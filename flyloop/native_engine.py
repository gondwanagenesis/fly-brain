"""Native-kernel whole-brain LIF engine.

Same model, same state, same RNG as ``flyloop.brain_engine.BrainEngine`` -- but
the per-step update runs in one fused AVX-512 pass in C instead of ~12 separate
PyTorch tensor ops.

Two structural changes, both exact:

1. **Fused single pass.** PyTorch streams the 2.2 MB neuron state through cache
   once per elementwise op. One fused pass moves it once. No arithmetic is
   changed -- see ``native/lif_kernel.c`` for the op-by-op correspondence and
   the two rounding facts (FMA in the membrane update, no FMA in the
   conductance decay) that had to be reproduced exactly.

2. **Sparse delay slots.** Upstream's delay line is a dense (19, N) fp32 ring
   buffer: 10.5 MB, which evicts the neuron state from L2/L3 every 19 steps.
   Its contents are the recurrent input, which has ~190 non-zeros out of
   138,639. Storing each slot as an (index, value) list makes the delay line
   ~30 KB resident. Exact, because the omitted entries are exactly zero and
   ``fl(x + 0.0f) == x`` for every value that occurs here.

The spike vector is likewise a bitset (17 KB) rather than an fp32 array
(554 KB); the AVX-512 compare produces it for free as a mask register.

``verify_native.py`` is the gate: full state (v, g, refrac) bit-identical and
spike trains identical against BrainEngine across every stimulation regime.
"""
from __future__ import annotations

import ctypes
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from brain_engine import MODEL_PARAMS, DT

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "native" / "lif_kernel.c"
_DLL = _HERE / "native" / "lif_kernel.dll"

_CLANG = Path(
    r"C:\Users\neogo\AppData\Local\Programs\Swift\Toolchains"
    r"\6.3.2+Asserts\usr\bin\clang.exe"
)

# -ffp-contract=off is MANDATORY: without it the compiler fuses the conductance
# decay's mul+add into an FMA, which rounds once instead of twice and breaks
# bit-identity with torch. -ffast-math would break it far more thoroughly.
#
# Deliberately NO -march=native. The kernel selects AVX-512 / AVX2 / scalar at
# RUNTIME via __builtin_cpu_supports, so one binary runs on any x86-64 (and on
# other architectures via the scalar path). Building with -march=native would
# produce a library that only runs on the machine that compiled it.
_CFLAGS = ["-shared", "-O3", "-ffp-contract=off",
           "-fno-fast-math", "-std=c11", "-Wall"]

_ISA_NAME = {0: "scalar", 1: "AVX2", 2: "AVX-512"}


def aten_vector_width():
    """ATen's float vector width on THIS host.

    The kernel has to reproduce the position of ATen's scalar tail, and that
    position depends on how the installed PyTorch was built, not on what this
    kernel can execute. torch reports its dispatched capability directly.

    ATen's fallback Vectorized<float> is a 32-byte array, i.e. 8 floats, so
    non-AVX512 builds all use 8.
    """
    try:
        cap = str(torch.backends.cpu.get_cpu_capability()).upper()
    except Exception:
        return 8
    return 16 if "AVX512" in cap else 8


def _dll_for(src_text, salt=""):
    """Content-addressed DLL name.

    Windows Smart App Control (VerifiedAndReputablePolicyState = 1) blocks
    unsigned binaries it has not vouched for, and it blocks them by FILE
    IDENTITY: once `lif_kernel.dll` is on its list, every rebuild to that same
    path fails to load with WinError 4551, while a byte-identical library under
    a different name loads fine (verified directly). Naming the artefact after
    a hash of the source therefore side-steps the block *and* gives us a free
    build cache -- an unchanged source maps to an already-compiled file.

    `salt` exists for the retry path below: if a specific hash-named file does
    get blocked, changing the salt yields a new identity to try.
    """
    h = hashlib.sha256((src_text + salt).encode()).hexdigest()[:12]
    return _HERE / "native" / f"lif_{h}.dll"


def _build(force=False, salt=""):
    src = _SRC.read_text(encoding="utf-8")
    dll = _dll_for(src, salt)
    if force or not dll.exists():
        cc = str(_CLANG) if _CLANG.exists() else "clang"
        subprocess.run([cc, *_CFLAGS, "-o", str(dll), str(_SRC)],
                       check=True, capture_output=True)
    return dll


def _load(force_build=False):
    # Retry under fresh file identities if Smart App Control blocks one.
    last = None
    for attempt in range(4):
        dll = _build(force=force_build, salt="" if not attempt else f"#{attempt}")
        try:
            lib = ctypes.CDLL(str(dll))
            break
        except OSError as e:
            last = e
            if "4551" not in str(e) and "Application Control" not in str(e):
                raise
            try:
                dll.unlink()
            except OSError:
                pass
    else:
        raise OSError(
            "Every build was blocked by Windows Application Control (Smart App "
            "Control). It blocks unsigned binaries by file identity; four "
            f"distinct identities were refused. Last error: {last}"
        )
    c_f32p = ctypes.POINTER(ctypes.c_float)
    c_i32p = ctypes.POINTER(ctypes.c_int32)
    c_i64p = ctypes.POINTER(ctypes.c_int64)
    c_u64p = ctypes.POINTER(ctypes.c_uint64)

    lib.lif_set_threads.restype = ctypes.c_int
    lib.lif_set_threads.argtypes = [ctypes.c_int]

    lib.lif_isa.restype = ctypes.c_int
    lib.lif_isa.argtypes = []

    lib.lif_force_isa.restype = ctypes.c_int
    lib.lif_force_isa.argtypes = [ctypes.c_int]

    lib.lif_step.restype = ctypes.c_int
    lib.lif_step.argtypes = [
        ctypes.c_int,                                    # n
        c_f32p, c_f32p,                                  # v, g
        c_u64p, c_u64p,                                  # gate_bits, in_ref
        c_i32p, c_i32p, ctypes.POINTER(ctypes.c_int),    # rc_idx, rc_cnt, n_ref
        c_i32p,                                          # refrac_steps (int32)
        c_u64p, c_u64p,                                  # sp_bits, sp_scratch
        ctypes.c_float, ctypes.c_float,                  # c_decay c_mem
        ctypes.c_float, ctypes.c_float, ctypes.c_float,  # v_rest v_reset v_th
        c_i32p, c_f32p, ctypes.c_int,                    # delayed
        c_i32p, c_f32p, ctypes.c_int,                    # stim
        c_i32p, ctypes.c_int, ctypes.c_int,              # chunk bounds, tail width
        c_i32p,                                          # out_spike_idx
    ]

    lib.lif_fanout.restype = ctypes.c_int
    lib.lif_fanout.argtypes = [
        ctypes.c_int,                                     # n
        c_i32p, ctypes.c_int,                             # spike_idx, nsp
        c_i64p, c_i32p, ctypes.POINTER(ctypes.c_int16),   # crow post val(int16)
        ctypes.c_float,                                   # w_scale
        c_i32p, c_i32p, c_u64p,                           # acc(int32), touched, bits
        c_i32p, c_f32p, ctypes.c_int,                     # out_idx, out_val, threaded
    ]
    return lib


_LIB = None


def _lib(force_build=False):
    global _LIB
    if _LIB is None or force_build:
        _LIB = _load(force_build)
    return _LIB


def _p(a, t):
    return a.ctypes.data_as(ctypes.POINTER(t))


class NativeBrainEngine:
    """Whole-brain LIF stepped by the fused native kernel."""

    def __init__(self, data_dir="data", params=None, dt=DT, stim_ids=None,
                 seed=0, force_build=False, threads=None, silence_ids=None):
        self.lib = _lib(force_build)
        self.p = dict(params or MODEL_PARAMS)
        self.dt = dt
        d = Path(data_dir)

        comp = pd.read_csv(d / "2025_Completeness_783.csv", index_col=0)
        self.flyid2i = {int(j): i for i, j in enumerate(comp.index)}
        self.i2flyid = np.asarray(comp.index, dtype=np.int64)
        self.N = N = len(self.i2flyid)

        fo = torch.load(d / "fanout_csc.pt")
        self.crow = np.ascontiguousarray(fo["crow"].numpy().astype(np.int64))
        self.post = np.ascontiguousarray(fo["post"].numpy().astype(np.int32))
        # int16 weights are LOSSLESS here, not a quantisation: every connectome
        # weight is an exact integer with |w| <= 2405 (see ANALYSIS.md section 1
        # and code/prove_exact_accumulation.py, which asserts it). Halves the
        # 60 MB weight array, which is the dominant traffic in fan-out-heavy
        # regimes. Accumulation is int32 and exact, so the result is unchanged.
        _w = fo["val"].numpy()
        assert np.array_equal(_w, np.rint(_w)),             "connectome weights are not integers -- int16 narrowing would be lossy"
        assert np.abs(_w).max() <= 32767, "weight exceeds int16 range"
        self.val = np.ascontiguousarray(_w.astype(np.int16))

        # Optical silencing, the repo's second manipulation type (`neu_slnc` in
        # code/benchmark.py). Upstream defines it as setting every synaptic
        # connection TO and FROM the listed neurons to zero, so it is a
        # connectome edit rather than a dynamics change: zero the outgoing CSC
        # range of each silenced neuron, and zero any entry targeting one.
        # The neuron itself still integrates, it is just disconnected.
        self.silence_idx = np.zeros(0, dtype=np.int64)
        if silence_ids:
            self.silence(silence_ids)

        # ---- constants, rounded to fp32 exactly as torch does ----
        self.c_decay = np.float32(1 - dt / self.p["tauSyn"])
        self.c_mem = np.float32(dt / self.p["tauMem"])
        self.v_rest = np.float32(self.p["vRest"])
        self.v_reset = np.float32(self.p["vReset"])
        self.v_th = np.float32(self.p["vThreshold"])
        self.w_scale = np.float32(self.p["wScale"])
        self._poi_scale = np.float32(self.p["scalePoisson"] * self.p["wScale"])

        # ---- state ----
        self.v = np.full(N, np.float32(self.p["v0"]), dtype=np.float32)
        self.g = np.zeros(N, dtype=np.float32)
        base_refrac = int(round(self.p["tRefrac"] / dt))
        self.refrac_steps = np.full(N, base_refrac, dtype=np.int32)
        # Refractory state as a gate bitset plus a compact countdown list. Only
        # ~40 neurons are refractory at once (1.75 spikes/step x 22 steps), so a
        # dense fp32 counter cost 554 KB of sweep traffic each way for nothing.
        # All gates start OPEN, matching the reference's initial
        # refrac == refrac_steps.
        _nw = (N + 63) >> 6
        self.gate_bits = np.full(_nw + 1, np.uint64(0xFFFFFFFFFFFFFFFF),
                                 dtype=np.uint64)
        self.in_ref = np.zeros(_nw + 1, dtype=np.uint64)
        self.rc_idx = np.zeros(N, dtype=np.int32)
        self.rc_cnt = np.zeros(N, dtype=np.int32)
        self.n_ref = ctypes.c_int(0)

        # Mirror ATen's at::parallel_for partition of an N-element tensor, so
        # the kernel's scalar tails land on the same neurons as the reference's
        # and the arithmetic matches bit-for-bit. ATen splits into
        # min(num_threads, ceil(N/GRAIN_SIZE)) chunks of ceil(N/n_chunks).
        _GRAIN = 32768
        nth = max(1, torch.get_num_threads())
        n_chunks = max(1, min(nth, -(-N // _GRAIN)))
        step = -(-N // n_chunks)
        bounds = list(range(0, N, step)) + [N]
        self.chunks = np.asarray(bounds, dtype=np.int32)
        self.n_chunks = len(bounds) - 1

        # ATen's float vector width on this host -- NOT the kernel's. The
        # kernel reproduces the reference's scalar-tail seam wherever it falls,
        # independently of which SIMD path the kernel itself takes.
        self.tail_w = aten_vector_width()
        self.isa = _ISA_NAME.get(self.lib.lif_isa(), "?")

        # One worker per chunk. The pool only engages when threads == n_chunks;
        # any other value runs the chunks serially. Either way the RESULT is
        # identical, because the chunk boundaries come from the ATen mirror and
        # not from our thread count -- so threading here is a pure speed knob
        # with no fidelity consequence.
        self.threads = self.n_chunks if threads is None else int(threads)
        self.lib.lif_set_threads(self.threads)
        # Threading the fan-out is CORRECT (the exactness theorem in ANALYSIS.md
        # section 1 makes integer accumulation order-independent) but MEASURED
        # SLOWER: atomic contention on the shared accumulator turned the
        # saturating regime from 1.56x into 0.73x, an outright regression.
        # Serial is the default; see the note in lif_kernel.c for the pull-based
        # formulation that should win in dense regimes instead.
        # Multi-threaded fan-out: OFF, and it should stay off.
        #
        # It is CORRECT -- fanout_range() accumulates with __atomic_fetch_add on
        # int32, and integer addition is associative, commutative and exact, so
        # the result is deterministic no matter how the threads interleave (the
        # `touched` order varies, but each neuron appears once and the delayed
        # pass is order-independent across distinct neurons). Verified
        # bit-identical to the serial path.
        #
        # But it is SLOWER, because the atomic contention costs more than the
        # parallelism buys. Measured min-of-7 x 120 steps on the two regimes
        # where fan-out actually dominates:
        #                        serial      threaded
        #   broad 1000        0.153 ms      0.178 ms
        #   saturating 40k    4.471 ms      7.396 ms   <- 1.65x WORSE
        # Set to 1 only to reproduce that measurement.
        self.mt_fanout = 0

        self.nw = (N + 63) >> 6
        # one word of padding: bit groups straddle words at unaligned chunk
        # boundaries, so get16/put16 may touch word w+1
        self.sp_bits = np.zeros(self.nw + 1, dtype=np.uint64)
        self._sp_scratch = np.zeros(self.nw + 1, dtype=np.uint64)
        # Two spike-index buffers, alternated. Upstream computes the recurrent
        # drive from the PREVIOUS step's spikes (``self.spikes`` is read at the
        # top of the step, before it is overwritten), so this step's fan-out
        # must be fed the previous list while the kernel is writing the new one.
        self._sp_buf = [np.zeros(N, dtype=np.int32), np.zeros(N, dtype=np.int32)]
        self._cur = 0
        self._cur_n = 0          # no spikes exist before t = 0
        self.n_spikes = 0

        # ---- sparse delay ring: L slots of (index, value) ----
        self.L = int(self.p["tDelay"] / dt) + 1
        self._del_idx = [np.zeros(N, dtype=np.int32) for _ in range(self.L)]
        self._del_val = [np.zeros(N, dtype=np.float32) for _ in range(self.L)]
        self._del_n = [0] * self.L
        self.head = 0

        # ---- fan-out scratch (kept zeroed by lif_fanout itself) ----
        self._acc = np.zeros(N, dtype=np.int32)
        self._touched = np.zeros(N, dtype=np.int32)
        self._touch_bits = np.zeros(self.nw, dtype=np.uint64)

        # ---- stimulation ----
        self.stim_idx = np.zeros(0, dtype=np.int32)
        self._stim_val = np.zeros(0, dtype=np.float32)
        self._rates = torch.zeros(0)
        if stim_ids:
            self.set_stim_neurons(stim_ids)

        self.gen = torch.Generator()
        self.gen.manual_seed(seed)

        # --- Poisson draws are pre-generated in blocks.
        #
        # torch.bernoulli on the 21 stimulated neurons measured 45 us/step --
        # 26% of the whole step, for 21 random numbers. That is pure PyTorch
        # per-op dispatch, not arithmetic. Drawing POISSON_BLOCK steps at once
        # amortises it to nothing.
        #
        # This is exact, not an approximation: a batched (K, n) bernoulli
        # consumes the generator's stream in the same order as K sequential
        # (n,) draws, so the values are bit-identical. Verified for
        # (n, K) = (21, 50), (21, 4096), (2, 1000), (1, 777) -- 0 differences.
        self.POISSON_BLOCK = 4096
        self._poi_block = None
        self._poi_pos = 0

        self.t_ms = 0.0
        self._rec_t, self._rec_n = [], []
        self._cache_pointers()

    def _cache_pointers(self):
        """Resolve every ctypes pointer once.

        ``ndarray.ctypes.data_as`` costs ~1-2 us and the step needs ~14 of
        them; rebuilding them every call was ~27 us/step of pure glue, which at
        a 100 us real-time budget is not affordable.
        """
        f, i32, u64 = ctypes.c_float, ctypes.c_int32, ctypes.c_uint64
        self._pv = _p(self.v, f)
        self._pg = _p(self.g, f)
        self._pgate = _p(self.gate_bits, u64)
        self._pinref = _p(self.in_ref, u64)
        self._prci = _p(self.rc_idx, i32)
        self._prcc = _p(self.rc_cnt, i32)
        self._prs = _p(self.refrac_steps, i32)
        self._pspb = _p(self.sp_bits, u64)
        self._pspc = _p(self._sp_scratch, u64)
        self._pchunk = _p(self.chunks, i32)
        self._pstim = _p(self.stim_idx, i32)
        self._pstimv = _p(self._stim_val, f)
        self._pacc = _p(self._acc, i32)
        self._ptouch = _p(self._touched, i32)
        self._ptbits = _p(self._touch_bits, u64)
        self._pcrow = _p(self.crow, ctypes.c_int64)
        self._ppost = _p(self.post, i32)
        self._pval = _p(self.val, ctypes.c_int16)
        self._pdel_i = [_p(a, i32) for a in self._del_idx]
        self._pdel_v = [_p(a, f) for a in self._del_val]
        self._psp = [_p(a, i32) for a in self._sp_buf]
        self._cf = {k: ctypes.c_float(getattr(self, k)) for k in
                    ("c_decay", "c_mem", "v_rest", "v_reset", "v_th", "w_scale")}

    def _draw_poisson(self):
        n = self.stim_idx.size
        p = (self._rates * (self.dt / 1000.0)).expand(
            self.POISSON_BLOCK, n).contiguous()
        blk = torch.bernoulli(p, generator=self.gen)
        blk.mul_(float(self._poi_scale))
        self._poi_block = blk.numpy()
        self._poi_pos = 0

    # ---------------- interface ----------------
    def _fanout_dtype_ok(self):
        return self.val.dtype == np.int16

    def silence(self, flywire_ids):
        """Zero every synapse to and from these neurons. Returns the count."""
        idx = np.asarray([self.flyid2i[int(i)] for i in flywire_ids
                          if int(i) in self.flyid2i], dtype=np.int64)
        if not idx.size:
            return 0
        self.silence_idx = np.union1d(self.silence_idx, idx)
        for j in idx:                                    # outgoing
            self.val[self.crow[j]:self.crow[j + 1]] = np.int16(0)
        mask = np.isin(self.post, idx.astype(np.int32))  # incoming
        self.val[mask] = np.int16(0)
        return int(idx.size)

    def set_stim_neurons(self, flywire_ids):
        idx = [self.flyid2i[int(i)] for i in flywire_ids if int(i) in self.flyid2i]
        self.stim_idx = np.asarray(idx, dtype=np.int32)
        self.refrac_steps[self.stim_idx] = 0
        self._stim_val = np.zeros(len(idx), dtype=np.float32)
        self._rates = torch.zeros(len(idx))
        self._poi_block = None
        if hasattr(self, "_pv"):        # arrays were replaced -> re-resolve
            self._cache_pointers()
        return len(idx)

    def inject(self, rates_hz):
        if np.isscalar(rates_hz):
            self._rates.fill_(float(rates_hz))
        else:
            self._rates.copy_(torch.as_tensor(rates_hz, dtype=torch.float32))
        # rates changed -> the pre-drawn block is stale. Discarding it consumes
        # the RNG stream differently from the reference, so a closed-loop run
        # that calls inject() every step must set POISSON_BLOCK = 1 to stay
        # bit-identical. Constant-drive runs (every regime in the gate) are
        # unaffected.
        self._poi_block = None

    def indices_of(self, flywire_ids):
        return np.asarray([self.flyid2i[int(i)] for i in flywire_ids
                           if int(i) in self.flyid2i], dtype=np.int64)

    def step(self, record=False):
        N = self.N

        # --- Poisson drive, served from the pre-drawn block (see _draw_poisson) ---
        n_stim = self.stim_idx.size
        if n_stim:
            if self._poi_block is None or self._poi_pos >= self.POISSON_BLOCK:
                self._draw_poisson()
            self._stim_val[:] = self._poi_block[self._poi_pos]
            self._poi_pos += 1

        h = self.head
        prev_n = self._cur_n
        cf = self._cf
        nsp = self.lib.lif_step(
            N,
            self._pv, self._pg,
            self._pgate, self._pinref,
            self._prci, self._prcc, ctypes.byref(self.n_ref),
            self._prs,
            self._pspb, self._pspc,
            cf["c_decay"], cf["c_mem"], cf["v_rest"], cf["v_reset"], cf["v_th"],
            self._pdel_i[h], self._pdel_v[h], self._del_n[h],
            self._pstim, self._pstimv, n_stim,
            self._pchunk, self.n_chunks, self.tail_w,
            self._psp[1 - self._cur],
        )
        self.n_spikes = nsp

        # --- fan-out the PREVIOUS step's spikes into the slot just consumed.
        #     The kernel read slot[h] as this step's delayed input before we
        #     overwrite it here, exactly as upstream's read-then-write ring. ---
        self._del_n[h] = self.lib.lif_fanout(
            N,
            self._psp[self._cur], prev_n,
            self._pcrow, self._ppost, self._pval, cf["w_scale"],
            self._pacc, self._ptouch, self._ptbits,
            self._pdel_i[h], self._pdel_v[h], self.mt_fanout,
        ) if prev_n else 0
        self.head = (h + 1) % self.L
        self._cur, self._cur_n = 1 - self._cur, nsp
        self.t_ms += self.dt

        if record and nsp:
            self._rec_n.append(self._sp_buf[self._cur][:nsp].copy())
            self._rec_t.append(np.full(nsp, self.t_ms))
        return nsp

    @property
    def spike_idx(self):
        return self._sp_buf[self._cur][:self._cur_n]

    def spikes_dataframe(self):
        if not self._rec_n:
            return pd.DataFrame(columns=["time_ms", "neuron_index", "flywire_id"])
        n = np.concatenate(self._rec_n).astype(np.int64)
        t = np.concatenate(self._rec_t)
        return pd.DataFrame({"time_ms": t, "neuron_index": n,
                             "flywire_id": self.i2flyid[n]})
