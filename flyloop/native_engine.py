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
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import models as nrn_models
from brain_engine import MODEL_PARAMS, DT
from native_lib import ISA_NAME as _ISA_NAME, aten_vector_width, lib as _lib, ptr as _p

class NativeBrainEngine:
    """Whole-brain LIF stepped by the fused native kernel."""

    def __init__(self, data_dir="data", params=None, dt=DT, stim_ids=None,
                 seed=0, force_build=False, threads=None, silence_ids=None,
                 reorder=None, model="lif_euler"):
        self.lib = _lib(force_build)
        self.p = dict(params or MODEL_PARAMS)
        self.dt = dt

        # ---- the membrane model ----
        # Everything below this line is network machinery and is IDENTICAL for
        # all nine models: the connectome, the 1.8 ms delay ring, the
        # event-driven fan-out, the refractory gate, the tiles and the thread
        # pool. Only `spec` changes. See models.py for why the non-reference
        # models are calibrated rather than copied out of their papers.
        self.spec = nrn_models.build(model, dt=dt) if isinstance(model, str) \
            else model
        self.model = self.spec.key
        self.n_aux = self.spec.n_aux
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

        # ---- optional neuron reordering, for tile-skip locality ----
        # Must happen before any state or index is derived below.
        self.perm = self.inv = None
        if reorder and reorder != "none":
            from reorder import build_permutation
            self._apply_perm(build_permutation(str(d), key=reorder))

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
        # Poisson drive is injected straight into v, bypassing the synapse, so
        # unlike the recurrent input it does not pass through the model's k_in.
        # Scale it by the ratio of threshold gaps instead, so that one sensory
        # event moves the membrane the same FRACTION of the way to threshold in
        # every model. Without this, switching to Izhikevich (a 20 mV gap) or
        # Hodgkin-Huxley (65 mV) would silently weaken the sensory drive by 3x
        # to 9x and the model comparison would be measuring that instead.
        _gap = float(self.spec.params.v_th) - float(self.spec.rest[0])
        self._gap_ratio = _gap / (MODEL_PARAMS["vThreshold"] - MODEL_PARAMS["vRest"])
        self._poi_scale = np.float32(
            self.p["scalePoisson"] * self.p["wScale"] * self._gap_ratio)

        # ---- state ----
        # v starts at the model's OWN resting fixed point, which models.build()
        # located by relaxing the kernel rather than by quoting a textbook: for
        # Izhikevich it is a root of a quadratic and for Hodgkin-Huxley the
        # solution of a transcendental system, and only a state the update maps
        # to itself bit-for-bit may be skipped as inert.
        self.v = np.full(N, self.spec.init[0], dtype=np.float32)
        self.g = np.zeros(N, dtype=np.float32)
        # Auxiliary state, (n_aux, N) contiguous: Izhikevich's u, AdEx's w,
        # resonate-and-fire's y, HH's m/h/n/armed, GLIF's theta.
        self.aux = (np.repeat(self.spec.init[1:].astype(np.float32), N)
                    .reshape(self.n_aux, N).copy()
                    if self.n_aux else np.zeros(1, dtype=np.float32))
        self.aux = np.ascontiguousarray(self.aux)
        self.rest = np.ascontiguousarray(self.spec.rest.astype(np.float32))
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

        # ---- 16-neuron tiles, defined RELATIVE TO EACH CHUNK ----
        #
        # The ATen chunk boundaries are not multiples of 16, so a globally
        # aligned tile would straddle the FMA/non-FMA seam at a chunk edge.
        # Tiling from each chunk's start makes the FMA region exactly
        # (vec_end - lo)/16 whole groups, with the non-FMA remainder as one
        # final partial tile -- so every tile has one uniform rounding mode.
        TILE = 16
        t_lo, t_fma, c_tile = [], [], [0]
        for c in range(self.n_chunks):
            lo, hi = int(bounds[c]), int(bounds[c + 1])
            vec_end = lo + ((hi - lo) // self.tail_w) * self.tail_w
            i = lo
            while i < vec_end:
                t_lo.append(i)
                t_fma.append(1)
                i = min(i + TILE, vec_end)
            if vec_end < hi:                      # ATen's scalar tail: no FMA
                t_lo.append(vec_end)
                t_fma.append(0)
            c_tile.append(len(t_lo))
        t_lo.append(N)
        self.tile_lo = np.asarray(t_lo, dtype=np.int32)
        self.tile_fma = np.asarray(t_fma, dtype=np.uint8)

        # --- ATen's rounding seam follows the NEURON, not the array slot ---
        #
        # The reference applies its non-fused scalar tail to the last
        # (chunk_len mod tail_w) neurons of each of ITS chunks, in the ORIGINAL
        # index space. Unpermuted, position == neuron and the tile-level flags
        # above are exact. Once neurons are permuted the two come apart, so the
        # flag is recomputed per neuron and carried through the permutation;
        # tiles that end up straddling the seam are marked MIXED (2) and take a
        # scalar path that looks the bit up. At most 31 neurons brain-wide.
        fma_orig = np.zeros(N, dtype=bool)
        for c in range(self.n_chunks):
            lo, hi = int(bounds[c]), int(bounds[c + 1])
            fma_orig[lo:lo + ((hi - lo) // self.tail_w) * self.tail_w] = True
        fma_new = fma_orig[self.perm] if self.perm is not None else fma_orig
        self.fma_bits = np.zeros(((N + 63) >> 6) + 1, dtype=np.uint64)
        _w = np.nonzero(fma_new)[0]
        np.bitwise_or.at(self.fma_bits, _w >> 6,
                         np.uint64(1) << (_w & 63).astype(np.uint64))
        for k in range(len(self.tile_fma)):
            seg = fma_new[self.tile_lo[k]:self.tile_lo[k + 1]]
            self.tile_fma[k] = 1 if seg.all() else (0 if not seg.any() else 2)
        self.chunk_tile = np.asarray(c_tile, dtype=np.int32)
        self.n_tiles = nt = len(t_fma)

        # neuron -> tile, used only by the sparse delayed pass (~190/step)
        self.neuron_tile = np.zeros(N, dtype=np.int32)
        for k in range(nt):
            self.neuron_tile[self.tile_lo[k]:self.tile_lo[k + 1]] = k

        _ntw = (nt + 63) >> 6
        # B1b: every tile starts LIVE. Initialising by scan would mark the whole
        # brain dead at t=0 (all neurons are at exact rest) and nothing would
        # ever run.
        self.tile_live = np.full(_ntw + 1, np.uint64(0xFFFFFFFFFFFFFFFF),
                                 dtype=np.uint64)
        self.tile_pin = np.zeros(_ntw + 1, dtype=np.uint64)
        self.rescan_every = 64
        self._since_rescan = 0
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
        self._ptlo = _p(self.tile_lo, i32)
        self._ptfma = _p(self.tile_fma, ctypes.c_uint8)
        self._ptlive = _p(self.tile_live, u64)
        self._pfmab = _p(self.fma_bits, u64)
        self._ptpin = _p(self.tile_pin, u64)
        self._pctile = _p(self.chunk_tile, i32)
        self._pntile = _p(self.neuron_tile, i32)
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
        self._paux = _p(self.aux.reshape(-1), f) if self.n_aux else None
        self._prest = _p(self.rest, f)
        self._pparams = ctypes.byref(self.spec.params)
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

    def _apply_perm(self, perm):
        """Renumber neurons so co-active ones land in the same tile.

        A permutation is pure relabeling, so this is exact -- it changes which
        neurons share a 16-wide tile, and nothing else. Neuron index is an
        arbitrary artefact of the completeness CSV's row order, and in that
        order the live neurons are scattered (mean run length 1.1), so 98.85%
        of tiles hold at least one live neuron and skipping saves 1.15%.
        Grouping by `cell_type` -- cells of a type share inputs, so they fall
        quiet together -- takes tile-16 from 70.6% live to 28.3% on sugar.

        Callers address neurons by FlyWire id everywhere (inject, silence,
        indices_of, spikes_dataframe), so the renumbering is invisible from
        outside: only the tile hit-rate changes.
        """
        N = self.N
        perm = np.asarray(perm, dtype=np.int64)
        if perm.shape != (N,):
            raise ValueError(f"permutation must have length {N}, got {perm.shape}")
        inv = np.empty(N, dtype=np.int64)
        inv[perm] = np.arange(N, dtype=np.int64)

        # The fan-out is CSC (grouped by PREsynaptic neuron), so the groups are
        # reordered with their sources and the postsynaptic ids relabelled.
        lens = (self.crow[1:] - self.crow[:-1])[perm]
        total = int(lens.sum())
        base = np.repeat(self.crow[perm], lens)
        ramp = (np.arange(total, dtype=np.int64)
                - np.repeat(np.cumsum(lens) - lens, lens))
        take = base + ramp

        new_crow = np.zeros(N + 1, dtype=np.int64)
        np.cumsum(lens, out=new_crow[1:])
        self.crow = np.ascontiguousarray(new_crow)
        self.post = np.ascontiguousarray(inv[self.post[take]].astype(np.int32))
        self.val = np.ascontiguousarray(self.val[take])

        self.i2flyid = np.ascontiguousarray(self.i2flyid[perm])
        self.flyid2i = {int(j): i for i, j in enumerate(self.i2flyid)}
        self.perm, self.inv = perm, inv

    # ---------------- interface ----------------
    def set_model(self, model):
        """Switch membrane model in place, keeping the connectome loaded.

        Rebuilding the engine to change model would re-read a 182 MB fan-out
        table and re-derive the tiling for something that does not depend on
        either: the connectome, the delay ring, the chunk partition, the tile
        map and the permutation are all properties of the NETWORK. Only the
        parameter block, the auxiliary state and the initial membrane value
        belong to the model. Swapping just those turns a ~15 s reload into a
        few milliseconds, which is what makes an interactive model switch
        possible at all.

        The simulation is reset, because carrying Izhikevich's membrane over
        into Hodgkin-Huxley would be meaningless -- the variables do not denote
        the same thing.
        """
        self.spec = nrn_models.build(model, dt=self.dt) if isinstance(model, str) \
            else model
        self.model = self.spec.key
        self.n_aux = self.spec.n_aux
        _gap = float(self.spec.params.v_th) - float(self.spec.rest[0])
        self._gap_ratio = _gap / (MODEL_PARAMS["vThreshold"] - MODEL_PARAMS["vRest"])
        self._poi_scale = np.float32(
            self.p["scalePoisson"] * self.p["wScale"] * self._gap_ratio)
        self.rest = np.ascontiguousarray(self.spec.rest.astype(np.float32))
        self.aux = np.ascontiguousarray(
            np.repeat(self.spec.init[1:].astype(np.float32), self.N)
            .reshape(self.n_aux, self.N).copy()
            if self.n_aux else np.zeros(1, dtype=np.float32))
        self.reset()
        return self.spec

    def reset(self):
        """Back to t = 0: state at the model's resting fixed point, delay ring
        empty, nothing refractory, every tile live."""
        N = self.N
        self.v[:] = self.spec.init[0]
        self.g[:] = 0.0
        if self.n_aux:
            for k in range(self.n_aux):
                self.aux[k, :] = self.spec.init[1 + k]
        self.gate_bits[:] = np.uint64(0xFFFFFFFFFFFFFFFF)
        self.in_ref[:] = 0
        self.rc_cnt[:] = 0
        self.n_ref = ctypes.c_int(0)
        self.sp_bits[:] = 0
        self._sp_scratch[:] = 0
        self._del_n = [0] * self.L
        self.head = 0
        self._cur, self._cur_n = 0, 0
        self.n_spikes = 0
        # Every tile starts LIVE. Initialising by scan would mark the whole
        # brain dead at t = 0 -- all neurons are at exact rest -- and nothing
        # would ever run again.
        self.tile_live[:] = np.uint64(0xFFFFFFFFFFFFFFFF)
        for i in self.stim_idx:
            tt = int(self.neuron_tile[int(i)])
            self.tile_pin[tt >> 6] |= np.uint64(1) << np.uint64(tt & 63)
        self._since_rescan = 0
        self._poi_block = None
        self._poi_pos = 0
        self.t_ms = 0.0
        self._rec_t, self._rec_n = [], []
        self._cache_pointers()

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
        # B1: stimulated neurons are driven OUTSIDE the delay ring, so nothing
        # in the delayed pass can mark their tile live. A stim neuron whose
        # Poisson draw is 0 sits at exact rest and would be declared inert, its
        # tile cleared, and the sensory drive silently disconnected. Pin them.
        if hasattr(self, "tile_pin"):
            for i in self.stim_idx:
                tt = int(self.neuron_tile[int(i)])
                self.tile_pin[tt >> 6] |= np.uint64(1) << np.uint64(tt & 63)
                self.tile_live[tt >> 6] |= np.uint64(1) << np.uint64(tt & 63)
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

        self._since_rescan += 1
        rescan = 1 if self._since_rescan >= self.rescan_every else 0
        if rescan:
            self._since_rescan = 0
        # A model whose resting state is not a bit-level fixed point of its own
        # update must never have tiles cleared: skipping would freeze a state
        # that is still drifting. models.build() reports this per model; every
        # one of the nine currently passes, but the guard is what makes that a
        # checked fact rather than an assumption.
        if not self.spec.extra.get("can_skip_tiles", True):
            rescan = 0

        h = self.head
        prev_n = self._cur_n
        nsp = self.lib.nrn_step(
            self.spec.mid, self._pparams,
            self._paux, N if self.n_aux else 0, self.n_aux, self._prest,
            N,
            self._pv, self._pg,
            self._pgate, self._pinref,
            self._prci, self._prcc, ctypes.byref(self.n_ref),
            self._prs,
            self._pspb, self._pspc,
            self._pdel_i[h], self._pdel_v[h], self._del_n[h],
            self._pstim, self._pstimv, n_stim,
            self._pchunk, self.n_chunks, self.tail_w,
            self._ptlo, self._ptfma, self._ptlive, self._pfmab,
            self._ptpin, self._pctile,
            self._pntile, self.n_tiles, rescan,
            self._psp[1 - self._cur],
        )
        self.n_spikes = nsp

        # --- fan-out the PREVIOUS step's spikes into the slot just consumed.
        #     The kernel read slot[h] as this step's delayed input before we
        #     overwrite it here, exactly as upstream's read-then-write ring. ---
        self._del_n[h] = self.lib.lif_fanout(
            N,
            self._psp[self._cur], prev_n,
            self._pcrow, self._ppost, self._pval, self._cf["w_scale"],
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
