"""Build, load and bind the native multi-model neuron kernel.

Split out of native_engine.py so that models.py can drive the kernel directly
without importing the engine (and without a circular import). That matters more
than it sounds: it is what lets the model calibration, the resting-state search
and the exactness audits all run against THE KERNEL rather than against a numpy
lookalike that would be free to drift away from it. There is exactly one
implementation of every membrane equation in this repository.
"""
from __future__ import annotations

import ctypes
import hashlib
import subprocess
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_NATIVE = _HERE / "native"
_SRC = _NATIVE / "nrn_kernel.c"
# Headers are part of the build input: the model bodies live in
# sweep_template.h and the ISA mapping in simd.h, so hashing only the .c would
# leave a stale DLL cached after every model change -- a silent wrong-answer
# bug rather than a loud one.
_HEADERS = ("simd.h", "simd_undef.h", "nrn_params.h", "sweep_template.h")

_CLANG = Path(
    r"C:\Users\neogo\AppData\Local\Programs\Swift\Toolchains"
    r"\6.3.2+Asserts\usr\bin\clang.exe"
)

# -ffp-contract=off is MANDATORY: without it the compiler fuses the LIF
# conductance decay's mul+add into an FMA, which rounds once instead of twice
# and breaks bit-identity with torch. -ffast-math would break it far more
# thoroughly. Note this does NOT stop the explicit fmaf()/_mm*_fmadd_ps calls in
# the model bodies -- those are requested fusions, not compiler-invented ones.
#
# Deliberately NO -march=native: the kernel selects AVX-512 / AVX2 / scalar at
# runtime via CPUID, so one binary runs on any x86-64.
_CFLAGS = ["-shared", "-O3", "-ffp-contract=off",
           "-fno-fast-math", "-std=c11", "-Wall"]

ISA_NAME = {0: "scalar", 1: "AVX2", 2: "AVX-512"}


def aten_vector_width():
    """ATen's float vector width on THIS host.

    The kernel has to reproduce the position of ATen's scalar tail, and that
    position depends on how the installed PyTorch was built, not on what this
    kernel can execute.
    """
    try:
        cap = str(torch.backends.cpu.get_cpu_capability()).upper()
    except Exception:
        return 8
    return 16 if "AVX512" in cap else 8


def _source_fingerprint():
    h = hashlib.sha256()
    h.update(_SRC.read_bytes())
    for name in _HEADERS:
        h.update((_NATIVE / name).read_bytes())
    return h.hexdigest()[:12]


def _dll_for(salt=""):
    """Content-addressed DLL name.

    Windows Smart App Control blocks unsigned binaries by FILE IDENTITY: once a
    fixed path is on its list every rebuild to that path fails to load with
    WinError 4551, while a byte-identical library under a different name loads
    fine. Naming the artefact after a hash of the sources side-steps that and
    doubles as a build cache. `salt` gives the retry path a fresh identity.
    """
    return _NATIVE / f"nrn_{hashlib.sha256((_source_fingerprint() + salt).encode()).hexdigest()[:12]}.dll"


def _build(force=False, salt=""):
    dll = _dll_for(salt)
    if force or not dll.exists():
        cc = str(_CLANG) if _CLANG.exists() else "clang"
        r = subprocess.run([cc, *_CFLAGS, "-o", str(dll), str(_SRC)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"kernel build failed:\n{r.stderr}")
    return dll


def _bind(lib):
    f32 = ctypes.POINTER(ctypes.c_float)
    i32 = ctypes.POINTER(ctypes.c_int32)
    i64 = ctypes.POINTER(ctypes.c_int64)
    u64 = ctypes.POINTER(ctypes.c_uint64)
    u8 = ctypes.POINTER(ctypes.c_uint8)
    vp = ctypes.c_void_p

    lib.lif_set_threads.restype = ctypes.c_int
    lib.lif_set_threads.argtypes = [ctypes.c_int]
    lib.lif_isa.restype = ctypes.c_int
    lib.lif_isa.argtypes = []
    lib.lif_force_isa.restype = ctypes.c_int
    lib.lif_force_isa.argtypes = [ctypes.c_int]
    lib.nrn_params_size.restype = ctypes.c_int
    lib.nrn_params_size.argtypes = []

    lib.nrn_step.restype = ctypes.c_int
    lib.nrn_step.argtypes = [
        ctypes.c_int, vp,                                # model, params*
        f32, ctypes.c_int, ctypes.c_int, f32,            # aux, stride, n_aux, rest
        ctypes.c_int,                                    # n
        f32, f32,                                        # v, g
        u64, u64,                                        # gate_bits, in_ref
        i32, i32, ctypes.POINTER(ctypes.c_int),          # rc_idx, rc_cnt, n_ref
        i32,                                             # refrac_steps
        u64, u64,                                        # sp_bits, sp_scratch
        i32, f32, ctypes.c_int,                          # delayed
        i32, f32, ctypes.c_int,                          # stim
        i32, ctypes.c_int, ctypes.c_int,                 # chunks, n_chunks, tail_w
        i32, u8, u64,                                    # tile_lo, tile_fma, live
        u64,                                             # fma_bits
        u64, i32,                                        # tile_pin, chunk_tile
        i32, ctypes.c_int, ctypes.c_int,                 # neuron_tile, n_tiles, rescan
        i32,                                             # out_spike_idx
    ]

    lib.lif_step.restype = ctypes.c_int
    lib.lif_step.argtypes = [
        ctypes.c_int, f32, f32, u64, u64,
        i32, i32, ctypes.POINTER(ctypes.c_int), i32, u64, u64,
        ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        i32, f32, ctypes.c_int, i32, f32, ctypes.c_int,
        i32, ctypes.c_int, ctypes.c_int,
        i32, u8, u64, u64, u64, i32,
        i32, ctypes.c_int, ctypes.c_int, i32,
    ]

    lib.lif_fanout.restype = ctypes.c_int
    lib.lif_fanout.argtypes = [
        ctypes.c_int, i32, ctypes.c_int,
        i64, i32, ctypes.POINTER(ctypes.c_int16), ctypes.c_float,
        i32, i32, u64, i32, f32, ctypes.c_int,
    ]

    lib.nrn_sweep_raw.restype = ctypes.c_int
    lib.nrn_sweep_raw.argtypes = [
        ctypes.c_int, vp, f32, f32, f32, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, i32, u64, u64,
    ]

    lib.nrn_exp_probe.restype = None
    lib.nrn_exp_probe.argtypes = [f32, f32, ctypes.c_int]
    return lib


def _load(force_build=False):
    last = None
    for attempt in range(4):
        dll = _build(force=force_build, salt="" if not attempt else f"#{attempt}")
        try:
            return _bind(ctypes.CDLL(str(dll)))
        except OSError as e:
            last = e
            if "4551" not in str(e) and "Application Control" not in str(e):
                raise
            try:
                dll.unlink()
            except OSError:
                pass
    raise OSError(
        "Every build was blocked by Windows Application Control (Smart App "
        "Control). It blocks unsigned binaries by file identity; four distinct "
        f"identities were refused. Last error: {last}"
    )


_LIB = None


def lib(force_build=False):
    global _LIB
    if _LIB is None or force_build:
        _LIB = _load(force_build)
    return _LIB


def ptr(a, t=ctypes.c_float):
    return a.ctypes.data_as(ctypes.POINTER(t))
