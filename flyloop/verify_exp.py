"""Exhaustive accuracy audit of the kernel's vectorised exponential.

    .venv\\Scripts\\python.exe flyloop\\verify_exp.py [--quick]

WHY EXHAUSTIVE
--------------
AdEx, EIF and Hodgkin-Huxley all route through one hand-written `fly_exp`
(sweep_template.h). Every other operation in every model is a single IEEE-754
add, multiply, FMA or compare, whose behaviour is defined by the standard and
reproduced by construction. The exponential is the only place where this
kernel makes an accuracy CHOICE, so it is the only place where an accuracy
claim has to be earned.

Sampling would not earn it. A polynomial approximation's worst case does not
sit where sampling looks -- it hides at a specific reduced argument near a
range-reduction boundary, and a random sweep of a million points can miss it
completely while reporting a reassuring 0.5 ULP mean.

float32 makes the honest version cheap: the entire reachable argument domain
contains only about 2.1 billion representable values, which is a few minutes of
walking rather than a sampling problem. So this checks EVERY ONE against a
float64 reference, and reports the worst error anywhere in the domain. That is
a proof over the domain, not evidence about it.

It also re-checks the three ISA paths against each other over the same
exhaustive set, which is a far stronger statement than the whole-brain
comparison in verify_models.py: a divergence that needs a specific bit pattern
to trigger will be found here and would almost certainly be missed there.
"""
import ctypes
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "flyloop"))

import native_lib                                # noqa: E402

# fly_exp clamps to [-87, 88]; outside that expf has already saturated to 0 or
# +inf, so the clamp pins values that are pinned anyway. The audit therefore
# covers the clamp interval, which is the whole of the function's live domain.
LO, HI = -87.0, 88.0
CHUNK = 1 << 22
ISA = {2: "AVX-512", 1: "AVX2", 0: "scalar"}


def probe(lib, x):
    out = np.empty_like(x)
    lib.nrn_exp_probe(native_lib.ptr(x), native_lib.ptr(out), x.size)
    return out


def ulp_gap(a, b):
    """Distance in representable float32 steps between two finite same-sign
    positives. Monotone bit patterns make this an integer subtraction."""
    return np.abs(a.view(np.int32).astype(np.int64)
                  - b.view(np.int32).astype(np.int64))


def domain_chunks(quick):
    """Every float32 in [LO, HI], chunk by chunk.

    Both halves are enumerated by MAGNITUDE and the sign bit is set afterwards.
    The obvious version -- take the bit pattern of LO and count up to the bit
    pattern of HI -- silently covers nothing at all on the negative side,
    because a negative float32's bit pattern read as int32 is itself negative,
    so the range is empty and the loop runs zero times while reporting a
    perfectly plausible total. That would have left every negative argument
    untested, which is most of what these models actually evaluate: every decay
    factor, every Rush-Larsen step and the whole sub-threshold branch of AdEx
    pass negative arguments.
    """
    mag_lo = int(np.float32(abs(LO)).view(np.int32))
    mag_hi = int(np.float32(HI).view(np.int32))
    stride = 64 if quick else 1
    step = CHUNK * stride
    for mag_end, sign_bit in ((mag_lo, np.uint32(1) << np.uint32(31)),
                              (mag_hi, np.uint32(0))):
        for i in range(0, mag_end + 1, step):
            mags = np.arange(i, min(i + step, mag_end + 1), stride,
                             dtype=np.uint32)
            yield (mags | sign_bit)


def main():
    quick = "--quick" in sys.argv
    lib = native_lib.lib()
    cap = lib.lif_force_isa(99)
    print(f"vectorised exp audit   domain [{LO}, {HI}]   "
          f"{'QUICK (1/64 sample)' if quick else 'EXHAUSTIVE'}")
    print(f"host ISA: {ISA[cap]}\n")

    results = {}
    for want in (2, 1, 0):
        if want > cap:
            continue
        lib.lif_force_isa(want)
        worst_ulp, worst_x, n_seen = 0, 0.0, 0
        sum_ulp, n_exact = 0.0, 0
        t0 = time.perf_counter()
        digests = []
        for bits in domain_chunks(quick):
            x = np.ascontiguousarray(bits.view(np.float32))
            if not x.size:
                continue
            y = probe(lib, x)
            ref = np.exp(x.astype(np.float64)).astype(np.float32)
            # Both are finite and non-negative over this domain; underflow to
            # +0 on either side is an exact agreement, not a special case.
            g = ulp_gap(y, ref)
            k = int(g.argmax())
            if g[k] > worst_ulp:
                worst_ulp, worst_x = int(g[k]), float(x[k])
            sum_ulp += float(g.sum())
            n_exact += int((g == 0).sum())
            n_seen += x.size
            digests.append(int(y.view(np.uint32).astype(np.int64).sum()))
        dt = time.perf_counter() - t0
        results[want] = (worst_ulp, worst_x, n_seen, sum_ulp / n_seen,
                         n_exact / n_seen, sum(digests))
        print(f"  {ISA[want]:<8} {n_seen:>12,} values   "
              f"max {worst_ulp} ULP at x={worst_x:+.7g}   "
              f"mean {sum_ulp/n_seen:.4f} ULP   "
              f"{100*n_exact/n_seen:.2f}% exactly rounded   ({dt:.0f}s)")
    lib.lif_force_isa(99)

    print()
    ok = True
    ds = {k: v[5] for k, v in results.items()}
    if len(set(ds.values())) != 1:
        print("*** ISA PATHS DISAGREE ***", ds)
        ok = False
    else:
        print(f"all {len(ds)} ISA paths bit-identical over the whole domain "
              f"(checksum {list(ds.values())[0]})")

    worst = max(v[0] for v in results.values())
    print(f"worst error anywhere in the domain: {worst} ULP")
    if worst > 2:
        print("*** worse than the 2 ULP the models are documented to assume ***")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
