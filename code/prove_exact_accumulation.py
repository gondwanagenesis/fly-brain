"""Verify the preconditions of the exact-accumulation theorem, then test it.

THEOREM (see ANALYSIS.md section 1)
For this connectome, float32 accumulation of synaptic weights incurs NO rounding
and is therefore INDEPENDENT OF SUMMATION ORDER.

Proof sketch: every weight is an exact integer; the total in-weight on any
neuron is at most 69,948; float32 represents every integer up to 2^24 =
16,777,216 exactly; so every partial sum is exact, and the result equals the
exact integer sum regardless of order.

WHY THIS SCRIPT EXISTS
The theorem is a property of the DATA, not of the model equations. If the
connectome is ever rescaled, resampled, or replaced with non-integer weights,
the theorem fails SILENTLY -- accumulation becomes order-dependent again and any
parallel fan-out quietly stops being reproducible. So the preconditions must be
asserted, not assumed. This script is cheap and safe to run in CI.

What the theorem buys: the fan-out can be parallelised in any order (CPU
threads, GPU atomicAdd, warp reductions) while staying bit-identical, weights
are losslessly int16, and a GPU port can be bit-reproducible -- which no GPU
backend in this repository currently is, because atomicAdd(float*) normally has
non-deterministic ordering.

Exit code 0 if the theorem holds, 1 if any precondition fails.

Run:  .venv\\Scripts\\python.exe code\\prove_exact_accumulation.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
FANOUT = ROOT / "data" / "fanout_csc.pt"

FP32_EXACT_INT = 2 ** 24        # float32 represents every integer below this
INT16_MAX = 32767
N_PERM = 200


def main():
    fo = torch.load(FANOUT)
    crow = fo["crow"].numpy()
    post = fo["post"].numpy()
    val = fo["val"].numpy()
    n = crow.size - 1
    ok = True

    print(f"connectome: {n:,} neurons, {val.size:,} edges\n")
    print("PRECONDITIONS")

    # (1) every weight is an exact integer
    max_frac = float(np.abs(val - np.rint(val)).max())
    p1 = max_frac == 0.0
    ok &= p1
    print(f"  [{'PASS' if p1 else 'FAIL'}] all weights integer-valued "
          f"(max |w - round(w)| = {max_frac})")

    # (2) individual weights fit int16 -> lossless narrowing
    wmax = float(np.abs(val).max())
    p2 = wmax <= INT16_MAX
    ok &= p2
    print(f"  [{'PASS' if p2 else 'FAIL'}] max |w| = {wmax:,.0f} <= int16 max "
          f"{INT16_MAX:,}")

    # (3) every PARTIAL sum is bounded by the total in-weight, which must stay
    #     inside float32's exactly-representable integer range
    tot = np.zeros(n, dtype=np.int64)
    np.add.at(tot, post, np.abs(val).astype(np.int64))
    tmax = int(tot.max())
    p3 = tmax < FP32_EXACT_INT
    ok &= p3
    print(f"  [{'PASS' if p3 else 'FAIL'}] max total |in-weight| = {tmax:,} "
          f"< 2^24 = {FP32_EXACT_INT:,}")
    print(f"         headroom: {FP32_EXACT_INT / max(tmax, 1):,.0f}x\n")

    if not ok:
        print("THEOREM DOES NOT HOLD for this connectome. Synaptic accumulation")
        print("is order-DEPENDENT; a parallel fan-out will not be reproducible.")
        return 1

    # empirical confirmation: permute the accumulation order
    print(f"EMPIRICAL ({N_PERM} random accumulation orders each)")
    rng = np.random.default_rng(0)
    for nspk in (50, 500, 5000):
        src = np.sort(rng.choice(n, nspk, replace=False))
        sel = np.concatenate([np.arange(crow[j], crow[j + 1]) for j in src])
        tgt, w = post[sel], val[sel]

        ref = None
        same = True
        for _ in range(N_PERM):
            p = rng.permutation(sel.size)
            acc = np.zeros(n, dtype=np.float32)
            np.add.at(acc, tgt[p], w[p])
            if ref is None:
                ref = acc.copy()
            elif not np.array_equal(acc.view(np.uint32), ref.view(np.uint32)):
                same = False
                break

        exact = np.zeros(n, dtype=np.int64)
        np.add.at(exact, tgt, w.astype(np.int64))
        matches = np.array_equal(ref, exact.astype(np.float32))
        ok &= (same and matches)
        print(f"  [{'PASS' if same and matches else 'FAIL'}] {nspk:5,} spiking "
              f"neurons / {sel.size:9,} synapses: "
              f"order-independent={same}, equals exact integer sum={matches}")

    print()
    print("THEOREM HOLDS. The fan-out may be accumulated in ANY order -- CPU")
    print("threads, GPU atomicAdd, warp reductions -- and remains bit-identical.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
