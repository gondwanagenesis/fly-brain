"""Where, and by how much, does the native kernel leave the torch reference
on this host? (verify_all.py reports STATE@N without saying.)

Runs the saturating regime (diverges earliest) in lockstep, unpermuted, and at
the first state mismatch reports: which neurons, v or g, the ULP distance,
whether each sits on ATen's FMA or scalar-tail side of the seam, and whether
the reference value equals the fused (FMA) or unfused rounding of the update.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "flyloop"))
from brain_engine import BrainEngine            # noqa: E402
from native_engine import NativeBrainEngine     # noqa: E402

D = str(ROOT / "data")


def ulp(a, b):
    ia, ib = a.view(np.int32).astype(np.int64), b.view(np.int32).astype(np.int64)
    return np.abs(ia - ib)


def main(n_stim=40000, rate=200.0, steps=200, seed=0):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    import pandas as pd
    ids = pd.read_csv(ROOT / "data" / "2025_Completeness_783.csv", index_col=0).index
    stim = rng.choice(np.asarray(ids), n_stim, replace=False).tolist()
    ref = BrainEngine(data_dir=D, device="cpu", stim_ids=stim, seed=seed)
    nat = NativeBrainEngine(data_dir=D, stim_ids=stim, seed=seed, reorder=None)
    ref.inject(rate)
    nat.inject(rate)
    print("torch", torch.__version__, "threads", torch.get_num_threads(),
          "cap", torch.backends.cpu.get_cpu_capability(), "tail_w", nat.tail_w,
          "chunks", nat.chunks.tolist())
    fma = np.zeros(nat.N, bool)
    for c in range(nat.n_chunks):
        lo, hi = int(nat.chunks[c]), int(nat.chunks[c + 1])
        fma[lo:lo + ((hi - lo) // nat.tail_w) * nat.tail_w] = True
    for s in range(steps):
        v_prev, g_prev = ref.v.numpy().copy(), ref.g.numpy().copy()
        ref.step()
        nat.step()
        rv, rg = ref.v.numpy(), ref.g.numpy()
        dv = np.flatnonzero(rv.view(np.uint32) != nat.v.view(np.uint32))
        dg = np.flatnonzero(rg.view(np.uint32) != nat.g.view(np.uint32))
        if dv.size or dg.size:
            print(f"first divergence at step {s}: {dv.size} v, {dg.size} g")
            for name, d, r, n in (("v", dv, rv, nat.v), ("g", dg, rg, nat.g)):
                if not d.size:
                    continue
                u = ulp(r[d], n[d])
                print(f"  {name}: ulp max {u.max()} median {np.median(u)}; "
                      f"on FMA side {fma[d].sum()}/{d.size}")
                for i in d[:8]:
                    print(f"    i={i} ref={r[i]!r} nat={n[i]!r} fma_side={fma[i]} "
                          f"v_prev={v_prev[i]!r} g_prev={g_prev[i]!r}")
            # which rounding did the reference use for v? recompute both
            c_mem = np.float32(nat.c_mem)
            for i in dv[:8]:
                vp, gp = np.float32(v_prev[i]), np.float32(g_prev[i])
                x = np.float32(np.float32(gp) - (vp - np.float32(nat.v_rest)))
                unf = np.float32(vp + np.float32(x * c_mem))
                fus = np.float32(np.float64(vp) + np.float64(x) * np.float64(c_mem))
                print(f"    i={i}: unfused={unf!r} fused={fus!r} ref={rv[i]!r} "
                      f"nat={nat.v[i]!r}")
            return
    print(f"no divergence in {steps} steps")


if __name__ == "__main__":
    main(*(int(a) if i in (0, 2) else float(a) for i, a in enumerate(sys.argv[1:])))
