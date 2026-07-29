"""Run delay-window stepping on the REAL FlyWire connectome.

This is the measurement that settles whether window stepping helps at the scale
and sparsity that actually matters. Everything before this was a toy network
that could not be held at the real firing regime (~31 spikes/window).

Reports the per-window cost breakdown so the O(N) predict/certify part can be
separated from the O(segments x candidates) repair part -- that split is what
determines whether the method wins here.
"""
import sys, time, pickle
from pathlib import Path
import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from window_step import TAU_M, TAU_S, THETA, DELAY, kappa, free_window_map, certified_no_spike

SUGAR = [720575940624963786,720575940630233916,720575940637568838,720575940638202345,
720575940617000768,720575940630797113,720575940632889389,720575940621754367,
720575940621502051,720575940640649691,720575940639332736,720575940616885538,
720575940639198653,720575940639259967,720575940617937543,720575940632425919,
720575940633143833,720575940612670570,720575940628853239,720575940611875570,
720575940629176663]


def load():
    import pandas as pd
    d = ROOT / "data"
    comp = pd.read_csv(d / "2025_Completeness_783.csv", index_col=0)
    ids = np.asarray(comp.index, dtype=np.int64)
    n = len(ids)
    fo = __import__("torch").load(d / "fanout_csc.pt")
    crow = fo["crow"].numpy().astype(np.int64)
    post = fo["post"].numpy().astype(np.int64)
    val = fo["val"].numpy().astype(np.float64) * 0.275     # wScale
    return n, ids, crow, post, val


if __name__ == "__main__":
    t0 = time.perf_counter()
    n, ids, crow, post, val = load()
    print(f"connectome: {n:,} neurons, {len(post):,} synapses "
          f"(loaded in {time.perf_counter()-t0:.1f}s)\n")

    src = np.repeat(np.arange(n), np.diff(crow))
    t0 = time.perf_counter()
    W = sp.csr_matrix((val, (post, src)), shape=(n, n))
    print(f"W built: {W.nnz:,} nnz  ({time.perf_counter()-t0:.1f}s)")

    w_pos_in = np.zeros(n)
    np.add.at(w_pos_in, post, np.maximum(val, 0.0))
    print(f"positive in-weight: median {np.median(w_pos_in):.2f}  "
          f"mean {w_pos_in.mean():.2f}  max {w_pos_in.max():.1f}")

    # --- cost of the O(N) parts, which every window pays ---
    u = np.zeros(n); g = np.zeros(n)
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        ue, ge = free_window_map(u, g, DELAY)
    t_pred = (time.perf_counter() - t0) / reps

    t0 = time.perf_counter()
    for _ in range(reps):
        safe = certified_no_spike(u, g, w_pos_in, THETA, DELAY)
    t_cert = (time.perf_counter() - t0) / reps

    # --- delivery, two ways ---
    # (a) sigma as a FULL sparse mat-vec: touches all 15.1M nnz every window
    #     no matter how few neurons fired. This is the same dense-spmv trap the
    #     event-driven work already removed from run_pytorch.py.
    sig = np.zeros(n); sig[:200] = 1.0
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = W @ sig
    t_spmv = (time.perf_counter() - t0) / reps

    # (b) sigma applied EVENT-DRIVEN: scatter only from sources that fired,
    #     weighted by that source's sigma. Same result, O(spikes x fanout).
    n_spk = 31                                   # ~1.75 spikes/step x 18 steps
    rng = np.random.default_rng(0)
    fired = rng.choice(n, n_spk, replace=False)
    sig_g = rng.random(n_spk); sig_u = rng.random(n_spk)
    t0 = time.perf_counter()
    for _ in range(reps):
        dg = np.zeros(n); du = np.zeros(n)
        for k, j in enumerate(fired):
            lo, hi = crow[j], crow[j + 1]
            if hi > lo:
                np.add.at(dg, post[lo:hi], val[lo:hi] * sig_g[k])
                np.add.at(du, post[lo:hi], val[lo:hi] * sig_u[k])
    t_evt = (time.perf_counter() - t0) / reps
    fanout = sum(crow[j + 1] - crow[j] for j in fired)
    print(f"\ndelivery, {n_spk} spikes/window ({fanout:,} synapses touched):")
    print(f"  (a) 2 x full W @ sigma    {2*t_spmv*1000:8.2f} ms   "
          f"({2*W.nnz:,} ops)")
    print(f"  (b) event-driven sigma    {t_evt*1000:8.2f} ms   "
          f"({2*fanout:,} ops)  -> {2*t_spmv/t_evt:.0f}x cheaper")

    print(f"\nper-window cost at full scale:")
    print(f"  predict (free_window_map) {t_pred*1000:8.2f} ms")
    print(f"  certify (silence bound)   {t_cert*1000:8.2f} ms")
    print(f"  deliver, event-driven     {t_evt*1000:8.2f} ms")
    win_fixed = t_pred + t_cert + t_evt
    print(f"  --------------------------------------")
    print(f"  total per window          {win_fixed*1000:8.2f} ms")
    print(f"  -> per simulated second   {win_fixed*(1000/DELAY):8.2f} s"
          f"   ({int(1000/DELAY)} windows)")

    # --- what fraction does the bound certify from a resting start? ---
    print(f"\ncertified-silent from rest: {100*safe.mean():.2f}%")
    thresh_w = THETA / 0.0720849531
    print(f"a resting neuron needs sum(w+) > {thresh_w:.1f} in one window to even "
          f"be a candidate")
    print(f"neurons meeting that: {int((w_pos_in > thresh_w).sum()):,} "
          f"({100*(w_pos_in > thresh_w).mean():.2f}%)")

    # --- grid baseline for the same O(N) work ---
    t0 = time.perf_counter()
    for _ in range(reps):
        xm, xs = np.exp(-0.1 / TAU_M), np.exp(-0.1 / TAU_S)
        _u = xm * u + (g * (TAU_S / (TAU_M - TAU_S))) * (xm - xs)
        _g = xs * g
        _ = _u > THETA
    t_grid_step = (time.perf_counter() - t0) / reps
    print(f"\ngrid dt=0.1ms: {t_grid_step*1000:.2f} ms/step "
          f"-> {t_grid_step*10000:.2f} s per simulated second (neuron update only)")
    print(f"window fixed cost is {t_grid_step*10000/(win_fixed*(1000/DELAY)):.2f}x "
          f"cheaper on the O(N) part alone")
