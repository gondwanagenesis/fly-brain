"""Verify the active-set optimisation across many stimulation regimes.

For each regime: run the dense kernel and the active-set kernel from identical
state/seed and require BIT-IDENTICAL spike trains, then report the speedup.
Regimes deliberately span tiny -> whole-brain drive so we prove the thing is
safe for every use of the network, not just the sugar demo.
"""
import sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "flyloop"))
from brain_engine import BrainEngine

SUGAR = [720575940624963786,720575940630233916,720575940637568838,720575940638202345,
720575940617000768,720575940630797113,720575940632889389,720575940621754367,
720575940621502051,720575940640649691,720575940639332736,720575940616885538,
720575940639198653,720575940639259967,720575940617937543,720575940632425919,
720575940633143833,720575940612670570,720575940628853239,720575940629176663,
720575940611875570]
P9 = [720575940627652358, 720575940635872101]

DATA = str(ROOT / "data")
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500

# a pool of real ids to build broad-stimulation regimes from
_all = BrainEngine(data_dir=DATA, stim_ids=SUGAR[:1], seed=0).i2flyid
rng = np.random.default_rng(0)


def regimes():
    yield "sugar GRNs (21 @200Hz)", SUGAR, 200.0
    yield "P9 walking (2 @100Hz)", P9, 100.0
    yield "single neuron (1 @200Hz)", SUGAR[:1], 200.0
    yield "silent (0 drive)", SUGAR, 0.0
    for n in (100, 1000, 10000):
        yield f"broad ({n} @100Hz)", rng.choice(_all, n, replace=False).tolist(), 100.0
    # worst case: drive a large fraction hard -> must fall back to dense
    yield "whole-brain (40k @200Hz)", rng.choice(_all, 40000, replace=False).tolist(), 200.0


def run(mode, ids, rate, steps):
    e = BrainEngine(data_dir=DATA, stim_ids=ids, seed=1234)
    e.active_mode = (mode == "active")
    e.inplace = True
    e.inject(rate)
    t0 = time.perf_counter()
    for _ in range(steps):
        e.step(record=True)
    dt = time.perf_counter() - t0
    df = e.spikes_dataframe()
    if len(df):
        key = np.stack([df.time_ms.to_numpy(), df.neuron_index.to_numpy()])
        order = np.lexsort(key)
        sig = (df.time_ms.to_numpy()[order], df.neuron_index.to_numpy()[order])
    else:
        sig = (np.array([]), np.array([]))
    frac = float(e.active.sum()) / e.N if hasattr(e, "active") else 1.0
    return dt, sig, len(df), frac


print(f"{'regime':<26} {'spikes':>8} {'active%':>8} "
      f"{'dense s':>9} {'active s':>9} {'speedup':>8}  match")
print("-" * 82)
tot_d = tot_a = 0.0
all_ok = True
for name, ids, rate in regimes():
    d_t, d_sig, d_n, _ = run("dense", ids, rate, STEPS)
    a_t, a_sig, a_n, frac = run("active", ids, rate, STEPS)
    ok = (d_n == a_n
          and np.array_equal(d_sig[0], a_sig[0])
          and np.array_equal(d_sig[1], a_sig[1]))
    all_ok &= ok
    tot_d += d_t; tot_a += a_t
    print(f"{name:<26} {d_n:>8d} {100*frac:>7.1f}% {d_t:>9.3f} {a_t:>9.3f} "
          f"{d_t/a_t:>7.2f}x  {'EXACT' if ok else 'MISMATCH'}")

print("-" * 82)
print(f"{'TOTAL':<26} {'':>8} {'':>8} {tot_d:>9.3f} {tot_a:>9.3f} "
      f"{tot_d/tot_a:>7.2f}x  {'ALL EXACT' if all_ok else 'FAILURES'}")
sys.exit(0 if all_ok else 1)
