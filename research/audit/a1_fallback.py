"""Audit A: active->dense fallback via the spike_switch branch does NOT
rebuild self.spikes, so the last active step's spikes are silently dropped
(no synaptic fan-out, no refractory reset)."""
import sys, os
from pathlib import Path
import numpy as np, torch
ROOT = Path(r"C:\Users\neogo\Documents\FlyBrain")
sys.path.insert(0, str(ROOT / "flyloop"))
from brain_engine import BrainEngine

DATA = str(ROOT / "data")
_all = BrainEngine(data_dir=DATA, stim_ids=None, seed=0).i2flyid
rng = np.random.default_rng(7)
# 6000 stim neurons  -> active set 6000 < 0.08*138639 = 11091
# 1000 Hz            -> ~600 spikes/step > 0.004*138639 = 554
IDS = rng.choice(_all, 6000, replace=False).tolist()
RATE = 1000.0
STEPS = 80

def run(mode, trace=False):
    e = BrainEngine(data_dir=DATA, stim_ids=IDS, seed=1234)
    e.active_mode = (mode == "active"); e.inplace = True
    e.inject(RATE)
    for k in range(STEPS):
        if trace:
            print(f"  step {k}: active={e._idx.numel()} spike_idx={e._spike_idx.numel()} "
                  f"dense_fallback={e._dense_fallback} sum(self.spikes)={float(e.spikes.sum())}")
        e.step(record=True)
    df = e.spikes_dataframe()
    if len(df)==0: return np.array([]), np.array([])
    o = np.lexsort([df.neuron_index.to_numpy(), df.time_ms.to_numpy()])
    return df.time_ms.to_numpy()[o], df.neuron_index.to_numpy()[o]

print("N =", len(_all), " dense_switch*N =", 0.08*len(_all), " spike_switch*N =", 0.004*len(_all))
print("active trace:")
at, an = run("active", trace=False)
dt_, dn = run("dense")
print("dense  spikes:", len(dt_))
print("active spikes:", len(at))
ok = np.array_equal(at, dt_) and np.array_equal(an, dn)
print("MATCH" if ok else "*** MISMATCH ***")
if not ok:
    # first differing timestep
    for t in sorted(set(np.concatenate([dt_, at]).tolist())):
        a = np.sort(an[at==t]); d = np.sort(dn[dt_==t])
        if not np.array_equal(a,d):
            print(f"first divergence at t={t}: dense n={d.size} active n={a.size}")
            break
