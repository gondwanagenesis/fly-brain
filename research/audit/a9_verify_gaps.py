"""Audit F: which fallback path do verify.py's own regimes take, and does the
active/dense equivalence survive prune_every=1 and full state comparison?"""
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(r"C:\Users\neogo\Documents\FlyBrain")
sys.path.insert(0, str(ROOT / "flyloop"))
import brain_engine as BE
from brain_engine import BrainEngine
DATA = str(ROOT / "data")

SUGAR = [720575940624963786,720575940630233916,720575940637568838,720575940638202345,
720575940617000768,720575940630797113,720575940632889389,720575940621754367,
720575940621502051,720575940640649691,720575940639332736,720575940616885538,
720575940639198653,720575940639259967,720575940617937543,720575940632425919,
720575940633143833,720575940612670570,720575940628853239,720575940629176663,
720575940611875570]
_all = BrainEngine(data_dir=DATA, stim_ids=SUGAR[:1], seed=0).i2flyid
rng = np.random.default_rng(0)

# instrument which fallback path fires
paths = []
orig_top = BrainEngine.step_active
def traced(self, record=False):
    N = self.N
    if not self._dense_fallback and (self._idx.numel() > self.dense_switch*N
                                     or self._spike_idx.numel() > self.spike_switch*N):
        paths.append(("TOP (no _rebuild_dense_state)", int(self._idx.numel()),
                      int(self._spike_idx.numel())))
    return orig_top(self, record=record)
BrainEngine.step_active = traced
orig_rebuild = BrainEngine._rebuild_dense_state
def traced_rb(self, tgt, vals, wS):
    paths.append(("GROWTH (rebuilds)", int(self._idx.numel()), int(self._spike_idx.numel())))
    return orig_rebuild(self, tgt, vals, wS)
BrainEngine._rebuild_dense_state = traced_rb

def regimes():
    yield "sugar GRNs (21 @200Hz)", SUGAR, 200.0
    yield "silent (0 drive)", SUGAR, 0.0
    for n in (100, 1000, 10000):
        yield f"broad ({n} @100Hz)", rng.choice(_all, n, replace=False).tolist(), 100.0
    yield "whole-brain (40k @200Hz)", rng.choice(_all, 40000, replace=False).tolist(), 200.0

print("=== which fallback path does each verify.py regime take? ===")
for name, ids, rate in regimes():
    paths.clear()
    e = BrainEngine(data_dir=DATA, stim_ids=ids, seed=1234)
    e.active_mode = True; e.inplace = True; e.inject(rate)
    for _ in range(300): e.step()
    print(f"  {name:<26} fallback: {paths[:1] if paths else 'never (stayed sparse)'}")

print("\n=== sugar regime: prune_every=1 + FULL STATE comparison, 3000 steps ===")
def run(mode, prune_every, steps):
    e = BrainEngine(data_dir=DATA, stim_ids=SUGAR, seed=1234)
    e.active_mode = (mode=="active"); e.inplace = True
    e.prune_every = prune_every
    e.inject(200.0)
    for _ in range(steps): e.step(record=True)
    df = e.spikes_dataframe()
    o = np.lexsort([df.neuron_index.to_numpy(), df.time_ms.to_numpy()]) if len(df) else np.array([],int)
    sig = (df.time_ms.to_numpy()[o], df.neuron_index.to_numpy()[o]) if len(df) else (np.array([]),np.array([]))
    return e, sig
for pe in (64, 1):
    ed, sd = run("dense", pe, 3000)
    ea, sa = run("active", pe, 3000)
    spikes_ok = np.array_equal(sd[0],sa[0]) and np.array_equal(sd[1],sa[1])
    v_ok  = torch.equal(ed.v, ea.v)
    g_ok  = torch.equal(ed.g, ea.g)
    rf_ok = torch.equal(ed.refrac, ea.refrac)
    buf_ok= torch.equal(ed.buf, ea.buf)
    spk_ok= torch.equal(ed.spikes, ea.spikes)
    print(f"  prune_every={pe:>3}  spikes={'EXACT' if spikes_ok else 'MISMATCH'}  "
          f"v={'ok' if v_ok else 'DIFF'}  g={'ok' if g_ok else 'DIFF'}  "
          f"refrac={'ok' if rf_ok else 'DIFF'}  buf={'ok' if buf_ok else 'DIFF'}  "
          f"self.spikes={'ok' if spk_ok else 'DIFF'}")
    if not rf_ok:
        d = (ed.refrac - ea.refrac)
        print(f"       refrac differs on {int((d!=0).sum())} neurons, max |diff| = {float(d.abs().max())}")
