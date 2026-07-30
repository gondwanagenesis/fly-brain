"""Audit G: the two engines are NEVER cross-checked. brain_engine.py claims
'Numerically equivalent to the original'; verify.py only compares brain_engine
against itself. Test it directly, with zero Poisson rate so the different RNG
consumption cannot confound the comparison."""
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(r"C:\Users\neogo\Documents\FlyBrain")
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "flyloop"))
import run_pytorch as rp
from benchmark import path_comp, path_con
from brain_engine import BrainEngine

STEPS = 120
SUGAR = [720575940624963786,720575940630233916,720575940637568838,720575940638202345,
720575940617000768,720575940630797113,720575940632889389,720575940621754367]

flyid2i, _ = rp.get_hash_tables(str(path_comp))
exc = [flyid2i[n] for n in SUGAR if n in flyid2i]
W = rp.get_weights(str(path_con), str(path_comp), str(path_con.parent), csr=True)
N = W.shape[0]

def upstream():
    model = rp.TorchModel(1, N, rp.DT, rp.MODEL_PARAMS, W, exc_indices=exc, device='cpu')
    cond, buf, spikes, v, refrac = model.state_init()
    v[0, exc] = -40.0                       # ignite
    rates = torch.zeros(1, N)
    gen = torch.Generator(device='cpu'); gen.manual_seed(1234)
    out = []
    with torch.no_grad():
        for t in range(STEPS):
            cond, buf, spikes, v, refrac = model(rates, cond, buf, spikes, v, refrac, generator=gen)
            nz = (spikes[0] > 0).nonzero(as_tuple=True)[0]
            for i in nz.tolist(): out.append((t+1, i))
    return out, v[0].clone(), cond[0].clone()

def engine(mode):
    e = BrainEngine(data_dir=str(ROOT/"data"), stim_ids=SUGAR, seed=1234)
    e.active_mode = (mode == "active"); e.inplace = True
    e.inject(0.0)
    e.v[e.stim_idx] = -40.0
    e.active[e.stim_idx] = True
    e._idx = e.active.nonzero(as_tuple=True)[0]
    out = []
    for t in range(STEPS):
        e.step()
        idx = e._spike_idx if (e.active_mode and not e._dense_fallback) else (e.spikes>0).nonzero(as_tuple=True)[0]
        for i in idx.tolist(): out.append((t+1, i))
    return out, e.v.clone(), e.g.clone()

up, vu, gu = upstream()
for mode in ("dense", "active"):
    en, ve, ge = engine(mode)
    same = (up == en)
    print(f"brain_engine[{mode}] vs run_pytorch: upstream {len(up)} spikes, "
          f"engine {len(en)} spikes -> {'IDENTICAL' if same else 'MISMATCH'}")
    if not same:
        su, se = set(up), set(en)
        print(f"   only-upstream {len(su-se)}   only-engine {len(se-su)}")
        d = sorted(su ^ se)[:6]
        print("   first differing (step, neuron):", d)
    print(f"   max |v diff| = {float((vu-ve).abs().max()):.6g}   "
          f"max |g diff| = {float((gu-ge).abs().max()):.6g}")
