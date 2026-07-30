"""Audit G2: cross-engine check with a real cascade (2000 ignited neurons, 400 steps)."""
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(r"C:\Users\neogo\Documents\FlyBrain")
sys.path.insert(0, str(ROOT / "code")); sys.path.insert(0, str(ROOT / "flyloop"))
import run_pytorch as rp
from benchmark import path_comp, path_con
from brain_engine import BrainEngine
STEPS = 400
flyid2i, _ = rp.get_hash_tables(str(path_comp))
W = rp.get_weights(str(path_con), str(path_comp), str(path_con.parent), csr=True)
N = W.shape[0]
rng = np.random.default_rng(11)
ig = np.sort(rng.choice(N, 2000, replace=False))
i2fly = np.asarray(list(flyid2i.keys()), dtype=np.int64)

def upstream():
    model = rp.TorchModel(1, N, rp.DT, rp.MODEL_PARAMS, W, exc_indices=[], device='cpu')
    cond, buf, spikes, v, refrac = model.state_init()
    v[0, torch.as_tensor(ig)] = -40.0
    rates = torch.zeros(1, N)
    gen = torch.Generator(device='cpu'); gen.manual_seed(1234)
    out = []
    with torch.no_grad():
        for t in range(STEPS):
            cond, buf, spikes, v, refrac = model(rates, cond, buf, spikes, v, refrac, generator=gen)
            for i in (spikes[0] > 0).nonzero(as_tuple=True)[0].tolist(): out.append((t+1, i))
    return out

def engine(mode):
    e = BrainEngine(data_dir=str(ROOT/"data"), stim_ids=None, seed=1234)
    e.active_mode = (mode == "active"); e.inplace = True
    t = torch.as_tensor(ig)
    e.v[t] = -40.0; e.active[t] = True
    e._idx = e.active.nonzero(as_tuple=True)[0]
    out = []
    for k in range(STEPS):
        e.step()
        idx = e._spike_idx if (e.active_mode and not e._dense_fallback) else (e.spikes>0).nonzero(as_tuple=True)[0]
        for i in idx.tolist(): out.append((k+1, i))
    return out, e._dense_fallback

up = upstream()
print("upstream spikes:", len(up))
for mode in ("dense", "active"):
    en, fb = engine(mode)
    same = up == en
    print(f"brain_engine[{mode}] {len(en)} spikes  fallback={fb}  -> "
          f"{'IDENTICAL' if same else 'MISMATCH'}")
    if not same:
        su, se = set(up), set(en)
        print("   only-upstream", len(su-se), " only-engine", len(se-su),
              " first diffs", sorted(su ^ se)[:5])
