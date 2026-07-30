"""Audit D (real engine): measure the implemented axonal delay end-to-end."""
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(r"C:\Users\neogo\Documents\FlyBrain")
sys.path.insert(0, str(ROOT / "flyloop"))
from brain_engine import BrainEngine
DATA = str(ROOT / "data")

for mode in ("dense", "active"):
    e = BrainEngine(data_dir=DATA, stim_ids=None, seed=0)
    e.active_mode = (mode == "active"); e.inplace = True
    e._rates = torch.zeros(0)
    # pick a neuron with fan-out
    src = int(torch.argmax(e.fo_crow[1:] - e.fo_crow[:-1]))
    tgts = e.fo_post[e.fo_crow[src]:e.fo_crow[src+1]]
    print(f"[{mode}] L = {e.L}   source {src} fanout {tgts.numel()}")
    e.v[src] = -40.0                 # above threshold -> spikes on step 1
    e.active[src] = True
    e._idx = e.active.nonzero(as_tuple=True)[0]
    t_spike = None; t_g = None
    for k in range(1, 40):
        sp = e.step()
        n_sp = int((sp > 0).sum()) if sp.dtype != torch.int64 else sp.numel()
        if t_spike is None and n_sp:
            t_spike = k
        if t_g is None and float(e.g[tgts].abs().sum()) > 0:
            t_g = k
            break
    print(f"[{mode}] spike emitted on step {t_spike} (t={0.1*t_spike:.1f} ms); "
          f"target g first changes on step {t_g} (t={0.1*t_g:.1f} ms)  "
          f"=> implemented delay {0.1*(t_g-t_spike):.1f} ms  (spec: 1.8 ms)")
