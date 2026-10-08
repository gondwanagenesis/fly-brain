"""Calibrate the synaptic weight scale of a new connectome against FlyWire.

Shiu et al.'s wScale (0.275 mV per synapse) was fitted on FlyWire. The male
CNS detects about twice as many synapses per connection (124M synapses on
165k neurons vs 54M on 139k), so the same scale over-drives it; others found
it runs away (review 10). This measures, per global gain, the responses the
published model is known for, with the same stimuli on both connectomes:

    rest     no input, 1 s                 -> mean rate, active fraction
    sugar    labellar sugar GRNs 150 Hz     -> MN9 (proboscis) rate
    bitter   bitter GRNs 150 Hz             -> MN9 (should stay ~0)
    wind     JO-C/E 150 Hz                  -> grooming DN rate
    shadow   LC4/LPLC2 150 Hz               -> giant fibre (DNp01) rate
and the fraction of the brain active in each (ignition check).

    SUPERFLY_CONNECTOME=male_cns python -m superfly.experiments.calibrate_connectome 1.0 0.75 0.55 0.45
    python -m superfly.experiments.calibrate_connectome 1.0          # FlyWire reference
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from superfly.anatomy import CONNECTOME
from superfly.bridge import OUT
from superfly.fly import SuperFly

STIM = {"rest": {}, "sugar": {"sugar": 1.0}, "bitter": {"bitter": 1.0},
        "wind": {"wind": 1.0}, "shadow": {"shadow": 1.0}}


def main(gains):
    fly = SuperFly(word_pns=0, plasticity=False, seed=0)
    e = fly.e
    rows = []
    t0 = time.perf_counter()
    for g in gains:
        e.set_gain(g)
        for name, c in STIM.items():
            fly.reset()
            e.rng = np.random.default_rng(1)
            tw = time.perf_counter()
            obs = fly.present(c, ms=600.0 if name != "rest" else 1000.0, settle_ms=100.0)
            wall = time.perf_counter() - tw
            w = fly._last_window
            r = {"gain": g, "stim": name, "MN9": round(obs.actions["feed"], 1),
                 "groom": round(obs.actions["groom"], 1), "escape": round(obs.actions["escape"], 1),
                 "turn": round(obs.actions["turn"], 1), "walk_forward": round(obs.actions["walk_forward"], 1),
                 "mean_hz": round(float(w[:e.n_native].mean()), 3),
                 "active_frac": round(float((w[:e.n_native] > 0).mean()), 4),
                 "realtime": round((obs.window_ms + 100.0) / 1000.0 / wall, 3)}
            rows.append(r)
            print(r, flush=True)
    rep = {"connectome": CONNECTOME, "N": int(e.n_native), "rows": rows,
           "wall_s": round(time.perf_counter() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration.json").write_text(json.dumps(rep, indent=1))
    print("wrote", OUT / "calibration.json")


if __name__ == "__main__":
    main([float(x) for x in sys.argv[1:]] or [1.0])
