"""What does the unmodified fly brain do with each sense?

Drives each sensory population in turn (Poisson, the paper's convention) and
records the response of every region and of the populations an uplift reads:
dopamine neurons (does the fly have its own teaching signal?), MBONs, the
descending neurons by behaviour, and MN9. The model has no spontaneous
activity, so the response IS the signal -- there is no baseline to subtract.

    python -m uplift.experiments.probe_senses [ms] [hz]

Writes data/results/uplift/probe_senses.csv and prints the table.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from uplift.anatomy import Atlas, REGIONS
from uplift.engine import UpliftEngine, Readout, ROOT

OUT = ROOT / "data" / "results" / "uplift"

CHANNELS = {
    "sugar (annotated LB3c)":  ["taste.sugar"],
    "sugar (Shiu 21 ids)":     ["taste.shiu_sugar"],
    "bitter":                  ["taste.bitter"],
    "water":                   ["taste.water"],
    "high salt":               ["taste.high_salt"],
    "CO2 (ORN_V)":             ["smell.V"],
    "geosmin (ORN_DA2)":       ["smell.DA2"],
    "cVA (ORN_DA1)":           ["smell.DA1"],
    "vinegar-like (DM1+DM4)":  ["smell.DM1", "smell.DM4"],
    "sound (JO-A+B)":          ["hearing_wind.JO_A", "hearing_wind.JO_B"],
    "wind/gravity (JO-C+E)":   ["hearing_wind.JO_C", "hearing_wind.JO_E"],
    "heat":                    ["temperature.heating"],
    "cold":                    ["temperature.cold", "temperature.cooling"],
    "dry":                     ["humidity.dry"],
    "humid":                   ["humidity.humid", "humidity.moist"],
}
READ = ["mb.PAM", "mb.PPL1", "mb.MBON", "mb.KC", "lh.all", "cx.all",
        "dn.walk_forward", "dn.turn", "dn.walk_backward", "dn.escape",
        "dn.groom", "dn.all", "motor.MN9", "motor.all"]


def sensory_union(at):
    ids = set()
    for pops in CHANNELS.values():
        for p in pops:
            ids.update(int(x) for x in at[p].ids)
    return sorted(ids)


def main(ms=500.0, hz=150.0):
    at = Atlas()
    eng = UpliftEngine(sensory_ids=sensory_union(at), seed=3)
    reg_slots = at.region_slots(eng)
    groups = {p: at[p].idx(eng) for p in READ}
    for i, (k, _, _) in enumerate(REGIONS):
        groups[f"region.{k}"] = np.flatnonzero(reg_slots == i)
    ro = Readout(eng, groups)
    rows = []
    for name, pops in CHANNELS.items():
        eng.full_reset()
        eng.clear_rates()
        slots = np.concatenate([at[p].idx(eng) for p in pops])
        eng.set_rates(eng.stim_positions(slots), hz)
        ro.reset()
        t0 = time.perf_counter()
        nsp = eng.run(ms)
        wall = time.perf_counter() - t0
        r = ro.read()
        r.update(channel=name, n_driven=int(slots.size), spikes=int(nsp),
                 ms_per_step=wall / (ms / eng.dt) * 1e3)
        rows.append(r)
        print(f"{name:26s} n={slots.size:4d} spikes={nsp:7d} "
              f"PAM={r['mb.PAM']:6.2f} PPL1={r['mb.PPL1']:6.2f} "
              f"MBON={r['mb.MBON']:6.2f} MN9={r['motor.MN9']:6.1f} "
              f"fwd={r['dn.walk_forward']:5.1f} groom={r['dn.groom']:5.1f} "
              f"({wall:.1f}s)", flush=True)
    df = pd.DataFrame(rows).set_index("channel")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "probe_senses.csv")
    print(f"\nsaved {OUT / 'probe_senses.csv'}")
    return df


if __name__ == "__main__":
    a = sys.argv[1:]
    main(float(a[0]) if a else 500.0, float(a[1]) if len(a) > 1 else 150.0)
