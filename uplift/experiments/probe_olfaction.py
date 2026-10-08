"""Is odour identity preserved by the model's antennal lobe?

probe_senses.py found every antennal input (any glomerulus, temperature,
humidity) driving the whole antennal lobe to ~94 Hz. This asks, per drive
rate: (1) when does the AL leave the stimulated glomerulus, (2) how specific
are uniglomerular PN responses -- the fraction of PN spikes coming from the
PNs of the stimulated glomerulus -- and (3) how much of the AL is recruited.

    python -m uplift.experiments.probe_olfaction
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from uplift.anatomy import Atlas
from uplift.engine import UpliftEngine, ROOT

OUT = ROOT / "data" / "results" / "uplift"
GLOMS = ["DA1", "DA2", "V", "DM1", "VA1v", "DL3"]


def pn_glom(at):
    """uPN cell_type -> glomerulus (FlyWire uPN types are '<glom>_<tract>PN')."""
    m = at.select(cell_class="ALPN", cell_sub_class="uniglomerular")
    ct = at.ann.cell_type.to_numpy()[m]
    ids = at.sim_ids[m]
    gl = np.array([t.split("_")[0] for t in ct])
    return ids, gl


def main(rates=(10, 25, 50, 100, 150), ms=300.0):
    at = Atlas()
    orn_ids = sorted({int(x) for g in GLOMS for x in at[f"smell.{g}"].ids})
    e = UpliftEngine(sensory_ids=orn_ids, seed=5)
    pid, pgl = pn_glom(at)
    pslot = e.indices_of(pid)
    al = at.select(cell_class=["ALPN", "ALLN", "ALIN", "ALON"])
    al_slots = e.indices_of(at.sim_ids[al])
    kc = at["mb.KC"].idx(e)
    rows = []
    for g in GLOMS:
        pos = e.stim_positions(at[f"smell.{g}"].idx(e))
        own = pgl == g
        for r in rates:
            e.full_reset()
            e.clear_rates()
            e.set_rates(pos, float(r))
            # time course in 25 ms bins
            c0 = e.counts.copy()
            t_runaway = None
            for b in range(int(ms / 25)):
                prev = e.counts[al_slots].sum()
                e.run(25.0)
                frac = (e.counts[al_slots] > c0[al_slots]).mean()
                if t_runaway is None and frac > 0.5:
                    t_runaway = (b + 1) * 25.0
            d = e.counts - c0
            pn_sp = d[pslot]
            spec = pn_sp[own].sum() / max(pn_sp.sum(), 1)
            rows.append(dict(
                glom=g, orn_hz=r, n_own_pn=int(own.sum()),
                own_pn_hz=pn_sp[own].mean() / (ms / 1000),
                other_pn_hz=pn_sp[~own].mean() / (ms / 1000),
                pn_specificity=spec,
                al_recruited=float((d[al_slots] > 0).mean()),
                kc_active=float((d[kc] > 0).mean()),
                t_al_over_half_ms=t_runaway))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "probe_olfaction.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
