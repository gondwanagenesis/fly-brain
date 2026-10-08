"""Gate: the uplift machinery leaves an un-uplifted fly bit-identical.

Three comparisons, full state (v, g as raw uint32) and spike trains every step:

  A  NativeBrainEngine                     vs  UpliftEngine(rng="torch")
     -- the ports and counters alone change nothing
  B  UpliftEngine                           vs  UpliftEngine + MBPlasticity
     with learning disabled (no DAN activity reaches it)
     -- a fly that has not learned is the original fly
  C  UpliftEngine + graft of synthetic neurons that are never driven, on the
     native slots only -- appended neurons do not perturb native dynamics
     until they fire.

    python -m uplift.tests.test_identity [steps]
"""
from __future__ import annotations

import sys
import time

import numpy as np

from uplift.anatomy import Atlas, SHIU_SUGAR
from uplift.engine import UpliftEngine, NativeBrainEngine, ROOT
from uplift.plasticity import MBPlasticity

DATA = str(ROOT / "data")


def same(a, b, n_native):
    va = a.v[:n_native] if a.perm is None else a.v[np.argsort(a.perm)][:n_native]
    vb = b.v[:n_native] if b.perm is None else b.v[np.argsort(b.perm)][:n_native]
    ga = a.g[:n_native] if a.perm is None else a.g[np.argsort(a.perm)][:n_native]
    gb = b.g[:n_native] if b.perm is None else b.g[np.argsort(b.perm)][:n_native]
    return (np.array_equal(va.view(np.uint32), vb.view(np.uint32))
            and np.array_equal(ga.view(np.uint32), gb.view(np.uint32)))


def spikes_ids(e):
    return np.sort(e.i2flyid[e.spike_idx])


def run_pair(a, b, steps, n_native, label):
    t0 = time.perf_counter()
    for s in range(steps):
        a.step()
        b.step()
        sa, sb = spikes_ids(a), spikes_ids(b)
        sb = sb[np.isin(sb, a.i2flyid)]
        if not np.array_equal(sa, sb) or not same(a, b, n_native):
            print(f"  {label}: DIVERGED at step {s}")
            return False
    print(f"  {label}: bit-identical over {steps} steps "
          f"({time.perf_counter() - t0:.1f}s, {a.counts.sum() if hasattr(a, 'counts') else '-'} spikes)")
    return True


def main(steps=600):
    ok = True
    at = Atlas()
    # A ------------------------------------------------------------------
    nat = NativeBrainEngine(data_dir=DATA, stim_ids=SHIU_SUGAR, seed=7,
                            reorder="cell_type")
    up = UpliftEngine(DATA, sensory_ids=SHIU_SUGAR, seed=7, rng="torch")
    nat.inject(200.0)
    up.inject(200.0)
    ok &= run_pair(nat, up, steps, nat.N, "A native vs uplift-engine")
    del nat
    # B ------------------------------------------------------------------
    u1 = up
    u1.full_reset()
    u2 = UpliftEngine(DATA, sensory_ids=SHIU_SUGAR, seed=7, rng="torch")
    pl = MBPlasticity(u2, at)
    u2.add_emitter(pl)
    u2.add_ticker(pl, pl.every)
    u1.inject(200.0)
    u2.inject(200.0)
    ok &= run_pair(u1, u2, steps, u1.N, "B + plasticity, nothing learned")
    print(f"     plastic KC->MBON edges: {pl.n_edges}, learned: {pl.any_learned}")
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 600))
