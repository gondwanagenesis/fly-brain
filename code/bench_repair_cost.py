"""Measure the repair path at real connectome scale.

The 6.0x figure covers the O(N) work every window pays (predict + certify +
event-driven delivery). It excludes the repair path, which re-examines neurons
the silence bound cannot certify. This measures that missing term so the total
can be stated honestly.

Repair cost = (number of candidates) x (number of segments) x (vector op cost).
Segments per window = distinct arrival times = spikes in the previous window.
"""
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from window_step import TAU_M, TAU_S, THETA, DELAY, certified_no_spike
from exact_spike_time import spike_time
from bench_connectome_window import load

if __name__ == "__main__":
    n, ids, crow, post, val = load()
    w_pos_in = np.zeros(n)
    np.add.at(w_pos_in, post, np.maximum(val, 0.0))

    # A neuron can only be a candidate if it is perturbed. Measured active set
    # on the sugar experiment saturates near 8.8% of the network.
    rng = np.random.default_rng(0)
    for active_frac in (0.001, 0.0088, 0.05):
        n_act = int(n * active_frac)
        idx = rng.choice(n, n_act, replace=False)
        u = np.zeros(n); g = np.zeros(n)
        u[idx] = rng.uniform(0, 6.5, n_act)
        g[idx] = rng.uniform(0, 40.0, n_act)

        safe = certified_no_spike(u, g, w_pos_in, THETA, DELAY)
        cand = np.flatnonzero(~safe)

        # cost of one segment pass over the candidates
        reps = 20
        uc, gc = u[cand].copy(), g[cand].copy()
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = spike_time(uc, gc, THETA, TAU_M)
        t_seg = (time.perf_counter() - t0) / max(reps, 1)

        n_seg = 31          # ~1.75 spikes/step x 18 steps, real sugar regime
        t_repair = t_seg * n_seg
        print(f"active {100*active_frac:5.2f}%  candidates {cand.size:7,} "
              f"({100*cand.size/n:5.2f}%)  1 segment {t_seg*1000:7.2f} ms  "
              f"x{n_seg} segs = {t_repair*1000:8.1f} ms/window "
              f"-> {t_repair*(1000/DELAY):7.2f} s/sim-sec")

    print()
    print("compare: O(N) part measured at 16.33 ms/window = 9.07 s/sim-second")
    print("         grid dt=0.1ms                          = 54.30 s/sim-second")
