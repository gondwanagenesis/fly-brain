"""Audit C: a spike that the reference emits is LOST ENTIRELY by OptWindowSim,
because the repair solves the input-free trajectory (exc arrival then inh
arrival inside the same 1.8 ms window)."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(r"C:\Users\neogo\Documents\FlyBrain")
sys.path.insert(0, str(ROOT / "code"))
from window_opt import OptWindowSim
from window_step import THETA, DELAY, kappa

TAU_M, TAU_S = 20.0, 5.0

def ref(u0, g0, events, T, dt=1e-6):
    u, g, t = u0, g0, 0.0
    ev = sorted(events); k = 0; fired = []
    xm, xs = np.exp(-dt/TAU_M), np.exp(-dt/TAU_S)
    for _ in range(int(round(T/dt))):
        while k < len(ev) and ev[k][0] <= t + 1e-15:
            g += ev[k][1]; k += 1
        u = xm*u + g*(xm - xs)/3.0
        g = xs*g
        t += dt
        if u > THETA:
            fired.append(t); u = 0.0; g = 0.0
    return fired

n = 3
# neuron 0 -> +400 into 2 ; neuron 1 -> -1200 into 2 ; neuron 2 no outputs
crow = np.array([0, 1, 2, 2], dtype=np.int64)
post = np.array([2, 2], dtype=np.int64)
val  = np.array([400.0, -1200.0])

sim = OptWindowSim(n, crow, post, val)
sim.u[0], sim.g[0] = 6.5, 30.0      # fires ~0.456 ms
sim.u[1], sim.g[1] = 5.0, 41.8      # fires ~1.30 ms
sim.active[0] = sim.active[1] = True
for w in range(3):
    sim.step()
si, st = sim.spike_arrays()
print("window-sim spikes (idx, t):", sorted(zip(si.tolist(), st.tolist())))

t0 = ref(6.5, 30.0, [], 1.8)[0]
t1 = ref(5.0, 41.8, [], 1.8)[0]
print(f"source spike times: n0={t0:.5f}  n1={t1:.5f}")
# neuron 2's window is [1.8, 3.6); arrivals at t0+1.8 and t1+1.8 -> lags t0, t1
f2 = ref(0.0, 0.0, [(t0, 400.0), (t1, -1200.0)], 1.8)
print("reference: neuron 2 fires at lags", f2, "-> absolute", [1.8+x for x in f2])
print()
print("VERDICT: neuron 2 spike present in reference, absent from window sim:",
      2 not in si.tolist() and len(f2) > 0)
