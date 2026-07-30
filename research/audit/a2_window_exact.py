"""Audit B: OptWindowSim's repair solves the INPUT-FREE trajectory, so
mid-window arrivals are invisible to the spike-time solver."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(r"C:\Users\neogo\Documents\FlyBrain")
sys.path.insert(0, str(ROOT / "code"))
from window_opt import OptWindowSim
from window_step import THETA, DELAY, kappa, KAPPA_WINDOW_MAX
from exact_spike_time import spike_time, sup_u_free

TAU_M, TAU_S = 20.0, 5.0

def ref_traj(u0, g0, events, T, dt=1e-5):
    """Fine-grid reference: events = list of (time, weight) impulses into g."""
    u, g, t = u0, g0, 0.0
    ev = sorted(events)
    k = 0
    fired = []
    n = int(round(T/dt))
    for i in range(n):
        while k < len(ev) and ev[k][0] <= t + 1e-12:
            g += ev[k][1]; k += 1
        xm, xs = np.exp(-dt/TAU_M), np.exp(-dt/TAU_S)
        u = xm*u + g*(xs and 1.0)*(xm - xs)/3.0
        g = xs*g
        t += dt
        if u > THETA:
            fired.append(t); u = 0.0; g = 0.0
            break
    return fired, u, g

# --- 2-neuron net: 0 -> 1 with a big weight ---
n = 2
crow = np.array([0, 1, 1], dtype=np.int64)   # neuron 0 has 1 synapse, neuron 1 none
post = np.array([1], dtype=np.int64)
val  = np.array([200.0])

sim = OptWindowSim(n, crow, post, val)
sim.u[0] = 6.5; sim.g[0] = 30.0    # neuron 0 crosses ~0.5 ms into window 0
sim.active[0] = True

for w in range(4):
    sim.step()
    print(f"window {w}: t={sim.t:.2f}  u={sim.u}  g={sim.g}")
si, st = sim.spike_arrays()
print("window-sim spikes (idx, t):", list(zip(si.tolist(), st.tolist())))

# ground truth for neuron 0
f0, _, _ = ref_traj(6.5, 30.0, [], 1.8)
print("\nreference: neuron 0 crosses at t =", f0)
t_sp0 = f0[0]
# neuron 1 receives w=200 at t_sp0 + 1.8
f1, u1, g1 = ref_traj(0.0, 0.0, [(t_sp0 + 1.8 - 1.8, 200.0)], 1.8)   # measured from window start of arrival window
print("reference: neuron 1 (arrival at lag %.4f ms into its window) crosses at t =" % (t_sp0,), f1)
print("           -> absolute time =", 1.8 + (f1[0] if f1 else float('nan')))
