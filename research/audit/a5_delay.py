"""Audit D: how many timesteps of axonal delay does the reference actually
implement?  Model spec says 1.8 ms = 18 steps at dt=0.1 ms."""
import numpy as np, torch
print("int(1.8/0.1)      =", int(1.8/0.1), "   (1.8/0.1 =", repr(1.8/0.1), ")")
print("brain_engine L    =", int(1.8/0.1)+1)
print("run_pytorch buflen=", int(1.8/0.1)+1)

# minimal reimplementation of the reference ordering
L = int(1.8/0.1)+1
N = 2
g = np.zeros(N); buf = np.zeros((L, N)); head = 0
spikes = np.zeros(N)
lag = None
for k in range(40):
    rec = np.zeros(N)
    if spikes[0] > 0:
        rec[1] += 1.0                       # neuron 0 -> neuron 1, weight 1
    delayed = buf[head].copy()
    g_new = g*(1-0.1/5.0) + delayed
    buf[head] = rec
    head = (head+1) % L
    g = g_new
    # neuron 0 fires exactly once, at the end of step 0
    spikes = np.zeros(N)
    if k == 0:
        spikes[0] = 1.0
    if g[1] != 0 and lag is None:
        lag = k
        print(f"spike emitted at end of step 0; g[target] first non-zero at step {k}"
              f"  -> {k} steps = {k*0.1:.1f} ms")
        break
