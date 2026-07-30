"""Audit E: floating-point behaviour of spike_time near the tangency
(peak == theta) and the u0 == theta boundary."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(r"C:\Users\neogo\Documents\FlyBrain")
sys.path.insert(0, str(ROOT / "code"))
from exact_spike_time import sup_u_free, certified_silent, spike_time
THETA = 7.0
TAU_M = 20.0

def u_of_h(u0, g0, h):
    x = np.exp(-h/TAU_M)
    return x*u0 + (g0/3.0)*(x - x**4)

# --- 1. tangency: find u0 where the peak == theta exactly, for a fixed g0 ---
print("=== tangency scan: peak(u0,g0) ~ theta ===")
for g0 in (1.0, 10.0, 100.0, 1000.0):
    lo, hi = 0.0, THETA
    for _ in range(200):
        mid = 0.5*(lo+hi)
        if float(sup_u_free(np.array([mid]), np.array([g0]))[0]) < THETA: lo = mid
        else: hi = mid
    for u0, tag in ((lo,"below"), (hi,"above")):
        h = float(spike_time(np.array([u0]), np.array([g0]), THETA, TAU_M)[0])
        sup = float(sup_u_free(np.array([u0]), np.array([g0]))[0])
        umax = max(u_of_h(u0, g0, hh) for hh in np.linspace(0, 60, 600001))
        print(f" g0={g0:7.1f} u0={u0!r:22} {tag:5} sup={sup:.17f} "
              f"true_max={umax:.17f} spike_time={h!r} "
              f"{'SPIKE' if np.isfinite(h) else 'silent'} "
              f"{'  <-- FALSE POSITIVE' if np.isfinite(h) and umax < THETA else ''}"
              f"{'  <-- FALSE SILENCE' if (not np.isfinite(h)) and umax > THETA else ''}")

# --- 2. u0 exactly == theta ---
print("\n=== u0 == theta exactly (reference test is  v > vth , STRICT) ===")
h = spike_time(np.array([THETA]), np.array([0.0]), THETA, TAU_M)
print(" spike_time(u0=theta, g0=0) =", h, "  -> fires at h=0")
print(" certified_silent(u0=theta, g0=0) =", certified_silent(np.array([THETA]), np.array([0.0]), THETA))
print(" reference dense kernel uses (v > vThreshold): u0==theta does NOT fire.")

# --- 3. large-scale randomized cross-check against a fine grid ---
print("\n=== randomized cross-check (50k samples) vs dt=1e-4 ms brute force ===")
rng = np.random.default_rng(0)
M = 50000
u0 = rng.uniform(-3, 7.5, M)
g0 = rng.uniform(-20, 200, M)
h = spike_time(u0, g0, THETA, TAU_M)
hh = np.arange(0, 60, 1e-4)
false_sil = false_pos = 0
maxdt = 0.0
for i in range(M):
    if i % 5000 == 0: pass
    hs = h[i]
    # analytic sup, high resolution
    if np.isfinite(hs):
        # check u actually reaches theta
        um = u_of_h(u0[i], g0[i], np.linspace(0, 60, 20001)).max()
        if um < THETA - 1e-12: false_pos += 1
    else:
        um = u_of_h(u0[i], g0[i], np.linspace(0, 60, 20001)).max()
        if um > THETA + 1e-12: false_sil += 1
print(f" false silences: {false_sil}   false positives: {false_pos}   (of {M})")
