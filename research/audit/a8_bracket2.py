"""Audit E2: does the claimed bracket  f(lo) >= 0 >= f(hi)  actually hold in fp
for every state that reaches the bisection loop?"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(r"C:\Users\neogo\Documents\FlyBrain")
sys.path.insert(0, str(ROOT / "code"))
from exact_spike_time import sup_u_free, certified_silent, spike_time
THETA, TAU_M = 7.0, 20.0

def bracket_check(u0, g0):
    u0 = np.asarray(u0, float); g0 = np.asarray(g0, float)
    now  = u0 >= THETA
    live = ~certified_silent(u0, g0, THETA) & (g0 > 0) & ~now
    u, g = u0[live], g0[live]
    a, b = u + g/3.0, g/3.0
    ratio = (3.0*u + g)/(4.0*g)
    lo = np.minimum(np.where(ratio > 0, np.cbrt(np.maximum(ratio, 0.0)), 0.0), 1.0)
    f  = lambda x: a*x - b*x**4 - THETA
    return live, f(lo), f(np.ones_like(lo))

# 1) random broad scan
rng = np.random.default_rng(1)
u0 = rng.uniform(-5, 7.0, 2_000_000); g0 = rng.uniform(0.0, 400.0, 2_000_000)
live, flo, fhi = bracket_check(u0, g0)
print(f"broad scan: live={live.sum()}  min f(lo)={flo.min():.3e}  "
      f"#f(lo)<0 = {(flo<0).sum()}   max f(1)={fhi.max():.3e}  #f(1)>0 = {(fhi>0).sum()}")

# 2) targeted tangency scan: drive peak to theta for many g0
bad = 0; tot = 0
for g0v in np.geomspace(0.01, 5000.0, 400):
    lo_, hi_ = -50.0, 7.0
    for _ in range(200):
        mid = 0.5*(lo_+hi_)
        if float(sup_u_free(np.array([mid]), np.array([g0v]))[0]) < THETA: lo_ = mid
        else: hi_ = mid
    for u0v in (hi_, np.nextafter(hi_, np.inf), np.nextafter(hi_, -np.inf)):
        live, flo, fhi = bracket_check(np.array([u0v]), np.array([g0v]))
        if live.any():
            tot += 1
            if flo[0] < 0:
                bad += 1
                if bad <= 5:
                    h = spike_time(np.array([u0v]), np.array([g0v]), THETA, TAU_M)[0]
                    print(f"  BRACKET FAILURE g0={g0v:.6g} u0={u0v!r}  f(lo)={flo[0]:.3e} "
                          f"f(1)={fhi[0]:.3e}  spike_time={h!r}")
print(f"tangency scan: {bad} bracket failures out of {tot} live cases")
