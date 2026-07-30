"""Red-team claims 1, 2, 5, 10."""
import numpy as np
from scipy.linalg import expm

tau_m, tau_s, dt, theta = 20.0, 5.0, 0.1, 7.0

# ---- Claim 1: matrix exponential ----
# State y = (u, g). du/dt = (g - u)/tau_m ; dg/dt = -g/tau_s
A = np.array([[-1/tau_m, 1/tau_m], [0.0, -1/tau_s]])
h = 1.8
M = expm(A*h)
x = np.exp(-h/tau_m)
print("Claim 1: ratio tau_m/tau_s =", tau_m/tau_s)
print("  x =", x, " x^4 =", x**4, " exp(-h/tau_s) =", np.exp(-h/tau_s))
print("  expm(A h) =\n", M)
print("  predicted [[x, (x - x^4)/3],[0, x^4]] =\n",
      np.array([[x, (x - x**4)/3], [0, x**4]]))
print("  max abs diff:", np.max(np.abs(M - np.array([[x, (x-x**4)/3],[0, x**4]]))))

# General formula: coupling coefficient = tau_s/(tau_m - tau_s) * (x - x^{tau_m/tau_s})?
# Derive: u(t) = x^t-term... solve exactly via sympy-free numeric scan over random ratios
rng = np.random.default_rng(0)
print("\n  General-coefficient test (random tau_m, tau_s, h):")
for _ in range(5):
    tm = rng.uniform(2, 50); ts = rng.uniform(0.5, tm*0.9); hh = rng.uniform(0.1, 5)
    Aa = np.array([[-1/tm, 1/tm],[0,-1/ts]])
    Mm = expm(Aa*hh)
    r = tm/ts
    xm = np.exp(-hh/tm)
    # candidate formulas
    c1 = (xm - np.exp(-hh/ts)) / (r - 1)          # (x - x^r)/(r-1)
    c2 = (xm - np.exp(-hh/ts)) / (1 - ts/tm)      # claim's (x - x^4)/(1 - ts/tm)
    print(f"    tm={tm:.3f} ts={ts:.3f} expm01={Mm[0,1]:.8f} "
          f"c1=(x-x^r)/(r-1)={c1:.8f} c2=(x-x^r)/(1-ts/tm)={c2:.8f}")

# ---- Claim 2: semigroup ----
print("\nClaim 2: semigroup P(h1)P(h2) = P(h1+h2), and in x-coords P(x1)P(x2)=P(x1*x2)")
def P(xv):
    return np.array([[xv, (xv - xv**4)/3],[0, xv**4]])
errs = []
for _ in range(1000):
    h1, h2 = rng.uniform(0, 10, 2)
    e = np.max(np.abs(P(np.exp(-h1/tau_m)) @ P(np.exp(-h2/tau_m)) - P(np.exp(-(h1+h2)/tau_m))))
    errs.append(e)
print("  max |P(x1)P(x2)-P(x1x2)| over 1000 random pairs:", max(errs))

# ---- Claim 5: silence kernel constant ----
print("\nClaim 5: max over h>=0 of (x - x^4)/3, x=exp(-h/20)")
hs = np.linspace(0, 100, 2_000_001)
xs = np.exp(-hs/tau_m)
K = (xs - xs**4)/3
i = np.argmax(K)
print(f"  max K = {K[i]:.10f} at h = {hs[i]:.6f} ms (x = {xs[i]:.6f})")
# analytic: d/dx (x - x^4) = 1 - 4x^3 = 0 -> x* = 4^(-1/3)
xa = 4**(-1/3)
print(f"  analytic: x* = 4^(-1/3) = {xa:.8f}, K* = {(xa - xa**4)/3:.10f}, h* = {-tau_m*np.log(xa):.6f}")
# check bound claim: sup_h u(h) <= max(u0,0) + 0.0720850*(g0 + sum w+)
# test with random u0<=0, g0>=0
worst = 0
for _ in range(200000):
    u0 = -rng.uniform(0, 20); g0 = rng.uniform(0, 500)
    hh = rng.uniform(0, 100)
    xv = np.exp(-hh/tau_m)
    u = xv*u0 + g0*(xv - xv**4)/3
    bound = max(u0,0) + 0.0720850*g0
    worst = max(worst, u - bound)
print("  worst (u - bound) over 200k random (u0<0,g0>0):", worst, "(should be ~0, marginally <=0 if exact)")
print("  exact constant vs 0.0720850: diff =", (xa - xa**4)/3 - 0.0720850)

# inhibitory: does bound logic break? bound says u <= max(u0,0) + K*(g0 + sum w+).
# With inhibition, g can go negative -> u dips; upper bound still holds since negative g only lowers u.
# But what breaks: claim 'resting neuron needs sum w+ > 97' assumes all excitatory.

# ---- Claim 10 ----
print("\nClaim 10: 7 / (0.0720850 * 0.275) =", 7/(0.0720850*0.275))
print("  with exact constant:", 7/(((xa-xa**4)/3)*0.275))
print("  ceil:", np.ceil(7/(((xa-xa**4)/3)*0.275)))
