"""Red-team claims 3, 4."""
import numpy as np

tau_m, tau_s = 20.0, 5.0
dt = 0.1
x = np.exp(-dt/tau_m)
print("x = exp(-0.1/20) =", x)

# ---- Claim 3: contraction ||P||_1 = x ? ----
P = np.array([[x, (x - x**4)/3],[0, x**4]])
# induced 1-norm = max column sum of absolute values
n1 = np.max(np.sum(np.abs(P), axis=0))
ninf = np.max(np.sum(np.abs(P), axis=1))
n2 = np.linalg.norm(P, 2)
print(f"  ||P||_1 (induced, max col sum) = {n1:.10f}")
print(f"  ||P||_inf (max row sum) = {ninf:.10f}")
print(f"  ||P||_2 (spectral) = {n2:.10f}")
print(f"  x = {x:.10f}")
print(f"  spectral radius = {max(abs(np.linalg.eigvals(P))):.10f}  (should equal x)")

# Round-off floor: repeated contraction u_{n+1} = fl(P u_n)
# standard model: fl(Pu) = Pu + e, |e| <= eps * |P u| (+ eps|u| style). Stationary error
# bound: e_inf <= eps * max||Pu|| / (1 - x)? Check claimed floor eps/(1-x) with eps fp32.
eps32 = np.float32(2.0**-24)
print("\n  eps(fp32, 2^-24) =", float(eps32), " eps/(1-x) =", float(eps32)/(1-x))
# empirical: iterate scalar decay u_{n+1}=x*u_n from 1.0 in fp32, measure fixed-point residue
u = np.float32(1.0)
x32 = np.float32(x)
hist = []
for n in range(200000):
    u = np.float32(x32 * u)
    hist.append(float(u))
hist = np.array(hist)
# where does it stall? geometric until rounding dominates
print("  scalar fp32 decay from 1.0: value after 200k iters:", hist[-1])
print("  true x^200000:", float(x)**200000)
# find n where relative error first exceeds 10%
truev = float(x)**np.arange(1, 200001)
rel = np.abs(hist - truev)/np.maximum(truev, 1e-45)
idx = np.argmax(rel > 0.5)
print("  first iter with >50% rel error:", idx, " value there:", hist[idx])
# stall point: u such that x*u rounds to u? never; but absolute error accumulates:
# e_n satisfies e_{n+1} = x e_n + delta_n, |delta|<= eps*|u_n|. steady floor ~ eps*|u|/(1-x) relative.
# So floor on ABSOLUTE error for u~O(1): eps/(1-x) ~ 1.4e-6, not 2.4e-5.
# But claim says floor is eps/(1-x) mV = 2.4e-5 mV -> that means eps ~ 1.2e-7*(0.086)...
# 1-x = 0.086069; eps/(1-x) with eps=2^-24 gives 6.9e-7. Claim 2.4e-5 implies eps used = 2.4e-5*0.086=2.06e-6?
print("  claim's 2.4e-5 mV implies eps used =", 2.4e-5*(1-x))
# maybe eps = 2^-23 (fp32 machine epsilon in the '1.0+eps' convention):
print("  2^-23/(1-x) =", 2.0**-23/(1-x))
# maybe theta-scale: eps*theta/(1-x) = 1.19e-7*7/0.086 = 9.7e-6. eps*g_scale?

# ---- Claim 4: quartic root analysis ----
print("\nClaim 4: u(h)=theta with x=exp(-h/20):")
print("  x*u0 + (g0/3)(x - x^4) = theta")
print("  => x^4 - (1 + 3u0/g0) x + 3 theta/g0 = 0   [multiply by -3/g0]")
print("  claim: p = -(3u0+g0)/g0 = -(1 + 3u0/g0)  -> matches; q = 3 theta/g0 -> matches")

def f(xv, u0, g0, theta=7.0):
    return xv**4 + (-(3*u0+g0)/g0)*xv + 3*theta/g0

def roots_info(u0, g0, theta=7.0):
    p = -(3*u0+g0)/g0; q = 3*theta/g0
    r = np.roots([1, 0, 0, p, q])
    real01 = sorted(r.real[np.abs(r.imag) < 1e-9])
    real01 = [rv for rv in real01 if -1e-12 <= rv <= 1+1e-12]
    f0 = q  # f(0)
    f1 = 1 + p + q
    xstar = ((3*u0+g0)/(4*g0))**(1/3) if (3*u0+g0)/g0 > 0 else None
    return real01, f0, f1, xstar

# claim: f unimodal, f(0)<0, f(1)<0 => unique root in (x*,1).
# f(0) = q = 3 theta/g0. For g0>0, theta>0: f(0) = +3theta/g0 > 0 !!! Check:
print("\n  f(0) = q = 3*theta/g0. For g0>0 this is POSITIVE, contradicting 'f(0)<0'.")
for (u0,g0) in [(0.0, 3.0), (2.0, 5.0), (-3.0, 10.0)]:
    real01, f0, f1, xstar = roots_info(u0, g0)
    print(f"  u0={u0}, g0={g0}: f(0)={f0:+.4f}, f(1)={f1:+.4f}, x*={xstar}, roots in [0,1]={np.round(real01,6)}")

# Correct reading: threshold crossing needs u(h) to REACH theta. f(x)=0 crossing as x decreases from 1 (h=0) toward 0 (h=inf).
# u(0)=u0. u(inf)=0. So a root in (0,1) exists iff max_h u(h) > theta (and u0 != theta...).
# Check number of roots in (0,1) for excitatory g0: quartic f' = 4x^3 + p, single real critical point x* = (-p/4)^(1/3)
# = ((3u0+g0)/(4g0))^(1/3) when 3u0+g0>0. f decreasing then increasing? f'' = 12x^2 >= 0: f is CONVEX.
# Convex => at most 2 real roots. f(0)=q>0 (g0>0). If f dips below zero, TWO roots in (0,1), not one!
print("\n  f'' = 12x^2 >= 0: f is convex, so generically 0 or 2 roots in (0,1), NOT unique.")
for (u0,g0) in [(0.0, 50.0), (0.0, 100.0), (-5.0, 100.0)]:
    real01, f0, f1, xstar = roots_info(u0, g0)
    print(f"  u0={u0}, g0={g0}: f(0)={f0:+.4f}, f(1)={f1:+.4f}, x*={xstar:.4f}, roots in [0,1]={np.round(real01,6)}")
# interpretation: u rises, crosses theta at large x (early h), peaks, decays, crosses again at small x.
# The FIRST spike time corresponds to the LARGER root (smaller h). So 'unique root in (x*,1)' — check:
# x* is the minimum of f. If f(x*)<0<f(1)? f(1) = 1 + p + q = (g0 - 3u0 - g0 + 3theta)/g0 = 3(theta-u0)/g0
print("\n  f(1) = 3(theta-u0)/g0  -> sign depends on u0 vs theta. For subthreshold u0<theta, g0>0: f(1)>0.")
print("  So with f(0)>0, f(1)>0, convex: either 0 or 2 roots in (0,1); the larger is in (x*,1) and IS unique there.")
# verify uniqueness within (x*, 1): f increasing on (x*, inf) since f' >0 for x>x*. Yes, monotonic => unique.
# Counterexamples: g0<0 (inhibitory): q<0 -> f(0)<0, f(1)=3(theta-u0)/g0 <0 for u0<theta -> could have root:
for (u0,g0) in [(0.0, -30.0), (8.0, -10.0), (-10.0, -50.0)]:
    real01, f0, f1, xstar = roots_info(u0, g0)
    print(f"  INH u0={u0}, g0={g0}: f(0)={f0:+.4f}, f(1)={f1:+.4f}, x*={xstar}, roots in [0,1]={np.round(real01,6)}")
# 3u0+g0 < 0 with g0>0: x* complex (negative radicand). f' = 4x^3+p with p>0 -> f' >0 on (0,1): monotone increasing.
u0, g0 = -10.0, 20.0
real01, f0, f1, xstar = roots_info(u0, g0)
print(f"  3u0+g0<0: u0={u0},g0={g0}: p positive, f strictly increasing on x>0, f(0)={f0:.3f}>0, f(1)={f1:.3f}; roots in [0,1]={real01}")
print("    -> no root: u starts negative, g excitatory but 3u0+g0<0 means du/dh at 0 = (g0-u0)/20? check dynamics")
