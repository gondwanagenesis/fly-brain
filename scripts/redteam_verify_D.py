"""Red-team claim 8 + refinements + missed opportunities."""
import numpy as np

tau_m, tau_s, dt, D = 20.0, 5.0, 0.1, 1.8

# ---- Claim 9 refinement: discrete Euler peak time ----
a_m, a_s = 1-dt/tau_m, 1-dt/tau_s
n = np.arange(0, 600)
u_e = (dt/tau_m)*(a_m**n - a_s**n)/(a_m - a_s)
i = np.argmax(u_e)
# parabolic interpolation on the discrete sequence
y0,y1,y2 = u_e[i-1],u_e[i],u_e[i+1]
shift = 0.5*(y0-y2)/(y0-2*y1+y2)
t_peak_euler = (i+shift)*dt
t_peak_exact = np.log(4)/(1/tau_s-1/tau_m)
print(f"Claim 9 refined: discrete Euler peak idx={i} (t={i*dt}), parabolic t={t_peak_euler:.4f}")
print(f"  exact peak t={t_peak_exact:.5f}; shift={t_peak_euler-t_peak_exact:+.4f} ms (claimed -0.066)")
# continuous Euler-equivalent (tau_eff) peak time:
te_m = -dt/np.log(a_m); te_s = -dt/np.log(a_s)
t_peak_eff = np.log(te_m/te_s)/(1/te_s-1/te_m)
print(f"  tau_eff-continuous peak t={t_peak_eff:.5f}, shift={t_peak_eff-t_peak_exact:+.4f} ms")
peak_exact = (np.exp(-t_peak_exact/tau_m)-np.exp(-t_peak_exact/tau_s))/3
print(f"  peak height: exact={peak_exact:.6f}, euler-disc={u_e[i]:.6f}, rel={(u_e[i]-peak_exact)/peak_exact*100:+.4f}%")

# ---- Claim 8: sigma factorization ----
rng = np.random.default_rng(1)
N_src, N_tgt = 50, 30
W = (rng.random((N_tgt, N_src)) < 0.3) * rng.integers(1, 5, (N_tgt, N_src)) * 0.275
s = rng.uniform(0, D, N_src)          # arrival lag within window, target-independent
fired = rng.random(N_src) < 0.5
sigma_g = np.where(fired, np.exp(-s/tau_s), 0.0)
sigma_u = np.where(fired, (np.exp(-s/tau_m)-np.exp(-s/tau_s))/3, 0.0)
dG = W @ sigma_g; dU = W @ sigma_u
# brute force per-target scatter
dG2 = np.zeros(N_tgt); dU2 = np.zeros(N_tgt)
for j in range(N_src):
    if fired[j]:
        dG2 += W[:,j]*np.exp(-s[j]/tau_s)
        dU2 += W[:,j]*(np.exp(-s[j]/tau_m)-np.exp(-s[j]/tau_s))/3
print(f"\nClaim 8: max|dG diff|={np.max(np.abs(dG-dG2)):.2e}, max|dU diff|={np.max(np.abs(dU-dU2)):.2e}")
print("  factorization exact (linearity); caveat re target reset is model-semantics, not algebra")

# ---- Missed opportunities ----
# 1) cancellation-free kernel eval: (x - x^4)/3 = x*(1-x)*(1+x+x^2)/3
print("\nExtras: fp32 kernel evaluation cancellation test (small h)")
for h in (0.1, 0.01, 0.001):
    x32 = np.float32(np.exp(-h/tau_m))
    naive = (x32 - np.float32(x32**4))/np.float32(3)
    fact = x32*np.float32(1-x32)*np.float32(1+x32+x32*x32)/np.float32(3)
    exact = (np.exp(-h/tau_m)-np.exp(-4*h/tau_m))/3
    print(f"  h={h}: exact={exact:.10e} naive_rel={(float(naive)-exact)/exact:+.2e} factored_rel={(float(fact)-exact)/exact:+.2e}")

# 2) Newton on the quartic from guaranteed bracket [x*, 1]
def spike_x(u0, g0, theta=7.0):
    p = -(3*u0+g0)/g0; q = 3*theta/g0
    xs = ((3*u0+g0)/(4*g0))**(1/3)
    f = lambda x: x**4 + p*x + q
    fp = lambda x: 4*x**3 + p
    if f(xs) > 0 or f(1.0) < 0: return None, 0
    x = 1.0  # f convex on [x*,1], increasing: Newton from right endpoint monotone converges
    for it in range(50):
        xn = x - f(x)/fp(x)
        if abs(xn-x) < 1e-15: x = xn; break
        x = xn
    return x, it+1
for (u0,g0) in [(0.0,50.0),(0.0,100.0),(-5.0,100.0),(3.0,20.0)]:
    xr, it = spike_x(u0,g0)
    if xr:
        print(f"  Newton u0={u0},g0={g0}: x={xr:.12f} -> h={-tau_m*np.log(xr):.6f} ms, {it} iters")
# verify against direct dynamics
u0,g0 = 0.0, 100.0
xr,_ = spike_x(u0,g0)
h = -tau_m*np.log(xr)
chk = np.exp(-h/tau_m)*u0 + g0/3*(np.exp(-h/tau_m)-np.exp(-h/tau_s))
print(f"  residual u(h)-theta = {chk-7.0:.2e}")

# 3) closed-form: depressed quartic x^4+px+q=0 (b=0,c=0). Ferrari reduces to cubic resolvent.
# Check discriminant sign patterns to confirm radicals viable but ill-conditioned vs Newton.
import numpy.polynomial.polynomial as P_
print("\n  numpy roots vs Newton agreement:")
r = np.roots([1,0,0,-(3*u0+g0)/g0, 21.0/g0])
print("  np.roots:", np.round(r,10), " Newton:", xr)
