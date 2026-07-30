"""Red-team claims 6, 9 (Euler), 7 (method of steps)."""
import numpy as np
from math import gcd

tau_m, tau_s, dt = 20.0, 5.0, 0.1

# ---- Claim 6: Euler effective time constants ----
# Forward Euler decay: g_{n+1} = (1 - dt/tau) g_n. Effective tau_eff: exp(-dt/tau_eff) = 1 - dt/tau
# => tau_eff = -dt / ln(1 - dt/tau)
# Expand: ln(1-e) = -e - e^2/2 - ... => tau_eff = tau / (1 + e/2 + ...) ~ tau(1 - dt/(2 tau)) = tau - dt/2
for tau in (tau_s, tau_m):
    alpha = 1 - dt/tau
    tau_eff = -dt/np.log(alpha)
    shift = tau_eff - tau
    print(f"  tau={tau}: tau_eff={tau_eff:.6f}  shift={shift:+.6f} ms  ({100*shift/tau:+.4f}%)  vs claimed -dt/2={-dt/2}")
print("  claimed: tau_syn 5->4.94983 (-1.003%), tau_mem 20->19.94996 (-0.250%)")
# Note: shift is NOT exactly dt/2; it's tau_eff - tau = -dt/2 - dt^2/(12 tau) - ...
for tau in (tau_s, tau_m):
    e = dt/tau
    tau_eff_series = tau*(1 - e/2 - e**2/12 - e**3/24)
    print(f"  tau={tau}: series tau_eff={tau_eff_series:.6f}")

# P_vg relative error: kernel (x - x^4)/3 exact vs Euler counterpart
# Euler propagator for the 2x2 system (forward Euler on both):
# g_{n+1} = a_s g_n, u_{n+1} = u_n + dt*(g_n - u_n)/tau_m = a_m u_n + (dt/tau_m) g_n
a_m, a_s = 1-dt/tau_m, 1-dt/tau_s
h = 1.8; n = int(round(h/dt))
# exact kernel over h:
x = np.exp(-h/tau_m)
K_exact = (x - x**4)/3
# Euler: g_n = a_s^n g0; u_n = a_m^n u0 + (dt/tau_m) g0 * sum_{k=0}^{n-1} a_m^{n-1-k} a_s^k
geom = (a_m**n - a_s**n)/(a_m - a_s)   # sum a_m^{n-1-k} a_s^k
K_euler = (dt/tau_m)*geom
print(f"\n  over h=1.8ms: K_exact={K_exact:.8f}  K_Euler={K_euler:.8f}  rel err={(K_euler-K_exact)/K_exact:+.6e}")
print("  claimed P_vg relative error +1.257e-02")
# also single-step relative error
x1 = np.exp(-dt/tau_m)
K1_exact = (x1 - x1**4)/3
K1_euler = dt/tau_m
print(f"  single-step: K_exact={K1_exact:.8f} K_Euler={K1_euler:.8f} rel={(K1_euler-K1_exact)/K1_exact:+.6e}")

# ---- Claim 9: PSP under exact vs Euler ----
# delta input g0 at t=0, u0=0. PSP u(t) = (g0/3)(e^{-t/20} - e^{-t/5}) * (with g0=1)
def psp_exact(t, g0=1.0):
    return g0/3*(np.exp(-t/tau_m) - np.exp(-t/tau_s))
def psp_euler(t, g0=1.0):
    n = int(round(t/dt))
    return g0*(dt/tau_m)*(a_m**n - a_s**n)/(a_m - a_s)
tt = np.linspace(dt, 60, 60000)
pe = np.array([psp_exact(t) for t in tt])
pu = np.array([psp_euler(t) for t in tt])
i_e, i_u = np.argmax(pe), np.argmax(pu)
print(f"\nClaim 9: exact peak={pe[i_e]:.6f} at t={tt[i_e]:.4f}; Euler peak={pu[i_u]:.6f} at t={tt[i_u]:.4f}")
print(f"  peak rel err = {(pu[i_u]-pe[i_e])/pe[i_e]*100:+.4f}%  (claimed +0.47%)")
print(f"  peak time shift = {tt[i_u]-tt[i_e]:+.4f} ms (claimed -0.066 ms)")
# analytic peak time exact: t* solves -e^{-t/20}/20 + e^{-t/5}/5 = 0 -> e^{t(1/5-1/20)} = 4 -> t* = ln4/(0.15)=9.24196
print(f"  analytic exact peak t* = {np.log(4)/(1/tau_s-1/tau_m):.5f} ms")

# ---- Claim 7 ----
print("\nClaim 7: refractory 2.2 > delay 1.8 => at most one spike per neuron per 1.8ms window")
print("  min inter-spike interval = 2.2ms > 1.8ms window => at most one spike per window: TRUE by pigeonhole")
print("  gcd(1800us, 2200us) =", gcd(1800, 2200), "us; in units of dt=0.1ms: delay=18 steps, refractory=22 steps, gcd =", gcd(18,22), "steps")
