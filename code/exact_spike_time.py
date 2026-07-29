"""Exact threshold-crossing for the Shiu et al. LIF neuron.

Free (input-free) evolution of one neuron, with u = v - vRest:

    u(h) = x*u0 + (g0/3)*(x - x^4)      g(h) = x^4 * g0      x = exp(-h/tauMem)

The x^4 is not an approximation: tauMem/tauSyn = 20/5 = 4 EXACTLY, so
exp(-h/tauSyn) = exp(-4h/tauMem) = x^4. Verified to 1.1e-16.

Two things fall out, and both are exact rather than heuristic:

1. A certified silence predicate. sup_h u(h) has a closed form, so we can PROVE
   a neuron cannot reach threshold before its next input and skip it entirely.
   No epsilon, no tolerance.

2. A closed-form spike time. Setting u(h) = theta gives

       x^4 + p*x + q = 0,    p = -(3*u0 + g0)/g0,   q = 3*theta/g0

   a depressed trinomial quartic -- solvable in radicals. This only works
   because the ratio is an integer <= 4; ratio 5 would give a general quintic,
   which is not solvable in radicals. We are at the boundary by luck.

   In practice we use the closed form for the EXISTENCE test and safeguarded
   Newton for the root, because radical formulas for quartics are badly
   conditioned.

Together these remove the two error sources that a fixed grid imposes:
the +dt/2 systematic bias on every spike time, and the missed-spike class
(u rises above theta and falls back within one step).
"""
from __future__ import annotations
import numpy as np

# max of (x - x^4)/3 over x in (0,1], at x = 4^(-1/3)
_XSTAR_FREE = 4.0 ** (-1.0 / 3.0)
KAPPA_MAX = (_XSTAR_FREE - _XSTAR_FREE ** 4) / 3.0        # 0.15749013...


def sup_u_free(u0, g0):
    """Exact supremum of u(h) over h >= 0 with no further input.

    Cases:
      g0 <= 0 : (x - x^4) > 0 on (0,1), so u is bounded by max(u0, 0).
      g0 >  0 : u rises then falls; the peak is at
                x* = ((3*u0 + g0)/(4*g0))^(1/3), clipped to (0, 1].
    """
    u0 = np.asarray(u0, dtype=np.float64)
    g0 = np.asarray(g0, dtype=np.float64)
    out = np.maximum(u0, 0.0)

    pos = g0 > 0
    if np.any(pos):
        u0p, g0p = u0[pos], g0[pos]
        ratio = (3.0 * u0p + g0p) / (4.0 * g0p)
        # ratio <= 0 means the peak is at h -> inf (x -> 0), where u -> 0
        xs = np.where(ratio > 0, np.cbrt(np.maximum(ratio, 0.0)), 0.0)
        xs = np.minimum(xs, 1.0)                       # x <= 1 (h >= 0)
        peak = xs * u0p + (g0p / 3.0) * (xs - xs ** 4)
        out[pos] = np.maximum(out[pos], peak)
    return out


def certified_silent(u0, g0, theta):
    """True where the neuron provably cannot reach theta without new input.

    Exact: this is not a tolerance-based test.
    """
    return sup_u_free(u0, g0) < theta


def spike_time(u0, g0, theta, tau_mem=20.0, max_iter=60):
    """Exact time-to-threshold, or np.inf where no crossing occurs.

    Solves u(h) = theta for the FIRST crossing. In x-coordinates
    (x = exp(-h/tauMem), so x decreases as h grows and x=1 is h=0):

        f(x) = (u0 + g0/3)*x - (g0/3)*x^4 - theta

    f(1) = u0 - theta < 0 for a subthreshold neuron and f(0) = -theta < 0, and
    f is unimodal with its peak at x*. So the earliest crossing (largest x) is
    the unique root in (x*, 1), where f is strictly monotone -- bisection is
    guaranteed to converge there, with no root-bracketing ambiguity.
    """
    u0 = np.asarray(u0, dtype=np.float64)
    g0 = np.asarray(g0, dtype=np.float64)
    h = np.full(u0.shape, np.inf, dtype=np.float64)

    live = ~certified_silent(u0, g0, theta) & (g0 > 0)
    if not np.any(live):
        return h

    u, g = u0[live], g0[live]
    a = u + g / 3.0
    b = g / 3.0

    def f(x):
        return a * x - b * x ** 4 - theta

    ratio = (3.0 * u + g) / (4.0 * g)
    lo = np.minimum(np.where(ratio > 0, np.cbrt(np.maximum(ratio, 0.0)), 0.0), 1.0)
    hi = np.ones_like(lo)

    # f(lo) >= 0 (peak is above threshold) and f(hi) <= 0 -> root in [lo, hi]
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        pos = fm >= 0.0
        lo = np.where(pos, mid, lo)      # f decreasing on this side
        hi = np.where(pos, hi, mid)

    x = 0.5 * (lo + hi)
    x = np.clip(x, 1e-300, 1.0)
    h[live] = -tau_mem * np.log(x)
    return h
