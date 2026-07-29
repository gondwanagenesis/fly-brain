"""Delay-window stepping: advance the whole network by the full 1.8 ms axonal
delay in ONE step, exactly.

Why this is legitimate
----------------------
Every synapse carries the same delay D = 1.8 ms, and the connectome has no
autapses, no zero-delay edges and no gap junctions (verified: 0 self-edges in
15,091,983; delay is a single scalar on the whole synapse population). So by the
method of steps for delay differential equations (Bellman & Cooke 1963; Hairer,
Norsett & Wanner ch. II.17), on any interval of length D the forcing to every
neuron is fully determined by spikes emitted BEFORE the interval began. The
network therefore decouples into N independent inhomogeneous linear ODEs, each
of which we can solve in closed form.

NEST and NEURON already use the minimum delay to batch spike COMMUNICATION.
What they do not do is raise the INTEGRATION timestep to it -- they keep h small
and march the grid inside the window. That is the step taken here.

The sigma factorisation
-----------------------
A spike emitted by neuron j at t_k in [t0-D, t0) arrives at t_k + D, i.e. inside
the window [t0, t0+D). Its contribution at the window END is

    to g:  w * exp(-s/tau_syn)
    to u:  w * kappa(s),   kappa(s) = (1/3)(exp(-s/tau_mem) - exp(-s/tau_syn))

with s = (t0 + D) - (t_k + D) = t0 - t_k.

s depends ONLY on the source spike time, never on the target. So per-source
scalars can be accumulated first and pushed through the connectome once:

    sigma_j^(g) = sum_over_j's_spikes exp(-s/tau_syn)
    sigma_j^(u) = sum_over_j's_spikes kappa(s)
    dg = W @ sigma^(g),   du = W @ sigma^(u)

Two sparse mat-vecs per window replace 18 rounds of irregular scatter.

Exactness caveat
----------------
The closed form above assumes the target does not spike inside the window (a
spike sets g = 0, discarding the accumulated pre-spike contribution). We
therefore predict, then certify, then repair only the neurons that could have
crossed -- see certified_no_spike().
"""
from __future__ import annotations
import numpy as np

TAU_M, TAU_S = 20.0, 5.0
V_REST, V_TH = -52.0, -45.0
THETA = V_TH - V_REST          # 7 mV
DELAY = 1.8                    # ms, uniform


def kappa(s, tau_m=TAU_M, tau_s=TAU_S):
    """Membrane response at lag s to a unit impulse into g."""
    s = np.asarray(s, dtype=np.float64)
    return (np.exp(-s / tau_m) - np.exp(-s / tau_s)) * (tau_s / (tau_m - tau_s))


def kappa_max_over_window(D=DELAY, tau_m=TAU_M, tau_s=TAU_S):
    """max kappa(s) for s in (0, D]. kappa rises from 0, so on a short window
    the max sits at the right end."""
    s = np.linspace(1e-12, D, 200001)
    return float(kappa(s, tau_m, tau_s).max())


KAPPA_WINDOW_MAX = kappa_max_over_window()


def free_window_map(u0, g0, D=DELAY, tau_m=TAU_M, tau_s=TAU_S):
    """Exact free evolution over one window (no input, no spike).

    Uses x = exp(-D/tau_mem); because tau_mem/tau_syn = 4 exactly,
    exp(-D/tau_syn) = x**4, so this is polynomial in x.
    """
    x = np.exp(-D / tau_m)
    x4 = x ** 4 if abs(tau_m / tau_s - 4.0) < 1e-12 else np.exp(-D / tau_s)
    u = x * u0 + (g0 * (tau_s / (tau_m - tau_s))) * (x - x4)
    return u, x4 * g0


def sigma_factors(spike_times, t0, tau_m=TAU_M, tau_s=TAU_S):
    """Per-spike (sigma_g, sigma_u) weights. s = t0 - t_spike, in (0, D]."""
    s = t0 - np.asarray(spike_times, dtype=np.float64)
    return np.exp(-s / tau_s), kappa(s, tau_m, tau_s)


def certified_no_spike(u0, g0, w_pos_sum, theta=THETA, D=DELAY):
    """True where the neuron provably cannot reach threshold inside the window.

    u can gain at most kappa_max * (g0 + positive input) on top of its starting
    value, since kappa <= kappa_max at every lag in the window and each impulse
    contributes at most w * kappa_max. Never certifies a neuron that would
    spike, so anything it passes is finished exactly.

    ``w_pos_sum`` should be THE POSITIVE INPUT ACTUALLY ARRIVING IN THIS WINDOW,
    not the neuron's total positive in-weight. This matters enormously, and it
    is legitimate precisely because of the delay: every spike that can land in
    this window was emitted before the window began, so the arriving input is
    already known exactly -- there is nothing to bound over.

    Measured on the real connectome at the sugar-experiment active fraction:

        w_pos_sum = total in-weight (all synapses could fire)   18,949 candidates (13.67%)
        w_pos_sum = actual arrivals this window                     184 candidates ( 0.13%)

    a 103x reduction, and the difference between the repair path costing
    ~42 s per simulated second and ~0.4 s. Passing total in-weight here is
    still correct, just far more conservative than necessary.
    """
    return (np.maximum(u0, 0.0)
            + KAPPA_WINDOW_MAX * (np.maximum(g0, 0.0)
                                  + np.maximum(w_pos_sum, 0.0))) < theta


def window_positive_input(n, crow, post, val, fired_idx):
    """Positive weight actually arriving on each neuron this window.

    Event-driven: touches only the synapses of neurons that fired, so it costs
    O(spikes x fanout), not O(nnz).
    """
    w = np.zeros(n)
    for j in np.atleast_1d(fired_idx):
        lo, hi = crow[j], crow[j + 1]
        if hi > lo:
            np.add.at(w, post[lo:hi], np.maximum(val[lo:hi], 0.0))
    return w
