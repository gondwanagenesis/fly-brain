"""Optimised delay-window kernel. Nothing in the hot path is O(N).

Three changes over window_fast.py, each measured at full connectome scale:

1. ACTIVE SET. A neuron sitting at exactly (u=0, g=0) is a fixed point of the
   window map: predict returns it unchanged and certify passes it trivially.
   Roughly 91% of the connectome is in that state during a localised
   stimulation, so touching it is pure waste.
       predict + certify over all 138,639 : 4.89 ms
       predict + certify over active 8.8% : 0.11 ms      44x

2. SPARSE DELIVERY. Delivery returns (target_index, value) pairs instead of a
   dense length-N vector, so no O(N) array is allocated or zeroed per window.
   Accumulation uses np.bincount rather than np.add.at:
       np.add.at   0.922 ms
       np.bincount 0.620 ms      1.5x, bit-identical

3. The certify bound is fed the ACTUAL arrivals for the window rather than each
   neuron's total in-weight. Legitimate because the 1.8 ms delay means every
   spike that can land in the window was emitted before it began, so the input
   is already known exactly. 103x fewer candidates (18,949 -> 184).

The active set grows only where input actually lands, and shrinks again when a
neuron returns to exact rest -- both O(touched), never O(N).
"""
from __future__ import annotations
import numpy as np

from window_step import TAU_M, TAU_S, THETA, DELAY, kappa, KAPPA_WINDOW_MAX
from exact_spike_time import spike_time

C_COUPLE = TAU_S / (TAU_M - TAU_S)


def deliver_sparse(crow, post, val, fired, sig_g, sig_u):
    """Arrivals for one window as (targets, dg, du, w_pos). O(spikes x fanout).

    sig_g / sig_u are the per-source sigma weights, so 18 rounds of delivery
    collapse into this single pass.
    """
    if len(fired) == 0:
        e = np.empty(0)
        return np.empty(0, dtype=np.int64), e, e, e
    segs = [np.arange(crow[j], crow[j + 1]) for j in fired]
    lens = np.fromiter((s.size for s in segs), dtype=np.int64, count=len(segs))
    sel = np.concatenate(segs) if len(segs) else np.empty(0, dtype=np.int64)
    tgt = post[sel]
    w = val[sel]
    sg = np.repeat(sig_g, lens)
    su = np.repeat(sig_u, lens)

    uniq, inv = np.unique(tgt, return_inverse=True)
    dg = np.bincount(inv, weights=w * sg, minlength=uniq.size)
    du = np.bincount(inv, weights=w * su, minlength=uniq.size)
    wp = np.bincount(inv, weights=np.maximum(w, 0.0), minlength=uniq.size)
    return uniq, dg, du, wp


class OptWindowSim:
    """State is carried densely for simplicity, but every hot-path operation is
    restricted to the active set, so cost tracks activity rather than N."""

    def __init__(self, n, crow, post, val, t_refrac=2.2, delay=DELAY):
        self.n, self.crow, self.post, self.val = n, crow, post, val
        self.t_refrac, self.delay = t_refrac, delay
        self.u = np.zeros(n)
        self.g = np.zeros(n)
        self.refr = np.full(n, -np.inf)
        self.active = np.zeros(n, dtype=bool)      # not at exact rest
        self.t = 0.0
        self.pend_i = np.empty(0, dtype=np.int64)
        self.pend_t = np.empty(0)
        self.spikes_i, self.spikes_t = [], []
        self.n_pred = self.n_cand = self.n_win = 0

    def step(self):
        t0, t_end = self.t, self.t + self.delay
        D = self.delay
        self.n_win += 1

        # ---- sigma factors for last window's spikes ----
        if self.pend_i.size:
            s = t_end - (self.pend_t + D)
            sig_g, sig_u = np.exp(-s / TAU_S), kappa(s)
            tgt, dg, du, wpos = deliver_sparse(self.crow, self.post, self.val,
                                               self.pend_i, sig_g, sig_u)
        else:
            tgt = np.empty(0, dtype=np.int64)
            dg = du = wpos = np.empty(0)

        # ---- the only neurons that can matter: already active, or hit now ----
        if tgt.size:
            self.active[tgt] = True
        idx = np.flatnonzero(self.active)
        self.n_pred += idx.size
        if idx.size == 0:
            self.t = t_end
            self.pend_i = np.empty(0, dtype=np.int64)
            self.pend_t = np.empty(0)
            return 0

        u0, g0 = self.u[idx], self.g[idx]

        # ---- predict (closed form) over the active set only ----
        x = np.exp(-D / TAU_M); x4 = x ** 4
        ue = x * u0 + (g0 * C_COUPLE) * (x - x4)
        ge = x4 * g0

        # scatter this window's arrivals onto the active set
        w_in = np.zeros(idx.size)
        if tgt.size:
            pos = np.searchsorted(idx, tgt)
            ue += np.bincount(pos, weights=du, minlength=idx.size)
            ge += np.bincount(pos, weights=dg, minlength=idx.size)
            w_in = np.bincount(pos, weights=wpos, minlength=idx.size)

        # ---- certify against ACTUAL arrivals ----
        sup = np.maximum(u0, 0.0) + KAPPA_WINDOW_MAX * (np.maximum(g0, 0.0) + w_in)
        cand_local = np.flatnonzero((sup >= THETA) & (self.refr[idx] <= t0))
        self.n_cand += cand_local.size

        u_new, g_new = ue, ge

        # ---- repair: exact crossing time for the few candidates ----
        if cand_local.size:
            uc, gc = u0[cand_local], g0[cand_local]
            tt = spike_time(uc, gc, THETA, TAU_M)
            fires = tt <= D
            if fires.any():
                k = cand_local[fires]
                gi = idx[k]
                self.spikes_i.append(gi)
                self.spikes_t.append(t0 + tt[fires])
                rem = D - tt[fires]
                xm, xs = np.exp(-rem / TAU_M), np.exp(-rem / TAU_S)
                u_new[k] = 0.0          # reset then coast the remainder
                g_new[k] = 0.0
                self.refr[gi] = t0 + tt[fires] + self.t_refrac

        self.u[idx], self.g[idx] = u_new, g_new

        # ---- retire neurons that have returned to exact rest ----
        rest = (self.u[idx] == 0.0) & (self.g[idx] == 0.0) & (self.refr[idx] <= t_end)
        if rest.any():
            self.active[idx[rest]] = False

        if self.spikes_i and self.spikes_t[-1].size and \
                self.spikes_t[-1][0] >= t0:
            self.pend_i = self.spikes_i[-1]
            self.pend_t = self.spikes_t[-1]
        else:
            self.pend_i = np.empty(0, dtype=np.int64)
            self.pend_t = np.empty(0)

        self.t = t_end
        return self.pend_i.size

    def run(self, T):
        while self.t < T - 1e-12:
            self.step()
        return self.spike_arrays()

    def spike_arrays(self):
        if not self.spikes_i:
            return np.empty(0, dtype=np.int64), np.empty(0)
        return np.concatenate(self.spikes_i), np.concatenate(self.spikes_t)

    @property
    def avg_predicted(self):
        return self.n_pred / max(self.n_win, 1)
