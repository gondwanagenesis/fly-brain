"""Vectorised delay-window simulator. Exact, no fine sub-grid anywhere.

The prototype in window_sim.py proved the algorithm but repaired candidates with
per-neuron Python loops over a fine sub-grid. This version removes both.

Key realisation: inside a window, arrivals land at a small number of DISTINCT
times (one per source spike, all shifted by the same delay D). Between two
consecutive arrivals nothing enters the system, so the dynamics are free -- and
free evolution has a closed form AND a closed-form threshold crossing (the
quartic, because tau_mem/tau_syn = 4 exactly).

So a window is handled as a short sequence of segments:

    for each segment between consecutive arrival times:
        advance every candidate analytically           (one vector op)
        solve for threshold crossings analytically     (one vector op)
        apply resets to whoever crossed
    at each arrival time: scatter that arrival's weights into g

No sub-stepping, no tolerance, no missed spikes -- the crossing test is exact
rather than sampled. Segment count is bounded by the number of source spikes in
the window, which is small precisely because activity is sparse.

Delivery uses scipy CSR: dg = W @ sigma_g, du = W @ sigma_u.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp

from window_step import TAU_M, TAU_S, THETA, DELAY, kappa, free_window_map, certified_no_spike
from exact_spike_time import spike_time

C_COUPLE = TAU_S / (TAU_M - TAU_S)


def advance_vec(u, g, h):
    """Exact free evolution by h. h may be a scalar OR a per-neuron vector,
    which is what lets the post-reset remainder be applied without a loop."""
    xm, xs = np.exp(-h / TAU_M), np.exp(-h / TAU_S)
    return xm * u + (g * C_COUPLE) * (xm - xs), xs * g


def build_W(n, crow, post, val):
    """Sparse W with W[target, source] = weight, for dg = W @ sigma."""
    src = np.repeat(np.arange(n), np.diff(crow))
    return sp.csr_matrix((val, (post, src)), shape=(n, n))


class FastWindowSim:
    def __init__(self, n, crow, post, val, t_refrac=2.2, delay=DELAY):
        self.n, self.crow, self.post, self.val = n, crow, post, val
        self.W = build_W(n, crow, post, val)
        self.t_refrac, self.delay = t_refrac, delay
        self.w_pos_in = np.zeros(n)
        np.add.at(self.w_pos_in, post, np.maximum(val, 0.0))
        self.u = np.zeros(n); self.g = np.zeros(n)
        self.refr = np.full(n, -np.inf)
        self.t = 0.0
        self.pend_i = np.empty(0, dtype=np.int64)
        self.pend_t = np.empty(0)
        self.spikes = []
        self.stat_cert = 0
        self.stat_cand = 0

    def step(self):
        t0, t_end = self.t, self.t + self.delay

        # ---- deliver: two sparse mat-vecs ----
        du = np.zeros(self.n); dg = np.zeros(self.n)
        if self.pend_i.size:
            s = t_end - (self.pend_t + self.delay)          # age at window end
            sig_g = np.zeros(self.n); sig_u = np.zeros(self.n)
            np.add.at(sig_g, self.pend_i, np.exp(-s / TAU_S))
            np.add.at(sig_u, self.pend_i, kappa(s))
            dg = self.W @ sig_g
            du = self.W @ sig_u

        # ---- predict ----
        ue, ge = free_window_map(self.u, self.g, self.delay)
        ue += du; ge += dg

        # ---- certify ----
        safe = certified_no_spike(self.u, self.g, self.w_pos_in, THETA, self.delay)
        safe &= (self.refr <= t0)
        cand = np.flatnonzero(~safe)
        self.stat_cert += int(safe.sum()); self.stat_cand += cand.size

        u_new, g_new = ue.copy(), ge.copy()
        fired = []

        if cand.size:
            # arrival times inside this window, plus the window end as a sentinel
            arr_times = np.unique(self.pend_t + self.delay) if self.pend_i.size \
                else np.empty(0)
            arr_times = arr_times[(arr_times > t0) & (arr_times < t_end)]
            seg_ends = np.concatenate([arr_times, [t_end]])

            uc, gc = self.u[cand].copy(), self.g[cand].copy()
            rc = self.refr[cand].copy()

            # All arrivals for all candidates, as ONE sparse product:
            # arr[c, s] = total weight landing on candidate c at segment time s.
            n_seg = seg_ends.size
            arr = None
            if self.pend_i.size and arr_times.size:
                col = np.searchsorted(arr_times, self.pend_t + self.delay)
                ok = (col < arr_times.size) & \
                     (arr_times[np.clip(col, 0, arr_times.size - 1)]
                      == self.pend_t + self.delay)
                S = sp.csr_matrix(
                    (np.ones(ok.sum()), (self.pend_i[ok], col[ok])),
                    shape=(self.n, arr_times.size))
                arr = np.asarray((self.W[cand] @ S).todense())

            t = t0
            for si, ev_t in enumerate(seg_ends):
                h = ev_t - t
                if h > 0:
                    live = rc <= t
                    tt = np.full(cand.size, np.inf)
                    if live.any():
                        tt[live] = spike_time(uc[live], gc[live], THETA, TAU_M)
                    cross = live & (tt <= h)
                    plain = live & ~cross

                    if plain.any():                      # no crossing: one hop
                        uc[plain], gc[plain] = advance_vec(uc[plain], gc[plain], h)

                    if cross.any():                      # reset, then remainder
                        ks = np.flatnonzero(cross)
                        fired.append((cand[ks], t + tt[ks]))
                        rem = h - tt[ks]                 # per-neuron remainder
                        uu, gg = advance_vec(np.zeros(ks.size), np.zeros(ks.size), rem)
                        uc[ks], gc[ks] = uu, gg
                        rc[ks] = t + tt[ks] + self.t_refrac
                t = ev_t
                if arr is not None and si < arr_times.size:
                    gc += arr[:, si]                     # vectorised delivery

            u_new[cand], g_new[cand] = uc, gc
            self.refr[cand] = rc

        if fired:
            self.pend_i = np.concatenate([f[0] for f in fired]).astype(np.int64)
            self.pend_t = np.concatenate([f[1] for f in fired])
            self.spikes.append((self.pend_i, self.pend_t))
        else:
            self.pend_i = np.empty(0, dtype=np.int64)
            self.pend_t = np.empty(0)
        self.u, self.g, self.t = u_new, g_new, t_end
        return self.pend_i.size

    def spike_arrays(self):
        """All spikes as (neuron_idx[], time[])."""
        if not self.spikes:
            return np.empty(0, dtype=np.int64), np.empty(0)
        return (np.concatenate([a for a, _ in self.spikes]),
                np.concatenate([b for _, b in self.spikes]))

    def run(self, T):
        while self.t < T - 1e-12:
            self.step()
        return self.spikes

    @property
    def certified_fraction(self):
        tot = self.stat_cert + self.stat_cand
        return self.stat_cert / max(tot, 1)
