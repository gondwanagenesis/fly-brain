"""Whole-network simulator that advances one full 1.8 ms delay window per step.

Loop per window (predict -> certify -> repair):

  1. sigma:   turn last window's spikes into two per-source scalars
  2. deliver: dg, du = W @ sigma   (two sparse mat-vecs, replacing 18 rounds of
              irregular scatter)
  3. predict: advance every neuron in closed form assuming it does not spike
  4. certify: a conservative exact bound proves most neurons could not have
              crossed threshold; those are finished, exactly
  5. repair:  only the un-certifiable minority is re-examined at fine
              resolution, where the quartic solver gives the exact spike time

Steps 1-4 are O(N) with tiny constants and touch the connectome exactly twice.
Step 5 is O(candidates), which is small precisely because the network is sparse.

Accuracy note: this does NOT reproduce a dt=0.1 ms grid run bit-for-bit, and it
should not. A grid run quantises every spike to a multiple of dt, which adds a
systematic +dt/2 bias to every spike time and inflates the effective axonal
delay by 2.8%. This integrates exactly and places spikes off-grid, so it should
sit CLOSER to a very fine reference than dt=0.1 ms does. That is the test in
validate_window_sim.py.
"""
from __future__ import annotations
import numpy as np

from window_step import (TAU_M, TAU_S, THETA, DELAY, kappa,
                         free_window_map, certified_no_spike,
                         KAPPA_WINDOW_MAX)

C_COUPLE = TAU_S / (TAU_M - TAU_S)      # 1/3 for tau_m/tau_s = 4


def advance(u, g, h):
    """Exact free evolution of (u, g) by h. Vectorised, no spike handling."""
    xm = np.exp(-h / TAU_M)
    xs = np.exp(-h / TAU_S)
    return xm * u + (g * C_COUPLE) * (xm - xs), xs * g


class WindowSim:
    """One neuron population advanced a full delay window at a time.

    fan_crow/fan_post/fan_val: connectome grouped by PREsynaptic index
    (CSC of W, equivalently CSR of W.T), so the outgoing synapses of neuron j
    are fan_post[fan_crow[j]:fan_crow[j+1]].
    """

    def __init__(self, n, fan_crow, fan_post, fan_val, w_pos_in,
                 t_refrac=2.2, delay=DELAY):
        self.n = n
        self.crow, self.post, self.val = fan_crow, fan_post, fan_val
        self.w_pos_in = w_pos_in          # per-target sum of positive in-weights
        self.t_refrac, self.delay = t_refrac, delay
        self.u = np.zeros(n)
        self.g = np.zeros(n)
        self.refrac_until = np.full(n, -np.inf)
        self.t = 0.0
        # spikes emitted in the window just finished: (neuron idx, exact time)
        self._pending_idx = np.empty(0, dtype=np.int64)
        self._pending_t = np.empty(0, dtype=np.float64)
        self.spike_log = []

    # ---- step 1+2: sigma factorisation and delivery -------------------------
    def _deliver(self, t_end):
        """Accumulate this window's arrivals at the window END, exactly."""
        du = np.zeros(self.n)
        dg = np.zeros(self.n)
        if self._pending_idx.size == 0:
            return du, dg
        # A spike emitted at t_k arrives at t_k + D. Its age at the window end
        # is therefore t_end - (t_k + D), which lies in [0, D). This lag depends
        # only on the SOURCE spike, never on the target -- that is exactly what
        # makes the two mat-vecs below valid.
        s = t_end - (self._pending_t + self.delay)
        sg = np.exp(-s / TAU_S)
        su = kappa(s)
        for k, j in enumerate(self._pending_idx):
            lo, hi = self.crow[j], self.crow[j + 1]
            if hi > lo:
                np.add.at(dg, self.post[lo:hi], self.val[lo:hi] * sg[k])
                np.add.at(du, self.post[lo:hi], self.val[lo:hi] * su[k])
        return du, dg

    # ---- steps 3-5 ---------------------------------------------------------
    def step_window(self, fine_dt=0.01):
        t0, t_end = self.t, self.t + self.delay
        du, dg = self._deliver(t_end)

        # 3. predict: free evolution + delivered input, assuming no spike
        u_end, g_end = free_window_map(self.u, self.g, self.delay)
        u_end, g_end = u_end + du, g_end + dg

        # 4. certify: which neurons provably could not have crossed?
        safe = certified_no_spike(self.u, self.g, self.w_pos_in, THETA, self.delay)
        safe &= (self.refrac_until <= t0)      # refractory ones are handled below
        cand = np.flatnonzero(~safe)

        u_new, g_new = u_end.copy(), g_end.copy()

        # 5. repair only the candidates, at fine resolution with exact reset
        for i in cand:
            ui, gi, t = self.u[i], self.g[i], t0
            lo, hi = 0, 0
            # this neuron's arrivals inside the window, in time order
            arr_t, arr_w = self._arrivals_for(i, t0, t_end)
            events = list(zip(arr_t, arr_w)) + [(t_end, 0.0)]
            for ev_t, ev_w in events:
                seg = ev_t - t
                nsteps = max(1, int(np.ceil(seg / fine_dt)))
                h = seg / nsteps
                for _ in range(nsteps):
                    if t >= self.refrac_until[i]:
                        ui, gi = advance(ui, gi, h)
                        if ui > THETA:
                            self.spike_log.append((i, t + h))
                            ui, gi = 0.0, 0.0
                            self.refrac_until[i] = t + h + self.t_refrac
                    t += h
                gi += ev_w
            u_new[i], g_new[i] = ui, gi

        # neurons that spiked this window become next window's sources
        newly = [(i, tt) for (i, tt) in self.spike_log if t0 < tt <= t_end]
        self._pending_idx = np.array([i for i, _ in newly], dtype=np.int64)
        self._pending_t = np.array([tt for _, tt in newly], dtype=np.float64)

        self.u, self.g, self.t = u_new, g_new, t_end
        return len(newly)

    def _arrivals_for(self, i, t0, t_end):
        """Arrival times and weights landing on neuron i inside this window."""
        if self._pending_idx.size == 0:
            return np.empty(0), np.empty(0)
        ts, ws = [], []
        for k, j in enumerate(self._pending_idx):
            lo, hi = self.crow[j], self.crow[j + 1]
            if hi <= lo:
                continue
            m = self.post[lo:hi] == i
            if m.any():
                ts.append(self._pending_t[k] + self.delay)
                ws.append(float(self.val[lo:hi][m].sum()))
        if not ts:
            return np.empty(0), np.empty(0)
        o = np.argsort(ts)
        return np.asarray(ts)[o], np.asarray(ws)[o]
