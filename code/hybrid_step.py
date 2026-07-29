"""Hybrid stepper: window mode when it pays, grid mode when it does not.

Why this exists
---------------
Delay-window stepping is spectacular on sparse activity and terrible on dense
activity. Measured against a dt=0.1 ms grid at full connectome scale:

    brain active     window vs grid
        0.88%            88x faster
        5%               10.8x
       25%               0.45x   <- already losing
       60%               0.12x
      100%               0.09x   <- 11x SLOWER

The cause is structural, not a bug. Window cost is
O(active) predict + O(candidates x segments) repair, and BOTH candidate count
and segment count rise with activity, so the repair term grows super-linearly.
Grid cost is a flat O(18 x N) regardless.

That makes plain window stepping useless for an embodied model driven by all
senses, where much of the brain is active at once -- exactly the case Eon cares
about. A speedup that only appears for a 21-neuron sugar stimulus is a special
case, not a contribution.

The fix is to decide per window, from quantities already computed, and take
whichever mode is cheaper. The window path is used only when its predicted cost
is below the grid's, so the hybrid is never slower than the grid by more than
the cost of the estimate itself, and keeps the full win when activity is low.

Both modes integrate exactly (Rotter & Diesmann propagator), so switching
between them changes only the spike-time resolution, never the subthreshold
trajectory.
"""
from __future__ import annotations
import numpy as np

from window_step import TAU_M, TAU_S, THETA, DELAY, KAPPA_WINDOW_MAX

C_COUPLE = TAU_S / (TAU_M - TAU_S)

STEPS_PER_WINDOW = int(round(DELAY / 0.1))


class ModeSelector:
    """Online auto-tuner: MEASURE both modes, then use the cheaper one.

    A static cost model was tried first and got this wrong. Calibrated on
    18,548 candidates it inferred 6.9e-8 s per quartic solve, but at 69,000
    candidates the true cost was 7.7e-7 s -- 11x higher, because the cost is
    superlinear once the working set leaves cache. A model fitted at one
    operating point cannot be trusted at another, and it would need
    recalibrating for every machine.

    So do not predict; measure. Run each mode for a few windows, keep the
    observed per-window cost, use whichever is cheaper, and re-probe
    occasionally so the choice tracks changing activity. The probe costs one
    window in the losing mode every `recheck` windows, i.e. O(1/recheck)
    overhead, in exchange for never being badly wrong on any hardware or at any
    activity level.
    """

    WINDOW, GRID = 0, 1

    def __init__(self, probe=3, recheck=64):
        self.probe, self.recheck = probe, recheck
        self.cost = [[], []]          # recent per-window seconds, per mode
        self.mode = self.WINDOW
        self.since = 0
        self._probing = self.WINDOW

    def next_mode(self):
        """Which mode to run this window."""
        if len(self.cost[self.WINDOW]) < self.probe:
            return self.WINDOW
        if len(self.cost[self.GRID]) < self.probe:
            return self.GRID
        if self.since >= self.recheck:          # re-probe the losing mode
            return self.GRID if self.mode == self.WINDOW else self.WINDOW
        return self.mode

    def record(self, mode, seconds):
        c = self.cost[mode]
        c.append(seconds)
        if len(c) > 8:
            del c[0]
        if len(self.cost[0]) >= self.probe and len(self.cost[1]) >= self.probe:
            mw = float(np.median(self.cost[self.WINDOW]))
            mg = float(np.median(self.cost[self.GRID]))
            new = self.WINDOW if mw < mg else self.GRID
            self.since = 0 if new != self.mode else self.since + 1
            self.mode = new
        else:
            self.since += 1

    @property
    def summary(self):
        f = lambda c: (float(np.median(c)) * 1000 if c else float('nan'))
        return {"mode": "window" if self.mode == self.WINDOW else "grid",
                "window_ms": f(self.cost[self.WINDOW]),
                "grid_ms": f(self.cost[self.GRID])}


def grid_window(u, g, arrivals, steps=STEPS_PER_WINDOW, dt=0.1,
                theta=THETA, t_refrac=2.2, refr=None, t0=0.0):
    """Advance one delay window by plain grid stepping, integrating exactly.

    arrivals: (step_index, target_index, weight) triples landing in this window.
    Returns (u, g, fired_idx, fired_t). Spike times land on the grid, which is
    the accuracy the grid mode gives up relative to window mode.
    """
    xm, xs = np.exp(-dt / TAU_M), np.exp(-dt / TAU_S)
    fired_i, fired_t = [], []
    by_step = {}
    if arrivals is not None and len(arrivals[0]):
        si, ti, wv = arrivals
        for k in range(len(si)):
            by_step.setdefault(int(si[k]), []).append((int(ti[k]), float(wv[k])))
    for s in range(steps):
        for tgt, w in by_step.get(s, ()):
            g[tgt] += w
        live = refr <= (t0 + s * dt) if refr is not None else slice(None)
        un = xm * u + (g * C_COUPLE) * (xm - xs)
        gn = xs * g
        if refr is not None:
            u = np.where(live, un, u); g = np.where(live, gn, g)
        else:
            u, g = un, gn
        hit = np.flatnonzero(u > theta)
        if hit.size:
            fired_i.append(hit)
            fired_t.append(np.full(hit.size, t0 + (s + 1) * dt))
            u[hit] = 0.0; g[hit] = 0.0
            if refr is not None:
                refr[hit] = t0 + (s + 1) * dt + t_refrac
    fi = np.concatenate(fired_i) if fired_i else np.empty(0, dtype=np.int64)
    ft = np.concatenate(fired_t) if fired_t else np.empty(0)
    return u, g, fi, ft
