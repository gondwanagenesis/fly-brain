"""Deterministic head-to-head: delay-window stepping vs fixed-grid stepping.

No Poisson input, so there is no RNG to confound the comparison -- the network
is released from a fixed initial condition and left to evolve. Any difference
between runs is purely integration error.

Reference ("truth") is a very fine grid, dt = 0.002 ms.
"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))

from window_step import THETA, DELAY, kappa, free_window_map, certified_no_spike
from window_sim import advance
from validate_window_sim import make_net

T_REF = 2.2


def grid_run(n, crow, post, val, u0, g0, dt, T):
    u, g = u0.copy(), g0.copy()
    refr = np.full(n, -np.inf)
    D = int(round(DELAY / dt))
    buf = np.zeros((D + 1, n)); head = 0
    spikes = []
    for it in range(int(round(T / dt))):
        t = it * dt
        g = g + buf[head]; buf[head] = 0.0
        live = t >= refr
        un, gn = advance(u, g, dt)
        u = np.where(live, un, u); g = np.where(live, gn, g)
        fired = np.flatnonzero((u > THETA) & live)
        if fired.size:
            tgt = (head + D) % (D + 1)
            for i in fired:
                spikes.append((int(i), t + dt))
                lo, hi = crow[i], crow[i + 1]
                if hi > lo:
                    np.add.at(buf[tgt], post[lo:hi], val[lo:hi])
            u[fired] = 0.0; g[fired] = 0.0
            refr[fired] = t + dt + T_REF
        head = (head + 1) % (D + 1)
    return spikes


def window_run(n, crow, post, val, u0, g0, T, fine=0.002):
    """Window stepping: predict in closed form, certify, repair the rest.

    Repair uses a fine sub-step so the spike time is resolved properly; the
    saving comes from only doing that for neurons the bound cannot certify.
    """
    u, g = u0.copy(), g0.copy()
    refr = np.full(n, -np.inf)
    w_pos_in = np.zeros(n)
    np.add.at(w_pos_in, post, np.maximum(val, 0.0))
    pend_i = np.empty(0, dtype=np.int64); pend_t = np.empty(0)
    spikes = []
    t0 = 0.0
    n_cert = n_cand = n_win = 0

    while t0 < T - 1e-12:
        t_end = t0 + DELAY
        # ---- deliver last window's spikes at their exact arrival times ----
        arr_t = pend_t + DELAY
        du = np.zeros(n); dg = np.zeros(n)
        per_target = {}
        if pend_i.size:
            s = t_end - arr_t
            sg, su = np.exp(-s / 5.0), kappa(s)
            for k, j in enumerate(pend_i):
                lo, hi = crow[j], crow[j + 1]
                if hi > lo:
                    np.add.at(dg, post[lo:hi], val[lo:hi] * sg[k])
                    np.add.at(du, post[lo:hi], val[lo:hi] * su[k])
                    for tt, ww in zip(post[lo:hi], val[lo:hi]):
                        per_target.setdefault(int(tt), []).append((arr_t[k], float(ww)))

        # ---- predict (closed form, assumes no spike) ----
        ue, ge = free_window_map(u, g, DELAY)
        ue += du; ge += dg

        # ---- certify ----
        safe = certified_no_spike(u, g, w_pos_in, THETA, DELAY) & (refr <= t0)
        cand = np.flatnonzero(~safe)
        n_cert += int(safe.sum()); n_cand += cand.size; n_win += 1

        u_new, g_new = ue.copy(), ge.copy()
        fired_this = []
        # ---- repair only the candidates ----
        for i in cand:
            ui, gi, t = u[i], g[i], t0
            evs = sorted(per_target.get(int(i), [])) + [(t_end, 0.0)]
            for ev_t, ev_w in evs:
                seg = ev_t - t
                if seg > 0:
                    ns = max(1, int(np.ceil(seg / fine)))
                    h = seg / ns
                    for _ in range(ns):
                        if t >= refr[i]:
                            ui, gi = advance(ui, gi, h)
                            if ui > THETA:
                                fired_this.append((int(i), t + h))
                                ui, gi = 0.0, 0.0
                                refr[i] = t + h + T_REF
                        t += h
                gi += ev_w
            u_new[i], g_new[i] = ui, gi

        spikes.extend(fired_this)
        pend_i = np.array([i for i, _ in fired_this], dtype=np.int64)
        pend_t = np.array([tt for _, tt in fired_this], dtype=np.float64)
        u, g, t0 = u_new, g_new, t_end

    return spikes, n_cert / max(n_cert + n_cand, 1)


def rate_vec(spikes, n, T):
    r = np.zeros(n)
    for i, _ in spikes:
        r[i] += 1
    return r * 1000.0 / T


if __name__ == "__main__":
    n, T = 400, 54.0
    crow, post, val = make_net(n)
    rng = np.random.default_rng(3)
    # Seed a cascade: most neurons carry substantial conductance (g feeds u with
    # gain up to 0.157, so g ~ 40 can lift u by ~6 mV), and a few start already
    # above the 7 mV threshold to ignite the network.
    u0 = rng.uniform(0.0, 6.5, n)
    g0 = rng.uniform(5.0, 45.0, n)
    u0[:15] = 7.5

    res = {}
    for lbl, dt in (("truth  dt=0.002ms", 0.002), ("grid   dt=0.1ms", 0.1)):
        t = time.perf_counter(); sp = grid_run(n, crow, post, val, u0, g0, dt, T)
        res[lbl] = (rate_vec(sp, n, T), len(sp), time.perf_counter() - t)
        print(f"{lbl:<20} spikes={len(sp):5d}   {res[lbl][2]:7.2f}s")

    t = time.perf_counter()
    sp_w, cert_frac = window_run(n, crow, post, val, u0, g0, T)
    res["window dt=1.8ms"] = (rate_vec(sp_w, n, T), len(sp_w), time.perf_counter() - t)
    print(f"{'window dt=1.8ms':<20} spikes={len(sp_w):5d}   {res['window dt=1.8ms'][2]:7.2f}s"
          f"   certified-silent {100*cert_frac:.1f}%")

    truth = res["truth  dt=0.002ms"][0]
    print("\nmean |rate - truth|  (Hz), lower is better:")
    for lbl in ("grid   dt=0.1ms", "window dt=1.8ms"):
        e = np.abs(res[lbl][0] - truth)
        print(f"  {lbl:<20} {e.mean():8.4f}    spike-count err {res[lbl][1]-res['truth  dt=0.002ms'][1]:+d}")
