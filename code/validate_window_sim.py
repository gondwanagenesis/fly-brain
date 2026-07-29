"""Accuracy test for delay-window stepping on a small random network.

The claim is NOT that window stepping reproduces a dt=0.1 ms grid run. It should
not: a grid run quantises every spike to a multiple of dt, adding a systematic
+dt/2 bias and inflating the effective 1.8 ms axonal delay by ~2.8%.

The claim is that window stepping is CLOSER TO THE TRUTH than dt=0.1 ms, where
truth is a very fine grid run (dt=0.002 ms). If window stepping -- while taking
18x fewer steps -- lands nearer the fine reference than dt=0.1 ms does, then it
is both faster and more accurate, which is the whole point.
"""
import sys, time
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

from window_step import TAU_M, TAU_S, THETA, DELAY
from window_sim import advance

T_REF = 2.2


def make_net(n=400, k=12, seed=0):
    rng = np.random.default_rng(seed)
    pre = np.repeat(np.arange(n), k)
    post = rng.integers(0, n, n * k)
    val = rng.normal(0.9, 0.45, n * k) * rng.choice([1.0, -1.0], n * k, p=[.6, .4])
    keep = pre != post                     # no autapses (precondition)
    pre, post, val = pre[keep], post[keep], val[keep]
    o = np.argsort(pre, kind='stable')
    pre, post, val = pre[o], post[o], val[o]
    crow = np.zeros(n + 1, dtype=np.int64)
    crow[1:] = np.cumsum(np.bincount(pre, minlength=n))
    return crow, post, val


def grid_run(n, crow, post, val, dt, T, drive_idx, drive_rate, seed=7):
    """Reference: plain clock-driven run, spikes quantised to the grid."""
    rng = np.random.default_rng(seed)
    u = np.zeros(n); g = np.zeros(n)
    refr = np.full(n, -np.inf)
    D = int(round(DELAY / dt))
    buf = np.zeros((D + 1, n)); head = 0
    spikes = []
    nsteps = int(round(T / dt))
    p_fire = drive_rate * dt / 1000.0
    for it in range(nsteps):
        t = it * dt
        g = g + buf[head]                    # arrivals scheduled D steps ago
        buf[head] = 0.0
        live = t >= refr
        un, gn = advance(u, g, dt)
        u = np.where(live, un, u); g = np.where(live, gn, g)
        hit = rng.random(len(drive_idx)) < p_fire
        if hit.any():
            u[drive_idx[hit]] += 3.2
        fired = np.flatnonzero((u > THETA) & live)
        if fired.size:
            for i in fired:
                spikes.append((int(i), t + dt))
                lo, hi = crow[i], crow[i + 1]
                if hi > lo:
                    np.add.at(buf[(head + D) % (D + 1)], post[lo:hi], val[lo:hi])
            u[fired] = 0.0; g[fired] = 0.0
            refr[fired] = t + dt + T_REF
        head = (head + 1) % (D + 1)
    return spikes


def rates(spikes, n, T):
    r = np.zeros(n)
    for i, _ in spikes:
        r[i] += 1
    return r * 1000.0 / T


if __name__ == "__main__":
    n, T = 400, 60.0
    crow, post, val = make_net(n)
    drive = np.arange(8)

    print(f"network: {n} neurons, {len(post)} synapses, T={T} ms, drive={len(drive)} @300Hz\n")
    out = {}
    for label, dt in (("fine dt=0.002ms (truth)", 0.002),
                      ("dt=0.1ms (their default)", 0.1),
                      ("dt=0.2ms", 0.2)):
        t0 = time.perf_counter()
        sp = grid_run(n, crow, post, val, dt, T, drive, 300.0)
        el = time.perf_counter() - t0
        out[label] = rates(sp, n, T)
        print(f"{label:<26} spikes={len(sp):5d}  {el:7.2f}s")

    truth = out["fine dt=0.002ms (truth)"]
    print()
    print("mean |rate - truth| across neurons (Hz):")
    for label in ("dt=0.1ms (their default)", "dt=0.2ms"):
        e = np.abs(out[label] - truth)
        print(f"  {label:<26} {e.mean():8.4f}   max {e.max():8.4f}")
    print()
    print("Interpretation: this establishes the grid-error baseline that window")
    print("stepping must beat. Window stepping places spikes off-grid, so its")
    print("error should fall below the dt=0.1ms row while taking 18x fewer steps.")
