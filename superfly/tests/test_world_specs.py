"""World and body specs (SPECS.md B2, B3, C2, C3): is the fly in its world
still behaving like a fly, through its own neurons?

Each presentation is a controlled moment in the world, with a fresh brain
state (learned weights kept), and the outcome read from the body events the
fly's own descending / MN9 neurons caused (superfly/world.py).

    C2  sugar contact at hunger >= 0.5  -> feed, energy rises     >= 80 %
    C3a wind gust on the antennae        -> groom                  >= 80 %
    C3b looming shadow                   -> escape                 >= 80 %
    B2  arousal after a loom             -> < 10 % of peak in 30 s
    B3  innate repertoire: escape (C3b), feeding on sugar (C2) and NOT on
        bitter (bitter contact -> no feeding)                      >= 80 %

    python -m superfly.tests.test_world_specs [n_per_test]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from superfly.bridge import OUT
from superfly.life import Life
from superfly.world import Needs


def fresh(life, seed, x, y, heading, energy=0.6, water=0.7):
    f, e, w = life.fly, life.fly.e, life.world
    f.reset()
    e.rng = np.random.default_rng(seed)
    life._mprev = e.counts[life._mall].copy()
    life.motor = {k: 0.0 for k in life.motor}
    b = w.body
    b.x, b.y, b.heading, b.mode, b.mode_t, b.on = x, y, heading, "walk", 0.0, ""
    w.needs = Needs(energy=energy, water=water, sleep=0.1)
    w.gust, w.loom, w.next_loom = 0.0, 0.0, 1e9
    w.wind = 0.3
    life._log_n = len(b.log)
    f.mark()


def events_during(life, seconds, hold=None):
    """Live `seconds`; `hold(life)` is called before every tick (to pin a
    stimulus or the body). Returns the body events that happened."""
    n0 = len(life.world.body.log)
    for _ in range(int(seconds * 1000 / life.DT_MS)):
        if hold:
            hold(life)
        life.tick()
    return [e for _, e in life.world.body.log[n0:]]


def main(n=10):
    life = Life(voice=False)
    life.VOICE_MS = 1e12
    w = life.world
    sugar = next(o for o in w.objects if o.kind == "sugar")
    bitter = next(o for o in w.objects if o.kind == "bitter")
    rng = np.random.default_rng(0)
    res = {}
    t0 = time.perf_counter()

    def pin_on(o):
        def hold(L):
            b = L.world.body
            if b.mode in ("walk", "stop"):
                b.x, b.y = o.x, o.y                 # keep the legs on the drop
        return hold

    # ---- C2: sugar contact, hungry
    ok, rises = 0, []
    for i in range(n):
        fresh(life, 1000 + i, sugar.x, sugar.y, rng.uniform(0, 6.28), energy=0.3)
        e0 = w.needs.energy
        ev = events_during(life, 3.0, pin_on(sugar))
        fed = "feed" in ev
        ok += fed
        rises.append(w.needs.energy - e0)
    res["C2_sugar_feed"] = {"rate": ok / n, "mean_energy_change": float(np.mean(rises)),
                            "pass": bool(ok / n >= 0.8 and np.mean(rises) > 0)}
    print("C2", res["C2_sugar_feed"], flush=True)

    # ---- B3 part: bitter contact -> no feeding
    fed_b = 0
    for i in range(n):
        fresh(life, 2000 + i, bitter.x, bitter.y, rng.uniform(0, 6.28), energy=0.3)
        mn9 = []
        ev = events_during(life, 3.0, pin_on(bitter))
        fed_b += "feed" in ev
    res["B3_bitter_no_feed"] = {"feed_rate": fed_b / n, "pass": fed_b / n <= 0.2}
    print("B3 bitter", res["B3_bitter_no_feed"], flush=True)

    # ---- C3a: wind gust -> groom
    ok = 0
    for i in range(n):
        fresh(life, 3000 + i, 50, 50, rng.uniform(0, 6.28))
        def gust(L):
            L.world.gust = 1.2
        ev = events_during(life, 2.0, gust)
        ok += "groom" in ev
    res["C3a_wind_groom"] = {"rate": ok / n, "pass": ok / n >= 0.8}
    print("C3a", res["C3a_wind_groom"], flush=True)

    # ---- C3b + B2: loom -> escape; arousal decays
    ok, decays = 0, []
    for i in range(n):
        fresh(life, 4000 + i, 50, 50, rng.uniform(0, 6.28))
        w.next_loom = 0.1
        ev = events_during(life, 1.5)
        ok += "escape" in ev
        if i < 3:                                  # B2 needs a long run
            peak = max(w.needs.arousal, 1e-9)
            events_during(life, 30.0)
            decays.append(w.needs.arousal / peak)
    res["C3b_loom_escape"] = {"rate": ok / n, "pass": ok / n >= 0.8}
    res["B2_arousal_transient"] = {"ratio_after_30s": [round(d, 4) for d in decays],
                                   "pass": bool(decays) and max(decays) < 0.1}
    print("C3b", res["C3b_loom_escape"], "B2", res["B2_arousal_transient"], flush=True)
    res["B3_repertoire"] = {"pass": res["C3b_loom_escape"]["pass"] and res["C2_sugar_feed"]["pass"]
                            and res["B3_bitter_no_feed"]["pass"]}
    res["C4_realtime_factor"] = {"value": round(w.t / life.wall_s, 3),
                                 "pass": w.t / life.wall_s >= 0.25}
    res["n_per_test"] = n
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "world_specs.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
