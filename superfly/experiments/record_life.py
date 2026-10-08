"""Record the fly's brain while it lives in FlyWorld, for the voice.

The first corpus (make_corpus.py) drove receptors at 60-150 Hz; the world
drives them at the rates its fields produce (odours <= 18 Hz, see world.py).
A voice trained on the first corpus misreports in the world (it said "i smell
co2" with no CO2 anywhere). So this corpus is recorded where the voice will
be used: short life segments from varied places, needs and events.

Each segment: fresh brain state (learned weights kept), a random start
(half near an object or the lamp), random hunger/thirst, sometimes a looming
shadow, sometimes someone speaking (sense drive as the hearing heads would
give it) or a word in the word lobe. 100 ms settle, then four 250 ms windows.

Labels per window -- what a decoder is trained to report:
    percepts   from the world's receptor drive, averaged over the window
               (max over sides) and present above a floor per sense;
               intensity = drive / reference
    states     hungry / thirsty when the need is >= 0.6 (the drive on the
               interoceptive neurons, superfly/life.py)
    words      the word in the word lobe
    actions    the fly's OWN descending / MN9 rates (>= 5 Hz), not the world

    python -m superfly.experiments.record_life [segments] [seed]
"""
from __future__ import annotations

import sys
import time

import numpy as np

from superfly.engine import ROOT
from superfly.language import CONCEPTS, ACTIONS
from superfly.life import Life
from superfly.world import Needs

from superfly.bridge import CACHE as OUT  # noqa: E402 (per connectome)
VOCAB = ["zap", "yum", "blip", "moo", "dax", "wug", "fep", "kiki", "bouba",
         "toma", "lupo", "rin", "sol", "nef", "vax", "pim"]
STATES = ["hungry", "thirsty"]
# (floor Hz for "present", reference Hz for intensity 1) per sense class
SCALE = {"taste": (20.0, 100.0), "smell": (5.0, 15.0), "wind": (20.0, 50.0),
         "thermo": (6.0, 15.0), "hygro": (6.0, 15.0), "shadow": (30.0, 120.0),
         "sound": (20.0, 100.0)}
CLASS = {"sugar": "taste", "bitter": "taste", "water": "taste", "salt": "taste",
         "umami": "taste", "co2": "smell", "geosmin": "smell",
         "male_pheromone": "smell", "fruit_odor": "smell", "sound": "sound",
         "wind": "wind", "heat": "thermo", "cold": "thermo", "dry": "hygro",
         "humid": "hygro", "shadow": "shadow"}


def main(n_seg=600, seed=0, windows=4, win_ms=250.0, settle_ms=100.0):
    rng = np.random.default_rng(seed)
    life = Life(seed=seed, voice=False)
    life.VOICE_MS = 1e12                     # windows are taken here
    f, e, w = life.fly, life.fly.e, life.world
    keys = [c.key for c in CONCEPTS] + STATES
    acc = []
    orig = w.sense

    def sensing():
        o = orig()
        acc.append(o)
        return o
    w.sense = sensing
    X, C, W, A, meta = [], [], [], [], []
    t0 = time.perf_counter()
    for sg in range(n_seg):
        # ---- a new moment of life
        f.reset()
        e.rng = np.random.default_rng(seed * 100003 + sg)
        life._mprev = e.counts[life._mall].copy()
        life.motor = {k: 0.0 for k in life.motor}
        b = w.body
        if rng.random() < 0.5:
            spots = [(o.x, o.y) for o in w.objects] + [w.lamp]
            x0, y0 = spots[rng.integers(len(spots))]
            r, a = rng.uniform(0, 14), rng.uniform(0, 2 * np.pi)
            b.x, b.y = float(np.clip(x0 + r * np.cos(a), 2, 98)), float(np.clip(y0 + r * np.sin(a), 2, 98))
        else:
            b.x, b.y = rng.uniform(5, 95, 2)
        b.heading = rng.uniform(0, 2 * np.pi)
        b.mode, b.mode_t = "walk", 0.0
        w.needs = Needs(energy=float(rng.uniform(0.05, 1.0)),
                        water=float(rng.uniform(0.05, 1.0)), sleep=0.1)
        w.gust = float(rng.choice([0.0, 0.0, rng.uniform(0.3, 1.2)]))
        w.wind_dir = rng.uniform(0, 2 * np.pi)
        w.loom = 0.0
        w.next_loom = float(rng.uniform(0.1, 1.0)) if rng.random() < 0.2 else 1e9
        life._hearing = []
        heard_c, word = {}, ""
        if rng.random() < 0.25:
            k = str(rng.choice([c.key for c in CONCEPTS]))
            heard_c = {k: float(rng.choice([0.4, 0.7, 1.0]))}
        if rng.random() < 0.25:
            word = str(rng.choice(VOCAB))
        if heard_c or word:
            life._hearing.append((w.t + 1e6, heard_c, [word] if word else []))
        # ---- settle, then record
        for _ in range(int(settle_ms / life.DT_MS)):
            life.tick()
        f.mark()
        for wi in range(windows):
            acc.clear()
            for _ in range(int(win_ms / life.DT_MS)):
                life.tick()
            obs = f.observe()
            X.append(f.voice_features(obs).astype(np.float16))
            side_mean = {}                   # window mean per (sense, side)
            for o in acc:
                for (k, side), hz in o.items():
                    side_mean[(k, side)] = side_mean.get((k, side), 0.0) + hz / len(acc)
            row = []
            for c in CONCEPTS:
                fl, ref = SCALE[CLASS[c.key]]
                m = max(side_mean.get((c.key, "left"), 0.0), side_mean.get((c.key, "right"), 0.0))
                v = min(1.0, m / ref) if m >= fl else 0.0
                v = max(v, heard_c.get(c.key, 0.0))
                row.append(round(v, 3))
            n = w.needs
            row += [round(n.hunger, 3) if n.hunger >= 0.6 else 0.0,
                    round(n.thirst, 3) if n.thirst >= 0.6 else 0.0]
            C.append(row)
            W.append(word)
            A.append([obs.actions[a] for a in ACTIONS])
            meta.append(obs.spikes)
        if (sg + 1) % 25 == 0:
            el = time.perf_counter() - t0
            print(f"  {sg + 1}/{n_seg} segments  {el:.0f}s  (~{el / (sg + 1) * (n_seg - sg - 1):.0f}s left)",
                  flush=True)
            if (sg + 1) % 100 == 0:
                save(X, C, W, A, meta, keys, f, e, seed)
    save(X, C, W, A, meta, keys, f, e, seed)


def save(X, C, W, A, meta, keys, f, e, seed):
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "corpus_world.npz"
    keep_t = np.array([t for t in f.feat_names if not t.startswith("KC")])
    kc = f.atlas["mb.KC"].idx(e)
    np.savez_compressed(out, X=np.asarray(X), C=np.asarray(C, np.float32),
                        W=np.asarray(W), A=np.asarray(A, np.float32),
                        spikes=np.asarray(meta), concept_keys=np.asarray(keys),
                        action_keys=np.asarray(list(ACTIONS)),
                        feat_names=np.concatenate([keep_t, [f"KC#{j}" for j in range(kc.size)]]),
                        model=e.model, gain=e.gain, seed=seed)
    print(f"saved {out}: {len(X)} windows", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 600, int(a[1]) if len(a) > 1 else 0)
