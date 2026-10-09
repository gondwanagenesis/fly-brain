"""SUPERFLY's life: one fly brain living in FlyWorld.

Every 20 ms of simulated time:
    world.sense()   -> Poisson rates on the fly's own receptor neurons, per side
    body state      -> the fly's interoceptive neurons (MBON11, PPL101, ISN, ITP)
    the brain       -> 20 ms of the 138,639-neuron network
    motor decode    -> descending / MN9 population rates (60 ms smoothing)
                       -> the minimal nerve cord in world.step()
Every 250 ms (the window the voice was trained on):
    observe         -> central-brain features -> FlyLM inner speech
    mouth           -> the inner speech is said aloud when its content is new
    memory          -> an episode when something happened

Nothing above the brain chooses an action: every movement other than the
nerve cord's own walking rhythm comes from the fly's descending neurons, and
everything the voice says comes from the central brain's activity.

    python -m superfly.life 120            # live 2 min, print what it says
    python -m superfly.life 300 --save     # and keep its memories
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict, field

import numpy as np

from superfly.bridge import parse_utterance, OUT, CACHE
from superfly.fly import SuperFly
from superfly.language import CONCEPTS, CONCEPT
from superfly.memory import EpisodicStore
from superfly.world import World, ARENA

INTERO = ("intero.hunger", "intero.satiety", "intero.ISN", "intero.thirst")


def intero_rates(n):
    """Body state -> rates (Hz) on interoceptive neurons (review 06 s3).
    Hunger raises MBON11 and lowers PPL101; ISNs rise with hunger and fall
    with thirst; ITP neurons rise with thirst."""
    return {"intero.hunger": 60.0 * n.hunger,
            "intero.satiety": 60.0 * (1.0 - n.hunger),
            "intero.ISN": 60.0 * min(1.0, max(0.0, 0.5 + n.hunger - n.thirst)),
            "intero.thirst": 60.0 * n.thirst}


@dataclass
class Moment:
    """One 250 ms window of the fly's life: the grounded record."""
    t: float                    # world time, s
    said: str                   # inner speech (grounded voice)
    labels: list                # parsed content of `said`
    actions: dict               # DN / MN9 rates over the window, Hz
    needs: dict
    valence: float              # MB output readout
    event: str = ""             # body event in the window
    place: str = ""
    heard: list = field(default_factory=list)
    aloud: bool = False
    spikes: int = 0
    groups: dict = field(default_factory=dict)   # mean Hz per region group (senses, thinking, action...)


class Life:
    DT_MS = 20.0
    VOICE_MS = 250.0
    TAU_MOTOR_MS = 60.0

    def __init__(self, seed=0, voice=True, learn=False, fly=None, world=None,
                 memory=None, threads=None, verbose=True):
        self.fly = fly or SuperFly(plasticity=True, drives=INTERO, seed=seed,
                                   threads=threads, verbose=verbose)
        f, e, at = self.fly, self.fly.e, self.fly.atlas
        if f.mb is not None:
            f.mb.enabled = bool(learn)
        self.world = world or World(seed=seed)
        self.voice = None
        if voice:
            from superfly.chat import TinyVoice
            self.voice = voice if not isinstance(voice, bool) else TinyVoice()
        self.memory = memory if memory is not None else EpisodicStore()
        # receptor positions per (concept, side)
        self.side_pos = {}
        for c in CONCEPTS:
            for side in ("left", "right"):
                ids = np.concatenate([at[p].side[side] for p in c.pops])
                pos = e.stim_positions(e.indices_of(ids))
                self.side_pos[(c.key, side)] = pos[pos >= 0]
        # motor readout: the fly's own descending / motor populations
        self.motor_slots = dict(f.act_groups)
        for side in ("left", "right"):
            self.motor_slots[f"turn_{side}"] = e.indices_of(at["dn.turn"].side[side])
        self._mkeys = list(self.motor_slots)
        self._mall = np.concatenate([self.motor_slots[k] for k in self._mkeys])
        self._msplit = np.cumsum([self.motor_slots[k].size for k in self._mkeys])[:-1]
        self._mprev = e.counts[self._mall].copy()
        self.motor = {k: 0.0 for k in self._mkeys}
        self.kc_slots = at["mb.KC"].idx(e)
        # bookkeeping
        self.lock = threading.RLock()
        self.moments: deque[Moment] = deque(maxlen=2400)   # 10 min
        self.aloud: deque = deque(maxlen=500)              # (t, text)
        self.events: deque = deque(maxlen=500)             # (t, event)
        self._hearing = []         # [(until_t, concepts, words)]
        self._teach_until = 0.0
        self._win_event = ""
        self._win_heard = []
        self._t_voice = 0.0
        self._last_aloud_t = -1e9
        self._spoken = {}          # label -> last time said aloud
        self._last_mem_t = -1e9
        self._log_n = len(self.world.body.log)
        self._frame_prev = e.counts.copy()
        self.listeners = []        # callables(kind, payload) for the Lab
        self.on_tick = None        # callable(life) after every 20 ms exchange
        self.wall_s = 0.0
        f.mark()

    # ------------------------------------------------------------ inputs
    def hear(self, text):
        """Someone speaks to the fly. Through the voice model's hearing heads
        the sentence becomes drive on the fly's own senses (and on the word
        lobe for quoted words) for 0.6 s. Praise/scolding is dopamine,
        which only changes anything when learning is on."""
        if self.voice is None:
            return {}, [], 0.0
        concepts, words, reward = self.voice.hear(text)
        with self.lock:
            self._hearing.append((self.world.t + 0.6, concepts, words))
            self._win_heard += words or [k for k in concepts]
            if abs(reward) > 0.5 and self.fly.mb is not None and self.fly.mb.enabled:
                self.fly.teach(np.sign(reward))
                self._teach_until = self.world.t + 0.3
        return concepts, words, reward

    def _apply_senses(self):
        f, e, w = self.fly, self.fly.e, self.world
        r = e._rate_np
        r[:] = 0.0
        for (k, side), hz in w.sense().items():
            if hz > 0.05:
                p = self.side_pos[(k, side)]
                r[p] = np.maximum(r[p], hz)
        self._hearing = [h for h in self._hearing if h[0] > w.t]
        for _, concepts, words in self._hearing:
            for k, inten in concepts.items():
                p = f.concept_pos[k]
                r[p] = np.maximum(r[p], CONCEPT[k].rate_hz * float(inten))
            for wd in words:
                p = f.word_channel(wd)
                if p is not None:
                    r[p] = 150.0
        for k, hz in intero_rates(w.needs).items():
            if k in f.drive_pos:
                r[f.drive_pos[k]] = hz
        if self._teach_until and w.t >= self._teach_until:
            f.teach(0)
            self._teach_until = 0.0
        e._refresh_refrac()
        e._poi_block = None

    # ------------------------------------------------------------ the loop
    def tick(self):
        with self.lock:
            f, e, w = self.fly, self.fly.e, self.world
            t0 = time.perf_counter()
            self._apply_senses()
            f.run(self.DT_MS)
            c = e.counts[self._mall]
            d = (c - self._mprev).astype(float)
            self._mprev = c.copy()
            a = 1.0 - math.exp(-self.DT_MS / self.TAU_MOTOR_MS)
            for k, part, s in zip(self._mkeys, np.split(d, self._msplit),
                                  (self.motor_slots[k] for k in self._mkeys)):
                hz = part.sum() / max(s.size, 1) / (self.DT_MS / 1000.0)
                self.motor[k] += a * (hz - self.motor[k])
            ev = w.step(self.DT_MS / 1000.0, self.motor)
            log = w.body.log
            for tt, name in log[self._log_n:]:
                self.events.append((tt, name))
                if name in ("shadow", "feed", "escape", "groom", "sleep", "wake"):
                    self._win_event = self._win_event or name
                self._emit("event", {"t": tt, "event": name})
            self._log_n = len(log)
            if ev:
                self._win_event = ev
            self._t_voice += self.DT_MS
            if self._t_voice >= self.VOICE_MS:
                self._t_voice = 0.0
                self._voice_tick()
            self.wall_s += time.perf_counter() - t0
            if self.on_tick is not None:
                self.on_tick(self)

    def _groups(self, regions):
        """Mean rate per region group: where in the brain the activity is."""
        from superfly.anatomy import REGIONS
        acc = {}
        for key, _, group in REGIONS:
            if key in regions and group in ("senses", "early", "thinking", "action"):
                acc.setdefault(group, []).append(regions[key])
        return {g: round(float(np.mean(v)), 3) for g, v in acc.items()}

    def place(self):
        b, w = self.world.body, self.world
        if b.on:
            return f"on the {b.on} drop"
        for o in w.objects:
            if math.hypot(b.x - o.x, b.y - o.y) < 12:
                return f"near the {o.kind} drop"
        if math.hypot(b.x - w.lamp[0], b.y - w.lamp[1]) < 12:
            return "near the warm lamp"
        if min(b.x, b.y, ARENA - b.x, ARENA - b.y) < 6:
            return "by the wall"
        return "in the open"

    def _voice_tick(self):
        f, w = self.fly, self.world
        obs = f.observe()
        feats = f.voice_features(obs)
        said = self.voice.speak(feats) if self.voice is not None else ""
        labels = sorted(parse_utterance(said))
        heard, ev = self._win_heard, self._win_event
        self._win_heard, self._win_event = [], ""
        m = Moment(round(w.t, 2), said, [list(l) for l in labels],
                   {k: round(v, 1) for k, v in obs.actions.items()},
                   w.needs.as_dict(), round(obs.valence, 3), ev, self.place(),
                   heard, False, obs.spikes, self._groups(obs.regions))
        # the mouth: say it when the content is new (or the body just acted)
        now = w.t
        fresh = [l for l in labels if now - self._spoken.get(l, -1e9) > 8.0]
        if labels and (fresh or ev in ("feed", "escape", "groom")) \
                and now - self._last_aloud_t >= 1.5:
            m.aloud = True
            self._last_aloud_t = now
            for l in labels:
                self._spoken[l] = now
            self.aloud.append((m.t, said))
            self._emit("say", {"t": m.t, "text": said, "labels": m.labels})
        self.moments.append(m)
        # memory: a stored brain state when something happened
        if ev or heard or m.aloud or now - self._last_mem_t >= 15.0:
            kc = np.flatnonzero(f._last_window[self.kc_slots] > 0)
            vec = self.voice.norm(feats) if self.voice is not None else np.log1p(feats)
            self.memory.add(m.t, kc, vec, said, m.place, m.needs, ev,
                            m.valence, heard)
            self._last_mem_t = now
        self._emit("moment", asdict(m))

    def _emit(self, kind, payload):
        for cb in self.listeners:
            try:
                cb(kind, payload)
            except Exception:
                pass

    def live(self, seconds, realtime=False, progress=None):
        n = int(round(seconds * 1000.0 / self.DT_MS))
        for i in range(n):
            t0 = time.perf_counter()
            self.tick()
            if realtime:
                time.sleep(max(0.0, self.DT_MS / 1000.0 - (time.perf_counter() - t0)))
            if progress and i % int(progress * 1000 / self.DT_MS) == 0:
                self.status()

    # ------------------------------------------------------------ for the Lab
    def frame(self):
        """Engine slots that spiked since the last call, with counts."""
        with self.lock:
            c = self.fly.e.counts
            d = c - self._frame_prev
            self._frame_prev = c.copy()
        nz = np.flatnonzero(d)
        return nz, d[nz]

    def status(self):
        w = self.world
        rt = w.t / max(self.wall_s, 1e-9)
        s = w.state()
        print(f"t={w.t:7.1f}s  x={s['fly']['x']:5.1f} y={s['fly']['y']:5.1f} "
              f"{s['fly']['mode']:6s} on={s['fly']['on'] or '-':6s} "
              f"E={w.needs.energy:.2f} W={w.needs.water:.2f} "
              f"A={w.needs.arousal:.2f}  speed {rt:.2f}x real time", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seconds", type=float, nargs="?", default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--learn", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--start", default=None, help="x,y,heading to start the fly at")
    a = ap.parse_args()
    life = Life(seed=a.seed, learn=a.learn)
    if a.start:
        x, y, h = (float(v) for v in a.start.split(","))
        b = life.world.body
        b.x, b.y, b.heading = x, y, h
    life.listeners.append(lambda k, p: print(f"   [{p['t']:6.1f}s] FLY: {p['text']}", flush=True)
                          if k == "say" else (print(f"   [{p['t']:6.1f}s] ({p['event']})", flush=True)
                                              if k == "event" else None))
    life.live(a.seconds, progress=10.0)
    life.status()
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"seconds": a.seconds, "seed": a.seed, "realtime_factor": life.world.t / life.wall_s,
           "events": list(life.events), "aloud": list(life.aloud),
           "moments": [asdict(m) for m in life.moments],
           "memory_episodes": len(life.memory.eps)}
    p = OUT / f"life_seed{a.seed}_{int(a.seconds)}s.json"
    p.write_text(json.dumps(rec, indent=1))
    print("wrote", p)
    if a.save:
        print("memories saved to", life.memory.save(CACHE / f"memory_seed{a.seed}.npz"))


if __name__ == "__main__":
    main()
