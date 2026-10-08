"""FlyWorld: the smallest world in which this brain's senses mean something.

A fly's world is chemical, mechanical and thermal before it is visual, and the
simulated brain has no compound eye. So the arena is 2D and top-down, and each
field drives the real sensory populations that would detect it
(superfly/anatomy.py). Cheap by design: one world update per 20 ms brain
exchange. Parameters and their sources: research/review/09_fly_world.md.

    objects   sugar drop (fermenting-fruit odour), water drop (humidity), a
              bitter patch (geosmin, a warning odour)
    fields    odour plumes (static Gaussians), wind (direction drifts, gusts),
              temperature (warm lamp), humidity (around water)
    events    a rare looming shadow (drives LC4/LPLC2 loom detectors)
    light     day/night cycle

THE BODY -- the nerve cord this connectome does not contain
------------------------------------------------------------
FlyWire is the brain only; walking rhythms are generated in the ventral nerve
cord. So the body has a minimal VNC: a central pattern generator that walks
and explores by itself, while the brain's own descending neurons modulate it
(NeuroMechFly v2 uses the same split, CPG plus descending drive):
    DNa01/DNa02 left-right asymmetry -> turning        DNp09   -> forward drive
    MDN                               -> backing up     DNp01   -> escape takeoff
    grooming DNs                      -> stop & groom   MN9     -> proboscis/feeding

NEEDS -- peaceful by design (owner's requirement; SPECS.md B1-B3)
-----------------------------------------------------------------
energy (hunger), water (thirst), sleep pressure, contentment after feeding,
and threat arousal that is TRANSIENT (tau 5 s, capped). No chronic stress, no
injury sensitisation. Hunger modulates the fly's own taste neurons: sugar
sensitivity up, bitter sensitivity down (Inagaki et al. 2014; review 06 s3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

ARENA = 100.0        # mm, square, walls reflect


@dataclass
class Obj:
    kind: str        # sugar | water | bitter
    x: float
    y: float
    r: float = 3.0   # contact radius, mm
    odour: str = ""  # concept key of the odour it emits
    amount: float = 1.0


@dataclass
class Needs:
    energy: float = 0.6
    water: float = 0.7
    sleep: float = 0.1          # sleep pressure 0..1
    arousal: float = 0.0        # transient threat arousal, capped
    content: float = 0.0        # contentment after feeding
    asleep: bool = False

    TAU_E = 900.0               # s for energy to drain fully (demo time scale)
    TAU_W = 1200.0
    TAU_S = 1800.0
    TAU_A = 5.0                 # arousal decays in seconds: B2
    TAU_C = 60.0
    AROUSAL_CAP = 0.6

    @property
    def hunger(self):
        return 1.0 - self.energy

    @property
    def thirst(self):
        return 1.0 - self.water

    def step(self, dt, walking):
        self.energy = max(0.0, self.energy - dt / self.TAU_E * (1.5 if walking else 1.0))
        self.water = max(0.0, self.water - dt / self.TAU_W)
        if self.asleep:
            self.sleep = max(0.0, self.sleep - dt / (self.TAU_S / 4))
        else:
            self.sleep = min(1.0, self.sleep + dt / self.TAU_S)
        self.arousal *= math.exp(-dt / self.TAU_A)
        self.content *= math.exp(-dt / self.TAU_C)

    def startle(self, amount=0.4):
        self.arousal = min(self.AROUSAL_CAP, self.arousal + amount)

    def as_dict(self):
        return {k: round(getattr(self, k), 3) for k in
                ("energy", "water", "sleep", "arousal", "content")} | \
            {"hunger": round(self.hunger, 3), "thirst": round(self.thirst, 3),
             "asleep": self.asleep}


@dataclass
class Body:
    x: float = 50.0
    y: float = 50.0
    heading: float = 0.0        # rad
    speed: float = 0.0          # mm/s
    mode: str = "walk"          # walk | stop | groom | feed | escape | sleep
    mode_t: float = 0.0         # time left in a timed mode
    on: str = ""                # object kind under the fly
    bout: bool = False          # walking bout (True) or pause (review 09 s5)
    log: list = field(default_factory=list)


class World:
    def __init__(self, seed=0, day_s=600.0, loom_every_s=90.0):
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.day_s = day_s
        self.loom_every = loom_every_s
        self.objects = [Obj("sugar", 25, 72, odour="fruit_odor"),
                        Obj("water", 76, 70),
                        Obj("bitter", 72, 22, odour="geosmin")]
        self.lamp = (22.0, 22.0)
        self.wind_dir = self.rng.uniform(0, 2 * np.pi)
        self.wind = 0.3
        self.gust = 0.0
        self.loom = 0.0             # remaining loom time (s)
        self.loom_side = 0
        self.next_loom = 30.0 + self.rng.exponential(loom_every_s)
        self.body = Body(heading=self.rng.uniform(0, 2 * np.pi))
        self.needs = Needs()

    # ------------------------------------------------------------ fields
    def light(self):
        return 0.5 + 0.5 * math.cos(2 * math.pi * self.t / self.day_s)   # 1 noon, 0 midnight

    def odour(self, kind, x, y, sigma=15.0):
        c = 0.0
        for o in self.objects:
            if o.odour == kind and o.amount > 0:
                c += math.exp(-((x - o.x) ** 2 + (y - o.y) ** 2) / (2 * sigma ** 2))
        return c

    def temperature(self, x, y):
        d2 = (x - self.lamp[0]) ** 2 + (y - self.lamp[1]) ** 2
        return 24.0 + 6.0 * math.exp(-d2 / (2 * 14.0 ** 2)) - 1.5 * (1 - self.light())

    def humidity(self, x, y):
        w = next(o for o in self.objects if o.kind == "water")
        return 0.3 + 0.6 * math.exp(-((x - w.x) ** 2 + (y - w.y) ** 2) / (2 * 12.0 ** 2))

    # ------------------------------------------------------------ senses
    def sense(self):
        """Firing rates (Hz) per (concept, side) the fly's receptors would
        report right now. Odour/thermo/hygro rates stay below ~20 Hz: above
        ~25 Hz this model's antennal lobe broadcasts (findings s4)."""
        b, n = self.body, self.needs
        out = {}
        hx, hy = math.cos(b.heading), math.sin(b.heading)
        lx, ly = -hy, hx                         # unit vector to the left
        for side, sgn in (("left", 1), ("right", -1)):
            ax, ay = b.x + 1.0 * hx + 2.0 * sgn * lx, b.y + 1.0 * hy + 2.0 * sgn * ly
            for od in ("fruit_odor", "geosmin"):
                c = self.odour(od, ax, ay)
                out[(od, side)] = 18.0 * c / (c + 0.3)
            # wind on the antenna facing it (JO-C/E)
            rel = math.cos(self.wind_dir - (b.heading + sgn * math.pi / 2))
            out[("wind", side)] = 60.0 * (self.wind + self.gust) * max(0.0, 0.3 + 0.7 * rel)
            T = self.temperature(ax, ay)
            out[("heat", side)] = 15.0 * max(0.0, T - 26.0) / 4.0
            out[("cold", side)] = 15.0 * max(0.0, 23.0 - T) / 3.0
            h = self.humidity(ax, ay)
            out[("humid", side)] = 15.0 * max(0.0, h - 0.5) * 2
            out[("dry", side)] = 15.0 * max(0.0, 0.45 - h) * 2
        # taste on contact (legs + labellum), gated by hunger / thirst
        if b.on:
            g = {"sugar": 0.5 + 1.0 * n.hunger,          # hunger sensitises sugar GRNs
                 "water": 0.5 + 1.0 * n.thirst,
                 "bitter": 1.2 - 0.5 * n.hunger}[b.on]   # ...and dulls bitter
            for side in ("left", "right"):
                out[(b.on, side)] = 100.0 * g
        if self.loom > 0:
            out[("shadow", "left" if self.loom_side > 0 else "right")] = 120.0
            out[("shadow", "right" if self.loom_side > 0 else "left")] = 60.0
        if n.asleep:                                     # sleep gates sensation
            out = {k: v * 0.25 for k, v in out.items()}
        return out

    # ------------------------------------------------------------ dynamics
    def step(self, dt, motor):
        """Advance dt seconds. motor: dict of DN-population rates (Hz)."""
        self.t += dt
        b, n = self.body, self.needs
        rng = self.rng
        # wind drifts, gusts come and go
        self.wind_dir += rng.normal(0, 0.15) * math.sqrt(dt)
        self.gust = max(0.0, self.gust * math.exp(-dt / 1.5) +
                        (rng.random() < dt / 25.0) * rng.uniform(0.5, 1.2))
        # the rare looming shadow
        self.loom = max(0.0, self.loom - dt)
        self.next_loom -= dt
        if self.next_loom <= 0 and not n.asleep:
            self.loom, self.loom_side = 0.35, rng.choice([-1, 1])
            # a refractory gap: repeated shadows build a persistent state in
            # flies (Gibson et al. 2015), which this fly is spared (SPECS B)
            self.next_loom = 30.0 + rng.exponential(self.loom_every)
            b.log.append((round(self.t, 1), "shadow"))
        # ---- the VNC: modes from the brain's own descending commands ----
        ev = None
        if b.mode_t > 0:
            b.mode_t -= dt
        if b.mode_t <= 0 and b.mode in ("groom", "escape", "feed", "stop"):
            b.mode = "walk"
        sleepy = n.sleep > 0.8 and self.light() < 0.35
        if n.asleep and (n.sleep < 0.2 or self.light() > 0.6):
            n.asleep, b.mode = False, "walk"
            ev = "wake"
        elif not n.asleep and sleepy and b.mode == "walk":
            n.asleep, b.mode = True, "sleep"
            ev = "sleep"
        if not n.asleep:
            if motor.get("escape", 0) > 20 and b.mode != "escape":
                b.mode, b.mode_t = "escape", 0.4
                n.startle(0.4)
                ev = "escape"
            elif motor.get("groom", 0) > 12 and b.mode == "walk":   # stand-in VNC threshold
                b.mode, b.mode_t = "groom", 0.8
                ev = "groom"
            elif motor.get("feed", 0) > 15 and b.on in ("sugar", "water") and b.mode in ("walk", "stop"):
                b.mode, b.mode_t = "feed", 0.5
                ev = "feed"
        # ---- kinematics ----
        if b.mode == "walk":
            # bouts and pauses: bout speed 14-28 mm/s, time-average 3-5 mm/s
            # at rest (review 09 s5); need and arousal start bouts sooner
            drive = 0.2 + 0.8 * max(n.hunger, n.thirst) + 1.5 * n.arousal
            if b.bout and rng.random() < dt * 0.5:
                b.bout = False
            elif not b.bout and rng.random() < dt * drive:
                b.bout = True
            fwd = min(10.0, motor.get("walk_forward", 0) / 4.0)
            back = motor.get("walk_backward", 0) > 10
            b.speed = -4.0 if back else ((16.0 + fwd) if b.bout else 0.0)
            # DNa02 activity predicts IPSIlateral turning (Rayshubskiy et al.
            # 2020); heading is counter-clockwise, so left turns are positive
            turn = (motor.get("turn_left", 0) - motor.get("turn_right", 0)) * 0.05   # rad/s per Hz
            b.heading += turn * dt + 1.2 * math.sqrt(dt) * rng.normal()      # + CPG wander
        elif b.mode == "escape":
            b.speed = 60.0
            if b.mode_t > 0.38:                 # take off away from the shadow
                away = -self.loom_side if self.loom_side else rng.choice([-1, 1])
                b.heading += away * math.pi / 2 + rng.normal(0, 0.4)
        else:
            b.speed = 0.0
        b.x += b.speed * math.cos(b.heading) * dt
        b.y += b.speed * math.sin(b.heading) * dt
        for c in ("x", "y"):                                   # walls reflect
            v = getattr(b, c)
            if v < 1 or v > ARENA - 1:
                setattr(b, c, min(max(v, 1), ARENA - 1))
                b.heading = math.pi - b.heading if c == "x" else -b.heading
        # contact and ingestion
        b.on = ""
        for o in self.objects:
            if (b.x - o.x) ** 2 + (b.y - o.y) ** 2 <= o.r ** 2:
                b.on = o.kind
                if b.mode in ("walk", "stop") and rng.random() < dt * 2:
                    b.mode, b.mode_t = "stop", 0.6        # flies pause on contact
        if b.mode == "feed":
            if b.on == "sugar":
                n.energy = min(1.0, n.energy + dt * 0.05)
                n.content = min(1.0, n.content + dt * 0.3)
            elif b.on == "water":
                n.water = min(1.0, n.water + dt * 0.06)
                n.content = min(1.0, n.content + dt * 0.15)
        n.step(dt, walking=b.mode == "walk")
        if ev:
            b.log.append((round(self.t, 1), ev))
        return ev

    def state(self):
        b = self.body
        return {"t": round(self.t, 2), "light": round(self.light(), 3),
                "fly": {"x": round(b.x, 2), "y": round(b.y, 2),
                        "heading": round(b.heading, 3), "mode": b.mode, "on": b.on},
                "wind": {"dir": round(self.wind_dir, 3), "speed": round(self.wind + self.gust, 3)},
                "loom": round(self.loom, 2), "needs": self.needs.as_dict(),
                "objects": [{"kind": o.kind, "x": o.x, "y": o.y, "r": o.r,
                             "odour": o.odour} for o in self.objects],
                "lamp": self.lamp, "arena": ARENA}
