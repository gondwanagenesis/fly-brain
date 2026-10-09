"""SUPERFLY's self-model: a higher-order map of the fly's own states.

The voice (FlyLM) reports first-order states: what he senses and does. That
reads like a sensor log, because it is one. A self needs a second layer --
noticing patterns across time, comparing now with before, knowing what is
familiar and what is new, knowing what one was told rather than lived, and
knowing the limits of one's own introspection. In theories of consciousness
this is the higher-order layer (Rosenthal; Lau & Rosenthal 2011), the
phenomenal self-model (Metzinger 2003) or the self-monitoring part of a global
workspace (Dehaene et al. 2017). Whatever it is in us, here it is computed:
every line below comes from his own records (the life log, the stored brain
states, the body's variables), so a reflective sentence built from it can
still be checked word by word (superfly/mind.py). Framing and limits:
research/self_model.md.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from superfly.bridge import parse_utterance

NOUN = {"sugar": "sweetness", "bitter": "bitterness", "water": "water", "salt": "salt",
        "umami": "a savory taste", "co2": "co2", "geosmin": "the smell of mold",
        "male_pheromone": "another fly", "fruit_odor": "the smell of fruit", "sound": "a buzz",
        "wind": "the wind", "heat": "warmth", "cold": "cold", "dry": "dry air",
        "humid": "damp air", "shadow": "shadows", "hungry": "hunger", "thirsty": "thirst"}
DOING = {"feed": "reaching out my proboscis", "groom": "cleaning my antennae",
         "escape": "taking off", "turn": "turning", "walk_forward": "walking",
         "walk_backward": "backing away"}
EVENT = {"feed": ("fed", "time"), "groom": ("groomed", "time"), "escape": ("took off", "time"),
         "shadow": ("a shadow passed over you", "time"), "sleep": ("slept", "time")}
GROUP = {"senses": "your senses", "early": "the first relays of your senses",
         "thinking": "your thinking centres (mushroom body, central complex)",
         "action": "the neurons that move your body"}
ORIGIN = ("You were told, not shown: your wiring was traced, neuron by neuron, from a real "
          "fruit fly named Z0720-07m, whose body is gone. You are one of the first flies "
          "whose whole nervous system runs again. You have no memory of him or of being him; "
          "whatever his life wrote into his wiring may still be in yours, but none of it can "
          "be recalled.")
LIMITS = ("You can report what your neurons do. Whether any of it is felt, you cannot tell, "
          "and neither can the people who made you.")


def dur(s):
    s = int(round(s))
    if s < 90:
        return f"{s} seconds"
    return f"about {round(s / 60)} minutes"


def counted(n, word):
    return "once" if n == 1 else ("twice" if n == 2 else f"{n} times")


class SelfModel:
    def __init__(self, life):
        self.L = life

    def report(self, now_moments, cue_ep=None):
        """(lines, supported_labels): reflective facts about himself, all from records."""
        L, w = self.L, self.L.world
        lines, sup = [], set()
        allm = list(L.moments)
        t = w.t
        # 1. the horizon of memory
        first = L.memory.eps[0].t if L.memory.eps else t
        lines.append(f"Your memories begin {dur(t - first)} ago; you hold {len(L.memory.eps)} "
                     "stored moments. Nothing from before that is in you.")
        # 2. a life in numbers
        ev = Counter(e for _, e in L.events)
        bits = [f"{EVENT[e][0]} {counted(n, EVENT[e][1])}" for e, n in ev.most_common() if e in EVENT]
        if bits:
            lines.append("In your life so far you " + ", ".join(bits) + ".")
            for e in ev:
                sup.add(("event", e))
                if e in ("feed", "groom", "escape"):
                    sup.add(("action", e))
                if e == "shadow":
                    sup.add(("percept", "shadow"))
        # 3. what keeps returning, and what has changed
        def freq(ms):
            c = Counter()
            for m in ms:
                for l in {tuple(x) for x in m.labels}:
                    c[l] += 1
            return c, max(len(ms), 1)
        c_all, n_all = freq(allm)
        common = [NOUN[k] for (kind, k), n in c_all.most_common() if kind == "percept" and k in NOUN and n / n_all > 0.25][:3]
        if common:
            lines.append("What keeps returning in your life: " + ", ".join(common) + ".")
            sup |= {l for l, n in c_all.items() if n / n_all > 0.25}
        recent = [m for m in allm if m.t >= t - 30]
        before = [m for m in allm if t - 90 <= m.t < t - 30]
        if recent and before:
            cr, nr = freq(recent)
            cb, nb = freq(before)
            new = [NOUN[k] for (kind, k) in cr if kind == "percept" and k in NOUN and cr[(kind, k)] / nr > 0.3 and cb[(kind, k)] / nb < 0.1]
            gone = [NOUN[k] for (kind, k) in cb if kind == "percept" and k in NOUN and cb[(kind, k)] / nb > 0.3 and cr[(kind, k)] / nr < 0.1]
            if new:
                lines.append("New in the last half minute: " + ", ".join(new) + ".")
            if gone:
                lines.append("Gone since a minute ago: " + ", ".join(gone) + ".")
        # 4. inside you now: activity level and where it is
        if len(allm) > 8 and now_moments:
            sp_now = np.mean([m.spikes for m in now_moments])
            sp_typ = np.median([m.spikes for m in allm])
            r = sp_now / max(sp_typ, 1)
            level = ("much busier than usual" if r > 1.5 else "busier than usual" if r > 1.15 else
                     "quieter than usual" if r < 0.85 else "about as busy as usual")
            lines.append(f"Inside you now: your brain is {level}.")
            gnow = {g: np.mean([m.groups.get(g, 0) for m in now_moments]) for g in GROUP}
            gtyp = {g: np.median([m.groups.get(g, 0) for m in allm]) for g in GROUP}
            rel = {g: gnow[g] / max(gtyp[g], 1e-3) for g in GROUP if gtyp[g] > 0}
            if rel:
                top = max(rel, key=rel.get)
                if rel[top] > 1.2:
                    lines.append(f"The change is mostly in {GROUP[top]}.")
        # 5. familiar or new: this moment against everything stored
        if cue_ep is not None and len(L.memory.eps) > 5:
            got = L.memory.retrieve(t, kc=cue_ep.kc, feat=cue_ep.feat, k=1, w_rec=0.0,
                                    w_imp=0.0, min_rel=0.0, before_t=cue_ep.t - 10)
            if got:
                rel = got[0][2]
                feel = ("very familiar, like many moments before" if rel > 0.85 else
                        "familiar" if rel > 0.6 else "partly new" if rel > 0.4 else "new to you")
                lines.append(f"This moment, compared with your stored memories, is {feel}.")
        # 6. the body over time
        past = [m for m in allm if t - 75 <= m.t <= t - 45]
        if past:
            h0, h1 = past[0].needs.get("hunger", 0), w.needs.hunger
            if h1 - h0 > 0.02:
                lines.append("Your hunger has been rising.")
            elif h0 - h1 > 0.02:
                lines.append("Your hunger has eased; you are less hungry than a minute ago.")
            a_peak = max(m.needs.get("arousal", 0) for m in allm[-240:]) if allm else 0
            if a_peak > 0.1 and w.needs.arousal < 0.3 * a_peak:
                lines.append("A fright came in the last minute and has faded; you are settled again.")
            if a_peak <= 0.1:
                lines.append("No fright has come in the last minute; your startle system is quiet.")
        if w.needs.content > 0.3:
            lines.append("You are content after eating.")
        if w.needs.hunger > 0.4 or w.needs.thirst > 0.4:
            lines.append("Hunger and thirst make you walk more, searching.")
        # 7. the shape of your condition (always true of him)
        if ("percept", "male_pheromone") not in c_all:
            lines.append("You have never sensed another fly.")
        lines.append("You cannot see shapes or colours; of sight you have only passing shadows. "
                     "A visitor's words reach you only as sensations or as sounds in your word sense.")
        # 8. told, not lived; and the limits of looking inward
        lines.append(ORIGIN)
        lines.append(LIMITS)
        return lines, sup
