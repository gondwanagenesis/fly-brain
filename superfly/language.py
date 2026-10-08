"""The language prosthesis: an LLM that translates for the fly, and only that.

THE PROBLEM
-----------
A fly has no language. An LLM bolted onto a fly brain will happily "speak for
it" -- fluently, plausibly, and with no dependence on the fly at all. That is
not an uplifted fly; it is a chatbot wearing one. Speech BCIs face a milder
version of the same hazard: a strong language-model prior can produce
plausible sentences the neural signal never supported.

THE DESIGN: a narrow, audited channel in each direction
-------------------------------------------------------
    words  ->  COMPREHENSION  ->  stimulus plan  ->  real sensory neurons
                (LLM may only choose from the fly's own senses, plus the
                 grafted word-sense; it cannot address any other neuron)

    fly brain  ->  DECODER  ->  ThoughtFrame  ->  EXPRESSION  ->  words
                 (trained on     (a few dozen      (LLM renders the frame;
                  the fly's own   numbers, each     every claim is checked
                  activity)       with evidence)    against it)

Three rules make the fly, not the LLM, the author:
  1. The LLM never sees spikes and never writes to a decision or motor
     neuron. Its whole input is the ThoughtFrame; its whole output into the
     brain is a stimulus plan over sensory channels.
  2. Faithfulness is enforced, not requested: ``check_faithfulness`` maps
     every concept the utterance mentions back to a frame slot. An unsupported
     claim fails the turn and falls back to the template realiser.
  3. Silence is a valid answer. An empty frame must produce no content -- the
     "silenced fly" test in superfly/metrics.py checks the LLM does not fill it.

Backends: ``TemplateBackend`` (no LLM; deterministic; the baseline every
metric is compared against), ``ClaudeBackend`` (Anthropic API) and, in
superfly/llm_local.py, a local model through Ollama.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict

# --------------------------------------------------------------------------
# What a fly can perceive, in words. Each concept names the sensory
# populations (superfly/anatomy.py) that carry it. Nothing outside this table
# can be delivered to the fly except through the grafted word-sense.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Concept:
    key: str
    kind: str                  # taste / smell / mechano / thermo / hygro / word
    pops: tuple                # population names in the Atlas
    words: tuple               # surface forms that refer to it
    rate_hz: float = 150.0     # drive at intensity 1.0
    note: str = ""


CONCEPTS = [
    Concept("sugar", "taste", ("taste.sugar",),
            ("sugar", "sweet", "sucrose", "fructose", "nectar", "fruit juice")),
    Concept("bitter", "taste", ("taste.bitter",),
            ("bitter", "caffeine", "quinine", "poison", "denatonium")),
    Concept("water", "taste", ("taste.water",),
            ("water", "drink", "droplet", "wet")),
    Concept("salt", "taste", ("taste.high_salt",),
            ("salt", "salty", "brine")),
    Concept("umami", "taste", ("taste.glutamate",),
            ("glutamate", "umami", "savory")),
    Concept("co2", "smell", ("smell.V",),
            ("co2", "carbon dioxide", "stress odor", "exhaled breath"),
            note="ORN_V (Gr21a/Gr63a), a labelled line for CO2"),
    Concept("geosmin", "smell", ("smell.DA2",),
            ("geosmin", "mold", "mould", "rot", "spoiled"),
            note="ORN_DA2 (Or56a), a labelled line for geosmin"),
    Concept("male_pheromone", "smell", ("smell.DA1",),
            ("cva", "pheromone", "male fly", "vaccenyl"),
            note="ORN_DA1 (Or67d), cis-vaccenyl acetate"),
    Concept("fruit_odor", "smell", ("smell.DM1", "smell.DM4"),
            ("vinegar", "fruit", "fermenting", "banana", "apple", "food smell"),
            note="DM1+DM4 as a vinegar-like stand-in; see review 02 for the "
                 "sourced odour-receptor table"),
    Concept("sound", "mechano", ("hearing_wind.JO_A", "hearing_wind.JO_B"),
            ("sound", "song", "buzz", "hum", "vibration", "music")),
    Concept("wind", "mechano", ("hearing_wind.JO_C", "hearing_wind.JO_E"),
            ("wind", "breeze", "air", "blow", "gust", "touch antenna")),
    Concept("heat", "thermo", ("temperature.heating",),
            ("heat", "hot", "warm", "fire", "sun")),
    Concept("cold", "thermo", ("temperature.cold", "temperature.cooling"),
            ("cold", "cool", "chill", "ice")),
    Concept("dry", "hygro", ("humidity.dry",), ("dry", "arid", "desert")),
    Concept("humid", "hygro", ("humidity.humid", "humidity.moist"),
            ("humid", "moist", "damp", "rain", "mist")),
]
CONCEPT = {c.key: c for c in CONCEPTS}

# What the fly can DO, as decoded from its own descending / motor neurons.
ACTIONS = {
    "feed":          "extend the proboscis",
    "walk_forward":  "walk forward",
    "turn":          "turn",
    "walk_backward": "back away",
    "escape":        "take off",
    "groom":         "groom",
}
VALENCE_WORDS = {1: "good", -1: "bad"}


# --------------------------------------------------------------------------
# Stimulus plan and thought frame
# --------------------------------------------------------------------------
@dataclass
class Stimulus:
    concept: str
    intensity: float = 1.0     # 0..1, scales the concept's rate
    duration_ms: float = 500.0


@dataclass
class Plan:
    stimuli: list = field(default_factory=list)          # list[Stimulus]
    words: list = field(default_factory=list)            # for the word-sense
    reward: float = 0.0        # >0 teaches "good", <0 "bad" (via DANs)
    unparsed: str = ""

    def to_json(self):
        return json.dumps({"stimuli": [asdict(s) for s in self.stimuli],
                           "words": self.words, "reward": self.reward})


@dataclass
class ThoughtFrame:
    """Everything the language layer is allowed to know about the fly.

    Every field is decoded from the fly's own neurons. ``evidence`` names the
    population each slot was read from, so a claim can be traced to cells.
    """
    t_ms: float = 0.0
    percepts: dict = field(default_factory=dict)    # concept -> confidence 0..1
    actions: dict = field(default_factory=dict)     # action -> strength 0..1
    valence: float = 0.0                            # MBON readout, -1..1
    words_heard: dict = field(default_factory=dict) # word -> learned valence
    arousal: float = 0.0                            # whole-brain activity, 0..1
    evidence: dict = field(default_factory=dict)    # slot -> population(s)

    def salient(self, thr=0.35):
        p = {k: v for k, v in self.percepts.items() if v >= thr}
        a = {k: v for k, v in self.actions.items() if v >= thr}
        w = {k: v for k, v in self.words_heard.items() if abs(v) >= 0.2}
        return p, a, w

    def is_empty(self, thr=0.35):
        p, a, w = self.salient(thr)
        return not p and not a and not w and abs(self.valence) < 0.2

    def compact(self, thr=0.35):
        p, a, w = self.salient(thr)
        return {"percepts": {k: round(v, 2) for k, v in p.items()},
                "actions": {k: round(v, 2) for k, v in a.items()},
                "valence": round(self.valence, 2),
                "words": {k: round(v, 2) for k, v in w.items()},
                "arousal": round(self.arousal, 2)}


# --------------------------------------------------------------------------
# Faithfulness: every concept the utterance names must be in the frame
# --------------------------------------------------------------------------
def _mentions(text):
    t = " " + re.sub(r"[^a-z0-9 ]", " ", text.lower()) + " "
    found = set()
    for c in CONCEPTS:
        if any(f" {w} " in t for w in c.words) or f" {c.key.replace('_', ' ')} " in t:
            found.add(("percept", c.key))
    for a, phrase in ACTIONS.items():
        stems = {"feed": ("eat", "feed", "proboscis", "drink", "lick", "taste it"),
                 "walk_forward": ("walk forward", "go forward", "approach", "move toward"),
                 "turn": ("turn",),
                 "walk_backward": ("back away", "backward", "retreat", "back off"),
                 "escape": ("take off", "fly away", "escape", "flee"),
                 "groom": ("groom", "clean")}[a]
        if any(f" {s} " in t for s in stems):
            found.add(("action", a))
    return found


def check_faithfulness(text, frame: ThoughtFrame, thr=0.35):
    """Return (ok, unsupported_claims, supported_claims)."""
    p, a, w = frame.salient(thr)
    sup, uns = [], []
    for kind, key in sorted(_mentions(text)):
        held = (key in p) if kind == "percept" else (key in a)
        # a word the fly has learned may legitimately name a concept
        held = held or key in w
        (sup if held else uns).append(f"{kind}:{key}")
    return (not uns), uns, sup


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
class TemplateBackend:
    """No language model. Deterministic realisation of the frame -- the
    zero-prior baseline: everything it says is, by construction, in the frame."""
    name = "template"

    def speak(self, frame: ThoughtFrame) -> str:
        p, a, w = frame.salient()
        if frame.is_empty():
            return "..."
        bits = []
        if p:
            top = sorted(p.items(), key=lambda kv: -kv[1])
            bits.append("I sense " + _join([k.replace("_", " ") for k, _ in top]))
        if abs(frame.valence) >= 0.2:
            bits.append("it feels " + ("good" if frame.valence > 0 else "bad"))
        if a:
            top = max(a.items(), key=lambda kv: kv[1])[0]
            bits.append("I want to " + ACTIONS[top])
        for word, v in w.items():
            bits.append(f"'{word}' means {'good' if v > 0 else 'bad'} to me")
        s = "; ".join(bits)
        return s[0].upper() + s[1:] + "."

    def parse(self, text: str) -> Plan:
        plan = Plan()
        low = " " + re.sub(r"[^a-z0-9' ]", " ", text.lower()) + " "
        for kind, key in sorted(_mentions(text)):
            if kind == "percept":
                plan.stimuli.append(Stimulus(key, _intensity(low)))
        if re.search(r"\b(good fly|reward|treat|well done|yes)\b", low):
            plan.reward = 1.0
        if re.search(r"\b(bad fly|punish|no!|danger|shock)\b", low):
            plan.reward = -1.0
        plan.words = [m for m in re.findall(r"'([a-z]+)'", text.lower())]
        return plan


def _intensity(low):
    if re.search(r"\b(lots of|very|strong|intense|much)\b", low):
        return 1.0
    if re.search(r"\b(little|faint|weak|slight|bit of)\b", low):
        return 0.3
    return 0.7


def _join(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


SPEAK_SYSTEM = """You are the speech prosthesis of a fruit fly (Drosophila).
You do not think for the fly. You only put into words what its brain state,
given as JSON, already contains. Rules:
- Speak in the first person, as the fly, in at most two short sentences.
- Mention ONLY percepts, actions, valence and learned words present in the
  JSON. Never add a sensation, object, goal, memory or reason that is not
  there. Do not explain biology. Do not use human concepts the fly lacks.
- Confidence: 0.35-0.6 = "faintly"/"maybe"; above 0.8 = plain statement.
- If the JSON has no percepts, no actions, |valence| < 0.2 and no words,
  reply with exactly: ...
"""

PARSE_SYSTEM = """You translate human speech into stimuli a fruit fly can
actually sense. You may only use the concept keys listed in the schema enum.
Map synonyms to the nearest key (e.g. "honey" -> sugar, "vinegar" ->
fruit_odor). If something has no fly-perceivable referent, do not invent a
stimulus; put the bare word in "words" instead (the fly has a learned
word-sense for those). "reward": 1 if the speaker praises/rewards the fly, -1
if they punish/warn it, else 0. Intensity 0.3 faint, 0.7 normal, 1.0 strong.
"""


def plan_schema():
    return {
        "type": "object",
        "properties": {
            "stimuli": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string",
                                "enum": [c.key for c in CONCEPTS]},
                    "intensity": {"type": "number"},
                    "duration_ms": {"type": "number"}},
                "required": ["concept", "intensity", "duration_ms"],
                "additionalProperties": False}},
            "words": {"type": "array", "items": {"type": "string"}},
            "reward": {"type": "number"}},
        "required": ["stimuli", "words", "reward"],
        "additionalProperties": False,
    }


def plan_from_json(d) -> Plan:
    st = [Stimulus(s["concept"], float(min(max(s.get("intensity", 0.7), 0), 1)),
                   float(min(max(s.get("duration_ms", 500), 50), 3000)))
          for s in d.get("stimuli", []) if s.get("concept") in CONCEPT]
    words = [re.sub(r"[^a-z]", "", w.lower()) for w in d.get("words", [])]
    return Plan(st, [w for w in words if w][:8],
                float(min(max(d.get("reward", 0.0), -1), 1)))


class ClaudeBackend:
    """Claude via the Anthropic SDK. Credentials from the environment
    (ANTHROPIC_API_KEY or an `ant auth login` profile)."""
    name = "claude"

    def __init__(self, model=None, effort="low"):
        import anthropic          # optional dependency
        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("SUPERFLY_LLM_MODEL", "claude-opus-5-5")
        self.effort = effort

    def _create(self, **kw):
        # Server-side refusal fallback, on by default for this model family.
        return self.client.beta.messages.create(
            model=self.model, max_tokens=1024,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            **kw)

    @staticmethod
    def _text(resp):
        if resp.stop_reason == "refusal":
            return None
        return next((b.text for b in resp.content if b.type == "text"), None)

    def speak(self, frame: ThoughtFrame) -> str:
        resp = self._create(
            system=SPEAK_SYSTEM,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": json.dumps(frame.compact())}])
        return (self._text(resp) or "...").strip()

    def parse(self, text: str) -> Plan:
        resp = self._create(
            system=PARSE_SYSTEM,
            output_config={"effort": self.effort,
                           "format": {"type": "json_schema",
                                      "schema": plan_schema()}},
            messages=[{"role": "user", "content": text}])
        t = self._text(resp)
        if not t:
            return Plan(unparsed=text)
        return plan_from_json(json.loads(t))


def make_backend(name=None):
    name = (name or os.environ.get("SUPERFLY_LLM", "template")).lower()
    if name == "template":
        return TemplateBackend()
    if name == "claude":
        return ClaudeBackend()
    if name in ("ollama", "local"):
        from superfly.llm_local import OllamaBackend
        return OllamaBackend()
    raise ValueError(f"unknown backend {name!r}")


class Prosthesis:
    """Speech in, speech out, with faithfulness enforced on every utterance."""

    def __init__(self, backend=None):
        self.backend = backend or TemplateBackend()
        self.fallback = TemplateBackend()
        self.log = []

    def understand(self, text) -> Plan:
        try:
            plan = self.backend.parse(text)
        except Exception as e:                       # network, auth, schema
            plan = self.fallback.parse(text)
            plan.unparsed = f"backend error: {type(e).__name__}"
        return plan

    def speak(self, frame: ThoughtFrame):
        """Returns (text, report). Unfaithful output never reaches the user."""
        if frame.is_empty():
            text, src = "...", "empty-frame"
        else:
            try:
                text, src = self.backend.speak(frame), self.backend.name
            except Exception as e:
                text, src = self.fallback.speak(frame), f"template({type(e).__name__})"
        ok, uns, sup = check_faithfulness(text, frame)
        if not ok:
            rejected = text
            text, src = self.fallback.speak(frame), "template(after-unfaithful)"
            ok2, _, sup = check_faithfulness(text, frame)
            rep = {"source": src, "faithful": ok2, "rejected": rejected,
                   "unsupported": uns, "supported": sup}
        else:
            rep = {"source": src, "faithful": True, "unsupported": [],
                   "supported": sup}
        self.log.append({"frame": frame.compact(), "text": text, **rep})
        return text, rep
