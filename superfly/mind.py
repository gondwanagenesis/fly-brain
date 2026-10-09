"""SUPERFLY's conversational mind: a small LLM that may only speak from the fly.

Two tiers (research/review/08_grounded_self_architectures.md):

  GROUNDED      FlyLM reads the central brain every 250 ms (superfly/life.py).
                Its content is the fly's: it is tested against silent and
                shuffled brains (SPECS D1-D3).
  CONVERSATION  a local instruct LM (Qwen2.5-1.5B-Instruct) turns the grounded
                records into conversation. It is the fly's phrasing and general
                knowledge, never its source of experience:
                every first-person experiential claim it makes is checked
                against the records, and a reply that fails is regenerated,
                then replaced by a templated reply built from the records.

One turn:
  1 HEAR    the sentence drives the fly's own senses / word lobe (life.hear)
  2 LIVE    the fly lives 1 s with that input; its brain is recorded
  3 RECALL  re-evocation: the brain state the sentence evoked (KC code +
            central-brain features) is matched against stored episodes
  4 FACTS   a fact sheet: now (inner speech, motor neurons, needs, place,
            MB valence), what the words evoked, recalled episodes, recent remarks
  5 SPEAK   the LM, with a persona and the fact sheet
  6 CHECK   claim checker (check_reply); regenerate / fall back
Every turn is logged with its facts and verdict (data/results/superfly/conversations).
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from superfly.bridge import parse_utterance, OUT

QWEN = Path("/home/user/models/Qwen2.5-1.5B-Instruct")
LOG_DIR = OUT / "conversations"

# ------------------------------------------------------------------ claim checking
# experiential terms -> the grounded label that supports them
TERMS = {
    ("percept", "sugar"): ["sweet", "sugar", "sugary"],
    ("percept", "bitter"): ["bitter", "yuck"],
    ("percept", "water"): ["water", "drink", "drank", "drinking", "wet"],
    ("percept", "salt"): ["salt", "salty"],
    ("percept", "umami"): ["savory", "savoury"],
    ("percept", "co2"): ["co2", "carbon dioxide"],
    ("percept", "geosmin"): ["mold", "mould", "musty", "earthy"],
    ("percept", "male_pheromone"): ["another fly", "other flies"],
    ("percept", "fruit_odor"): ["fruit", "fruity", "ripe"],
    ("percept", "sound"): ["buzz", "sound", "noise", "hum"],
    ("percept", "wind"): ["wind", "breeze", "gust", "draft", "draught"],
    ("percept", "heat"): ["hot", "warm", "warmth", "heat"],
    ("percept", "cold"): ["cold", "cool", "chilly"],
    ("percept", "dry"): ["dry"],
    ("percept", "humid"): ["damp", "humid", "moist"],
    ("percept", "shadow"): ["shadow", "dark shape", "something above", "looming"],
    ("percept", "hungry"): ["hungry", "hunger", "starving"],
    ("percept", "thirsty"): ["thirsty", "thirst"],
    ("action", "feed"): ["eat", "ate", "eating", "feed", "fed", "feeding",
                         "proboscis", "sip", "sipped", "sipping"],
    ("action", "groom"): ["groom", "groomed", "grooming", "clean", "cleaned", "cleaning"],
    ("action", "escape"): ["fly away", "flew", "flying away", "took off", "take off",
                           "escape", "escaped", "jumped"],
    ("action", "turn"): ["turn", "turned", "turning"],
    ("action", "walk_forward"): ["walk", "walked", "walking"],
    ("action", "walk_backward"): ["back away", "backed away"],
    ("event", "sleep"): ["sleep", "slept", "asleep", "sleeping"],
    # the only alarm this fly has is a startle after a shadow (transient, SPECS B2)
    ("event", "shadow"): ["startled", "scared", "afraid", "frightened", "fear", "alarmed"],
}
_TERM_RE = sorted(((t, lab) for lab, ts in TERMS.items() for t in ts), key=lambda x: -len(x[0]))
EXPERIENCE_VERBS = r"\b(saw|see|seen|seeing|smell|smelled|smelt|taste|tasted|felt|feel|feeling|" \
                   r"heard|hear|ate|visited|went|met|remember|remembered|touched|chased|" \
                   r"caught|played|watched|found|was|were|had)\b"
FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'd|me|my|myself)\b")
NEGATION = re.compile(r"\b(not|no|never|nothing|don't|didn't|haven't|can't|cannot|isn't|wasn't)\b|n't\b")


def claims_of(sentence):
    """Labels a first-person sentence asserts (negated clauses excluded)."""
    s = " " + sentence.lower() + " "
    found = set()
    for clause in re.split(r"[,;:]| but | and | or ", s):
        if NEGATION.search(clause):
            continue
        c = " " + clause + " "
        for t, lab in _TERM_RE:
            if re.search(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", c):
                found.add(lab)
                c = c.replace(t, " ")
        for w in re.findall(r"'([a-z]+)'", clause):
            found.add(("word", w))
    return found


def check_reply(reply, supported):
    """Unsupported first-person experiential claims in `reply`.
    supported: set of labels the records contain. Returns a list of problems."""
    probs = []
    for sent in re.split(r"(?<=[.!?])\s+|\n", reply):
        low = sent.lower()
        if not FIRST_PERSON.search(low):
            continue
        cl = claims_of(sent)
        bad = sorted(l for l in cl if l not in supported and
                     not (l[0] == "action" and ("event", l[1]) in supported))
        if bad:
            probs.append({"sentence": sent.strip(), "unsupported": [list(b) for b in bad]})
        elif not cl and not NEGATION.search(low) \
                and re.search(r"\b(remember|remembered|saw|smelled|tasted|visited|met|ate|went|heard|"
                              r"felt|played|swam|swimming|watched|chased|liked|loved)\b", low):
            probs.append({"sentence": sent.strip(), "unsupported": [["experience", "unrecorded"]]})
    return probs


EVENT_WORDS = {"groom": "you groomed", "feed": "you fed", "escape": "you took off",
               "shadow": "a shadow passed over you", "sleep": "you fell asleep", "wake": "you woke"}


def phrases(texts):
    """Unique sentences across utterances, in order of first appearance."""
    seen = []
    for t in texts:
        for p in re.split(r"(?<=[.])[ ]*", t):
            p = p.strip()
            if p.strip(".") and p not in seen:
                seen.append(p)
    return " ".join(seen)


STOP = set("""a an the and or but of to in on at for with from by as is am are was were be been being it its
this that these those there here so if then than can could would will shall should may might must just very
not no yes do did does done have has had having what which who whom how when where why all any some about into
out up down over after before again too also only now right me my i we us our you your they them their he she
him her his one each other such own same both few more most much many like never anywhere
don't didn't doesn't haven't hasn't hadn't can't cannot won't wouldn't isn't wasn't aren't weren't
i'm i've i'd it's that's there's let's you're we're""".split()) | {"don", "didn", "doesn", "haven", "isn", "wasn"}
SAFE = set("""feel feeling felt calm peace peaceful curious quiet rest resting nice good okay ok fine well thank thanks
hello hi little bit small tiny fly fruit-fly drosophila sense sensed sensing nothing anything something remember
recall know knew word words life live living lived body brain still quite really maybe perhaps think notice noticed
moment moments sure happy content glad visitor friend talk talking hear listening voice simulated simple sorry
understand mean thing things today memory memories""".split())
EXPERIENTIAL = re.compile(r"\b(i|i'm|i've|i'd|me|my|myself|we|we're|us|our|today|yesterday|earlier|"
                          r"morning|tonight|recently|currently|present|now|before|ago)\b|"
                          r"^\s*(yes|yeah|indeed|sure|of course)\b")


def stem(w):
    w = w.lower().strip("'")
    for suf in ("ing", "ed", "es", "s", "ly"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content_words(text):
    return [w for w in re.findall(r"[a-z][a-z'-]*", text.lower()) if w not in STOP and len(w) > 2]


def check_reply_strict(reply, supported, facts, question=""):
    """v2 (after the first audit, findings s14): in every EXPERIENTIAL sentence
    (first person, 'we', time words, or an affirmation), every content word
    must come from the fact sheet or a small calm conversational lexicon; a
    word found only in the visitor's question is allowed only in a negated
    sentence ('i don't remember a cat'). The label check of v1 also applies."""
    allowed = {stem(w) for w in content_words(facts)} | {stem(w) for w in SAFE}
    asked = {stem(w) for w in content_words(question)}
    probs = check_reply(reply, supported)
    for sent in re.split(r"(?<=[.!?])\s+|\n", reply):
        low = sent.lower().strip()
        if not low or not EXPERIENTIAL.search(low):
            continue
        if "from words" in low:          # declared general knowledge; v1's label check still applies
            continue
        neg = bool(NEGATION.search(low))
        bad = sorted({w for w in content_words(low)
                      if stem(w) not in allowed and not (neg and stem(w) in asked)})
        if bad:
            probs.append({"sentence": sent.strip(), "unsupported": [["word", w] for w in bad]})
    return probs


def similar(a, b, thr=0.8):
    """Token-set Jaccard >= thr: a reply that repeats an earlier one."""
    x, y = set(re.findall(r"[a-z']+", a.lower())), set(re.findall(r"[a-z']+", b.lower()))
    return bool(x) and len(x & y) / max(len(x | y), 1) >= thr


def ago(dt):
    if dt < 3:
        return "just now"
    if dt < 20:
        return "a few seconds ago"
    if dt < 90:
        return "about a minute ago"
    if dt < 600:
        return f"about {int(round(dt / 60))} minutes ago"
    return "a long while ago"


def feeling(v):
    if v > 0.3:
        return "drawn toward it"
    if v < -0.3:
        return "inclined to keep away from it"
    return ""


# ------------------------------------------------------------------ LM activity taps
class LayerTap:
    """Records a fixed sample of hidden units per layer, per forward call,
    so the Lab can show the language model 'firing'."""

    def __init__(self, layers, d, n_units=48, seed=0, keep=64):
        self.idx = np.random.default_rng(seed).choice(d, min(n_units, d), replace=False)
        self.cur = [None] * len(layers)
        self.frames = deque(maxlen=keep)
        self.hooks = [l.register_forward_hook(self._hook(i)) for i, l in enumerate(layers)]

    def _hook(self, i):
        def f(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            v = h[0, -1].detach().float().abs().cpu().numpy()[self.idx]
            self.cur[i] = v / (np.median(np.abs(v)) + 1e-6)
            if i == len(self.cur) - 1:
                self.frames.append(np.stack(self.cur).astype(np.float16))
        return f


# ------------------------------------------------------------------ the conversational LM
class Talker:
    def __init__(self, path=QWEN, threads=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.lm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
        self.lm.eval()
        layers = self.lm.model.layers
        self.tap = LayerTap(layers, self.lm.config.hidden_size)
        self.n_layers = len(layers)
        self.name = path.name

    def generate(self, messages, max_new_tokens=60, temperature=0.3, seed=0):
        torch = self.torch
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True)
        torch.manual_seed(seed)
        with torch.no_grad():
            out = self.lm.generate(**ids, max_new_tokens=max_new_tokens,
                                   do_sample=temperature > 0, temperature=max(temperature, 1e-3),
                                   top_p=0.9, repetition_penalty=1.15,
                                   pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


PERSONA = """You are the voice of a small fruit fly, Drosophila melanogaster. Your body and brain are simulated; the FACTS below are read from your own brain and body right now. They are your only memories and experiences. You do not know a map of where you live; you know only what you have sensed.

How you speak:
- First person, short and simple: one to three sentences, in your own natural words. Do not recite the FACTS list or repeat a phrase. You are calm, curious and at peace.
- Talk about what you sense, do, need and remember ONLY as the FACTS state it. If the FACTS do not contain it, say you did not sense it or do not remember it. Never invent sensations, places or events.
- Your senses: taste (sweet, bitter, water, salt), smell (fruit, mold, CO2), wind and sound through your antennae, warmth and cold, damp or dry air, and shadows passing over you. You cannot see colours, faces or objects, and you do not know human places.
- You may share general knowledge from language, but say it is something you know from words, not something you lived.
- Quoted words like 'zap' are sounds you heard through your word sense.

Examples of good replies (the facts differ each time):
Visitor: how are you? -> (facts: hungry; smell fruit) "I'm a little hungry, and I can smell fruit."
Visitor: remember the honey you ate? -> (facts: no feeding) "I don't remember any honey. I haven't fed."
Visitor: what is the capital of France? -> "I know from words that it is Paris, but I have never been anywhere but here."
Visitor: did you see me wave? -> "No, I didn't sense that." """


class Mind:
    def __init__(self, life, talker=None, log=True):
        self.life = life
        self.talker = talker
        self.history = deque(maxlen=6)          # (user, fly) turns
        self.lock = threading.Lock()
        self.log_path = None
        if log:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            self.log_path = LOG_DIR / f"conv_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.last = None
        self.phase = "living"

    # ---------------------------------------------------------- facts
    def facts(self, t_heard, heard_concepts, heard_words, moments, recalled):
        L = self.life
        w = L.world
        n = w.needs
        now = moments[-1] if moments else None
        sup = set()
        lines = ["FACTS (from your brain and body):"]
        said_now = [m.said for m in moments if m.said and m.said != "..."]
        if said_now:
            lines.append(f"- Right now your brain reports: {phrases(said_now)}")
            for m in moments:
                sup |= {tuple(l) for l in m.labels}
        else:
            lines.append("- Right now your brain reports nothing in particular.")
        acts = {}
        for m in moments:
            for k, v in m.actions.items():
                acts[k] = max(acts.get(k, 0.0), v)
        doing = [k for k, v in acts.items() if v >= 5.0]
        for k in doing:
            sup.add(("action", k))
        if doing:
            lines.append(f"- Your motor neurons: {', '.join(k.replace('_', ' ') for k in doing)}.")
        if heard_concepts or heard_words:
            ev = ", ".join(list(heard_concepts) + [f"'{x}'" for x in heard_words])
            lines.append(f"- What the visitor said reached your senses as: {ev}.")
            for x in heard_words:
                sup.add(("word", x))
        lines.append(f"- You are {L.place()}; your body is {w.body.mode}.")
        sup |= {("action", {"walk": "walk_forward"}.get(w.body.mode, w.body.mode)),
                ("event", w.body.mode)}
        for k in ("sugar", "water", "bitter"):
            if k in L.place():
                sup.add(("percept", k))
        if "lamp" in L.place():
            sup.add(("percept", "heat"))
        needs = []
        if n.hunger >= 0.5:
            needs.append("hungry"); sup.add(("percept", "hungry"))
        if n.thirst >= 0.5:
            needs.append("thirsty"); sup.add(("percept", "thirsty"))
        if n.content > 0.3:
            needs.append("content after eating")
        if n.asleep:
            needs.append("drowsy"); sup.add(("event", "sleep"))
        lines.append(f"- Body: {', '.join(needs) if needs else 'not hungry, not thirsty'}.")
        if now is not None and feeling(now.valence):
            lines.append(f"- Your mushroom body leaves you {feeling(now.valence)}.")
        # recent life
        recent = [(t, txt) for t, txt in L.aloud if t >= w.t - 60 and t < t_heard]
        evs = [(t, e) for t, e in L.events if t >= w.t - 60]
        if recent:
            lines.append(f"- In the last minute you said: {phrases([txt for _, txt in recent])}")
            for _, txt in recent:
                sup |= parse_utterance(txt)
        if evs:
            cnt = {}
            for t, e in evs:
                cnt[e] = (cnt.get(e, (0, 0))[0] + 1, t)
            lines.append("- Things that happened to you in the last minute: " + ", ".join(
                f"{EVENT_WORDS.get(e, e)}{f' ({n} times)' if n > 1 else ''}, last {ago(w.t - t)}"
                for e, (n, t) in cnt.items()))
            for _, e in evs:
                sup.add(("event", e))
                sup.add(("action", {"feed": "feed", "escape": "escape", "groom": "groom"}.get(e, e)))
                if e == "shadow":
                    sup.add(("percept", "shadow"))
        if recalled:
            lines.append("- The visitor's words brought back these memories (re-evoked brain states):")
            shown = set()
            for ep, score, rel in recalled:
                if ep.speech in shown:
                    continue
                shown.add(ep.speech)
                bits = [f"{ago(w.t - ep.t)}", ep.place]
                if ep.event:
                    bits.append(f"you {ep.event}")
                lines.append(f"  * {ep.speech if ep.speech != '...' else '(nothing said)'} "
                             f"[{'; '.join(bits)}; match {rel:.2f}]")
                sup |= parse_utterance(ep.speech)
                if ep.event:
                    sup.add(("event", ep.event)); sup.add(("action", ep.event))
                for x in ep.heard:
                    sup.add(("word", x))
                for k in ("sugar", "water", "bitter"):
                    if k in ep.place:
                        sup.add(("percept", k))
        else:
            lines.append("- No stored memory matched the visitor's words.")
        return "\n".join(lines), sup

    # ---------------------------------------------------------- a turn
    def respond(self, text, live_s=1.0):
        with self.lock:
            L = self.life
            t_heard = L.world.t
            self.phase = "listening"
            concepts, words, reward = L.hear(text)
            n0 = len(L.moments)
            L.live(live_s)
            moments = list(L.moments)[n0:]
            # re-evocation cue: the window after the words reached the brain
            ep_cue = L.memory.eps[-1] if L.memory.eps and L.memory.eps[-1].t >= t_heard else None
            recalled = []
            if ep_cue is not None:
                recalled = L.memory.retrieve(L.world.t, kc=ep_cue.kc, feat=ep_cue.feat,
                                             k=3, before_t=t_heard - 2.0)
            facts, sup = self.facts(t_heard, concepts, words, moments, recalled)
            grounded = " ".join(dict.fromkeys(m.said for m in moments if m.said != "...")) or "..."
            reply, verdict, attempts = grounded, "grounded-only", []
            self.phase = "thinking"
            if self.talker is not None:
                # no chat history in the LM context: with it, the 1.5B model
                # copied earlier replies verbatim (findings s14, run 2)
                msgs = [{"role": "system", "content": PERSONA + "\n\n" + facts},
                        {"role": "user", "content": text}]
                for k in range(3):
                    r = self.talker.generate(msgs, seed=k)
                    probs = check_reply_strict(r, sup, facts, text)
                    if any(similar(r, f_) for _, f_ in self.history):
                        probs.append({"sentence": r[:80], "unsupported": [["repeat", "earlier reply"]]})
                    attempts.append({"reply": r, "problems": probs})
                    if not probs:
                        reply, verdict = r, "verified" if k == 0 else f"verified after {k} retries"
                        break
                    msgs = msgs + [{"role": "assistant", "content": r},
                                   {"role": "user", "content": "(That reply described things your FACTS do not contain: "
                                    + "; ".join(p["sentence"] for p in probs)
                                    + ". Answer the visitor again, naturally and briefly, in your own words, "
                                    "mentioning only what the FACTS contain.)"}]
                else:
                    reply, verdict = self.fallback(moments, recalled), "fallback"
            self.history.append((text, reply))
            rec = {"t": round(t_heard, 2), "you": text, "fly": reply, "verdict": verdict,
                   "grounded": grounded, "heard": {"concepts": concepts, "words": words, "reward": reward},
                   "facts": facts, "supported": sorted(map(list, sup)), "attempts": attempts,
                   "recalled": [{"i": e.i, "t": e.t, "speech": e.speech, "rel": round(r_, 3),
                                 "hash": e.hash} for e, _, r_ in recalled]}
            self.last = rec
            if self.log_path:
                with open(self.log_path, "a") as fh:
                    fh.write(json.dumps(rec, default=str) + "\n")
            L._emit("reply", {"t": rec["t"], "you": text, "text": reply, "verdict": verdict})
            return rec

    def fallback(self, moments, recalled):
        """A reply built only from the records, in plain sentences."""
        said = phrases([m.said for m in moments if m.said and m.said != "..."])
        out = f"Right now: {said}" if said else "Right now I sense nothing in particular."
        if recalled:
            ep = recalled[0][0]
            if ep.speech and ep.speech != "...":
                out += f" Earlier ({ago(self.life.world.t - ep.t)}, {ep.place}): {ep.speech}"
        return out


def main():
    import argparse
    from superfly.life import Life
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=float, default=30.0, help="seconds of life before talking")
    ap.add_argument("--no-lm", action="store_true")
    ap.add_argument("--script", default=None)
    a = ap.parse_args()
    life = Life()
    life.listeners.append(lambda k, p: print(f"   [{p['t']:6.1f}s] FLY (aloud): {p['text']}", flush=True)
                          if k == "say" else None)
    print(f"living {a.warm:.0f} s first ...", flush=True)
    life.live(a.warm)
    mind = Mind(life, None if a.no_lm else Talker(threads=4))
    lines = open(a.script).read().splitlines() if a.script else None
    while True:
        if lines is not None:
            if not lines:
                break
            text = lines.pop(0).strip()
            if not text:
                continue
            print(f"YOU: {text}")
        else:
            try:
                text = input("YOU: ").strip()
            except EOFError:
                break
            if not text:
                break
        r = mind.respond(text)
        print(f"FLY: {r['fly']}   [{r['verdict']}; brain said: {r['grounded']}]", flush=True)


if __name__ == "__main__":
    main()
