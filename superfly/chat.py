"""Talk to SUPERFLY.

    python -m superfly.chat                 # FlyLM voice (from scratch)
    python -m superfly.chat --voice smollm  # frozen SmolLM2 + neural tokens
    python -m superfly.chat --once "here is some sugar"

Every turn is one real experience for the fly:

  1. HEARING   your sentence -> the voice model's hearing heads -> which of
               the fly's senses to drive, how hard, and whether you praised or
               scolded it. Quoted words ('zap') go to the grafted word lobe.
  2. THE FLY   a fresh episode of the 138,639-neuron brain (learned weights
               kept): its real sensory neurons are driven, praise/scolding
               activates its reward/punishment dopamine neurons, and its own
               circuits do the rest.
  3. SPEECH    the central brain's activity -> neural tokens -> the voice.
               Shown with the evidence: what the fly's own motor neurons did,
               and how active each brain region was.

The voice never sees your sentence when it speaks -- only the fly's brain.
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import torch

from superfly.bridge import CACHE, parse_utterance
from superfly.flylm import FlyLM, FlyLMConfig, Vocab
from superfly.fly import SuperFly
from superfly.language import CONCEPTS

GROUPS = [("senses", ["taste", "smell", "vision", "hearing_wind", "touch",
                      "temperature", "humidity"]),
          ("early", ["antennal_lobe", "optic_lobe", "visual_proj"]),
          ("thinking", ["mushroom_body", "lateral_horn", "central_complex",
                        "protocerebrum"]),
          ("action", ["descending", "motor"]),
          ("grafted", ["grafted"])]


class TinyVoice:
    def __init__(self, path=CACHE / "flylm_tiny.pt"):
        ck = torch.load(path, weights_only=False)
        self.cfg = FlyLMConfig(**ck["cfg"])
        self.m = FlyLM(self.cfg)
        self.m.load_state_dict(ck["state"])
        self.m.eval()
        self.vocab = Vocab.from_state(ck["vocab"])
        self.mu, self.sd, self.keep = ck["mu"], ck["sd"], ck["keep"]
        self.keys = [c.key for c in CONCEPTS]

    def norm(self, x):
        return (((np.log1p(x) - self.mu) / self.sd)[self.keep]).astype(np.float32)

    def speak(self, feats):
        return self.m.speak(torch.as_tensor(self.norm(feats)), self.vocab)[0]

    @torch.no_grad()
    def hear(self, text):
        ids = torch.as_tensor([self.vocab.encode(text, bos=False, eos=False)])
        c, _, r = self.m.hear(ids)
        from superfly.language import _intensity
        inten = _intensity(" " + text.lower() + " ")
        concepts = {k: inten for k, v in zip(self.keys, c[0]) if v > 0.5}
        words = re.findall(r"'([a-z]+)'", text.lower())
        return concepts, words, float(r[0])


def bar(x, scale=40.0, width=20):
    n = int(min(width, round(width * np.log1p(x) / np.log1p(scale))))
    return "#" * n + "." * (width - n)


def turn(fly, voice, text, ms=250.0, verbose=True):
    concepts, words, reward = voice.hear(text)
    fly.reset()
    fly.sense(concepts, words)
    fly.run(50.0)
    if abs(reward) > 0.5:
        fly.teach(np.sign(reward))
    fly.mark()
    fly.run(ms)
    fly.teach(0)
    obs = fly.observe()
    said = voice.speak(fly.voice_features(obs))
    turn.last = dict(said=said, concepts=concepts, words=words, reward=reward,
                     actions={k: round(v, 1) for k, v in obs.actions.items()},
                     regions={k: round(v, 2) for k, v in obs.regions.items()},
                     spikes=obs.spikes)
    if verbose:
        heard = ", ".join(f"{k} {v:.1f}" for k, v in concepts.items()) or "nothing it can sense"
        print(f"  [heard -> senses: {heard}"
              + (f"; words {words}" if words else "")
              + (f"; {'praise' if reward > 0 else 'scolding'} -> dopamine" if abs(reward) > 0.5 else "")
              + "]")
        acts = ", ".join(f"{a} {v:.0f} Hz" for a, v in obs.actions.items() if v >= 1) or "none"
        print(f"  [motor neurons: {acts}; {obs.spikes} spikes]")
        for g, keys in GROUPS:
            vals = [obs.regions.get(k, 0.0) for k in keys]
            print(f"  [{g:8s} {bar(max(vals))} {max(vals):6.1f} Hz]")
        print(f"FLY: {said}")
    return said, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="tiny", choices=["tiny"])
    ap.add_argument("--once", default=None)
    ap.add_argument("--script", default=None, help="file with one line per turn")
    ap.add_argument("--learn", action="store_true",
                    help="dopamine-gated learning on (a slightly different "
                         "network from the one the bundled voice was trained on)")
    a = ap.parse_args()
    voice = TinyVoice()
    fly = SuperFly(plasticity=a.learn)
    if a.once:
        turn(fly, voice, a.once)
        return
    if a.script:
        for line in open(a.script):
            if line.strip():
                print(f"YOU: {line.strip()}")
                turn(fly, voice, line.strip())
                print()
        return
    print("\nSUPERFLY. Talk to the fly (empty line to quit). Try: 'here is some "
          "sugar', 'a strong breeze', 'say 'zap', bad fly'.\n")
    while True:
        try:
            text = input("YOU: ").strip()
        except EOFError:
            break
        if not text:
            break
        turn(fly, voice, text)


if __name__ == "__main__":
    main()
