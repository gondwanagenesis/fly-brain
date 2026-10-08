"""Record the fly's own brain under random experiences, for the language bridge.

Each episode presents a random mixture of fly-perceivable stimuli (tastes,
touch, smells, temperature, humidity; superfly/language.py CONCEPTS) and/or a
heard word (the grafted word lobe), at random intensities, or nothing at all.
What is recorded is what the language model will be allowed to see:

    features   rates of the CENTRAL brain only -- every cell type outside
               super_class 'sensory', plus each Kenyon cell individually
               (a word's identity is in WHICH KCs fire, which a cell-type
               average would erase). Receptor neurons are excluded: a
               percept has to reach the brain to be spoken about.
    labels     the stimulus (what a perception decoder is trained to report,
               as in any neural-decoding study), and the fly's OWN actions,
               read from its descending / motor neurons -- not from the
               stimulus.

Nothing about the language model enters here; this is a dataset about the fly.

    python -m superfly.experiments.make_corpus [episodes] [out.npz]
"""
from __future__ import annotations

import sys
import time

import numpy as np

from superfly.fly import SuperFly
from superfly.language import CONCEPTS, ACTIONS
from superfly.engine import ROOT

OUT = ROOT / "data" / "superfly_cache"
VOCAB = ["zap", "yum", "blip", "moo", "dax", "wug", "fep", "kiki", "bouba",
         "toma", "lupo", "rin", "sol", "nef", "vax", "pim"]
ACT_HZ = {"feed": 5.0, "groom": 5.0, "walk_forward": 5.0, "turn": 5.0,
          "walk_backward": 5.0, "escape": 5.0}


def episode_plan(rng):
    keys = [c.key for c in CONCEPTS]
    r = rng.random()
    concepts, words = {}, []
    if r < 0.12:
        pass                                            # silence
    else:
        n = rng.choice([0, 1, 1, 1, 2], p=None)
        for k in rng.choice(keys, n, replace=False):
            concepts[str(k)] = float(rng.choice([0.4, 0.7, 1.0]))
        if rng.random() < 0.35 or (n == 0):
            words = [str(rng.choice(VOCAB))]
    return concepts, words


def main(n_ep=600, out=None, seed=0, ms=250.0, settle=50.0):
    rng = np.random.default_rng(seed)
    # The final SUPERFLY network: dopamine acts through plasticity (no fast
    # DAN->KC/MBON excitation), with learning switched OFF while recording, so
    # the voice is trained on exactly the brain `chat --learn` runs.
    fly = SuperFly(plasticity=True, seed=seed)
    fly.mb.enabled = False
    e = fly.e
    kc = fly.atlas["mb.KC"].idx(e)
    sens = (fly.atlas.ann.super_class == "sensory").to_numpy(bool)
    keep_t = np.array([t for t in fly.feat_names if not t.startswith("KC")])
    tix = {t: i for i, t in enumerate(fly.feat_names)}
    keep_ix = np.array([tix[t] for t in keep_t])
    X, C, W, A, meta = [], [], [], [], []
    keys = [c.key for c in CONCEPTS]
    t0 = time.perf_counter()
    for i in range(n_ep):
        concepts, words = episode_plan(rng)
        fly.reset()
        fly.e.rng = np.random.default_rng(seed * 100003 + i)
        obs = fly.present(concepts, words, ms=ms + settle, settle_ms=settle)
        X.append(np.concatenate([obs.features[keep_ix],
                                 fly._last_window[kc]]).astype(np.float16))
        C.append([concepts.get(k, 0.0) for k in keys])
        W.append(words[0] if words else "")
        A.append([obs.actions[a] for a in ACTIONS])
        meta.append(obs.spikes)
        if (i + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print(f"  {i + 1}/{n_ep} episodes  {el:.0f}s  (~{el / (i + 1) * (n_ep - i - 1):.0f}s left)",
                  flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = out or OUT / "corpus.npz"
    np.savez_compressed(out, X=np.asarray(X), C=np.asarray(C, np.float32),
                        W=np.asarray(W), A=np.asarray(A, np.float32),
                        spikes=np.asarray(meta), concept_keys=np.asarray(keys),
                        action_keys=np.asarray(list(ACTIONS)),
                        feat_names=np.concatenate([keep_t, [f"KC#{j}" for j in range(kc.size)]]),
                        model=e.model, gain=e.gain)
    print(f"saved {out}: {len(X)} episodes x {len(X[0])} features")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 600, a[1] if len(a) > 1 else None)
