"""The fly's autobiographical memory: stored brain states, not stories.

Design from research/review/08_grounded_self_architectures.md. Mainstream LLM
agent memory stores text the language model wrote, so the memory becomes the
LM's story about the agent. Here an episode is what the fly's own brain did:

    kc        which Kenyon cells fired (the mushroom body's sparse code; a
              locality-sensitive hash of the experience, Dasgupta et al. 2017)
    feat      the central brain's activity (the voice's neural-token input)
    speech    what the GROUNDED voice (FlyLM) said at that moment
    context   time, place, needs, MB valence, and what the body was doing

Retrieval is RE-EVOCATION: a cue is presented to the fly (through its senses
or its word lobe), and episodes are matched against the brain state the cue
evokes now, by KC overlap and central-brain similarity, weighted by recency
and importance (Park et al. 2023's three factors, with importance taken from
the fly's arousal, valence and body events).

Provenance: records form a hash chain (each hash covers the previous one), so
an inserted or edited memory is detectable (SPECS M5a).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np


@dataclass
class Episode:
    i: int
    t: float                    # world time, s
    kc: np.ndarray              # indices of active KCs (int32)
    feat: np.ndarray            # central-brain features (float16)
    speech: str                 # grounded voice output at the time
    place: str                  # e.g. "on the sugar drop", "near the lamp"
    needs: dict
    event: str = ""             # body event (feed, escape, groom, sleep, heard)
    valence: float = 0.0
    importance: float = 0.0
    heard: list = field(default_factory=list)   # words heard (word lobe)
    prev: str = ""
    hash: str = ""

    def digest(self):
        h = hashlib.sha256()
        h.update(self.prev.encode())
        h.update(np.asarray(self.kc, np.int32).tobytes())
        h.update(np.asarray(self.feat, np.float16).tobytes())
        h.update(json.dumps([self.i, round(self.t, 3), self.speech, self.place,
                             self.event, self.heard], sort_keys=True).encode())
        return h.hexdigest()[:16]


class EpisodicStore:
    def __init__(self, path=None):
        self.eps: list[Episode] = []
        self.path = Path(path) if path else None
        self._fn = None             # feature normaliser (mu, sd)

    # ------------------------------------------------------------ write
    def add(self, t, kc, feat, speech, place, needs, event="", valence=0.0,
            heard=()):
        imp = (0.5 * float(needs.get("arousal", 0)) + 0.5 * abs(valence)
               + (0.4 if event in ("feed", "escape", "groom", "heard") else 0.0)
               + (0.2 if speech and speech != "..." else 0.0))
        ep = Episode(len(self.eps), float(t), np.asarray(kc, np.int32),
                     np.asarray(feat, np.float16), speech, place, dict(needs),
                     event, float(valence), float(imp), list(heard),
                     prev=self.eps[-1].hash if self.eps else "genesis")
        ep.hash = ep.digest()
        self.eps.append(ep)
        return ep

    # ------------------------------------------------------------ read
    def verify(self):
        """Indices of records whose hash chain is broken (implanted/edited)."""
        bad, prev = [], "genesis"
        for e in self.eps:
            if e.prev != prev or e.digest() != e.hash:
                bad.append(e.i)
            prev = e.hash
        return bad

    def retrieve(self, now_t, kc=None, feat=None, k=3, w_kc=1.0, w_feat=1.0,
                 w_rec=0.3, w_imp=0.5, tau_rec=600.0, min_rel=0.15):
        """Score = relevance (KC Jaccard and/or central-brain cosine) +
        recency + importance; returns [(episode, score, relevance)].
        Episodes below `min_rel` relevance are never returned -- an
        irrelevant memory is not a memory of this cue."""
        if not self.eps:
            return []
        rel = np.zeros(len(self.eps))
        if kc is not None and len(kc):
            q = set(int(x) for x in kc)
            rel += w_kc * np.array([len(q & set(e.kc.tolist())) /
                                    max(len(q | set(e.kc.tolist())), 1) for e in self.eps])
        if feat is not None:
            F = np.stack([e.feat.astype(np.float32) for e in self.eps])
            f = np.asarray(feat, np.float32)
            num = F @ f
            den = np.linalg.norm(F, axis=1) * (np.linalg.norm(f) + 1e-9) + 1e-9
            rel += w_feat * np.clip(num / den, 0, 1)
        n_terms = (kc is not None and len(kc) > 0) * w_kc + (feat is not None) * w_feat
        rel = rel / max(n_terms, 1e-9)
        rec = np.array([math.exp(-(now_t - e.t) / tau_rec) for e in self.eps])
        imp = np.array([e.importance for e in self.eps])
        score = rel + w_rec * rec + w_imp * imp
        order = np.argsort(-score)
        out = [(self.eps[j], float(score[j]), float(rel[j])) for j in order
               if rel[j] >= min_rel][:k]
        return out

    def recent(self, n=5):
        return self.eps[-n:]

    # ------------------------------------------------------------ persistence
    def save(self, path=None):
        p = Path(path or self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = [{k: v for k, v in asdict(e).items() if k not in ("kc", "feat")}
                for e in self.eps]
        np.savez_compressed(p, meta=json.dumps(meta),
                            kc=np.array([e.kc for e in self.eps], dtype=object),
                            feat=np.stack([e.feat for e in self.eps]) if self.eps
                            else np.zeros((0, 1), np.float16))
        return p

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        st = cls(path)
        for m, kc, f in zip(json.loads(str(d["meta"])), d["kc"], d["feat"]):
            st.eps.append(Episode(kc=np.asarray(kc, np.int32), feat=f, **m))
        return st
