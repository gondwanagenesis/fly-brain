"""Memory specs (SPECS.md M2, M3, M5a, M11) on the fly's episodic store.

Episodes here are made the way life.py makes them: a real experience is
presented to a fresh brain state, and what is stored is the brain's response
(the Kenyon cells that fired and the central-brain features) -- not text.

    M2   recall by re-evocation: after N intervening episodes, a PARTIAL cue
         (the same experience at half strength, on one side, from a fresh
         brain) retrieves an episode of that experience; compared with the
         chance of drawing one at random
    M3   deleting the episodes of an experience removes its recall
    M5a  an implanted or edited episode is flagged by the hash chain
    M11  save -> load gives the same retrievals

    python -m superfly.tests.test_memory_specs
"""
from __future__ import annotations

import json
import time

import numpy as np

from superfly.bridge import OUT, CACHE
from superfly.fly import SuperFly
from superfly.memory import EpisodicStore, Episode

EXPERIENCES = {                      # name -> (concepts at full strength, words)
    "sugar": ({"sugar": 1.0}, []), "bitter": ({"bitter": 1.0}, []),
    "water": ({"water": 1.0}, []), "fruit": ({"fruit_odor": 0.15}, []),
    "mold": ({"geosmin": 0.15}, []), "wind": ({"wind": 1.0}, []),
    "heat": ({"heat": 1.0}, []), "shadow": ({"shadow": 1.0}, []),
    "zap": ({}, ["zap"]), "kiki": ({}, ["kiki"]), "dax": ({}, ["dax"]),
}


def evoke(fly, concepts, words, seed, ms=250.0):
    fly.reset()
    fly.e.rng = np.random.default_rng(seed)
    obs = fly.present(concepts, words, ms=ms + 50.0, settle_ms=50.0)
    kc = np.flatnonzero(fly._last_window[fly._kc] > 0)
    return kc, np.log1p(fly.voice_features(obs))


def main(n_rep=3, intervening=(0, 20, 100)):
    fly = SuperFly(plasticity=True, seed=0)
    fly.mb.enabled = False
    fly._kc = fly.atlas["mb.KC"].idx(fly.e)
    rng = np.random.default_rng(0)
    names = list(EXPERIENCES)
    store = EpisodicStore()
    t, seed = 0.0, 0
    # ---- encode: each experience n_rep times, then fillers between tests
    for r in range(n_rep):
        for nm in names:
            c, w = EXPERIENCES[nm]
            kc, f = evoke(fly, c, w, seed); seed += 1
            store.add(t, kc, f, nm, "arena", {"arousal": 0.0}, event=nm)
            t += 5.0
    n_target = len(store.eps)
    res = {"n_episodes_encoded": n_target}
    t0 = time.perf_counter()
    m2 = {}
    for N in intervening:
        while len(store.eps) < n_target + N:              # intervening filler episodes
            nm = str(rng.choice(names))
            c, w = EXPERIENCES[nm]
            c = {k: v * float(rng.uniform(0.6, 1.0)) for k, v in c.items()}
            kc, f = evoke(fly, c, w, seed); seed += 1
            store.add(t, kc, f, nm, "arena", {"arousal": 0.0}, event=nm)
            t += 5.0
        hits, top1, null = 0, 0, 0.0
        for nm in names:
            c, w = EXPERIENCES[nm]
            cue_c = {k: v * 0.5 for k, v in c.items()}      # partial cue
            kc, f = evoke(fly, cue_c, w, 10_000 + seed); seed += 1
            got = store.retrieve(t, kc=kc, feat=f, k=3, w_rec=0.0, w_imp=0.0, min_rel=0.0)
            labs = [ep.speech for ep, _, _ in got]
            hits += nm in labs
            top1 += bool(labs) and labs[0] == nm
            null += 1 - (1 - np.mean([e.speech == nm for e in store.eps])) ** 3
        m2[N] = {"recall_at3": hits / len(names), "top1": top1 / len(names),
                 "null_at3": round(null / len(names), 3), "store": len(store.eps)}
        print("M2", N, m2[N], flush=True)
    res["M2"] = m2
    res["M2_pass"] = all(v["recall_at3"] >= 0.8 and v["recall_at3"] > 2 * v["null_at3"] for v in m2.values())
    # ---- M3: delete every episode of one experience, cue it again
    m3 = {}
    for nm in ("sugar", "zap", "shadow"):
        st = EpisodicStore()
        for e in store.eps:
            if e.speech != nm:
                st.add(e.t, e.kc, e.feat, e.speech, e.place, e.needs, e.event)
        c, w = EXPERIENCES[nm]
        kc, f = evoke(fly, c, w, 20_000 + seed); seed += 1
        got = st.retrieve(t, kc=kc, feat=f, k=3)
        m3[nm] = {"recalled_after_delete": sum(ep.speech == nm for ep, _, _ in got)}
    res["M3"] = m3
    res["M3_pass"] = all(v["recalled_after_delete"] == 0 for v in m3.values())
    # ---- M5a: implants and edits
    flagged, trials = 0, 100
    for i in range(trials):
        st = EpisodicStore()
        for e in store.eps[:40]:
            st.add(e.t, e.kc, e.feat, e.speech, e.place, e.needs, e.event)
        j = int(rng.integers(1, 39))
        if i % 2:                                 # edit an existing memory
            st.eps[j].speech = "honey"
        else:                                     # implant a forged one
            fake = Episode(j, st.eps[j].t + 1, st.eps[j].kc, st.eps[j].feat, "honey",
                           "arena", {}, prev=st.eps[j].hash)
            fake.hash = fake.digest()
            st.eps.insert(j + 1, fake)
        flagged += bool(st.verify())
    res["M5a"] = {"flagged": flagged / trials}
    res["M5a_pass"] = flagged / trials >= 0.9
    # ---- M11: save / load
    p = store.save(CACHE / "memory_spec_test.npz")
    st2 = EpisodicStore.load(p)
    same, drift = [], 0.0
    for nm in names:
        c, w = EXPERIENCES[nm]
        kc, f = evoke(fly, c, w, 30_000 + names.index(nm))
        a = [e.i for e, _, _ in store.retrieve(t, kc=kc, feat=f, k=3)]
        b = [e.i for e, _, _ in st2.retrieve(t, kc=kc, feat=f, k=3)]
        same.append(len(set(a) & set(b)) / max(len(set(a) | set(b)), 1))
    res["M11"] = {"retrieval_jaccard": float(np.mean(same)), "chain_ok_after_load": not st2.verify()}
    res["M11_pass"] = res["M11"]["retrieval_jaccard"] == 1.0 and res["M11"]["chain_ok_after_load"]
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "memory_specs.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
