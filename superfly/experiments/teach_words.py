"""Teach the fly the meaning of words, with its own learning rule.

Protocol: the in-silico version of DAN-substitution conditioning (real flies
learn when an odour is paired with optogenetic activation of PAM or PPL1
dopamine neurons; Claridge-Chang et al. 2009, Aso & Rubin 2016).

    word-sense  : a grafted second antennal lobe for words (150 PNs onto the
                  fly's real KCs with real PN->KC statistics); a word is a
                  combinatorial PN pattern from its letter trigrams
    teacher     : PAM (reward) or PPL1 (punishment) dopamine neurons, driven
                  through the synaptic port so they keep their own dynamics
    learning    : the fly's dopamine-gated KC->MBON plasticity, compartments
                  from the connectome (superfly/plasticity.py)
    MBONs       : given their spontaneous firing (~5-6 Hz), which the Shiu
                  model lacks, so a memory has a baseline to move

Every trial and every test starts from a quiet brain (activity reset,
learned weights kept), because the model is bistable: long drive tips it into
a brain-wide broadcast (research/superfly_findings.md section 4). Every phase is
monitored and an episode that ignites (KCs > IGNITE of the population) is
reported, not hidden.

Readouts, both measured on the word presented ALONE after training:
  memory trace   change of the word-evoked response of the MBONs in the
                 compartments the teacher innervates (reward word -> PAM
                 compartments, punished word -> PPL1 compartments). Learning
                 predicts a DROP in the paired compartments only.
  valence        net approach drive of the MBON population (each MBON signed
                 by the dopamine that teaches it).
Controls: 'blip' is presented in training without dopamine; 'moo' is never
presented. Both must change less than the trained words.

    python -m superfly.experiments.teach_words [trials]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from superfly.fly import SuperFly
from superfly.engine import ROOT

OUT = ROOT / "data" / "results" / "superfly"
WORDS = {"yum": +1, "zap": -1, "blip": 0, "moo": None}   # None: never trained
import os
WORD_HZ = float(os.environ.get("SUPERFLY_WORD_HZ", 70.0))   # ~5-9 % of KCs
# MBON spontaneous activity. Published LIF: 600 Hz -> ~5-6 Hz MBONs (800 ignites
# within ~1.8 s); adaptive LIF stays stable to >= 1500 Hz.
TONE = (float(os.environ.get("SUPERFLY_TONE_HZ", 600.0)), 2.0)
IGNITE = 0.25           # KC active fraction that marks a brain-wide broadcast


class Session:
    def __init__(self, fly):
        self.fly, self.e = fly, fly.e
        self.kc = fly.mb.kc_slots
        self.ignited = []

    def fresh(self, settle_ms=100.0):
        f = self.fly
        f.reset()
        f.tone(*TONE)
        f.run(settle_ms)

    def window(self, ms, label):
        """Run `ms` and return per-MBON rate; record ignition."""
        e = self.e
        c0 = e.counts.copy()
        self.fly.run(ms)
        d = e.counts - c0
        kc_frac = float(np.mean(d[self.kc] > 0))
        if kc_frac > IGNITE:
            self.ignited.append((label, round(kc_frac, 3)))
        return d[self.fly.mbon_slots] / (ms / 1000.0), kc_frac

    def respond(self, word, ms=600.0, reps=4):
        """Word-evoked MBON change from baseline, averaged over `reps` fresh
        episodes (a single 300 ms window is ~3 spikes per MBON -- Poisson
        noise of the same size as the learned change)."""
        ds, ks = [], []
        for r in range(reps):
            self.fresh()
            base, _ = self.window(ms, f"baseline:{word}:{r}")
            self.fly.sense(words=[word], word_hz=WORD_HZ)
            self.fly.run(50.0)
            resp, kcf = self.window(ms, f"test:{word}:{r}")
            self.fly.sense({})
            ds.append(resp - base)
            ks.append(kcf)
        return np.mean(ds, 0), float(np.mean(ks))


def main(trials=6, eta_dep=0.01, eta_pot=0.0025, f_min=0.3):
    # f_min: KC->MBON depression is partial in the fly (Hige et al. 2015 report
    # pairing leaves a fraction of the response), and a fully silenced MBON
    # removes inhibition that keeps this model's brain out of its broadcast.
    fly = SuperFly(plasticity_kw=dict(eta_dep=eta_dep, eta_pot=eta_pot, f_min=f_min))
    e, mb = fly.e, fly.mb
    S = Session(fly)
    print(f"model {e.model} gain {e.gain}; plastic edges {mb.n_edges}; "
          f"DA modulatory: {fly.da_modulatory}", flush=True)
    pam_c = mb.reward_share > 0.5            # MBONs taught mainly by PAM
    ppl_c = mb.punish_share > 0.5            # ... mainly by PPL1
    print(f"compartments: {pam_c.sum()} PAM-taught MBONs, {ppl_c.sum()} PPL1-taught")

    mb.enabled = False
    pre = {}
    for w in WORDS:
        d, kcf = S.respond(w)
        pre[w] = d
        print(f"  pre  '{w}': KCs {100 * kcf:4.1f}%  PAM-cmp {d[pam_c].mean():+6.2f} Hz  "
              f"PPL1-cmp {d[ppl_c].mean():+6.2f} Hz", flush=True)

    mb.enabled = True
    excluded = []
    t0 = time.perf_counter()
    for k in range(trials):
        for w, sign in WORDS.items():
            if sign is None:
                continue
            S.fresh()
            snap = mb.state_dict()
            n_ign = len(S.ignited)
            fly.sense(words=[w], word_hz=WORD_HZ)
            S.window(200.0, f"train{k}:{w}:word")      # word first ...
            fly.teach(sign)
            S.window(600.0, f"train{k}:{w}:paired")    # ... then dopamine
            fly.teach(0)
            fly.sense({})
            if len(S.ignited) > n_ign:                 # the brain seized:
                mb.load_state_dict(snap)               # exclude the trial
                excluded.append(f"train{k}:{w}")
        print(f"  trial {k + 1}/{trials} ({time.perf_counter() - t0:.0f}s)", flush=True)
    mb.enabled = False

    res = {"model": e.model, "gain": e.gain, "trials": trials,
           "eta_dep": eta_dep, "eta_pot": eta_pot, "words": {}}
    for w in WORDS:
        d, kcf = S.respond(w)
        dd = d - pre[w]
        trace_pam, trace_ppl = float(dd[pam_c].mean()), float(dd[ppl_c].mean())
        val = float(np.dot(mb.valence, dd))
        res["words"][w] = dict(kc_frac=kcf, d_pam_cmp=trace_pam,
                               d_ppl1_cmp=trace_ppl, d_valence=val)
        print(f"  post '{w}': KCs {100 * kcf:4.1f}%  change PAM-cmp {trace_pam:+6.2f} Hz  "
              f"PPL1-cmp {trace_ppl:+6.2f} Hz  valence {val:+7.2f}", flush=True)
    W = res["words"]
    # the memory-trace prediction, compartment by compartment
    spec_yum = W["yum"]["d_pam_cmp"] < 0 and W["yum"]["d_pam_cmp"] < W["yum"]["d_ppl1_cmp"]
    spec_zap = W["zap"]["d_ppl1_cmp"] < 0 and W["zap"]["d_ppl1_cmp"] < W["zap"]["d_pam_cmp"]
    trained = min(abs(W["yum"]["d_pam_cmp"]), abs(W["zap"]["d_ppl1_cmp"]))
    controls_quiet = all(abs(W[c]["d_pam_cmp"]) < trained and abs(W[c]["d_ppl1_cmp"]) < trained
                         for c in ("blip", "moo"))
    res.update(edges_changed=int((mb.f != 1).sum()),
               ignited=S.ignited, excluded_trials=excluded, f_min=f_min, specific_reward=bool(spec_yum),
               specific_punish=bool(spec_zap), controls_quiet=bool(controls_quiet))
    test_ignited = [x for x in S.ignited if not x[0].startswith("train")]
    ok = spec_yum and spec_zap and controls_quiet and not test_ignited
    res["learned_correctly"] = bool(ok)
    print(f"edges changed {res['edges_changed']}/{mb.n_edges}; ignited episodes: "
          f"{S.ignited or 'none'}; training trials excluded: {excluded or 'none'}")
    print(f"reward trace specific: {spec_yum}  punishment trace specific: {spec_zap}  "
          f"controls quiet: {controls_quiet}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"teach_words_{e.model}_g{e.gain}.json").write_text(json.dumps(res, indent=1))
    print("\nLEARNED CORRECTLY" if ok else "\nDID NOT LEARN AS PREDICTED")
    return res


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
