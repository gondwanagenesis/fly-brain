# SUPERFLY

**An uplifted fruit fly that is still the fruit fly.**

The goal is to give a simulated *Drosophila* more capacity, through the
learning it already has, grafted neurons and a language model, while the
original fly stays in charge of what it perceives and does. Every addition
is measured for how much of the behaviour is still the fly's.

This document is the project's working front page. Everything in it has a
command that reproduces it. Where something failed, it says so.

---

## The architecture: the fly at the centre, additions at the edges

```
 L4  VOICE      a language model wired into the fly's neurons, both ways
                  fly -> neural tokens -> LM          (speech)
                  words -> LM hidden state -> senses  (hearing)
                superfly/flylm.py, superfly/bridge.py, superfly/chat.py
 L3  GRAFTS     new neurons appended to the connectome as first-class cells
                  word lobe: a second antennal lobe, for words
                  expanded mushroom body: more Kenyon cells (memory capacity)
                superfly/graft.py
 L2  INTERFACE  senses in through real sensory neurons; readouts from the
                fly's own descending and motor neurons; region map
                superfly/anatomy.py, superfly/fly.py
 L1  LEARNING   the fly's own rule: dopamine-gated KC->MBON plasticity,
                compartments read from the connectome
                superfly/plasticity.py
 L0  THE FLY    all 138,639 FlyWire v783 neurons, Shiu et al. 2024 LIF, on
                this repo's bit-exact native kernel
                superfly/engine.py -> flyloop/native_engine.py
```

Rules that keep the fly the author:

1. **L0 is never edited in place.** Every graft or correction is opt-in,
   applied when the engine is built, and named. With nothing attached, the
   engine is bit-identical to the unmodified one
   (`superfly/tests/test_identity.py`).
2. **Additions talk to the fly only through its own neurons.** Sensory drive
   goes into real receptor neurons. Grafts and teaching go through the same
   1.8 ms delay ring as native synapses and respect each target's
   refractoriness.
3. **The voice sees the central brain only.** It never sees the stimulus,
   your sentence or the receptor neurons. Its only route back into the fly is
   the senses.
4. **External yardstick.** "Is it still a fly?" is answered by
   [flybench](https://github.com/brandoncho369/flybench), an independent
   36-task behavioural benchmark, not by tests we wrote ourselves.

---

## What has been measured

Full log with commands: [research/superfly_findings.md](research/superfly_findings.md).
Survey of everyone else's work: [research/review/05_existing_software.md](research/review/05_existing_software.md).

| result | value |
|---|---|
| Native kernel exact on Linux (was Windows-only) | 8/8 regimes bit-identical after detecting the PyTorch wheel's rounding at run time |
| **flybench, published model on this engine** (gain 1.0, 35 tasks, 3 seeds) | **graded 0.796**, core 1.00, hard 0.682. flybench's best leaderboard entry is 0.791; our run covers 23 applicable tasks against their 25, on this repo's connectome |
| flybench, gain 0.45 / adaptive LIF 0.45 | 0.686 / 0.658: both fail sugar→proboscis on this engine |
| "Sugar GRNs" in Shiu et al. that are sugar neurons by v783 annotation | 10 of 21 |
| Antennal lobe | **bistable**: above ~25 Hz any glomerulus recruits 55–81 % of the AL; odour identity lost. Open problem, also found by flybench |
| Word lobe coding | 5–9 % of Kenyon cells per word, pairwise overlap ~0.08: sparse, like odours in a real fly |
| Dopamine teacher | PAM activation → PAM 17.8 Hz, PPL1 3.9 Hz; PPL1 activation → PPL1 20.5 Hz, PAM 0 |
| Word learning | memory traces form in the **correct compartment with the correct valence sign** and no seizures, but are **< 1 Hz and do not yet exceed untrained controls**. Not yet a demonstrated memory (see below) |

## Problems found in the model itself, and what was done

| problem | evidence | response |
|---|---|---|
| AL broadcast | probe_olfaction under 4 settings | not fixed. Four hypotheses tested (gain, adaptation, transmitter signs, terminal inputs). Routed around: the word lobe feeds Kenyon cells directly |
| Low-confidence serotonin/dopamine LNs drive the AL | NT prediction confidence 0.28–0.6 | **not** flipped: literature ground truth says those LN types are cholinergic (Shang et al. 2007) |
| Dopamine signed as fast excitation | teaching excited KCs broadly; 50k of 62k synapses changed | with plasticity on, DAN→KC/MBON fast synapses are removed: dopamine acts through the modelled plasticity |
| No spontaneous activity | sparse KC codes never reach MBON threshold (0/96 fire) | opt-in MBON background (`SuperFly.tone`). The published LIF ignites above ~800 Hz of it; adaptive LIF is stable to ≥1500 Hz |
| Brain-wide ignition during learning | depressing MBONs disinhibits the network | partial depression (f ≥ 0.3), fresh episodes, ignited trials excluded and reported |
| Weak MBON readout | words evoke ~1 Hz MBON changes under uniform weights | next: the expanded mushroom body, more Kenyon cells → a larger evoked signal |

---

## Running it

```bash
# once: data the repo does not ship
python code/build_fanout_csc.py
curl -o data/flywire_meta/neuron_annotations.tsv \
  https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv

python -m superfly.anatomy                       # the regions and populations
python -m superfly.tests.test_identity           # un-uplifted fly == original fly
python -m superfly.experiments.probe_senses      # what the brain does with each sense
python -m superfly.experiments.probe_olfaction   # the antennal-lobe broadcast
python -m superfly.experiments.teach_words       # word learning, with controls
python -m superfly.experiments.make_corpus 800   # record the fly for the voice
python -m superfly.bridge tiny                   # train + test the from-scratch voice
python -m superfly.bridge pretrained             # train + test the SmolLM2 voice
python -m superfly.chat                          # talk to it

# the external benchmark
python -m superfly.bench.flybench_adapter        # build flybench's connectome from repo data
PYTHONPATH=. flybench run -c flywire783_repo --gain 1.0 --seeds 3 \
  --simulator superfly.bench.flybench_adapter:NativeSim
```

Environment variables: `SUPERFLY_MODEL` (`lif_euler` = published, `lif_adapt`
= with spike-frequency adaptation), `SUPERFLY_GAIN`, `SUPERFLY_THREADS`.

## Status

Built and measured: L0–L3 machinery, the flybench integration, the word lobe,
the fly's plasticity and the voice pipeline. **Not yet achieved:** a word memory
that beats its controls, and the voice's fly-mind test results (in progress;
written to `data/results/superfly/bridge_*.json`).
