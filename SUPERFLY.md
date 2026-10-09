# SUPERFLY

**An uplifted fruit fly that is still the fruit fly.**

The goal is to give a simulated *Drosophila* more capacity, through the
learning it already has, grafted neurons and a language model, while the
original fly stays in charge of what it perceives and does. Every addition
is measured for how much of the behaviour is still the fly's.

This document is the project's working front page. Everything in it has a
command that reproduces it. Where something failed, it says so.

## Try it on your computer

Download the zip from the [README](README.md#run-it-yourself-one-download-one-double-click),
unzip, and double-click `Start-SUPERFLY-Windows.bat` (Windows),
`Start-SUPERFLY-Mac.command` (macOS) or run `./start-superfly.sh` (Linux).
Only Python 3.10+ must be installed first; the launcher sets up everything
else (including a C compiler for the simulation core, via the `ziglang`
package) and opens the Lab. By hand:

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -r requirements-superfly.txt
python -m superfly.quickstart                     # --small-lm, --no-lm, --flywire
```

Screenshot of the live Lab: [research/lab_live_male_cns.png](research/lab_live_male_cns.png).

---

## The architecture: the fly at the centre, additions at the edges

```
 L6  LAB        3D view: the connectome at the centre, every module around it,
                wired, all firing live; the world; the conversation
                superfly/lab.py, superfly/lab/index.html
 L5  MIND       conversation: a local Qwen2.5-1.5B that may only claim, in the
                first person, what the fly's records contain; every claim checked
                superfly/mind.py
 L4  VOICE      FlyLM: central-brain activity -> neural tokens -> inner speech,
                and hearing heads: words -> the fly's senses
                superfly/flylm.py, superfly/bridge.py
     MEMORY     episodes = stored brain states (Kenyon-cell code + central
                features), hash-chained, recalled by re-evocation
                superfly/memory.py
 L3  GRAFTS     word lobe (a second antennal lobe, for words); expanded MB
                superfly/graft.py
 L2  INTERFACE  senses in through real receptor neurons, per side; body state
                in through real interoceptive neurons; actions out from the
                fly's own descending and motor neurons
                superfly/anatomy.py, superfly/fly.py
     WORLD      a 2D arena with odour plumes, wind, warmth, humidity, light,
                looming shadows, food, water; a body with needs
                superfly/world.py, superfly/life.py
 L1  LEARNING   dopamine-gated KC->MBON plasticity, compartments from the connectome
                superfly/plasticity.py
 L0  THE FLY    the connectome, Shiu et al. 2024 LIF, on this repo's bit-exact
                native kernel. Two connectomes (SUPERFLY_CONNECTOME):
                  flywire783  FlyWire FAFB v783, adult female brain, 138,639 neurons
                  male_cns    Janelia male CNS v1.0: brain + optic lobes + nerve
                              cord, 165,122 neurons (superfly/connectomes/male_cns.py)
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
| **Untaught SUPERFLY vs original fly** | **bit-identical** (identity gate PASS, 600 steps) |
| **Voice A: from-scratch FlyLM, 2.9M params** | held-out F1 **0.927** (final network; 0.935 on the pre-correction network); silent brain → "..." **100 %**; shuffled brain: follows the given brain **0.927**, leaks the true stimulus 0.197 |
| Voice B: frozen SmolLM2-360M + neural prefix | F1 0.38 after 400 CPU steps; silent brain → "..." 100 % (no confabulation, undertrained) |

| **Living in a world** (FlyWire) | 0.33x real time on 4 cores, brain + world; loom -> giant fibre 50-131 Hz -> take-off |
| Voice trained in the world | held-out F1 0.857 (linear 0.698); silent-brain failure found and fixed in the data (findings s10) |
| **Male CNS** (brain + optic lobes + nerve cord) | converted in 51 s; broadcasts at Shiu's scale; calibrated gain 0.65: sugar->MN9 44 Hz, bitter 0, wind->groom 50 Hz, loom->GF 280 Hz (findings s11) |

## Problems found in the model itself, and what was done

| problem | evidence | response |
|---|---|---|
| AL broadcast | probe_olfaction under 4 settings | not fixed. Four hypotheses tested (gain, adaptation, transmitter signs, terminal inputs). Routed around: the word lobe feeds Kenyon cells directly |
| Low-confidence serotonin/dopamine LNs drive the AL | NT prediction confidence 0.28–0.6 | **not** flipped: literature ground truth says those LN types are cholinergic (Shang et al. 2007) |
| Dopamine signed as fast excitation | teaching excited KCs broadly; 50k of 62k synapses changed | with plasticity on, DAN→KC/MBON fast synapses are removed: dopamine acts through the modelled plasticity |
| No spontaneous activity | sparse KC codes never reach MBON threshold (0/96 fire) | opt-in MBON background (`SuperFly.tone`). The published LIF ignites above ~800 Hz of it; adaptive LIF is stable to ≥1500 Hz |
| Brain-wide ignition during learning | depressing MBONs disinhibits the network | partial depression (f ≥ 0.3), fresh episodes, ignited trials excluded and reported |
| Male CNS over-driven at Shiu's scale | 10 % of neurons active on any stimulus | global gain 0.65 from a calibration against FlyWire; bitter/wind still recruit 3-6 % |
| Silent brain in the world corpus | the world-trained voice confabulated on silence | silent windows added, as in the first corpus |
| Weak MBON readout | words evoke ~1 Hz MBON changes under uniform weights | next: the expanded mushroom body, more Kenyon cells → a larger evoked signal |

---

## Talking to it

```
YOU: here is some sugar for you      [MN9 feeding neuron 56 Hz]    FLY: something sweet. i want to eat.
YOU: careful, bitter poison          [no motor output]             FLY: yuck, bitter.
YOU: a strong breeze is blowing      [grooming DNs 11 Hz]          FLY: a breeze. i clean my antennae.
YOU: say 'zap'                       [word lobe; turning DNs 3 Hz] FLY: i hear 'zap'. i turn.
YOU: lots of honey and a bit of salt [MN9 70 Hz]                   FLY: something sweet. i want to eat.
YOU: listen to this buzz             [hearing missed 'buzz']       FLY: ...
YOU: hello fly                       [nothing reaches its senses]  FLY: ...
```

Errors visible here: hearing missed "buzz" (an earlier training run caught it,
and the giant fibre then fired at 24 Hz: "a sound. i fly away."); "salt" was
dropped from a two-sense sentence; and "i turn" was said with the turning
neurons at 3 Hz, below the 5 Hz the training labels used.

(`data/results/superfly/chat_demo_tiny.txt`.) It **reports**; it does not converse.
The content is the fly's: what it sensed, what its motor neurons did, which word
it heard. The phrasing is ours. It has nothing more to say because the
simulated brain has no drives, moods or working memory yet. That is the next
stage (research/review/06–08).

Live, in 3D: `python flyloop/studio.py --superfly`, then open
http://127.0.0.1:8765. Every message runs one real episode of the brain, and the
render shows which neurons fired.

## Living, talking, and watching it

```bash
python -m superfly.life 120                  # the fly lives 2 min in its world and says what it says
python -m superfly.mind                      # live 30 s, then talk; replies verified against its records
python -m superfly.lab                       # http://127.0.0.1:8770 : brain, modules, world, chat, live
SUPERFLY_CONNECTOME=male_cns python -m superfly.lab      # the same, on the male CNS
python -m superfly.tests.test_world_specs    # SPECS B2, B3, C2, C3, C4
```

How a reply is made (superfly/mind.py): your sentence drives the fly's own
senses (or its word lobe) for 0.6 s; the fly lives 1 s with that input; the
brain state it evoked is matched against stored episodes (re-evocation); a fact
sheet is built from the fly's inner speech, motor neurons, needs, place, and
the recalled episodes; Qwen writes the reply from it; a claim checker rejects
any first-person sensation, action, word or event the facts do not contain,
asks again, and finally falls back to a reply built from the facts alone.
Every turn is logged with its facts and verdict.

The Lab: `research/lab_mock.png` shows the layout (mock data).

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

Environment variables: `SUPERFLY_CONNECTOME` (`flywire783` default, `male_cns`; run `python -m superfly.connectomes.male_cns` once), `SUPERFLY_MODEL` (`lif_euler` = published, `lif_adapt`
= with spike-frequency adaptation), `SUPERFLY_GAIN`, `SUPERFLY_THREADS`.

## Status

Built and measured: L0-L3 machinery on two connectomes, the flybench
integration (FlyWire), the word lobe, the fly's plasticity, the voices with
their fly-mind tests, the world and the life loop, episodic memory, the
conversational tier with its claim checker, and the Lab. **In progress:** the
voice and the world specs on the male CNS; the memory specs (M2-M11) and the
conversation specs (D4, D5). **Not yet achieved:** a word memory that beats its
controls (M5b). Acceptance specs and their status: [SPECS.md](SPECS.md).
