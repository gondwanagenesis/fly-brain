# SUPERFLY findings log

Every result here has the command that produced it. Negative results are kept.
Host: 4-core Xeon (AVX-512), Linux, torch 2.14.1, FlyWire v783 as shipped in
this repo (`data/2025_Connectivity_783.parquet`, all synapses, min count 1).

---

## 1. The kernel now runs, and is exact, on Linux

`flyloop/verify_all.py 400` failed on Linux in 3 of 8 regimes (sugar STATE@394,
broad-10000 STATE@141, saturating STATE@42). Cause: the kernel reproduced a
Windows artefact. ATen's scalar tail of `add_(t, alpha=)` rounds **unfused** in
MSVC-built Windows wheels and **fused** in GCC-built Linux wheels. The first
divergence was 1 ULP on neuron 69,316, a scalar-tail neuron
(`research/audit/a13_linux_torch_divergence.py`). `native_lib.aten_tail_fused()`
now measures the wheel's behaviour at start-up. Result: **ALL BIT-IDENTICAL,
8/8 regimes** on Linux. Windows behaviour is unchanged.

## 2. Ten of the 21 "sugar GRNs" are sugar neurons

The 21 ids used by Shiu et al. 2024 and every script in this repo map, in the
v783 annotation (Schlegel et al. 2024), to 10 × LB3c (sugar), 5 × LB3d
(high salt), 4 × LB4b (putative attractive) and 2 × LB3b (sugar/low salt).
High-salt GRNs alone drive MN9 at 66 Hz in the model (sugar LB3c: 65 Hz), so the
mixture doesn't change the headline result. flybench's task 2 independently
restricts sugar to LB3b–c on receptor-line evidence.
`superfly/anatomy.py` exposes both `taste.shiu_sugar` and `taste.sugar`.

## 3. What the unmodified brain does with each sense

`python -m superfly.experiments.probe_senses 500 150` (Poisson, 150 Hz, 500 ms):

| sense | result |
|---|---|
| sugar (LB3c) | MN9 65 Hz, ~7.9k spikes — reproduces Shiu |
| bitter, water | MN9 0 Hz |
| wind/gravity (JO-C+E) | grooming DNs 10.6 Hz |
| sound (JO-A+B) | stays in the auditory pathway |
| **any odour, temperature, humidity** | **~220k spikes, near-identical across channels**: the whole antennal lobe at ~94 Hz, KCs 30 Hz mean, PPL1 ~102 Hz, MBONs ~77 Hz |

Neither PAM nor PPL1 dopamine neurons respond to sugar or bitter through the
fly's own circuits in this model (0 Hz). The dopamine neurons only fire as part
of the olfactory broadcast. **So teaching signals have to be supplied**, as
optogenetic DAN activation supplies them in real experiments.

## 4. The antennal lobe is bistable, and odour identity is lost

`python -m superfly.experiments.probe_olfaction` (6 glomeruli × rate, 300 ms):

| model / gain | below threshold | above threshold (≥25 Hz for most) |
|---|---|---|
| Shiu LIF, 1.0 | specific (PN specificity 1.0) | 81 % of AL recruited in 25–75 ms, 65 % of KCs active, PN specificity 1–9 % |
| Shiu LIF, 0.45 | specific | 60 % of AL, 6 % of KCs, specificity 2–9 % |
| adaptive LIF, 1.0 | specific | 78 % of AL, 52 % of KCs |
| **adaptive LIF, 0.45** | specific | 55 % of AL, **KCs 1.6–2.3 % (sparse)**, specificity 3–11 % |

The bistability survives every setting. flybench found the same independently
(FINDINGS 2026-09-14: one glomerulus → 84 % of PNs; DA1 sparseness 0.0009 vs
0.90 measured).

### Hypotheses tested and refuted

1. **"Monoamine-mispredicted local neurons are really inhibitory."** The AL's
   net excitation comes mostly from LNs predicted serotonergic (+158k synapses,
   prediction confidence 0.28–0.6) or dopaminergic (+45k). But the literature
   ground truth (`data/external/nt_gt_data.csv`; Shang et al. 2007,
   confidence 5) lists lLN1_bc, lLN2T_a/b/c and lLN2X03 as **cholinergic**:
   excitatory LNs are real, so the excitatory sign is right. Flipping them
   would have been a wrong "fix". Not applied.
2. **"Central input fires receptor-neuron terminals."** Sensory neurons have
   their somata in the periphery, so all their in-brain input is axo-axonic.
   Removing it (`GraftBuilder.peripheral_terminals`, 16,352 neurons) is
   biologically correct but changes almost nothing (AL recruited 0.77 vs 0.81
   at gain 1.0). The broadcast is generated inside the PN/LN network.

Open: the excitatory LNs in the fly act largely through gap junctions and onto
inhibitory partners, and the AL has strong GABA-B presynaptic inhibition. A
point-neuron, chemical-only, sign-only model represents none of that.

**Consequence for SUPERFLY.** Odour learning through the native antennal lobe
is not credible in this model at any setting tested. The design routes around
it: taste and mechanosensory channels are specific; odours below threshold are
specific; and the word-sense graft enters the mushroom body directly, where
adaptive LIF at gain 0.45 gives sparse KC coding.

## 5. flybench on this engine

`superfly/bench/flybench_adapter.py` runs the native kernel under flybench's
contract. Core tier, published model (gain 1.0), 1 seed: **5/5 tasks pass, all
graded 1.00** (sugar→MN9 60.9 Hz at 0.3 % of the brain active; bitter suppresses
sugar; loom→giant fibre 186 Hz; bitter alone does not drive MN9). flybench's
own "Shiu 2024, gain 1.0" row scores 0.77 on the core tier. The differences are
known: this kernel reproduces the PyTorch refractory semantics (input discarded
while refractory, g reset on spike), and the graph includes 1–4 synapse edges.
The full-suite run is in `data/results/superfly/flybench/`.

## 6. The fly's own learning: what works and what does not yet

`python -m superfly.experiments.teach_words 6` (DAN-substitution conditioning:
word + PAM = reward, word + PPL1 = punishment, 'blip' unpaired, 'moo' never
presented; every episode from a quiet brain; 4 × 600 ms tests per word).

| setting | coding | reward trace | punishment trace | controls | ignition |
|---|---|---|---|---|---|
| published LIF 1.0, tone 600 | 5–6 % KCs | wrong compartment | wrong | moved most | 6 episodes |
| adaptive LIF 1.0, tone 1000, f ≥ 0.3 | 6–9 % KCs | **correct compartment** (−0.87 Hz) | **correct** (−0.34 Hz) | as large as the traces | **none** |
| + expanded MB (20,000 grafted KCs) | 7–8 % (native 3.1 %, grafted 2.8 %) | **correct**, net approach +27.5 | not specific | contaminated by test ignition | 4 episodes |

**Not yet a demonstrated memory.** The machinery is in place and verified: the
teacher drives the right dopamine neurons, the word codes are sparse and
distinct, and the traces form in the paired compartments. The blocker is the
model's bistability. Word-evoked MBON changes are ~1 Hz under uniform synaptic
weights. Anything that strengthens them, a larger mushroom body or more MBON
drive, moves the brain toward its broadcast state.

Related corrections, each opt-in:
- **Dopamine as plasticity**: DAN→KC/MBON fast excitation removed while
  plasticity is on (with it, 50,629 of 62,261 synapses changed in 3 trials).
- **MBON spontaneous activity** (`SuperFly.tone`): without it, 0/96 MBONs fire
  to a sparse KC code. The published LIF ignites above ~800 Hz of tonic input;
  the adaptive LIF is stable to ≥ 1500 Hz.

## 7. The untaught SUPERFLY is the original fly

`python -m superfly.tests.test_identity 600` — **PASS**. Native engine vs
SuperflyEngine (ports attached, torch RNG): bit-identical state and spikes for
600 steps. SuperflyEngine vs SuperflyEngine + MBPlasticity, nothing learned:
bit-identical (906 spikes).

## 8. What the fly does, read from its own motor neurons

From the 800-episode corpus (`make_corpus`, published LIF 1.0), descending and
motor populations ≥ 5 Hz:

- feeding (MN9): with sugar (49) and high salt (29)
- grooming (DNg11/DNg62/DNge078): only with wind/JO-C+E (21/21), as in Shiu Fig. 5
- escape (giant fibre DNp01): with sound/JO-A+B (32/32)
- turning (DNa01/02): whenever the brain is broadly active (451 episodes)
- forward or backward walking: never

**Correction.** An earlier version of this log called a linear probe's F1
0.575 a "ceiling" on what the brain allows any decoder to recover. That was
wrong. The probe overfits (7,553 varying features, 640 training episodes;
recall 0.99, precision 0.41), and the voice below reaches 0.92 on the same
held-out episodes. With per-label thresholds the linear baseline scores 0.45.
It is a weak baseline, not a ceiling.

## 9. The voice: a language model wired into the fly's neurons

`python -m superfly.bridge tiny` — voice A, a 2.9M-parameter transformer
trained from scratch. Its only input is neural tokens projected from the
central brain (cell-type rates plus each Kenyon cell); it never sees the
stimulus. 800-episode corpus, 20 % held out.

| fly-mind test | voice A (first run) |
|---|---|
| held-out content F1 | **0.924** (precision 0.941, recall 0.908) |
| zero brain (silent fly) | **"..." every time** (content rate 0.0) |
| shuffled brain: follows the brain it is given | F1 **0.924** |
| shuffled brain: leaks the true stimulus | F1 **0.193** |

Example (held out): the fly tasted salt with mold present, its MN9 fired,
and the word 'bouba' was played. It said *"i taste salt. i smell mold. i hear
'pim'. i extend my proboscis. i turn."* The percepts and actions are right;
the word is wrong.

Hearing (human sentence → senses) first run: the correct sense ranked first
in most test sentences but with low confidence (0.14–0.5), so most requests
never reached the fly. Retrained with presence targets, synonyms and more
templates (results below once recorded).


## 10. Living in a world (2026-10-08)

**Why.** A brain that only answers one stimulus at a time has nothing to talk
about. Reviews 06 (feelings), 08 (grounded self) and 09 (world) all point to
the same minimum: a body with needs, in a place with structure, so that the
brain's activity has causes and consequences over time.

**What was built.** `superfly/world.py` (a 100 mm arena: sugar drop with a
fruit-odour plume, water drop with a humidity field, bitter patch with a
geosmin plume, a warm lamp, drifting wind with gusts, a day/night cycle, rare
looming shadows), and `superfly/life.py` (the brain and the world exchange
every 20 ms). Every receptor population is driven per side; hunger and thirst
drive the interoceptive neurons named in review 06 (MBON11, PPL101, ISN, ITP);
every movement other than the stand-in nerve cord's walking rhythm comes from
the fly's own descending neurons and MN9. Review 09's corrections were applied:
walking in bouts and pauses (time-average 3-5 mm/s at rest), a 30 s minimum gap
between shadows (repeated looms build a persistent state in real flies,
Gibson et al. 2015, which this fly is spared), take-off away from the shadow,
lamp peak 30 C. The world costs ~27 us per exchange, 0.3 % of the brain.

**First measurements (FlyWire).** 0.33x real time on 4 cores (SPECS C4
threshold 0.25x). A looming shadow drives the giant fibre to 50-131 Hz and the
fly takes off. Wind drives the turning DNs (DNa01/02) at 15-30 Hz, so the fly
"turns" in most windows; that is the brain's own output, not a decoding error.

**The voice failed in the world, and why.** The first voice, trained on 60-150 Hz
stimuli, said "i smell co2" with no CO2 source: world odours are <= 18 Hz
(the antennal lobe broadcasts above ~25 Hz, s4), outside its training range.
So the corpus was re-recorded inside the world (`experiments/record_life.py`,
800 life segments, 3,200 windows; labels from the world's receptor drive, the
needs, the word lobe and the fly's own motor neurons). Result: held-out F1
**0.857** (linear probe 0.698); shuffled brain follows 0.857. Two failures:
(1) a silent brain produced "i smell mold" (content rate 1.0), because the
world corpus contains no silent windows (interoception is always on) -- the
loader now adds them, as the first corpus had by design; (2) raw shuffled
"leak" 0.359 is above the D3 threshold, but labels such as "turn" and "wind"
are frequent, so leak is now reported against its base-rate null
(`leak_null_f1`: another window's true labels scored against this one's).

## 11. The male CNS: brain, optic lobes and nerve cord of one animal

**Why.** The owner's direction (2026-10-08): use the most complete connectome.
FlyWire FAFB is one female's brain and stops at the neck, which is why the body
needed a stand-in nerve cord. The Janelia male CNS v1.0 (Berg et al., Cell
2026) has brain, both optic lobes and the ventral nerve cord in one animal;
BANC (female brain + cord, Nature 2026) is the fallback for anything missing
(review 10).

**Conversion** (`superfly/connectomes/male_cns.py`, 51 s): 165,122 traced
neurons, 25,563,197 edges, 124,025,046 synapses (FlyWire: 138,639 / 15.1M /
~54M). Signs by the rule found in Shiu et al.'s own FlyWire file (GABA and
glutamate inhibitory, everything else excitatory; checked against the file:
84,162 of 85,881 cholinergic neurons excitatory, 21,863 of 24,804
glutamatergic inhibitory), plus histamine inhibitory. 34.7 % of neurons
inhibitory. Labels transferred from FlyWire by cell type, preferring the male
type when FlyWire has it (it is often finer: LB3c vs flywireType LB3); every
SUPERFLY population resolves (23 labellar sugar GRNs, 38 bitter, DNp01, MN9,
MBON11, PPL101, ITP, LC4/LPLC2 311, KC 4,064).

**Calibration** (`experiments/calibrate_connectome.py`, same stimuli on both):

| connectome, gain | sugar->MN9 | active (sugar) | bitter->MN9 | wind->groom | loom->GF | active (bitter) |
|---|---|---|---|---|---|---|
| FlyWire, 1.0 | 75 Hz | 0.3 % | 0 | 12.8 Hz | 180 Hz | 0.09 % |
| male CNS, 1.0 | 63 | **10 %** | 1 | 36 | 305 | 10 % |
| male CNS, 0.75 | 46 | 7.8 % | 0 | 57 | 289 | 7.9 % |
| male CNS, 0.70 | 45 | 7.3 % | 0 | 47 | 285 | 7.1 % |
| **male CNS, 0.65** | **44** | **0.6 %** | **0** | **50** | **280** | 6.5 % |
| male CNS, 0.60 | 20 | 0.4 % | 0 | 44 | 275 | 5.8 % |
| male CNS, 0.55 | 14 | 0.2 % | 0 | - | - | 0.4 % |
| male CNS, 0.45 | 0 | 0.1 % | 0 | 38 | 256 | 0.2 % |

At Shiu's FlyWire-fitted scale the male CNS broadcasts (10 % of neurons, and
every stimulus triggers grooming), as other groups also reported (review 10:
synapse density ~1.8x FlyWire). A sharp transition separates 0.65 from 0.70.
**Default male gain: 0.65** -- the highest gain at which the four signature
responses stay specific. Open difference: bitter and wind still recruit
3-6 % of the male network (FlyWire < 1 %).

## 12. The male CNS in its world: body specs pass, the voice is grounded

World specs on the male CNS (gain 0.65, 10 trials each,
`tests/test_world_specs.py`): sugar contact while hungry -> MN9 -> feeding in
10/10 (energy +0.12); bitter contact -> feeding 0/10; wind gust -> grooming
10/10; looming shadow -> giant fibre -> take-off 10/10; arousal 30 s after a
shadow at 0.25 % of its peak (the fly settles; SPECS B2). Real time 0.29x.

Voice trained in the world on the male CNS (2,000 windows + 12 % silent):
held-out F1 0.842 (linear probe 0.615); silent brain -> "..." (W2 pass);
shuffled brain followed at 0.842 and the true stimulus "leaks" at 0.332
against a base-rate null of 0.331 (W3 pass: content follows the brain, not
the world). W1 (>= 0.90) is not met. In the examples the fly often reports
"i extend my proboscis" when hungry -- MN9 fires in many windows when the
interoceptive hunger drive is on, a state-dependent proboscis readiness that
is the brain's own output.
