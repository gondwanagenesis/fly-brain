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
