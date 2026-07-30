---
paper: "Neuromorphic Simulation of the Fruit Fly Connectome on Loihi 2"
authors: Wang et al. (Sandia National Laboratories)
year: 2025
arxiv: https://arxiv.org/abs/2508.16792
pdf: wang2025_loihi2_drosophila.pdf
text: wang2025_loihi2_drosophila.txt
scanned: 2026-07-30
relevance: "THE benchmark to beat. Contains Table 1, the only published wall-clock numbers for whole-brain FlyWire LIF simulation."
---

# Wang et al. 2025 — FlyWire connectome on Loihi 2

MD scan companion. Page/line refs are to `wang2025_loihi2_drosophila.txt`
(`pdftotext -layout`).

## Setup

- Same model as ours: two-state LIF, τ_syn = 5 ms, refractory 2.2 ms, uniform
  axonal delay 1.8 ms, 138,639 neurons / ~50M synapses, FlyWire.
- **12 Loihi 2 chips** (1440 neurocores), Kapoho Point system (L22, L581).
  Shared axon routing is what let the *unmodified* connectome fit on 12 rather
  than 20 chips (L578–581).
- Ran at **both dt = 0.1 ms and dt = 1 ms** (L374–377).
  - dt = 0.1 ms: refractory = 22 steps, delay = 18 steps — **exact, no rounding**.
  - dt = 1 ms: both approximated to **2 steps** (i.e. 2.0 ms), for speed (L737–738).
- STACS reference: forward Euler, 0.1 ms update, **8 MPI processes**, 1 ms
  communication interval (L288, L302–305). *Not* a large HPC allocation.

## Table 1 — runtime for 1 s of simulated time (L763–771)

⚠️ **The FlyWire column is in MILLISECONDS.** The header reads
`FlyWire (ms)`; the baseline-rate columns are in seconds. The rows for
Loihi 2 are line-wrapped in the text extraction — `53.76` belongs to the
0.1 ms row and `12.40` to the 1 ms row.

| Platform | sugar expt, per 1 s sim | = s / sim-second | vs real time |
|---|---|---|---|
| Brian 2 (reference) | 4419 ± 236 ms | **4.42** | 4.4× slower |
| STACS (8 processes) | 2656 ± 79.8 ms | **2.66** | 2.7× slower |
| **Loihi 2 @ dt=0.1 ms** | 53.76 ± 0.896 ms | **0.0538** | **18.6× faster** |
| **Loihi 2 @ dt=1 ms** | 12.40 ± 0.279 ms | **0.0124** | **81× faster** |

Consistency checks that confirm the ms reading:
- Loihi 2 (0.1 ms) vs Brian 2 = 4419 / 53.76 = **82×**.
- The paper's claim of ">100× at sparser activity" matches the 0.5 Hz column:
  10.13 / 0.0958 = **106×** (L778).
- The paper states Loihi 2 was "capable of performing better than realtime"
  (L776) — only true under the ms reading (0.054 s per simulated second).
- Brian 2 at 4.42 s/sim-second = 0.44 ms/step for 138,639 neurons, which is a
  believable rate for Brian 2's C++ standalone codegen. The alternative reading
  (4419 s/sim-second = 442 ms/step) would be ~3 µs per neuron-step, far too slow
  for compiled code.

## Fidelity compromises Loihi 2 makes (we do not)

These matter for any speed comparison — Loihi 2 is fast *and* approximate:

- **9-bit fixed-point integer weights** (1 sign bit), and weights **capped**:
  theoretical max 512 vs an original range of −2405…1897 (L240, L491–494).
  454 negative and 637 positive weights were clipped — 0.007% of weights
  (L706–707) — with a visible off-parity cluster in spike rates (Fig. 13).
- **Fixed-point neuron state** and fixed-point update constants (L355, L370).
- **Conductance-only input** approximation (L713, L745).
- At dt = 1 ms, delay 1.8→2.0 ms (+11%) and refractory 2.2→2.0 ms (−9%).

So Loihi 2 @0.1 ms is the fair speed target; Loihi 2 @1 ms trades documented
accuracy for a further 4.3×.

## Consequences for this project

1. **Our HANDOFF.md Table in §8 is wrong by 1000×** on the Brian 2, STACS and
   Loihi 2 rows — it recorded the ms figures as seconds. See
   `research/CORRECTION_benchmark_units.md`. Every claim of the form "we beat
   Loihi 2" derived from that table is void.
2. The real bar is **0.0538 s/sim-second**. Our best measured native kernel is
   3.31 s/sim-second on the same protocol, so Loihi 2 @0.1 ms is **~62× ahead**.
3. Brian 2 at 4.42 s/sim-second is only 1.33× behind our native kernel. Brian 2
   generates C++; this is strong evidence that compiled codegen (what we just
   did) is the correct axis, and that we should benchmark directly against
   Brian 2 standalone rather than against the repo's PyTorch backend, which is
   an unusually weak baseline.
4. Their headline conclusion matches ours independently: speedup is
   **activity-dependent**, maximal at sparse activity (L856–857).
5. They cite Knight & Nowotny, "GPUs outperform current HPC and neuromorphic
   solutions in terms of speed and energy" (ref [33], L1002) — worth reading as
   the counterpoint to the neuromorphic framing.
