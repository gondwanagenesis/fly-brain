---
date: 2026-07-30
severity: critical
status: NOT YET MERGED INTO HANDOFF.md
---

# Correction: HANDOFF §8 prior-art table is wrong by 1000×

## What is wrong

`HANDOFF.md` §8 records:

```
| Simulator                | s/sim-second |
| Brian 2 (reference)      | 4419 ± 236   |
| STACS (Sandia, Charm++, 64 Summit nodes) | 2656 ± 80 |
| Eon PyTorch (our measurement)            | ~487       |
| Loihi 2 @ 1 ms           | 53.76 ± 0.9  |
```

Three errors, verified against the paper text
(`research/papers/wang2025_loihi2_drosophila.txt`, Table 1 at L763–771):

1. **Units.** The source column header is `FlyWire (ms)` — **milliseconds** per
   1 s of simulated time, not seconds. Every figure taken from it is 1000× too
   large.
2. **Row misassignment.** `53.76` is Loihi 2 at **dt = 0.1 ms**, not dt = 1 ms.
   The dt = 1 ms figure is `12.40 ± 0.279` ms, which the text extraction places
   on a wrapped line.
3. **STACS hardware.** The paper used **8 MPI processes** (L302–305), not
   64 Summit nodes. The "algorithmic structure beats parallelism" argument in
   HANDOFF §8, which rested on STACS getting only 1.66× "on 64 HPC nodes",
   loses its force: 1.66× on 8 processes is unremarkable, not evidence.

## Corrected table

| Platform | published | = s / sim-second | vs real time |
|---|---|---|---|
| Brian 2 (reference) | 4419 ± 236 ms | 4.42 | 4.4× slower |
| STACS (8 processes) | 2656 ± 79.8 ms | 2.66 | 2.7× slower |
| Loihi 2 @ dt=0.1 ms | 53.76 ± 0.896 ms | **0.0538** | **18.6× faster** |
| Loihi 2 @ dt=1 ms | 12.40 ± 0.279 ms | **0.0124** | **81× faster** |

Cross-checks confirming the ms reading are in the paper's MD companion. The
decisive one: the paper says Loihi 2 performed "better than realtime", which is
only true at 0.054 s per simulated second.

Note `~487 s/sim-second` for "Eon PyTorch" was **our own local measurement** of
the unoptimised upstream backend on this laptop, not a figure from the paper, so
it is unaffected by the unit error — but it must never again be tabulated
alongside the paper's numbers without saying so, because the two were measured
on different machines.

## What this invalidates

- **Any claim that this project is faster than Loihi 2.** It is not. On the
  sugar protocol, measured 2026-07-30 on the target laptop:

  | | s/sim-second | ms/step |
  |---|---|---|
  | Upstream PyTorch, unoptimised (local, prior session) | ~487 | ~48.7 |
  | Repo PyTorch engine, optimised (local, today) | 25.7 | 2.57 |
  | **Native fused AVX-512 kernel (local, today)** | **3.31** | **0.331** |
  | Brian 2 (published) | 4.42 | 0.44 |
  | Loihi 2 @0.1 ms (published) | 0.0538 | 0.0054 |

  → Loihi 2 @0.1 ms is **~62× ahead** of our best. Loihi 2 @1 ms is ~267× ahead.
- The framing in HANDOFF §8 that "their compromise is our opportunity" because
  "at dt = 1 ms they rounded both the 1.8 ms delay and 2.2 ms refractory to
  2 ms". They **also** ran at dt = 0.1 ms with delay = 18 steps and refractory =
  22 steps, i.e. **exact**. The compromise applies only to their optional fast
  mode, and even the 0.1 ms configuration is 62× ahead of us.

## What survives, and is still worth claiming

- The **same-machine, bit-identical** speedups are real and independently
  measured: **7.8×** over the repo's current optimised PyTorch engine on the
  sugar protocol (1.99× at saturating activity), with full state (v, g, refrac)
  bit-identical at every step across all regimes.
- **Fidelity is genuinely ours.** Loihi 2 uses 9-bit capped fixed-point weights
  (0.007% of weights clipped, with visible spike-rate deviation), fixed-point
  state, and a conductance-only input approximation. We are bit-identical fp32.
  A like-for-like comparison must say this.
- **Real time is close.** 3.31 s/sim-second is 3.3× from real time (1.0), which
  is a plausible target for a consumer laptop. That is the honest headline —
  not "we beat neuromorphic hardware".

## The repo already contains real GPU baselines — use them

`data/benchmark-results.csv` (721 successful rows) has the *same* sugar
experiment measured across six backends. Host GPU is an **RTX 4070**
(README L410: "tested on RTX 4070"). Spike counts agree across backends
(16.7k–17.2k per simulated second), so they are all simulating the same thing.

`t_run = 1 s`, `n_run = 1`, median of 6 rounds, `sim_time` only:

| backend | hardware | s / sim-second | vs real time |
|---|---|---|---|
| GeNN | RTX 4070 | **0.542** | 1.85× faster |
| NEST GPU | RTX 4070 | 1.066 | 0.94× |
| Brian2GeNN | RTX 4070 | 1.910 | 0.52× |
| Brian 2 C++ standalone | host CPU | 2.904 | 0.34× |
| **our native kernel** | **i7-1185G7 laptop, no dGPU** | **3.31** | **0.30×** |
| PyTorch (CUDA) | RTX 4070 | 9.817 | 0.10× |
| Brian2CUDA | RTX 4070 | 11.963 | 0.08× |

Per-trial cost with batching (`sim_time / n_run`, t_run = 1 s):

| backend | n=1 | n=4 | n=8 | n=16 | n=32 |
|---|---|---|---|---|---|
| GeNN | 0.542 | 0.182 | **0.140** | 0.549 | 0.667 |
| NEST GPU | 1.066 | 1.084 | 1.055 | 1.143 | 1.055 |
| Brian2GeNN | 1.910 | 1.934 | 1.999 | 2.007 | 1.981 |
| Brian 2 C++ | 2.904 | 2.012 | 1.695 | 1.725 | 2.225 |
| PyTorch (CUDA) | 9.817 | 8.255 | 6.044 | 5.672 | 5.749 |
| Brian2CUDA | 11.963 | 12.167 | 12.066 | 12.149 | 12.393 |

### What this establishes

1. **PyTorch CUDA is 18× slower than GeNN on the SAME RTX 4070** (9.817 vs
   0.542). Identical hardware, identical model. The gap is entirely
   dispatch/fusion — GeNN code-generates fused CUDA kernels, PyTorch issues ~30
   separate kernel launches per step. This is direct empirical confirmation that
   the axis we attacked on CPU is the dominant one on GPU too, and *larger*
   there, because a PyTorch op on GPU costs a ~5–10 µs launch rather than just a
   cache pass.
2. **The repo's PyTorch backend is the second-worst of six backends.** Any
   speedup quoted against it is quoting against a weak baseline. `brian2cpp` and
   `genn` are the honest references.
3. **138,639 neurons cannot saturate a modern GPU.** RTX 4070 has 5,888 CUDA
   cores → 23.5 neurons per core, one wave. Compute is ~1.4 MFLOP/step (≈0.05 µs
   at 29 TFLOP/s) and traffic ~3.9 MB/step (≈8 µs at 500 GB/s), yet GeNN needs
   54 µs/step. So ≈85% of even the best GPU backend is launch/sync latency, not
   work. This is why the GPU's ~100× nominal FLOPS advantage yields only ~6×
   (3.31 → 0.542) at batch 1.
4. **Batching confirms the latency diagnosis.** GeNN improves 3.9× per-trial
   from n=1 to n=8 (0.542 → 0.140) purely by amortising launches, then degrades
   at n≥16 (capacity limit). Backends that were never launch-bound
   (Brian2CUDA, NEST GPU, Brian2GeNN) are flat under batching, as expected.

### Strategic consequence

Two different goals need two different machines, and they should be stated
separately:

- **Real-time whole brain on a consumer device** → the CPU path. 3.31 s/sim-s
  single-threaded now; 4-core threading is the obvious next step and should give
  ~2.5–3.5×, i.e. ≈1.0 s/sim-second. No discrete GPU needed, which is a *better*
  consumer story than requiring an RTX card.
- **Large-scale perturbation studies** (single-neuron ablation sweeps, the
  interpretability use case) → GPU plus batching. GeNN at n=8 is 0.140
  s/sim-second/trial, so 138,639 one-second ablations ≈ 5.4 h on one RTX 4070.

A CUDA port of this kernel only adds value if it beats GeNN. Given GeNN is ~85%
latency, CUDA graphs + the sparse delay line + the bitset spike vector plausibly
reach 10–20 µs/step (0.1–0.2 s/sim-second) — but that is a projection, not a
measurement, and Rule 2 of this project says measure.

## Actions

1. Rewrite HANDOFF §8 with the corrected table and delete the
   "their compromise is our opportunity" paragraph.
2. **Change the baseline.** Benchmarking against the repo's PyTorch backend
   flatters us: Brian 2's C++ standalone codegen is only 1.33× behind our
   AVX-512 kernel and 5.8× ahead of the repo's optimised PyTorch. Build a
   Brian 2 standalone run of the same protocol on this machine and make it the
   reference. `code/compare_ground_truth.py` already exists for the accuracy
   half of this.
3. Re-target: the goal is 0.0538 s/sim-second, needing ~62× beyond today.
   Multithreading (4 cores, ~3.5×) plus the activity-dependent paths gets part
   of the way; the gap is large and should be stated as such.
4. Add a units check to the benchmarking protocol. This error survived a full
   session and a HANDOFF rewrite because no one re-derived it against a
   sanity bound (0.44 ms/step for compiled C on 138k neurons is reasonable;
   442 ms/step is not).
