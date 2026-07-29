# Accelerating the whole-brain Drosophila LIF model — literature review & plan

Status 2026-07-29. Model: Shiu et al. LIF on FlyWire v783, 138,639 neurons,
15.1M synapses, dt=0.1 ms, uniform 1.8 ms delay, single exponential τ_syn=5 ms.

---

## 1. What we measured (not from literature)

| Fact | Value | How |
|---|---|---|
| dense neuron update : synaptic propagation | **728 : 1** | op count/step |
| spikes per timestep (sugar) | 1.75 (0.00126% of N) | instrumented run |
| neurons at **exact** rest after 500 ms | **91.2%** | state probe |
| 4-hop reachability from 21 GRNs | 96.8% | graph BFS |
| arithmetic intensity | ~0.3 FLOP/byte | ~10 FLOP / ~32 B |
| **current bottleneck** | **68% PyTorch dispatch** | cProfile, ~45 µs × ~30 ops/step |

Two consequences:
- **Synaptic propagation is 0.14% of runtime.** Optimizing spike delivery
  (GeNN's core strength, sparse-matmul libraries, synapse pruning) is nearly
  worthless *for this workload*. This is the counterintuitive headline.
- Sparsity is **stimulus-dependent**. 91% quiescence holds for localized drive
  (sugar, P9), not for broad stimulation. Every claim must be checked across
  protocols, not just the demo.

Reachability (96.8% within 4 hops) proves the quiescence is **dynamical
attenuation**, not topological isolation.

---

## 2. Literature — what applies, what doesn't

### Directly applicable

- **SparseProp** — Engelken, NeurIPS 2023. https://arxiv.org/abs/2312.17216
  Coordinate transform so only threshold/reset move; per-neuron decay never
  executes. >4 orders of magnitude on 1M-neuron sparse LIF.
  *Published form excludes us* (delta synapses, no delays, no Poisson) **but the
  transform generalizes**: with one uniform τ_syn store `ĝ=g·e^(t/τ)` plus a
  single global `e^(−t/τ)`. Overflow → periodic renormalization.
  → **This is the direct fix for our dispatch bottleneck.**

- **Bullet Trains** — Morrill, Pehle, Zador, ICML 2026. https://arxiv.org/abs/2603.13283
  Sub-threshold LIF transitions as **affine maps** that compose associatively →
  Blelloch scan. Hard reset kept **exact** via speculative execution
  (scan assuming no spike, analytically test for crossing, roll back). 44×.
  → Validates the math we want. Event-chunked and training-focused, *not* a
  fixed-Δt simulator, and does not use delay as the block.

- **FPT** — Feng et al., ICML 2025. https://arxiv.org/abs/2506.12087
  Fixed-point iteration for LIF: O(T) → O(K), **K≈3**. Handles reset by
  iterating it to a fixed point rather than removing it — the exactness-friendly
  alternative to speculation.

- **Exact propagator** — Rotter & Diesmann, Biol. Cybern. 1999.
  https://link.springer.com/article/10.1007/s004220050570
  Matrix exponential advances linear LIF+exp-synapse **exactly** over arbitrary
  Δt. Legalises lazy catch-up: store `t_last`, jump on event. Exact, not approximate.

- **Brette 2006** — Neural Computation 18(8). https://pubmed.ncbi.nlm.nih.gov/16771661/
  Exact event-driven IF with **exponential conductances, single τ** — our model.
  (2007 sequel lifts single-τ via polynomial root-finding.)

- **Mattia & Del Giudice 2000** — https://pubmed.ncbi.nlm.nih.gov/11032036/
  Minimum synaptic delay = causality horizon. Our 1.8 ms = **18 steps of
  guaranteed lookahead**; events inside a window are order-independent.

- **ADSEQ 2025** — https://arxiv.org/abs/2512.05906
  Empirical: CPU favours tree/FIFO queues, GPU favours ring buffers. With a
  *uniform* delay a fixed 18-slot ring buffer is provably optimal — **no priority
  queue at all**. (We already implement exactly this.)

### Context / partial fit
- Bautembach et al., HPEC 2021 (GPU lazy + work queues, 1.5–2.5×) https://arxiv.org/abs/2107.04092
- Magalhães et al., ICCS 2020 (async variable timestep, 24.6–228.5×, built for HH — take the asynchrony idea, not the implicit solver) https://arxiv.org/abs/1907.00670
- Cessac et al. 2008 (analytic bounds to prune event processing) https://arxiv.org/abs/0810.3992
- Kunkel et al. 2011 (fail-safe threshold-crossing cascade; miss prob ≤2.3e-4, **worst in our regime**) https://pmc.ncbi.nlm.nih.gov/articles/PMC3240333/
- Hanuschkin et al. 2010 — states event-driven *outperforms* hybrid in sparse/low-rate regimes, i.e. ours. https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2010.00113/full

### Deliberately rejected
- **Parareal / MGRIT** — no SNN application exists, but converges poorly on
  discontinuous systems and spike-reset is a hard discontinuity. Note the gap in
  a paper; do not build on it.
- **PSN / PSU / SPSN / DSN / SpikingSSMs** — all achieve parallelism by
  *removing or approximating the reset* (stochastic firing, probabilistic reset,
  learned surrogate reset, dynamic decay). Fine for ML accuracy, **fatal for
  simulation fidelity**. Do not adopt.
- **Procedural connectivity** (Knight & Nowotny 2021) — solves memory, not our
  problem; 15M synapses already fit. Revisit only at full 50M scale.
- **Synapse pruning / compression / faster spmv** — attacks the 0.14%. Trap.

### Novelty gaps found (no hits across multiple query formulations)
1. **Uniform synaptic delay as the parallel-scan block length** ← strongest claim.
   The `d_min` literature uses it for *spatial* decoupling across MPI ranks; the
   scan literature chunks by *event count*. The synthesis is unpublished — and is
   **stronger than Bullet Trains' speculation**, because within our window
   non-interaction is *provable*, so no rollback is needed for cross-neuron
   effects (only for a neuron's own self-reset).
2. Parallel scan inside a fixed-Δt **simulator** (all scan work is ML training).
3. **Two-state (V,G)** conductance scan — all scan papers use scalar-V.
4. Exactness-preserving parallel simulation — only Bullet Trains qualifies.

Independent confirmation the headroom is unclaimed: the reference Shiu et al.
implementation (https://github.com/philshiu/Drosophila_brain_model) is
clock-driven Brian2.

---

## 3. Plan

Ordered by (impact × confidence) / effort. **Every step must pass
`flyloop/verify.py` (bit-identical spikes across 8 regimes) before landing.**

### Phase 1 — kill the dispatch bottleneck (proven, low risk)
1. **State packing.** Pack V, G, refrac, refrac_steps into one `(4,N)` tensor →
   1 gather + 1 scatter instead of 4–5 each. Pure engineering, exact.
   *Expect ~1.5–2× on the active path.*
2. **SparseProp global rescale.** Carry `ĝ=g·e^(t/τ)` and one global scalar;
   per-neuron decay disappears. Renormalize every ~2000 steps.
   ⚠ Renormalization perturbs rounding — **must quantify whether bit-exactness
   survives**; if not, report as "exact to N ulp" with a drift bound, not "exact".
3. **Exact exponential decay** `exp(−dt/τ)` instead of Euler `1−dt/τ`. Pairs
   naturally with (2), which needs the true exponential anyway. Enables larger dt
   later. ⚠ Changes numerics vs upstream → coordinate across all backends.

### Phase 2 — the novel contribution
4. **Delay-window parallel scan.** Block = 18 steps (uniform delay, provable
   non-interaction). Associative scan over **(V,G) 2×2 affine maps** (Bullet
   Trains math extended to two states — gap #3). Intra-neuron self-reset via
   **FPT fixed-point iteration, K≈3**, or Bullet Trains speculation; prefer FPT
   for exactness.
   Also amortizes dispatch ~18× — same fix as Phase 1, different axis, and they
   compose (fewer ops/step × fewer steps/batch).

### Phase 3 — hardening & generality
5. Kunkel 2011 fail-safe cascade if we stay hybrid rather than fully event-driven.
6. Heterogeneous delays: block collapses to *min* delay — quantify the loss.
7. GPU port. ⚠ Scout warns active-set compaction may cost more than a dense
   139k-wide kernel at only ~190 active neurons. **Measure before porting.**

### Validation
- `flyloop/verify.py` — 8 regimes, 1 neuron → whole-brain, bit-identical.
- Extend to every Eon experiment (sugar, P9) **plus silencing**.
- Cross-check against Brian2 ground truth via their `compare_ground_truth.py`.

### Kill risks, ranked
1. **Tonic/Poisson drive to all neurons** destroys the active set — SparseProp
   names this explicitly. We are safe only because Poisson hits 21 neurons.
2. **High-firing protocols** — re-verify spikes/step per protocol.
3. **Heterogeneous delays** would collapse the 18-step lookahead.
4. **True conductance coupling** `g·(V−E)` is nonlinear → exact propagator
   invalid. Our model uses additive current, so we are currently fine; this
   binds if the model is ever made conductance-based.
