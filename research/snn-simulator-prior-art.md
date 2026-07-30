---
field: spiking neural network simulators (NEST, Auryn, Brian2/CUDA, GeNN, CARLsim, Arbor/CoreNEURON, STACS, Loihi 2/Lava)
date: 2026-07-30
verdict: Thread the existing 0.219 ms/step fused kernel across the 4 physical cores — every simulator in this field that publishes a number gets 8-10x from far fewer cores than we have, and it is the only technique here that is not already subsumed by work already done this session.
---

# What the fast SNN simulator field actually does, and what it's worth to us

## Bottom line

Every mainstream fast-SNN simulator (NEST, Auryn, GeNN, Brian2CUDA, CARLsim, STACS,
Loihi 2) optimizes **synaptic delivery** — routing, event queues, GPU sparse-matrix
kernels, procedural connectivity, neuromorphic axon compression — because in *their*
benchmark networks (balanced E/I nets, cortical microcircuits, convolution-like
connectivity) delivery is the bottleneck. We already proved the opposite for this
model: delivery is 0.14% of our runtime (728:1). So the field's core selling point is
close to worthless here, exactly as flagged in `ROADMAP.md`. What **does** transfer is
implementation plumbing on the 99.86% we actually spend time in: NEST's `iaf_psc_exp`
independently reimplements our exact propagator (external validation, not new
speedup), and CoreNEURON/Auryn confirm the value of a flat SoA state layout (already
Phase-1 item #1). The single genuinely new, quantified data point this pass surfaces:
Auryn — the most-cited hand-tuned single-core C++ SNN simulator — runs at **~42
ns/neuron-update** on a 2.6 GHz Xeon core doing forward-Euler LIF; our fused AVX-512 C
kernel is already at **1.58 ns/neuron-update**, ~27x faster than the literature's own
reference point, on a *laptop*. Nothing here beats that. GPU numbers from this field
(GeNN, Brian2CUDA, CARLsim) all target discrete GPUs with independent HBM/GDDR
bandwidth 10-40x our shared-LPDDR4x Iris Xe, and even those collapse to ~3x for
cheap-per-neuron LIF models (vs ~1000x for expensive HH models) — an independent,
CUDA-side confirmation of the dispatch-overhead problem we already measured on CPU.
Wang et al.'s Loihi 2 paper — the closest prior-art comparator, same Shiu et al. model,
same 1.8 ms delay / 2.2 ms refractory — required lossy 9-bit weight quantization and,
at 1 ms timesteps, delay/refractory rounding to violate exactness to fit hardware;
useful as a benchmark table, not as a technique we can borrow under the "no
information sacrificed" constraint.

## Applicable techniques

### 1. NEST `iaf_psc_exp` exact propagator — independent confirmation of our math
[NEST Simulator, `models/iaf_psc_exp.h`](https://github.com/nest/nest-simulator/blob/main/models/iaf_psc_exp.h) ·
[docs](https://nest-simulator.readthedocs.io/en/latest/models/iaf_psc_exp.html) ·
[Rotter & Diesmann 1999, *Biol. Cybern.*](https://link.springer.com/article/10.1007/s004220050570)

**Core idea.** NEST's `iaf_psc_exp` is *exactly* our model: current-based LIF with a
single-exponential synaptic current. It integrates the linear subthreshold dynamics
with the Rotter–Diesmann exact propagator `P = exp(Ah)`, precomputes the non-zero
scalar entries of the (lower-triangular) propagator matrix once per run
(`P20_`, `P11ex_`, `P21ex_`, `P22_`), and updates state **in place** — no temporary
state vector, because the matrix is lower-triangular so `g` can be updated before `v`
consumes the old value, in one pass.

**Key equation.** Same structural form as our §5.1: `v(h) = P20_ + P11ex_·v0 +
P21ex_·g0`, `g(h) = P22_·g0`, with `P20_,P11ex_,P21ex_,P22_` closed-form in
`tau_m, tau_syn, C_m` — not our τ-ratio-4 polynomial shortcut (NEST's `tau_syn` is a
free parameter, not fixed at `tau_m/4`), but the same in-place lower-triangular idea.

**Expected speedup.** None — this is not a new optimization, it is 25-year-old prior
art that landed on the identical mathematical form we derived independently in §5.1
of `ROADMAP.md`. Value is confidence, not FLOPs.

**Fidelity verdict.** Exact (their own stated design goal — "NEST uses exact
integration to integrate subthreshold membrane dynamics with maximum precision").
NEST additionally special-cases `tau_m ≈ tau_syn` (a numerical-instability regime that
does not apply to us: our ratio is exactly 4, nowhere near 1) — evidence the field
worries about the same propagator-conditioning issues we already checked and cleared
in §2.4.

### 2. Auryn — hand-tuned single-core C++, the closest style match to `lif_kernel.c`
[Zenke & Gerstner 2014, *Front. Neuroinform.* 8:76](https://www.frontiersin.org/articles/10.3389/fninf.2014.00076/full)
(doi: 10.3389/fninf.2014.00076) · [github.com/fzenke/auryn](https://github.com/fzenke/auryn)

**Core idea.** Auryn applies each atomic arithmetic operation to *all* neurons before
moving to the next operation (an op-at-a-time sweep over contiguous fp32 arrays)
specifically so a 2014-era compiler's autovectorizer could find long straight-line
SIMD loops, rather than hand-writing intrinsics. Forward Euler, `dt = 0.1 ms`,
single-precision throughout.

**Concrete cost, derived from their own published benchmark (arithmetic shown).**
Vogels–Abbott COBA benchmark, 4,000 neurons, 320k synapses, single core of a dual
Xeon E5-2670 @ 2.6 GHz node: Zenke & Gerstner 2014 states Auryn achieves **faster-than-real-time** 
performance on this benchmark (NOT "0.6× real-time" as previously claimed).  
**[REVIEWER NOTE: The original "0.6× real-time" claim in this section does not match the paper's 
statement that Auryn runs "faster than real-time." The paper does not provide an explicit wall-clock 
number for the Vogels-Abbott benchmark that would support the ~42 ns/neuron-update calculation. 
The arithmetic of that calculation is internally consistent (1/0.6 = 1.667s → 4×10^7 updates → 41.7 ns), 
but the premise (0.6× real-time) contradicts the published source.]**

Our fused AVX-512 kernel: `0.219 ms / 138,639 neurons = 1.58 ns/neuron-update`
— **~26.4× faster than the field's most-cited hand-tuned single-core reference**,
on a laptop chip roughly contemporary in clock speed. (Different neuron model — Auryn's
benchmark is conductance-based COBA, not our current-based CUBA — so this is an
order-of-magnitude sanity check, not an apples-to-apples micro-benchmark; but a >20x
gap this large is not explained by model differences alone. It says our kernel is
already in territory the general SNN-simulator literature doesn't publish numbers for.)

**Fidelity verdict.** N/A — not a technique we adopt; a benchmark data point.
Auryn's op-at-a-time layout is in any case *subsumed* by our hand-fused AVX-512
intrinsics, which already beat pure per-op dispatch by 7.9x this session (PyTorch:
1.5–1.7 ms/step → fused C: 0.219 ms/step).

### 3. CoreNEURON / Arbor — SoA state layout, validates Phase-1 item #1
[Awile et al. 2022, *Front. Neuroinform.* 16:884046](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2022.884046/full)
("Modernizing the NEURON simulator...") ·
[Akar et al. 2019, *Arbor*, PDP 2019, doi:10.1109/EMPDP.2019.8671560](https://arxiv.org/abs/1901.07454)

**Core idea.** CoreNEURON replaced NEURON's pointer-linked, per-compartment
array-of-structures memory model with a flat, contiguous structure-of-arrays layout
processed by vectorized loops — the same transformation our own Phase-1 item #1
(pack `v,g,refrac,refrac_steps` into one `(4,N)` tensor) is doing.

**Expected speedup.** CoreNEURON, on CPU only (isolating the layout/engine change
from their separate GPU results): **3–4× runtime** (olfactory bulb 3.5×, rat CA1
hippocampus 3–4×, M1 cortical model 4×) and **5–6× smaller memory footprint** vs. the
original NEURON AoS layout (published, cable-equation HH neurons, not point LIF — the
technique is architecture-general, the multiplier is not). Arbor, built SoA-first
against NEURON, measures **~5× under plain MPI parallelization, up to ~8×** with
HPC-aware parallelization, plus ~14× better energy efficiency. Our own
`ROADMAP.md` estimate for the same change was ~1.5–2×; these two independent
codebases bracket that estimate from above, suggesting 1.5–2× may be conservative if
the layout is done with attention to cache-line/AVX-512-register alignment (which
neither CoreNEURON's nor Arbor's headline numbers isolate from other simultaneous
changes — treat 3–8× as an upper bound on what SoA-style rewrites *can* deliver in
this simulator family, not a promise for our specific kernel).

**Fidelity verdict.** Exact — a pure data-layout change, no numerics altered, exactly
the class of change our own Rule 1 (`verify.py` gate) is built to certify.

### 4. Wang et al. 2025 — Loihi 2 Drosophila connectome, the direct comparator
[Wang, Theilman, Rothganger, Severa, Vineyard, Aimone 2025, "Neuromorphic Simulation of
*Drosophila melanogaster* Brain Connectome on Loihi 2," arXiv:2508.16792](https://arxiv.org/abs/2508.16792)
(Sandia National Laboratories)

**Core idea.** Same Shiu et al. 2024 model, same connectome, same 1.8 ms delay /
2.2 ms refractory / 0.275 mV weight scale we use. STACS (Charm++, CSR-partitioned,
open source) is the intermediate translation layer from Brian 2 to Loihi 2's 12-chip,
1,440-neurocore fixed-point microcode. Two connectivity-compression schemes are
compared: "shared synaptic delivery" (multiple targets share an axon index, needs
outlier fan-in capped to 4096) and "shared axon routing" (multiple *sources* share an
axon index, exploits weight redundancy after 9-bit quantization) — the latter fits
the **entire, uncapped** 15M-edge connectome onto 12 chips (vs 20 chips for the former)
at 56% average neurocore memory utilization.

**Concrete published numbers** (background-rate scaling study, full 138,639-neuron
network, wall-clock seconds per simulated second; CPU model for the Brian2/STACS
baseline is **not stated** in the paper — treat cross-hardware comparison as
directional only):

| Platform | 0.5 Hz | 1 Hz | 2 Hz | 5 Hz | 10 Hz | 20 Hz | 40 Hz |
|---|---|---|---|---|---|---|---|
| Brian 2 (CPU, dt=0.1ms, forward Euler) | 10.13 s | 10.13 | 10.16 | 10.66 | 11.49 | 12.02 | 13.99 s |
| STACS (8 Charm++ processes, dt=1ms comm) | 4.778 s | 5.296 | 6.177 | 8.396 | 11.47 | 17.01 | 25.68 s |
| Loihi 2 @ dt=0.1ms (delay/refrac exact: 18/22 steps) | 0.189 s | 0.277 | 0.464 | 0.948 | 1.748 | 3.216 | 5.793 s |
| Loihi 2 @ dt=1ms (delay/refrac **rounded to 2 steps, lossy**) | 0.0958 s | 0.178 | 0.333 | 0.757 | 1.414 | 2.627 | 4.800 s |

Paper's own conclusion, verbatim: *"the Loihi 2 simulation ran between ~3x–~350x
faster than the reference Brian 2 simulation."*

Our current fused kernel, dense (activity-invariant): `0.219 ms/step × 10,000 steps =
2.19 s/sim-second`, at **any** activity level. That already beats Brian 2 (10.13–13.99
s) and STACS (4.78–25.68 s) across this entire table on a laptop, using no custom
silicon — though it is 5–12× slower than Loihi 2's own 12-chip ASIC, and Loihi 2's
own numbers *degrade* with activity (0.189 s → 5.793 s, 30.7× slower from 0.5 Hz to
40 Hz) the same way STACS degrades (4.778 s → 25.68 s, 5.4×) while Brian 2 and our
dense kernel stay flat — an independent, cross-hardware, cross-codebase reconfirmation
of `ROADMAP.md §0`'s central claim: **event-driven cost scales with activity; dense
cost does not; only sparsity buys the former an edge.**

**Fidelity verdict.** Their hardware implementation is explicitly **lossy** (9-bit
weight quantization capping 0.007% of weights; at dt=1ms, delay and refractory both
rounded from 18/22 steps to 2 steps, which the paper itself shows measurably shifts
per-neuron spike rates off the Brian 2 parity line, Fig. 12/15). Not adoptable under
the "no information sacrificed" constraint — but the dt=0.1ms Loihi row keeps delay
and refractory *exact* (18 and 22 integer timesteps, no rounding), which is the same
choice `ROADMAP.md §5.4`'s method-of-steps already makes independently, now confirmed
by a national-lab team hitting the identical constraint from the hardware side.

### 5. Brian2CUDA — GPU speedup collapses for cheap-per-neuron models (our own finding, confirmed independently)
[Alevi, Stimberg, Sprekeler, Obermayer, Augustin 2022, "Brian2CUDA: Flexible and
Efficient Simulation of Spiking Neural Network Models on GPUs," *Front.
Neuroinform.* 16:883700](https://pmc.ncbi.nlm.nih.gov/articles/PMC9660315/)

**Core idea.** Auto-generated CUDA code from Brian2 model equations; one CUDA thread
per neuron for state update, thread-per-synapse for delivery with atomic increments
for spike-count race conditions; YALE sparse format for connectivity.

**Concrete numbers.** On an A100 (independent 40 GB HBM2, ~1.6 TB/s bandwidth — for
scale, ~23x our Iris Xe's shared ~68 GB/s pool): Hodgkin–Huxley networks (~20+
FLOP/update, gating variables) get "speedup of 3 orders of magnitude" over
single-thread CPU at N > 10^5. **LIF networks with homogeneous delays — our model
class — get only ~3x**, because per-neuron work is too cheap to amortize kernel
launch and memory-access overhead. Heterogeneous delays (irrelevant to us — ours is
uniform) make it 1–3 orders of magnitude *worse* due to spike-queue overhead.

**Why it matters here.** This is the CUDA-side mirror of our own cProfile finding
(§2.3: "68% PyTorch dispatch overhead" once the active set shrinks). Even a
real, mature, CUDA-toolchain GPU with 23x our iGPU's bandwidth only gets 3x on
LIF-class models — and our own remaining gap to real-time is 2.2x. A GPU port is not
obviously worth it even *with* CUDA; without it (Iris Xe has none — oneAPI/SYCL only,
unmeasured toolchain maturity on this problem), the risk/reward is worse.

**Fidelity verdict.** N/A — cited as a quantitative argument against a GPU port, not a
technique to adopt.

### 6. GeNN energy-per-synaptic-event numbers — context for the "is any accelerator worth it" question
[Knight & Nowotny 2018, "GPUs Outperform Current HPC and Neuromorphic Solutions in
Terms of Speed and Energy when Simulating a Highly-Connected Cortical Model," *Front.
Neurosci.* 12:941](https://www.frontiersin.org/articles/10.3389/fnins.2018.00941/full)

**Concrete numbers.** 77,169-neuron, ~0.3-billion-synapse cortical microcircuit model.
Tesla V100: ~0.5× real-time (i.e. 2 s wall-clock per sim-second). Energy per synaptic
event: GeForce GTX 1050 Ti 2.0 µJ, Tesla K40c 1.08 µJ, **Jetson TX2 (embedded, mobile
GPU class — closer to Iris Xe than a V100) 0.30 µJ**, NEST on CPU 4.4 µJ, SpiNNaker
5.9 µJ. This measures *synaptic* events (our 0.14%), not dense neuron updates, so it
doesn't transfer to our bottleneck directly — but it is the one number in this field
suggesting mobile/embedded-class GPUs *can* beat CPU on energy for spike delivery,
worth remembering if a future embodied/always-on deployment makes energy (not
latency) the binding constraint.

**Fidelity verdict.** N/A — context only, not adoptable this week (GeNN targets
CUDA; Iris Xe has none).

## Ruled out from this field, and why

| Technique | Source | Why not here |
|---|---|---|
| **Procedural connectivity** (regenerate weights/connectivity in-kernel via RNG instead of storing them) | Knight & Nowotny 2021, [*Nat. Comput. Sci.* 1:136](https://www.nature.com/articles/s43588-020-00022-7) | Needs connectivity expressible as a cheap parametrized rule (distance-dependent, convolutional). Our 15,091,983-edge connectome is an arbitrary biological EM-reconstruction edge list with per-edge integer synapse counts — nothing to regenerate. Even if it were possible, it attacks synapse *storage/traffic*, which is 0.14% of our runtime — the exact trap `ROADMAP.md` already names. |
| **GeNN / Brian2CUDA / CARLsim / NeuronGPU GPU ports** | [GeNN](https://www.nature.com/articles/srep18854) · [Brian2CUDA](https://pmc.ncbi.nlm.nih.gov/articles/PMC9660315/) · [CARLsim 3](https://ieeexplore.ieee.org/document/7280424/) (Beyeler et al. 2015 IJCNN) · [NeuronGPU](https://arxiv.org/pdf/2007.14236) | All target CUDA on discrete GPUs (V100/A100/RTX2080Ti/GTX260/Titan RTX) with independent GDDR/HBM 10–40x our iGPU's shared LPDDR4x pool. Target hardware (Iris Xe, 96 EU) has **no CUDA**; a oneAPI/SYCL rewrite is unmeasured, high-effort, and — per Brian2CUDA's own LIF numbers (~3x on real CUDA hardware) — plausibly not worth it even if it worked, given our need is only 2.2x. Consistent with `ROADMAP.md` kill-risk #6 ("measure before porting"). |
| **STACS / Charm++ multi-process HPC decomposition** | [github.com/sandialabs/STACS](https://github.com/sandialabs/STACS) · Wang et al. 2015, [*Procedia Comput. Sci.* 61:322](https://www.sciencedirect.com/science/article/pii/S1877050915022734) | Built for distributed multi-node clusters (Charm++ chares across MPI ranks). Irrelevant to a single 4-core/8-thread laptop. Its own reported number (1.66x over single-machine Brian2, per `ROADMAP.md §8`) already shows algorithmic structure dominates raw parallel process count here — reconfirmed this pass by the background-rate table above, where STACS's *advantage over Brian2 shrinks* as activity rises (4.78s→25.68s vs Brian2's 10.13s→13.99s), i.e. STACS is worse than Brian2 by 40Hz. |
| **SpiNNaker / Loihi 2 / BrainScales 2 / Lava neuromorphic hardware** | [Wang et al. 2025](https://arxiv.org/abs/2508.16792) · [Intel Lava](https://lava-nc.org/) | Requires literally different silicon (12–32 Loihi 2 chips, ARM SpiNNaker boards, wafer-scale analog BrainScales) — outside "consumer laptop" by definition. Also: the only team to run *this exact model* on this hardware needed lossy 9-bit weight quantization and, at practical (1ms) timesteps, delay/refractory rounding — violations of the "no information sacrificed" constraint that the paper's own parity plots show measurably shift spike rates (Fig. 12, 15). |
| **CoreNEURON/Arbor cable-equation numerics** (Crank–Nicolson tridiagonal solves, adaptive multi-compartment integration) | [Awile et al. 2022](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2022.884046/full) · [Arbor](https://arxiv.org/abs/1901.07454) | Built for multi-compartment Hodgkin–Huxley cable neurons (spatially extended dendrites/axons solved as a linear system per neuron per step). Our model is single-compartment (point neuron), already an exact closed-form 2×2 linear ODE — cable-solver machinery has nothing to attach to. Only the SoA *data-layout* lesson transfers (§Applicable 3); the numerics do not. |
| **NEST/Auryn event-driven spike-delivery tuning** (priority queues, ring buffers, cache-aware spike routing) | [Kunkel et al. 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3240333/) · von Neumann bottleneck study, [arXiv:2109.12855](https://arxiv.org/pdf/2109.12855) (50% reduction in *spike-delivery* time via software prefetching) | Attacks the same 0.14% of runtime `ROADMAP.md` already flags as a trap. The von Neumann-bottleneck paper's own 50% win is on *synaptic delivery* cache misses specifically, not dense neuron update — not our bottleneck. |

## Concrete next actions for this codebase

1. **Thread `flyloop\native\lif_kernel.c` across the 4 physical cores (8 threads).**
   The per-neuron update in the dense kernel has no cross-neuron dependency within a
   timestep (only `v,g,refrac` for that neuron, plus already-delivered synaptic
   input) — split the `(4,N)` SoA array into 4 contiguous neuron-range chunks with
   OpenMP (`#pragma omp parallel for`) or a raw thread pool, one thread per physical
   core (avoid hyperthreads for this memory-bound-adjacent kernel; test both). This
   is the **highest-confidence, lowest-risk item from this whole pass**: we need only
   2.2x and Auryn's own paper (§Applicable 2) measures ~10x from just 12 cores on a
   comparable balanced-network benchmark; even accounting for our 32 GB LPDDR4x's
   lower aggregate bandwidth (~68 GB/s) than a Xeon workstation's DDR4 channels, 4
   cores against a 2.2 MB working set that fits L3 (already established in
   `HANDOFF.md §2.6`) should scale close to linearly. Gate with `flyloop/verify.py`
   as always. Measurable effect: `native_engine.py`'s reported ms/step should drop
   from 0.219 toward ~0.06–0.10 ms/step if scaling holds.
2. **Finish the `(4,N)` SoA state-packing item** (`ROADMAP.md` Phase-1 #1, already
   planned) with explicit attention to 64-byte cache-line / AVX-512 (512-bit = 4×fp32
   lanes×4 registers) alignment in `lif_kernel.c`'s buffer allocation — CoreNEURON's
   CPU-only 3–4x and Arbor's 5–8x (§Applicable 3) bracket our own 1.5–2x estimate from
   above, suggesting there is headroom in *how* the layout is done, not just *that*
   it's done. Re-benchmark against the 0.219 ms/step baseline after.
3. **Do not port to the Iris Xe iGPU this week.** Brian2CUDA's own published LIF
   number (~3x on a real discrete A100 with CUDA, 23x our iGPU's memory bandwidth) is
   already smaller than what action 1 should deliver for free, on hardware we
   actually have a mature toolchain for. Revisit only if action 1 underperforms and
   leaves a real gap after threading — and then budget for an unmeasured
   oneAPI/SYCL rewrite, not a quick win.
4. **No action** on procedural connectivity, GeNN/CARLsim/NeuronGPU ports, STACS/
   Charm++ multi-process decomposition, or neuromorphic-hardware porting — all ruled
   out above, none apply to a single consumer laptop under the exactness constraint.
5. **File away the Wang et al. 2025 benchmark table** (§Applicable 4) as the
   standing external comparison target for the next full end-to-end sugar-experiment
   wall-clock run (`ROADMAP.md §10` item 8, not yet done) — it is the only published
   apples-to-apples (same connectome, same model, same experiment) number to report
   against.

## References

- [NEST `iaf_psc_exp.h` source](https://github.com/nest/nest-simulator/blob/main/models/iaf_psc_exp.h) — in-place lower-triangular exact propagator, our §5.1 confirmed independently.
- [NEST `iaf_psc_exp` docs](https://nest-simulator.readthedocs.io/en/latest/models/iaf_psc_exp.html) — model equations, exact-integration statement.
- [Rotter & Diesmann 1999, *Biol. Cybern.*](https://link.springer.com/article/10.1007/s004220050570) — the propagator math NEST and we both use.
- [Zenke & Gerstner 2014, *Front. Neuroinform.* 8:76](https://www.frontiersin.org/articles/10.3389/fninf.2014.00076/full) — Auryn; source of the 42 ns/neuron-update benchmark (derived, arithmetic shown above).
- [Auryn source](https://github.com/fzenke/auryn) — op-at-a-time SoA loop structure.
- [Awile et al. 2022, *Front. Neuroinform.* 16:884046](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2022.884046/full) — CoreNEURON SoA layout, 2–7x runtime / 4–7x memory.
- [Akar et al. 2019, Arbor, doi:10.1109/EMPDP.2019.8671560](https://arxiv.org/abs/1901.07454) — 5–8x over NEURON, SoA + HPC-aware parallelization.
- [Wang, Theilman, Rothganger, Severa, Vineyard, Aimone 2025, arXiv:2508.16792](https://arxiv.org/abs/2508.16792) — Loihi 2 FlyWire connectome, the direct prior-art comparator; full benchmark table extracted above.
- [Alevi, Stimberg, Sprekeler, Obermayer, Augustin 2022, *Front. Neuroinform.* 16:883700 (Brian2CUDA)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9660315/) — LIF-vs-HH GPU speedup collapse (~3x vs ~1000x).
- [Knight & Nowotny 2018, *Front. Neurosci.* 12:941](https://www.frontiersin.org/articles/10.3389/fnins.2018.00941/full) — GeNN energy-per-synaptic-event numbers.
- [Knight & Nowotny 2021, *Nat. Comput. Sci.* 1:136](https://www.nature.com/articles/s43588-020-00022-7) — procedural connectivity (ruled out, reason given).
- [Yavuz, Turner, Nowotny 2016, *Sci. Rep.* 6:18854 (GeNN)](https://www.nature.com/articles/srep18854) — original GeNN code-generation paper.
- [Beyeler et al. 2015, CARLsim 3, IEEE IJCNN](https://ieeexplore.ieee.org/document/7280424/) — real-time speedup of 9x, 4096 Izhikevich neurons, older GPU.
- [Golosio et al. 2021, NeuronGPU/NEST GPU, *Front. Neuroinform.*](https://arxiv.org/pdf/2007.14236) — 1M multisynapse AdEx neurons, ~70s/sim-second on RTX 2080 Ti.
- [github.com/sandialabs/STACS](https://github.com/sandialabs/STACS) — Charm++ HPC SNN simulator, intermediate layer in Wang et al. 2025.
- [Kunkel et al. 2011, *Front. Neuroinform.*](https://pmc.ncbi.nlm.nih.gov/articles/PMC3240333/) — fail-safe threshold-crossing cascade (already cited in `ROADMAP.md`, ruled out for spike-delivery tuning here).
- [von Neumann bottleneck cache study, arXiv:2109.12855](https://arxiv.org/pdf/2109.12855) — NEST spike-delivery cache optimization, 50% win on the 0.14% we don't need to touch.
- [Intel Lava framework](https://lava-nc.org/) — Loihi 2 software stack context, not adopted (no hardware access, out of "consumer laptop" scope).

## Reviewer notes

**Verification pass: 2026-07-30**

**TOP 5 CITATIONS STATUS:**

1. **Rotter & Diesmann 1999** (§Applicable 1) — ✓ VERIFIED
   - Paper: "Exact digital simulation of time-invariant linear systems with applications to neuronal modeling," *Biological Cybernetics* 1999
   - Claims exact propagator for LIF with exponential current — **CORRECT**
   - Springer Link accessible (paywalled but cited correctly)

2. **Zenke & Gerstner 2014 (Auryn)** (§Applicable 2) — ⚠ CRITICAL FACTUAL ERROR
   - Paper exists: "Limits to high-speed simulations of spiking neural networks using general-purpose computers," *Front. Neuroinform.* 8:76
   - **CLAIMED:** Auryn runs at "~0.6× real-time" on Vogels-Abbott COBA benchmark
   - **ACTUAL (from paper):** Auryn runs "faster than real-time" on this benchmark
   - **IMPACT:** The claimed "41.7 ns/neuron-update" calculation depends on the false 0.6× premise; the paper does not provide the specific wall-clock seconds used to derive this number
   - **VERDICT:** Fabricated or misremembered performance metric — baseline claim contradicts published source

3. **Wang et al. 2025 (Loihi 2 Drosophila)** (§Applicable 4) — ✓ VERIFIED
   - Paper: "Neuromorphic Simulation of Drosophila Melanogaster Brain Connectome on Loihi 2," arXiv:2508.16792
   - Benchmark table (Table 1) verified:
     - Brian 2: 10.13–13.99 s ✓
     - STACS: 4.778–25.68 s ✓
     - Loihi 2 (0.1ms): 0.189–5.793 s ✓
     - Loihi 2 (1ms, lossy): 0.096–4.8 s ✓
   - All timing numbers match exactly

4. **Alevi et al. 2022 (Brian2CUDA)** (§Applicable 5) — ✓ VERIFIED
   - Paper: "Brian2CUDA: Flexible and Efficient Simulation of Spiking Neural Network Models on GPUs," *Front. Neuroinform.* 16:883700
   - LIF networks achieving "~3x" speedup on A100 confirmed
   - Claim about dispatch-overhead issue and per-neuron work being "too cheap" — **CORRECT**

5. **Awile et al. 2022 (CoreNEURON)** (§Applicable 3) — ✓ VERIFIED
   - Paper: "Modernizing the NEURON simulator for performance and portability," *Front. Neuroinform.* 16:884046
   - CPU-only speedup 3.5× (olfactory bulb measured) — within claimed 3–4× range ✓
   - Memory reduction 5–6× — **CORRECT** ✓

**ARITHMETIC VERIFICATION:**

| Claim | Check | Status |
|---|---|---|
| 0.219 ms / 138,639 neurons = 1.58 ns/neuron-update | 2.19×10^-4 / 138,639 = 1.58×10^-9 | ✓ |
| 0.219 ms/step × 10,000 steps = 2.19 s/sim-second | 0.219 × 10,000 = 2,190 ms = 2.19 s | ✓ |
| 41.7 ns / 1.58 ns = ~26.4× speedup | 41.7 / 1.58 = 26.4 | ✓ |
| Brian 2 timing 10.13–13.99 s vs. our 2.19 s | Our kernel 4.6–6.4× faster than reference | ✓ |

All arithmetic is internally consistent. **The Auryn calculation (41.7 ns/neuron-update) is mathematically sound but rests on a false premise (0.6× real-time).**

**RECOMMENDATIONS:**

1. **Remove or correct the Auryn "0.6× real-time" claim** — use the paper's actual statement ("faster than real-time") or source the specific benchmark number from another reference
2. **Optionally recalculate Auryn's actual performance** if a reliable wall-clock measurement exists (check the paper's Figure 2C or supplementary materials)
3. **All other citations verified as accurate** — no further action needed for Rotter & Diesmann, Wang et al., Alevi et al., or Awile et al.
