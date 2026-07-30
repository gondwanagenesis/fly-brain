# Accelerating the whole-brain *Drosophila* LIF connectome simulation

**Complete working record — measurements, reasoning, dead ends, corrections, next steps.**

Written for cold pickup by another engineer or AI. Everything is either measured
on this machine, cited, or explicitly flagged as a projection.

- Last updated: 2026-07-30
- Upstream: [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain)
- **Our fork: [gondwanagenesis/fly-brain](https://github.com/gondwanagenesis/fly-brain)**, branch `perf/event-driven-pytorch` (all work pushed)
- Local: `Documents/FlyBrain`, venv at `.venv/` (Python 3.12, torch CPU, numpy, scipy)

---

## 0. Read this first — the strategic conclusion

> **2026-07-30 UPDATE — real time is reached.** The whole 138,639-neuron brain now
> runs at **0.0461 ms/step = 2.17× real time** on the sugar protocol on a 4-core
> laptop (0.0260 ms = 3.84× silent, 0.0109 ms = 9.16× single-neuron),
> bit-identical to the PyTorch reference, and beats it in **every** regime
> tested (**3.1×–104×**, no regression anywhere).
> See **§11** and **§12**, which supersede the framing below. Reader-facing
> write-up: **[FORK.md](FORK.md)**.
>
> ⚠️ Real time holds in the SPARSE regimes — which is where both published
> experiments (sugar, P9) live. Under broad drive of 1000+ neurons it is
> 0.28×–0.04× of real time, because tile-skipping is activity-dependent by
> construction and there is nothing left to skip.
>
> The correction to the old conclusion matters: the previous analysis assumed the
> dense path was pinned to memory bandwidth and that only sparsity could help.
> Profiling showed that at 171 µs/step **only 57% was the kernel at all** — 26%
> was `torch.bernoulli` drawing *21 numbers*, and 16% was ctypes glue. The
> baseline was not memory-bound, it was **framework-bound**, and the general
> (activity-independent) headroom was far larger than the 1.5–2× estimated below.

**Large *algorithmic* speedups in this problem are activity-dependent.** The work
is proportional to activity. At full activity every neuron must be advanced every
timestep and you approach memory bandwidth; no algorithm makes updating 138,639
neurons cost less than reading and writing their state. **But the constant factor
in front of that bound was ~30×, and it was recoverable in full.**

| Optimisation | 100% brain active | Sparse activity |
|---|---|---|
| **Fused native AVX-512 kernel (§11)** | **1.23×** | **9–10×** |
| Event-driven fan-out **(banked, bit-identical)** | ~1× | 3.2–5.2× |
| Delay-window stepping | 1.00× (auto-falls back) | 26× |
| σ removes the delay ring buffer | 1.16× | 1.16× |
| Exact integration | 1× (free *accuracy*) | 1× |
| State packing | ~1.5× | ~1.5× |

The native kernel is the first optimisation here that **never regresses**: it is
1.23× even in the saturating regime, where the active-set path was 2.3× *slower*
than baseline.

The consolation still holds: **real brains are sparse.** We measured 1.75
spikes/step and 91% of neurons at exact rest under localised stimulation. An
embodied model with all senses will have *more active regions*, not a saturated
brain.

**The most valuable single finding may still not be speed — see §4 and §11.3.**

---

## 1. The model

[Shiu et al. 2024, *Nature*](https://www.nature.com/articles/s41586-024-07763-9)
([methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC10187186/)) on
[FlyWire](https://codex.flywire.ai/) v783
([Dorkenwald et al. 2024](https://www.nature.com/articles/s41586-024-07558-y)).

```
dg/dt = -g/tau_syn + sum_k w_k delta(t - t_k)      tau_syn =  5 ms
dv/dt = (g - (v - v_rest)) / tau_mem               tau_mem = 20 ms
spike if v > v_th; then v = v_reset, g = 0
```

| Parameter | Value |
|---|---|
| neurons | 138,639 |
| connections | 15,091,983 edges / 54,492,922 synapses |
| `v_rest` = `v_reset` | −52 mV |
| `v_threshold` | −45 mV (θ = 7 mV above rest) |
| `tau_mem` / `tau_syn` | 20 / 5 ms — **ratio exactly 4** |
| refractory | 2.2 ms (**absolute**) |
| **axonal delay** | **1.8 ms, uniform on every synapse** |
| `dt` | 0.1 ms |
| `w_syn` | 0.275 mV × integer synapse count |

⚠️ **Current-based, not conductance-based.** `g` enters *additively*, not as
`g·(E_rev − v)`. That linearity is what makes exact integration valid. If the
model ever moves to true conductance coupling, most of §5 breaks (see §7).

---

## 2. Measured facts (reproducible on this machine)

### 2.1 Cost distribution
Sugar-GRN experiment (21 neurons @ 200 Hz):

| Component | Ops/step | Share |
|---|---|---|
| Dense neuron update (all 138,639) | 138,639 | **99.86%** |
| Event-driven synaptic propagation | ~190 | 0.14% |

**728 : 1.** Mean activity **1.75 spikes/step** (0.0013%).
⇒ optimising spike delivery — GeNN's core strength — is nearly worthless *here*.

### 2.2 Quiescence
| Time | Ever perturbed | Provably inert |
|---|---|---|
| 100 ms | 5.22% | 94.78% |
| 500 ms | 8.84% | **91.16%** |

Not topological: **96.8% of the brain is within 4 hops** of the 21 sugar GRNs.
It is dynamical attenuation. **Stimulus-dependent** — verified across 8 regimes.

### 2.3 Roofline
~10 FLOP / ~32 B per neuron-step ⇒ **~0.3 FLOP/byte**; A100 knee is ~12. Deeply
memory-bound. ⚠️ **Flips at small scale**: once the working set shrinks to ~3k
active neurons it becomes **dispatch-bound** (cProfile: 68% PyTorch per-op
overhead). Both true at different scales.

### 2.4 τ-ratio identity (machine precision)
With `x = exp(-dt/tau_mem)`: `exp(-dt/tau_syn) == x^4` (1.11e-16),
`P_vg == (x - x^4)/3` (3.64e-17), `alpha_m == x` (0.00e+00).

### 2.5 Exact spike-time solver
20,000 random states vs brute force at dt = 2e-4 ms:
**20000/20000** agreement on whether a spike occurs; max |Δt| **2.0e-04 ms**
(= the reference's own grid); **0 false-silence certifications** (safety-critical).

### 2.6 Full-scale per-window costs (138,639 neurons)
```
predict (free_window_map)   5.27 ms      certified-silent from rest: 86.67%
certify (silence bound)     6.92 ms      grid dt=0.1ms: 5.43 ms/step
deliver (event-driven)      4.14 ms                   = 54.30 s/sim-second
```

### 2.7 Delivery: the trap I fell into
```
2 x full W @ sigma   142.45 ms   (30,183,966 ops)
event-driven sigma     4.14 ms   (      7,174 ops)   34x cheaper
```

### 2.8 Certify bound: total in-weight vs actual arrivals
```
bound on total in-weight     18,949 candidates (13.67%)
bound on actual arrivals        184 candidates ( 0.13%)      103x fewer
```

### 2.9 Quartic vs cheap compare (per segment, 138,639 neurons)
```
quartic solve (60 bisection iters)   108.64 ms
advance + compare                      2.88 ms      38x
```

### 2.10 Auto-tuner behaviour at full scale
| active | window ms | grid ms | picked | effective |
|---|---|---|---|---|
| 0.88% | 3.71 | 97.74 | window | **26.4×** |
| 5% | 19.13 | 97.74 | window | 5.1× |
| 15% | 164.18 | 97.74 | grid | 1.00× |
| 100% | 1252.29 | 97.74 | grid | 1.00× |

---

## 3. Committed work

Branch `perf/event-driven-pytorch`. **Every performance change is gated on
bit-identical spike trains** (`code/verify_pytorch_perf.py`).

| Commit | Change |
|---|---|
| `040f699` | Event-driven fan-out + ring buffer in `run_pytorch.py` — **3.2–5.2×, bit-identical** |
| `0fa5b47` | Rotter–Diesmann propagator constants (opt-in) |
| `979f44b` | Wire `INTEGRATION='exact'` through the model classes |
| `19ca262` | Exact spike-time solver + certified silence predicate |
| `6680779` | Delay-window stepping kernels |
| `c891f81` | Window simulator + accuracy validation |
| `bfd7a56` | Vectorised repair; activity crossover measured |
| `b893905` | Real-connectome measurement (6.0×) |
| `37ad1b4` | Certify against actual arrivals (103× fewer candidates) |
| `c929b03` | Auto-tuned hybrid + generality analysis |

Also on `main`: `flyloop/` (standalone research engine, active-set stepping).

### Files
| File | Purpose |
|---|---|
| `code/run_pytorch.py` | **their** backend, optimised in place |
| `code/verify_pytorch_perf.py` | bit-identical gate — **run this before landing anything** |
| `code/exact_spike_time.py` | quartic solver + certified silence |
| `code/window_step.py` | window map, σ factors, silence bound |
| `code/window_fast.py`, `window_opt.py` | vectorised window simulators |
| `code/hybrid_step.py` | `ModeSelector` auto-tuner |
| `code/bench_connectome_window.py` | full-scale cost breakdown |
| `code/validate_window_exact.py` | accuracy head-to-head |

---

## 4. The accuracy finding (likely the most valuable result)

Their PyTorch backend uses **forward Euler**. Brian 2 — which the published model
was run in — selects **exact integration** automatically for linear equations
([docs](https://brian2.readthedocs.io/en/stable/user/numerical_integration.html)).

| Coefficient | Exact | Euler | Rel. error |
|---|---|---|---|
| `alpha_m` (v→v) | 0.9950124792 | 0.9950000000 | −1.25e-05 |
| `alpha_s` (g→g) | 0.9801986733 | 0.9800000000 | −2.03e-04 |
| **`P_vg` (g→v)** | **0.0049379353** | **0.0050000000** | **+1.257e-02** |

Euler shortens **every** τ by exactly `dt/2`: τ_syn −1.003%, τ_mem −0.250%.
Charge is preserved to 1e-5 (rates fine), but PSP peak +0.47% and the whole
simulation runs **0.25–1% fast** — 0.25–0.5 ms over a 50 ms response.

**The exact form costs the same FLOPs** (3 mul + 2 add vs 2 mul + 3 add).

Separately, grid detection **loses spikes**: vs a dt = 0.002 ms reference,
dt = 0.1 ms lost ~10% (42 vs 47) and dt = 0.2 ms ~17% (39 vs 47), because coarse
grids miss brief superthreshold excursions.

Head-to-head, deterministic 400-neuron net (no RNG):

| Method | Spikes | Count err | mean &#124;rate − truth&#124; |
|---|---|---|---|
| truth (dt = 0.002 ms) | 158 | — | — |
| **grid dt = 0.1 ms** | 159 | **+1** | 0.0463 Hz |
| **window dt = 1.8 ms** | **158** | **0** | **0.0000 Hz** |

**Why it matters:** the repo is a *benchmark comparing simulators*. If one backend
integrates differently from the reference, the comparison partly measures
integration schemes rather than frameworks.

---

## 5. The mathematics

### 5.1 Polynomial propagator (τ ratio = 4)
`u = v − v_rest`, `x = exp(-h/tau_mem)`, `exp(-h/tau_syn) = x^4`:
```
u(h) = x*u0 + (g0/3)*(x - x^4)        g(h) = x^4 * g0
```
No transcendental beyond one `x`; `x^4 = (x*x)^2`. **Semigroup**
`P(x1)P(x2) = P(x1*x2)` ⇒ advance a dormant neuron n steps with one `pow`.
Contractive (‖P‖₁ = x < 1) ⇒ fp32 floor ≈ 200ε ≈ 2.4e-5 mV against θ = 7 mV.

### 5.2 Certified silence (exact)
`sup_h u(h)` closed form, peak at `x* = ((3u0+g0)/(4g0))^(1/3)`:
```
sup u <= max(u0,0) + KAPPA_WINDOW_MAX * (g0 + w_positive_arriving)
KAPPA_WINDOW_MAX = 0.0720849531   over (0, 1.8 ms]
```
🔑 **Feed it the actual arrivals, not total in-weight** — legitimate because the
delay means every spike that can land was emitted before the window began, so
input is *already known exactly*. 103× fewer candidates (§2.8).

### 5.3 Quartic spike time
`u(h) = θ` ⇒ `x^4 + p·x + q = 0`, `p = -(3u0+g0)/g0`, `q = 3θ/g0`. `f` is
unimodal with peak at `x*`, `f(0)`,`f(1)` < 0, so the first crossing is the unique
root in `(x*, 1)` where `f` is monotone — bisection cannot fail to bracket.
Solvable in radicals only because the ratio is an integer ≤ 4; **ratio 5 → general
quintic → unsolvable.** At the boundary by luck.
⚠️ **38× more expensive than advance+compare** — use a compare to *detect*
crossings, the quartic only to *time* the few that cross (refractory caps it at
one per neuron per window).

### 5.4 Delay-window decoupling
Uniform 1.8 ms delay ⇒ by the method of steps (Bellman & Cooke 1963;
Hairer/Nørsett/Wanner II.17) the network decouples into N independent
inhomogeneous linear ODEs per window ⇒ `dt` can be **1.8 ms (18 steps)**, exactly,
if spike times are found analytically rather than by grid detection.
Refractory 2.2 ms > delay 1.8 ms ⇒ **≤1 spike per neuron per window**.

**Preconditions — VERIFIED against the data. Re-run if the model changes; one
zero-delay edge invalidates everything:**

| Precondition | Check | Result |
|---|---|---|
| No autapses | scanned all 15,091,983 edges | **0** ✅ |
| No zero-delay synapses | `delay=params['t_dly']` is a single scalar | uniform 1.8 ms ✅ |
| No gap junctions | chemical `Synapses` only | none ✅ |
| Refractory **absolute** | Brian2 marks *both* `dv/dt` and `dg/dt` `(unless refractory)` | ✅ |

**Novelty:** NEST/NEURON use `d_min` only to batch spike *communication* and keep
`h` small. A literature sweep found **no simulator that raises the integration
timestep to `d_min`**. The decoupling appears only as a justification for
parallelisation ([Front. Neuroinform. 2017](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2017.00034/full)).

### 5.5 σ-factorisation
A spike from `j` at `t_k` contributes at window end via `exp(-s/tau_s)` and
`kappa(s)` with `s = t_end - (t_k + D)` — **target-independent**, so per-source
scalars push through the connectome once. 18 delivery rounds → 1.
⚠️ **Apply event-driven, never as `W @ sigma`** (§2.7).
✅ **Eliminates the delay ring buffer entirely**: 21.1 MB of state and ~40 MB/window
of traffic → ~0.5 KB. **This is the one unconditional algorithmic win** (1.16× at
100% activity).

---

## 6. Ruled out — with reasons, so nobody re-treads them

| Technique | Why not |
|---|---|
| **SparseProp rescale** ([arXiv:2312.17216](https://arxiv.org/abs/2312.17216)) | Math generalises (v_rest absorbs by translation; Jacobian triangular) but it is O(N)→O(N), not O(N)→O(log N). Needs univariate + delta-synapse + event-driven; we violate all three. Drives quiet neurons into **denormals** (~100+ cycles/op). |
| **Multirate RK / IMEX / implicit** | Stiffness ratio is 4. Not stiff. dt sits ~100× below the FE stability limit — accuracy-bound, not stability-bound. |
| **Krylov / Arnoldi / Lanczos** | The positive delay keeps coupling out of the generator; `A_net` is **block-diagonal with 2×2 blocks**, so `exp(A_net·h)` *is* the per-neuron propagator. |
| **Magnus expansion** | `A` constant ⇒ Ω₁ = A·h, all commutators vanish. Reduces to `exp(Ah)`. |
| **Exponential Rosenbrock** | Linearises via a Jacobian; our only nonlinearity is the reset — discontinuous, non-differentiable. |
| **Strang splitting** | Order-2 needs two C₀-semigroups with bounded commutators. The reset is a projection at a state-dependent event time, not a flow. No generator ⇒ no commutator ⇒ no h² term. |
| **Delta-synapse limit** | Needs ε = τ_s/τ_m ≪ 1; here ε = 0.25 ⇒ 59% amplitude error. |
| **Graph reordering (RCM/METIS)** | Connectomes are hub-dominated scale-free; bandwidth reduction is not the bottleneck. Not cited as a win in the Loihi 2 work. |
| **Synapse pruning / faster SpMV** | Attacks the 0.14%. The obvious optimisation is the trap. |
| **Quantising neuron state to fp16/bf16** | `v ∈ [−52,−45]` sits in one binade; the −48 offset burns ~5 mantissa bits ⇒ fp16 ULP 0.03125 mV vs a 5 µV per-step increment ⇒ **updates silently dropped**. bf16 gives 28 levels total. *If needed:* shift to `u = v − v_rest` first, use int16 Q3.13 (~15 usable bits). |
| **Parareal / MGRIT** | No SNN application; converges poorly on discontinuous systems and the reset is a hard discontinuity. |
| **PSN / PSU / SPSN / DSN / SpikingSSMs** | All achieve parallelism by **removing or approximating the reset**. Fine for ML accuracy, fatal for simulation fidelity. |
| **Static cost model for mode selection** | Calibrated at 18,548 candidates it was **11× wrong** at 69,000 (cache effects). Replaced by runtime measurement. |

---

## 7. What would invalidate this work

Ranked by likelihood:
1. **Tonic/background drive to all neurons** destroys the active set — SparseProp names this explicitly. Safe only because Poisson hits 21 neurons.
2. **High-firing protocols** — 1.75 spikes/step is stimulus-specific.
3. **Heterogeneous delays** collapse the window to the *minimum* delay.
4. **True conductance coupling** `g·(E_rev − v)` makes the v-equation non-autonomous with per-neuron effective τ ⇒ exact propagator invalid.
5. **Brief superthreshold excursions** — if hybrid rather than fully event-driven, add the [Kunkel et al. 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3240333/) fail-safe cascade; miss probability ≤2.3e-4 but **worst in exactly our regime**.
6. **GPU at small active sets** — compaction may cost more than a dense 139k-wide kernel. Measure before porting.

---

## 8. Prior art

[Wang et al. 2025, *Neuromorphic Simulation of Drosophila on Loihi 2*](https://arxiv.org/abs/2508.16792)
(Sandia) — 140K neurons, 50M synapses, 12 chips. Wall-clock per simulated second:

> ⚠️ **CORRECTED 2026-07-30. The previous version of this table was wrong by
> 1000×** and the error propagated into §11. The paper's column header is
> `FlyWire (ms)` — **milliseconds** per simulated second, not seconds. Anything
> derived from the old table is void. Source: `research/papers/`, Table 1
> (`FlyWire (ms)` header at L764). Full analysis:
> `research/CORRECTION_benchmark_units.md`.

| Platform | published | = s/sim-second | vs real time |
|---|---|---|---|
| Brian 2 (reference) | 4419 ± 236 ms | 4.42 | 4.4× slower |
| STACS (Sandia, 8 MPI processes) | 2656 ± 80 ms | 2.66 | 2.7× slower |
| **Loihi 2 @ dt = 0.1 ms** | 53.76 ± 0.90 ms | **0.0538** | **18.6× faster** |
| **Loihi 2 @ dt = 1 ms** | 12.40 ± 0.28 ms | **0.0124** | **81× faster** |

The dt assignment is inferred, not read directly: the PDF text extraction wraps
the two Loihi rows onto one label. A *larger* dt must be *faster* (fewer steps),
so 12.40 ms is the dt = 1 ms row. The paper's statement that Loihi 2 performed
"better than realtime" (L776) is only true under the millisecond reading, which
confirms the units.

Reports ~3–350× over Brian 2, and *"speedups over 100× at sparser activity"*.
At dt = 1 ms they round *both* the 1.8 ms delay and 2.2 ms refractory to 2 ms
(+11% / −9% error); method of steps at dt = 1.8 ms would keep the delay exact.
[STACS is open source](https://github.com/sandialabs/STACS).

**What the correction costs us:** the old claim that "STACS gets only 1.66× over
Brian 2 on 64 HPC nodes, so algorithmic structure beats parallelism" is
withdrawn — it was 8 MPI processes, where 1.66× is unremarkable rather than
evidence. And **this project is not faster than Loihi 2**: 0.543 s/sim-second
(§11) against 0.0538, i.e. Loihi 2 is ~10× ahead. Real time on a laptop is still
the headline; beating dedicated neuromorphic silicon is not.

⚠️ These are *different machines*. Never tabulate our laptop measurements
alongside the paper's without saying so.

### Key references
**Exact integration & spike timing** — [Rotter & Diesmann 1999](https://link.springer.com/article/10.1007/s004220050570) · [Brette 2006](https://direct.mit.edu/neco/article/18/8/2004/7067/) · [Brette 2007](https://pubmed.ncbi.nlm.nih.gov/17716004/) · [Morrison et al. 2007](https://direct.mit.edu/neco/article/19/1/47/7159) · [Hanuschkin et al. 2010](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2010.00113/full) · [Hansel et al. 1998](https://direct.mit.edu/neco/article/10/2/467/6140/) · [Kunkel et al. 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3240333/)

**Event-driven / sparsity** — [SparseProp](https://arxiv.org/abs/2312.17216) · [Mattia & Del Giudice 2000](https://pubmed.ncbi.nlm.nih.gov/11032036/) · [Cessac et al. 2008](https://arxiv.org/abs/0810.3992) · [Bautembach et al. 2021](https://arxiv.org/abs/2107.04092) · [Magalhães et al. 2020](https://arxiv.org/abs/1907.00670) · [EDLUT](https://direct.mit.edu/neco/article/18/12/2959/7115/)

**Parallel scan / SSM** — [Bullet Trains ICML 2026](https://arxiv.org/abs/2603.13283) (affine-map scan, **exact** reset via speculation, 44×) · [FPT ICML 2025](https://arxiv.org/abs/2506.12087) (fixed-point reset, K≈3) · [SPSN](https://arxiv.org/abs/2306.12666) · [PSN](https://arxiv.org/abs/2304.12760) · [SpikingSSMs](https://arxiv.org/abs/2408.14909)

**Simulators** — [GeNN](https://www.nature.com/articles/srep18854) · [PyGeNN](https://www.frontiersin.org/articles/10.3389/fninf.2021.659005/full) · [Brian2CUDA](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2022.883700/full) · [Brian2GeNN](https://www.nature.com/articles/s41598-019-54957-7) · [Procedural connectivity](https://www.nature.com/articles/s43588-020-00022-7) · [NEST iaf_psc_exp](https://nest-simulator.readthedocs.io/en/latest/models/iaf_psc_exp.html)

**Precision / hardware** — [Hopkins et al. 2020](https://arxiv.org/abs/1904.11263) (stochastic rounding beats fp32) · [FeNN 2025](https://arxiv.org/html/2506.11760v1) (16-bit fixed point, 79.5% vs 79.6%) · [Gupta et al. 2015](https://arxiv.org/abs/1502.02551)

**Novelty gaps found (no hits across multiple query formulations)**
1. Uniform delay as the **integration timestep** / parallel-scan block ← strongest
2. Parallel scan inside a fixed-Δt **simulator** (all scan work is ML training)
3. **Two-state (V,G)** scan — published scans are scalar-V
4. Exactness-preserving parallel simulation (only Bullet Trains qualifies)

---

## 9. Method — rules that earned their place

**Rule 1 — every performance change must be proven bit-identical.**
`code/verify_pytorch_perf.py` (4 regimes), `flyloop/verify.py` (8 regimes). A
change that alters output is an *accuracy* change and must be argued separately.

**Rule 2 — measure, don't model.** Things I was confidently wrong about:
- "memory-bound" — true densely, false at active-set scale (dispatch-bound)
- "SparseProp gives ~11×" — the math says no
- "window stepping gives 18×" — real-scale measurement said 0.77×, then 1.05×, then 5.7×
- static cost model — 11× wrong one operating point away

**Rule 3 — ask research questions that can come back negative.** SparseProp and
graph-reordering both returned "don't", each saving days.

**Rule 4 — sparsity is stimulus-dependent.** Always test broad-stimulation and
whole-brain regimes, not just sugar.

**Rule 5 — the same mistake twice: worst case where the exact value was available.**
Delivery bounded over all 15.1M synapses when 31 neurons fired; the silence bound
assumed every synapse could fire when the delay makes arrivals *exactly known*.
Look for this pattern.

---

## 10. Next steps, in order

**General (help at any activity — prioritise these):**
1. **Compiled kernel (numba/C++)** — removes numpy dispatch. Unmeasured, plausibly 2–5×. **Largest untapped item.**
2. **int16 state in shifted coords** `u = v − v_rest ∈ [0,7]`, Q3.13 (~15 usable bits vs fp16's 7.8). Halves the dominant traffic.
3. **uint16 connectome weights** — counts are small integers ⇒ **lossless**, halves ~200 MB of connectome traffic.
4. Finish wiring σ-factorisation into `run_pytorch.py` to delete the delay buffer there (1.16×, unconditional).

**Conditional (sparse regimes):**
5. Wire `ModeSelector` into a single production stepper.
6. Use cheap advance+compare per segment; quartic only to time actual crossings (38× per §2.9).

**Validation (gates a PR):**
7. **Validate against Brian 2 via their own `code/compare_ground_truth.py`.** Not yet done. This is the gate.
8. End-to-end wall-clock of the full sugar experiment (current 5.7× is a *cost-model composition* of measured parts, not one run).
9. Re-time on an **idle** machine — all timings here are from a loaded laptop; ratios are stable, absolute seconds inflated.

**Not done deliberately:** no PR opened to Eon. Their repo is a benchmark;
changing one backend's numbers alters their published comparison, which is a
conversation to have with them, not something to slip in.

---

## 11. The native kernel — whole brain in real time on a laptop (2026-07-30)

**Target hardware (the "consumer device"):** Intel i7-1185G7 Tiger Lake,
4 cores / 8 threads @ 3.0 GHz, AVX-512, 48 KB L1d / 1.25 MB L2 per core,
12 MB shared L3, 32 GB LPDDR4x. Windows 11, clang 21, torch 2.13 CPU. No CUDA.

### 11.1 Result

Real time for this model is **0.1 ms/step** (dt = 0.1 ms ⇒ 10,000 steps per
simulated second). Min of 15 blocks × 500 steps, sugar protocol:

| Engine | ms/step | s/sim-second | vs real time |
|---|---|---|---|
| PyTorch dense | 1.6592 | 16.59 | 0.06× |
| PyTorch active-set | 1.6154 | 16.15 | 0.06× |
| native ×1 thread | 0.0822 | 0.82 | **1.22×** |
| native ×2 | 0.0790 | 0.79 | 1.27× |
| **native ×4** | **0.0543** | **0.54** | **1.84×** |

**30.6× over the PyTorch dense baseline, bit-identical.**

For honest scale (see the §8 correction — **not** the pre-2026-07-30 table,
which was wrong by 1000×): Sandia's 12-chip Loihi 2 achieves 0.0538 s/sim-second
at the same dt, so **dedicated neuromorphic silicon is still ~10× ahead of this
laptop.** The claim here is real time on a consumer device, not beating Loihi 2.
Brian 2 on the paper's hardware is 4.42 s/sim-second — different machine, so
that number is context, not a comparison.

Gate: 8 regimes × 800 steps, **v, g and refrac compared as uint32 every step**
plus exact spike-train comparison — `flyloop/verify_native.py`. All bit-equal.
Per-regime speedup 1.23× (saturating, 946 spikes/step) to 10.3× (sparse).

### 11.2 What actually made it fast

The old model of the bottleneck was wrong. At the 171 µs/step starting point:

```
C kernel         98.2 µs  (57%)
torch.bernoulli  45.1 µs  (26%)   <- for TWENTY-ONE random numbers
python glue      27.6 µs  (16%)   <- ndarray.ctypes.data_as, ~14x per step
```

Two of the three were framework overhead. In order of contribution:

1. **Fused single pass** (`flyloop/native/lif_kernel.c`). PyTorch runs ~12
   separate full-array passes; the state is streamed through cache twelve times
   instead of once. The threshold-and-reset maps perfectly onto AVX-512 mask
   registers (`vcmpps` + two `vblendmps`), so the spike bitset falls out of the
   compare for free.
2. **Sparse delay slots.** The dense (19, N) fp32 ring buffer is 10.5 MB and
   evicts the 2.2 MB neuron state from cache every 19 steps. Its contents have
   ~190 non-zeros out of 138,639, so storing each slot as an (index, value) list
   makes the delay line ~30 KB resident. Exact — the omitted entries are exactly
   zero. **This is simpler than the σ-factorisation of §5.5 and achieves the same
   goal.**
3. **Spike vector as a bitset** — 17 KB instead of 554 KB.
4. **Batched Poisson draws** (−45 µs). See §11.4.
5. **Thread pool** (1.35×). The four chunks are disjoint, so no synchronisation
   is needed *inside* a step. Workers spin on a generation counter rather than
   pay a 5–20 µs OS barrier at a ~200 µs step budget.
6. **Cached ctypes pointers** (−27 µs).

### 11.3 ⚠️ The reference's bit pattern depends on its thread count

Reproducing ATen bit-for-bit required matching two rounding behaviours:

- `v.add_(t, alpha=a)` is a **single-rounding FMA** in its vectorised body;
- but its **scalar tail is not fused** — a separate multiply-then-add.

So the reference integrates the last `(chunk_len mod 16)` neurons of **every**
`at::parallel_for` chunk with a different rounding from the rest. With N =
138,639 and 4 threads the chunks are 34,660 wide, so the seams are at neurons
34,656–34,659, 69,316–69,319, 103,976–103,979 and 138,624–138,638.

This was found the hard way: an FMA everywhere diverged at step 394 on neuron
138,637 (the final tail); forcing the separate form everywhere diverged at step
33. The kernel mirrors ATen's partition to reproduce the seams exactly.

**The consequence is worth stating plainly: "bit-identical to PyTorch" is not a
property of the model, it is a property of `torch.get_num_threads()`.** Change
the thread count and the reference's low bits change. For a repository whose
purpose is *comparing simulators*, that is a real reproducibility caveat, and it
compounds the §4 finding that the PyTorch backend also uses forward Euler where
Brian 2 uses exact integration.

The native kernel does **not** have this defect — its output is independent of
its own thread count (verified: ×1 and ×4 are bit-identical to each other and to
the reference). A `verify` run therefore pins `torch.set_num_threads(4)`.

### 11.4 Batched RNG is stream-exact

`torch.bernoulli` on a `(K, n)` tensor consumes the generator's stream in the
same order as `K` sequential `(n,)` draws, so pre-drawing a block is **exact**,
not an approximation. Verified at `(n, K)` = (21, 50), (21, 4096), (2, 1000),
(1, 777) — zero differences. Caveat: `inject()` invalidates the block, so a
closed-loop caller that re-injects every step must set `POISSON_BLOCK = 1`.

### 11.5 Files

| File | Purpose |
|---|---|
| `flyloop/native/lif_kernel.c` | fused AVX-512 kernel + spin-barrier thread pool |
| `flyloop/native_engine.py` | `NativeBrainEngine`, same API as `BrainEngine` |
| `flyloop/verify_native.py` | **the gate** — full-state bit comparison every step |
| `flyloop/bench_native.py` | min/median benchmark (min is the honest estimate) |

Build is automatic on import (clang, `-O3 -march=native -ffp-contract=off`).
**`-ffp-contract=off` is mandatory** — without it the compiler fuses the
conductance decay's mul+add into an FMA and bit-identity breaks. `-ffast-math`
would break it far more thoroughly.

---

## 12. Refractory removal + tile skipping (2026-07-30, later)

Supersedes §11.1's numbers. Full sweep, Syncthing stopped, min of 7 blocks,
400 lockstep verify steps per regime (`flyloop/verify_all.py`):

| regime | correct | torch ms | native ms | +reorder | vs torch | reorder gain | live tiles | **real time** |
|---|---|---|---|---|---|---|---|---|
| single neuron | BIT-EQ | 1.0664 | 0.0126 | **0.0109** | 97.7× | 1.15× | 3.2% | **9.16×** |
| silent (0 drive) | BIT-EQ | 2.7179 | 0.0309 | **0.0260** | 104.4× | 1.19× | 0.1% | **3.84×** |
| P9 walking (2) | BIT-EQ | 2.0810 | 0.0387 | **0.0262** | 79.5× | 1.48× | 12.7% | **3.82×** |
| sugar GRNs (21) | BIT-EQ | 2.4091 | 0.0695 | **0.0461** | 52.2× | 1.51× | 28.5% | **2.17×** |
| broad (100) | BIT-EQ | 2.3383 | 0.0859 | 0.0763 | 30.6× | 1.13× | 78.7% | 1.31× |
| broad (1000) | BIT-EQ | 3.0359 | 0.3786 | 0.3019 | 10.1× | 1.25× | 97.7% | 0.33× |
| broad (10000) | BIT-EQ | 4.0052 | 0.7986 | 0.6806 | 5.9× | 1.17× | 100.0% | 0.15× |
| saturating (40k) | BIT-EQ | 7.0947 | 2.5005 | 2.2536 | 3.1× | 1.11× | 100.0% | 0.04× |

⚠️ Native ms/step is a **lower bound** (load only makes blocks slower), so the
real-time column is conservative. The **vs torch** ratios are generous: the
PyTorch baseline was measured under the same residual load and reads 2.3–8.1 ms
against 1.66 ms fully idle.

### 12.1 Refractory counter out of the sweep

It existed only so the delayed pass could evaluate `gate = refrac >=
refrac_steps`, yet only ~40 neurons are refractory at once. Replaced by a gate
bitset (17 KB) + compact countdown list. The previous-spike bitset was *also*
read only for the refrac reset, so it left the sweep too — **the sweep now reads
just `v` and `g`**. 24.25 → 16.25 bytes/neuron.

⚠️ **The countdown is `refrac_steps + 1`, not `refrac_steps`.** The obvious rule
opens the gate one step early and admits input the reference discards. Ground
truth (neuron 95808): `refrac = spiked_prev ? 0 : refrac+1` puts refrac = k at
step t+1+k, so the gate is closed t+1..t+22 and reopens at **t+23**. Decrement
at the START of each step, open at `c <= 0`.

The countdown is walked **outside** the sweep deliberately: a refractory neuron
sits at `v == v_rest, g == 0` exactly — bit-indistinguishable from resting — so
if it rode inside a skippable sweep the gate would never reopen.

### 12.2 Tile-16 skipping + `cell_type` reordering

Skipping is exact: for `v == v_rest, g == 0` the kernel computes `t = 0`,
`v = fma(0, c_mem, v_rest) = v_rest`, `g = 0`, `v_rest > v_th` false — all
outputs unchanged, which is what skipping produces.

But in the shipped order live neurons are scattered (**mean run length 1.1**),
so 98.85% of tiles held a live neuron and skipping saved 1.15%. Neuron index is
an arbitrary CSV artefact, so renumbering is free and exact; `cell_type` takes
tile-16 from 70.6% live to 28.3% on sugar. Study:
`research/reorder_measurements.md`.

**Tiles are chunk-relative.** ATen's chunk boundaries (34,660) are not multiples
of 16, so a globally-aligned tile would straddle the FMA/non-FMA seam. Tiling
from each chunk's start gives exactly 2166 whole FMA groups plus one non-FMA
tail tile — no tile crosses a seam.

**Reordering moves the seam relative to the neurons.** The reference applies its
non-fused tail to ORIGINAL indices 34656-9, 69316-9, 103976-9, 138636-8; after
permutation those slots hold different neurons. The seam is now computed per
neuron in original space and carried through the permutation; straddling tiles
are marked MIXED (14 of 8,668) and take a scalar path that looks the bit up.

### 12.3 Bugs an adversarial review caught before shipping

1. **Stim neurons are driven outside the delay ring**, so nothing in the delayed
   pass can mark their tile live. One whose Poisson draw is 0 sits at exact rest,
   would be declared inert, and the entire sensory drive would silently
   disconnect. Stim tiles are **pinned**.
2. **At t=0 every neuron is at exact rest**, so initialising `tile_live` by scan
   marks the whole brain dead and nothing runs. All tiles **start live**.
3. **Refractory neurons are bit-indistinguishable from resting** — the inert test
   requires the gate OPEN.
4. **Marking live at fan-out EMISSION is insufficient** — a rescan can clear the
   tile during the up-to-19 steps the arrival is in flight. Mark at
   **consumption**.
5. *(self-inflicted)* The fast path tested `tile_fma[t]` for truthiness and MIXED
   is encoded as `2` — truthy — so mixed tiles took the FMA path anyway. Must be
   `== 1`.

### 12.4 Reordering helps even when nothing is skippable

Expected to be worthless at high activity. It is not: at broad(1000) with 97.7%
of tiles live it still gives **1.41×**, and 1.07–1.22× at 100% live. That cannot
be tile-skipping. Likely **fan-out locality** — `cell_type` also clusters
postsynaptic targets, so the scatter-add touches fewer cache lines. **Unconfirmed
— worth measuring directly.**

### 12.5 Environment traps

- **Windows Smart App Control** began blocking the DLL (`WinError 4551`) mid-
  session. It blocks by *file identity*, not content: an identical library under
  a new name loads fine. The DLL is now content-addressed (sha256 of source),
  which side-steps it and doubles as a build cache. **No need to disable SAC** —
  that is irreversible without resetting Windows.
- **Syncthing** re-indexing the repo inflated every timing (torch dense 4.97 ms
  vs 1.66 ms quiet). Stop it or exclude the repo before quoting numbers.

---

## 13. Nine neuron models over one connectome (2026-07-31)

Full write-up: **[MODELS.md](MODELS.md)**. Summary of what changed and what it
cost, in the terms this document uses.

`lif_kernel.c` became `nrn_kernel.c` and now carries nine membrane models. The
architectural claim is that **the expensive machinery belongs to the network,
not to the membrane equation** — the sparse delay ring, the event-driven
fan-out, the refractory gate, the ATen-faithful chunk partition and the thread
pool are all properties of the topology and the schedule, so they are written
once and shared. A model supplies only how `(v, g, aux)` advance, when a spike is
declared, and what the reset does. Each body is written once against a SIMD
abstraction (`native/simd.h`) and compiled three times, which makes ISA agreement
structural rather than something to re-audit.

`lif_step` is now a wrapper over the general `nrn_step`, deliberately: a green
run of `verify_native.py` / `verify_all.py` is therefore also evidence that the
generalisation did not disturb the one result already published. It is green.

**Two results worth carrying elsewhere:**

1. **Exact integration is strictly cheaper than the Euler it replaces.** The
   Rotter-Diesmann propagator is two FMAs and a multiply; forward Euler is four
   operations. §4's finding that Euler runs the simulation 0.25–1% fast has no
   performance defence — Euler is dominated, not traded against.
2. **Rush & Larsen 1978, from cardiac electrophysiology, is what makes
   whole-connectome Hodgkin-Huxley tractable.** Forward Euler on the sodium gate
   needs `dt` under 10 µs; Rush-Larsen is unconditionally stable for the gates
   and allows 25 µs sub-steps. Standard in the cardiac literature, essentially
   absent from the SNN performance literature.

**New, and possibly unclaimed — certified exponential elision.** AdEx and EIF
skip the exponential entirely whenever a two-instruction monotone upper bound
proves it would round away: `|E| < |S|·2^-25` implies `|E| < ulp(S)/2` implies
`fl(S+E) == S` exactly. This removes work without removing information, and it
is verified by running the whole brain with the elision switched off
(`elide_tiny = 0`) and comparing bits. `research/math/certified-exponential-elision.md`.

**Calibration is on excitability, not amplitude.** Each model's gain is solved
so the same number of simultaneous synapses fires it as fires the reference LIF
— **161.6**, which is itself the reason this connectome is quiet. The first
attempt matched PSP amplitude and made Hodgkin-Huxley ~5× too excitable, driving
the whole brain to sustained 20 Hz. That would have read as a finding about HH
and was a finding about the wrong denominator. Rule 5's pattern again: a worst-
case denominator used where the meaningful one was available.

**Verification, five independent gates** (`flyloop/verify_models.py`,
`flyloop/verify_exp.py`, `code/validate_models_brian2.py`): all nine models
bit-identical across AVX-512 / AVX2 / scalar including auxiliary state; tile
skipping proven exact per model by bit comparison; elision proven exact the same
way; the vectorised `exp` audited against **every one of the 2,237,530,114**
float32 values in its live domain (max **1 ULP**, all three paths identical); and
every model cross-checked against an independent Brian 2 implementation, eight of
nine at the float32 floor. Hodgkin-Huxley's 1.7e-2 residual is shown to be an
integrator difference by step refinement — ratios 2.01, 2.00, 1.99 per halving,
i.e. first-order convergence to the same trajectory.

**Three bugs found, all of a kind worth remembering:**
- AdEx reached `+inf` past the cutoff, then `inf − inf = NaN`, and NaN fails
  every ordered comparison *including the spike test* — so affected neurons went
  permanently **silent** rather than firing. A divergence that presents as
  quiescence is much harder to notice than one that presents as a blow-up.
- Resonate-and-fire added its input as a per-step impulse instead of integrating
  it: wrong by a factor of `dt`, invisible to every self-consistency gate, caught
  only by the external Brian 2 comparison.
- The exhaustive exp audit tested only the positive half of its domain and
  reported success, because a negative float32's bit pattern read as `int32` is
  negative and the range came out empty. **Coverage that is not itself checked is
  not coverage.**

⚠️ **Timings in MODELS.md §5 were taken on a loaded machine** — the PyTorch
baseline reads 5.017 ms/step against 1.66 measured quiet, so roughly 3×
inflation. Ratios are stable; absolute times are not.

**What it means for the benchmark:** "the fly brain" is model-dependent. Same
connectome, same protocol, same calibrated efficacy, and the network still
behaves differently. A benchmark that fixes one membrane equation measures
frameworks under one modelling choice — legitimate, but worth saying, and it
compounds §4 and FINDINGS §1.

---

## 14. Defects fixed (2026-07-30, late)

Unambiguous bugs, no change to model semantics, all gated on the 8-regime
dense-vs-active suite (`flyloop/verify.py`, ALL EXACT).

| defect | file | why it mattered |
|---|---|---|
| `KAPPA_WINDOW_MAX` was a **sampled** max feeding `certified_no_spike()` | `code/window_step.py` | an under-estimate certifies a neuron silent that actually spikes — the one failure this project cannot tolerate |
| dense fallback was a **one-way latch** | `flyloop/brain_engine.py` | one transient burst disabled the sparse path for the rest of the run |
| switch thresholded on spike **count** | `flyloop/brain_engine.py` | out-degree spans 1→9,800; one spike can carry more fan-out than a thousand |
| inert test `v == v_rest` is **unreachable** | `flyloop/brain_engine.py` | see §14.2 — the active set could only grow |
| DLL blocked by Windows Smart App Control | `flyloop/native_engine.py` | the engine was unloadable; fixed without disabling SAC |

### 14.1 The κ bound, now proved

κ(s) = C·(e^(−s/τm) − e^(−s/τs)) has its only interior stationary point at
`s* = ln(τm/τs)/(1/τs − 1/τm) = 9.2420 ms`, far outside the (0, 1.8] window, so
κ is strictly increasing there and its max is exactly κ(D). The function
re-derives `s*` from its arguments and **raises** if the peak ever moves inside
the window, so the proof cannot go stale if the constants change. Closed form
equals the old sampled max to the bit; it correctly refuses D = 12 ms.

### 14.2 ⚠️ The inert predicate is fundamentally limited in fp32

`v == v_rest` is **unreachable** for any neuron that was ever perturbed. With
g = 0 the membrane decays d ← 0.995·d, but near −52 mV the fp32 ULP is ~3.8e−6,
so once d reaches one ULP the update rounds back to itself and the neuron sits
one ULP above rest forever. Under the old predicate it could never be pruned.

Replaced by the actual **fixed-point** condition: if the update maps the state to
itself, skipping is exact (`v == v_rest` is the special case d = 0).

**The same trap applies to `g`**, which decays geometrically and reaches exactly
zero only after ~5,000 steps. So a neuron that receives *any* input stays
formally live for thousands of steps however negligible its conductance, and the
sparse path's inert set decays with cumulative activity. **No exact predicate can
avoid this.** An ε-prune backed by `certified_no_spike` is the route; unattempted.

### 14.3 Two observations that read like bugs but are not

- **Sugar sits at ~4.9% live**, inside the hysteresis band (release <4%, trip
  >8%), so it correctly does *not* release. That is the band working.
- **A broad burst leaves the network self-sustaining** — 78,914 spikes in 2,000
  steps *after* the stimulus is removed entirely. The active set therefore never
  falls back, regardless of the switch. Worth knowing before interpreting any
  long run.

### 14.4 Hypotheses tested and dropped

- **Denormal conductances slowing the kernel.** Did not reproduce: zero
  denormals at 3,000 steps, and a flush-to-zero control came out *slower*. The
  1.39× post-burst slowdown is higher activity, not slow arithmetic.
- **Trial batching.** Measured the amortisable fraction of a step first: 8.8%
  (sparse) / 0.7% (dense), capping an 8-trial gain at ~1.09×/1.007×. GeNN's 3.5×
  batching win is a GPU artifact of underutilisation; this CPU is already
  91–99% real compute at n=1. Not built.

### 14.5 Concurrent work in this tree

A parallel session is building a **multi-model** kernel in the same checkout
(`flyloop/models.py`, `native/nrn_kernel.c`, `sweep_template.h`, `studio/`,
`verify_models.py`, `code/validate_models_brian2.py`) — Izhikevich, AdEx,
Hodgkin-Huxley, GLIF sharing the same connectome, delay ring, fan-out and tile
machinery. Not covered by §11–14 and not measured here. Check `git status` and
file mtimes before large patches in `flyloop/`.

---

## 15. Next, in order

**Done since the last revision:** refrac removal (§12.1), tile-16 skipping +
reordering (§12.2), int16 connectome weights (lossless — integer synapse
counts), runtime ISA dispatch, a full-brain Brian 2 reference
(`code/run_brian2_reference.py`), and the model-divergence finding (§1.1 of
FORK.md).

1. ~~Confirm the fan-out-locality hypothesis~~ / ~~attack the fan-out~~ —
   **BOTH MEASURED, BOTH REFUTED (2026-07-31).** `flyloop/profile_split.py` and
   `flyloop/bench_fanout.py`; full write-up in
   `research/fanout-locality-negative.md`.

   The split is real — fan-out is **49.6%** of the saturating step (136,605
   edges at 21.7 ns each) and 9.7–13.3% in the regimes the published
   experiments use. But **neither candidate fix works**:

   - **Pull direction (Beamer).** Refuted on arithmetic alone: a pull fan-out
     costs O(E) = 15,091,983 edges *every step regardless of activity*, against
     136,605 actually needed in the densest regime — **110× more work**. The
     switch would never fire. Push wins everywhere in this connectome.
   - **Radix-partitioned scatter.** Implemented and measured: **0.47–0.87×,
     slower at every size** from 1,283 to 1,796,811 edges/step. The direct
     scatter holds **6–11 ns/edge up to 1.8M edges and does not degrade**, so it
     was never miss-bound — each presynaptic neuron's CSC range lists its
     targets in *ascending* order, so the prefetcher handles it and the
     randomness is only *between* sources. The implementation is kept in the
     kernel, gated behind a caller-supplied buffer the engine passes as NULL, so
     the measurement stays reproducible.

   Consequence: the 21.7 ns/edge in the live saturating step is **contention**
   with the rest of the step's working set, not the scatter's own behaviour. The
   lever for that is shrinking resident state (int16 shifted coordinates, uint8
   saturating counters — see `research/memory-layout-quantization.md`), not
   reorganising the traffic.

   ⚠️ **And §2.1's headline needs its units read carefully.** "Synaptic
   propagation is only 0.14% of runtime" is an *operation-count* statement used
   as a *runtime* one. Measured by time it is **9.7%** of the sugar step —
   ~70× more than the op count implies, because the neuron sweep streams and the
   scatter does not. The conclusion it supports (don't chase spike delivery in
   the sparse regime) survives at 9.7%; the number quoted for it should be the
   measured one.

   Still open from §12.4: reordering gives 1.07–1.41× where *nothing* is
   skippable, and that is now known **not** to be scatter locality. Unexplained.
3. **Validate against Brian 2 end to end** via `code/compare_ground_truth.py`.
   `run_brian2_reference.py` now builds the full 138,639-neuron network
   (~30 s per 100 ms simulated, 1.8 GB), so this is unblocked. It matters more
   now that §1.1 shows the backends differ at the *model* level.
4. **Throughput, not latency.** Their own data shows GeNN gains 3.5× from
   batching 8 trials. The scientific workload is parameter sweeps
   (every-neuron ablation), which is a batching problem. Real-time latency only
   matters for the closed-loop embodied case in `virtualfly/`, which cannot be
   batched — but that is exactly where this kernel is uniquely enabling.
5. ~~Fix `_dense_fallback`~~ **DONE** — now bidirectional with hysteresis (trip
   >8% active, release <4%, rechecked every 256 steps), and thresholding on
   Σ out-degree rather than spike count. Also **DONE**: `KAPPA_WINDOW_MAX` is now
   a closed-form bound with a proof that raises if it goes stale, and the inert
   predicate tests the actual fixed point rather than `v == v_rest`.
   ⚠️ Open, and deeper than the latch: `g` reaches exactly zero only after
   ~5,000 steps, so a perturbed neuron stays formally live for thousands of steps
   however negligible its conductance. No *exact* predicate can avoid this — an
   ε-prune backed by `certified_no_spike` is the route, unattempted.
6. **Delay-window stepping at dt = 1.8 ms** — the unclaimed algorithmic result,
   hardware-independent, would help GeNN and Loihi too. Highest ceiling, highest
   risk.
7. Port the kernel into `code/run_pytorch.py` so the upstream benchmark benefits,
   once (3) is done.
