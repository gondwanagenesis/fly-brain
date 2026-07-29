# Accelerating the whole-brain *Drosophila* LIF connectome simulation

**Complete working record — measurements, reasoning, dead ends, corrections, next steps.**

Written for cold pickup by another engineer or AI. Everything is either measured
on this machine, cited, or explicitly flagged as a projection.

- Last updated: 2026-07-29
- Upstream: [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain)
- **Our fork: [gondwanagenesis/fly-brain](https://github.com/gondwanagenesis/fly-brain)**, branch `perf/event-driven-pytorch` (all work pushed)
- Local: `Documents/FlyBrain`, venv at `.venv/` (Python 3.12, torch CPU, numpy, scipy)

---

## 0. Read this first — the strategic conclusion

**Large speedups in this problem are activity-dependent, not general.** The work
is proportional to activity. At full activity every neuron must be advanced every
timestep and you are pinned to memory bandwidth; no algorithm makes updating
138,639 neurons cost less than reading and writing 138,639 neurons' state.

| Optimisation | 100% brain active | Sparse activity |
|---|---|---|
| Event-driven fan-out **(banked, bit-identical)** | ~1× | **3.2–5.2×** |
| Delay-window stepping | 1.00× (auto-falls back) | **26×** |
| σ removes the delay ring buffer | **1.16×** | 1.16× |
| Exact integration | 1× (free *accuracy*) | 1× |
| State packing | ~1.5× | ~1.5× |

**Unconditional total ≈ 1.5–2×. Everything larger requires sparse firing.**

This is not a limitation of the approach — it is the structure of the problem, and
the field agrees. Sandia's Loihi 2 paper reports *"performance advantages increase
with sparser activity"* and reaches 100×+ only in the sparse regime. Twelve
neuromorphic chips obey the same law.

The consolation: **real brains are sparse.** We measured 1.75 spikes/step and 91%
of neurons at exact rest under localised stimulation. An embodied model with all
senses will have *more active regions*, not a saturated brain. The auto-tuner
(§5.4) handles wherever it lands — window mode when it pays, grid when it does
not, never worse.

**The most valuable single finding is probably not speed at all — see §4.**

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

| Simulator | s/sim-second |
|---|---|
| Brian 2 (reference) | 4419 ± 236 |
| STACS (Sandia, Charm++, 64 Summit nodes) | 2656 ± 80 |
| **Eon PyTorch** (our measurement) | ~487 |
| **Loihi 2 @ 1 ms** | **53.76 ± 0.9** |

Reports ~3–350× over Brian 2, and *"speedups over 100× at sparser activity"*.
🔑 **Their compromise is our opportunity:** at dt = 1 ms they rounded *both* the
1.8 ms delay and 2.2 ms refractory to 2 ms (+11% / −9% error). Method of steps at
dt = 1.8 ms keeps the delay exact by construction.
Note STACS gets only **1.66×** over Brian 2 on 64 HPC nodes — algorithmic structure
matters far more than parallelism here. [STACS is open source](https://github.com/sandialabs/STACS).

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
