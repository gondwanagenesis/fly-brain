# Accelerating the whole-brain *Drosophila* LIF connectome simulation

**A complete working record: measurements, reasoning, literature, dead ends, and next steps.**

Written to be picked up cold by another engineer or AI. Everything here is either
measured on this machine, cited, or explicitly labelled as a projection.

Last updated: 2026-07-29 · Repo: `Documents/FlyBrain` (clone of
[eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain))
· Work branch: `perf/event-driven-pytorch`

---

## 0. TL;DR for whoever picks this up

1. **The bottleneck is not where the field usually looks.** Synaptic propagation is
   **0.14%** of runtime for this workload; the dense per-timestep neuron update is
   **99.86%**. Ratio 728:1. Optimising spike delivery — the thing GeNN is built
   for — is nearly worthless here.
2. **91.2% of neurons are provably inert** (at exactly `v=vRest, g=0`) and can be
   skipped losslessly.
3. **τ_mem/τ_syn = 20/5 = 4 EXACTLY.** This is a gift: the propagator becomes a
   polynomial, and exact spike times solve as a quartic (solvable in radicals;
   ratio 5 would be a general quintic — unsolvable).
4. **The uniform 1.8 ms delay decouples the network into 18-step windows.** This
   is the largest single remaining win (18×) and is *exact*.
5. **Their PyTorch backend does not reproduce the published Brian 2 baseline** —
   it uses forward Euler where Brian 2 uses exact integration. This may matter
   more than any speedup.

---

## 1. The model

From [Shiu et al. 2024, *Nature*](https://www.nature.com/articles/s41586-024-07763-9)
([methods, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10187186/)), on the
[FlyWire](https://codex.flywire.ai/) v783 connectome
([Dorkenwald et al. 2024](https://www.nature.com/articles/s41586-024-07558-y)).

```
dg/dt = -g/tau_syn + sum_k w_k delta(t - t_k)      tau_syn =  5 ms
dv/dt = (g - (v - v_rest)) / tau_mem               tau_mem = 20 ms
spike if v > v_th; then v = v_reset, g = 0
```

| Parameter | Value |
|---|---|
| neurons | 138,639 (139,255 incl. unreconstructed) |
| connections | 15,091,983 edges / 54,492,922 synapses |
| excitatory : inhibitory | 60 : 40 |
| `v_rest` = `v_reset` | −52 mV |
| `v_threshold` | −45 mV (θ = 7 mV above rest) |
| `tau_mem` / `tau_syn` | 20 ms / 5 ms — **ratio exactly 4** |
| refractory | 2.2 ms |
| **axonal delay** | **1.8 ms, uniform on every synapse** |
| `dt` | 0.1 ms |
| `w_syn` | 0.275 mV × integer synapse count |

⚠️ **The model is current-based, not conductance-based.** `g` enters *additively*
(`dv/dt = (g − (v−v_rest))/tau_mem`), not as `g·(E_rev − v)`. This linearity is
what makes exact integration and the whole approach below valid. If the model
ever moves to true conductance coupling, most of this breaks — see §7.

---

## 2. What we measured (on this machine, reproducible)

### 2.1 The cost distribution — the central finding

Sugar-GRN experiment (21 gustatory neurons @ 200 Hz), instrumented:

| Component | Ops/step | Share |
|---|---|---|
| Dense neuron update (all 138,639) | 138,639 | **99.86%** |
| Event-driven synaptic propagation | ~190 | 0.14% |
| Poisson input (21 neurons) | 21 | ~0% |

**Ratio 728 : 1.** Mean activity is **1.75 spikes/step** = 0.0013% of neurons.

### 2.2 Quiescence

| Time | Neurons ever perturbed | Provably inert |
|---|---|---|
| 100 ms | 7,232 (5.22%) | 94.78% |
| 300 ms | 11,136 (8.03%) | 91.97% |
| 500 ms | 12,251 (8.84%) | **91.16%** |

Not a topology limit — **96.8% of the brain is reachable within 4 hops** of the
21 sugar GRNs. It is *dynamical attenuation*: the connectome does not propagate a
small taste input brain-wide.

⚠️ **Stimulus-dependent.** Verified across 8 regimes (1 neuron → 40k neurons
driven). Broad stimulation collapses the active set. Any claim must be re-checked
per protocol.

### 2.3 Arithmetic intensity / roofline

~10 FLOP against ~32 B traffic per neuron per step ⇒ **~0.3 FLOP/byte**. An
A100's roofline knee is ~12, so the dense path is deeply memory-bound.
Traffic: 44 GB/sim-second dense, 4 GB with quiescence skipped.

⚠️ **This flips at small scale.** Once the working set shrinks to ~3k active
neurons, it becomes **dispatch-bound**: cProfile shows **68% of runtime is
PyTorch per-op overhead** (~45 µs × ~30 ops/step), while the arithmetic on 13 KB
of state is sub-microsecond. *Both statements are true at different scales.*

### 2.4 τ ratio identity — verified to machine precision

With `x = exp(-dt/tau_mem)`:

| Identity | Agreement |
|---|---|
| `exp(-dt/tau_syn)` == `x^4` | 1.11e-16 |
| `P_vg` == `(x - x^4)/3` | 3.64e-17 |
| `alpha_m` == `x` | 0.00e+00 |

### 2.5 Exact spike-time solver — validated against brute force

20,000 random `(u0, g0)` states vs. brute-force integration of the exact
propagator at dt = 2e-4 ms:

| Check | Result |
|---|---|
| Agreement on *whether* a spike occurs | **20000 / 20000** |
| max &#124;t_analytic − t_brute&#124; | 2.0e-04 ms (== brute grid step) |
| **False silence certifications** | **0** ← safety-critical |

The residual *is* the brute-force discretisation, i.e. the analytic solution is
exact to the resolution the test can measure.

---

## 3. What is implemented and committed

Branch `perf/event-driven-pytorch`. **Every performance change is verified
bit-identical** by `code/verify_pytorch_perf.py`.

| Commit | Change | Result |
|---|---|---|
| `040f699` | Event-driven fan-out + ring-buffer delay line in `run_pytorch.py` | **3.2–5.2×, bit-identical** |
| `0fa5b47` | Rotter–Diesmann propagator constants (opt-in) | accuracy fix |
| `979f44b` | Wire `INTEGRATION='exact'` through the model classes | no regression |
| `19ca262` | Exact spike-time solver + certified silence predicate | validated §2.5 |

On `main`: `flyloop/` — a standalone steppable engine (active-set stepping, state
packing) used as a research testbed, plus `flyloop/ROADMAP.md`.

### Measured speedups (bit-identical, 4 regimes)

| Regime | Speedup |
|---|---|
| sugar GRNs (21 @ 200 Hz) | 5.20× |
| single neuron (1 @ 200 Hz) | 3.78× |
| no drive (0 Hz) | 3.40× |
| P9 walking (2 @ 100 Hz) | 3.19× |

⚠️ Timings taken on a loaded laptop CPU. **Ratios are stable; absolute seconds
are inflated.** Re-measure on an idle machine before quoting publicly.

---

## 4. The accuracy finding (possibly the most valuable result)

Their PyTorch backend uses **forward Euler**. Brian 2 — which the published
Shiu et al. model was run in — selects **exact integration** automatically for
linear equations (`method='auto'` → `'exact'`,
[docs](https://brian2.readthedocs.io/en/stable/user/numerical_integration.html)).

Forward Euler shortens **every** time constant by exactly `dt/2`:

| Coefficient | Exact | Euler | Relative error |
|---|---|---|---|
| `alpha_m` (v→v) | 0.9950124792 | 0.9950000000 | −1.25e-05 |
| `alpha_s` (g→g) | 0.9801986733 | 0.9800000000 | −2.03e-04 |
| **`P_vg` (g→v)** | **0.0049379353** | **0.0050000000** | **+1.257e-02** |

⇒ `tau_syn` 5.000 → 4.94983 ms (−1.003%), `tau_mem` 20.000 → 19.94996 ms (−0.250%).

**Consequences:** total synaptic charge preserved to ~1e-5 (so firing *rates* are
fine), but PSP peak +0.47%, peak time −0.066 ms, and the whole simulation runs
**0.25–1% fast** — a 0.25–0.5 ms error over a 50 ms response, an order of
magnitude larger than the dt/2 grid quantisation.

**The exact form costs the same FLOPs** (3 mul + 2 add vs 2 mul + 3 add).

**Why this matters for their paper:** the repo is a *benchmark comparing
simulators*. If one backend integrates differently from the reference, the
comparison partly measures integration schemes rather than frameworks.

---

## 5. The mathematical structure worth exploiting

### 5.1 Polynomial propagator (from τ ratio = 4)

Let `u = v − v_rest`, `x = exp(-h/tau_mem)`. Then `exp(-h/tau_syn) = x^4` and:

```
u(h) = x*u0 + (g0/3)*(x - x^4)
g(h) = x^4 * g0
```

- **No transcendental beyond one `x`**; `x^4 = (x*x)^2`.
- **Semigroup: P(x1)·P(x2) = P(x1·x2)** ⇒ advance a dormant neuron `n` steps with
  one `pow`, not `n` matrix applications.
- **Contractive** (‖P‖₁ = x < 1) ⇒ round-off does not accumulate; the fp32 floor
  is ε/(1−x) ≈ 200ε ≈ 2.4e-5 mV against a 7 mV threshold. fp32 is provably safe.

### 5.2 Certified silence predicate (exact, not heuristic)

`sup_h u(h)` has a closed form (peak at `x* = ((3u0+g0)/(4g0))^(1/3)`), so we can
**prove** a neuron cannot reach threshold before its next input. Cheap bound:

```
sup u(t) <= max(u0, 0) + 0.0720850 * (g0 + sum w+)
```

A resting neuron needs `sum w+ > 97` within one 1.8 ms window even to become a
candidate. Implemented and validated (§2.5) in `code/exact_spike_time.py`.

### 5.3 Quartic spike time

Setting `u(h) = θ` gives the depressed trinomial quartic `x^4 + p·x + q = 0`,
`p = -(3u0+g0)/g0`, `q = 3θ/g0`. `f` is unimodal with peak at `x*`; `f(0)` and
`f(1)` are both negative, so the first crossing is the unique root in `(x*, 1)`
where `f` is strictly monotone — bisection cannot fail to bracket.

Only solvable in radicals because the ratio is an integer ≤ 4. **Ratio 5 → general
quintic → not solvable.** We are at the boundary by luck.

### 5.4 Delay-window decoupling — the biggest remaining win

The uniform 1.8 ms delay means nothing emitted at time `t` affects anything before
`t + 1.8 ms`. By the *method of steps* for delay differential equations, within
one window the network is **N independent inhomogeneous linear ODEs** whose forcing
is fully determined by history. ⇒ `dt` can be **1.8 ms (18×)** with *zero* accuracy
loss, provided spike times inside the window are found exactly (§5.3) rather than
by grid detection.

**Refractory 2.2 ms > delay 1.8 ms ⇒ at most one spike per neuron per window**,
which makes the window map single-valued and bounds repair work at one root-find.

#### Preconditions — VERIFIED against the actual data (2026-07-29)

Method of steps is sound (Bellman & Cooke 1963; Hairer/Nørsett/Wanner ch. II.17)
but fails on zero-delay coupling. All blockers checked and cleared:

| Precondition | Check | Result |
|---|---|---|
| No autapses | scanned all 15,091,983 edges for `pre == post` | **0** ✅ |
| No zero-delay synapses | `delay=params['t_dly']` is a **single scalar** on the whole synapse population — no per-synapse delays | uniform 1.8 ms ✅ |
| No gap junctions | model uses `Synapses` (chemical) only | none ✅ |
| Refractory is **absolute** | Brian2 marks **both** `dv/dt` *and* `dg/dt` `(unless refractory)` — state frozen, neuron provably cannot fire | absolute, 2.2 > 1.8 ms ✅ |

⚠️ **Re-run these checks if the connectome or model is ever changed.** A single
zero-delay edge or autapse invalidates the entire window decoupling.

#### Novelty status

NEST/NEURON separate the *integration* step `h` from the *communication* step
`d_min`, and use `d_min` only for **spike batching / MPI parallelisation** — they
keep `h` small. A literature sweep found **no simulator that raises the
integration timestep to `d_min`**. The decoupling itself is stated in the
literature as the *theoretical basis for parallelisation*
([Front. Neuroinform. 2017](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2017.00034/full)),
never as a licence to enlarge `dt`. **That gap is the contribution.**

### 5.5 σ-factorisation of spike delivery

For a spike at `t_k` arriving at `t_k + D`, the contribution at window end is

```
Δg_i = Σ_j W_ij · e^(-s/tau_s)                 s = t0 - t_k
Δu_i = Σ_j W_ij · κ(s),  κ(s) = (1/3)(e^(-s/tau_m) - e^(-s/tau_s))
```

`s` is **target-independent**, so define per-source scalars `σ_j` and get
`Δg = W σ^(g)`, `Δu = W σ^(u)` — **one sparse mat-mul with a 2-column right factor
per window**, replacing 18 rounds of irregular atomic scatter. Exact, works with
fully off-grid spike times.

⚠️ Exact only for targets that do **not** spike inside the window (a spike resets
`g=0`, wiping accumulated pre-spike contribution) ⇒ predict-and-repair (§5.2/5.3).

---

## 6. Ruled out — with reasons, so nobody re-treads these

| Technique | Verdict | Why |
|---|---|---|
| **SparseProp global rescale** ([arXiv:2312.17216](https://arxiv.org/abs/2312.17216)) | ❌ | Math *does* generalise to our 2-state system (v_rest absorbs by translation; Jacobian is triangular), but it is O(N)→O(N), not O(N)→O(log N). SparseProp's win needs univariate + delta-synapse + event-driven; we violate all three. Also drives quiet neurons into **denormals** (~100+ cycles/op on x86) — likely a net loss. |
| **Multirate RK / IMEX / implicit** | ❌ | Stiffness ratio is 4. Not stiff. dt sits 100× below the FE stability limit — the step is set by accuracy, not stability. Every stiffness-motivated method targets a problem we don't have. |
| **Krylov / Arnoldi / Lanczos on the network matrix** | ❌ | The positive delay means coupling never enters the generator; `A_net` is **block-diagonal with 2×2 blocks**, so `exp(A_net·h)` *is* the per-neuron propagator. One Arnoldi step costs more than the exact answer. |
| **Magnus expansion** | ❌ | `A` is constant ⇒ Ω₁ = A·h and all commutator terms vanish. Reduces to `exp(Ah)`. (Would become relevant *if* the model went conductance-based.) |
| **Exponential Rosenbrock** | ❌ | Linearises the nonlinearity via a Jacobian; our only nonlinearity is the reset, which is discontinuous and non-differentiable. No Jacobian to expand. |
| **Strang splitting (flow vs reset)** | ❌ | Order-2 needs both operators to be C₀-semigroups with bounded commutators. The reset is a projection at a state-dependent event time, not the flow of any vector field. **No generator ⇒ no commutator ⇒ no h² term to cancel.** Correct framework is impulsive/hybrid ODE with event location: global order = min(p, q) where q is event-location order. |
| **Delta-synapse (singular perturbation) limit** | ❌ | Needs ε = tau_s/tau_m ≪ 1; here ε = 0.25. Gives 59% amplitude error and 9.2 ms timing error. |
| **Graph reordering (RCM / METIS)** | ❌ | Connectomes are hub-dominated scale-free graphs; bandwidth reduction is not the bottleneck. RCM excels on banded FEM matrices. Not cited as a win in the Loihi 2 work. |
| **Synapse pruning / weight compression / faster SpMV** | ❌ | All attack the 0.14%. The obvious optimisation is the trap. |
| **Quantising neuron state (fp16/bf16/int8)** | ❌ | State is only ~1.1 MB and cache-resident — wrong target. Also, `v ∈ [−52,−45]` sits in one binade so the −48 offset burns ~5 mantissa bits: fp16 ULP = 0.03125 mV vs a 5 µV per-step increment ⇒ **updates silently dropped entirely**. bf16 gives 28 levels across the whole range. *If ever needed:* shift to `u = v − v_rest` first (worth ~3 bits), use fp32 accumulators, or int16 Q3.13 (~15 usable bits). |
| **Parareal / MGRIT** | ❌ | No SNN application exists; converges poorly on discontinuous systems and spike-reset is a hard discontinuity. *Possible* reframing: mean-field coarse propagator + exact fine, converging in Wasserstein rather than sup-norm — genuinely open, but speculative. |
| **PSN / PSU / SPSN / DSN / SpikingSSMs** | ❌ | All obtain parallelism by **removing or approximating the reset** (stochastic firing, probabilistic reset, learned surrogate reset, dynamic decay). Fine for ML accuracy; **fatal for simulation fidelity**. |

---

## 7. Assumptions that would invalidate this work

Ranked by likelihood:

1. **Tonic/background Poisson drive to *all* neurons** destroys the active set —
   every neuron becomes an event. SparseProp names this explicitly. We are safe
   only because Poisson hits 21 neurons.
2. **High-firing protocols.** The 1.75 spikes/step figure is stimulus-specific.
3. **Heterogeneous delays** collapse the 18-step window to the *minimum* delay.
4. **True conductance coupling** `g·(E_rev − v)` makes the v-equation
   non-autonomous with a per-neuron effective time constant ⇒ exact propagator
   invalid, and the SparseProp-style rescale definitively dies.
5. **Brief superthreshold excursions.** If staying hybrid rather than fully
   event-driven, add the [Kunkel et al. 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3240333/)
   fail-safe cascade; miss probability ≤2.3e-4, but **worst in exactly our regime**
   (low firing rate, low connectivity, strong coupling).
6. **GPU at small active-set sizes** — compaction may cost more than a dense
   139k-wide kernel. Measure before porting.

---

## 8. Prior art and comparison

### 8.1 Existing whole-brain *Drosophila* simulations

[Wang et al. 2025, *Neuromorphic Simulation of Drosophila Melanogaster Brain
Connectome on Loihi 2*](https://arxiv.org/abs/2508.16792) (Sandia National Labs)
— 140K neurons, 50M synapses on 12 Loihi 2 chips. Wall-clock per 1 s simulated,
sugar-neuron experiment:

| Simulator | s / sim-second |
|---|---|
| Brian 2 (their reference) | 4419 ± 236 |
| STACS (Sandia) | 2656 ± 80 |
| Loihi 2 @ 0.1 ms | — |
| **Loihi 2 @ 1 ms** | **53.76 ± 0.9** |

Reported **~3× to ~350× over Brian 2**, and — critically —
*"speedups over 100× at **sparser** activity levels"* and *"performance advantages
**increase with sparser activity**."* **This independently confirms our central
thesis.**

🔑 **Their compromise is our opportunity:** at dt = 1 ms they rounded *both* the
1.8 ms delay and 2.2 ms refractory to 2 ms (+11% / −9% error). Method-of-steps at
dt = 1.8 ms keeps the delay exact by construction, and off-grid spike times (§5.3)
keep refractory exact. **Same speed, without the accuracy loss.**

They also validate against Brian 2 as ground truth — reinforcing §4.

### 8.2 Key references

**Exact integration & spike timing**
- [Rotter & Diesmann 1999, *Biol Cybern* 81:381](https://link.springer.com/article/10.1007/s004220050570) — matrix-exponential propagator (what NEST uses)
- [Brette 2006, *Neural Comput* 18:2004](https://direct.mit.edu/neco/article/18/8/2004/7067/) — exact event-driven IF with exponential conductances
- [Brette 2007, *Neural Comput* 19:2604](https://pubmed.ncbi.nlm.nih.gov/17716004/) — exponential currents via polynomial root-finding
- [Morrison et al. 2007, *Neural Comput* 19:47](https://direct.mit.edu/neco/article/19/1/47/7159) — exact subthreshold integration + continuous spike times
- [Hanuschkin et al. 2010, *Front Neuroinform* 4:113](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2010.00113/full) — precise spike times in time-driven simulation
- [Krishnan et al. 2017, *Front Neuroinform*](https://arxiv.org/abs/1706.05702) — **perfect spike detection via time reversal** (necessary+sufficient crossing test)
- [Hansel et al. 1998, *Neural Comput* 10:467](https://direct.mit.edu/neco/article/10/2/467/6140/) — grid detection destroys synchrony; interpolation restores O(h²)
- [Kunkel et al. 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3240333/) — fail-safe threshold-crossing detection

**Event-driven / sparse-activity**
- [SparseProp, Engelken NeurIPS 2023](https://arxiv.org/abs/2312.17216)
- [Mattia & Del Giudice 2000](https://pubmed.ncbi.nlm.nih.gov/11032036/) — min-delay as causality horizon
- [Cessac et al. 2008](https://arxiv.org/abs/0810.3992) — numerical bounds to prune events
- [Bautembach et al. HPEC 2021](https://arxiv.org/abs/2107.04092) — GPU lazy + work queues
- [Magalhães et al. ICCS 2020](https://arxiv.org/abs/1907.00670) — async variable timestep, 24.6–228.5×
- [Ros et al. 2006, EDLUT](https://direct.mit.edu/neco/article/18/12/2959/7115/)

**Parallel scan / SSM**
- [Bullet Trains, Morrill/Pehle/Zador ICML 2026](https://arxiv.org/abs/2603.13283) — affine-map scan, **exact** hard reset via speculation, 44×
- [FPT, Feng et al. ICML 2025](https://arxiv.org/abs/2506.12087) — fixed-point reset iteration, O(K), K≈3
- [SPSN, Yarga & Wood 2023](https://arxiv.org/abs/2306.12666) · [PSN, Fang et al. 2023](https://arxiv.org/abs/2304.12760) · [SpikingSSMs](https://arxiv.org/abs/2408.14909) — *all drop/approximate reset*

**Simulators**
- [GeNN, Nature Sci Rep 2016](https://www.nature.com/articles/srep18854) · [PyGeNN](https://www.frontiersin.org/articles/10.3389/fninf.2021.659005/full)
- [Brian2CUDA](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2022.883700/full) · [Brian2GeNN](https://www.nature.com/articles/s41598-019-54957-7)
- [Procedural connectivity, Knight & Nowotny 2021](https://www.nature.com/articles/s43588-020-00022-7)
- [NEST `iaf_psc_exp`](https://nest-simulator.readthedocs.io/en/latest/models/iaf_psc_exp.html) · [Brian2 numerical integration](https://brian2.readthedocs.io/en/stable/user/numerical_integration.html)

**Hardware / precision**
- [Hopkins et al. 2020, *Phil Trans R Soc A*](https://arxiv.org/abs/1904.11263) — stochastic rounding beats fp32
- [FeNN, Knight & Nowotny 2025](https://arxiv.org/html/2506.11760v1) — 16-bit fixed point, 79.5% vs 79.6% fp32
- [Gupta et al. ICML 2015](https://arxiv.org/abs/1502.02551) — limited numerical precision / update swamping

**Novelty gaps found (no hits across multiple query formulations)**
1. Uniform delay as the **parallel-scan block length** ← strongest claim
2. Parallel scan inside a fixed-Δt **simulator** (all scan work is ML training)
3. **Two-state (V,G)** scan — published scans are scalar-V
4. Exactness-preserving parallel simulation (only Bullet Trains qualifies)

---

## 9. Method — how to work on this

**Rule 1: every performance change must be proven bit-identical.**
`code/verify_pytorch_perf.py` compares spike trains (time + neuron index) between
old and new paths across 4 regimes. `flyloop/verify.py` does the same for the
research engine across 8 regimes. A change that alters output is an *accuracy*
change and must be argued separately (as §4 is).

**Rule 2: measure before believing.** Two of my own confident claims were wrong:
- "memory-bound" — true densely, false at active-set scale (dispatch-bound)
- "SparseProp will give ~11×" — the math says it won't; a scout caught it *before*
  implementation.

**Rule 3: ask research questions that can come back negative.** The SparseProp and
graph-reordering investigations both returned "don't do this", each saving days.

**Rule 4: sparsity is stimulus-dependent.** Always test the broad-stimulation and
whole-brain regimes, not just sugar.

---

## 10. Next steps, in order

1. **Exact propagator + dt = 0.2 ms.** `gcd(1.8, 2.2) = 0.2`, so 0.2 ms is the
   largest step keeping delay *and* refractory exact integers. **Free 2×.**
2. **dt = 1.8 ms via method of steps** (§5.4) + the quartic solver (already built
   and validated). **18×.** Requires off-grid refractory handling.
3. **σ-factorised delivery** (§5.5) — 18 scatter rounds → 2 sparse mat-muls.
4. **Validate against Brian 2** using their own `code/compare_ground_truth.py`.
   *This is the gate that turns our numbers into a mergeable PR.*
5. Re-time on an **idle** machine before quoting anything publicly.
6. Only then: GPU port, warp-aggregated atomics, CUDA Graphs / persistent kernels.

**Not yet done:** fork not created (`gh auth login` pending — **use HTTPS**, Git
Credential Manager is already configured). No PR opened. Nothing pushed publicly.
