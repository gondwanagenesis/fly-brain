---
field: Parallel discrete-event simulation and lookahead
date: 2026-07-30
verdict: The uniform 1.8 ms delay IS a Chandy-Misra-Bryant lookahead bound equal to the whole window, which licenses a zero-communication barrier once per 1.8 ms instead of once per 0.1 ms step — thread the existing per-window kernel across the 4 physical cores of the i7-1185G7, partitioned by contiguous neuron-index range, exchange only cross-partition spike events at the barrier.
---
# Parallel discrete-event simulation and lookahead, applied to whole-brain LIF on 4 cores

## Bottom line

This field's one load-bearing idea — **lookahead** — is already the mathematical
object §5.4/§5.5 of ROADMAP.md built (uniform 1.8 ms delay, method of steps,
σ-factorisation). What it has *not* yet been used for is **threading**: the
lookahead theorem says a partition of neurons can run an entire 1.8 ms window
with **zero reads of any other partition's state**, so 4 disjoint index-ranges
can run on 4 physical cores with one barrier per window (18 grid steps or 1
analytic window-step) instead of one per step — an 18× reduction in
synchronisation points, for free, with **provably zero risk of a causality
error inside the window** (not "usually safe" — provably safe, because the
delay is uniform and verified against all 15,091,983 edges). This is not a new
result in the field (NEST/NEURON have parallelised min-delay networks across
MPI ranks since 2005/2008) but it has not yet been applied *within this
kernel*, on *shared-memory cores*, *combined with the exact analytic
propagator* that avoids both grid-detection spike loss (§4) and the
delivery-ring-buffer traffic (§5.5). Time Warp / optimistic synchronization is
actively the wrong tool here: it exists to hide a *small or unknown*
lookahead behind speculative rollback, and this problem has the opposite —
a *large, exact, structural* lookahead that makes rollback strictly worse
(adds a correctness-risk surface for a bit-identical gate that conservative
sync doesn't need at all). Arithmetic below projects 3–3.5× from 4-core
partitioning (bandwidth-limited, not 4× linear) — enough on its own to reach
real time from the current 0.219 ms/step, but **this is a model, not a
measurement**, per the repo's own Rule 2.

## Applicable techniques

### 1. Chandy-Misra-Bryant (CMB) conservative synchronization
Chandy, K.M. & Misra, J. (1979). "Distributed Simulation: A Case Study in
Design and Verification of Distributed Programs." *IEEE Trans. Software Eng.*
SE-5(5), 440–452. https://dl.acm.org/doi/10.1109/tse.1979.230182 · independently,
Bryant, R.E. (1977). "Simulation of Packet Communication Architecture Computer
Systems." MIT-LCS-TR-188. https://dspace.mit.edu/handle/1721.1/149478 ·
survey: Misra, J. (1986). "Distributed Discrete-Event Simulation." *ACM
Computing Surveys* 18(1), 39–65. https://dl.acm.org/doi/10.1145/6462.6485

**Core idea.** Model the simulation as logical processes (LPs) exchanging
timestamped events with no shared clock. An LP may safely process (or advance
past) all events up to time T only once it can *guarantee* no message with
timestamp < T can still arrive from any other LP — otherwise it risks
processing events out of causal order. CMB's original mechanism enforces this
with null messages: each LP periodically tells its neighbours a lower bound on
its own next possible output, even when it has nothing real to send, purely so
neighbours can compute their own safe-advance time.

**Key relation.** LP *i* may advance to `T + L_i` without further input where
`L_i` is a *lookahead*: a proven lower bound on the timestamp of the next
message *i* can ever emit, given its clock is at `T`.

**Applied here.** Each of the 4 core-partitions is an LP. Because every edge
has *exactly* 1.8 ms delay (verified, ROADMAP §5.4 precondition table: 0
autapses, 0 zero-delay edges, uniform), `L_i = 1.8 ms` for every partition,
unconditionally — no null messages are even needed, because the bound doesn't
need computing at runtime, it's a static model fact. This is CMB in its
simplest, strongest form: static, uniform, network-wide lookahead.

**Expected effect.** Not a speedup by itself — a *removal of unnecessary
synchronisation*, which is the precondition for the multicore speedup below.

**Fidelity verdict.** Exact. CMB's safety condition is a *sufficient*
condition for zero causality error; it introduces no approximation, no
reordering of any computation that already happens, only a change to when
partitions are allowed to observe each other's output.

---

### 2. Fujimoto's lookahead framework
Fujimoto, R.M. (1990). "Parallel Discrete Event Simulation." *Commun. ACM*
33(10), 30–53. https://dl.acm.org/doi/10.1145/84537.84545 · textbook: Fujimoto,
R.M. (2000). *Parallel and Distributed Simulation Systems*. Wiley.
https://www.amazon.com/dp/0471183830

**Core idea.** Fujimoto formalises lookahead as *the* variable governing
conservative-algorithm efficiency: performance scales with the ratio of
lookahead to event granularity, because larger lookahead lets LPs run further
ahead of the global synchronisation point before blocking. He also identifies
"conservative time-stepped" simulation — where *all* LPs advance by a fixed Δt
and exchange at each Δt boundary — as the degenerate, easiest-to-implement
case of conservative PDES when a usable global lookahead bound is known in
advance.

**Key quantity, computed for this problem.**
```
lookahead / step   =  1.8 ms / 0.1 ms  =  18
```
This is *exactly* the number already named in ROADMAP.md's key-references
table and in the task prompt. Fujimoto's framework says: this ratio bounds how
rarely you are *required* to synchronise. Barrier count can be reduced from
1-per-step to 1-per-18-steps with no loss of correctness.

**Expected effect (arithmetic).** A barrier/rendezvous between 4 cores costs
on the order of 1 µs (atomic countdown + memory fence; no network, no OS
IPC — this is shared memory, not MPI). At the current 0.219 ms/step:
```
per-step barrier overhead ratio   :  1 µs / 219 µs  ≈ 0.46%  (paid 18×/window)
per-window barrier overhead ratio :  1 µs / (18 × 219 µs) ≈ 0.025%  (paid 1×/window)
```
18× fewer rendezvous points also means 18× fewer chances for OS scheduling
jitter (preemption, page fault, SMT contention) on *any* one of the 4 threads
to stall all 4 — a per-step barrier exposes that risk 18 times per window,
a per-window barrier once.

**Fidelity verdict.** Exact — a scheduling change, not a numerical one.

---

### 3. YAWNS — windowed conservative synchronization
Nicol, D.M. (1993). "The Cost of Conservative Synchronization in Parallel
Discrete-Event Simulations." *J. ACM* 40(2), 304–333.
https://dl.acm.org/doi/10.1145/151261.151266

**Core idea.** YAWNS ("Yet Another Windowing Network Simulator") is exactly
Fujimoto's conservative-time-stepped case, analysed rigorously: LPs compute
forward to a common barrier time (the window boundary), exchange, repeat.
Nicol proves that for a broad class of models this windowed protocol is
**asymptotically optimal** among conservative algorithms once the window
matches the model's provable lookahead — there is no cleverer conservative
scheme that synchronises less often without losing safety. Independently
confirmed as already the standard structure used by neuronal-network
simulators: NEURON and NEST parallelise across MPI ranks using exactly this
windowed/min-delay pattern (see refs below) — this problem's window (the
axonal delay) plays the same role their `d_min` plays across ranks, just
applied here across shared-memory cores instead.

**Key algorithm, mapped onto this kernel:**
```
for each window m (length 1.8 ms):
    barrier                                   # only sync point
    each of 4 threads, independently, no reads across partitions:
        advance its N/4 neurons through the window
        (18 grid substeps of 0.1 ms, OR 1 exact analytic step — either works,
         see "why grid mode is also covered" below)
        record any spikes into ITS OWN outbox, keyed by destination partition
    barrier
    each thread reads the outbox entries destined for its partition,
    applies them as the known, exact input for window m+1
```

**Why grid mode is also covered, not just window mode.** The licensing
argument is about the delay (1.8 ms) exceeding the window length (also
1.8 ms here), not about whether the *internal* stepping inside a partition is
one analytic jump or 18 Euler/exact grid substeps. A spike produced inside
window *m* cannot be *consumed* inside window *m*, regardless of whether the
producer used 1 substep or 18 to compute it. So this threading scheme applies
unchanged to §2.10's "grid" mode and "window" mode both — it parallelises
whichever the ModeSelector already picked.

**Expected speedup, worked from measured numbers.**
- Current: 0.219 ms/step (`native_engine.py` / `lif_kernel.c`, single-threaded,
  dense/AVX-512, per HANDOFF.md).
- Working set 2.2 MB (v, g, refrac, refrac_steps × N, fp32) fits the *shared*
  12 MB L3 today. Split 4 ways, each partition's slice is **0.55 MB — fits
  the *private* 1.25 MB L2 per core**, so 4-way partitioning is also a cache
  locality win independent of core-count parallelism.
- Achieved bandwidth today: 4.4 MB touched (read+write of the state) per
  0.219 ms step ⇒ ≈ 20 GB/s realised, against a 68 GB/s LPDDR4x ceiling
  (measured aggregate, not necessarily achievable in full by one core). If 4
  cores each demand ≈20 GB/s concurrently, aggregate demand (≈80 GB/s) exceeds
  the 68 GB/s ceiling, so the realistic ceiling is **≈68/20 ≈ 3.4×**, not 4×.
- Projected: `0.219 ms / 3.4 ≈ 0.064 ms/step` — **under the 0.1 ms real-time
  target**, with margin, *before* any sparse-activity win.
- ⚠️ This is arithmetic from measured single-core numbers, not a
  multi-thread measurement. Per Rule 2 of ROADMAP.md ("measure, don't
  model" — the window-stepping estimate was wrong three times in this exact
  way), **treat 3.4× and 0.064 ms as a hypothesis to benchmark, not a result.**

**What must be exchanged at the window boundary — precisely.** Only spikes
whose source is in partition *i* and whose synaptic target is in partition
*j ≠ i*, tagged with the source's exact intra-window spike time (needed by
the already-existing σ-factorisation, ROADMAP §5.5, to compute the correct
`exp(-s/tau_syn)` phase on arrival). This is a strict subset of what the
existing serial "deliver" step (§2.6, 4.14 ms at full scale, event-driven)
already computes — partitioning only requires *routing* those existing
per-edge deliveries into 4 disjoint destination arrays instead of 1, plus a
one-time startup pass that tags each connectome edge as intra-partition
(skip the barrier's outbox, deliver directly) or cross-partition (route
through the outbox). Same total delivery arithmetic, no new numerics.

**Fidelity verdict.** Exact / bit-identical, **conditional on one thing**:
the connectome partition boundary must not change *what* gets summed into any
neuron's `g`, only *when the memory write for it becomes visible to another
thread*. Since delivery is already event-driven addition of `w_k` into a
target's accumulator (linear, order-independent — additions commute), and the
barrier guarantees every partition's window-*m* spikes are fully recorded
before any partition begins reading them for window *m+1*, execution order
across threads cannot change the arithmetic result. This should be added as a
9th regime to `flyloop/verify.py` / a new mode in `code/verify_pytorch_perf.py`
before landing, per the repo's existing Rule 1.

---

### 4. Prior art: min-delay MPI parallelization in NEURON and NEST (context, not directly adoptable code)
Hines, M.L. & Carnevale, N.T. (2008). "Translating network models to parallel
hardware in NEURON." *J. Neurosci. Methods* 169(2), 349–363.
https://www.neuron.yale.edu/neuron/static/papers/jnm/parallelizing_models_jnm2008.pdf
· Morrison, A., Mehring, C., Geisel, T., Aertsen, A., Diesmann, M. (2005).
"Advancing the boundaries of high-connectivity network simulation with
distributed computing." *Neural Computation* 17(8), 1776–1801.
https://dl.acm.org/doi/10.1162/0899766054026648 · Jordan, J., Ippen, T.,
Helias, M., Kitayama, I., Diesmann, M., et al. (2018). "Extremely Scalable
Spiking Neuronal Network Simulation Code: From Laptops to Exascale
Computers." *Front. Neuroinform.* 12:2.
https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2018.00002/full

**Core idea.** Both simulators partition neurons across MPI ranks and
synchronise communication once per `d_min` (their minimum-delay-across-all-
synapses analogue of our 1.8 ms), i.e. this is the field's standard practice
for exactly this model class, already running in production on other
people's connectomes.

**Why it's "context" not "directly adoptable."** They synchronise
*communication batching* over MPI (network messages, serialisation, distinct
address spaces) with **grid-only Euler stepping inside the window** — no
analytic exact propagator, no certified-silence pruning, no σ-factorisation.
ROADMAP.md's own §4 already shows their-style grid stepping loses ~10% of
spikes vs a fine-grid reference at dt=0.1ms. The genuinely new combination
this repo can claim (flagged as "unclaimed research" in memory) is: exact
analytic within-window propagation (§5.1–5.3) **+** the NEST/NEURON-style
windowed barrier **+** shared-memory (no MPI serialisation cost) instead of
distributed ranks. NEST/NEURON solve "how do I not send a message every
step across a cluster"; this repo's version solves "how do I not synchronise
a shared cache line every step across 4 cores while staying numerically
exact," which is a different (cheaper, single-machine) engineering problem
with the same license from the same theorem.

**Fidelity verdict.** N/A — cited for precedent only, not proposed as code to
port (their Euler-grid inner loop is exactly what §4 shows loses spikes; do
not adopt that part).

## Ruled out from this field, and why

| Technique | Why not, specifically |
|---|---|
| **Time Warp / optimistic synchronization** (Jefferson, D.R. (1985). "Virtual Time." *ACM TOPLAS* 7(3), 404–425. https://dl.acm.org/doi/10.1145/3916.3988) | Time Warp trades synchronisation overhead for speculative execution + rollback (antimessages) when lookahead is *small or unknown*, so LPs would otherwise stall too often. Here lookahead is **large (18× the step), static, and exactly provable from the connectome** — there is nothing to hide behind speculation, and rollback adds a correctness-risk surface (state must be checkpointed and precisely unwound) that directly threatens the bit-identical gate this repo lives by. Strictly worse: more implementation risk, no upside, for a problem CMB already solves with zero risk. |
| **Null-message algorithm's runtime lookahead computation** (part of CMB, ibid.) | Null messages exist because in general LPs must *compute* a lookahead bound at runtime and it may vary. Here the bound (1.8 ms) is a **static model constant**, verified once at load time against all 15,091,983 edges (ROADMAP §5.4 precondition table). Sending null messages to establish a bound that's already a compile-time constant is pure overhead — use the static barrier directly. |
| **Bounded-lag algorithm** (Lubachevsky, B.D. (1988). "Bounded lag distributed discrete event simulation." *Proc. SCS Multiconf. on Distributed Simulation* 19(3), 183–191. https://dl.acm.org/doi/10.1145/63238.63247 (1989 CACM version)) | Generalises CMB to *state-dependent, runtime-computed* lookahead when no static bound exists. Strictly more general — and strictly more complex — than what's needed here, where the bound is static and uniform. Using bounded-lag machinery to compute what is already a known constant is solving a harder problem than the one that exists. |
| **MPI-based distributed PDES (general)** | This is a single 4-core/8-thread laptop with a shared 32 GB memory space, not a cluster. MPI's entire cost model (serialization, network RTT, per-rank address-space isolation) doesn't apply; a `std::barrier`/atomic-countdown costs ~1 µs vs MPI collective costs of ~10–100 µs+. Adopting MPI-style batching logic here would import overhead the hardware doesn't have. |
| **Heterogeneous-delay / per-edge lookahead (general CMB null-message case)** | Not applicable to *this* model (delay is uniform, verified) but flagged because it's the field's answer to ROADMAP §7 kill-risk #3 ("heterogeneous delays collapse the window to the minimum delay") — if the model ever gains non-uniform delays, the correct fallback is *not* to abandon this scheme but to shrink the global window to `min(delay)` across all edges (CMB's standard degradation), which the repo already anticipated. |

## Concrete next actions for this codebase

1. **`flyloop/native_engine.py` + `flyloop/native/lif_kernel.c`** — add a
   4-way static partition of the neuron index range `[0, 138639)` into 4
   contiguous chunks (≈34,660 neurons each, 0.55 MB state per chunk — fits
   L2). Spawn 4 pinned OS threads (not 8 — Tiger Lake's AVX-512 execution
   ports are shared per physical core, SMT siblings won't add AVX-512
   throughput), each running the existing per-window stepping loop
   unmodified except for the index range.
2. **New file, e.g. `flyloop/native/partition_barrier.c`** — implement the
   barrier as an atomic countdown + spin/futex wait (cheaper than
   `std::barrier<>` for 4 threads and avoids a libstdc++ dependency in the C
   kernel), called once per 1.8 ms window (once per 18 grid substeps, or once
   per analytic window-step, per whichever mode `ModeSelector` picked).
3. **Connectome preprocessing (one-time, at load)** — extend whatever builds
   the CSR/edge structures consumed by `native_engine.py` to tag each edge as
   intra-partition (source and target in the same 1/4 chunk) or
   cross-partition, and to allocate 4 disjoint per-destination-partition
   outbox arrays for cross-partition spikes. Intra-partition edges deliver
   directly with no synchronisation at all (majority of edges, since
   partitions are large contiguous chunks of a hub-dominated but not
   partition-correlated graph — measure the actual intra/cross ratio, it's
   free information once the tagging pass runs).
4. **`flyloop/verify.py`** — add a 9th regime: run the 4-thread partitioned
   kernel against the existing serial kernel across all current regimes
   (1 neuron → whole-brain) and assert bit-identical spike trains. This is
   the actual gate; nothing above should be treated as done until this
   passes, per the repo's Rule 1.
5. **Benchmark, not model, the achieved speedup** — measure wall-clock
   ms/step for the 4-thread version on the actual i7-1185G7, on an idle
   machine (ROADMAP §10 item 9 already flags this same caveat for other
   numbers). Compare against the 3.4× / 0.064 ms projection above; if the
   shared LPDDR4x bus is the binding constraint the achieved number could be
   lower — this is exactly the kind of estimate Rule 2 warns has been wrong
   before (window-stepping's 18×→0.77×→1.05×→5.7× history).
6. **Defer, don't block on**: load-imbalance across partitions. The *current*
   dense/grid kernel does the same fixed op-count per neuron regardless of
   whether it spikes, so a static contiguous-index partition is
   automatically load-balanced today. This only becomes a real risk if/when
   ROADMAP §10 item 5 (wiring the sparse active-set `ModeSelector` into
   production) lands *and* stimuli are localized enough to cluster all
   activity in one partition (§2.2's 91% quiescence is not uniformly
   distributed — 96.8% of the brain is within 4 hops of the 21 sugar GRNs,
   which could concentrate in one index range). If/when that happens, the
   fix is a connectome-community-aware partition (balance by expected
   activity, not raw index count) rather than anything from this field.

## References

- [Chandy & Misra 1979, IEEE TSE](https://dl.acm.org/doi/10.1109/tse.1979.230182) — origin of conservative (CMB) synchronization; safety condition via lookahead/null messages.
- [Bryant 1977, MIT-LCS-TR-188](https://dspace.mit.edu/handle/1721.1/149478) — independent origin of the same algorithm (hence "CMB").
- [Misra 1986, ACM Computing Surveys](https://dl.acm.org/doi/10.1145/6462.6485) — accessible survey of distributed DES and the lookahead concept.
- [Fujimoto 1990, CACM](https://dl.acm.org/doi/10.1145/84537.84545) — canonical PDES survey; defines lookahead formally, names conservative time-stepping as the easy case.
- [Fujimoto 2000, *Parallel and Distributed Simulation Systems*, Wiley](https://www.amazon.com/dp/0471183830) — textbook treatment, null-message algorithm detail.
- [Nicol 1993, J. ACM](https://dl.acm.org/doi/10.1145/151261.151266) — YAWNS windowed conservative algorithm; proves asymptotic optimality of window = lookahead.
- [Jefferson 1985, ACM TOPLAS](https://dl.acm.org/doi/10.1145/3916.3988) — Time Warp / optimistic synchronization; why it's the wrong tool when lookahead is large and known.
- [Lubachevsky 1988/1989, bounded lag](https://dl.acm.org/doi/10.1145/63238.63247) — generalization to runtime-computed lookahead; strictly more machinery than a static uniform delay needs.
- [Hines & Carnevale 2008, J. Neurosci. Methods](https://www.neuron.yale.edu/neuron/static/papers/jnm/parallelizing_models_jnm2008.pdf) — NEURON's MPI min-delay parallelization; prior art for the barrier structure, not the exact numerics.
- [Morrison et al. 2005, Neural Computation](https://dl.acm.org/doi/10.1162/0899766054026648) — NEST's distributed-computing min-delay communication scheme.
- [Jordan et al. 2018, Front. Neuroinform.](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2018.00002/full) — NEST at exascale, same windowed-communication pattern at much larger scale.

## Reviewer notes

**Citation verification (2026-07-30):** All top 5 citations verified against academic databases and open-access archives:
- ✓ Chandy & Misra 1979 (IEEE TSE SE-5(5), 440–452) — foundational CMB algorithm
- ✓ Bryant 1977 (MIT-LCS-TR-188) — independent parallel-simulation discovery
- ✓ Fujimoto 1990 (Commun. ACM 33(10), 30–53) — canonical PDES framework and lookahead formalization
- ✓ Nicol 1993 (J. ACM 40(2), 304–333) — YAWNS windowed protocol, asymptotic optimality proof
- ✓ Hines & Carnevale 2008 (J. Neurosci. Methods 169(2), 349–363) — NEURON MPI min-delay parallelization

**Arithmetic audit:** All speedup and bandwidth calculations verified:
- Lookahead/step ratio: 1.8 ms / 0.1 ms = 18 ✓
- Per-step barrier overhead: 1 µs / 219 µs ≈ 0.46% ✓
- Per-window barrier overhead: 1 µs / 3,942 µs ≈ 0.025% ✓
- Per-partition working set: 2.2 MB / 4 cores = 0.55 MB, fits 1.25 MB L2 ✓
- Measured bandwidth: 4.4 MB / 0.219 ms ≈ 20.1 GB/s ✓
- LPDDR4x ceiling: 68 GB/s; realistic multicore speedup = 68 / 20.1 ≈ 3.38× ✓
- Projected per-step time: 0.219 ms / 3.38 ≈ 0.065 ms (sub-0.1 ms target with margin) ✓

**No fabricated citations or arithmetic errors detected.** Document is citation-clean and numerically sound. Caveat remains (Rule 2: "measure, don't model") — 3.4× projection is a hypothesis pending empirical benchmark on i7-1185G7.
