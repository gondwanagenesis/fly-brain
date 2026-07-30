---
title: Synthesis — what to build next for real-time whole-brain Drosophila LIF
date: 2026-07-30
inputs: 8 research notes + HANDOFF.md + flyloop/ROADMAP.md + research/CORRECTION_benchmark_units.md + research/audit/*.py (re-run this session) + data/benchmark-results.csv
verdict: Real time is ALREADY REACHED on an idle machine (0.0543 ms/step = 1.84x) and MISSED on a loaded one (0.129 ms/step = 0.77x, measured today). The remaining work is not "get to real time", it is "get 3x of margin so real time survives a laptop that is also doing something else" — and, more urgently, three unfixed fidelity defects found by this repo's own audit scripts that no note or handoff mentions.
---

# Synthesis: the arithmetic, the ranked plan, and the three things nobody wrote down

> **Measurement environment warning (read first).** Every number I measured today
> was taken on a machine under heavy concurrent load, and **another session was
> actively rewriting `flyloop/native/lif_kernel.c` and `flyloop/native_engine.py`
> while I measured** (`git status` shows both modified; one build failed
> mid-synthesis; the file gained runtime AVX-512/AVX2 CPUID dispatch between my
> first and second read). Treat my absolute numbers as ~2.4x inflated and my
> ratios as sound. The committed reference point is git `5dbd161`.

---

## 1. Arithmetic of the target

### 1.1 What 0.1 ms/step demands

`dt = 0.1 ms` ⇒ 10,000 steps per simulated second ⇒ **budget = 100 µs/step**
for N = 138,639 neurons.

Per neuron-step the fused kernel executes (counted directly off
`lif_kernel.c:sweep_scalar`, not estimated): `refrac+1`, `g*c_decay`,
`v_rest-v`, `+g`, `fma(t,c_mem,v)` (2 FLOP), `v>v_th` — **≈ 7 FLOP**.

| Resource | Demand at 100 µs/step | Machine capability | Utilisation required |
|---|---|---|---|
| **FLOP/s** | 138,639 × 7 / 100 µs = **9.7 GFLOP/s** | 4 cores × 16 lanes × 2 × 3.0 GHz = **384 GFLOP/s** | **2.5%** |
| **Bytes/s** | read v,g,refrac + write v,g,refrac = 24 B/neuron ⇒ 3.36 MB/step / 100 µs = **33.6 GB/s** | LPDDR4x-4266 measured all-core **57–68 GB/s**; per-core L2 ≈ 10x that | **49–59% of DRAM, ~5% of L2** |
| **Working set** | v,g,refrac,refrac_steps fp32 = **2.218 MB** | 1.25 MB L2/core, 12 MB shared L3 | fits L3 whole; **416–554 KB/thread fits L2** |
| **Delay line** | sparse (index,value), ~190 nnz | ~**30 KB** resident (was 10.5 MB dense) | trivial |
| **Connectome** | 15,091,983 edges, 122 MB, read-only | ~1.5 KB touched/step at sugar activity | irrelevant to the step |

**Conclusion: the target is not remotely FLOP-bound (2.5% of peak) and is not
DRAM-bound either, because the entire live state fits in L2 once partitioned
4 ways.** The 100 µs budget is generous by roughly an order of magnitude
against both rooflines. Any failure to hit it is an implementation failure,
not a hardware one.

### 1.2 Where we actually are

| | ms/step | s/sim-second | vs real time | source |
|---|---|---|---|---|
| PyTorch dense (this repo, optimised) | 1.6592 | 16.59 | 0.06x | HANDOFF §11.1, idle |
| native ×1 thread | 0.0822 | 0.82 | 1.22x | HANDOFF §11.1, idle |
| native ×2 threads | 0.0790 | 0.79 | 1.27x | HANDOFF §11.1, idle |
| **native ×4 threads** | **0.0543** | **0.543** | **1.84x** | HANDOFF §11.1, idle |
| native ×4, **measured today under load** | 0.129 | 1.29 | **0.77x — MISSES** | `bench_native.py`, this session |
| pure kernel ×1 / ×4, today | 0.1252 / 0.0627 | — | — | scratch harness, this session |

Efficiency achieved at 0.0543 ms/step:

- **61.9 GB/s** effective (3.36 MB ÷ 54.3 µs). This *exceeds* the LPDDR4x
  ceiling, which is the positive proof that the traffic is being served from
  L2/L3 and the sparse-delay-slot change did what it claimed.
- **17.9 GFLOP/s = 4.7% of peak.**
- **0.392 ns/neuron-step = 1.17 core-cycles/neuron-step across 4 cores**, i.e.
  **4.7 cycles/neuron-step per core** for ~7 ops on a 16-wide unit. The loop is
  retiring roughly one vector instruction every 10 cycles per core.

### 1.3 Is real time achievable here? Yes — and there is ~5x still on the table

Floor for this kernel on this chip, two independent bounds:

- **Issue-limited:** 16 neurons/iteration, ~8 vector instructions/iteration,
  IPC 2 ⇒ 4 cycles/iteration. 8,665 iterations ÷ 4 threads = 2,166/thread ×
  4 cycles = 8,664 cycles = **2.9 µs/step**.
- **L2-bandwidth-limited:** 3.36 MB ÷ (4 cores × ~100 GB/s L2) = **8.4 µs/step**.

Realistic floor ≈ **8–12 µs/step = 8–12x real time**. We are at 54 µs.
**About 5x of pure engineering headroom remains on the i7-1185G7 itself**,
before any algorithm change and before any activity-dependent path.

**Plain answer:** real time on this consumer laptop is achievable, has been
achieved, and is currently *fragile* — it holds at 1.84x on an idle machine
and fails at 0.77x on a machine that is also compiling something. The
engineering goal should be restated as **≥3x margin (≤33 µs/step)**, which the
floor analysis says is comfortably reachable.

### 1.4 Honest external scale

Two numbers matter, both from this repo's own data, not from a paper:

| Platform | s/sim-second | vs real time | source |
|---|---|---|---|
| **GeNN on an RTX 4070** | **0.5415** | 1.85x | `data/benchmark-results.csv`, n_run=1, t_run=1, median of 6 |
| **This laptop, native ×4, no GPU** | **0.543** | 1.84x | HANDOFF §11.1 |
| NEST GPU (RTX 4070) | 1.0655 | 0.94x | same csv |
| Brian 2 C++ standalone (host CPU) | 2.9035 | 0.34x | same csv |
| PyTorch CUDA (RTX 4070) | 9.7865 | 0.10x | same csv |
| PyTorch CPU (upstream, host CPU) | 686.27 | 0.0015x | same csv |
| Loihi 2, 12 chips, dt=0.1 ms | 0.0538 | 18.6x | Wang et al. 2025 Table 1 (9-bit weights, lossy) |

**A 4-core laptop CPU at full fp32 bit-identical fidelity now ties a
discrete-GPU GeNN build to within 0.3%, on the same model and the same
protocol.** That is the headline result of this entire project and it is
currently written down nowhere except a stale correction file. Loihi 2 remains
**~10x ahead** (not 62x — see §5, contradiction C3) and is lossy.

**Hardware that clears the bar with real margin, if this laptop is ever judged
too fragile:** AMD Strix Halo (16 Zen 5 cores, same 512-bit AVX-512 FPU,
212–215 GB/s measured) projects **4–7x** by pure core scaling of the existing
kernel — the single highest-confidence port target, and it needs no new
algorithm. Any RTX-class GPU via GeNN already sits at 1.85x. Nothing else in
the consumer space (iGPU, NPU, Apple AMX) clears it: see §4.

---

## 2. Ranked opportunity table

Ranked by **(impact × confidence) / effort**. Impact is the multiplier on
ms/step. "Exact" = provably no arithmetic change. "Bit-identical" = arithmetic
changes representation but the gate proves identical output.

| # | Technique | Speedup | Fidelity | Hours | Conf | File to change | Source note |
|---|---|---|---|---|---|---|---|
| 1 | **Fix `threads != n_chunks` → silent serial fallback.** `sweep_chunks` runs serially unless `nthreads == n_chunks` exactly, but `lif_set_threads(2)` still spawns a worker that spins forever on a generation counter that never increments. **Measured: threads=2 is 0.89x — a real slowdown** (reproduced twice). Let workers take multiple chunks. | 1.0x at 4 threads, **1.4–1.8x at 2 threads**; removes a burned core | exact | 1 | 0.95 | `lif_kernel.c` `sweep_chunks` | measured this session |
| 2 | **Guard `put16` on a non-zero mask.** The bitset read-modify-write executes for **every** 16-neuron group (8,665/step) even when the spike mask is 0, on a 17 KB buffer shared by all 4 threads, with a boundary-word branch each time. At 1.75 spikes/step, ≥99.98% of those RMWs write zero. One `if (sp)` skips them. | 1.15–1.4x | exact | 2 | 0.6 | `lif_kernel.c` `sweep_avx512` | measured/derived this session |
| 3 | **Diagnose why 4-thread scaling is 1.8–2.0x, not the 3.4–4.0x that four separate notes projected.** Not a speedup — it is the gate on items 2, 4, 5. VTune / `perf` counters on the sweep alone (a scratch `probe_sweep_only` entry point already exists in my scratchpad and can be lifted). | information | n/a | 4 | 1.0 | new bench | §5 C1 |
| 4 | **Drop `refrac` from the dense sweep.** Only ever consumed as `refrac >= refrac_steps`, a monotone comparison. Replace the 554 KB fp32 array with a gate bitset (17 KB) + a sparse countdown list (≤40 entries at sugar activity). 24 B → 16 B per neuron = **33% traffic cut**. | 1.15–1.3x | exact (monotonicity proof); gate must compare the *derived* gate, not raw `refrac` | 6 | 0.7 | `lif_kernel.c`, `native_engine.py`, `verify_native.py` | HANDOFF §11.6.1; memory-layout §2 |
| 5 | **64-byte-aligned allocation + aligned load/store.** numpy gives 64-byte alignment only by luck; every `loadu`/`storeu` that straddles a line costs a split. Allocate v/g/refrac with `_aligned_malloc` and keep chunk starts at multiples of 16. | 1.0–1.2x | exact | 2 | 0.5 | `native_engine.py`, `lif_kernel.c` | compilers-codegen action 5 |
| 6 | **Sustained-run thermal/power measurement.** Every published number is min-of-15 × 500-step blocks. A 15 W U-series part running 10,000 steps/second **continuously for 60 s has never been measured.** This decides whether "1.84x real time" is a real capability or a burst. | information | n/a | 2 | 1.0 | new bench | nobody wrote this down |
| 7 | **Delete `refrac_steps` as a dense N-array.** It is binary (22 or 0) and the sweep never reads it — only the ~190-entry sparse delayed-input loop does. Frees 554 KB of L3 pressure; ~0x on the hot path. | 1.0–1.05x | exact | 3 | 0.9 | `lif_kernel.c`, `native_engine.py` | memory-layout §1 |
| 8 | **`uint16` connectome weights.** Verified against `data/fanout_csc.pt`: all 15,091,983 values are exact integers in [−2405, 1897], fits int16 with 13x margin. 0x at sugar activity (fan-out is 0.14%) but up to ~1.3x in the saturating regime where fan-out dominates (HANDOFF §11.6.2). | 1.0x sparse / 1.2–1.3x saturating | exact / lossless (verified on the real data) | 4 | 0.8 | connectome load path | memory-layout §6 |
| 9 | **Brian 2 ground-truth validation** via the repo's own `code/compare_ground_truth.py`. Not a speedup; it is the outstanding scientific gate and it is now *more* important because §11.3 shows the PyTorch reference has its own numerical quirks and the audit (§4 below) shows a 2.0 ms delay. | 0x | this is the fidelity gate | 8 | 1.0 | `code/compare_ground_truth.py` | HANDOFF §10.7, all notes |
| 10 | **Machine-check `KAPPA_WINDOW_MAX` with Gappa/FPTaylor.** Converts "0/20,000 sampled false-silence certifications" into a proof. Offline, no runtime cost. *I re-ran `audit/a4_kappa.py`: the bound is exact at D = 1.8 ms because the interior peak sits at 9.24 ms, outside the window — so the current constant is sound and this is hardening, not a bug fix.* | 0x | upgrades empirical → proved | 6 | 0.7 | `code/window_step.py` | numerical-exactness §9 |
| 11 | **Delay-window (v,g) affine-map parallel scan.** The one genuinely unclaimed algorithmic item. 1.00x at ≥15% activity (auto-falls back), up to 26x at 0.88%. **Currently broken** — `audit/a3_lost_spike.py` shows the window repair loses a spike the reference emits. | 1.0x dense / up to 26x sparse | claimed exact, **empirically NOT exact today** | 40+ | 0.3 | `code/window_*.py` | ROADMAP §5.4; contradicted by audit |
| 12 | **iGPU / SYCL / Level Zero port.** Do not. Launch latency alone is 33–156 µs against a 100 µs budget; Tiger Lake Xe-LP is outside PyTorch XPU support entirely. | ≤1.0x | moot | 40+ | 0.9 (that it fails) | — | igpu-and-consumer-hardware |

**Not on this list, deliberately:** ISPC/Halide/TVM/torch.compile (three
independent sources put SPMD compilers at 97–100% of hand intrinsics — no
headroom); graph reordering; synapse pruning; faster SpMV; hierarchical tile
bitsets (mean tile occupancy 0.0008 at measured activity — 64x waste);
Elias-Fano/WebGraph/k2-trees (no locality, decode cost lands on the 0.14%
path); ReproBLAS on the dense sweep (there is no reduction to reorder).

---

## 3. Critical path

Real time is reached. The path below targets **robust real time: ≤33 µs/step
(3x margin)**, which is what survives a laptop that is also doing other work.

| Step | Action | Expected ms/step after | How verified |
|---|---|---|---|
| 0 | **Baseline on an idle machine, sustained 60 s** (opportunity 6). Kill background load, disable turbo-boost variance, run continuously. | 0.0543 confirmed **or the claim is withdrawn** | `bench_native.py` extended to a continuous run; report min, median AND last-decile |
| 1 | Fix the `threads != n_chunks` serial fallback (opp. 1) | 0.0543 at 4 threads; 2-thread stops being a regression | `verify_native.py` 8 regimes at threads ∈ {1,2,3,4}; assert bit-equal across all |
| 2 | Diagnose the scaling gap (opp. 3) — this is where the plan either accelerates or stops | no change; produces the number | VTune counters: L2 miss rate, split loads, store-buffer stalls, `pause` cycles in the spin barrier |
| 3 | `put16` zero-mask guard (opp. 2) | **0.0434** (×1.25) | `verify_native.py` full-state memcmp every step, 8 regimes × 800 steps |
| 4 | Drop `refrac` from the dense sweep (opp. 4) | **0.0362** (×1.20) | same gate, but comparing the *derived* gate bit rather than the raw `refrac` float — the gate script must be edited first, and that edit reviewed |
| 5 | 64-byte aligned buffers (opp. 5) | **0.0335** (×1.08) | same gate; also re-run `lif_has_avx512` and confirm the AVX-512 path is still selected |
| 6 | Whatever step 2 found | **0.0291** (×1.15) | same gate |

**Honest multiplication.** Naively 1.25 × 1.20 × 1.08 × 1.15 = **1.86x**, giving
0.029 ms/step = 3.4x real time. But items 3, 4, 5 all attack the *same*
bottleneck (memory traffic and issue pressure in the same inner loop), so their
multipliers are not independent and the product must be haircut. **Realistic
landing: 0.034–0.042 ms/step = 2.4–2.9x real time on an idle machine**, and
roughly **1.3–1.6x on a machine under the load I measured today.**

That clears "robust real time" only just. If step 2 reveals the scaling gap is
L3/ring saturation rather than a fixable inner-loop issue, the ceiling on this
chip is ~0.040 ms/step and further gains require either the activity-dependent
path (opportunity 11, currently broken) or different silicon (Strix Halo).

**Parallel track that is not optional:** opportunities 9 and the three fidelity
defects in §4 must land regardless of any performance work. A fast simulator
with a 2.0 ms delay where the model specifies 1.8 ms is not a faster
simulation of the Shiu model; it is a fast simulation of a different model.

---

## 4. What would make this impossible, ranked by likelihood

**1. The machine is not idle. (Likelihood: certain. Already happened today.)**
Measured 0.129 ms/step under load versus 0.0543 idle — a 2.4x inflation that
takes the result from 1.84x real time to **0.77x, i.e. failure**. No note
addresses this. "Real time on a consumer laptop" is only a true claim if it
holds while the laptop is being a laptop. **Mitigation:** the 3x-margin target
in §3, plus a thread-count-aware fallback that degrades gracefully.

**2. Sustained thermal/power throttling. (Likelihood: high. Never measured.)**
Every published figure is the minimum of 500-step blocks on a 15 W U-series
part. A real embodied run is minutes of continuous 10,000 steps/second with
4 cores spinning (the barrier burns cores by design). Nobody has run this for
60 seconds and looked at the last decile. This is the cheapest unmeasured risk
in the whole project and it could invalidate the headline outright.

**3. Broad or tonic drive. (Likelihood: high; severity moderate — lower than
the notes claim.)** The dense kernel is activity-invariant, so 0.0543 ms/step
does *not* degrade with activity — that is the native kernel's structural
advantage over Loihi 2 and STACS, both of which degrade 5–31x from 0.5 Hz to
40 Hz. What *does* degrade is the fan-out: at 946 spikes/step it is ~104k
synapse updates/step, and HANDOFF's own headline (§2.1, "synaptic propagation
is 0.14%") **stops holding**. HANDOFF §11.1 already measures this: 1.23x over
PyTorch in the saturating regime versus 10.3x sparse. Opportunity 8 (uint16
weights) is the direct mitigation and it is currently ranked too low for this
reason.

**4. Three unfixed fidelity defects, found by this repo's own audit scripts,
which I re-ran this session and which appear in no note and no handoff.**
(Likelihood: they are already true.)

- **The implemented axonal delay is 2.0 ms, not 1.8 ms.**
  `research/audit/a5_delay.py`, re-run: `L = int(1.8/0.1) + 1 = 19` slots, and
  an end-to-end probe shows a spike emitted at the end of step 0 first raises
  `g` at the target on step 20 — **20 steps = 2.0 ms**. Both `brain_engine.py`
  and `run_pytorch.py` have `buflen = 19`. This is a **+11% delay error**, which
  is precisely the error HANDOFF §8 criticises Loihi 2's dt = 1 ms mode for
  committing. It also means the §5.4 method-of-steps window is 20 steps, not
  18, and the refractory-vs-delay margin is 2.2 vs 2.0 ms, not 2.2 vs 1.8. Every
  "1.8 ms" in the window math, the novelty claim, and the lookahead arithmetic
  is off by two steps.
- **The delay-window simulator loses spikes.** `research/audit/a3_lost_spike.py`,
  re-run: "neuron 2 spike present in reference, absent from window sim: **True**".
  The repair solves the *input-free* trajectory, so mid-window arrivals are
  invisible to the spike-time solver. HANDOFF §4's table ("window dt = 1.8 ms:
  158 spikes, count error 0, mean rate error 0.0000 Hz") is a result on one
  benign protocol, not a general exactness property. The window path is **not
  exact** as shipped.
- **The active→dense fallback drops spikes.** `research/audit/a1_fallback.py`,
  re-run at whole-brain scale: dense 51,251 spikes vs active 51,144, first
  divergence at t = 2.7 ms. And `audit/a9_verify_gaps.py` shows why the gate
  misses it: the sugar regime **never takes the fallback path at all**, and in
  the regimes that do, `refrac` differs on 138,288 of 138,639 neurons (max diff
  3000) between the two paths. Rule 1's "bit-identical across 8 regimes" is a
  weaker statement than it reads: `verify.py` compares spikes for the active
  path, not full state, and the regimes do not cover the branch that breaks.

  *(Good news from the same audit: `a11_cross_engine.py` confirms
  `brain_engine.py` is bit-identical to upstream `run_pytorch.py` for both
  dense and active modes at small scale — max |v diff| = 0, max |g diff| = 0.
  And `a4_kappa.py` confirms `KAPPA_WINDOW_MAX` is exact over the actual
  window. So the foundations are sound; the defects are in the optional
  fast paths.)*

**5. A `torch` upgrade. (Likelihood: moderate; severity total-but-recoverable.)**
Bit-identity is conditional on ATen's `at::parallel_for` chunk seams, which are
a function of `torch.get_num_threads()` and of the ATen build. Intel's own term
for this tier is *conditional* numerical reproducibility. The golden spike
train and the `chunks[]` array must be pinned to a recorded torch version +
thread count, or a future upgrade produces a mysterious gate failure.

**6. Concurrent editing of the same files.** Already caused one build failure
during this synthesis. `native_engine.py` gained runtime CPUID dispatch and
lost `-march=native` between my first and second read of it. Without a
build-and-gate step after each edit, the AVX-512 path can silently stop being
compiled and the kernel drops to scalar with no error.

**7. Model changes: heterogeneous delays, or true conductance coupling
`g·(E_rev − v)`.** Both invalidate the window/propagator mathematics (not the
fused kernel, which would survive). Verified today as not present.

---

## 5. Contradictions between notes, and my adjudication

**C1 — Threading: four notes projected 3.4–4.0x; measured is 1.8–2.0x.**
`hpc-simd-threading` §1 gives Amdahl 3.983x; `parallel-des-lookahead` §3 gives
3.38x from a bandwidth argument; `compilers-codegen` §5 gives 4x ideal / 2.4x
at 60% efficiency; `numerical-exactness` §1 says "3–4x of unused headroom".
Measured, twice, on the pure kernel with the Python and RNG paths removed:
**1.81x and 2.00x**. End-to-end, HANDOFF §11.1's own numbers give
0.0822 → 0.0543 = **1.51x**.
**Adjudication: all four notes are wrong in the same direction, for the same
reason.** They applied Amdahl to the 99.86% figure, but that number is a ratio
of *operation counts* (dense update vs synaptic propagation), not a ratio of
*time*, and they assumed the whole 0.219 ms was the parallel sweep. It is not:
`lif_step` also does a 17 KB `memset`, a 2,166-word bitset scan, a 17 KB
`memcpy`, and the sparse delay loop, all serial, plus ~1.6 µs of ctypes glue.
Solving `s + p = 1, s + p/4 = 1/1.9` gives **s ≈ 30% serial**, so the Amdahl
ceiling with infinite cores is ~3.3x, not 715x, and the achieved 1.9x is 58% of
*that*. **Verdict: 4-core threading is a ~1.8–2.0x technique on this kernel, it
is already banked, and no further multicore gain should be budgeted until the
serial residue is attacked directly (opportunities 2 and 4 do exactly that).**

**C2 — `hpc-simd-threading` §2 says partition boundaries MUST be 64-aligned or
the spike-bitset RMW is a data race. The shipped code uses non-aligned
boundaries (34,660 / 69,320 / 103,980).**
**Adjudication: the note is wrong and its recommendation must not be
implemented.** 64-aligning the boundaries would *break bit-identity*, because
the boundaries must mirror ATen's `ceil(N/nthreads)` partition to reproduce the
FMA/non-FMA seams at the chunk tails (HANDOFF §11.3). The shipped code resolves
the real race correctly and more cheaply: `or_word` does an atomic OR on the
two boundary words per chunk and a plain OR on every interior word — 2 atomics
per chunk instead of one per 16 neurons. The note identified a genuine hazard
and prescribed a cure that conflicts with the project's hardest constraint.

**C3 — `CORRECTION_benchmark_units.md` says the native kernel is 0.331 ms/step
(3.31 s/sim-second); HANDOFF §11 says 0.0543 ms/step. Same day, same machine.**
**Adjudication: CORRECTION is stale by exactly two commits** (`5e891f1`
threading + batched Poisson + cached pointers, and the follow-on work recorded
in `ab6e688`). Every derived claim in it is wrong by ~6x:
- "Loihi 2 @0.1 ms is **~62x ahead**" → corrected to **~10x** (0.0538 vs 0.543).
- "GeNN on RTX 4070 beats us 6x (0.542 vs 3.31)" → corrected to **a dead heat**
  (0.5415 vs 0.543, verified by re-querying `data/benchmark-results.csv` this
  session: n_run=1, t_run=1, median of 6 rounds).
- "4-core threading should give ~2.5–3.5x, i.e. ≈1.0 s/sim-second" → it gave
  1.51x and 0.543 s/sim-second; the projection was wrong on the multiplier and
  right on the destination for the wrong reason.
**Action: fold the surviving parts of CORRECTION into HANDOFF §8 and delete the
performance table from it, or someone will quote the 62x.**

**C4 — `snn-simulator-prior-art` §2's Auryn benchmark is fabricated.** Its own
reviewer flagged it: the note claims Auryn runs at "0.6x real-time" and derives
"41.7 ns/neuron-update" and hence "our kernel is 26.4x faster than the field's
most-cited hand-tuned single-core simulator." Zenke & Gerstner 2014 state
*faster* than real time. The arithmetic is internally consistent; the premise is
false.
**Adjudication: withdraw the 26.4x claim entirely.** Do not attempt to repair it
by hunting for the real Auryn number — there is a *better* comparator already in
this repo: GeNN on an RTX 4070 at 0.5415 s/sim-second, same model, same
protocol, same spike counts (16,978 vs 16,708–17,233 across backends, so they
are all simulating the same thing). Use that.

**C5 — `memory-layout` §2 (saturating uint8 `refrac`) vs HANDOFF §11.6.1 (drop
`refrac` from the dense sweep entirely).**
**Adjudication: do HANDOFF's version.** A gate bitset (1 bit/neuron) plus a
sparse countdown list strictly dominates a uint8 array (8 bits/neuron) and
removes the array from the hot loop rather than shrinking it. The memory note's
monotonicity proof — the only consumer is `refrac >= 22`, a monotone comparison
against a constant reachable far below saturation — is nevertheless the correct
correctness argument for *either* representation, and should be lifted verbatim
into the commit message.

**C6 — `memory-layout`'s headline ("crosses from fits-L3 to fits-L2") counts
`refrac_steps` in the working set, but the same note concedes the sweep never
reads it.** So 554 KB of the claimed saving is L3 *pressure*, not hot-path
traffic, and the L2-crossing narrative is measured against an array the inner
loop does not touch. **Adjudication: real but small.** Ranked #7, not #1.

**C7 — `igpu-and-consumer-hardware` and `compilers-codegen` both price the iGPU
at 98–102 µs using "4.4 MB/step ÷ 68 GB/s = 65 µs of memory".** On the CPU the
same traffic is served at an *effective* 61.9 GB/s while finishing in 54 µs —
because it comes from cache, not DRAM. The arithmetic as written would also
"prove" the CPU cannot make budget.
**Adjudication: the conclusion survives, the reasoning does not.** Rule out the
iGPU on **launch latency alone** (33–156 µs against a 100 µs budget, plus no
PyTorch XPU support for Tiger Lake Xe-LP) — that argument is sound and
independently sufficient. Xe-LP's 1.5–3 MB L3 does make the memory term
genuinely worse there than on the CPU, but that needs stating as a cache-tier
argument, not a DRAM one.

**C8 — `parallel-des-lookahead` recommends one barrier per 1.8 ms window
instead of per step (18x fewer barriers).**
**Adjudication: dead as a barrier optimisation, live as a scan enabler.** The
measured barrier cost is ~1 µs against a 54 µs step = under 2% of runtime, and
collecting it requires cross-partition spike outboxes plus a one-time connectome
edge-tagging pass. Not worth it for 2%. It *is* the necessary substrate for the
delay-window parallel scan (opportunity 11), which is a different and much
larger idea — keep the note for that reason only. (And note C-4 above: the
window is 20 steps, not 18.)

**C9 — `activity-structure-frontier` targets `brain_engine.py`'s dense/sparse
switches; `native_engine.py` has no such switch and is unconditionally dense.**
**Adjudication: the note is aimed at the deprecated path.** Its findings are
nevertheless *validated* by the audit: the one-way latch it identified is real,
and `audit/a1_fallback.py` shows the fallback additionally *drops spikes*, which
is worse than the note's performance concern. **Park the performance
recommendations; promote the correctness finding to §4 defect #3.** The
work-weighted delivery switch (Σ out-degree instead of spike count) becomes
relevant only if an active-set mode is ever added to the native engine.

---

## 6. Novel contributions worth publishing

Each claim below states its prior-art check *before* the novelty assertion. Two
of the five do not survive the check.

### N1 — Uniform axonal delay raised from a communication-batching interval to the *integration timestep*
**Prior art checked:** NEST (Morrison et al. 2005), NEURON (Hines & Carnevale
2008), NEST-at-exascale (Jordan et al. 2018), STACS — all use `d_min` to batch
*communication* across MPI ranks and keep the integration step small. Nicol 1993
(YAWNS) proves window = lookahead is asymptotically optimal for
*synchronisation*. Chandy–Misra 1979, Bryant 1977, Fujimoto 1990 treat lookahead
as a synchronisation quantity throughout. Front. Neuroinform. 2017 uses the
decoupling only to justify parallelisation. No simulator was found that sets
`h = d`.
**Novelty: STANDS, with two hard caveats.** (a) HANDOFF §2.10 measures the
technique at 1.00x for ≥15% activity and 26.4x at 0.88% — it is a *sparse-regime*
result and must be published as one. (b) **The delay is 2.0 ms, not 1.8 ms**
(§4 defect 1), so every number in the derivation is off by two steps and must be
recomputed before submission. Also: the literature search was query-based; a
formal search of the NEST/Arbor/Brian issue trackers and the NEST source has not
been done and should be, before a novelty claim goes in writing.

### N2 — Two-state (v,g) affine-map parallel scan with an exact reset, blocked by the delay
**Prior art checked:** Bullet Trains (ICML 2026, arXiv:2603.13283) does a
scalar-V affine-map scan with an exact reset via speculation + rollback, 44x —
but it is event-chunked and training-focused, not a fixed-Δt simulator, and does
not use delay as the block. FPT (ICML 2025, arXiv:2506.12087) handles reset by
fixed-point iteration, K≈3, also training. PSN/SPSN/DSN/SpikingSSMs all remove or
approximate the reset. Published scans are scalar-V.
**Novelty: STANDS on the combination** — (v,g) two-state extension, delay as the
block length (which makes cross-neuron non-interaction *provable*, so no rollback
is needed for anything except a neuron's own self-reset — strictly stronger than
Bullet Trains' speculation), inside a fixed-Δt simulator.
**Status: NOT IMPLEMENTED, and the nearest existing implementation is broken**
(§4 defect 2: the window repair solves the input-free trajectory and loses
spikes). This is the only genuinely unclaimed algorithmic item in the record and
it is also the only one that would justify a methods paper rather than an
engineering note.

### N3 — "Bit-identical to PyTorch" is a property of `torch.get_num_threads()`, not of the model
**Prior art checked:** arXiv:2408.05148 (2024) documents FMA-fused-vector-body
vs unfused-scalar-tail divergence as a general HPC reproducibility phenomenon.
Intel oneMKL's Conditional Numerical Reproducibility names the "conditional on
thread count" tier explicitly. So the *phenomenon* is known.
**Novelty: STANDS, but narrowed.** The contribution is not the mechanism — it is
(a) the demonstration that a published *simulator-comparison benchmark's*
numerical output silently depends on an environment variable, in a repository
whose entire purpose is to compare simulators, and (b) the constructive
reproduction of ATen's chunk seams — reproducing the reference's *own*
inconsistency (15 of 138,639 neurons obey a marginally different update rule)
in order to prove equivalence. That is a publishable reproducibility note, and
it is currently a §11.3 aside.

### N4 — The reference PyTorch backend uses forward Euler where Brian 2 uses exact integration
**Prior art checked:** Rotter & Diesmann 1999 is the exact propagator; Hansel et
al. 1998 is the diagnosis of grid-detection desynchronisation; Morrison et al.
2007 and Hanuschkin et al. 2010 are the accepted off-grid fix. NEST's
`iaf_psc_exp` implements exactly the same in-place lower-triangular propagator
independently. **The method is 25-year-old prior art. Nothing is novel about the
fix.**
**Novelty: the AUDIT FINDING stands, the method does not.** Euler shortens every
τ by exactly dt/2 (τ_syn −1.003%, τ_mem −0.250%), the P_vg coefficient is off by
+1.257e-2 relative, and the whole simulation runs 0.25–1% fast — in a repository
that exists to compare simulators, so part of what it measures is integration
schemes rather than frameworks. Combined with §4 defect 1 (the delay is 2.0 ms,
+11%), **this is the single highest-value thing to send upstream** and it should
go as a correction, not as a paper.

### N5 — Sparse (index, value) delay slots replacing the dense (L, N) ring buffer
**Prior art checked:** ADSEQ 2025 (arXiv:2512.05906) benchmarks queue structures
and finds CPU favours tree/FIFO; with a uniform delay a fixed-slot ring is the
standard answer and NEST's spike registers are already sparse.
**Novelty: DOES NOT STAND.** This is standard event-driven practice re-derived
from first principles. It is a large engineering win here (10.5 MB → ~30 KB,
and it is what makes the state L2-resident) and it deserves a prominent place in
the engineering write-up, but it is not a contribution.

### N6 — Real time for a whole insect connectome, bit-identical fp32, on a 4-core consumer laptop with no accelerator
**Prior art checked:** Knight & Nowotny 2021 got "very close to real time" for
77k neurons / 3×10⁸ connections on an RTX 2080 Ti, *using procedural
connectivity* (inapplicable to an EM-reconstructed edge list). Wang et al. 2025
got 18.6x real time on 12 Loihi 2 chips, *with 9-bit weight quantization* that
visibly shifts spike rates. arXiv:2505.21185 finds the RTX 4090 the fastest
consumer GPU for this class. This repo's own benchmark file has GeNN on an RTX
4070 at 0.5415 s/sim-second. **No published result runs a whole connectome at
real time on a consumer CPU at full fp32 fidelity.**
**Novelty: STANDS as an engineering result**, with the caveats stated plainly:
it holds on an idle machine (1.84x), fails under load (0.77x, measured), is
measured at sugar-protocol activity, has never been run sustained for 60 s, and
the saturating-activity regime only reaches 1.23x over the PyTorch baseline. The
strongest supportable sentence is: **"a 4-core laptop CPU matches a
discrete-GPU GeNN build to within 0.3% on the same model and protocol, at full
fp32 bit-identical fidelity."**

---

## 7. Index of the research directory

| File | One line |
|---|---|
| `_SYNTHESIS.md` | This file — arithmetic of the target, ranked plan, contradictions, novelty audit. |
| `hpc-simd-threading.md` | 4-way static partition + spin-wait pool + physical-core pinning; projects 3.98x (measured: 1.8–2.0x — see C1); its 64-alignment "correctness blocker" conflicts with bit-identity and must not be implemented (C2). |
| `compilers-codegen.md` | ISPC/Halide/TVM/torch.compile all land at 97–100% of hand intrinsics — no codegen headroom left; the only gap the field identifies is multicore dispatch. Its VTune/cycles-per-iteration diagnosis (action 5) is the most useful part and is still unrun. |
| `parallel-des-lookahead.md` | The uniform delay is a Chandy–Misra–Bryant lookahead equal to the whole window, licensing one barrier per window instead of per step; sound theory, ~2% payoff (C8), valuable only as substrate for the parallel scan. |
| `numerical-exactness-reproducibility.md` | Rotter–Diesmann/Morrison/Hanuschkin exact integration is already fully absorbed; the live issue is *conditional* reproducibility (Intel CNR terminology), and Gappa/FPTaylor is the right-sized tool to make `KAPPA_WINDOW_MAX` machine-checked. |
| `memory-layout-quantization.md` | Delete `refrac_steps` (binary, unread in the sweep) and saturating-uint8 `refrac`; int16 connectome weights verified lossless against the real data (all 15,091,983 values integers in [−2405, 1897]); v/g fixed-point is a bounded-error change needing an unmeasured range first. |
| `activity-structure-frontier.md` | Beamer/Ligra direction-optimizing switch applied to `brain_engine.py`: the dense fallback is a one-way latch and the delivery switch counts spikes where it should count Σ out-degree — aimed at the deprecated path (C9), but its latch finding is corroborated and worsened by the audit. |
| `snn-simulator-prior-art.md` | NEST/Auryn/CoreNEURON/GeNN/Brian2CUDA/STACS/Loihi 2 all optimise synaptic delivery, which is 0.14% here; contains one fabricated benchmark (Auryn 0.6x real-time, C4) that must be withdrawn; the Wang et al. 2025 table is verified and is the correct external comparator. |
| `igpu-and-consumer-hardware.md` | Iris Xe is a dead end (Tiger Lake Xe-LP outside PyTorch XPU support; 33–156 µs launch latency vs a 100 µs budget); survey of devices that do clear the bar — Strix Halo 4–7x projected, any CUDA GPU via GeNN 1.85x measured, NPUs architecturally excluded. |
| `CORRECTION_benchmark_units.md` | Fixes a 1000x unit error in HANDOFF §8's prior-art table (paper column is ms, not s); **its own performance table is now stale by 6x** (C3) and should be pruned before anyone quotes the "62x behind Loihi 2" figure. |
| `audit/a1_fallback.py` | Re-run today: active→dense fallback drops spikes — 51,251 vs 51,144 at whole-brain scale, first divergence t = 2.7 ms. **Unfixed.** |
| `audit/a2_window_exact.py` | The window simulator's repair solves the input-free trajectory, so mid-window arrivals are invisible to the spike-time solver. |
| `audit/a3_lost_spike.py` | Re-run today: concrete case where the reference emits a spike and the window sim does not. **The window path is not exact.** |
| `audit/a4_kappa.py` | Re-run today: `KAPPA_WINDOW_MAX` is exact at D = 1.8 ms (interior peak at 9.24 ms lies outside the window); unsound only for D > 9.24 ms, which never occurs. **Clean.** |
| `audit/a5_delay.py` | Re-run today: `L = 19` slots and end-to-end latency is **20 steps = 2.0 ms**, not the specified 1.8 ms. **+11% delay error, unfixed, in both engines.** |
| `audit/a6_delay_real.py` | Same measurement against the real engine. |
| `audit/a7_bracket.py`, `a8_bracket2.py` | Floating-point behaviour of the quartic spike-time solver near tangency (peak == θ) and at the u0 == θ boundary. |
| `audit/a9_verify_gaps.py` | Re-run today: shows which `verify.py` regimes take which fallback path — the sugar regime **never** exercises the fallback, and `refrac` differs on 138,288 neurons between active and dense paths. **Rule 1's coverage is weaker than it reads.** |
| `audit/a10_weights.py` | Direct dtype/range measurement on `data/fanout_csc.pt` — the evidence base for the lossless int16 claim. |
| `audit/a11_cross_engine.py` | Re-run today: `brain_engine.py` is bit-identical to upstream `run_pytorch.py` in both dense and active modes at small scale (max |v diff| = 0, max |g diff| = 0). **Clean — the foundation is sound.** |
| `audit/a12_cross_big.py` | The same cross-engine check with a 2,000-neuron ignition cascade over 400 steps. |
| `papers/wang2025_loihi2_drosophila.{pdf,txt,md}` | Sandia's Loihi 2 run of this exact model — the only apples-to-apples published comparator; Table 1 is the source of the corrected benchmark numbers. |

---

## 8. The three sentences that matter

1. **Real time is done; margin is not.** 0.0543 ms/step idle (1.84x), 0.129 under
   load (0.77x, fails). Spend the next week on the 3x-margin target in §3, not on
   new algorithms.
2. **Four notes promised 3.4–4.0x from threading and delivered 1.8–2.0x, because
   ~30% of the kernel is serial bookkeeping nobody counted.** Fix `put16`, drop
   `refrac` from the sweep, and measure the residue before budgeting anything
   else.
3. **The simulator implements a 2.0 ms axonal delay where the model specifies
   1.8 ms, its window path loses spikes, and its active-set fallback drops
   them.** All three are demonstrated by scripts already in this repository and
   mentioned in none of its documentation. Under a "no information sacrificed"
   constraint, these outrank every performance item on this page.
