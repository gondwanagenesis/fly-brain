# What this fork changes, and why

Fork of [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain),
branch `perf/event-driven-pytorch`.

Two independent contributions:

1. **Correctness.** Three places where the backends do not simulate the same
   model. One is large enough to dominate the cross-backend comparison this
   repository exists to make.
2. **Performance.** The whole 138,639-neuron brain runs **faster than real time
   on a 4-core laptop with no GPU**, bit-identical to the PyTorch reference, in
   every regime the published experiments use.

Everything below is measured on an Intel i7-1185G7 (Tiger Lake, 4 cores,
AVX-512, 32 GB LPDDR4x), Windows 11, torch 2.13 CPU, with reproduction commands
given per claim. The complete working record, including dead ends and
corrections, is in [HANDOFF.md](HANDOFF.md).

---

## Part 1 — Correctness

### 1.1 The backends implement different models (largest finding)

**Brian 2** — `code/run_brian2_cuda.py:52`, and the designated ground truth in
`code/compare_ground_truth.py`:

```python
dg/dt = -g / tau : volt (unless refractory)
Synapses(..., on_pre='g += w', delay=t_dly)
```

During a neuron's 2.2 ms refractory period `g` is **frozen** and arriving
synaptic input **still accumulates**. Brian 2 applies `(unless refractory)` to
differential equations only — never to synaptic `on_pre` statements.

**PyTorch** — `code/run_pytorch.py`, `flyloop/brain_engine.py:203`:

```python
gate  = (refrac >= refrac_steps)
g_new = g * decay + delayed * gate
```

During refractoriness `g` **keeps decaying** and arriving input is
**discarded** — not deferred, dropped. Over the 22-step window the retained
charge decays by `0.98²² ≈ 0.64` and every spike that lands is lost.

A neuron is effectively **deaf for 2.2 ms after each of its own spikes** in one
backend and integrates through the window in the other.

**How much input is discarded** (`code/measure_refractory_divergence.py`):

| regime | arriving weight discarded |
|---|---|
| sugar GRNs — *the headline experiment* | **6.40%** |
| broad (1000 neurons) | **15.23%** |
| saturating (40k @ 200 Hz) | 8.74% |

**Does it move the observable?** Yes. Single-variable ablation — same backend,
same seed, same RNG stream, same ordering, only that one term changed
(`code/compare_semantics.py`, 700 steps):

| regime | arm | spikes | Δ | active-neuron Jaccard | rate *r* |
|---|---|---|---|---|---|
| sugar (21) | input delivered | 987 → 1209 | **+22.5%** | **0.871** | 0.961 |
| sugar (21) | true Brian 2 | 987 → 1279 | **+29.6%** | **0.863** | 0.943 |
| broad (1000) | input delivered | 30606 → 91626 | **+199%** | 0.666 | 0.782 |
| broad (1000) | true Brian 2 | 30606 → 96213 | **+214%** | 0.659 | 0.777 |

Jaccard overlap of the active-neuron set is *precisely* the headline metric
`compare_ground_truth.py` reports. For scale, exact-vs-Euler integration (§1.2)
measures Jaccard 0.913 on the same protocol. **This model divergence is larger
than the integration-scheme divergence**, and mean first-spike latency shifts by
2.5–2.8 ms — comparable to the entire response window the sugar experiment
measures.

**Not claimed:** which semantics is *correct*. Shiu et al. ran Brian 2, so
Brian 2's behaviour is the published model and PyTorch is the deviation — but
whether the published model *intended* accumulate-through-refractoriness is a
question for the authors. What is claimed is that the two backends are not
simulating the same system, so part of what the benchmark attributes to
framework performance is model divergence.

Cross-backend spike-for-spike comparison is impossible (different PRNG streams),
which is why this is a controlled single-variable ablation inside one backend.

```bash
.venv/Scripts/python.exe code/measure_refractory_divergence.py
.venv/Scripts/python.exe code/compare_semantics.py 700
```

### 1.2 PyTorch uses forward Euler; Brian 2 uses exact integration

Brian 2 selects exact integration automatically for linear equations
([docs](https://brian2.readthedocs.io/en/stable/user/numerical_integration.html)),
so forward Euler is not the published baseline — it is the exact solution of a
slightly different model.

| coefficient | exact | Euler | rel. error |
|---|---|---|---|
| `alpha_m` (v→v) | 0.9950124792 | 0.9950000000 | −1.25e−05 |
| `alpha_s` (g→g) | 0.9801986733 | 0.9800000000 | −2.03e−04 |
| **`P_vg` (g→v)** | **0.0049379353** | **0.0050000000** | **+1.26e−02** |

Euler shortens **every** time constant by exactly `dt/2` (τ_syn −1.00%,
τ_mem −0.25%). Charge is preserved to ~1e−5 so firing *rates* survive, but PSP
peak amplitude is +0.47% and the simulation runs 0.25–1% fast. **The exact form
costs the same FLOPs** (3 mul + 2 add vs 2 mul + 3 add), so there is no
performance argument for Euler.

Measured directly in Brian 2 on the full brain
(`code/run_brian2_reference.py`, 100 ms sugar): exact 1517 spikes / 323 active,
Euler 1509 / 337, active-neuron Jaccard **0.913**, count correlation 0.993.

### 1.3 The reference's bit pattern depends on its thread count

ATen's `v.add_(t, alpha=a)` is a **single-rounding FMA in its vectorised body**
but a **separate multiply-then-add in its scalar tail**. So the reference
integrates the last `(chunk_len mod W)` neurons of every `at::parallel_for`
chunk with a different rounding from the rest, where `W` is ATen's float vector
width — 16 under AVX-512, 8 under AVX2.

With N = 138,639 and 4 threads the seams fall on neurons 34,656–34,659,
69,316–69,319, 103,976–103,979 and 138,636–138,638 — 15 neurons.

Consequence: **"bit-identical to PyTorch" is a property of
`torch.get_num_threads()`, not of the model.** For a repository whose purpose is
comparing simulators, that is worth documenting.

Found by bisection: an FMA everywhere diverges at step 394 on neuron 138,637;
forcing the separate form everywhere diverges at step 33. The kernel here does
**not** have this defect — its output is independent of its own thread count
(verified: ×1 and ×4 bit-identical to each other and to the reference).

---

## Part 2 — Performance

### 2.1 Result

Real time is **0.1 ms/step** — `dt = 0.1 ms` means 10,000 steps per simulated
second. Full sweep, Syncthing stopped, min of 7 blocks, correctness verified
over 400 lockstep steps per regime (`flyloop/verify_all.py`):

| regime | correct | torch ms | native ms | +reorder | vs torch | reorder gain | live tiles | **real time** |
|---|---|---|---|---|---|---|---|---|
| silent (0 drive) | BIT-EQ | 2.2970 | 0.0269 | **0.0259** | 88.6× | 1.04× | 0.1% | **3.86×** |
| single neuron | BIT-EQ | 3.5611 | 0.0464 | 0.0510 | 69.8× | 0.91× | 3.2% | **1.96×** |
| sugar GRNs (21) | BIT-EQ | 3.1020 | 0.0917 | **0.0562** | 55.2× | 1.63× | 28.5% | **1.78×** |
| P9 walking (2) | BIT-EQ | 3.8922 | 0.1237 | **0.0586** | 66.4× | 2.11× | 12.7% | **1.71×** |
| broad (100) | BIT-EQ | 2.7857 | 0.1096 | 0.0858 | 32.5× | 1.28× | 78.7% | 1.17× |
| broad (1000) | BIT-EQ | 3.5575 | 0.4957 | 0.3518 | 10.1× | 1.41× | 97.7% | 0.28× |
| broad (10000) | BIT-EQ | 3.9043 | 0.8006 | 0.6575 | 5.9× | 1.22× | 100.0% | 0.15× |
| saturating (40k) | BIT-EQ | 8.0741 | 2.5337 | 2.3618 | 3.4× | 1.07× | 100.0% | 0.04× |

**ALL BIT-IDENTICAL.** The kernel beats PyTorch in every regime, 3.4×–88.6×,
with no regression anywhere.

⚠️ **Read the two columns differently.** The native ms/step figures are a
*lower bound* on performance — background load can only make a block slower, and
the minimum over repeats is the honest estimate. The **vs torch** ratios carry
more uncertainty, because the PyTorch baseline was measured under the same
residual load and reads 2.3–8.1 ms/step here against 1.66 ms on a fully idle
machine. Treat the real-time column as solid and the ratios as generous.

### 2.2 Is the speedup general? Partly — and the distinction matters

**The core kernel work is unconditional.** Fused sweep, sparse delay line,
bitset spikes, batched RNG, threading, refractory-counter removal — none depend
on what is being simulated. They deliver 3.4×–88.6× in every regime and never
regress.

**Tile-skipping is activity-dependent by construction.** A 16-neuron tile is
skippable only if all 16 are at exact rest, so as the brain wakes up the win
evaporates: 71% of the sweep skipped on sugar, 87% on P9, but 0% once ~100% of
tiles hold a live neuron. That is not a defect — it is the same law every
neuromorphic result obeys, and it is why the `live tiles` column is reported
next to the speedup.

**Real time therefore holds for the regimes the published experiments actually
use.** Sugar and P9 are the two experiments in `code/benchmark.py`, and both
clear real time. Broad stimulation of 1000+ neurons is an artificial stress case
where no ordering has headroom left.

**An unexpected second effect.** Reordering was expected to be worthless once
tiles stop being skippable. It is not: at broad(1000), 97.7% of tiles are live —
essentially nothing is skipped — yet reordering still gives **1.41×**, and
1.07–1.22× at 100% live. That gain cannot be tile-skipping. The likely mechanism
is **fan-out locality**: grouping by `cell_type` also clusters postsynaptic
targets, so the scatter-add touches fewer cache lines. Flagged as needing direct
confirmation rather than asserted.

### 2.3 Theory of the changes

The prior analysis held that the dense path was memory-bandwidth-bound, so only
sparsity could help, and general headroom was ~1.5–2×. **That was wrong.**
Profiling the step at 171 µs found:

```
C kernel         98.2 µs  (57%)
torch.bernoulli  45.1 µs  (26%)   <- for TWENTY-ONE random numbers
python glue      27.6 µs  (16%)   <- ndarray.ctypes.data_as, ~14x per step
```

Two of three were **framework overhead, not work**.

**(a) Fuse the step into one pass.** PyTorch runs ~12 separate full-array
elementwise ops, so the 2.2 MB state is streamed through cache twelve times
instead of once. Fusing changes no arithmetic — only the order memory is
touched.

**(b) The threshold-and-reset fits AVX-512 mask registers.** `v > v_th` yields a
`__mmask16` from `vcmpps`; both resets become single `vblendmps`; and the same
mask *is* the spike bitset, so the bitset costs nothing.

**(c) Sparse delay line.** Upstream's `(19, N)` fp32 ring buffer is 10.5 MB and
evicts the neuron state every 19 steps, but its contents have ~190 non-zeros out
of 138,639. Storing `(index, value)` lists makes it ~30 KB resident. Exact — the
omitted entries are exactly zero.

**(d) Spike vector as a bitset** — 17 KB instead of 554 KB.

**(e) Batch the RNG.** `torch.bernoulli` on 21 neurons cost 45 µs/step of pure
dispatch. Drawing 4096 steps at once is **exact, not approximate**: a batched
`(K, n)` bernoulli consumes the generator's stream in the same order as `K`
sequential `(n,)` draws. Verified at `(n, K)` = (21, 50), (21, 4096), (2, 1000),
(1, 777) — zero differences.

**(f) Thread the chunks.** Disjoint neuron ranges, so no synchronisation inside
a step. A step is ~30–200 µs, so an OS barrier (5–20 µs) would cost real
percentage points; workers spin on a generation counter. The only shared state
is the spike bitset, whose per-chunk boundary words are OR-ed atomically.

**(g) Cache the ctypes pointers** — `data_as` costs ~1–2 µs and the step needed
~14.

**(h) Take the refractory counter out of the sweep.** It existed only so the
delayed pass could evaluate `gate = refrac >= refrac_steps`, but only ~40
neurons are refractory at once. Replaced by a gate bitset (17 KB) plus a compact
countdown list. The previous-spike bitset was *also* only read for the refrac
reset, so it leaves the sweep too: **the sweep now reads just `v` and `g`.**
24.25 → 16.25 bytes per neuron.

**(i) Skip inert 16-neuron tiles, after reordering neurons by `cell_type`.**
See §2.4.

### 2.4 Why reordering was necessary

91% of neurons sit at exact rest and are provably unchanged by the update — for
`v == v_rest, g == 0` the kernel computes `t = 0`, `v = fma(0, c_mem, v_rest) =
v_rest`, `g = 0`, and `v_rest > v_th` is false. So skipping is exact.

But in the shipped neuron order the live ones are **scattered — mean run length
1.1 neurons** — so 98.85% of tiles held at least one live neuron and skipping
saved 1.15%. Neuron index is an arbitrary artefact of the completeness CSV's row
order, so renumbering is free and exact. Grouping by `cell_type` (cells of a
type share inputs, so they fall quiet together) collapses it:

| ordering | live tiles (sugar, tile-16) | sweep skippable |
|---|---|---|
| shipped CSV order | 70.6% | 1.15% (tile-64) |
| soma position (Morton) | 42.0% | — |
| **`cell_type`** | **28.3%** | **71.7%** |
| oracle ceiling | 7.7% | 92.3% |

Full generalisation study across regimes, keys and tile sizes:
[`research/reorder_measurements.md`](research/reorder_measurements.md).

The reordering is **invisible to callers** — `inject`, `silence`,
`indices_of` and `spikes_dataframe` all address neurons by FlyWire id.

### 2.5 Portability

Compiled **without `-march=native`**, selecting AVX-512 / AVX2 / scalar at
runtime by direct CPUID — including the XCR0 check, since a CPU can report
AVX-512 while the OS has not enabled ZMM state saving. One binary runs anywhere;
non-x86 uses the scalar path. All three paths verified **bit-identical to
PyTorch and to each other**:

| path | ms/step (1 thread) |
|---|---|
| AVX-512 | 0.133 |
| AVX2 | 0.263 |
| scalar | 1.215 |

Even with no SIMD the fused kernel beats PyTorch; on AVX2 — essentially every
x86 machine since 2013 — it is ~6× single-threaded.

Because ATen's tail seam (§1.3) depends on the *host's* PyTorch build, `tail_w`
is read from `torch.backends.cpu.get_cpu_capability()` and the kernel reproduces
the seam wherever it falls. Under reordering the seam is computed **per neuron
in original index space** and carried through the permutation, so it follows the
neuron rather than the array slot.

### 2.6 What transfers to GPU and neuromorphic

The wins are largely **removal of per-step framework overhead**, so the gain on
any backend is proportional to the overhead it already carries. Projections, not
measurements — no CUDA device was available.

Context, from this repo's own `data/benchmark-results.csv` (s per simulated
second, best of 24, single trial, RTX 4070):

| backend | n=1 | n=8 batched |
|---|---|---|
| GeNN (GPU) | 0.450 | 0.128 |
| Brian2GeNN (GPU) | 0.810 | 0.816 |
| NEST GPU | 0.883 | 0.936 |
| Brian2 (CPU) | 2.318 | 0.907 |
| Brian2CUDA (GPU) | 2.678 | 2.723 |
| PyTorch (CUDA) | 6.509 | 5.628 |
| PyTorch (CPU) | 686.3 | — |
| **this fork (laptop CPU, sugar)** | **0.562** | — |

- **PyTorch CUDA (6.509)** — largest opportunity. ~30 kernel launches per step
  at ~5–10 µs each is 150–300 µs before any work happens. (a), (c), (e) and
  CUDA Graphs attack that directly.
- **GeNN (0.450)** — smallest. It already generates code and manages its own
  ring buffers. Its own data shows n=1 is launch-bound (3.5× from batching,
  while Brian2GeNN / NEST GPU / Brian2CUDA are flat), so CUDA Graphs is the
  lever, not this kernel.
- **Neuromorphic (Loihi 2)** — essentially nothing transfers; the hardware is
  natively event-driven with no dense sweep, no DRAM ring buffer, no dispatch.
  What *would* transfer is fidelity: they run `dt = 1 ms` and round both the
  1.8 ms delay and 2.2 ms refractory to 2 ms (+11% / −9%), where method-of-steps
  at `dt = 1.8 ms` keeps the delay exact by construction.

Loihi 2 achieves 0.0538 s/sim-second at the same `dt`
([Wang et al. 2025](https://arxiv.org/abs/2508.16792)) — roughly 10× this.
**Real time on a consumer device is the claim; beating Loihi 2 is not.**

---

## Part 3 — Verification

**Rule 1 of this fork: every performance change is proven bit-identical.**

`flyloop/verify_all.py` is the gate. It compares state as **raw uint32 at every
single step**, so a divergence is caught where it first appears rather than
whenever it happens to flip a threshold crossing. Since the engines now number
neurons differently, `v` and `g` are compared **under the permutation** and
spike trains matched **by FlyWire id** — the same strength of check, no longer
assuming both engines index identically.

`refrac` is no longer compared, because the native engine has none. That is not
weaker: the refractory state's only influence on the dynamics is the factor it
applies to arriving input, which lands in `g`, so an off-by-one would diverge
`g` on the very next arrival. Bit-identical `g` over thousands of steps and
thousands of spikes proves the timing exactly.

Correctness and timing run in **separate passes** — `verify_native.py`
interleaves both engines and wraps every step in two `perf_counter` calls, which
is how a bogus 0.64× regression got recorded and then withdrawn.

```bash
.venv/Scripts/python.exe flyloop/verify_all.py 400   # correctness + timing, 8 regimes
.venv/Scripts/python.exe flyloop/bench_native.py     # min/median timing detail
.venv/Scripts/python.exe flyloop/verify_native.py    # legacy strict-array gate
```

**Benchmarking note:** Syncthing was continuously re-indexing this repository and
inflating every absolute timing (PyTorch dense read 4.97 ms/step against 1.66 ms
quiet). Stop it, or exclude the repo, before quoting numbers.

---

## Files added

| file | purpose |
|---|---|
| `flyloop/native/lif_kernel.c` | fused kernel, runtime ISA dispatch, tile skipping, thread pool |
| `flyloop/native_engine.py` | `NativeBrainEngine`, same API as `BrainEngine` |
| `flyloop/reorder.py` | `cell_type` / Morton / class permutations + liveness study |
| `flyloop/verify_all.py` | **the gate** — correctness + timing, all regimes |
| `flyloop/verify_native.py` | strict array-order gate (pre-reordering) |
| `flyloop/bench_native.py` | min/median benchmark |
| `code/run_brian2_reference.py` | full-brain Brian 2 ground truth, exact + euler |
| `code/measure_refractory_divergence.py` | quantifies §1.1 discard fraction |
| `code/compare_semantics.py` | single-variable ablation for §1.1 |
| `research/` | cross-field literature notes, `_SYNTHESIS.md`, reorder study |
| `ANALYSIS.md` | exactness theorem, cost bound, GPU roofline |

Build is automatic on import (clang, `-O3 -ffp-contract=off`).
**`-ffp-contract=off` is mandatory** — without it the compiler fuses the
conductance decay's mul+add into an FMA and bit-identity breaks. `-ffast-math`
breaks it thoroughly.

The DLL is **content-addressed** (named by a hash of the source). Windows Smart
App Control blocks unsigned binaries by *file identity*, so a rebuild to the same
path fails to load with `WinError 4551` while an identical library under a new
name loads fine. Hashing side-steps that and doubles as a build cache.

---

## Status

No pull request has been opened upstream. This repository is a benchmark;
changing one backend's numbers alters a published comparison, and the §1.1
finding changes what the comparison *means*. Both are conversations to have with
the maintainers rather than changes to slip in.
