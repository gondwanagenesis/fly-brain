# What this fork changes, and why

Fork of [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain),
branch `perf/event-driven-pytorch`.

Two independent contributions:

1. **Correctness.** Three places where the backends do not simulate the same
   model. One of them is large enough to dominate the cross-backend comparison
   this repository exists to make.
2. **Performance.** The whole 138,639-neuron brain now runs **faster than real
   time on a 4-core laptop**, bit-identical to the PyTorch reference, with no
   GPU.

Everything below is measured on an Intel i7-1185G7 (Tiger Lake, 4 cores,
AVX-512, 32 GB LPDDR4x) running Windows 11, torch 2.13 CPU. Reproduction
commands are given for each claim. The complete working record, including dead
ends and corrections, is in [HANDOFF.md](HANDOFF.md).

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
`compare_ground_truth.py` reports. For scale, exact-vs-Euler integration —
§1.2 below — measures Jaccard 0.913 on the same protocol. **This model
divergence is larger than the integration-scheme divergence**, and mean
first-spike latency shifts by 2.5–2.8 ms, comparable to the entire response
window the sugar experiment measures.

**Not claimed:** which semantics is *correct*. Shiu et al. ran Brian 2, so
Brian 2's behaviour is the published model and PyTorch is the deviation — but
whether the published model *intended* accumulate-through-refractoriness is a
question for the authors. What is claimed is that the two backends are not
simulating the same system, so part of what the benchmark attributes to
framework performance is model divergence.

Cross-backend spike-for-spike comparison is impossible (different PRNG
streams), which is why this is a controlled single-variable ablation inside one
backend rather than a direct Brian 2 diff.

```bash
.venv/Scripts/python.exe code/measure_refractory_divergence.py
.venv/Scripts/python.exe code/compare_semantics.py 700
```

### 1.2 PyTorch integrates with forward Euler; Brian 2 uses exact integration

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
peak amplitude is +0.47% and the whole simulation runs 0.25–1% fast.

**The exact form costs the same FLOPs** (3 mul + 2 add vs 2 mul + 3 add), so
there is no performance argument for keeping Euler. `INTEGRATION='exact'` is
wired through the model classes and is opt-in.

Measured directly in Brian 2 on the full brain (`code/run_brian2_reference.py`,
100 ms sugar): exact 1517 spikes / 323 active, Euler 1509 / 337, active-neuron
Jaccard **0.913**, per-neuron count correlation 0.993.

### 1.3 The PyTorch reference's bit pattern depends on its thread count

ATen's `v.add_(t, alpha=a)` is a **single-rounding FMA in its vectorised body**
but a **separate multiply-then-add in its scalar tail**. So the reference
integrates the last `(chunk_len mod W)` neurons of every `at::parallel_for`
chunk with a different rounding from the rest, where `W` is ATen's float vector
width — 16 under AVX-512, 8 under AVX2.

With N = 138,639 and 4 threads the chunks are 34,660 wide, so the seams fall on
neurons 34,656–34,659, 69,316–69,319, 103,976–103,979 and 138,624–138,638.

Consequence: **"bit-identical to PyTorch" is a property of
`torch.get_num_threads()`, not of the model.** For a repository whose purpose
is comparing simulators, that is a reproducibility caveat worth documenting.

Found by bisection: an FMA everywhere diverges at step 394 on neuron 138,637
(the final tail); forcing the separate form everywhere diverges at step 33.

The kernel in this fork does **not** have this property — its output is
independent of its own thread count (verified: ×1 and ×4 bit-identical to each
other and to the reference).

---

## Part 2 — Performance

### 2.1 Result

Real time is **0.1 ms/step** — `dt = 0.1 ms` means 10,000 steps per simulated
second. Minimum of 15 blocks × 500 steps, sugar protocol
(`flyloop/bench_native.py`):

| engine | ms/step | s/sim-second | vs real time |
|---|---|---|---|
| PyTorch dense | 1.6592 | 16.59 | 0.06× |
| PyTorch active-set | 1.6154 | 16.15 | 0.06× |
| native ×1 | 0.0822 | 0.82 | 1.22× |
| native ×2 | 0.0790 | 0.79 | 1.27× |
| **native ×4** | **0.0543** | **0.54** | **1.84×** |

**30.6× over the PyTorch dense baseline, bit-identical.**

⚠️ Timing caveat: that is the *minimum* over repeated blocks on an otherwise
idle machine, which for a sub-millisecond kernel is the honest estimate —
interference can only ever make a block slower. Under heavy background load the
same kernel measures ~0.13 ms/step. Report the methodology with the number.

**In context of this repo's own benchmark data** (`data/benchmark-results.csv`,
s per simulated second, best of 24, single trial):

| backend | n=1 | n=8 batched |
|---|---|---|
| GeNN (GPU) | 0.450 | 0.128 |
| Brian2GeNN (GPU) | 0.810 | 0.816 |
| NEST GPU | 0.883 | 0.936 |
| Brian2 (CPU) | 2.318 | 0.907 |
| Brian2CUDA (GPU) | 2.678 | 2.723 |
| PyTorch (CUDA) | 6.509 | 5.628 |
| PyTorch (CPU) | 686.3 | — |
| **native ×4 (laptop CPU)** | **0.543** | — |

A 4-core laptop CPU beats four of the five GPU backends and sits within 20% of
GeNN on a discrete card. Note these are **different machines** — the fork's
numbers are from the laptop above, the rest from the repo's benchmark runs.

Dedicated neuromorphic hardware remains far ahead: Loihi 2 achieves 0.0538
s/sim-second at the same `dt` ([Wang et al. 2025](https://arxiv.org/abs/2508.16792)),
roughly 10× this. Real time on a consumer device is the claim; beating Loihi 2
is not.

### 2.2 Theory of the changes

The prior analysis held that the dense path was memory-bandwidth-bound, so only
activity sparsity could help, and general headroom was ~1.5–2×. **That was
wrong.** Profiling the step at 171 µs found:

```
C kernel         98.2 µs  (57%)
torch.bernoulli  45.1 µs  (26%)   <- for TWENTY-ONE random numbers
python glue      27.6 µs  (16%)   <- ndarray.ctypes.data_as, ~14x per step
```

Two of the three were **framework overhead, not work**. The workload was
framework-bound, and the activity-independent headroom was ~30×.

**(a) Fuse the step into one pass.** PyTorch runs ~12 separate full-array
elementwise ops (refrac add, compare, mul, add, sub, neg, add, fma, gt, two
masked_fills, copy). Each reads and writes N floats, so the 2.2 MB state is
streamed through cache twelve times instead of once. Fusing changes no
arithmetic — only the order memory is touched.

**(b) The threshold-and-reset is a natural fit for AVX-512 mask registers.**
`v > v_th` produces a `__mmask16` straight from `vcmpps`; both resets become
single `vblendmps`; and the same mask *is* the spike bitset, so the bitset costs
nothing to produce. No branches, no select-by-arithmetic.

**(c) Make the delay line sparse.** Upstream stores the 1.8 ms axonal delay as
a dense `(19, N)` fp32 ring buffer — 10.5 MB, which alone exceeds L2 and evicts
the neuron state every 19 steps. But its contents are the recurrent input, which
has ~190 non-zeros out of 138,639 (1.75 spikes/step × ~110 mean fan-out).
Storing each slot as an `(index, value)` list makes the delay line ~30 KB
resident and the whole working set cache-resident. **Exact**, because the
omitted entries are exactly zero and `fl(x + 0.0f) == x` for every value that
occurs here.

**(d) Spike vector as a bitset** — 17 KB instead of 554 KB.

**(e) Batch the RNG.** `torch.bernoulli` on 21 stimulated neurons cost 45 µs
per step of pure dispatch. Drawing 4096 steps at once amortises it away. This is
**exact, not an approximation**: a batched `(K, n)` bernoulli consumes the
generator's stream in the same order as `K` sequential `(n,)` draws. Verified at
`(n, K)` = (21, 50), (21, 4096), (2, 1000), (1, 777) — zero differences.

**(f) Thread the chunks.** They are disjoint neuron ranges, so no
synchronisation is needed inside a step. A step is ~50–200 µs, so an OS barrier
(5–20 µs) would cost real percentage points; workers spin on a generation
counter and back off to a yield. The only shared state is the spike bitset,
whose per-chunk boundary words are OR-ed atomically — two atomics per chunk, not
one per vector group.

**(g) Cache the ctypes pointers.** `ndarray.ctypes.data_as` costs ~1–2 µs and
the step needed ~14 of them.

### 2.3 Portability

The kernel is compiled **without `-march=native`** and selects its path at
runtime by direct CPUID — including the XCR0 check, since a CPU can report
AVX-512 while the OS has not enabled ZMM state saving. One binary runs
anywhere; non-x86 uses the scalar path.

All three paths are verified **bit-identical to PyTorch and to each other**
(1500 steps, single-threaded):

| path | ms/step |
|---|---|
| AVX-512 | 0.133 |
| AVX2 | 0.263 |
| scalar | 1.215 |

Even with no SIMD at all the fused kernel beats PyTorch; on AVX2 — essentially
every x86 machine since 2013 — it is ~6× single-threaded.

Subtlety: because ATen's tail seam (§1.3) depends on the *host's* PyTorch build,
`tail_w` is read from `torch.backends.cpu.get_cpu_capability()` and the kernel
reproduces the seam wherever it actually falls, independently of which SIMD path
it takes internally.

### 2.4 What transfers to GPU and neuromorphic

The wins are overwhelmingly **removal of per-step framework overhead**, so the
gain on any backend is proportional to how much overhead it already carries.
Projections, not measurements — no CUDA device was available:

- **PyTorch CUDA (6.509 s/sim-s)** — largest opportunity. ~30 kernel launches
  per step at ~5–10 µs each is 150–300 µs before any work happens. (a), (c),
  (e) and CUDA Graphs all attack that directly.
- **GeNN (0.450)** — smallest. It already does code generation and its own ring
  buffers. Its own data shows n=1 is launch-bound (3.5× from batching to n=8,
  while Brian2GeNN / NEST GPU / Brian2CUDA are flat), so CUDA Graphs is the
  lever, not this kernel.
- **Neuromorphic (Loihi 2)** — essentially nothing transfers; the hardware is
  natively event-driven with no dense sweep, no DRAM ring buffer, no dispatch.
  What *would* transfer is fidelity: they run `dt = 1 ms` and round both the
  1.8 ms delay and 2.2 ms refractory to 2 ms (+11% / −9%), where method-of-steps
  at `dt = 1.8 ms` keeps the delay exact by construction.

---

## Part 3 — Verification

**Rule 1 of this fork: every performance change is proven bit-identical.**

`flyloop/verify_native.py` is stricter than comparing spike trains. It compares
the **entire floating-point state (v, g, refrac) as uint32 at every single
step**, so a divergence is caught on the step it first appears rather than
whenever it happens to flip a threshold crossing. A kernel can emit identical
spikes for thousands of steps while its state drifts by ULPs and then diverge
under a different stimulus; comparing raw bit patterns removes the possibility.

Eight regimes, 1 neuron → whole-brain, including saturating drive:

```bash
.venv/Scripts/python.exe flyloop/verify_native.py 800    # ALL BIT-IDENTICAL
.venv/Scripts/python.exe flyloop/bench_native.py         # min/median timing
```

The gate pins `torch.set_num_threads(4)`, because §1.3 means the reference's low
bits are a function of its thread count.

Per-regime speedup ranges 1.23× (saturating, 946 spikes/step, where fan-out
dominates) to 10.3× (sparse). Note the native kernel **never regresses** — the
previously existing active-set path was 2.3× *slower* than baseline at high
activity.

---

## Files added

| file | purpose |
|---|---|
| `flyloop/native/lif_kernel.c` | fused kernel, runtime ISA dispatch, thread pool |
| `flyloop/native_engine.py` | `NativeBrainEngine`, same API as `BrainEngine` |
| `flyloop/verify_native.py` | **the gate** — full-state bit comparison every step |
| `flyloop/bench_native.py` | min/median benchmark |
| `code/run_brian2_reference.py` | full-brain Brian 2 ground truth, exact + euler |
| `code/measure_refractory_divergence.py` | quantifies §1.1 discard fraction |
| `code/compare_semantics.py` | single-variable ablation for §1.1 |
| `research/` | cross-field literature notes + `_SYNTHESIS.md` |

Build is automatic on import (clang, `-O3 -ffp-contract=off`).
**`-ffp-contract=off` is mandatory** — without it the compiler fuses the
conductance decay's mul+add into an FMA and bit-identity breaks. `-ffast-math`
breaks it thoroughly.

---

## Status

No pull request has been opened upstream. This repository is a benchmark;
changing one backend's numbers alters a published comparison, and the §1.1
finding changes what the comparison *means*. Both are conversations to have with
the maintainers rather than changes to slip in.
