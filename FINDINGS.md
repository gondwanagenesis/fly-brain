# Findings for the fly-brain team

Work on [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain),
fork `gondwanagenesis/fly-brain`, branch `perf/event-driven-pytorch`.

Everything here is reproducible with the commands given. Depth and dead ends are
in [HANDOFF.md](HANDOFF.md); the full write-up is [FORK.md](FORK.md).

---

## TL;DR

1. **Your PyTorch backend and your Brian 2 ground truth do not simulate the same
   model.** They differ in how synaptic input is handled during the refractory
   period, and separately the PyTorch axonal delay is one timestep longer. On the sugar experiment this discards **6.4% of arriving synaptic
   weight**, changes spike count by **+22.5%**, and drops active-neuron Jaccard
   to **0.871** — a larger divergence than the forward-Euler-vs-exact gap
   (0.913). This is the finding we would most like you to check.

2. **The whole 138,639-neuron model runs faster than real time on a 4-core
   laptop with no GPU**, bit-identical to the PyTorch reference: 0.046 ms/step
   on sugar (**2.2× real time**), 0.026 ms on P9 (**3.8×**). That is
   0.46 s/simulated-second, level with **GeNN on an RTX 4070** (0.450) at
   roughly an order of magnitude less power.

3. **This is a latency result, not a throughput one.** Your GPU backends still
   win batched work — GeNN at n=8 is 0.128 s/sim-s per trial. We measured that
   batching cannot close that on CPU (see §4).

---

## 1. The backends implement different models

`code/run_brian2_cuda.py:52` — and `compare_ground_truth.py` treats this as
ground truth:

```python
dg/dt = -g / tau : volt (unless refractory)
Synapses(..., on_pre='g += w', delay=t_dly)
```

During refractoriness `g` is **frozen** and arriving input **accumulates**.
Brian 2 applies `(unless refractory)` to differential equations only, never to
synaptic `on_pre` statements.

`code/run_pytorch.py` / `flyloop/brain_engine.py:203`:

```python
gate  = (refrac >= refrac_steps)
g_new = g * decay + delayed * gate
```

During refractoriness `g` **keeps decaying** and arriving input is
**discarded** — not deferred, dropped. Over the 22-step window the retained
charge decays by `0.98²² ≈ 0.64` and every spike that lands is lost. A neuron is
effectively deaf for 2.2 ms after each of its own spikes in one backend and
integrates through the window in the other.

**Discarded weight** (`code/measure_refractory_divergence.py`):
sugar **6.40%**, broad-1000 **15.23%**, saturating 8.74%.

**Effect on the observable** — single-variable ablation, same backend, same
seed, same RNG stream, only that term changed (`code/compare_semantics.py`):

| regime | arm | spikes | Δ | active-neuron Jaccard |
|---|---|---|---|---|
| sugar (21) | input delivered | 987 → 1209 | +22.5% | **0.871** |
| sugar (21) | true Brian 2 semantics | 987 → 1279 | +29.6% | **0.863** |
| broad (1000) | input delivered | 30606 → 91626 | +199% | 0.666 |

Jaccard of the active-neuron set is exactly what `compare_ground_truth.py`
reports. Mean first-spike latency shifts 2.5–2.8 ms, comparable to the whole
response window the sugar experiment measures.

**We are not claiming which is correct.** Shiu et al. ran Brian 2, so Brian 2 is
the published model and PyTorch is the deviation — but whether the published
model *intended* accumulate-through-refractoriness is your call. What we are
claiming is that the two are not the same system, so part of what the benchmark
attributes to framework performance is model divergence.

Cross-backend spike-for-spike comparison is impossible (different PRNG streams),
which is why this is an ablation inside a single backend.

```bash
python code/measure_refractory_divergence.py
python code/compare_semantics.py 700
```

## 1b. The axonal delay differs between backends by one timestep

Measured directly — force one neuron to spike, watch when its postsynaptic
targets' `g` first changes:

| backend | spike → effect | effective delay |
|---|---|---|
| Brian 2 (`delay=1.8*ms`) | 19 steps | 18 steps + 1 update lag ✓ |
| PyTorch / `brain_engine.py` | **20 steps** | **19 steps + 1 update lag** |

The PyTorch backend delivers every spike **one timestep (0.1 ms) later than
Brian 2** — a 5.6% longer axonal delay, on every synapse, compounding with each
hop. Root cause is the ring buffer: `L = int(tDelay/dt) + 1 = 19` slots combined
with read-then-write in the same step holds each value for 19 steps rather than
18.

Worth noting alongside §1: Wang et al. round the 1.8 ms delay to 2.0 ms on
Loihi 2 and flag it as a fidelity compromise. The PyTorch backend arrives at
1.9 ms by accident.

## 1c. Ground truth — and why the current methodology could not catch any of this

We ran the comparison (`code/compare_to_brian2.py`, sugar, 100 ms). Cross-backend
spike-for-spike matching is impossible (different PRNGs), so this uses the same
statistical metrics as your `compare_ground_truth.py`, plus a **noise floor**:
Brian 2 against *itself* at two seeds.

| comparison | spikes | active | Jaccard | rate *r* |
|---|---|---|---|---|
| **noise floor** — brian2 exact seed 0 vs seed 1 | 1517 / 1513 | 323 / 341 | **0.850** | 0.941 |
| brian2 exact vs brian2 euler | 1517 / 1509 | 323 / 337 | 0.913 | 0.993 |
| brian2 exact vs **native kernel** | 1517 / 1443 | 323 / 322 | **0.931** | 0.935 |

**The native kernel agrees with Brian 2 better than Brian 2 agrees with itself
across seeds.** By this test it passes cleanly.

**But that is the point.** The seed-to-seed noise floor is 0.850, and the
refractory divergence in §1 measures 0.871 by ablation — *barely above the
noise*. Forward-Euler-vs-exact (0.913) is likewise inside it. So this
methodology, at this duration and single seed, **does not have the statistical
power to detect either divergence**. That is almost certainly why they went
unnoticed: a stochastic Poisson protocol over 100 ms produces ~1500 spikes and
~330 active neurons, and trial-to-trial variability swamps the effect.

The ablation in §1 is the sensitive test precisely because it removes RNG
variability — same seed, same stream, one term changed. If you want
`compare_ground_truth.py` to be able to catch this class of bug, it needs either
many seeds compared as distributions, longer runs, or deterministic
(non-Poisson) drive for validation runs.

```bash
python code/compare_to_brian2.py 100
```

## 2. Two further reproducibility issues

**Forward Euler vs exact integration.** Brian 2 auto-selects exact integration
for linear equations; the PyTorch backend uses forward Euler, which shortens
every time constant by exactly `dt/2` (τ_syn −1.00%, τ_mem −0.25%) and inflates
`P_vg` by +1.26%. Rates survive (charge preserved to ~1e−5) but PSP peak is
+0.47% and the sim runs 0.25–1% fast. Measured in Brian 2 on the full brain:
exact 1517 spikes / 323 active vs Euler 1509 / 337, **Jaccard 0.913**. The exact
form costs the same FLOPs — there is no performance argument for Euler.
`code/run_brian2_reference.py` builds the full 138,639-neuron network either way.

**The reference's bit pattern depends on `torch.get_num_threads()`.** ATen's
`add_(t, alpha=)` is a single-rounding FMA in its vectorised body but a separate
multiply-then-add in its scalar tail, so the last `(chunk_len mod W)` neurons of
every `at::parallel_for` chunk integrate with a different rounding. 15 neurons
at 4 threads; the positions move with the thread count. Minor numerically,
but it means bitwise reproducibility claims need to pin the thread count.

## 3. Performance

Real time = 0.1 ms/step (`dt = 0.1 ms` ⇒ 10,000 steps/simulated second).
i7-1185G7, 4 cores, AVX-512, no GPU. Correctness and timing measured in separate
passes; min of 7 blocks; 500 lockstep verification steps per regime.

| regime | correct | torch ms | ours ms | vs torch | live tiles | **real time** |
|---|---|---|---|---|---|---|
| single neuron | BIT-EQ | 1.0664 | **0.0109** | 97.7× | 3.2% | **9.16×** |
| silent | BIT-EQ | 2.7179 | **0.0260** | 104.4× | 0.1% | **3.84×** |
| P9 walking (2) | BIT-EQ | 2.0810 | **0.0262** | 79.5× | 12.7% | **3.82×** |
| sugar GRNs (21) | BIT-EQ | 2.4091 | **0.0461** | 52.2× | 28.5% | **2.17×** |
| broad (100) | BIT-EQ | 2.3383 | 0.0763 | 30.6× | 78.7% | 1.31× |
| broad (1000) | BIT-EQ | 3.0359 | 0.3019 | 10.1× | 97.7% | 0.33× |
| broad (10000) | BIT-EQ | 4.0052 | 0.6806 | 5.9× | 100.0% | 0.15× |
| saturating (40k) | BIT-EQ | 7.0947 | 2.2536 | 3.1× | 100.0% | 0.04× |

**ALL BIT-IDENTICAL** — full state compared as raw uint32 at every step, spike
trains matched exactly, 8 regimes from one neuron to whole-brain drive.

**In context of your own `data/benchmark-results.csv`** (s/simulated-second,
best of 24, n=1, RTX 4070):

| backend | n=1 | n=8 |
|---|---|---|
| GeNN (GPU) | 0.450 | **0.128** |
| **this fork (laptop CPU, sugar)** | **0.461** | — |
| Brian2GeNN (GPU) | 0.810 | 0.816 |
| NEST GPU | 0.883 | 0.936 |
| Brian2 (CPU) | 2.318 | 0.907 |
| Brian2CUDA (GPU) | 2.678 | 2.723 |
| PyTorch (CUDA) | 6.509 | 5.628 |
| PyTorch (CPU) | 686.3 | — |

Different machines and harnesses, so read this as "same class", not a tie.

### What is general and what is not

**Unconditional** — fused single-pass kernel, sparse delay line, bitset spikes,
batched RNG, threading, refractory-counter removal. These beat PyTorch in every
regime (3.1×–104×) with no regression anywhere.

**Conditional** — inert-tile skipping is activity-dependent by construction: a
16-neuron tile is skippable only if all 16 are at exact rest. 71–99% of the
sweep skipped when activity is sparse, nothing once the brain is broadly driven.
Real time therefore holds for the regimes your two shipped experiments use, and
not for artificial broad stimulation.

### How

The prior assumption was that the dense path was memory-bandwidth-bound with
~1.5–2× of headroom. Profiling at 171 µs/step found otherwise:

```
C kernel         98.2 µs  (57%)
torch.bernoulli  45.1 µs  (26%)   <- for TWENTY-ONE random numbers
python glue      27.6 µs  (16%)
```

It was **framework-bound**, and the headroom was ~30×. The changes: fuse ~12
full-array passes into one; map the threshold-and-reset onto AVX-512 mask
registers so the spike bitset is free; replace the dense `(19, N)` delay ring
(10.5 MB, ~190 non-zeros) with sparse `(index, value)` slots; batch the Poisson
draws (exact — a batched `(K,n)` bernoulli consumes the RNG stream in the same
order as `K` sequential draws); thread the disjoint chunks with a spin barrier;
drop the refractory counter to a gate bitset; and skip inert tiles after
renumbering neurons by `cell_type`, which is exact because neuron index is an
arbitrary CSV artefact.

One binary, runtime-dispatched AVX-512 / AVX2 / scalar, all three verified
bit-identical to each other and to PyTorch. It runs on any x86-64 since ~2013;
even the no-SIMD path beats PyTorch.

## 4. What we tested and rejected

**Trial batching does not transfer from GPU to CPU.** GeNN gains 3.5× from
batching at n=8, so the obvious move is to batch on CPU too. We measured the
amortisable fraction of a step:

| regime | full step | python-side (amortisable) | real per-trial C work |
|---|---|---|---|
| sugar | 33.01 µs | 2.92 µs = **8.8%** | 91.2% |
| saturating | 1594.61 µs | 11.27 µs = **0.7%** | 99.3% |

At 8 trials that caps the gain at ~1.09× sparse and ~1.007× dense. GeNN's
batching win is a GPU artifact of underutilisation — a GPU at n=1 is
launch-latency-bound and mostly idle. Our CPU at n=1 is already 91–99% real
compute, so there is nothing to fill. **We did not build it.** Your GPU
throughput advantage at n=8 is real and a CPU cannot close it this way.

**Threaded fan-out is correct but 1.65× slower** than serial (atomic
contention). Off by default.

Also ruled out with reasons in HANDOFF.md §6: SparseProp rescale, multirate
RK/IMEX, Krylov, Magnus, Strang splitting, delta-synapse limit, RCM/METIS
reordering for bandwidth, fp16 neuron state, Parareal/MGRIT, and the
PSN/SPSN/SpikingSSM family (all approximate or remove the reset).

## 5. What we are not claiming

- **Not faster than Loihi 2.** It achieves 0.0538 s/sim-s at the same `dt`
  (Wang et al. 2025), roughly 10× ours. We corrected a 1000× unit error in our
  own notes that had briefly implied otherwise — the paper's column is
  `FlyWire (ms)`.
- **Not faster than GeNN.** Level with it at n=1; behind at n=8.
- **The Brian 2 comparison passes, but it is a weak test.** The native kernel
  scores Jaccard 0.931 against Brian 2, above the 0.850 seed-to-seed noise
  floor — but that floor is high enough to hide the §1 and §2 divergences
  entirely (0.871 and 0.913 respectively). Passing it is necessary, not
  sufficient. See §1c.
- **The fan-out-locality effect is a hypothesis.** Reordering gives 1.1–1.5×
  even where nothing is skippable; the likely mechanism is cache locality in the
  scatter-add, unconfirmed.
- **Timing caveats.** Ours are minima over repeats (load only makes blocks
  slower, so the real-time column is conservative). Watch for background
  indexers — Syncthing re-indexing the repo inflated PyTorch dense from 1.66 to
  4.97 ms/step before we noticed.

## 6. Suggested next steps

1. **Decide the refractory semantics** (§1) and **the one-step delay offset**
   (§1b). Whichever way each goes, the backends should agree.
2. **Give `compare_ground_truth.py` the power to catch this class of bug** (§1c)
   — multiple seeds compared as distributions, longer runs, or deterministic
   drive for validation. As it stands its noise floor is larger than the
   divergences it would need to detect.
3. **Consider `method='exact'`** for the PyTorch backend — same FLOP count,
   matches what Brian 2 actually does.
4. **PyTorch CUDA is your biggest headroom**: 6.509 vs GeNN's 0.450 on the same
   card. ~30 kernel launches per step at 5–10 µs each is 150–300 µs before any
   work happens. CUDA Graphs plus the sparse delay line should move it a long way.
5. **The CPU path makes a GPU optional** for single-trial and closed-loop work —
   relevant to `virtualfly/`, since a sensorimotor loop cannot be batched and
   latency is all that matters.

## 7. Reproducing everything

```bash
python flyloop/verify_all.py 500        # correctness + timing, 8 regimes
python flyloop/bench_native.py          # min/median timing detail
python code/compare_semantics.py 700    # the model divergence
python code/measure_refractory_divergence.py
python code/run_brian2_reference.py --duration-ms 100 --method exact
python code/compare_to_brian2.py 100    # ground truth, with a noise floor
```

No pull request has been opened. This repo is a benchmark; changing one
backend's numbers alters a published comparison, and finding §1 changes what the
comparison means. Both seemed like conversations to have first.
