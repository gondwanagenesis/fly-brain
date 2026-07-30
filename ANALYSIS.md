# Mathematical analysis: exactness structure, cost bounds, and the GPU roofline

Three results, none of which required a GPU to derive, and two of which are
theorems about this specific model rather than engineering measurements.

**Scope honesty:** no CUDA device was available. Everything about GPU behaviour
below is a **roofline/decomposition argument built from this repository's own
published benchmark numbers** (`data/benchmark-results.csv`, RTX 4070 per the
README). The exactness results are proven and verified on CPU. Rule 2 of this
project is *measure, don't model* — §3 is a model and is labelled as one.

---

## 1. Theorem: synaptic accumulation is exact and order-independent

### Statement

Let `w_ij` be the connectome weights as used by this model. Then for any
postsynaptic neuron `i`, any subset `S` of presynaptic neurons, and **any order
of summation**, the float32 accumulation

```
rec[i] = Σ_{j∈S} w_ij
```

incurs **no rounding whatsoever** and equals the exact integer sum. In
particular it is independent of the order of operations.

### Proof

Measured over all 15,091,983 edges (`data/fanout_csc.pt`):

- every `w_ij` is an **exact integer**: `max |w − round(w)| = 0.0`
- `max |w_ij| = 2405`
- `max_i Σ_j |w_ij| = 69,948`

Every partial sum is therefore an integer of magnitude at most 69,948.
float32 represents every integer in `[−2²⁴, 2²⁴] = [−16777216, 16777216]`
exactly, and 69,948 ≪ 2²⁴. An IEEE-754 addition whose two operands *and* whose
exact result are all representable is performed exactly. By induction on the
summation, every partial sum is exact, so the final value equals the exact
integer sum — which is manifestly independent of order. ∎

### Verification

`code/prove_exact_accumulation.py`: 200 random permutations of the accumulation
order, at 50 / 500 / 5000 spiking neurons (up to 550,918 synapses), every result
bit-identical, and every result equal to the exact `int64` sum.

### Why this matters

Floating-point addition is not associative. That single fact is why the fan-out
in this fork had to be accumulated in one specific order to preserve
bit-identity, and it is why GPU scatter-adds are normally non-deterministic.
**For this connectome that constraint does not exist.** Consequences:

1. **The fan-out parallelises freely and remains bit-identical** — CPU threads,
   GPU `atomicAdd`, warp-level reductions, segmented scans, any order at all.
   This is the single largest untapped win in the saturating regime, which is
   currently this fork's weakest (1.23×) precisely because fan-out is serial.
2. **A GPU port becomes bit-reproducible.** `atomicAdd(float*)` on CUDA is
   normally run-to-run non-deterministic because the reduction order varies.
   Here, determinism follows from *exactness*, not from ordering — so a GPU
   implementation can be bit-reproducible, which **no current GPU backend in
   this repository is**.
3. **Weights are losslessly `int16`** (`|w| ≤ 2405 < 32767`), halving the 60 MB
   weight array and the dominant fan-out memory traffic.
4. **`int32` accumulation is also exact** and x86/CUDA have *native* integer
   atomics, whereas atomic float add on x86 requires a CAS loop.

### Precondition — must be checked, not assumed

This is a property of *this dataset*, not of the model equations. If the
connectome is ever rescaled, resampled, or replaced with non-integer weights,
the theorem fails silently and order-dependence returns. Any implementation
relying on it must assert `all(w == round(w))` and
`max_i Σ_j |w_ij| < 2²⁴` at load time. `code/prove_exact_accumulation.py`
performs exactly this check and is safe to run in CI.

---

## 2. Theorem: the sustained cost is bounded above

The refractory period is **absolute** and 2.2 ms = 22 steps, so in any 22
consecutive steps each neuron fires at most once. Therefore the total fan-out
work in any 22-step window is at most the total edge count, giving a **sustained
bound that holds for every possible stimulus**:

| quantity | bound |
|---|---|
| dense sweep | 138,639 neuron updates/step (fixed) |
| sustained fan-out | ≤ 15,091,983 / 22 = **685,999 synapse updates/step** |
| worst-case sustained ratio | **4.95 : 1** (synaptic : dense) |
| measured on sugar | 1 : 728 (the other way round) |

⚠️ **This inverts the project's headline finding by ~3600×.** HANDOFF §2.1
records "synaptic propagation is 0.14% of runtime, so optimising spike delivery
is a trap." That is true *for the sugar protocol* and false in general: at
maximum sustained firing, synaptic work is ~5× the dense sweep. The 728:1
number is a property of the stimulus, not of the model, and any design that
assumes it will fall over under broad or embodied drive.

The **instantaneous** peak is not bounded by N/22 — all N neurons could fire on
one step and then be silent for 22 — so peak single-step fan-out is the full
edge count. A hard real-time guarantee therefore needs either the sustained
bound plus buffering, or an admission-control argument.

This bound is what makes a **real-time guarantee** possible for the closed-loop
embodied case in `virtualfly/`: you can provision for 686k synapse updates/step
and prove the deadline is never missed on a sustained basis, without ever
measuring a specific stimulus.

---

## 3. Where the GPU time actually goes (model, not measurement)

### 3.1 GeNN is already at the memory roofline; its problem is fixed overhead

GeNN's own two data points in `data/benchmark-results.csv` are enough to
separate fixed per-step cost from per-trial work. With `F` the fixed cost and
`W` the work per trial:

```
n=1:  F +  W = 0.450    s/sim-second
n=8:  F + 8W = 8 × 0.128422 = 1.0274 s/sim-second
```

Solving: **W = 8.25 µs/step, F = 36.75 µs/step — 81.7% of GeNN's single-trial
cost is fixed overhead.**

Now the roofline. The dense per-neuron state is `v, g, refrac` in fp32, read and
written: 24 B/neuron × 138,639 = 3.33 MB/step. At the RTX 4070's ~504 GB/s:

```
3.33 MB / 504 GB/s = 6.60 µs/step   (= 0.066 s/sim-second)
```

**GeNN's measured work of 8.25 µs/step is within 25% of the 6.60 µs bandwidth
floor.** Its kernel is essentially optimal. Nothing in this fork improves it.
The entire opportunity is the 36.75 µs of fixed cost, which is kernel-launch
latency — confirmed by the fact that batching (which amortises launches across
trials) is what recovers it, and by the fact that Brian2GeNN, NEST GPU and
Brian2CUDA are all *flat* under batching, i.e. not launch-bound.

**The lever for GeNN is CUDA Graphs, not a better kernel.** And the method-of-
steps decoupling (HANDOFF §5.4) says something stronger: because the uniform
1.8 ms delay makes the network provably non-interacting within a window, an
**entire 18-step window can be captured as one CUDA graph** with no host
intervention. That amortises the fixed cost 18×:

```
36.75/18 + 8.25 ≈ 10.3 µs/step  ->  ~0.10 s/sim-second
```

i.e. ~4.4× faster than today, and within ~2× of 12 Loihi 2 chips (0.0538) — on
one consumer GPU.

### 3.2 PyTorch CUDA: 80% of it is accounted for by three fixable things

At 6.509 s/sim-second = 651 µs/step:

| cost | µs/step | fixed by |
|---|---|---|
| dense spmv over 15.1M nnz (val+idx, 8 B) | 240 | event-driven fan-out (already upstream, `040f699`) |
| ~30 kernel launches @ ~8 µs | 240 | fusion + CUDA Graphs (§2.2 of FORK.md) |
| `torch.roll` of the (19, N) delay buffer | 42 | sparse delay slots (§2.2c of FORK.md) |
| **accounted** | **522 of 651** | |

Every one of these is something this fork already fixes on CPU, and all three
port to CUDA unchanged. This is why PyTorch CUDA — not GeNN — is where the
exploitable headroom is: **~30–60× is available, versus ~4× for GeNN.**

### 3.3 Why neuromorphic wins by less than expected

Loihi 2 reaches 0.0538 s/sim-second on 12 chips versus 0.543 on four laptop CPU
cores — only ~10×. §2 explains why: neuromorphic silicon accelerates **sparse
spike delivery**, and on the measured protocol spike delivery is 0.14% of the
work. The architecture is aimed at the part that costs nothing here.

Note this cuts both ways. Under the §2 worst case the ratio inverts to 4.95:1
in favour of synaptic work, which is the regime Loihi is built for — so the
neuromorphic advantage should *grow* with activity, and the 10× measured on a
sparse protocol understates it for dense ones. The honest statement is that
**the comparison is stimulus-dependent**, which is exactly the trap HANDOFF
Rule 4 warns about.

---

## 4. Ranked recommendations

| # | change | applies to | expected | fidelity |
|---|---|---|---|---|
| 1 | Parallelise fan-out (§1 licenses it) | CPU + GPU | large in dense regimes | **bit-identical, proven** |
| 2 | `int16` weights | all | halves fan-out traffic | **lossless, proven** |
| 3 | CUDA Graphs over a full 18-step delay window | GeNN, PyTorch CUDA | ~4× GeNN | exact |
| 4 | Sparse delay slots + event-driven fan-out on CUDA | PyTorch CUDA | ~30–60× | exact |
| 5 | Assert the §1 precondition at load time | all | none | prevents silent breakage |
| 6 | Provision from the §2 bound, not from measured activity | embodied/real-time | guarantee | n/a |

Items 1, 2 and 5 are implementable now and need no GPU. Items 3 and 4 need
hardware this machine does not have, and should be **measured before being
believed** — this project has been wrong by 3× in both directions on exactly
this kind of estimate before (HANDOFF Rule 2).
