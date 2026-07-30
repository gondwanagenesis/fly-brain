# Nine neuron models over one connectome

**Switchable membrane models for the whole-brain *Drosophila* simulation, each
optimised, each independently validated.**

- Branch `perf/event-driven-pytorch`, fork
  [gondwanagenesis/fly-brain](https://github.com/gondwanagenesis/fly-brain)
- Kernel: `flyloop/native/nrn_kernel.c` + `sweep_template.h` + `simd.h`
- Registry: `flyloop/models.py` · Engine: `flyloop/native_engine.py`
- Live 3D interface: `flyloop/studio.py`

---

## 0. What this is

The repository simulates 138,639 neurons and 15,091,983 connectome edges under
one membrane equation: leaky integrate-and-fire, forward Euler. This adds eight
more and makes the choice a runtime switch.

The network is untouched. The connectome, the uniform 1.8 ms axonal delay, the
5 ms exponential synapse, the 2.2 ms absolute refractory period and both
published stimulation protocols are identical across all nine. What changes is
the differential equation each neuron obeys.

| # | model | year | extra state | flop/neuron/dt | exact? |
|---|---|---|---|---|---|
| 0 | **LIF (forward Euler)** — as shipped | — | — | ~5 | no |
| 1 | **LIF (exact propagator)** Rotter & Diesmann | 1999 | — | **~4** | yes, between arrivals |
| 2 | **Izhikevich** | 2003 | `u` | ~10 | no |
| 3 | **AdEx** Brette & Gerstner | 2005 | `w` | ~12–24 | no |
| 4 | **EIF** Fourcaud-Trocmé et al. | 2003 | — | ~9–21 | no |
| 5 | **QIF / theta** Ermentrout & Kopell | 1986 | — | ~5 | no |
| 6 | **Resonate-and-fire** Izhikevich | 2001 | `y` | ~8 | **yes, sub-threshold** |
| 7 | **Hodgkin-Huxley** | 1952 | `m,h,n,armed` | ~180 | gates exact |
| 8 | **GLIF (adaptive threshold)** Allen Institute | 2018 | `θ` | ~7 | **yes, both states** |

---

## 1. The architecture, and why adding a model is cheap

Simulator cost divides into work that depends on the membrane equation and work
that does not:

| component | model-dependent? |
|---|---|
| sparse 1.8 ms delay ring | no |
| event-driven fan-out over 15.1M edges | no |
| refractory gate + compact countdown | no |
| ATen-faithful chunk partition, thread pool | no |
| 16-neuron tile skipping | only through the resting state |
| **per-neuron state update, spike test, reset** | **yes** |

Everything in the first group is a property of the *network and the schedule*,
so it is written once and shared. A model supplies three things and inherits the
rest. Adding one costs a macro in `sweep_template.h` and an entry in
`models.py`.

**Each model body is written once and compiled three times** — AVX-512, AVX2 and
portable scalar — against the small SIMD abstraction in `native/simd.h`. That
makes "the three instruction-set paths agree" a structural property of the build
rather than a claim to re-audit whenever a model is touched. The model switch is
hoisted *outside* the neuron loop, so every model gets a fully specialised loop
with no per-iteration branch.

Model 0 keeps its own hand-written sweep, because it alone has to reproduce
ATen's mixed FMA/non-FMA rounding seam to stay bit-identical to the PyTorch
backend. `lif_step` is now a thin wrapper over the general `nrn_step`, so the
existing gate proves the refactor did not disturb it.

---

## 2. Models are calibrated on excitability, not on millivolts

Each model's published parameters live in its own units and voltage scale: AdEx
in pF/nS/pA around −70.6 mV, Izhikevich in a dimensionless polynomial around
−70 mV, Hodgkin-Huxley in µA/cm² around −65 mV. Dropped onto a connectome whose
weights are calibrated in millivolts against a 7 mV threshold, they produce a
network that is silent or in seizure — and the model comparison then measures
unit mismatch rather than dynamics.

So each model keeps its published *shape* exactly, and one free gain `k_in` is
solved for numerically so that **the same number of simultaneous synapses fires
it as fires the reference LIF**. Here that number is **161.6**. Nothing is
hand-tuned; `models.calibrate()` bisects against the kernel itself.

**The first attempt used PSP amplitude instead, and it was wrong.** Matching
millivolts requires a "distance to threshold", and Hodgkin-Huxley has no
threshold in the LIF sense. Using its action-potential detection level (65 mV
above rest) as the denominator made HH about **five times too excitable** and put
the whole connectome into sustained ~20 Hz firing. That would have read as a
finding about HH dynamics and was entirely a finding about the wrong
denominator. Excitability is also the quantity that governs network behaviour:
what a neuron does depends on how much convergent input recruits it, not on the
units of its membrane variable.

The 161.6 figure is worth keeping in view on its own — it is *why* this
connectome is quiet. A typical neuron needs a coincident volley, not a trickle.

---

## 3. What makes each model fast

**LIF exact is cheaper than the Euler it replaces.** The propagator
`u ← α_m u + P_vg g`, `g ← α_s g` is two FMAs and a multiply; forward Euler is a
subtract, an add, an FMA and a multiply. It is *more accurate and fewer
operations* — Euler is simply dominated, and the usual "Euler is cheaper"
justification does not hold for a linear membrane with an exponential synapse.

**AdEx and EIF: certified exponential elision.** The exponential is ~12 of
~20 operations and is below the rounding floor for most neurons most of the
time. The kernel computes a rigorous upper bound first —
`exp(x) ≤ 2^(round(x·log₂e)+1)`, one round and one scalef — and skips the
polynomial entirely when no lane can matter. The condition `|E| < |S|·2^-25`
implies `|E| < ulp(S)/2`, which implies `fl(S+E) == S` exactly, so this removes
work without removing information. Full argument:
[research/math/certified-exponential-elision.md](research/math/certified-exponential-elision.md).

**Hodgkin-Huxley: Rush & Larsen 1978, from cardiac electrophysiology.** Forward
Euler on the sodium activation gate needs `dt` well under 10 µs because `τ_m`
falls to ~0.05 ms during the upstroke. Rush-Larsen integrates each gate by its
exact solution for frozen `v`, which is unconditionally stable and is what makes
25 µs sub-steps — and therefore whole-connectome HH — viable at all. The method
is standard in the cardiac literature and essentially absent from the SNN
performance literature.

**Resonate-and-fire is exact, including its input.** The sub-threshold flow is a
scaled rotation, and the response to an input held over the step has a closed
form too, so both halves are constants: six multiplies, four adds, no
transcendental at run time.

**Izhikevich is nearly free.** The quadratic is a Horner chain — two FMAs — so
the cost is the second state array's memory traffic, not the arithmetic. It
benchmarks *faster* than LIF on the sugar protocol.

---

## 4. Verification

Five independent gates, because self-consistency alone would not notice a model
that is *consistently* wrong.

### `flyloop/verify_models.py` — the kernel's own gates

```
B  LIF regression vs PyTorch : BIT-EQ

model       aux  rest fp  AVX-512   AVX2  scalar   C skip  D elide
lif_euler     0     True      ref     OK      OK   BIT-EQ        -
lif_exact     0     True      ref     OK      OK   BIT-EQ        -
izhikevich    1     True      ref     OK      OK   BIT-EQ        -
adex          1     True      ref     OK      OK   BIT-EQ   BIT-EQ
eif           0     True      ref     OK      OK   BIT-EQ   BIT-EQ
qif           0     True      ref     OK      OK   BIT-EQ        -
raf           1     True      ref     OK      OK   BIT-EQ        -
hh            4     True      ref     OK      OK   BIT-EQ        -
glif          1     True      ref     OK      OK   BIT-EQ        -

ALL GATES PASS
```

- **B — no regression.** LIF is still bit-identical to the PyTorch backend after
  the kernel grew eight models and `lif_step` became a wrapper.
- **A — ISA equivalence.** Whole-brain state compared as raw uint32 every step
  under forced AVX-512 / AVX2 / scalar. Vector width is a pure speed knob.
- **C — tile-skipping exactness.** Each model run with skipping on and off,
  compared bit-for-bit. This is what catches a model whose "rest" is one ulp off.
- **D — elision exactness.** `elide_tiny = 0` forces the exponential to be
  computed always; identical bits.
- **E — auxiliary state.** `u`, `w`, `y`, `m/h/n/armed`, `θ` are compared as
  uint32 alongside `v` and `g`, so a model cannot pass with a right membrane and
  a drifting second variable.

### `flyloop/verify_exp.py` — exhaustive, not sampled

The vectorised `expf` is the only place the kernel makes an accuracy *choice*.
float32's small cardinality makes the honest check cheap, so **every one of the
2,237,530,114 representable values** in the live domain `[-87, 88]` is checked
against a float64 reference:

```
  AVX-512  2,237,530,114 values   max 1 ULP   mean 0.0077 ULP   99.23% exactly rounded
  AVX2     2,237,530,114 values   max 1 ULP   mean 0.0077 ULP   99.23% exactly rounded
  scalar   2,237,530,114 values   max 1 ULP   mean 0.0077 ULP   99.23% exactly rounded
  all 3 ISA paths bit-identical over the whole domain
```

The first version of this audit silently tested only the positive half — a
negative float32's bit pattern read as `int32` is itself negative, so the range
was empty and the run reported success. That would have left every decay factor,
every Rush-Larsen step and AdEx's whole sub-threshold branch untested. Details:
[research/math/vectorised-exp-exactness.md](research/math/vectorised-exp-exactness.md).

### `code/validate_models_brian2.py` — the external check

Every model written out again in **Brian 2** from the published equations,
sharing no code with the kernel, both sides driven identically at the same step
size. float32-fused-single-pass against float64-general-purpose:

| model | sub-threshold rel. error | spike counts (kernel/Brian) |
|---|---|---|
| lif_exact | 1.5e-05 | 1/1 |
| izhikevich | 1.4e-05 | 2/2 |
| adex | 3.6e-05 | 5/5 |
| eif | 3.3e-05 | 10/10 |
| qif | 7.2e-05 | 1/1 |
| **raf** | **1.5e-06** | 33/34 |
| hh | 1.7e-02 | 1/1 |
| glif | 1.5e-05 | 1/1 |

Eight of nine sit at the float32 floor. Hodgkin-Huxley's 1.7e-2 is an
*integrator* difference — ours is Rush-Larsen gates with an Euler membrane,
Brian's is exponential-Euler throughout — and rather than argue that, the script
measures it:

```
step refinement -- integrator difference, or wrong coefficient?
  model        n_sub    rel err   ratio
  hh               4  1.720e-02      -
  hh               8  8.575e-03    2.01
  hh              16  4.278e-03    2.00
  hh              32  2.153e-03    1.99
```

Clean first-order convergence to the same trajectory. A wrong coefficient would
hold the ratio near 1.

Two real bugs were found by this comparison and fixed: resonate-and-fire was
adding its input as a per-step impulse rather than integrating it (wrong by a
factor of `dt`), and Brian's exact solver *rejects* the resonator outright —
"the solution to the linear system contains complex values" — which is precisely
the case the kernel handles in closed form.

---

## 5. Performance

138,639 neurons, 4 threads, AVX-512, no GPU. Real time is 0.1 ms/step. Minimum
of 7 × 400 steps.

> ⚠️ **Measured on a loaded machine.** The PyTorch baseline reads 5.017 ms/step
> on the sugar protocol here against **1.66 ms/step measured quiet** — roughly
> 3× inflation across the board. Ratios are stable; absolute times are not.
> Stop background indexers before quoting these.

| model | silent | P9 (2) | sugar (21) | broad (1000) |
|---|---|---|---|---|
| | ms · rt · live% | ms · rt · live% | ms · rt · live% | ms · rt · live% |
| lif_euler | 0.0522 · 1.91× · 0.1% | 0.0684 · 1.46× · 14.6% | 0.0966 · 1.04× · 28.5% | 0.7651 · 0.13× · 96.3% |
| lif_exact | 0.0490 · 2.04× · 0.1% | 0.0614 · 1.63× · 14.6% | 0.1080 · 0.93× · 28.8% | 0.7177 · 0.14× · 96.3% |
| **izhikevich** | 0.0556 · 1.80× · 0.1% | **0.0521 · 1.92×** · 10.8% | **0.0745 · 1.34×** · 7.5% | 0.3428 · 0.29× · 94.6% |
| adex | 0.0542 · 1.84× · 0.1% | 0.1320 · 0.76× · 13.6% | 0.2922 · 0.34× · 58.8% | 1.3699 · 0.07× · 96.8% |
| eif | 0.0556 · 1.80× · 0.1% | 0.1024 · 0.98× · 71.4% | 0.2907 · 0.34× · 69.7% | 1.6737 · 0.06× · 98.7% |
| qif | 0.0487 · 2.05× · 0.1% | 0.0779 · 1.28× · 16.6% | 0.1154 · 0.87× · 31.0% | 0.6452 · 0.15× · 97.1% |
| raf | 0.0569 · 1.76× · 0.1% | 0.0613 · 1.63× · 11.1% | 0.0850 · 1.18× · 58.2% | 1.1076 · 0.09× · 95.1% |
| hh | 0.0724 · 1.38× · 0.1% | 0.2629 · 0.38× · 3.3% | 0.2380 · 0.42× · 2.6% | 3.8232 · 0.03× · 94.1% |
| glif | 0.0543 · 1.84× · 0.1% | 0.0624 · 1.60× · 14.6% | 0.0976 · 1.02× · 17.8% | 0.6133 · 0.16× · 96.2% |

`live%` is the fraction of tiles that could **not** be skipped, and it explains
most of the variation. Two models at the same `live%` and different speeds differ
in **arithmetic**; two at the same speed and different `live%` differ in
**dynamics**.

Things worth noticing:

- **Izhikevich is the fastest model on the real protocols**, beating LIF, because
  it drives the network less hard (7.5% live tiles on sugar against LIF's 28.5%).
  A "more complex" model running faster than the simple one is a dynamics
  result, not an arithmetic one.
- **Hodgkin-Huxley is the quietest** — 2.6% live tiles on sugar — because its
  potassium conductance provides intrinsic adaptation no I&F model here has. Its
  ~180 flop/neuron are only paid on 2.6% of the brain, which is why full HH lands
  at 0.42× real time rather than the ~0.03× its arithmetic alone would suggest.
- **EIF has the highest live fraction** on P9 (71.4% against LIF's 14.6%): the
  exponential term keeps weakly depolarised neurons off their resting fixed point
  for longer, so fewer tiles qualify as inert.
- **All nine exceed real time when the brain is quiet**, and none does under
  broad 1000-neuron drive. Tile skipping is activity-dependent by construction.

---

## 6. FlyBrain Studio — the live interface

```bash
.venv/Scripts/python.exe flyloop/studio.py
# -> http://127.0.0.1:8765
```

All 138,639 neurons rendered at their real FlyWire coordinates (99.99% have one),
driven live by the native kernel, with the membrane model switchable at runtime.
Every point is a real neuron and lights when that neuron actually spikes — the
render is the simulation's output, not an animation of it.

- **Model switching costs milliseconds, not a reload.** The connectome, delay
  ring, tiling and permutation are network properties; only the parameter block,
  the auxiliary state and the initial membrane value belong to the model.
- Stimulation protocols (sugar, P9, broad, silent), Poisson rate, playback speed
  from 0.02× to unlimited, spike-trail persistence, exposure.
- Live telemetry: ms/step, multiples of real time, spikes/step, live-tile
  fraction, per-region firing rates, spike-rate history.
- Figure capture writes a PNG named for the model and protocol that produced it.
- Standard library only — no Flask, no websockets, no build step.

The simulation runs in its own thread; `engine.step()` is a ctypes call that
releases the GIL for its whole duration, so the simulation and the server
genuinely overlap. Activity decays once per *frame* rather than per step —
applying an exponential decay to a 138,639-element array 10,000 times a second
in Python would cost far more than the simulation it displays.

---

## 7. Three bugs this work surfaced

1. **AdEx went NaN and fell silent.** Above the cutoff the exponential diverges;
   with the threshold checked once per `dt` rather than per sub-step, `v` reached
   `+inf`, then `w_inf = a(inf − E_L) = inf`, then
   `fma(w − inf, decay, inf) = NaN`. A NaN fails every ordered comparison
   including the spike test, so affected neurons went permanently quiet rather
   than firing — a whole population lost to one arithmetic edge case. Fixed by
   testing per sub-step and integrating `w` toward the cutoff rather than toward
   a divergent `v`.
2. **Resonate-and-fire was wrong by a factor of `dt`**, adding its input as a
   per-step impulse instead of integrating it. Caught by Brian 2, fixed with the
   closed-form `(e^{λh} − 1)/λ` — which made the model *more* exact, not less.
3. **The exhaustive exp audit was testing half its domain** and reporting
   success. Coverage that is not itself checked is not coverage.

---

## 8. Reproducing everything

```bash
.venv/Scripts/python.exe flyloop/verify_models.py 250      # the kernel gates
.venv/Scripts/python.exe flyloop/verify_exp.py --quick     # exp audit, 7 s
.venv/Scripts/python.exe flyloop/verify_exp.py             # exhaustive, ~11 min
.venv/Scripts/python.exe code/validate_models_brian2.py    # external check
.venv/Scripts/python.exe flyloop/bench_models.py 400 7     # the table in §5
.venv/Scripts/python.exe flyloop/verify_all.py 400         # the original LIF gate
.venv/Scripts/python.exe flyloop/studio.py                 # the live interface
```

Research notes:
[multi-model-neuron-kernels.md](research/multi-model-neuron-kernels.md) ·
[certified-exponential-elision.md](research/math/certified-exponential-elision.md) ·
[vectorised-exp-exactness.md](research/math/vectorised-exp-exactness.md)

---

## 9. What this means for the benchmark

This repository exists to compare simulators. The nine models make explicit
something the single-backend framing leaves implicit: **"the fly brain" is
model-dependent.** Same connectome, same protocol, same calibrated synaptic
efficacy, and the network still behaves differently — `lif_exact` produces ~0.5%
fewer spikes than `lif_euler` (exactly the direction the Euler-shortens-τ
analysis predicts), GLIF fewer still, and HH is the quietest of all under
sustained drive.

A benchmark that fixes one membrane equation is measuring frameworks under one
modelling choice. That is a legitimate thing to measure, but it is worth saying
out loud, and it compounds the divergences already documented in
[FINDINGS.md](FINDINGS.md) §1 and §2.
