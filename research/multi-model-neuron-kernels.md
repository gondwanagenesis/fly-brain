---
field: Neuron model families, numerical integration, and simulator architecture
date: 2026-07-31
verdict: Nine membrane models can share one connectome kernel because the expensive machinery (delay ring, event-driven fan-out, refractory gate, tile skipping, thread pool) belongs to the NETWORK, not the membrane equation. Two results are worth carrying elsewhere: exact integration (Rotter-Diesmann) is both more accurate AND cheaper than the forward Euler it replaces, so there is no trade to make; and Rush-Larsen, a 1978 cardiac-electrophysiology method, is what makes whole-connectome Hodgkin-Huxley tractable at all.
---

# Multiple neuron models over one connectome

## The architectural claim

A spiking network simulator's cost divides into work that depends on the
membrane equation and work that does not. In this codebase, measured:

| component | depends on the model? |
|---|---|
| 1.8 ms uniform delay ring (sparse slots) | no |
| event-driven fan-out over 15.1M edges | no |
| refractory gate + compact countdown list | no |
| 16-neuron tile skipping | only through the resting state |
| ATen-faithful chunk partition | no |
| spin-barrier thread pool | no |
| **per-neuron state update** | **yes** |
| **spike condition and reset** | **yes** |

Everything in the first group is a property of the *network topology and the
schedule*. So a model contributes three things — how `(v, g, aux)` advance over
one `dt`, when a spike is declared, and what the reset does — and inherits the
rest.

The practical consequence is that adding a model costs one macro in
`sweep_template.h` and one entry in `models.py`, and cannot regress the others:
each model's body is instantiated into its own fully specialised loop, with the
model switch hoisted outside the neuron loop.

The one coupling is **tile skipping**, which needs the model's resting state to
be a bit-level fixed point of its own update. That is not assumed; it is found,
by relaxing an isolated neuron in the kernel until one further step changes
nothing (`models.find_rest`). All nine models converge to one. If a model ever
did not, the engine disables skipping for it rather than freezing a state that
is still drifting.

## Model families and what each is for

| model | year | states | why anyone uses it |
|---|---|---|---|
| LIF | 1907 (Lapicque) | v, g | the baseline; everything else is measured against it |
| LIF exact | 1999 | v, g | same model, exact propagator instead of Euler |
| QIF / theta | 1986 | v, g | canonical normal form of a type-I excitable neuron |
| resonate-and-fire | 2001 | v, y, g | responds to input *frequency*; inhibition can cause firing |
| Izhikevich | 2003 | v, u, g | four parameters reproduce most cortical firing patterns |
| EIF | 2003 | v, g | spike onset derived from the real sodium current |
| AdEx | 2005 | v, w, g | predicts real cortical spike times to ~2 ms |
| GLIF | 2018 | v, θ, g | adaptation without leaving the linear regime |
| Hodgkin-Huxley | 1952 | v, m, h, n, g | the ground truth the others approximate |

## Result 1: exact integration is strictly cheaper than forward Euler

For the linear LIF with an exponential synapse, the exact propagator over a step
(Rotter & Diesmann 1999) is

```
u ← α_m u + P_vg g       g ← α_s g       u = v − v_rest
α_m = e^(−dt/τ_m)   α_s = e^(−dt/τ_s)   P_vg = τ_s/(τ_m−τ_s) (α_m − α_s)
```

Written with FMAs that is **two FMAs and a multiply**. The forward Euler form
the repository ships is a subtract, an add, an FMA and a multiply — **four**.

So the exact form is more accurate *and* uses fewer operations. There is no
accuracy/speed trade-off to weigh: Euler is dominated. This matters beyond this
repository because the usual reason given for Euler in SNN kernels is cost, and
for the linear-membrane + exponential-synapse case that reason is simply wrong.

Measured here (see `FINDINGS.md` §2): Euler shortens every time constant by
exactly `dt/2`, so `τ_syn` runs 1.00% short and `τ_mem` 0.25% short, and the
whole simulation runs 0.25–1% fast. Against Brian 2's own `method='exact'`, our
exact propagator agrees to **1.5e-5 relative** — the float32 floor.

Caveat: exactness is between synaptic arrivals. Input arriving mid-step is still
quantised to the step boundary. The τ-ratio identity `α_s = α_m⁴` (exact to
1.1e-16 at these constants) makes the constants cheap to derive but is not
required.

## Result 2: Rush-Larsen is what makes whole-connectome HH possible

Hodgkin-Huxley's gating variables obey `dy/dt = α(v)(1−y) − β(v)y`. Forward
Euler on `m` is unstable unless `dt ≪ τ_m`, and `τ_m` falls to about 0.05 ms
during the spike upstroke — so a naive HH needs `dt` well under 10 µs.

[Rush & Larsen 1978](https://pubmed.ncbi.nlm.nih.gov/700741/), from **cardiac
electrophysiology**, notes that with `v` frozen over the step the gate equation
is linear and has a closed solution:

```
y_inf = α/(α+β)      τ = 1/(α+β)
y ← y_inf + (y − y_inf) e^(−dt(α+β))
```

This is unconditionally stable for the gates, which is what makes 25 µs
sub-steps viable and whole-connectome HH tractable at all. It is standard in the
cardiac literature — the method is the reason ten-thousand-cell cardiac tissue
simulations are routine — and essentially absent from the spiking-neural-network
performance literature, which tends to treat HH as simply unaffordable and move
on.

The generalisations are worth knowing about if HH ever becomes the bottleneck:
- [Perego & Veneziani 2009](https://www.researchgate.net/publication/237811008),
  second-order Rush-Larsen.
- [Marsh, Ziaratgahi & Spiteri 2012](https://pubmed.ncbi.nlm.nih.gov/22736685/),
  "The secrets to the success of the Rush-Larsen method and its
  generalizations" — explains *why* it outperforms its formal order.
- [Chen & Zhang 2017](https://arxiv.org/pdf/1712.02260), orders 3 and 4 as
  explicit multistep exponential integrators.

Measured here: our HH (Rush-Larsen gates + Euler membrane, 4 sub-steps) against
Brian 2's `exponential_euler` shows 1.7e-2 relative difference, which **halves
exactly as the step halves** (ratios 2.01, 2.00, 1.99 over three refinements).
That is first-order convergence to the same trajectory — an integrator
difference, not a coefficient error. The refinement test is the thing that
distinguishes those two, and it is cheap; assertions about "it's just float32"
are not.

## Result 3: models must be calibrated on EXCITABILITY, not amplitude

Each model's published parameters live in its own units and voltage scale: AdEx
in pF/nS/pA around −70.6 mV, Izhikevich in a dimensionless polynomial around
−70 mV, HH in µA/cm² around −65 mV. Dropping those onto a connectome whose
synaptic weights are calibrated in millivolts against a 7 mV threshold produces
a network that is silent or in seizure, and any model comparison then measures
unit mismatch rather than dynamics.

The first fix attempted here was to match **PSP amplitude** as a fraction of the
distance to threshold. That is wrong for Hodgkin-Huxley, and the failure is
instructive: HH's "threshold" in the LIF sense does not exist, and using the
action-potential detection level (0 mV, i.e. 65 mV above rest) as the
denominator made HH roughly **five times too excitable**. The whole connectome
went into sustained ~20 Hz firing — a result that looked like a finding about HH
dynamics and was entirely a finding about the wrong denominator.

The invariant that works is **excitability**: solve each model's synaptic gain
so that the same number of simultaneous synapses is needed to fire it as fires
the reference LIF. Here that number is **161.6** — a quantity worth keeping in
view independently, since it is why the connectome is quiet. A typical neuron
needs a coincident volley of ~162 synapses, not a trickle.

Excitability is also the quantity that governs network behaviour: what a neuron
does depends on how much convergent input recruits it, not on the arbitrary
units of its membrane variable.

## What switching the model actually changes

Same connectome, same protocol, same drive, same synaptic efficacy in the
calibrated sense — and the network still behaves differently, because adaptation
and refractoriness differ. `lif_exact` produces ~0.5% fewer spikes than
`lif_euler` on the sugar protocol, which is exactly the direction and magnitude
the Euler-shortens-τ analysis predicts. GLIF's adaptive threshold reduces the
count further. HH is the quietest of the nine under sustained drive, because its
potassium conductance provides intrinsic adaptation that none of the I&F models
have.

This is worth stating plainly for a repository whose purpose is benchmarking
simulators: **"the fly brain" is model-dependent**, and a benchmark that fixes
one membrane equation is measuring frameworks under one modelling choice, not
measuring the fly.

## References

**Models** — Lapicque 1907 · [Ermentrout & Kopell 1986](https://doi.org/10.1137/0146017) ·
[Hodgkin & Huxley 1952](https://physoc.onlinelibrary.wiley.com/doi/10.1113/jphysiol.1952.sp004764) ·
[Izhikevich 2001](https://www.izhikevich.org/publications/resfire.htm) ·
[Izhikevich 2003](https://www.izhikevich.org/publications/spikes.pdf) ·
[Fourcaud-Trocmé et al. 2003](https://www.jneurosci.org/content/23/37/11628) ·
[Brette & Gerstner 2005](https://journals.physiology.org/doi/full/10.1152/jn.00686.2005) ·
[Teeter et al. 2018 (Allen GLIF)](https://www.nature.com/articles/s41467-017-02717-4)

**Integration** — [Rotter & Diesmann 1999](https://link.springer.com/article/10.1007/s004220050570) ·
[Rush & Larsen 1978](https://pubmed.ncbi.nlm.nih.gov/700741/) ·
[Marsh et al. 2012](https://pubmed.ncbi.nlm.nih.gov/22736685/) ·
[Chen & Zhang 2017](https://arxiv.org/pdf/1712.02260)

**Reference implementations** — [Brian 2 AdEx example](https://brian2.readthedocs.io/en/stable/examples/frompapers.Brette_Gerstner_2005.html) ·
[NEST aeif_cond_exp](https://nest-simulator.readthedocs.io/en/latest/models/aeif_cond_exp.html) ·
[Scholarpedia: AdEx](http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model)
