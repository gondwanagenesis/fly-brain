---
field: Floating-point analysis / SIMD kernel design
date: 2026-07-31
verdict: The exponential in AdEx and EIF can be SKIPPED, bit-exactly, whenever a two-operation upper bound proves it would round away. In this connectome that is most neurons most of the time, which is what brings the "expensive" spike-initiation models within a factor of two of plain LIF. The argument is a sufficient condition on |E| < ulp(S)/2, so it removes work without removing information -- and it is verified by running the kernel with the elision switched off and comparing bits.
---

# Certified exponential elision

## The problem

AdEx (Brette & Gerstner 2005) and EIF (Fourcaud-Trocmé et al. 2003) both carry
an exponential spike-initiation term:

```
C dv/dt = -g_L (v - E_L) + g_L Δ_T exp((v - V_T)/Δ_T)  [- w]  + I
```

In a fused float32 SIMD kernel the exponential is roughly twelve of the model's
twenty operations — a Cody-Waite range reduction, a degree-6 polynomial and a
scaling. It dominates the cost of both models.

It also does nothing at all for most neurons most of the time. In the whole-brain
*Drosophila* connectome under either published protocol, 70–99% of neurons sit
at their resting fixed point on any given step, where `v - V_T ≈ -20 mV` and
`Δ_T = 2 mV`, so the exponential's argument is around −10 and the term is
`60 × e^-10 ≈ 2.7e-3` against a leak term of order 1–100. It is far below the
rounding floor of the update it feeds.

The obvious move — skip it when `v` is small — is an approximation, and this
project's rule is that a performance change must not alter output. The question
is whether the skip can be made *exact*.

## The argument

Write one sub-step of the AdEx membrane update the way the kernel executes it:

```
S     = -g_L (v - E_L) - w + I          (everything except the exponential)
E     = g_L Δ_T exp((v - V_T)/Δ_T)      (the exponential term)
v_new = fma(dt/C, S + E, v)
```

**Claim.** If `|E| < ulp(S)/2`, then `fl(S + E) == S` exactly, and therefore
`v_new` is bit-identical to the value computed with `E` set to zero.

**Proof.** Under IEEE-754 round-to-nearest, `fl(x)` is the representable value
closest to `x`. The representable neighbours of `S` are at distance `ulp(S)`
above and (at least) `ulp(S)/2` below. If `|E| < ulp(S)/2` then `S + E` lies
strictly inside the interval that rounds to `S`, so `fl(S + E) = S`. Every
subsequent operation sees the identical operand, so the whole step is
bit-identical. ∎

The condition needs `ulp(S)`, which is awkward to compute — but it only needs to
be *implied*, and `ulp(S) ≥ |S| · 2^-24` for every normal float32. So

```
|E| < |S| · 2^-25   ⟹   |E| < ulp(S)/2   ⟹   the term rounds away
```

is sufficient, and costs one absolute value, one multiply and one compare.

## Bounding E without computing it

The remaining difficulty is circular: the test needs `E`, and `E` is what we are
trying to avoid computing. It is broken with a *monotone upper bound* that costs
two instructions.

For `x = (v - V_T)/Δ_T`,

```
exp(x) = 2^(x log₂e) ≤ 2^(round(x log₂e) + 1)
```

because `round(y) ≥ y - 1/2`, so `round(y) + 1 ≥ y + 1/2 > y`. Both operations
are single instructions on every target: `VROUND` (`vroundps` / `vrndscaleps`)
and `VSCALEF` (`vscalefps` on AVX-512, an exponent-field construction on AVX2
and scalar).

So the guard is:

```c
nb     = VROUND(x * LOG2E) + 1
Ebound = VSCALEF(gL*DT, nb)              /* ≥ the true term */
need   = Ebound ≥ |S| * 2^-25
if (VANY(need)) E = VSEL(need, gLDT * fly_exp(x), 0);
else            E = 0;                    /* proven to round away */
```

Five operations replace twelve, and when no lane in the vector needs the term
the polynomial is not executed at all. Because the bound is an *upper* bound,
`need == false` guarantees the true term also rounds away — the elision can be
conservative but never wrong.

## Why it is checked rather than trusted

The proof above is only as good as its transcription into a kernel, so the
threshold is a runtime parameter (`elide_tiny`, normally `2^-25`). Setting it to
zero makes the test `bound ≥ 0`, which is always true, so the exponential is
computed on every lane. `flyloop/verify_models.py` gate D runs the whole
138,639-neuron brain both ways for 250 steps and compares `v`, `g` and `w` as
raw uint32 every step.

Result: **bit-identical**. If the bound were ever too loose, that gate fails
loudly rather than the results drifting quietly.

## Measured effect

| regime | live tiles | AdEx elision fires |
|---|---|---|
| silent | 0.1% | essentially always |
| sugar GRNs (21) | ~29% | most vectors |
| broad (1000) | ~98% | rarely |

The elision is activity-dependent by construction, exactly like the tile
skipping it composes with, and it degrades to "no benefit, one extra compare"
rather than to a regression.

## Where else this applies

The argument uses nothing specific to the exponential. It generalises to any
*additive* term in a floating-point update for which a cheap monotone bound
exists:

- **Hodgkin-Huxley** — the sodium conductance `ḡ_Na m³ h` is below the rounding
  floor of the total current whenever `m` is small, which is most of the time at
  rest. Not implemented here (HH is dominated by the six rate functions, which
  are needed to *advance* the gates regardless), but the same test applies.
- **Conductance-based synapses** with many receptor types, where individual
  channels are frequently negligible against the total.
- Any **multi-scale sum** where terms span many orders of magnitude — the
  classic case being gravitational or electrostatic N-body far-field terms,
  where the analogous statement is the basis of fast multipole error control.
  The difference is that FMM bounds a *truncation error* and accepts it, whereas
  this bounds the term against the *machine's own rounding* and therefore
  accepts nothing.

## Literature position

Approximate versions of this idea are everywhere: "skip small terms" is the
oldest optimisation in numerical computing. What appears to be unclaimed is the
combination of (a) a *sufficient condition for bit-identity* rather than a
tolerance, (b) a *branch-free monotone bound* cheap enough to compute in the
inner loop, and (c) *whole-system verification* that the elided and non-elided
runs agree bit-for-bit.

The nearest published relatives are lazy evaluation in event-driven spiking
simulators — [Bautembach et al. 2021](https://arxiv.org/abs/2107.04092) skip
neurons rather than terms, and their criterion is a work queue rather than a
rounding argument — and interval-arithmetic term dropping in validated numerics,
which bounds error rather than eliminating it. Neither states the ulp-level
identity used here.

## Caveats

1. **The bound is on the SUB-STEP, not the step.** With `n_sub > 1` the test is
   re-evaluated per sub-step, which is correct but means the branch is taken
   more often than the per-`dt` figure suggests.
2. **`S == 0` forces evaluation.** Then `|S| · 2^-25 = 0`, `need` is true, and
   the exponential is computed. Correct, and the conservative direction.
3. **The `|S| ≥ ulp` step assumes normal floats.** If `S` were subnormal the
   inequality `ulp(S) ≥ |S| 2^-24` still holds (subnormals have a *fixed* ulp,
   larger relative to their magnitude), so the condition remains sufficient.
4. **It does not help at high activity**, by construction. Under broad
   stimulation almost every neuron is depolarised, the bound fails, and AdEx
   pays full price. The benchmark reports this rather than hiding it.
