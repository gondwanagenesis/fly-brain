---
field: Floating-point accuracy, SIMD portability, verification method
date: 2026-07-31
verdict: The kernel's hand-written vectorised expf was audited against EVERY float32 in its live domain -- all 2,237,530,114 of them -- against a float64 reference. Worst error anywhere: 1 ULP. 99.23% of inputs are correctly rounded, and the AVX-512, AVX2 and scalar paths are bit-identical across the entire domain. float32's small cardinality makes exhaustive verification cheap enough that sampling a transcendental's accuracy is a choice, not a necessity.
---

# Exhaustive audit of the vectorised exponential

## Why this needed doing at all

Every operation in every neuron model in this kernel is a single IEEE-754 add,
multiply, FMA or compare — behaviour fixed by the standard and reproduced by
construction across instruction sets. The exponential is the one exception. It
is the only place where the kernel makes an *accuracy choice*, and therefore the
only place where an accuracy claim has to be earned rather than inherited.

It is also load-bearing: AdEx, EIF and Hodgkin-Huxley all route through it, and
HH calls it six times per neuron per sub-step.

## Why sampling would not have been enough

A polynomial approximation's worst case does not sit where random sampling
looks. It hides at a particular reduced argument, usually near a range-reduction
boundary, where the polynomial's error and the reduction's error align. A sweep
of a million random points can miss it entirely and report a reassuring mean
error of 0.5 ULP.

The same reasoning applies to cross-ISA agreement, and more sharply. A
divergence between the AVX-512 and AVX2 paths would need a specific bit pattern
to trigger — a value where the two `VSCALEF` implementations round the exponent
construction differently, say. A whole-brain simulation comparison, which is how
`verify_models.py` checks ISA agreement, exercises maybe a few million distinct
arguments drawn from a narrow, correlated distribution. It would very plausibly
never hit the one that differs.

## float32 makes the honest version cheap

The live domain of the function is `[-87, 88]`. Below `-87.34` `expf` underflows
to zero; above `88.72` it overflows to infinity; the kernel clamps to `[-87, 88]`
first, which pins only values that are already saturated.

That interval contains **2,237,530,114** representable float32 values. That is
not a sampling problem, it is a few minutes of walking. So the audit checks
every one against `np.exp` in float64 rounded to float32, and reports the worst
error anywhere in the domain — a proof over the domain rather than evidence
about it.

## Result

```
vectorised exp audit   domain [-87.0, 88.0]   EXHAUSTIVE
host ISA: AVX-512

  AVX-512  2,237,530,114 values   max 1 ULP at x=-1.490116e-07   mean 0.0077 ULP   99.23% exactly rounded   (238s)
  AVX2     2,237,530,114 values   max 1 ULP at x=-1.490116e-07   mean 0.0077 ULP   99.23% exactly rounded   (163s)
  scalar   2,237,530,114 values   max 1 ULP at x=-1.490116e-07   mean 0.0077 ULP   99.23% exactly rounded   (287s)

all 3 ISA paths bit-identical over the whole domain (checksum 2383807292697390920)
worst error anywhere in the domain: 1 ULP
```

Three statements, all exhaustive rather than sampled:

1. **Max error 1 ULP** — faithful rounding. Not correctly-rounded (that would be
   0 ULP everywhere), which no vectorised expf of this cost achieves.
2. **99.23% correctly rounded**, mean error 0.0077 ULP.
3. **The three ISA paths are bit-identical on all 2.2 billion inputs.** Vector
   width is a pure performance knob with no numerical consequence at all.

## The bug this method caught in itself

The first version of the audit reported 1,118,830,593 values — a plausible
number, a clean run, and exactly half the domain. It walked from
`float32(-87).view(int32)` up to `float32(88).view(int32)`, which is empty for
the negative half, because a negative float32's bit pattern read as a signed
integer is itself negative. The loop ran zero times on the negative side and
reported success.

That would have left **every negative argument untested** — which is most of
what these models evaluate: every decay factor, every Rush-Larsen gate step, and
the entire sub-threshold branch of AdEx pass negative arguments to this
function.

The fix is to enumerate by *magnitude* and set the sign bit afterwards. The
lesson is more general and worth writing down: an exhaustive verification whose
coverage is not itself checked is not exhaustive, and it fails silently and
confidently. The audit now prints its value count, and the number is the thing
to look at first.

## Implementation

Standard Cephes `expf`, expressed once against the SIMD abstraction in
`native/simd.h` and compiled three times:

```
x  ← clamp(x, -87, 88)
n  ← round(x · log₂e)                       nearest-even on all three paths
r  ← x − n·ln2_hi − n·ln2_lo                Cody-Waite, ln2_hi exact in binary32
p  ← degree-5 Horner minimax
y  ← p·r² + r + 1
    ← scalef(y, n)                          2ⁿ by exponent construction
```

Two details do the cross-ISA work:

- **`VROUND` is nearest-even everywhere.** `nearbyintf` (scalar, default rounding
  mode), `_mm256_round_ps(_MM_FROUND_TO_NEAREST_INT)`, and
  `_mm512_roundscale_ps(x, 0)` are the same operation.
- **`VSCALEF` is exact in range.** AVX-512 has `vscalefps`. AVX2 and scalar build
  `2ⁿ` by writing the biased exponent field. These agree exactly for
  `n ∈ [-126, 127]`, and clamping `x` first guarantees `n` stays there — which
  is why the clamp is a correctness measure and not just a saturation guard.

## Cost of the audit

~11 minutes for all three paths on a 4-core laptop, in numpy, once. It is not
part of the routine gate (`verify_exp.py --quick` samples 1-in-64 and runs in
7 seconds); the exhaustive form is run when the exp implementation changes.

## Transferable point

Exhaustive verification of single-argument float32 functions costs minutes. The
habit of reporting "max error over 10⁶ random samples" for such functions is
inherited from float64, where the domain really is intractable, and it is simply
unnecessary one precision down. Anywhere a float32 transcendental is on a
critical path — SNN kernels, ML inference, graphics, DSP — the exhaustive
statement is available for the price of a coffee break, and it is a categorically
stronger claim.
