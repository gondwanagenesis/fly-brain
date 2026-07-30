---
topic: provable neuron/tile skipping, GPU-friendly
date: 2026-07-30
verdict: sup_{s in (0,D]} kappa(s) = kappa(D) exactly, because D=1.8ms sits below the interior peak s*=(tau_m tau_s/(tau_m-tau_s))ln(tau_m/tau_s)=9.242ms where kappa is provably still rising -- this replaces the repo's 200,001-point sampled max with a closed-form theorem and a load-time-checkable precondition (D <= s*); tile-granular GPU skipping is derivable and lossless from the same state-fixed-point argument the CPU code already uses, but at the measured no-locality dirty fraction (~8.84% of neurons) it saves too little (<=5.2% of 32-tiles, <=0.3% of 64-tiles) to be worth deploying until ID-locality of the "ever touched" set is actually measured.
fidelity: provably exact (both results); the tile-occupancy payoff estimate is an honest quantitative bound, not itself a fidelity claim
---

# Provable neuron/tile skipping on GPU: an exact inertness predicate, a closed-form kappa bound, and a Beamer-style delivery-switch fix

Three results. The first two are theorems (closed form, proof, load-time
precondition). The third is a quantitative honesty check on the first, done
with the repo's own measured numbers, that argues against deploying the tile
version of it as-is -- which is itself the useful output, per Rule 3
(HANDOFF §9: "ask research questions that can come back negative").

**Scope honesty**, matching ANALYSIS.md's own framing: no CUDA device was
available. The GPU kernel design in §3 is pseudocode built from the roofline
numbers this repo already published (`data/benchmark-results.csv`), not a
measurement. The exactness proofs in §1 and §2 need no GPU and are complete
as written.

---

## 1. Result A: the exact inertness predicate, generalized to tile granularity

### 1.1 What "at rest" must mean -- enumerate every state slot

`flyloop/brain_engine.py`, `_prune_active` (lines 269-273), already computes
the correct per-neuron predicate; nobody has stated it as a theorem or lifted
it to tile granularity. The four conditions it checks, in the paper's
notation (`u = v - v_rest`):

1. `u_i = 0` (equivalently `v_i = v_rest`)
2. `g_i = 0`
3. `rf_i >= rs_i` (refractory countdown has finished -- the neuron is not
   mid-reset)
4. `buf[l, i] = 0` for **every** `l` in the delay ring, `l = 0..L-1`,
   `L = floor(t_dly/dt) + 1 = 19` (`flyloop/brain_engine.py:74`, confirmed by
   grep: `self.L = int(self.p["tDelay"] / dt) + 1`) -- **this is condition
   (4) the topic asks to enumerate explicitly: all 19 delay-line slots, not
   just the head.** The code checks it with
   `self.buf[:, idx].abs().amax(0) == 0`, an O(L) reduction per neuron.

Plus a structural precondition, not a state variable: the neuron must not be
externally driven this step (`stim_idx` neurons are excluded from the inert
set unconditionally, line 274-275) -- Poisson injection can perturb `u_i`
even when all four conditions above hold, so "inert" must be conditioned on
"not itself a stimulus target."

### 1.2 Proof that the all-zero state is an exact fixed point

Claim: if conditions 1-4 hold for neuron `i` at step `t`, and no fan-out
delivery targets `i` at step `t` (i.e. `i` is not a scatter target of any
neuron that fires at `t`), then conditions 1-4 hold again at step `t+1`,
**bit-for-bit**.

Proof. The free-flow propagator (ROADMAP.md §5.1, verified to machine
precision) is

```
u(h) = x*u0 + (g0/3)*(x - x^4)        g(h) = x^4 * g0        x = exp(-h/tau_mem)
```

Substituting `u0 = 0`, `g0 = 0`:

```
u(h) = x*0 + (0/3)*(x - x^4) = 0        g(h) = x^4 * 0 = 0
```

Every multiplication is `finite * 0.0`, which IEEE-754 defines as exactly
`0.0` (or `-0.0`, sign-immaterial here since both operands are non-negative),
and `0.0 + 0.0 = 0.0` exactly -- no rounding step in this derivation can
produce a nonzero result. So conditions 1-2 persist exactly, for any kernel
that implements the propagator (window-mode `free_window_map`, grid-mode
Euler-or-exact single-step, or the AVX-512 fused kernel) because all of them
reduce to the same zero-in/zero-out algebra.

Condition 3 (`rf_i >= rs_i`) is a **monotone, saturating comparison**: once
true it stays true under any non-decrementing update to `rf_i` (the only way
`rf_i` resets is a spike, which is excluded by "no fan-out delivery targets
`i`" combined with "`i` does not itself spike," and a neuron with `u=0` is by
construction not above threshold). So condition 3 persists trivially.

Condition 4 persists by the window's own semantics: the ring rotates
(`head = (head+1) % L`) and the vacated slot is overwritten by the current
step's delivery. If no delivery targets `i` this step, the newly-written slot
value is `0.0` (an explicit write of zero, or no write at all with the array
pre-zeroed), and the other 18 slots are untouched, so all `L=19` remain zero.

Hence (1)-(4) form an invariant set under the free-flow map, established by
induction on `t`. It breaks only at the exact step a delivery writes a
nonzero value into some `buf[l,i]` -- which is precisely the "fresh" wake
event the existing code already detects
(`fresh = tgt[~self.active[tgt]]`, line 314). **∎**

This is not a new algorithm; it is the proof that the CPU code's predicate
was already correct, stated so it can be trusted on a different hardware
target without re-deriving it from scratch.

### 1.3 Lifting to a tile

Partition the `N = 138,639` neurons into tiles of `W` (32 or 64), padding the
last tile with dummy neurons that have zero in-degree and zero out-degree
(permanently, structurally inert -- assert this at load time, see §5).

```
N = 138,639
W = 32:  T = ceil(N/32) = 4,333 tiles   (last tile: 15 real + 17 pad)
W = 64:  T = ceil(N/64) = 2,167 tiles   (last tile: 15 real + 49 pad)
```

Define the per-neuron "dirty" bit `d_i = NOT(conditions 1-4 hold for i)`.
Define the per-tile occupancy word `word_t = OR_{i in tile t} d_i`. Tile `t`
is exactly skippable at step `t` iff `word_t == 0` **and** no delivery this
step targets any neuron in tile `t`. By §1.2 applied neuron-by-neuron, if
`word_t == 0` and the tile receives no delivery, **every** neuron in the tile
is an exact fixed point, so the whole tile can be skipped -- no read, no
write, no compute -- with the post-state identical to what a full update
would have produced. This is the "one test skips a whole warp/tile" claim:
`word_t` is a single machine word, tested once per tile per step.

**Fidelity: bit-identical.** The proof above is neuron-local and composes
trivially over a tile; nothing in it depends on `W`.

---

## 2. Result B: the closed-form supremum of kappa over (0, D] -- replacing the sampled max

### 2.1 The problem with the current implementation

`code/window_step.py:60-67`:

```python
def kappa_max_over_window(D=DELAY, tau_m=TAU_M, tau_s=TAU_S):
    s = np.linspace(1e-12, D, 200001)
    return float(kappa(s, tau_m, tau_s).max())

KAPPA_WINDOW_MAX = kappa_max_over_window()   # = 0.0720849531
```

This is a **numerical experiment**, not a proof of an upper bound. It reports
the max over 200,001 sample points; the true continuous supremum over `(0,
D]` could in principle exceed the sampled max between grid points, and
nothing in the code certifies that it doesn't. `KAPPA_WINDOW_MAX` gates
`certified_no_spike` (`window_step.py:88-113`), which is used to prove a
neuron *cannot* cross threshold and therefore needs no repair. **An
under-estimated bound here would produce false-silence certifications** --
exactly the failure mode `code/exact_spike_time.py`'s own docstring calls
"safety-critical" and reports as "0 false-silence certifications" for the
unrelated, already-closed-form `KAPPA_MAX` constant in that file. The
sampled `KAPPA_WINDOW_MAX` has never been given the same certificate.

It happens to be numerically correct today only because `np.linspace(...,
200001)` includes the right endpoint `s=D` by construction (`endpoint=True`
is the default), and — as proven below — the true maximum happens to sit
exactly at that endpoint. That is a property of this specific `D`, `tau_m`,
`tau_s`, discovered by luck of the sampling grid, not asserted by the code.

### 2.2 Closed form

`kappa(s) = (tau_s/(tau_m - tau_s)) * (exp(-s/tau_m) - exp(-s/tau_s))`. With
this model's exact values `tau_m = 20`, `tau_s = 5`:

```
kappa(s) = (1/3) * (exp(-s/20) - exp(-s/5))
```

Differentiate:

```
kappa'(s) = (1/3) * exp(-s/20) * ( -1/20 + (1/5)*exp(-3s/20) )
```

The prefactor `(1/3)*exp(-s/20)` is strictly positive for all finite `s`, so
`kappa'(s)` has the sign of the bracket. Setting the bracket to zero:

```
(1/5)*exp(-3s/20) = 1/20
exp(-3s/20) = 1/4
-3s/20 = -ln(4)
s* = (20/3)*ln(4)  =  20*ln(4)/3
```

Numerically: `ln(4) = 1.3862943611`, so `s* = 27.7258872/3 = 9.2419624` ms.
(Cross-check: `x* = exp(-s*/tau_m) = 4^(-1/3) = 0.6299605...`, exactly the
`_XSTAR_FREE` constant already defined in `code/exact_spike_time.py:36` --
the two files' derivations are consistent, as they must be, since
`kappa(s) = (x - x^4)/3` with `x = exp(-s/tau_m)` is the same function in
both places, one parameterized by `s`, one by `x`.)

For `s < s*`, `exp(-3s/20) > exp(-3s*/20) = 1/4` (the exponent is monotone
decreasing in `s`), so the bracket `-1/20 + (1/5)exp(-3s/20) > -1/20 + (1/5)(1/4) = 0`
strictly, hence `kappa'(s) > 0` on `(0, s*)`. **kappa is strictly increasing
on the entire interval `(0, s*)`.**

The model's delay is `D = 1.8 ms`. Since `D = 1.8 < s* = 9.2420` (`D` is only
`19.5%` of the way to the peak), the window `(0, D]` lies entirely inside the
increasing region, so:

```
sup_{s in (0,D]} kappa(s) = kappa(D)
```

exactly, attained at the right endpoint. Evaluating:

```
kappa(1.8) = (1/3) * (exp(-0.09) - exp(-0.36))
           = (1/3) * (0.9139311853 - 0.6976763261)
           = (1/3) * 0.2162548592
           = 0.0720849531
```

**This matches the code's sampled value to all 10 reported digits** -- the
theorem confirms the number, and supplies the certificate the sampling
method could not.

### 2.3 General closed form and the load-time-checkable precondition

For arbitrary `tau_m, tau_s` with `r = tau_m/tau_s > 1`:

```
s* = tau_m*tau_s*ln(r) / (tau_m - tau_s)           (interior peak location)
x* = r^(-1/(r-1)) = exp(-s*/tau_m)
kappa(s*) = (x* - x*^r) / (r - 1)                   (= KAPPA_MAX, unbounded-horizon sup)

sup_{s in (0,D]} kappa(s) = kappa(D)      if D <= s*
                          = kappa(s*)     if D >  s*
```

The **precondition to assert at load time** (mirroring the repo's existing
convention, e.g. ANALYSIS.md §1's `assert all(w==round(w))`):

```python
assert DELAY <= TAU_M*TAU_S*np.log(TAU_M/TAU_S)/(TAU_M-TAU_S), \
    "delay exceeds kappa's interior peak; KAPPA_WINDOW_MAX must switch branches"
```

If the model's delay or time constants ever change such that `D > s*`, the
correct bound becomes the constant `KAPPA_MAX = 0.15749013` already computed
in `code/exact_spike_time.py` (same function, different closed form, no new
derivation needed) -- but the current `D=1.8ms, s*=9.242ms` sits safely in
the first branch by a factor of >5.

**Fidelity: provably exact.** Not an approximation of the sampled value --
a proof that replaces it, with a runtime-checkable precondition guarding the
one case (`D > s*`) where the formula would need to switch branches.

---

## 3. GPU implementation: hierarchical bitset + popcount, warp-uniform skip

This is a design (projection, no CUDA device available), built directly on
Result A (§1).

### 3.1 Two-level bitset

```
Level 0 (per-tile occupancy word): one bit per neuron, packed W-per-word,
         W = 32 or 64 to match warp / sub-warp granularity.
         word[t] = OR over the W neurons in tile t of their dirty bit d_i.

Level 1 (per-meta-tile word): one bit per Level-0 word, packed W-per-word.
         meta[m] = OR over W consecutive tile words of (word[t] != 0).
```

`T = ceil(N/W)` level-0 words; `ceil(T/W)` level-1 words. At `W=64`: 2,167
level-0 words (~17 KB) and 34 level-1 words (272 B) -- both resident in a
single SM's shared memory or L1, matching the sibling note's own sizing
observation (`research/activity-structure-frontier.md`, technique 3) for the
CPU case, reused here for a different purpose.

### 3.2 Kernel skeleton (pseudocode, GeNN-style one-block-per-tile)

```c
// One CUDA block per tile of W neurons. Block-uniform read: every thread
// in the block/warp reads the SAME word[t], so the early-exit branch is
// warp-coherent -- no divergence penalty, unlike a per-neuron predicated
// skip.
__global__ void lif_dense_step_tiled(...) {
    int t = blockIdx.x;
    if (word[t] == 0u) return;          // whole tile provably unchanged
                                          // -- zero bytes read or written
    int i = t * W + threadIdx.x;
    if (i >= N) return;                 // padding lane
    // ... normal LIF update for this neuron, exactly as today ...
    // on write-back, update word[t] and meta[t/W] via atomicOr if this
    // neuron's post-state is non-inert (the wake path, mirrors
    // "fresh = tgt[~active[tgt]]" in brain_engine.py:314)
}
```

The `meta[]` level lets a **launch-time** grid-shaping step (or a
`__ballot_sync`/`__popc` reduction inside a single coarse kernel) test 64
tiles -- 4,096 neurons -- with one popcount before even issuing the
per-tile blocks, cutting kernel-launch count on top of per-tile skip.
This is the concrete mechanism the topic asks for: "a whole warp/tile is
skipped with one test," extended to "64 tiles skipped with one popcount."

### 3.3 Maintaining `word[]` / `meta[]` cheaply -- avoid the O(L) reduction

The existing CPU predicate re-derives condition 4 (all 19 delay-ring slots
zero) with an O(L) reduction every prune (`self.buf[:, idx].abs().amax(0)`).
On GPU that is a per-neuron 19-wide read on every wake/sleep test -- avoid it
by maintaining a per-neuron **nonzero-slot counter** `c_i in [0, L]`
incrementally instead of re-scanning:

```
on write of a nonzero value into buf[head, i]:  c_i += 1
on rotation-out of a slot whose old value was nonzero:  c_i -= 1
delay-line-clean  <=>  c_i == 0
```

This turns condition 4 into an O(1) compare, matching the O(1) nature of
conditions 1-3, and composes into `d_i` (§1.3) as one more bit, not an
L-wide scan -- the GPU-appropriate version of the same exact predicate.

---

## 4. Quantified payoff against the roofline, and the honest verdict

### 4.1 Occupancy math

Tile-skip only pays when tiles are actually empty. Given a dirty fraction
`p` and **no assumed ID-locality** (the conservative default per Rule 4 --
"sparsity is stimulus-dependent," and per the sibling frontier note's own
observation that the measured spike distribution shows no evidence of
ID-clustering across its 64-wide tiling):

```
P(tile of W neurons fully clean) = (1 - p)^W
```

This repo's own measured dirty fractions (HANDOFF §2.2), read as
*instantaneous* snapshots at each horizon:

```
100 ms:  ever-perturbed 5.22%  ->  provably inert 94.78%   (p = 0.0522)
500 ms:  ever-perturbed 8.84%  ->  provably inert 91.16%   (p = 0.0884)
```

```
p = 0.0884, W = 32:  (0.9116)^32 = exp(32*ln(0.9116)) = exp(-2.961) = 0.0518   ->  5.2% of tiles skippable
p = 0.0884, W = 64:  (0.9116)^64 = exp(-5.922)         = 0.00268   ->  0.27% of tiles skippable
p = 0.0522, W = 32:  (0.9478)^32 = exp(-1.716)         = 0.180    ->  18.0% of tiles skippable
p = 0.0522, W = 64:  (0.9478)^64 = exp(-3.432)         = 0.0324   ->  3.2% of tiles skippable
```

### 4.2 Why the dirty fraction is this large, not the ~0.001% spike rate

The naive intuition is that "dirty" should track the 1.75 spikes/step
(0.0013% of N) headline number. It does not, and the reason is IEEE-754
underflow timing, not network dynamics. `g` only reaches **exact** `0.0`
(condition 2, §1.1) by repeated multiplication by `x_syn =
exp(-dt/tau_syn) = exp(-0.02) = 0.98019867` every step until the value
underflows through the fp32 denormal range. Steps to underflow from an
initial kick `g0` to the smallest positive fp32 denormal `g_min ~ 1.4e-45`:

```
n = ln(g_min/g0) / ln(x_syn)
```

For a single-synapse kick, `g0 = w_syn = 0.275` mV (one edge, weight count
1): `n = ln(1.4e-45/0.275)/ln(0.98019867) = -101.99/-0.020 ~ 5,100 steps ~ 510 ms`.

**Any neuron that ever receives even one synaptic event stays exactly-dirty
for roughly half a second of simulated time afterward, independent of how
small its contribution has become.** This is exactly why the measured
"provably inert" fraction is 94.78% at 100 ms and *drops* to 91.16% at
500 ms (more neurons have been touched by then, and none of the earlier ones
have had time to decay back to exact zero yet -- 500 ms is still under the
~510 ms underflow horizon for a minimal single-synapse kick, and most
touched neurons receive more than one synapse, extending it further). This
repository's own benchmark protocols (`verify_native.py`, `bench_native.py`)
run 500-800 steps = 50-80 ms, well inside this regime, so for the actual
measured experiments, "ever touched" and "currently dirty" are close to the
same set -- the 8.84%/5.22% numbers above are the right ones to use, not an
optimistic instantaneous-spike-count number.

**A tempting but rejected fix: flush `g` to exact zero early once it is
"negligible."** This is *not* free the way it looks. Near the operating
point that actually matters (`u ≈ 0`, an inert neuron), the addition
`u_new = u + g * P_vg` (`P_vg = 0.0049379353` per HANDOFF §1.2, the one-step
`kappa` coefficient) only rounds away to bit-identical-with-`u` once
`g * P_vg` is itself below the smallest representable increment near `u`'s
current magnitude -- which, for `u` near zero, means `g` must already be
within a small constant factor of denormal-small. Flushing "early" at any
looser threshold changes the bit pattern of some future `u`, which is a
lossy change with no rigorous bound offered here, so **it is ruled out**,
not adopted -- consistent with ROADMAP.md's existing rejection of fp16/bf16
neuron-state quantization for the same class of reason (silently dropped
sub-ULP updates).

### 4.3 What this buys, against the stated rooflines

GeNN (RTX 4070, from ANALYSIS.md §3): fixed cost `F = 36.75 us/step`, work
`W = 8.25 us/step`, bandwidth floor `6.60 us/step` for the dense 24 B/neuron
state sweep. CUDA-graph batching over an 18-step window (ANALYSIS.md §3.1)
already brings the *amortized* cost to `~10.3 us/step`, with `~8.25 us` of
that being the (near-floor) dense work `W`.

Applying §4.1's *pessimistic, no-locality* tile-skip fraction to that
residual `W`:

```
W=32, p=0.0884 (500ms regime):  8.25us * (1 - 0.052) = 7.82 us/step   -> ~5% off W, ~4% off the 10.3us total
W=64, p=0.0884:                 8.25us * (1 - 0.0027) = 8.23 us/step  -> negligible
W=32, p=0.0522 (100ms regime):  8.25us * (1 - 0.180) = 6.77 us/step   -> ~18% off W, ~14% off the 10.3us total
```

Compare this to CUDA-graph batching alone, which is already a measured-model
**4.4x** (ANALYSIS.md §3.1) on the *fixed* term `F`. Tile-skip at the
measured, no-locality dirty fraction adds at most **~14-18%** on top of that
at `W=32`, and is essentially free money at `W=64`, where GPU warps are 32
lanes but many architectures group two warps (64 threads) per scheduling
unit -- the tile size that matters for actual divergence-free skip is
architecture-specific and should be benchmarked, not assumed.

**Honest verdict: this is a real, lossless, small-to-moderate win, gated
entirely on how the ~5-9% dirty set distributes across neuron IDs.** If a
neuron-ID permutation exists under which dirty neurons cluster (untested --
same open question the sibling frontier note raises for the CPU tile idea,
§3 there), the tile-skip fraction improves superlinearly (clustering `k`
dirty neurons into `k/W` tiles instead of spreading them across up to `k`
different tiles). Measuring the ID-locality of the dirty set under the
sugar protocol is the single experiment that would tell whether this is
worth deploying past a design doc.

### 4.4 CPU vs GPU context

CPU native kernel: `0.0543 ms/step = 54.3 us/step` (4 threads, HANDOFF
§11.1). Even the full 4.4x CUDA-graph GeNN projection (`~10.3 us/step`)
plus the optimistic 18% tile-skip bonus (`~8.8 us/step`) would put a GPU
roughly **6x ahead of this laptop's already-real-time CPU kernel** — a
different regime from where the CPU work sits, and worth stating precisely
because HANDOFF's own "next steps" (§10, §11.6) currently prioritize CPU-side
`refrac`-bitset and `uint16` weight work that do not port to this GPU
picture without re-deriving their own occupancy math.

### 4.5 Neuromorphic: why this result does not apply there

Loihi 2 (and event-driven neuromorphic silicon generally) has no dense sweep
to skip in the first place -- it is natively event-driven, so "skip an inert
tile" is not an optimization on top of its architecture, it is the
architecture (FORK.md §2.4: "essentially nothing transfers; the hardware is
natively event-driven with no dense sweep, no DRAM ring buffer, no
dispatch"). Result A/B in this note are GPU-specific by construction: they
exist to make a bandwidth-bound dense-array simulator (GeNN, or a custom
CUDA kernel) behave more like the event-driven hardware it is competing
against, without abandoning the dense representation. Nothing here should be
read as adding value to a Loihi-class target.

---

## 5. Failure modes / preconditions

Assert all of the following at load time; any violation invalidates a claim
in this note.

1. **`DELAY <= s* = tau_mem*tau_syn*ln(tau_mem/tau_syn)/(tau_mem-tau_syn)`**
   (§2.3). Currently `1.8 <= 9.2420` with margin >5x. If the model's delay or
   time constants change, `KAPPA_WINDOW_MAX` must switch to the `kappa(s*)`
   branch (already implemented as `KAPPA_MAX` in `code/exact_spike_time.py`
   -- reuse it, do not re-derive).
2. **`tau_mem/tau_syn` stays an exact ratio usable in the closed form.**
   §2's derivation used the specific numeric ratio 4 only insofar as it
   matches the model's own stated exactness (ROADMAP.md §2.4); the general
   formula in §2.3 does not require the ratio to be an integer, so this is
   milder than the quartic-solvability precondition ROADMAP.md §5.3 flags
   ("ratio 5 -> general quintic") -- that precondition binds §5.3's spike
   *timing* solve, not this note's bound, and is unaffected by anything
   here.
3. **Padding neurons in the last tile must have zero in-degree and
   zero out-degree**, asserted once at load (`assert
   outdeg[N:] .sum() == 0 and indeg[N:].sum() == 0` after padding to a
   multiple of `W`), otherwise a "phantom" neuron could receive delivery and
   force a tile awake for no real reason -- harmless to correctness (it would
   just waste one tile's work) but worth asserting so a connectome change
   that adds real neurons up to a tile boundary does not silently start
   depending on stale padding.
4. **Tile-skip Mechanism A (this note) is orthogonal to, and must not be
   confused with, the certify/quartic-repair skip already in
   `code/window_step.py`/`code/exact_spike_time.py`.** That mechanism uses
   the *weaker* `certified_no_spike` condition (does not require `g==0`
   exactly, only a proven-below-threshold bound) and currently achieves a
   much larger candidate reduction (86.67%-99.87%, ANALYSIS.md §2.6, §2.8)
   for a *different* purpose -- skipping the expensive quartic root-find, not
   skipping memory traffic in a dense per-step sweep. Do not report this
   note's smaller (5-18%) tile numbers as if they supersede that existing,
   larger result; they answer a different question (bytes moved vs.
   root-finds avoided).
5. **The occupancy numbers in §4.1/§4.3 assume no neuron-ID locality of the
   dirty set.** This has not been measured in this session (same gap the
   sibling frontier note flags for its own, CPU-context tile rejection). If
   measured and found favorable, redo §4.1-4.3's arithmetic with the
   measured clustering coefficient before claiming a larger number.
6. **The frontier/Beamer material this topic also asked for --** switching
   the delivery fallback on `Σ out-degree(spiking)` rather than spike count,
   and the `_dense_fallback` one-way-latch bug -- **is already covered in
   full, with its own arithmetic and a concrete patch plan, in
   `research/activity-structure-frontier.md`** (Applicable Techniques 1 and
   2 there). This note does not re-derive it; it only adds the
   tile/GPU-specific angle that document explicitly does not cover (that
   document's own hierarchical-bitset rejection, technique 3, is scoped to
   the CPU active-set/searchsorted representation and its measured
   occupancy at 1.75 spikes/step -- a different comparison than the
   GPU-dense-kernel-vs-tile-skip comparison made here in §4, which compares
   against the GeNN roofline's bandwidth floor, not against a CPU
   sorted-index gather).

---

## References

Real citations only, all previously verified in this repository's own
research notes (no new citation risk introduced):

- Bellman, R., Cooke, K.L. (1963). *Differential-Difference Equations.*
  Academic Press. -- method of steps, already cited in ROADMAP.md §5.4 and
  `code/window_step.py`'s own docstring; underlies why the window (0, D] is
  the correct domain for the kappa supremum in §2.
- Hairer, E., Nørsett, S.P., Wanner, G. *Solving Ordinary Differential
  Equations II*, ch. II.17 -- same role, already cited in ROADMAP.md §5.4.
- Beamer, S., Asanović, K., Patterson, D. (2012). "Direction-Optimizing
  Breadth-First Search." *SC12*. DOI
  [10.1109/SC.2012.50](https://dl.acm.org/doi/10.1109/SC.2012.50). Source of
  the edge-weighted (not vertex-count) switching criterion this note's §5.6
  points back to in `research/activity-structure-frontier.md`.
- Shun, J., Blelloch, G.E. (2013). "Ligra: A Lightweight Graph Processing
  Framework for Shared Memory." *PPoPP 2013*.
  https://www.cs.cmu.edu/~guyb/papers/SB13.pdf -- independent confirmation
  of edge-count switching; also the closest precedent for the two-level
  bitset representation used in §3.1.
- Zhang, Y. et al. (2018). "GraphIt: A High-Performance DSL for Graph
  Analytics." arXiv: [1805.00923](https://arxiv.org/abs/1805.00923) --
  frontier representation (bitmap vs. sparse array) as an explicit design
  choice, same framing as §3.1-3.2 here.
- Bautembach, D., Oikonomidis, I., Argyros, A. (2021). "Even Faster SNN
  Simulation with Lazy+Event-driven Plasticity and Shared Atomics." *HPEC
  2021*. arXiv: [2107.04092](https://arxiv.org/abs/2107.04092) -- per-neuron
  bitfield + intrinsic popcount/`__clz()` precedent for §3's hierarchical
  bitset, in an SNN-specific context.
- ANALYSIS.md §1-3 (exact accumulation theorem, sustained-cost bound, GPU
  roofline derivation), HANDOFF.md §2.2 (quiescence measurements used in
  §4.1), §2.4 (tau-ratio identity used in §2.2), §5.1-5.2 (propagator and
  `KAPPA_WINDOW_MAX`), §11 (native kernel numbers used in §4.4) -- all reused
  directly from this repository's existing, previously-verified record.
- `research/activity-structure-frontier.md` -- covers the Beamer/Ligra/
  GraphIt delivery-switch fix and the `_dense_fallback` latch bug this
  topic also asked about; cited rather than duplicated per §5, item 6 above.
