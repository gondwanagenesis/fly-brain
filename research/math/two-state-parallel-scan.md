---
topic: Associative scan over 2x2 affine maps for GPU
date: 2026-07-30
verdict: The two-state (u,g) scan monoid collapses to a bare 2-vector with a 3-FMA combine because A^n = P(x^n), and refractory 2.2 ms > delay 1.8 ms makes the reset exact at K=1 with zero rollback -- but on an RTX 4070 the scan can never beat the *bit-identical* fused 18-step in-register window, whose 6.60 us/window memory floor already sits 12.8x above its own 0.51 us of arithmetic. Take the fused window (~11x on GeNN, bit-identical); reject the scan.
fidelity: bit-identical for the recommended fused-window path; provably exact (with a <=2.4e-4 mV fp32 steady-state bound) for the scan path, which is rejected on cost, not on fidelity
---

# The two-state (u,g) parallel scan: exact, cheaper than published, and still the wrong answer on GPU

This note does three things.

1. **Derives the two-state scan properly** (§1). The composition rule for the
   (u,g) affine maps collapses to a *one-parameter* family, so the associative
   operator carries a 2-vector and 3 FMAs, not a 2x2 matrix and 8 flops. This is
   the missing "two-state" version of the scalar-V scans in Bullet Trains
   (arXiv:2603.13283) and FPT (arXiv:2506.12087), and it is 2.5x cheaper than
   the naive extension of either.
2. **Proves the reset needs neither speculation nor iteration** (§2). Because
   the absolute refractory period (2.2 ms) strictly exceeds the delay window
   (1.8 ms), *free-run-then-truncate* is exact. Bullet Trains' speculation has
   rollback probability exactly 0 here; FPT's fixed-point iteration converges at
   K=1. This is a strictly stronger exactness statement than either paper, it is
   an algebraic property of the model rather than of the hardware, and it is the
   part that transfers to GPU **and** neuromorphic.
3. **Kills the scan on cost, with arithmetic** (§4). On an RTX 4070 the fused
   18-step window is memory-bound by 12.8x. The scan cannot move memory. Its
   maximum possible benefit over a sequential in-register loop is therefore
   **zero**, and its measured-by-roofline cost is 1.04x (perfectly overlapped)
   to 1.43x (realistic) *slower*. §1 and §2 are not wasted: they are exactly the
   theory that licenses the fused window, which is the thing that pays.

---

## Result

### 1. The composition rule

#### 1.1 The per-step map

State `z = (u, g)` with `u = v - v_rest`. One timestep `h = dt = 0.1 ms`:

```
    z_{k+1} = A z_k + b_k ,        A = [ a  c ]        b_k = ( 0  )
                                       [ 0  d ]              ( I_k)
```

`I_k` is the total synaptic weight arriving at sub-step `k`. **Under the uniform
1.8 ms delay every `I_k` in a window is known before the window begins** (method
of steps; Bellman & Cooke 1963), so `b_k` is data, not a coupling. That is what
makes the map affine and the scan legal at all.

Two instantiations, both used in this repo:

| scheme | `a` | `c` | `d` |
|---|---|---|---|
| exact (Rotter-Diesmann) | `x = e^{-h/tau_m}` | `(x - x^4)/3` | `x^4 = e^{-h/tau_s}` |
| forward Euler (current `lif_kernel.c`) | `1 - h/tau_m` | `h/tau_m` | `1 - h/tau_s` |

Numerically at `h = 0.1 ms`: `a = 0.9950124791926823`, `c = 0.0049379352953090`,
`d = 0.9801986733067554` — these are exactly `alpha_m`, `P_vg`, `alpha_s` from
HANDOFF §4.

#### 1.2 The monoid

Let `T_k(z) = A z + b_k`. Composition of affine maps is associative and closed:

```
    (A1, b1) . (A2, b2)  :  z -> A2(A1 z + b1) + b2 = (A2 A1) z + (A2 b1 + b2)
```

(reading left-to-right in time: `(A1,b1)` applied first). Upper-triangular 2x2
matrices are closed under product, so the monoid is well-defined. The composite
of `m` steps is

```
    z_m = A^m z_0 + r_m ,      r_m = sum_{k=1}^{m} A^{m-k} b_k .          (1)
```

#### 1.3 The collapse — why this is cheaper than the general 2x2 affine scan

For upper-triangular `A` with `a != d`,

```
    A^n = [ a^n   c (a^n - d^n)/(a - d) ]
          [  0            d^n           ]
```

and for **both** schemes in the table above,

```
    c / (a - d) = tau_s / (tau_m - tau_s) = 5/15 = 1/3      exactly.       (2)
```

*Exact scheme:* `c = (x - x^4)/3` and `a - d = x - x^4`, so the ratio is `1/3`
identically. *Euler:* `c = h/tau_m`, `a - d = h/tau_s - h/tau_m =
h(tau_m - tau_s)/(tau_m tau_s)`, so the ratio is again `tau_s/(tau_m - tau_s)`.
It is a property of the *model*, not of the discretisation.

Therefore

```
    A^n = [ a^n   (a^n - d^n)/3 ]
          [  0         d^n      ]                                          (3)
```

and for the **exact** scheme, where `d = a^4` because `tau_m/tau_s = 4` exactly,

```
    A^n = P(y) := [ y   (y - y^4)/3 ]        with y = x^n .                (4)
                  [ 0       y^4     ]
```

**`{A^n}` is a one-parameter group.** `P(y1) P(y2) = P(y1 y2)` — the semigroup
identity already recorded in HANDOFF §5.1, here recognised as the statement that
the matrix part of the scan monoid is *free*: it is a known constant at every
scan level, indexed by the (statically known) span length.

Two consequences, and they are the contribution:

- **The scan operator carries no matrix.** Define the monoid on pairs `(n, r)`
  with `n` the span length and `r` in R^2:

```
    (n1, r1) * (n2, r2) = ( n1 + n2 ,  P(x^{n2}) r1 + r2 )                (5)
```

  Associativity (direct):
  `((n1,r1)*(n2,r2))*(n3,r3) = (n1+n2+n3, P(x^{n2+n3}) r1 + P(x^{n3}) r2 + r3)`
  `(n1,r1)*((n2,r2)*(n3,r3)) = (n1+n2+n3, P(x^{n2+n3}) r1 + P(x^{n3}) r2 + r3)`. Equal.
  Identity `(0, 0)`. It is a monoid, so Blelloch/Ladner-Fischer applies.

- **In a fixed 18-step window the span `n2` is known at compile time at every
  scan level**, so `y = x^{n2}`, `y^4`, and `(y - y^4)/3` are literals. The
  combine is

```
    r_u' = fma( y , r1_u , fma( (y-y^4)/3 , r1_g , r2_u ) )
    r_g' = fma( y^4 , r1_g , r2_g )
```

  = **3 FMA, 6 FLOP, 2 floats of carried state.**

Cost comparison against the naive two-state extension of the published scalar
scans, which would carry the full affine map `(a,c,d,r_u,r_g)`:

| operator | carried state | combine cost |
|---|---|---|
| general 2x2 affine (naive extension of Bullet Trains / FPT to (V,G)) | 5 floats | 8 flop (3 mul + 1 mul + 1 fma for the matrix, 3 for the vector) |
| **collapsed, this note** | **2 floats** | **6 flop (3 FMA)** |
| scalar-V (published) | 2 floats | 2 flop |

So the two-state scan costs **3x** the published scalar-V scan, not the ~4-5x a
naive matrix-carrying implementation would cost. Register/shared-memory pressure
drops 2.5x, which matters more than the flops on GPU.

*Euler caveat.* Under Euler `d != a^4` (`0.98` vs `0.98019867...`), so (4) fails
and the family is two-parameter: carry `(a^n, d^n)` instead of `y`. Both are
still compile-time literals per level and the combine is still 3 FMA on 2 floats
via (3). **Only the "one `pow` advances a dormant neuron `n` steps" shortcut of
HANDOFF §5.1 needs the exact scheme.** Nothing in this note's cost accounting
changes.

#### 1.4 The window response is a fixed 18-tap convolution

Substituting `b_k = (0, I_k)` into (1) with (4):

```
    u_m = x^m u_0 + ((x^m - x^{4m})/3) g_0 + sum_{k=1}^{m} I_k (x^{m-k} - x^{4(m-k)})/3
    g_m = x^{4m} g_0                        + sum_{k=1}^{m} I_k  x^{4(m-k)}       (6)
```

i.e. the prefix trajectory is a rank-2 homogeneous term plus a **lower-triangular
Toeplitz 18x18 matvec** with taps `C_n = (x^n - x^{4n})/3` and `D_n = x^{4n}`.

Sanity check that ties this to existing code: at `n = 18` (`x^18 = e^{-0.09} =
0.9139311852712282`),

```
    C_18 = (0.9139311852712282 - 0.6976763260710310)/3 = 0.07208495306673239
```

which is bit-for-bit `KAPPA_WINDOW_MAX = 0.0720849531` in `code/window_step.py`.
**The certified-silence constant is exactly the (1,2) entry of `A^18`** — the
bound in `certified_no_spike()` is `u_0 + C_18 (g_0 + w_pos)`, i.e. it is the
scan's own window map with every arrival optimistically moved to the tap of
maximal gain. Worth knowing: that is why the bound is tight rather than lucky,
and it is exactly `kappa_max` because `C_n` is increasing on `n in [1,18]`
(`C_n` peaks at `x^n = 4^{-1/3}`, i.e. `n = 92.4` steps, well past the window).

---

### 2. The reset: exact at K=1, no speculation, no iteration

This is the part that generalises past this hardware discussion, so it is stated
as a theorem.

**Setup.** `D = 1.8 ms = 18 steps` (uniform delay), `T_ref = 2.2 ms = 22 steps`
(absolute), `v_reset = v_rest = -52 mV` so the reset state is `z = (0,0)` in
shifted coordinates. Windows are `W_w = ` steps `[18w+1, 18w+18]`.

**Lemma 1 (at most one spike per window).** Two spikes of the same neuron are at
least `T_ref + 1 = 23` steps apart. `23 > 18`. Hence at most one spike per
neuron per window, *including* the case where the neuron is refractory at window
start and released mid-window. []

**Lemma 2 (the reset state is a fixed point of the gated flow).** Under the
PyTorch/`lif_kernel.c` semantics, during refractoriness arriving input is
*discarded* (`gate = refrac >= refrac_steps`) and the linear flow continues. From
`z = (0,0)`:

```
    g' = 0 * c_decay = 0                                   (exact in IEEE-754)
    t  = (v_rest - v_reset) + 0 = 0                        (exact: v_reset == v_rest)
    v' = fma(0, c_mem, v_reset) = v_reset                  (exact)
```

so `z` stays bit-exactly `(0,0)` for the whole refractory period. Not
approximately — the operations are exact on these operands. []

**Theorem (window map).** Let `u^free_m`, `m = 1..18`, be the prefix trajectory
(6) computed *ignoring the threshold*, started from the neuron's state at window
open, with input gated by whatever refractory countdown is in force. Let
`m* = min{ m : u^free_m > theta }` (`= inf` if none). Then the true trajectory is

```
    z_m = z^free_m          for m <= m*
    z_m = (0, 0)            for m* < m <= 18
```

and the neuron's spike set in the window is `{m*}` or empty.

*Proof.* For `m < m*` no threshold has been crossed, so the true and free
trajectories obey the same affine recursion from the same initial state and are
equal. At `m*` the free value crosses, hence so does the true one; the neuron
fires and is reset to `(0,0)`. By Lemma 1 it cannot fire again in the window; by
Lemma 2 it remains at `(0,0)` for the remaining `18 - m* < 22` steps. []

**Corollary A (Bullet Trains speculation, rollback probability 0).** Bullet
Trains scans assuming no spike and rolls back on a detected crossing. Here the
speculative trajectory is *never wrong before the crossing*, and *never used
after it* — the post-crossing values are discarded and replaced by the constant
`(0,0)`, not recomputed. **The rollback is a truncation, not a re-execution.**
Cost of "handling the reset" = one `min`-index reduction over 18 booleans plus a
predicated zero-fill.

**Corollary B (FPT fixed point, K = 1).** FPT iterates the reset to a fixed
point at `K ~ 3`. Here one pass suffices and is exact, because the reset can fire
at most once and annihilates the state. `K = 1` with a proof, not an empirical
convergence rate.

**Corollary C (segment structure).** Per neuron per window there are at most
three segments: `[refractory-frozen at (0,0)]`, `[free affine]`,
`[reset-frozen at (0,0)]`, in that order. A refractory release inside the window
implies a spike >= 22 steps earlier, hence in a *previous* window, so release and
spike cannot collide.

**Corollary D (the delay line collapses to double-buffering).** `D/dt = 18` is
*exactly* the window length. A spike at sub-step `j` of window `W` arrives at
sub-step `j` of window `W+1`. So the arrival list for the next window is
literally this window's spike list with its sub-step indices preserved. The
19-slot ring buffer (10.5 MB dense; ~30 KB sparse in this fork) becomes a single
handed-over list — no ring, no `torch.roll`, no modular indexing.

**Precondition, and it is not free.** Lemma 2 is *semantics-specific*. Under the
Brian 2 semantics documented in FORK §1.1 (state frozen, input **accumulates**
through refractoriness), the post-spike state is not `(0,0)` but
`(0, sum of arrivals during refractoriness)`. The theorem survives — that branch
is still closed form and still at most one spike — but the third segment is
"accumulate, do not decay" rather than "hold zero". Any implementation must
assert which semantics it is reproducing.

---

## Why it is exact

Three separate claims, three separate fidelity classes. Stating them separately
is the point.

### 3.1 The recommended path (fused window) is bit-identical

The fused 18-step in-register window (§5) performs *literally the same
floating-point operations on the same operands in the same order* as the current
per-step kernel. Nothing is reassociated; only residency changes (registers
instead of a round trip to L2/DRAM). Therefore it is **bit-identical by
construction**, and `flyloop/verify_native.py`'s full-state uint32 comparison
applies unchanged.

The two facts that make the fusion legal, both already proven in this repo:

- *No intra-window feedback.* Uniform delay `D = 18` steps, 0 autapses, 0
  zero-delay edges, no gap junctions (HANDOFF §5.4, verified over all 15,091,983
  edges). So no arrival inside window `W` depends on a spike emitted inside
  window `W`.
- *No intra-window ordering hazard on the fan-out.* ANALYSIS §1: every weight is
  an exact integer, `max|w| = 2405`, `max_i sum_j |w_ij| = 69,948 < 2^24`, so
  fp32 accumulation of arrivals is exact and order-independent. The fused kernel
  may therefore emit spikes and scatter fan-out in any order.

### 3.2 The scan is exact in exact arithmetic

Proved in §1.2 (monoid, associativity) and §2 (theorem). No approximation is
introduced at the level of the mathematics: `sum_k A^{m-k} b_k` is the same
quantity however the tree is shaped.

### 3.3 The scan is *not* bit-identical in fp32, and here is the bound

The scan reassociates the sum in (1). It is not bit-identical to the sequential
recursion, and no amount of engineering makes it so. But the direction of the
error is the opposite of what one might fear.

Both schemes evaluate the same inner product `r_m = sum_k I_k C_{m-k}` with the
same exact coefficients. Standard backward-error analysis of summation (Higham,
*Accuracy and Stability of Numerical Algorithms*, 2nd ed., ch. 4) gives
`|fl(S) - S| <= gamma_D * sum|terms|` where `D` is the *depth* of the summation
tree and `gamma_k = k*eps/(1 - k*eps)`, `eps = 2^-24 = 5.9605e-8` for fp32:

```
    sequential, depth 17 :  gamma_17 = 1.0133e-6
    tree scan,  depth  5 :  gamma_5  = 2.9802e-7          ->  3.40x SMALLER
```

**The scan is ~3.4x more accurate than the sequential loop, not less.** The
usual "scan breaks bit-identity" objection is really "scan breaks bit-identity
*to a less accurate reference*".

Accumulation across windows is bounded because the map is a strict contraction:
`||P(x^18)||_1 = x^18 = 0.9139311853 < 1`, so a per-window error `e` reaches a
steady state of at most

```
    e / (1 - x^18) = 11.62 * e .
```

With the realistic envelope observed in this repo (`u in [-10, +7] mV`,
`|g_0| + arriving weight ~ 20 mV`):

```
    e_window   <= gamma_17 * 20 mV      = 2.03e-5 mV
    e_steady   <= 11.62 * 2.03e-5 mV    = 2.36e-4 mV        (3.4e-5 of theta = 7 mV)
```

Worst case, using the connectome's `max_i sum_j |w_ij| = 69,948` synapse counts
at `w_syn = 0.275 mV` and gain `C_18 = 0.0720849531`, the arriving term is
bounded by `0.0720849531 * 69948 * 0.275 = 1387 mV`, giving
`e_steady <= 1.65e-2 mV`, i.e. 0.24% of `theta`. Still a bound, but it must be
reported as such, not as "exact".

**Spike-train invariance certificate.** Spike trains are provably unchanged if,
for every neuron and every sub-step, `|u_m - theta| > e_steady`. That is a
runtime-checkable margin test costing one extra compare per neuron-step. Neurons
failing the margin are re-integrated sequentially (Lemma 1 caps this at one
crossing each), giving **provably identical spike trains with a bounded-size
repair set**. This is the honest way to ship a scan; it is also more machinery
than the fused window needs, since the fused window needs none of it.

---

## What it buys

### 4.1 GPU: the scan's maximum possible benefit is zero

The decisive argument is one line: **the fused window is memory-bound by 12.8x,
and a scan cannot move memory.**

RTX 4070 (AD104): 46 SMs, 5,888 FP32 lanes, 2,475 MHz boost, 29.15 TFLOP/s FP32
(FMA counted as 2), 504.2 GB/s, 1,536 resident threads/SM.

Per **window** (18 steps, whole brain, `N = 138,639`):

| quantity | value | derivation |
|---|---|---|
| dense state traffic | **3.327 MB** | 24 B/neuron (read `v,g,refrac` = 12 B, write 12 B), once per window |
| **memory floor** | **6.599 us** | 3,327,336 B / 504.2e9 B/s |
| sequential in-register compute | **0.514 us** | 138,639 x 18 x 3 FMA = 7.487e6 FMA = 1.497e7 FLOP / 29.15e12 |
| compute : memory | **1 : 12.8** | 0.514 / 6.599 |

The sequential fused window spends 7.8% of its budget on arithmetic. Making
arithmetic *faster* — which is the only thing a scan does — cannot reduce a cost
that is 92.2% memory. **Upper bound on the scan's benefit: 0.00x.**

Now the cost. Warp-level Hillis-Steele inclusive scan over 18 sub-steps,
`ceil(log2 18) = 5` levels, generously assuming perfect lane packing (no padding
waste to 32):

| term | count | rate | time |
|---|---|---|---|
| FMA | 138,639 x 18 x 5 x 3 = 3.743e7 FMA = 7.487e7 FLOP | 29.15 TFLOP/s | **2.568 us** |
| `__shfl_up_sync` | 138,639 x 18 x 5 x 2 = 2.496e7 lane-shuffles | 46 SM x 32 lanes/clk x 2.475 GHz = 3.643e12/s | **6.850 us** |

The shuffles dominate: warp-shuffle throughput on compute capability 8.9 is ~32
threads/SM/clk (CUDA C++ Programming Guide, arithmetic-instruction throughput
table) — a quarter of the FP32 rate — and the scan issues 10 of them per lane.

| kernel | memory | FP32 | MIO/shuffle | window cost |
|---|---|---|---|---|
| **sequential fused window** | 6.599 us | 0.514 us | 0 | **6.599 us** (memory-bound) |
| scan, perfectly overlapped | 6.599 | 2.568 | 6.850 | **6.850 us** (1.04x slower) |
| scan, pipes serialised | 6.599 | 2.568 | 6.850 | **9.418 us** (1.43x slower) |

And that ignores three further costs the scan pays and the sequential loop does
not:

1. **Thread slots.** The sequential window needs `N = 138,639` threads = 1.96
   full-occupancy waves (`46 x 1536 = 70,656` resident). The GPU is *already
   saturated in the neuron dimension*. The scan needs `N x 18 = 2.5e6` threads
   for parallelism the machine has no use for.
2. **Occupancy.** Carrying `(r_u, r_g)` per sub-step plus scan scratch raises
   register pressure on a kernel whose only real constraint is keeping enough
   warps resident to hide DRAM latency. Lower occupancy directly raises the
   6.599 us floor.
3. **Segmentation.** Corollary C means each neuron's window has a per-neuron
   refractory-release boundary. The sequential loop handles it with a predicated
   add; the scan needs a *segmented* scan with data-dependent segment starts.

**Statement of the negative, sharpened.** The scan trades work for depth. This
problem has 138,639-way natural parallelism and needs ~70,656-way. Depth is not
the binding constraint anywhere on the dense path — and the Snir depth-size
tradeoff (`size + depth >= 2n - 2`, Snir 1986) guarantees any prefix circuit on
`n = 18` costs at least `29` combines against the sequential `17`, i.e. **>=1.7x
work inflation, permanently, in exchange for a resource that is already free.**

**The one regime where step-parallelism has somewhere to go, and why it does not
matter.** Step-parallelism only helps when the active set under-fills the
machine: `N_active x 18 <= 70,656`, i.e. `N_active <= 3,925` (2.8% of the brain)
— which *is* this repo's measured sugar regime (~190 active). But at
`N_active = 3,925` the whole window's arithmetic is
`3925 x 18 x 3 FMA = 2.12e5 FMA = 0.0145 us`, against GeNN's measured
**36.75 us** of fixed launch cost. **The scan's entire win window fits 2,500x
inside the kernel-launch overhead.** It is unmeasurable.

### 4.2 GPU: what actually pays, quantified against the GeNN roofline

GeNN today: 0.450 s/sim-second = 45.0 us/step, decomposed in ANALYSIS §3.1 into
`F = 36.75 us/step` fixed (launch) and `W = 8.25 us/step` work, of which the
6.599 us/step dense-state bandwidth floor is the bulk, leaving ~1.65 us/step of
"other" (fan-out, ring buffer, spike bookkeeping).

Both levers are unlocked by §1-§2, i.e. by the same delay-decoupling theory the
scan needed — but neither is a scan:

| lever | mechanism | effect on the decomposition |
|---|---|---|
| **window-as-one-graph** (ANALYSIS §3.1) | 18 steps have no host interaction, capture as one CUDA graph | `F: 36.75 -> 36.75/18 = 2.042 us/step` |
| **window fusion** (this note, §5) | state lives in registers for all 18 sub-steps; loaded once, stored once per window | state traffic: `6.599 -> 6.599/18 = 0.367 us/step` |
| **sparse delay slots + Corollary D** (already in this fork) | ring buffer -> handed-over spike list | "other" `1.65 -> ~0` (upper bound kept conservative below) |

```
    conservative :  2.042 + 0.367 + 1.65 = 4.06 us/step = 0.0406 s/sim-second
    optimistic   :  2.042 + 0.367        = 2.41 us/step = 0.0241 s/sim-second
```

Against the reference points:

| backend | s/sim-second | vs today's GeNN |
|---|---|---|
| GeNN today (RTX 4070) | 0.450 | 1.0x |
| **GeNN + graph + fused window (conservative)** | **0.0406** | **11.1x** |
| GeNN + graph + fused window (optimistic) | 0.0241 | 18.7x |
| Loihi 2, 12 chips, dt = 0.1 ms (Wang et al. 2025) | 0.0538 | -- |
| this fork, native x4, laptop CPU | 0.543 | -- |

**The conservative number beats a 12-chip Loihi 2 by 1.33x on one consumer GPU,
bit-identically.** That is the deliverable, and the scan contributes nothing to
it.

Adding the scan *on top* of the fused window costs `(6.850 - 0.514)/18 =
+0.35 us/step` of MIO in the perfectly-overlapped case: `4.06 -> 4.41 us/step`,
**8.6% slower**. In the launch-free limit (perfect graphs, `F -> 0`) the penalty
is the full 1.43x. There is no operating point at which it wins.

*Labelling, per HANDOFF Rule 2:* §4.1 is a roofline model built from this repo's
own benchmark data plus published RTX 4070 and CUDA-throughput figures. No CUDA
device was available. The shuffle-throughput figure in particular should be
confirmed with a microbenchmark before anyone reports it.

### 4.3 CPU: the scan loses outright; window fusion is the remaining win

Current: **0.0543 ms/step = 54.3 us/step** on 4 laptop cores.

Arithmetic floor. The kernel body is ~8 vector ops per 16 neurons
(`refrac` add, `g` mul, `sub`, `add`, `fma`, 2 blends, compare):
`138,639/16 = 8,665` vectors x 8 = 69,320 vector-ops/step; at ~2 512-bit ops per
cycle per core over 4 cores = 8,665 cycles/step; at a sustained AVX-512 clock of
~2.5-3.0 GHz that is **~2.9-3.5 us/step**.

```
    measured 54.3 us  /  arithmetic floor ~3 us  =  ~18x
```

The state is 1.66 MB across 4 chunks = 0.42 MB/core, which *fits* in the 1.25 MB
L2 per core — so the gap is not DRAM. It is per-step overhead: the ctypes/FFI
boundary, the spin-barrier rendezvous, and the L2 round-trip that the 12-pass
PyTorch path already showed dominates. **Window fusion amortises every one of
those 18x**, and it is the same code change as the GPU one. Expected direction:
large; magnitude unknown until measured (Rule 2 — this project has been wrong by
3x in both directions on exactly this kind of estimate).

The scan on CPU is worse than useless. Its parallel axis is the *step* index,
which is the one axis that is not in the SIMD lanes; exposing it needs an
18x16 transpose per neuron-block, and the work inflation is 1.9x (Blelloch) to
5.0x (Hillis-Steele) against a kernel that is already ~18x away from being
arithmetic-bound. **Confirming the brief's own suspicion: on 4 cores with 18
steps per block, the scan loses to the sequential loop. It does.**

### 4.4 Neuromorphic

A Loihi 2 neuron core is a sequential state machine time-multiplexing ~90
neurons (138,639 over 12 chips x 128 cores) behind a hardware per-timestep
barrier. There is no intra-neuron step-parallel resource, so a parallel scan is
not merely unprofitable but *inexpressible*.

What does transfer is §2 — and it is the most valuable thing here for
neuromorphic. Sandia's dt = 1 ms configuration rounds both the 1.8 ms delay and
the 2.2 ms refractory to 2 ms (+11% / -9%). The theorem in §2 says the window
can be taken as `dt = D = 1.8 ms` **exactly**, with the delay preserved by
construction and the reset handled by free-run-then-truncate at K = 1 — i.e.
18x fewer hardware barriers with *no* rounding of either time constant. That is
an exactness argument, not a speed argument, and it is the strongest form of the
"uniform delay as the integration timestep" claim this project has.

Note the precondition inverts on other hardware: the theorem needs
`T_ref >= D`. Here `2.2 >= 1.8` with 0.4 ms of margin. If the delay were raised
past 2.2 ms, or the refractory shortened below 1.8 ms, Lemma 1 fails, multiple
spikes per window become possible, and *then* FPT-style iteration or Bullet
Trains speculation genuinely would be needed.

---

## How to implement

**Do not implement the scan.** Implement the fused window. §1 and §2 are what
prove it correct.

### 5.1 New files (nothing existing is modified)

| file | contents |
|---|---|
| `flyloop/native/window_kernel.c` | `sweep_window_chunk()` — 18 sub-steps held in registers |
| `flyloop/window_engine.py` | `WindowBrainEngine`, same API as `NativeBrainEngine` |
| `flyloop/verify_window.py` | the gate: full-state uint32 comparison at every *sub-step*, 8 regimes |
| `flyloop/cuda/window_kernel.cu` | `lif_window_kernel<<<>>>`, one thread per neuron, graph-capturable |
| `code/scan_window.py` | reference scan implementation **for the record only**, so the negative in §4.1 is reproducible rather than asserted |

### 5.2 The fused window body (per neuron; one GPU thread, or one AVX-512 lane)

Mirrors `lif_kernel.c`'s documented operation order exactly — that is the
bit-identity requirement, and the only difference is that `v`, `g`, `refrac`
never leave registers between sub-steps.

```c
/* arrivals for THIS window are the PREVIOUS window's spike list (Corollary D),
   pre-scattered into CSR-by-target: arr_row[i], arr_sub[p], arr_w[p].
   arr_sub[p] in [0,17] is the sub-step, preserved from the emitting sub-step. */

float v = v_state[i], g = g_state[i], rf = refrac_ctr[i];   /* 12 B read */
int   p = arr_row[i], pe = arr_row[i+1];
uint32_t spike_mask = 0;                                    /* <=1 bit set, Lemma 1 */

for (int m = 0; m < 18; ++m) {
    /* 1. refractory counter, matching `refrac = spiked_prev ? 0 : refrac + 1` */
    rf = spiked_prev ? 0.0f : rf + 1.0f;

    /* 2. g decays; arrivals land on the POST-decay g, gated by refractoriness,
          exactly as the caller does today (two roundings, same order)         */
    const float gi = g;                    /* v uses the OLD g */
    float gn = gi * c_decay;
    const float gate = (rf >= refrac_steps_i) ? 1.0f : 0.0f;
    while (p < pe && arr_sub[p] == m) { gn = gn + arr_w[p] * gate; ++p; }

    /* 3. external (Poisson) injection, pre-drawn in a block -- exact, §11.4    */
    v += stim_window[m][i];                /* sparse; 21 neurons */

    /* 4. membrane, ATen's exact op sequence */
    const float t  = (v_rest - v) + gi;
    const float vn = use_fma ? fmaf(t, c_mem, v) : ((t * c_mem) + v);

    /* 5. threshold + reset. Theorem §2: after this fires, (v,g) is bit-exactly
          the fixed point (v_reset, 0) for the rest of the window.              */
    const int sp = (vn > v_th);
    v = sp ? v_reset : vn;
    g = sp ? 0.0f    : gn;
    spiked_prev = sp;
    if (sp) spike_mask |= (1u << m);
}
v_state[i] = v; g_state[i] = g; refrac_ctr[i] = rf;         /* 12 B write */
if (spike_mask) emit(i, spike_mask);   /* one append; feeds the NEXT window */
```

Notes that matter for the gate:

- `-ffp-contract=off` remains mandatory; step 2's `gi * c_decay` must not fuse.
- The `use_fma` / scalar-tail split of FORK §1.3 must be reproduced *inside* the
  fused loop at the same neuron indices, or bit-identity to PyTorch breaks at the
  ATen chunk seams (34,656-34,659, 69,316-69,319, 103,976-103,979,
  138,624-138,638 at 4 threads).
- `emit()` may run in any order: ANALYSIS §1 makes the downstream accumulation
  exact, so a GPU `atomicAdd` scatter is deterministic.
- `spike_mask` has at most one bit set (Lemma 1) — assert it in debug builds; a
  second bit means a precondition broke.

### 5.3 The scan reference (for the negative result only)

`code/scan_window.py::scan_free_prefix(u0, g0, I)` implementing (5)-(6) with
Hillis-Steele over 18, so that §4.1's 1.04x-1.43x can be *measured* on a CUDA
device rather than modelled. Ship it alongside a microbenchmark of
`__shfl_up_sync` throughput, which is the single number §4.1 leans on hardest.

### 5.4 Load-time assertions

All of these must be checked, not assumed. Each one, if violated, silently
invalidates a different theorem above.

| assertion | protects |
|---|---|
| `delay` is a single scalar, `== 18 * dt` | §1 affinity (method of steps), Corollary D |
| `refrac_steps * dt >= delay` (`2.2 >= 1.8`) | Lemma 1, Corollaries A & B |
| `v_reset == v_rest` | Lemma 2 (reset is a fixed point); otherwise carry `P(x^n)(u_reset,0)` |
| no autapses, no zero-delay edges, no gap junctions | §1 affinity |
| `all(w == round(w))` and `max_i sum_j |w_ij| < 2^24` | order-independent `emit()` (ANALYSIS §1) |
| `abs(tau_m/tau_s - 4) < 1e-12` | the one-parameter collapse (4); else use the two-parameter form (3) |
| refractory semantics flag (input discarded vs accumulated) | Lemma 2's third segment |
| closed-loop injection is not per-step | the Poisson block and the window itself |

---

## Failure modes / preconditions

1. **Heterogeneous delays.** The window collapses to `d_min`. If `d_min < 22`
   steps, Lemma 1 fails and multiple spikes per window become possible — *then*
   FPT (K~3) or Bullet Trains speculation is genuinely required, and this note's
   Corollaries A and B are void. Below `d_min = 1` the whole construction is gone.
2. **`T_ref < D`.** Same failure. The 0.4 ms margin here is not large; a model
   revision that shortens refractoriness to 1.5 ms breaks it silently.
3. **`v_reset != v_rest`.** Lemma 2's *bit-exactness* is lost (the state decays
   from `u_reset` instead of sitting at 0). The theorem survives with an extra
   `x^n u_reset` term, but the "post-spike segment is free" claim does not.
4. **Brian 2 refractory semantics.** Input accumulating through refractoriness
   changes the third segment (FORK §1.1). Exactness survives; the code does not.
5. **True conductance coupling** `g (E_rev - v)`. The map stops being affine, `A`
   stops being constant, the monoid stops existing. Everything above dies. This
   is the single largest model risk and it is shared with HANDOFF §5.
6. **Closed-loop / embodied injection.** A caller that injects stimulus computed
   from the *previous step's* output cannot use an 18-step window at all —
   the window is exactly the horizon over which the outside world must not
   intervene. Sensor input must be quantised to window boundaries, or the window
   shortened to the control latency.
7. **Register pressure on GPU.** The fused window keeps `v, g, rf`, the arrival
   cursor, and the stimulus pointer live across 18 iterations. If the compiler
   spills, the 18x traffic reduction partly evaporates. Check
   `-Xptxas -v` register counts and target <= 40 registers/thread to hold 1,536
   threads/SM.
8. **The scan's fp32 bound is a bound, not a measurement.** If anyone ships the
   scan path anyway, the margin certificate of §3.3 must run, and the repair set
   must be logged — a growing repair set is the signal that the `u`-envelope
   assumption (`|u| <= 10 mV`) has been violated by a new stimulus protocol.
9. **`sum |w|` worst case.** The §3.3 worst-case bound (0.24% of `theta`) uses
   `max_i sum_j |w_ij| = 69,948`. If the connectome is rescaled or replaced, both
   that bound and ANALYSIS §1's exactness theorem must be recomputed.

---

## References

- Blelloch, G. *Prefix Sums and Their Applications.* CMU-CS-90-190, 1990.
  https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf
- Ladner, R. & Fischer, M. *Parallel Prefix Computation.* JACM 27(4):831-838, 1980.
  https://dl.acm.org/doi/10.1145/322217.322232
- Snir, M. *Depth-size trade-offs for parallel prefix computation.* Journal of
  Algorithms 7(2):185-201, 1986.
  https://www.sciencedirect.com/science/article/pii/0196677486900037
- Morrill, Pehle & Zador. *Bullet Trains* (affine-map scan for LIF, exact reset
  by speculation). ICML 2026. https://arxiv.org/abs/2603.13283
- Feng et al. *FPT: fixed-point iteration for parallel LIF training*, K~3.
  ICML 2025. https://arxiv.org/abs/2506.12087
- Rotter, S. & Diesmann, M. *Exact digital simulation of time-invariant linear
  systems with applications to neuronal modeling.* Biol. Cybern. 81:381-402, 1999.
  https://link.springer.com/article/10.1007/s004220050570
- Bellman, R. & Cooke, K. *Differential-Difference Equations.* Academic Press, 1963.
  (method of steps)
- Hairer, E., Norsett, S. & Wanner, G. *Solving Ordinary Differential Equations I*,
  ch. II.17 (delay equations, method of steps).
- Higham, N. *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002,
  ch. 4 (summation error, `gamma_n` bounds).
- Wang et al. *Neuromorphic Simulation of the Drosophila Brain on Loihi 2*, 2025.
  https://arxiv.org/abs/2508.16792
- NVIDIA. *Ada GPU Architecture Whitepaper* (RTX 4070: 46 SMs, 5,888 CUDA cores,
  504 GB/s). https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
- NVIDIA. *CUDA C++ Programming Guide*, arithmetic-instruction throughput table
  for compute capability 8.9 (warp shuffle ~32 threads/SM/clk).
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- In-repo: `ANALYSIS.md` §1 (exact order-independent accumulation), §2 (sustained
  cost bound), §3.1 (GeNN F/W decomposition); `HANDOFF.md` §5.1-§5.5, §11;
  `FORK.md` §1.1, §1.3, §11.4; `code/window_step.py` (`KAPPA_WINDOW_MAX = C_18`);
  `flyloop/native/lif_kernel.c` (the operation order the fused body must mirror).
