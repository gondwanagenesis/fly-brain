---
field: Memory layout, bit-packing and lossless connectome compression
date: 2026-07-30
verdict: Delete refrac_steps as a dense N-array (it is binary-valued and already read only for ~190 sparse indices/step, never in the hot sweep) and pack refrac to a SATURATING uint8 — together this shrinks the per-step working set from 2.22 MB to 1.19 MB, crossing from "fits L3 only" to "fits the 1.25 MB/core L2", with zero information loss and no new verification burden beyond a memcmp.
---

# Memory layout, bit-packing and lossless compression for the whole-brain LIF kernel

## Bottom line

Two changes to `flyloop/native/lif_kernel.c` + `native_engine.py` are provably
exact, cost nothing, and are worth landing this week: eliminate `refrac_steps`
as an N-length array (it is binary — 22 or 0 — and the kernel already never
reads it inside the dense sweep, only in the ~190-entry sparse delayed-input
loop) and replace fp32 `refrac` with a **saturating** uint8 counter, which is
observationally identical because the only consumer is a monotone `>=`
comparison. That drops the resident dense state from 2.22 MB to 1.19 MB,
crossing under the 1.25 MB/core L2 for the first time (it currently only fits
the shared 12 MB L3). Quantizing `v` and `g` to fixed-point is a different
risk class entirely — `v`'s range is not the textbook `[0, θ]` once inhibitory
undershoot is admitted, and `g` is measured to be **unbounded above** by any
hard mechanism except decay (raw connectome weights run to ±2405 synapses ×
0.275 mV ≈ ±661 mV, dwarfing the 7 mV threshold), so it needs an empirical
extremum measurement across all eight validated regimes before any fixed
range can be trusted, and even then it is a bounded-error change requiring a
new gate, not a free one. Connectome weight compression to int16 is real and
lossless (verified directly against `data/fanout_csc.pt`: all 15,091,983
values are exact integers in `[-2405, 1897]`) but it will not move the
0.219 ms step time at all — fan-out touches ~190 of 15.1M edges/step, so
compressing the other 99.999% changes disk footprint, not the critical path.
Succinct graph codes (Elias-Fano, WebGraph, k2-trees) should not be built
here: their entire value proposition is amortized decode cost against a
*scanned* graph, and this workload never scans the graph — it touches an
already-known, already-tiny arrival set once per step, so any decode
constant is pure overhead layered directly onto the 0.14% path this codebase
has already identified as the trap.

## Applicable techniques

### 1. Hot/cold field splitting — delete `refrac_steps` as a dense array

**Core idea.** Data-oriented design's standard move: fields with different
access frequency should not share a layout tier, because co-locating a
rarely-touched field with hot fields buys it a free ride into cache lines
that the hot fields don't need shared ([Data-oriented design, Wikipedia](https://en.wikipedia.org/wiki/Data-oriented_design); [AoS and SoA, Wikipedia](https://en.wikipedia.org/wiki/AoS_and_SoA)).
Here it's stronger than the usual case: `refrac_steps` in this codebase is not
merely cold, it is **provably binary and provably unread in the hot loop**.
Grep confirms every use is a comparison against `refrac`:
`brain_engine.py:197` (`torch.ge(self.refrac, self.refrac_steps, out=gb)`),
`:272`, `:419`; and in the C kernel `refrac_steps` is a parameter of
`sweep_scalar`/`sweep_avx512` that the function body never touches — it's
only read in `lif_step`'s sparse delayed-input loop (`lif_kernel.c:269`,
`n_del ≈ 190` iterations/step). The array holds exactly two values: the
refractory-step count (22, for `2.2 ms / 0.1 ms`) for every ordinary neuron,
and 0 for the handful of stimulation targets (`set_stim_neurons` zeroes it for
~21 neurons in the sugar protocol). A 554,556-byte dense fp32 array is pure
waste for one bit of information per neuron that only ~190 sparse reads/step
ever consult.

**Key change.** Replace `refrac_steps[i]` with a direct check against a tiny
"always-excitable" index list (the same `stim_idx` array the engine already
carries — no new data structure needed): in the sparse gate loop, `gate =
(refrac[i] >= 22) || is_stim[i]`, where `is_stim` is either the existing
`stim_idx` array (≤ a few hundred entries, checked once at setup by writing a
sentinel into `refrac`, e.g. pre-biasing stim neurons' `refrac` so the plain
`>= 22` test already returns true) or, if a runtime check is wanted, the
existing `nw`-word bitset infrastructure (17,330 B) reused for a second
purpose. Either way the N-length `refrac_steps` array disappears.

**Arithmetic.** Saves `138,639 × 4 B = 554,556 B` exactly, unconditionally —
no precision argument needed because no precision existed to lose (the field
was already exact-representable as 1 bit).

**Fidelity verdict: exact.** Removes a field, changes no arithmetic that
executes, requires no new tolerance in `verify_native.py`.

### 2. Saturating uint8 refractory counter

**Core idea.** `refrac` evolves as `r ← 0` on self-spike, else `r ← r + 1`,
every step, for as long as a neuron stays quiescent — which for the 91%
provably-inert population (HANDOFF §2.2) can be the entire run. In fp32 this
just grows monotonically (24-bit mantissa holds any integer up to 16.7M
exactly, so no rounding ever occurs regardless of run length). The only
consumer of `refrac` anywhere in the codebase is the boolean gate `refrac >=
refrac_steps` (constant 22) — confirmed by grep, `refrac` is never logged,
returned, or used as a "time since last spike" quantity in `brain_engine.py`
or `native_engine.py`.

**Key argument (why saturating, and why that's exact, not approximate).**
A plain `uint8` with **wraparound** would break correctness: at `r = 255`,
`r + 1 = 0 < 22`, which flips the gate back to false — a real behavior change
(a neuron the fp32 version treats as long-since-excitable would spuriously
re-enter its own refractory window). A **saturating** add (`r ← min(r + 1,
255)`) does not have this failure mode, and the reason is a one-line
monotonicity proof: the comparison `r >= 22` partitions the reachable domain
into `{0..21} → false` and `{22..255} → false-to-true, forever-true}`. Once a
saturating counter enters `{22..255}` it can only move within that set
(saturating add never decreases, never wraps below 255), so it is a *no-op on
the observable predicate* forever after — bit-for-bit the same forever-true
behavior the unbounded fp32 counter exhibits. The two representations are
observationally indistinguishable for every possible trajectory: both start
at 0, both reset to 0 on the same spike event, both cross into "gate = true"
at the same step (the 22nd consecutive non-spike step), and both stay there
until the next reset. This is not "usually works" — it is exact because the
only operation ever performed on the value is a monotone comparison against a
fixed constant that is reachable well below the saturation ceiling (22 ≪
255), so saturation past the decision boundary is unobservable by
construction. The one hazard: if any future code ever wants `refrac` as an
actual elapsed-time value (e.g., telemetry, "ms since this neuron last
fired"), a saturating uint8 silently discards that information above 25.5 ms
— fine for the gate, wrong for telemetry. Recommend a code comment pinning
this invariant so nobody repurposes the field later.

**Arithmetic.** `138,639 × 4 B → 138,639 × 1 B` saves 415,917 B.

**Fidelity verdict: exact** (provable by the monotonicity argument above, not
merely bounded-error), conditional on `refrac` never being read for anything
but the `>= 22` comparison — true today, must stay true.

### 3. Working-set arithmetic: SoA fits L3 today, L2 after (1) + (2)

| State | Bytes | vs 1.25 MB/core L2 | vs 12 MB shared L3 |
|---|---|---|---|
| Current: v,g,refrac,refrac_steps (fp32×4) | 2,218,224 B (2.22 MB) | **does not fit** (1.78×) | fits (5.4× headroom) |
| − `refrac_steps` (§1) | 1,663,668 B (1.66 MB) | does not fit (1.33×) | fits |
| − `refrac_steps`, `refrac`→uint8 (§1+§2) | 1,247,751 B (1.25 MB) | **fits** (Intel's own spec is 1,310,720 B = 1280 KiB, so ~63 KB headroom; even the decimal-MB reading of "1.25 MB" leaves ~2 KB) | fits, 9.6× headroom |
| + spike bitset ×2 (current + scratch, 17,330 B each) | +34,660 B | still fits (with the 1280 KiB reading) | fits |

This crosses a real tier boundary: L2 latency on Tiger Lake/Willow Cove is on
the order of 12–14 cycles vs. L3's 40+ cycles. Whether this actually shows up
in the 0.219 ms/step number is **unmeasured** — the access pattern in
`sweep_avx512` is a pure linear stream, which hardware prefetchers are good
at hiding regardless of which cache tier ultimately serves it, so the honest
claim is "plausible, cheap, not yet benchmarked," per this repo's own Rule 2
(measure, don't model). It composes with the four-way static core
partitioning in `research/hpc-simd-threading.md`: **that** plan already gets
each thread down to `N/4 × 16 B ≈ 554 KB` even at current fp32 width, which
already fits L2 per thread on its own — so this packing's marginal benefit is
largest for **today's single-core number** and for leaving more L3 headroom
for the fan-out scratch and delay ring once threads share the 12 MB pool, not
for the multicore case where the cache-fit problem is mostly solved by
partitioning alone. Land it anyway: it is free, it helps the number you can
measure right now, and it does not fight the threading plan.

### 4. Minimum bits that provably preserve every threshold crossing

Given `Δ_min = 5 µV` (the smallest real per-step increment measured this
session) and `θ = 7 mV`, a fixed-point format with resolution `r` never
silently drops an update only if `r` is small enough that round-to-nearest
cannot annihilate the smallest real increment. The clean sufficient condition
is `r ≤ Δ_min` (equality risks exact-tie edge cases; `r ≤ Δ_min/2` removes
even that): `r ≤ 2.5 µV ⇒ F ≥ log2(1 mV / 2.5 µV) = 8.64 ⇒ F = 9` fractional
bits is the **theoretical minimum**, giving `r = 1.953 µV` (2× margin). The
number this repo's prior session settled on, Q3.13 (`F = 13`, `r = 0.122
µV`), is not the minimum — it's the minimum plus a ~40× safety margin, which
is the right call given the hard "no information sacrificed" bar, but should
be labeled as a safety-margined choice, not a tight bound. **The gap the
prior note left open:** Q3.13 assumed range `[0, 8]` mV, i.e. `M = 3` unsigned
integer bits and *no undershoot*. But `dv/dt = (g − (v − v_rest))/tau_mem`
with signed `g` (inhibitory synapses are negative weights in
`fanout_csc.pt` — measured `val.min() = −2405`) can pull `u = v − v_rest`
**below zero**, and nothing in the model clips it there; only decay pulls it
back toward rest. A range assumption that only covers `[0, 8]` mV will
silently saturate on any real inhibitory transient outside that guess — a
hard violation of the "no information sacrificed" constraint the moment it
happens once. **Concrete fix:** widen to a *signed* Q-format with enough
integer headroom for the real undershoot, e.g. Q4.10 (`1` sign bit, `M = 4`
integer bits ⇒ range `±16 mV`, `F = 10` ⇒ `r = 0.977 µV`, still ~5× margin
under `Δ_min`), which fits 16 bits total — but the `±16 mV` envelope is a
guess, not a measurement, and must be replaced with the actual observed
extremum before this is trustworthy (see next action #3).

**Fidelity verdict: bounded-error by construction, contingent on an unmeasured
range.** This is explicitly *not* the same class of claim as §1/§2 above —
it requires (a) an empirical range measurement this session did not perform,
(b) a new tolerance-based entry in the verification gate (not a bit-exact
memcmp), and (c) an explicit error-propagation bound through the one
non-representable operation in the pipeline, the `g *= c_decay` multiply,
which is the only place fixed-point rounding is introduced (additions of two
already-representable fixed-point values are exact; the decay multiply is
not, generically).

### 5. Spike bitset + tzcnt/popcount iteration (already correctly implemented)

**Core idea.** Represent the N-neuron spike vector as a 64-bit-word bitset
(`(N+63)/64 = 2166` words, 17,330 B) instead of a float array, and recover
the sparse index list by repeatedly clearing the lowest set bit
(`b &= b - 1`) and reading its position with a hardware count-trailing-zeros
instruction — the classic bit-manipulation idiom (Kernighan's lowest-set-bit
trick; hardware `TZCNT`/`BSF`, documented in Warren, *Hacker's Delight*, 2nd
ed., Addison-Wesley 2012, and exposed as `__builtin_ctzll`/BMI1 `TZCNT` in
current toolchains). `lif_kernel.c:276-287` already does exactly this, and
gets the bitset itself "for free" as a side effect of the AVX-512 `vcmpps`
mask register (`_mm512_cmp_ps_mask`) rather than as a separate reduction
pass — the best-case realization of this pattern.

**Arithmetic.** No change recommended; this is already correct and already
near-optimal. One micro-inefficiency exists (the `nw = 2166`-word outer scan
in `lif_step` runs every step regardless of whether any word is nonzero,
costing on the order of 2166 predictable branches ≈ 0.7 µs/step, ~0.3% of the
219 µs budget) — a hierarchical "any-set" summary bitmap (one bit per 64
words) could skip that, but at 0.3% of budget it is not worth the complexity
or the risk of a new bug on the hot path.

**Fidelity verdict: exact**, already implemented, no action needed beyond
optionally the summary-bitmap micro-optimization if every last percent is
being chased later.

### 6. Lossless connectome weight compression (int16, verified against real data)

**Core idea.** `w = 0.275 mV × integer synapse count` — verified directly
against this repo's own `data/fanout_csc.pt` (`.venv/Scripts/python.exe`,
this session): `val` is `float32`, length 15,091,983, and **every single
value is an exact integer**, range `[−2405, 1897]`. That fits `int16`
(`[−32768, 32767]`) with a >13× margin, so storing raw synapse counts as
`int16` and applying `w_scale` at the point of use (already how
`lif_fanout` applies `w_scale` — after accumulation, not before) is a
zero-loss cast, not an approximation.

**Arithmetic (measured, not estimated).**

| Array | Current | dtype-narrowed | Saved |
|---|---|---|---|
| `val` (edge weight) | `float32`, 60,367,932 B | `int16`, 30,183,966 B | 30,183,966 B |
| `crow` (CSC row ptr, len 138,640) | `int64`, 1,109,120 B | `int32` (max value 15,091,983 ≪ 2³¹) fits, 554,560 B | 554,560 B |
| `post` (postsynaptic index, max 138,638) | `int32`, 60,367,932 B | already minimal (needs ≥18 bits; `int32` is the natural width, no narrower standard type applies) | 0 |
| **Total** | **121,844,984 B (121.8 MB)** | **91,106,458 B (91.1 MB)** | **30,738,526 B (25.2%)** |

**Why it doesn't move the 0.219 ms number.** Fan-out touches `crow[j]:crow[j+1]`
only for the ~1.75 neurons/step that actually spike (sugar protocol), i.e.
~190 of 15,091,983 edges/step — 0.0013% of the array, matching the
independently-measured 728:1 dense:sparse op ratio. The bytes actually
streamed per step from the connectome are on the order of `190 × (4 B post +
4 B val) ≈ 1.5 KB`, regardless of whether the backing array is 122 MB or
91 MB — the other 99.999% of the array sits untouched in RAM either way, and
32 GB of RAM makes even the uncompressed 122 MB immaterial for capacity. The
saving is real for one-time load time and disk footprint, not for the
per-step critical path.

**Fidelity verdict: exact / lossless**, verified against the actual dataset
in this repo (not assumed from the model description).

## Ruled out from this field, and why

| Technique | Why not, specifically for this codebase |
|---|---|
| **Elias-Fano encoding** ([Ottaviano & Venturini, *Partitioned Elias-Fano Indexes*, SIGIR 2014](http://groups.di.unipi.it/~ottavian/files/elias_fano_sigir14.pdf); survey: [Besta & Hoefler, *Survey and Taxonomy of Lossless Graph Compression*, arXiv:1806.01799](https://arxiv.org/pdf/1806.01799)) | Designed to compress *monotone sorted sequences* (e.g. sorted adjacency lists) with O(1) random access at a small constant-factor decode cost per lookup. That constant, however small, lands directly on the 0.14%-of-runtime fan-out path this codebase has already flagged as the trap ("Synapse pruning / faster SpMV: attacks the 0.14%," HANDOFF §6) — for a workload that touches ~190 edges/step out of 15.1M, shaving bytes off the untouched 99.999% cannot show up in wall-clock, and decoding the touched 0.001% adds pure overhead. |
| **WebGraph / BV compression** ([Boldi & Vigna, *The WebGraph Framework I: Compression Techniques*, WWW 2004](https://dl.acm.org/doi/10.1145/988672.988752)) | Its compression ratio depends on **lexicographic URL locality and link similarity between nearby-numbered nodes** — properties of web graphs, not of a synaptic connectome numbered by opaque FlyWire segment ID. This repo already tested and rejected the locality precondition these techniques need: "Graph reordering (RCM/METIS): connectomes are hub-dominated scale-free; bandwidth reduction is not the bottleneck" (HANDOFF §6). Without that locality, BV-style delta/interval/reference coding degrades toward raw storage while still paying its decode complexity. |
| **k2-trees** ([Brisaboa, Ladra & Navarro, *k2-Trees for Compact Web Graph Representation*, SPIRE 2009](https://link.springer.com/chapter/10.1007/978-3-642-03784-9_3)) | Same locality precondition as WebGraph (exploits large empty/clustered blocks of the adjacency matrix), same 0.14%-path decode-cost problem, and additionally requires a *tree descent per query* rather than O(1) array indexing — strictly worse for this access pattern than either fp32 CSR or the int16-narrowed CSR in §6. |
| **AoSoA tiling of v/g/refrac** ([AoS and SoA, Wikipedia](https://en.wikipedia.org/wiki/AoS_and_SoA); worked example: [Noormofidi, *Simulating Nonlinear Neutrino Oscillations on Next-Generation Many-Core Architectures*, arXiv:1907.05560](https://arxiv.org/pdf/1907.05560)) | AoSoA earns its keep when a per-record access pattern is irregular (gather/scatter) or when a vector width doesn't divide a natural record size, forcing padding under pure SoA. Neither applies here: `sweep_avx512` processes 16 fp32 lanes per sub-iteration, and `16 × 4 B = 64 B` **is exactly one cache line** — plain SoA already gives one clean cache-line fetch per array per 16-neuron block, which is the AoSoA ideal outcome without AoSoA's added address-computation complexity. Interleaving fields into an AoSoA tile would in fact *hurt* here, by pulling the cold `refrac_steps` field into the same cache lines as the hot `v`/`g`/`refrac` fields every block — the opposite of the hot/cold split in §1. |
| **Quantizing `g` to any fixed-point width** | Ruled out for now, not forever: `g` has no reset floor/ceiling other than the neuron's own spike (`g = sp ? 0 : g·c_decay`), and the connectome's raw edge weights run to ±2405 synapses × 0.275 mV ≈ ±661 mV — two orders of magnitude past the 7 mV threshold that bounds `v`. A quiet neuron could in principle receive a burst of such inputs under the "tonic/broad drive" regime this repo has already flagged as the top kill-risk (HANDOFF §7.1). Fixing any finite range for `g` without first measuring its realized extrema across all eight validated regimes risks silent saturation — a genuine information-loss failure mode, not a rounding one. See next action #3. |
| **Bit-packing `post` to 18 bits** (N = 138,639 needs `⌈log2 N⌉ = 18` bits, `int32` wastes 14) | Technically lossless and would save another ~26.4 MB (`15,091,983 × 14 bits / 8 ≈ 26.4 MB`), but requires per-edge bit-unpacking (shift + mask) on every fan-out touch — again landing decode cost squarely on the 0.14% path for a saving that (per §6's arithmetic) does not affect per-step bandwidth at all, only static footprint that RAM capacity already makes irrelevant. Not worth the code complexity or the new correctness surface. |
| **Shrinking `_acc`/`_touched` fan-out scratch below N-length** (`native_engine.py:170-172`, `1,126,742 B` combined, currently dense-`N`-sized despite ~190 sparse touches/step) | Looked at and *not* recommended, on reflection: because access is index-by-neuron-ID (`acc[p] += val[k]`) and only the touched cache lines are ever pulled into any cache tier, the array's *nominal* size costs nothing extra per step under a streaming/scattered access model — no O(N) clear happens (the kernel comment confirms it self-zeros only touched entries) and TLB pressure from ~136 backing pages is far below Tiger Lake's L2 STLB capacity. Shrinking it would require a neuron-ID→slot indirection, i.e. attacking the 0.14% path for a benefit that doesn't show up in either bandwidth or TLB terms. Flagged and dismissed in the same session so nobody re-derives it. |

## Concrete next actions for this codebase

1. **`flyloop/native/lif_kernel.c` + `native_engine.py`: delete `refrac_steps`
   as an N-array.** Replace with the existing `stim_idx` list (pre-bias those
   neurons' `refrac` so the plain `>= 22` test already returns true at setup,
   or gate the sparse delayed-input loop with an `is_stim` check on the ~190
   touched indices only). Saves 554,556 B, exact, no new tolerance needed in
   `verify_native.py` — a plain `memcmp` against the existing baseline still
   applies.
2. **`native_engine.py:147-148`, `lif_kernel.c` sweep signatures: change
   `refrac` from `float32` to `uint8` with saturating increment**
   (`r = (r < 255) ? r + 1 : 255`, or the branchless `_mm512_adds_epu8`-style
   saturating add if vectorized). Saves 415,917 B. Exact per the monotonicity
   argument in §2 — add a one-line comment at the field declaration
   documenting the "gate-only, never telemetry" invariant so it can't be
   silently repurposed later.
3. **Before touching `v` or `g`: instrument `flyloop/verify.py`'s eight
   validated regimes to log `v.min()/v.max()/g.min()/g.max()` every step**
   and report the true observed envelope, including the broad-stimulation and
   whole-brain regimes explicitly called out as the risk case in HANDOFF §7.
   This is the prerequisite for §4's fixed-point proposal to be anything more
   than a guess — do this before writing a single line of quantization code
   for `v`/`g`.
4. **`data/fanout_csc.pt` load path (wherever it's regenerated, likely
   upstream of `native_engine.py`'s `torch.load`): narrow `val` to `int16`
   and `crow` to `int32`.** Verified lossless against the live dataset this
   session (§6 table). Do it for the 25% footprint/load-time win; do not
   expect it to move `0.219 ms/step`, and say so in the commit message so a
   future reader doesn't re-measure and get confused by a null result.
5. **After 1–2 land, rerun the existing single-core timing harness** (whatever
   currently produces the "0.219 ms/step" figure in HANDOFF §2, presumably
   near `flyloop/verify_native.py` or a dedicated bench script) to get a real
   measurement of whether crossing the L2 boundary actually moves the number,
   rather than asserting it does. Report the result — positive or null —
   back into `HANDOFF.md` §2 either way, per this repo's Rule 2.

## References

- [Data-oriented design — Wikipedia](https://en.wikipedia.org/wiki/Data-oriented_design) — hot/cold field-splitting rationale used in §1.
- [AoS and SoA — Wikipedia](https://en.wikipedia.org/wiki/AoS_and_SoA) — SoA vs AoSoA tradeoff, used to rule out AoSoA in the ruled-out table.
- [Noormofidi, *Simulating Nonlinear Neutrino Oscillations on Next-Generation Many-Core Architectures*, arXiv:1907.05560](https://arxiv.org/pdf/1907.05560) — worked AoSoA example from HPC particle simulation, cited as the general-case comparator.
- [Ottaviano & Venturini, *Partitioned Elias-Fano Indexes*, SIGIR 2014](http://groups.di.unipi.it/~ottavian/files/elias_fano_sigir14.pdf) — Elias-Fano algorithm and its O(1)-random-access decode-cost tradeoff.
- [Besta & Hoefler, *Survey and Taxonomy of Lossless Graph Compression and Space-Efficient Graph Representations*, arXiv:1806.01799](https://arxiv.org/pdf/1806.01799) — general survey covering Elias-Fano and successors in the graph-compression context.
- [Boldi & Vigna, *The WebGraph Framework I: Compression Techniques*, WWW 2004 / ACM DL](https://dl.acm.org/doi/10.1145/988672.988752) — locality-dependent web-graph compression, ruled out here for lack of locality precondition.
- [Brisaboa, Ladra & Navarro, *k2-Trees for Compact Web Graph Representation*, SPIRE 2009 / Springer](https://link.springer.com/chapter/10.1007/978-3-642-03784-9_3) — k2-tree structure, same locality precondition, ruled out.
- [Intel Core i7-1185G7 Processor specifications — Intel ARK](https://www.intel.com/content/www/us/en/products/sku/208664/intel-core-i71185g7-processor-12m-cache-up-to-4-80-ghz-with-ipu/specifications.html) — confirms 12 MB shared L3, 1.25 MB (1280 KiB) per-core L2 used throughout the working-set arithmetic in §3.
- `C:\Users\neogo\Documents\FlyBrain\data\fanout_csc.pt` — this session's direct measurement of `val`/`crow`/`post` dtypes and value ranges (§6), via `.venv\Scripts\python.exe`; not from documentation, from the actual bytes on this machine.
- `C:\Users\neogo\Documents\FlyBrain\flyloop\native\lif_kernel.c`, `native_engine.py`, `brain_engine.py` — this session's read of the current implementation; all "already implemented" and "grep confirms" claims in §1, §2, §5 are grounded in these files as they stand on branch `perf/event-driven-pytorch` as of 2026-07-30.
- `C:\Users\neogo\Documents\FlyBrain\research\hpc-simd-threading.md` — sibling research note this session; cross-referenced in §3 for how per-thread working-set size interacts with the four-way static partitioning plan.
- Warren, H. S., *Hacker's Delight*, 2nd ed., Addison-Wesley, 2012 — canonical reference for the lowest-set-bit-clear (`b &= b-1`) and count-trailing-zeros bitset-iteration idiom already implemented in `lif_kernel.c` (§5); no single web URL, cited as the standard print reference.

## Reviewer notes

**Citation verification (2026-07-30 review):**
- **CORRECTED**: arXiv:1907.05560 was incorrectly attributed to "Zoni et al."; the sole author is **Vahid Noormofidi**. Changed both references (§3 worked example and ruled-out table) to correct attribution. Paper title, arXiv ID, and content claims verified correct.
- **VERIFIED**: Ottaviano & Venturini, *Partitioned Elias-Fano Indexes*, SIGIR 2014 (ACM SIGIR conf, May 2014) ✓
- **VERIFIED**: Besta & Hoefler, *Survey and Taxonomy of Lossless Graph Compression*, arXiv:1806.01799 ✓
- **VERIFIED**: Boldi & Vigna, *The WebGraph Framework I: Compression Techniques*, WWW 2004 ✓
- **VERIFIED**: Brisaboa, Ladra & Navarro, *k2-Trees for Compact Web Graph Representation*, SPIRE 2009 ✓

**Arithmetic verification (all checks passed):**
- Working-set table (§3): all byte calculations exact (2.22 MB → 1.66 MB → 1.25 MB transitions verified)
- Cache ratios: L2 fit margin 1.78× → 1.33× → 0.95× all correct
- Connectome compression (§6): total saves 30,738,526 B (25.2% of 121.8 MB) verified across val/crow/post arrays
- Fixed-point resolution (§4): Q3.13 F=13 gives r=0.122 µV with ~41× safety margin vs. Δ_min=5 µV (note claims ~40×) ✓
- Q4.10 alternative: r=0.977 µV gives ~5.1× margin (note says ~5×) ✓
- Fan-out bandwidth: 190 edges × 8 B = 1,520 B ≈ 1.5 KB ✓
- Bitset size: 138,639 neurons ⇒ 17,330 B (ceil(138,639/8)) ✓

**No fabricated citations found.** All TOP-5 papers exist, are correctly titled, and the technical claims about them are accurate. The single error was author misattribution (Zoni→Noormofidi), now corrected.
