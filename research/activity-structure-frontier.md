---
field: Activity structure and frontier processing (graph BFS / SNN sparse-update literature)
date: 2026-07-30
verdict: The existing 8%/0.4% dense-fallback is a one-way latch (never resets) — any transient burst permanently sacrifices the sparse path for the rest of the run; make it bidirectional with Beamer-style hysteresis, and separately switch the *delivery* fallback on Σ out-degree of spiking neurons (edge work) instead of spike count, because the connectome is hub-dominated and a single high-degree spike can blow the fan-out budget while looking cheap under a count-only threshold.
---
# Activity structure and frontier processing, applied to the active-set LIF kernel

## Bottom line

This field's central idea — decide your *representation* (sparse index vs.
dense array/bitmap) per step from a *cost estimate*, not a fixed rule of
thumb — is already half-built in `brain_engine.py`: the code already has a
push-style sparse path (`step_active`) and a pull-style dense fallback
(`step_inplace`), switched on a hand-tuned threshold, which is exactly
Beamer's 2012 direction-optimizing BFS pattern. What the literature adds is
three things this repo does not yet have: (1) the switch criterion should be
**work** (Σ degree of the active/spiking set), not raw count, because this
connectome is hub-dominated scale-free; (2) the switch must be **bidirectional
with hysteresis** — the current `_dense_fallback` flag is set but never
cleared, which is a real, checkable design gap, not a hypothetical one; and
(3) hierarchical bitset frontier representations (from GPU graph frameworks
and one SNN paper) are worth pricing out explicitly — and the arithmetic says
they **do not pay off** at this workload's measured activity density, so this
is a documented rejection, not a gap. Nothing here is a new algorithm; it is
three engineering changes to `brain_engine.py`, all provably bit-identical
because none of them touch the update math, only which code path runs it.

## Applicable techniques

### 1. Direction-optimizing switch criterion, made work-weighted
Beamer, S., Asanović, K., Patterson, D. (2012). "Direction-Optimizing
Breadth-First Search." *SC12*. DOI
[10.1109/SC.2012.50](https://dl.acm.org/doi/10.1109/SC.2012.50), PDF:
https://parlab.eecs.berkeley.edu/sites/all/parlab/files/main.pdf · companion
threshold precedent: Shun, J., Blelloch, G.E. (2013). "Ligra: A Lightweight
Graph Processing Framework for Shared Memory." *PPoPP 2013*.
https://www.cs.cmu.edu/~guyb/papers/SB13.pdf

**Core idea.** Beamer's hybrid BFS switches between a "push" traversal (walk
the frontier, examine its edges) and a "pull" traversal (walk all unvisited
vertices, check for a frontier neighbour) using a cost *estimate*, not a
vertex-count threshold: `m_f` = edges to check from the frontier (sum of
out-degree over frontier vertices), `m_u` = edges to check from unvisited
vertices (upper bound on pull-side work). Switch push→pull when `m_f/α > m_u`
(tuned `α=14`); switch back when frontier size `n_f < n/β` (`β=24`). Ligra
independently converged on the same principle with a single fixed edge
threshold: switch representations when Σ degree(frontier) exceeds `|E|/20`.
Both frameworks reject a pure vertex-count threshold because in scale-free
graphs vertex count and edge-examination work diverge by orders of magnitude.

**Where this maps onto this codebase, precisely.** There are *two* different
switches in `brain_engine.py`, and the literature applies to only one of
them:
- `dense_switch = 0.08` (line 252) gates the **neuron-update** kernel
  (`step_active` vs `step_inplace`). Per-neuron update cost is O(1) —
  independent of connectome degree (§2.1 of ROADMAP.md: dense update is
  728:1 vs synaptic work). A vertex-count threshold is *already the right
  resource metric* here; Beamer's edge-weighting does not apply to this
  switch (see Ruled-out §2).
- `spike_switch = 0.004` (line 253) gates the **delivery fan-out** path
  (lines 300–323: `src = self._spike_idx`, fan-out over `fo_crow`/`fo_post`,
  `tot = cnt.sum()` = actual edges to process this step). This *is* an
  edge-count-dominated cost, and it is checked using `self._spike_idx.numel()`
  — a pure vertex (spike) count — **before** `tot` is computed. This is
  exactly Beamer's `m_f` vs. frontier-count distinction: the code checks
  frontier size when it should check frontier *work*.

**Key relation, adapted.** Replace
`self._spike_idx.numel() > self.spike_switch * N`
with
`self.outdeg[self._spike_idx].sum() > self.edge_switch * E`,
where `self.outdeg = fo_crow[1:] - fo_crow[:-1]` (one-time O(N) precompute,
already implicit in the existing `fo_crow` CSR row pointers — no new data
structure) and `E = 15,091,983`. Cost of the check itself: one gather + sum
over `_spike_idx`, same asymptotic cost as the `.numel()` it replaces (both
O(|spiking|), typically ≤ a few hundred at the measured 1.75 spikes/step).

**Expected effect, with arithmetic.** Mean out-degree in this connectome is
`15,091,983 / 138,639 ≈ 108.9`. A count-only check passes a single spiking
neuron through the O(active) fan-out branch unconditionally, regardless of
its degree, because `1 ≤ 0.004×138,639 ≈ 554` by a wide margin. If that one
neuron is a high-degree hub — plausible, since ROADMAP.md's own "ruled out"
table (§6) already asserts this connectome is "hub-dominated scale-free" —
`tot` (the actual number of fan-out edges processed in the O(active) branch,
line 306) can be far above the mean while `_spike_idx.numel()==1` sails
through the guard by ~500×. This is not a fidelity bug (output is unaffected)
but it *is* a hole in the "a fully-active network must never be slower than
before" guarantee the code claims for itself (comment, line 251). The actual
maximum out-degree in this connectome has not been measured in this
session — that is the first thing to check before tuning `edge_switch`
(Rule 2, ROADMAP.md: "measure, don't model").

**Fidelity verdict: exact / bit-identical.** This changes only which code
path executes, never the arithmetic performed once a path is chosen. Must
still pass `flyloop/verify.py` before landing, per repo convention.

### 2. Bidirectional switching with hysteresis (fixing the one-way latch)
Same source as above (Beamer et al. 2012) — the α/β *two*-threshold design
is the load-bearing idea here, not just the edge-weighting.

**Core idea.** Beamer's hybrid explicitly switches back to push mode once
the frontier shrinks below `n/β`, using a *different, lower* threshold than
the one that triggered the switch to pull mode (`m_f/α > m_u`). This
asymmetry (hysteresis) is what prevents oscillation between representations
on graphs whose frontier hovers near the switch point, while still allowing
the algorithm to return to the cheap representation once the expensive phase
passes. Ligra and GraphIt (Zhang, Y. et al. (2018). "GraphIt: A
High-Performance DSL for Graph Analytics." arXiv:
[1805.00923](https://arxiv.org/abs/1805.00923)) both retain this
bidirectionality — none of the frontier literature treats "dense" as a
one-way absorbing state, because in general graph workloads (PageRank,
connected components, iterative traversal) frontier size legitimately shrinks
back down after a peak.

**What the current code actually does.** `_dense_fallback` (line 258) is set
to `True` at two call sites (lines 297, 321) and is read at line 382, but
**there is no code path that ever sets it back to `False`.** The comment at
lines 255–257 defends this explicitly: *"the network tends to stay [broadly
active] ... so the fallback latches rather than flip-flopping."* That
argument holds for a monotone BFS frontier (which, once large on a
low-diameter graph, stays large by construction) but does **not** hold for
an LIF simulation under a realistic stimulus protocol: ROADMAP.md's own risk
ranking (§7, items 1–2) names "tonic/background drive to all neurons" and
"high-firing protocols" as the top two things that would invalidate the
sparsity story — and an embodied model with transient bursts (a startle, a
brief loud tone, a saccade) riding on an otherwise-quiescent baseline is
exactly the regime where a one-shot latch is worst: the *entire remainder*
of the simulation pays dense-kernel cost after a single brief burst, rather
than paying it only while the burst is active. That is not graceful
degradation; it is a permanent cliff triggered by a transient event.

**Key relation, adapted.** Add `self.recover_switch = 0.5 * self.dense_switch`
(hysteresis band, tune empirically — do not assume the value transfers).
Reuse the existing `prune_every` cadence (already a periodic checkpoint,
lines 245, 362–365) to test, every 64 steps while in dense mode,
`active.sum() < recover_switch * N`; if true, rebuild `_idx` via
`active.nonzero(as_tuple=True)[0]` (identical logic already used in
`_init_active`, line 243 — the dense kernel already maintains the full
`active` boolean array every step, so this reconstruction is exact, not an
approximation) and clear `_dense_fallback`.

**Expected effect, with arithmetic.** ROADMAP.md §2.10 already measured the
achievable ratio at low activity on this exact connectome: 26.4× at 0.88%
active, 5.1× at 5% active, both via the window path, but the same relative
gap exists between the flat 0.219 ms/step (native, dense, fixed regardless
of activity) and whatever `step_active` costs at low `na` — which is *why*
the code chooses sparse mode there in the first place. Currently, any run
that ever transiently crosses 8% active pays the dense rate for 100% of the
remaining simulated time, however quiet it becomes; with hysteresis it pays
the dense rate only for the fraction of time actually above
`recover_switch`. The multiplier recovered is exactly however much of a
protocol's duration is quiet after its burst — for a plausible embodied
protocol (brief salient event, long quiescent baseline) this could dominate
total wall-clock more than any other single change in this document.

**Fidelity verdict: exact / bit-identical.** No update math changes; only
representation. State reconstruction from the dense array is exact by
construction (dense mode never drops information). Must pass
`flyloop/verify.py` / `verify_native.py` before landing — this doc proposes
a `verify.py` regime addition (burst-then-quiet protocol) specifically to
exercise the new re-arm path (see Concrete next actions, item 4).

### 3. Hierarchical bitset activity tracking — evaluated, not adopted
Two-level bitmap precedent: search results point to "Bit-GraphBLAS: Bit-Level
Optimizations of Matrix-Centric Graph Processing on GPU," arXiv:
[2201.08560](https://arxiv.org/pdf/2201.08560) (two-level Bit-Block CSR
container) and Roaring Bitmaps (container-based hierarchical bitmap,
https://roaringbitmap.org/, "Roaring Bitmaps: Implementation of an Optimized
Software Library," https://luca.ntop.org/roaring.pdf). Closest neuro-specific
analog: Bautembach, D., Oikonomidis, I., Argyros, A. (2021). "Even Faster SNN
Simulation with Lazy+Event-driven Plasticity and Shared Atomics." *HPEC
2021*. arXiv: [2107.04092](https://arxiv.org/abs/2107.04092) — stores each
neuron's recent firing history as a 64-bit bitfield and enumerates set bits
with the `__clz()` integer intrinsic to skip inactive steps in a synapse's
own update history.

**Core idea (as it would map here).** Partition the `N = 138,639` neurons
into `T = ⌈N/64⌉ = 2,167` tiles of 64. Maintain one 64-bit occupancy word per
tile (2,167 × 8 B ≈ 17 KB, fits comfortably in L1). Before touching a tile's
neuron state, test `tile_bits[t] != 0` (one comparison, or a batched
`popcount` across all tiles at once) and skip the whole tile with zero cost
if empty — replacing the current `torch.searchsorted`-based gather/scatter
(binary search, O(log(active count)) per lookup, random-access memory
pattern) with a cheap membership test plus contiguous SIMD access for tiles
that are occupied.

**Arithmetic against this workload's measured density.** Sugar-protocol
activity is 1.75 spikes/step (ROADMAP.md §2.1) spread, with no evidence of
ID-locality, across 2,167 tiles — mean occupancy is `1.75/2,167 ≈ 0.0008`
active neurons per tile. Touching a full 64-wide tile to service on average
< 0.001 truly-active neurons wastes ~64× the necessary element work per
truly active neuron; this is *worse* than the current O(active-count) sorted
gather/scatter, not better, at the density this workload actually exhibits.
The technique only starts to pay for itself once per-tile occupancy climbs
high enough that touching all 64 elements is cheaper than the equivalent
number of separate binary-searched gathers — which, by rough proportion,
lands close to the *existing* `dense_switch = 0.08` boundary (at 8% active,
mean occupancy is ≈ 5/tile). But by that density the already-built AVX-512
dense kernel (`native/lif_kernel.c`, 0.219 ms/step measured, fully
contiguous, no gather instructions at all) is already a better answer than a
"mostly-skip" tiled kernel — it has zero per-tile branching and is already
tuned to that regime. The one condition that could revive this idea is
reordering neuron IDs so that neurons that tend to co-activate under a given
stimulus cluster into the same 64-ID blocks — untested here, and adjacent to
(but not identical to) the graph-reordering (RCM/METIS) idea ROADMAP.md
already ruled out for a *different* objective (delivery bandwidth, not
activity clustering); it would need its own measurement before being taken
seriously.

**Fidelity verdict: exact, if implemented** (a membership test changes
nothing about the arithmetic performed) **— but rejected on cost grounds**,
not fidelity grounds, for the density this workload actually shows.

## Ruled out from this field, and why

| Technique | Why not, here |
|---|---|
| **Edge-weighted (Beamer `m_f`/`m_u`) threshold applied to `dense_switch`** (the neuron-update switch, not the delivery switch) | Per-neuron update cost is O(1) and degree-independent (ROADMAP §2.1: dense update is 728:1 vs. synaptic work — the update itself does not touch the connectome at all). A vertex-count threshold is *already the correct resource metric* for this specific switch; importing edge-weighting here would optimize for a cost that isn't the bottleneck at this switch point. Applies only to the delivery-fan-out switch (`spike_switch`), covered in Applicable Technique 1. |
| **Hierarchical 64-neuron-tile bitset for the sparse active-set path** | Arithmetic above: at the measured 1.75 spikes/step, mean tile occupancy is ≈0.0008 — the technique wastes ~64× work per truly-active neuron in the regime that matters, and is dominated by the existing dense kernel in the regime where it would start to help. |
| **Adopting GraphIt's scheduling DSL / autotuner wholesale** | GraphIt's value is *automatically* discovering the frontier-representation choice (bitmap vs. array vs. hybrid) for an *unknown or varying* workload across many algorithms. This repo has exactly one fixed workload, already hand-characterized in more numeric detail (ROADMAP §2, §5) than any autotuner would rediscover, and GraphIt's cost model targets classic traversal (BFS/PageRank/SSSP edge-examination counts) — it has no notion of refractory period, threshold, or reset, so nothing beyond the frontier-representation idea (already extracted above) transfers. A general compiler layer would also add dispatch overhead, which ROADMAP §2.3 already identifies as the dominant cost once the active set is small — precisely the wrong direction. |
| **Ligra's `vertexSubset` abstraction as a general replacement for the current C/PyTorch split** | Same reasoning as GraphIt: Ligra is built for arbitrary traversal algorithms across arbitrary graphs; here the two concrete representations (sorted index array, dense AVX-512 pass) are already purpose-built and measured (0.219 ms/step dense; sparse wins to ~8%). Re-deriving them from a general framework adds abstraction cost without adding new capability for this one workload. |

## Concrete next actions for this codebase

1. **`flyloop/brain_engine.py`, `_init_active` (~line 239) and `step_active`
   (~line 295):** add `self.outdeg = self.fo_crow[1:] - self.fo_crow[:-1]`
   once at init; replace the fan-out guard
   `self._spike_idx.numel() > self.spike_switch * N` with
   `self.outdeg[self._spike_idx].sum() > self.edge_switch * E` (E =
   15,091,983). Before tuning `edge_switch`, measure the actual max/tail
   out-degree in this connectome (`self.outdeg.max()`, `.topk(20)`) — this
   number is not yet known in this session and determines how exposed the
   current count-only guard actually is.
2. **`flyloop/brain_engine.py`, lines 252–258, 295–323, 380–383:** implement
   bidirectional switching. Add `self.recover_switch` (start at
   `0.5 * dense_switch`, tune empirically per Rule 2). On the existing
   `prune_every`-cadence check (lines 362–365), while `_dense_fallback` is
   `True`, additionally test whether `active.sum() < recover_switch * N`;
   if so, rebuild `_idx = active.nonzero(as_tuple=True)[0]` and set
   `_dense_fallback = False`. Measurable effect: run a "burst-then-quiet"
   protocol (append a brief broad stimulus to the existing sugar protocol,
   then remove it) through `bench.py` and confirm the post-burst step rate
   returns to the sparse-mode rate instead of staying pinned at the
   dense-mode rate for the rest of the run.
3. **`flyloop/brain_engine.py`, `step_active` (lines 280–370):** this file's
   active set is evicted only every `prune_every = 64` steps (line
   363–365), meaning a neuron that spikes once and then receives no further
   input is still fully gathered/updated/scattered on every one of those up
   to 64 intervening steps even though its decay has a closed-form solution
   already derived in ROADMAP.md §5.1 (`u(h) = x·u0 + (g0/3)(x−x^4)`,
   semigroup `P(x1)P(x2)=P(x1·x2)`). Track a per-neuron `last_touch` step
   index; for neurons receiving no new input this step, defer the full
   update and instead apply one `pow`-based catch-up (reusing the exact
   constants already implemented in `flyloop/window_step.py` /
   `code/exact_spike_time.py`) plus the existing certified-silence bound
   (`KAPPA_WINDOW_MAX`, ROADMAP §5.2) to prove no threshold crossing was
   missed, when the neuron is finally touched again (new input, or the next
   `prune_every` check). Measurable effect: instrument `na = idx.numel()`
   over the course of the sugar regime and show its steady-state value
   drops (fewer "coasting straggler" neurons inflating every step's tensor
   size) — this directly attacks the dispatch-bound regime ROADMAP §2.3
   already measured (68% PyTorch per-op overhead once the active set is
   small). This is a distinct, smaller-scope idea from the flyloop
   delay-window parallel scan already flagged as separate unclaimed
   research — it applies inside the existing grid-stepped `step_active`
   path, not as a replacement for it.
4. **`flyloop/verify.py`:** add two regimes to the existing bit-identical
   gate before landing items 1–3: (a) a single-highest-out-degree-neuron
   spike in isolation (exercises item 1's blind spot), and (b) a
   burst-then-quiet protocol (exercises item 2's re-arm path). Both must
   produce bit-identical spike trains against the current dense reference,
   per the repo's existing Rule 1.

## References

- [Beamer, Asanović, Patterson — "Direction-Optimizing Breadth-First Search" (SC12, 2012)](https://parlab.eecs.berkeley.edu/sites/all/parlab/files/main.pdf) — source of the `m_f`/`m_u`/α/β hybrid switching heuristic and the sparse-queue/dense-bitmap representation split.
- [Beamer et al. — DOI record](https://dl.acm.org/doi/10.1109/SC.2012.50) — formal citation.
- [Shun & Blelloch — "Ligra: A Lightweight Graph Processing Framework for Shared Memory" (PPoPP 2013)](https://www.cs.cmu.edu/~guyb/papers/SB13.pdf) — independent confirmation of edge-count (`|E|/20`) switching threshold; explicitly built on Beamer et al.
- [Zhang et al. — "GraphIt: A High-Performance DSL for Graph Analytics" (arXiv:1805.00923)](https://arxiv.org/abs/1805.00923) — frontier representation (bitmap/bytemap/sparse array) as a schedule choice; grounds the "don't adopt the DSL, extract the idea" verdict.
- [Bautembach, Oikonomidis, Argyros — "Even Faster SNN Simulation with Lazy+Event-driven Plasticity and Shared Atomics" (arXiv:2107.04092, HPEC 2021)](https://arxiv.org/abs/2107.04092) — per-neuron 64-bit firing-history bitfield + `__clz()` intrinsic; closest SNN-specific precedent for hierarchical bit-tracking, cited in the negative (arithmetic) verdict.
- [Bautembach, Oikonomidis, Kyriazis, Argyros — "Faster and Simpler SNN Simulation with Work Queues" (arXiv:1912.07423, IJCNN 2020)](https://arxiv.org/abs/1912.07423) — the lazy-plasticity precursor to the 2021 paper above (age = now − last-update, replay via closed-form skip), same family as the semigroup catch-up idea in Concrete next action 3.
- [Bit-GraphBLAS: Bit-Level Optimizations of Matrix-Centric Graph Processing on GPU (arXiv:2201.08560)](https://arxiv.org/pdf/2201.08560) — two-level Bit-Block CSR container, general precedent for hierarchical bitmaps, cited for context only.
- [Roaring Bitmaps](https://roaringbitmap.org/) and [implementation paper](https://luca.ntop.org/roaring.pdf) — general hierarchical/compressed bitmap precedent, cited for context only.
- ROADMAP.md §5.1 (exact polynomial propagator, semigroup property), §5.2 (certified silence bound, `KAPPA_WINDOW_MAX`), §2.1 (728:1 cost ratio), §2.3 (dispatch-bound regime), §2.10 (auto-tuner crossover table), §6 (hub-dominated scale-free finding), §7 (invalidation risk ranking) — all reused directly from this repo's existing, previously-verified record; not re-derived here.

## Reviewer notes

**Citation verification (2026-07-30):** All top 5 citations verified as real, correctly attributed, and claiming what the document states.

- **Beamer et al. 2012 SC12** (DOI 10.1109/SC.2012.50): Real paper; correctly cites alpha/beta hybrid BFS switching heuristic with `m_f`/`m_u` edge-weighted cost model and bidirectional direction optimization.
- **Shun & Blelloch 2013 Ligra** (PPoPP 2013): Real paper; correctly cites edge-count threshold (`|E|/20`) and frontier representation switching (dense vs. sparse).
- **GraphIt 2018** (arXiv:1805.00923): Real paper accepted at OOPSLA 2018; correctly cites frontier representation (bitmap/bytemap/sparse array) as scheduling choices in a DSL context.
- **Bautembach et al. 2021** (arXiv:2107.04092, HPEC 2021): Real paper; correctly cites bitfields and integer intrinsics (including `__clz()`) for SNN spike-history tracking.
- **Bautembach et al. 2020** (arXiv:1912.07423, IJCNN 2020): Real paper; correctly cites work-queue design and lazy-plasticity approach; predecessor to 2021 paper.

**Arithmetic audit:** All numeric claims verified as correct.
- Mean out-degree: 15,091,983 / 138,639 = 108.89 ✓ (stated as "≈ 108.9")
- Tile count: ⌈138,639 / 64⌉ = 2,167 ✓
- Mean tile occupancy at measured density: 1.75 / 2,167 = 0.000807 ✓ (stated as "≈ 0.0008")
- Per-tile occupancy at 8% active: (138,639 × 0.08) / 2,167 = 5.11 ✓ (stated as "≈ 5/tile")
- Spike-count threshold: 0.004 × 138,639 = 554.556 ✓ (stated as "≈ 554")

**Verdict:** No fabricated citations, no arithmetic errors, no fidelity issues. Document is audit-clean.
