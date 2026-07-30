---
field: Memory hierarchy, sparse scatter, graph traversal direction
date: 2026-07-31
verdict: BOTH candidate fan-out optimisations are refuted by measurement. Beamer's push/pull switch loses by 110x on arithmetic alone (a pull costs O(E) = 15.1M edges every step against the 136,605 actually needed in the densest regime). Radix partitioning the scatter loses 0.47-0.87x at every size from 1,283 to 1,796,811 edges/step, because the direct scatter was never miss-bound: each presynaptic neuron's CSC range lists its targets in ASCENDING order, so the prefetcher handles it. The 21.7 ns/edge seen in the live saturating step is cache contention with the rest of the step, not the scatter's own behaviour.
---

# Fan-out: two optimisations, both refuted

HANDOFF §15 item 2 identified the fan-out as the thing to attack in the broad
and saturating regimes, and named Beamer's direction-optimising push/pull switch
as the candidate. Both that and the obvious alternative — radix partitioning the
scatter — were measured before being built. Both lose. Recording why, so nobody
re-treads them.

## First, the measurement that made this look promising

`flyloop/profile_split.py`, whole brain, 4 threads, min of 9 × 200 steps:

| regime | spikes/step | edges/step | full ms | fan-out ms | fan-out % | ns/edge |
|---|---|---|---|---|---|---|
| sugar (21) | 1.6 | 233 | 0.1010 | 0.0098 | 9.7% | 42.3 |
| broad (1000) | 60.7 | 14,996 | 0.8771 | 0.1124 | 12.8% | 7.5 |
| broad (10000) | 168.2 | 31,427 | 1.9876 | 0.2652 | 13.3% | 8.4 |
| saturating (40k) | 972.3 | 136,605 | 5.9875 | 2.9685 | **49.6%** | **21.7** |

Two things worth extracting before the negative results.

**Fan-out is half the saturating step.** HANDOFF §15 item 2 is right about that.

**And §2.1's headline needs a caveat.** "Synaptic propagation is only 0.14% of
runtime" is an *operation-count* statement — 233 edges against 138,639 neuron
updates — being read as a *runtime* statement. By actual time the fan-out is
**9.7%** of the sugar step, roughly 70× more than the op count suggests, because
the neuron sweep streams and the scatter does not. The conclusion drawn from it
(don't spend effort on spike delivery in the sparse regime) still holds at 9.7%,
but the number quoted for it should be the measured one.

## Refuted 1: the pull direction

Beamer's direction-optimising BFS switches from push (iterate the frontier, scatter
to neighbours) to pull (iterate all vertices, gather from the frontier) when the
frontier grows large enough that gathering is cheaper. The switch is a genuine
win in BFS on scale-free graphs and the connectome is scale-free, so it looked
applicable.

The arithmetic settles it without an implementation. A pull fan-out costs O(E)
per step regardless of activity, and:

```
E                             = 15,091,983 edges
edges/step, densest regime    =    136,605      (0.9% of E)
```

Pull would do **110× more edge work** than push in the *densest regime tested*,
and thousands of times more in the regimes the published experiments actually
use. Beamer's heuristic would never fire. The condition for it to pay is that
the frontier's out-edges approach a constant fraction of E, and 972 simultaneously
spiking neurons out of 138,639 do not come close — the spike raster is a very
sparse frontier by BFS standards, even when the brain is saturated.

Push wins everywhere in this connectome. Its advantage does not need atomics
either, since a single-threaded push has no contention to avoid — see the
separate measured negative on threading the fan-out (correct but 1.65× slower).

## Refuted 2: radix-partitioning the scatter

The `ns/edge` column above is the interesting one: 7.5 ns at broad(1000),
8.4 ns at broad(10000), then **21.7 ns** when saturating. The work per edge is
identical — one indexed int32 add — so a nearly-3× rise looked like a residency
problem. The accumulator is 138,639 int32 = 542 KB; a random scatter over all of
it should be missing L2.

The textbook fix is a radix-partitioned scatter, the same structure as a radix
join's partition phase or the counting phase of a sparse transpose: bucket the
`(post, val)` pairs by the high bits of `post` in one sequential pass, then
accumulate a bucket at a time so the live accumulator slice is 16 KB and stays
in L1. It moves *more* bytes, but all sequentially.

Implemented (`fanout_partitioned` in `nrn_kernel.c`, PART_BITS = 12 → 4096
neurons per bucket) and measured (`flyloop/bench_fanout.py`):

| edges/step | direct | partitioned | speedup | ns/edge direct | ns/edge part |
|---|---|---|---|---|---|
| 1,283 | 8.5 µs | 14.0 µs | 0.61× | 6.66 | 10.87 |
| 5,715 | 30.3 µs | 36.0 µs | 0.84× | 5.30 | 6.29 |
| 28,322 | 182.3 µs | 310.3 µs | 0.59× | 6.44 | 10.96 |
| 51,564 | 582.1 µs | 667.9 µs | 0.87× | 11.29 | 12.95 |
| 106,124 | 1.07 ms | 1.44 ms | 0.74× | 10.04 | 13.57 |
| 228,609 | 2.12 ms | 2.90 ms | 0.73× | 9.28 | 12.70 |
| 447,089 | 3.41 ms | 5.27 ms | 0.65× | 7.63 | 11.78 |
| 907,682 | 5.42 ms | 9.96 ms | 0.54× | 5.97 | 10.98 |
| 1,796,811 | 12.44 ms | 26.48 ms | 0.47× | 6.93 | 14.74 |

**It never wins.** And the reason is visible in the direct column: the direct
scatter holds **6–11 ns/edge all the way to 1.8 million edges per step** and does
not degrade with size. It was never miss-bound.

The cause of that is structural and easy to miss: within one presynaptic
neuron's CSC range, the postsynaptic indices are stored in **ascending order**.
Each source's fan-out is therefore a sequential walk over `acc`, which the
hardware prefetcher handles perfectly. The access pattern is only random
*between* sources, and with a mean out-degree of 108.9 each source contributes a
long enough run to amortise the jump.

So the 21.7 ns/edge in the *live* saturating step is not the scatter's intrinsic
cost. It is **contention**: in a real step the fan-out shares cache with the
neuron sweep's state (138,639 × 8 bytes of `v` and `g`), the tile arrays and the
delay slots, all of which are being touched in the same millisecond.
Partitioning attacks the wrong thing and makes contention worse, adding ~1 MB of
extra traffic that evicts more of the competing working set.

Both paths produce **bit-identical** compacted output at every size, as they must
— integer accumulation is associative and the totals never round (ANALYSIS.md
§1) — so this is purely a performance verdict, verified rather than assumed.

## What this leaves

The saturating regime's remaining cost is not one hot spot with a known fix. It
is the whole step's working set exceeding cache at once, and the levers for that
are the ones already identified in HANDOFF §13 and the memory-layout note:
shrink the resident state (int16 in shifted coordinates, uint8 saturating
counters) rather than reorganise the traffic.

The measured negative also slightly re-frames the priority. Fan-out is 49.6% of
the saturating step, but the saturating regime is an artificial stress test —
neither published experiment goes near it. In the regimes the science actually
uses, fan-out is 9.7–13.3% and the direct scatter is already running at 6–8
ns/edge. There is not much there.

## Method note

Both refutations cost one afternoon and one benchmark each. The partitioned
implementation is kept in the kernel rather than deleted, gated behind a
caller-supplied buffer that the engine passes as NULL, so the measurement stays
reproducible by anyone who suspects the conclusion. That seems the right way to
retire an optimisation: leave the evidence runnable.
