---
field: SIMD kernel engineering and multicore scaling
date: 2026-07-30
verdict: Static 4-way partition of the existing branchless AVX-512 sweep, aligned to 64-neuron bitset words and pinned 1 thread/physical core, should land at ~55-60 us/step against a 100 us budget — implement a persistent spin-wait pool (not per-step OpenMP fork/join), leave the 0.14% fan-out/delivery loops single-threaded, and never add non-temporal stores.
---

# Multicore scaling of the fused AVX-512 LIF kernel

## Bottom line

The dense sweep in `native/lif_kernel.c` is embarrassingly data-parallel across
neurons — no cross-neuron dependency, no data-dependent branch, uniform
per-element cost — so it is exactly the case multicore scaling was built for,
and it should come far closer to linear 4x than almost any other kernel in
this codebase. Amdahl's law on the measured 99.86%-parallel fraction gives
~3.98x, i.e. ~55 us of compute against the 100 us/step budget, leaving ~45 us
of headroom for synchronization and the un-parallelized 0.14%. The one
correctness-critical detail is that the spike-bitset word (64 neurons) must
never be split across a partition boundary, because it is updated with a
non-atomic read-modify-write. AVX-512 downclocking, the classic Skylake-X
trap, is very likely a non-issue on this specific chip (Willow Cove, same
license-simplified generation as Ice Lake/Rocket Lake client) but has not been
directly measured on the 1185G7 — that measurement is a five-minute action
item, not a research question. The one thing this field does NOT offer here:
non-temporal stores, cache-oblivious blocking, or software prefetch. The
working set already fits L2 per thread; those techniques solve problems this
kernel does not have and non-temporal stores would actively break the one
correct answer (cache residency across steps).

## Applicable techniques

### 1. Static contiguous partitioning across 4 physical cores

**Core idea.** Split the N=138,639-neuron range into 4 static, contiguous
chunks, one per physical core, each processed by `sweep_avx512`/`sweep_scalar`
unchanged — no reduction, no cross-thread write, because every neuron's
update reads and writes only its own `v[i], g[i], refrac[i]`. This is the
textbook "embarrassingly parallel" case for the roofline/multicore literature
(Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance
Model for Multicore Architectures*, CACM 2009,
https://cacm.acm.org/research/roofline-an-insightful-visual-performance-model-for-multicore-architectures/):
operational intensity and per-element cost are both constant across the
array, so there is no load-imbalance term to amortize and a **static**
schedule dominates any **dynamic**/work-stealing schedule, which would only
add per-chunk dispatch overhead for zero balancing benefit. Dynamic/guided
OpenMP scheduling is the wrong tool here specifically because the kernel is
branchless — every neuron costs the same number of cycles whether it spikes,
decays, or sits at rest, by construction of the masked-blend design already
in `sweep_avx512`.

**Arithmetic.** Measured single-core fused kernel: 0.219 ms/step (`HANDOFF.md`
§2, this session). The un-parallelizable fraction is the event-driven fan-out
+ delayed-input application, independently measured at 0.14% of runtime
(`ROADMAP.md` §1, 728:1 ratio). Amdahl's law:

```
speedup(n) = 1 / ((1-p) + p/n),   p = 0.9986, n = 4
           = 1 / (0.0014 + 0.24965)
           = 3.983x
```

`0.219 ms / 3.983 ≈ 0.055 ms = 55 us` of compute, against the 100 us/step
real-time budget — **45 us of margin** for synchronization plus the serial
0.14% (≈0.3 us, negligible). At n=2 the same formula gives 1.997x → 0.1097 ms,
i.e. **landing almost exactly ON the 100 us line with zero margin for sync
overhead** — a concrete reason 4 cores are not merely "better" but load-bearing
for this specific budget on this specific chip.

**Fidelity verdict: bit-identical, exactly, not merely bounded-error.**
Unlike a reduction (sum, histogram, matmul), there is no accumulation across
threads to reorder, so floating-point associativity is not in play at all.
Each neuron's `fmaf`/`mul`/`add` sequence executes on exactly the core it
would have executed on serially, with the same inputs and the same rounding.
The only requirement to preserve is that partition boundaries be
**deterministic and identical on every run** (fixed at engine construction,
not runtime-load-balanced) — which a static partition trivially gives for
free. This should be provable in `verify_native.py` with the *same* 8-regime
suite already gating single-threaded changes, just run once with 1 thread and
once with 4.

### 2. Bitset-word (64-neuron) and cache-line (16-float) aligned partition boundaries — correctness, not just performance

**Core idea.** `lif_kernel.c`'s spike output is a packed bitset,
`sp_out[i>>6] |= 1ULL << (i&63)` / `&= ~(...)`, one 64-bit word per 64
neurons. That line is a **non-atomic read-modify-write**. If a partition
boundary falls inside a 64-neuron group, two threads write the same word
concurrently and the update is a genuine data race (lost updates), not merely
a slow one — this is stricter than ordinary false sharing. Cache-line
alignment (16 floats = 64 B) on `v`/`g`/`refrac` is the softer, throughput-only
version of the same idea: classic false-sharing guidance (Sun/Oracle Studio
OpenMP guide, *6.2 False Sharing and How to Avoid It*,
https://docs.oracle.com/cd/E19205-01/819-5270/6n7c71veg/index.html) is that
cache coherency invalidates a whole 64 B line even when two cores write
logically disjoint elements inside it, causing line ping-pong between core
private caches every step.

**Arithmetic / concrete boundaries.** `N = 138,639`. Round chunk START boundaries to
multiples of 64 (which is also a multiple of the 16-float AVX-512 vector
width, so no partition also splits a `sweep_avx512` 16-lane group):
`138,639 / 4 = 34,659.75` → chunks of `34,688 / 34,688 / 34,688 / 34,575`
(each chunk a multiple of 64 except the necessarily-short last one, which ends
at N — `34,575` is not a multiple of 64; instead use `34,624/34,624/34,624/34,767` 
where the three internal boundaries at 34,624, 69,248, and 103,872 are multiples of 64, 
or any split that keeps every *internal* boundary on a multiple of 64; the exact split 
is a one-line arithmetic check to add to `NativeBrainEngine.__init__`). This is pure 
bookkeeping, no algorithmic change, and it is the single correctness-blocking detail 
before any thread touches the bitset.

**Fidelity verdict: exact** (it is a memory-layout precondition for
bit-identity, not an approximation).

### 3. Persistent spin-wait thread pool, not per-step OpenMP fork/join

**Core idea.** The step budget is 100 us; a spawn/join or a cold OpenMP
parallel-region entry can cost far more than that. Intel's own guidance on
`KMP_BLOCKTIME`/`OMP_WAIT_POLICY` documents the tradeoff directly: threads
spin for `KMP_BLOCKTIME` (default 200 ms) before sleeping at a barrier, and
setting it to `0` makes a thread check once and sleep immediately — cheap to
re-wake become expensive; setting it high/infinite keeps threads hot at the
cost of burning a core while idle (Intel Community, *OMP_WAIT_POLICY /
KMP_BLOCKTIME*, https://community.intel.com/t5/Intel-C-Compiler/OMP-WAIT-POLICY-OMP-Barrier-question/td-p/1148461).
For a step budget this tight, the only sane choice is to burn the core: 4
threads, pinned, permanently spinning on a small atomic generation counter,
started once in `NativeBrainEngine.__init__` and never joined until the
simulation ends. This is exactly the design point Ash Vardanian's Fork Union
work targets — commodity OS-scheduled thread pools (`std::thread` + queue +
mutex) are reported up to ~195x slower than a warm fork-join dispatch on a
reduction-style workload, while a purpose-built spin-wait pool closes to
within ~20% of OpenMP's own (already-warm) dispatch latency (Ash Vardanian,
*Beyond OpenMP in C++ & Rust: Taskflow, Rayon, Fork Union*,
https://ashvardanian.com/posts/beyond-openmp-in-cpp-rust/, and
https://github.com/ashvardanian/ForkUnion). OpenMP itself, once warm
(`KMP_BLOCKTIME` high), typically dispatches a parallel region in on the
order of 1 us (same source) — small compared to the 45 us margin computed
above, so either a hand-rolled spin-wait pool (2 barriers/step: one after the
dense sweep, done) or OpenMP with the wait policy tuned aggressively are both
viable; the hand-rolled pool is lower-risk because it removes the runtime's
own heuristics from a 100 us-budget hot path entirely.

**Arithmetic.** Cross-core synchronization latency is bounded above by the
core-to-core cache-coherency round trip. Tiger Lake's four cores are on one
monolithic ring/L3 (unlike multi-socket or tiled server parts), which is the
regime in which these tools (e.g. https://github.com/nviennot/core-to-core-latency)
typically report tens of nanoseconds, not the 100+ ns seen crossing NUMA
nodes or chiplets — no measurement specific to this exact SKU was found this
session (flagged below as an action item), but the order of magnitude is
consistent with the ~1 us dispatch figure above being dominated by pool
bookkeeping, not raw wire latency. Either way it sits at least an order of
magnitude below the 45 us of headroom.

**Fidelity verdict: exact** (a barrier only orders execution; it changes no
arithmetic).

### 4. 4 threads pinned to physical cores, not 8 via Hyper-Threading

**Core idea.** Tiger Lake client cores (like Ice Lake client) carry a single
512-bit-wide FMA unit per physical core — server SKUs (Skylake-X, Cascade
Lake, Ice Lake-SP) have two. Two SMT siblings on one physical core would
contend for that one FMA port on a kernel whose inner loop is dominated by
vector FMA/blend/compare, so the second logical thread buys nothing on the
execution-port-limited part of the loop and adds L1/L2 contention on the
part that is not. Pin exactly 4 worker threads, one per physical core
(`SetThreadAffinityMask` on Windows), and leave the other 4 logical
processors alone.

**Arithmetic (why there is headroom to give up SMT).** FMA utilization at
n=1 is already low: ~10 FLOP/neuron x 138,639 neurons / 0.219 ms ≈
6.3 GFLOP/s achieved, against ≈96 GFLOP/s theoretical peak for one core
(16 fp32 lanes x 2 flops/FMA x 3.0 GHz) — **≈7% of peak**. The kernel is
nowhere near saturating the single FMA port even single-threaded, which is
independent confirmation (not just architectural folklore) that the 0.219 ms
is not FMA-throughput-bound; it is far more likely load/store-latency or
front-end/dependency-chain bound. That means 4-way physical-core scaling has
no shared-execution-unit ceiling to hit, and 8-way SMT scaling has nothing to
gain from doubling occupancy on an already-underutilized port while risking
the L1D/L2 the two logical threads would now be time-sharing.

*Note on Windows scheduling*: Tiger Lake (11th gen mobile) predates Intel's
hybrid P-core/E-core designs (Alder Lake, 12th gen+), so there is no
Thread Director / QoS-class complication here — all 4 cores are identical
Willow Cove cores, one less variable than a 12th-gen-or-later target would
have.

**Fidelity verdict: exact** (this is a scheduling/affinity choice, changes no
arithmetic; it is a throughput argument for *why 8 threads would not help*,
not a correctness claim).

### 5. Roofline check: confirms this is not a bandwidth problem at either scale

**Core idea/equation.** Williams/Waterman/Patterson roofline: performance is
bounded by `min(peak FLOP/s, operational intensity x peak bandwidth)`. Two
bounds, using the working-set figures already measured in this repo:

- **Pessimistic (DRAM-bound) bound.** Treat every neuron-step as if it had to
  round-trip DRAM: reads `v,g,refrac,refrac_steps` (4x4 B) + writes
  `v,g,refrac` (3x4 B) = 28 B/neuron worst case (the task brief's ~19 B/neuron
  figure is inside this envelope — the gap is exactly the fraction of that
  traffic that is *not* DRAM traffic once the state is cache-resident, per
  the second bound below). At `138,639 x 28 B ≈ 3.88 MB/step` against Tiger
  Lake's measured all-core LPDDR4x-4266 bandwidth of ~57 GB/s (Notebookcheck,
  citing Tiger Lake vs Ice Lake bandwidth scaling,
  https://www.notebookcheck.net/Here-is-why-LPDDR4x-4266-RAM-seems-considerably-faster-when-coupled-with-Intel-Tiger-Lake-U-CPUs-compared-to-equivalent-AMD-Renoir-APUs.480844.0.html):
  `3.88e6 / 57e9 ≈ 68 us` — under the 100 us budget even in the *worst case*
  where nothing is cache-resident.
- **Realistic (L2-resident) bound.** `HANDOFF.md` §2.6 already establishes the
  full working set (v, g, refrac, refrac_steps, all fp32) is 2.2 MB, which
  fits the 12 MB shared L3 whole, and — once quartered across 4 static
  partitions — **554 KB per thread, comfortably inside the 1.25 MB private
  L2 per core.** L2 bandwidth per core is roughly an order of magnitude above
  DRAM bandwidth on any modern x86 core, so once the pool is warm and each
  thread's slice stays L2-resident across steps (guaranteed by the static
  partition — the same 554 KB is touched every step, nothing else competes
  for that L2), this bound is not binding at all.

**Conclusion from the roofline.** The observed 0.219 ms is **not** explained
by either bandwidth bound (68 us pessimistic, effectively free optimistic) —
it is compute/dispatch-bound, at only ~7% of FMA peak (see technique 4). This
is the roofline telling us *not* to spend effort on cache-blocking beyond the
already-correct partition sizing, non-temporal stores, or prefetch tuning:
there is no bandwidth roof anywhere near the observed ceiling, on this
kernel, at this problem size.

**Fidelity verdict: exact** (this is a measurement/diagnostic technique, it
changes no code by itself — it says *where* to invest, not what to compute).

### 6. AVX-512 downclocking on this generation — verify, don't assume

**Core idea.** The Skylake-X/Cascade Lake license-based downclock (separate
L0/L1/L2 frequency states triggered by 256-bit-"light", 512-bit-"light", and
512-bit-"heavy" instruction mixes) is well documented and was a real trap on
server parts of that generation. Ice Lake client collapsed this to a single
L1 level (width alone matters, the light/heavy distinction is gone) and
measured only ~100 MHz single-core reduction with **no measurable
multi-core AVX-512 downclock** (Travis Downs, *Ice Lake AVX-512 Downclocking*,
https://travisdowns.github.io/blog/2020/08/19/icl-avx512-freq.html, using his
own `avx-turbo` tool: https://github.com/travisdowns/avx-turbo). Rocket Lake
(same-generation Cypress Cove core, desktop) showed **no license-based
downclock at any core count** in the same tests. Intel's own public response
to the "AVX-512 downclocking" controversy characterized the client-generation
impact as "insignificant or zero" (reported via PCWorld/Slashdot coverage of
Intel's statement, https://www.pcworld.com/article/393372/intel-defends-avx-512-against-critics-who-wish-it-to-die-a-painful-death.html).
Tiger Lake's Willow Cove core is the same license-simplified generation
(10 nm SuperFin, one generation past Ice Lake client), so by strong
architectural analogy it should behave the same way — **but no
`avx-turbo` run specific to an i7-1185G7 was found this session**; one
commenter on Downs' own post reported 1165G7 numbers only under
virtualization, which he flags as unreliable. This is a gap, not a
conclusion.

**Arithmetic if the worst case held anyway.** Even if Tiger Lake behaved like
1st-gen Skylake-X (it almost certainly does not), the all-core AVX-512 clock
floor documented for that generation was still typically ≥80–85% of
all-core non-AVX clock, i.e. a ≤15-20% single-kernel slowdown — the 45 us of
Amdahl headroom computed above absorbs a hit that size without missing the
100 us budget (55 us x 1.2 ≈ 66 us, still comfortably under 100 us). So even
the pessimistic historical case does not break real time; it only shrinks
the safety margin.

**Fidelity verdict: N/A** — this is a throughput risk assessment, not an
accuracy question; clock frequency does not change any bit of the
computation.

## Ruled out from this field, and why

| Technique | Why not, here |
|---|---|
| **Non-temporal (streaming) stores** (`_mm512_stream_ps` / `MOVNTPS`) | NT stores exist to avoid cache pollution for data that will **not** be re-read soon (Intel Optimization Reference Manual, streaming-store usage models, https://www.intel.com/content/dam/doc/manual/64-ia-32-architectures-optimization-manual.pdf). Every element of `v`, `g`, `refrac` is re-read on the very next step, 10,000 times/simulated-second — the opposite of the NT-store use case. Using them here would force every step to round-trip DRAM instead of reusing the L2-resident 554 KB/thread established in technique 5, actively converting a ~free-bandwidth loop into a DRAM-bound one. **Do not add these.** |
| **Software prefetch** (`_mm_prefetch`) | The access pattern is fully sequential, unit-stride, over arrays the hardware L2 streamer and L1 IP-based prefetchers are built to detect trivially. Software prefetch earns its keep on irregular/gather access (exactly the delayed-synapse and fan-out lists, which are already only 0.14% of runtime and explicitly excluded from parallelization) or on strides the HW prefetcher can't see — not on this loop. |
| **Cache-oblivious / recursive blocking algorithms** | Those solve the problem of a working set that does *not* fit any single cache level cleanly. §5 already shows the per-thread partition (554 KB) fits comfortably inside one core's L2 (1.25 MB) with room to spare; there is no larger structure to block for. Cache-oblivious recursion would add branching and index-computation overhead to a kernel whose entire value proposition is being branchless. |
| **Dynamic/guided OpenMP scheduling** | Per technique 1: the kernel is branchless with uniform per-element cost by design (the whole reason for the AVX-512 masked-blend rewrite). Dynamic scheduling amortizes *load imbalance*, which does not exist here; it would only add chunk-dispatch overhead against a 45 us margin that is better spent on nothing at all. |
| **8-way SMT / Hyper-Threading** | Per technique 4: one 512-bit FMA port per physical core is already at ~7% utilization single-threaded; the second logical thread has no spare execution-port capacity to exploit and would only add L1D/TLB pressure. |
| **GPU offload (Iris Xe)** | Out of scope for *this* field brief (SIMD/multicore), and already flagged as a measure-first item in `HANDOFF.md` §7.6 — active-set compaction may cost more than the dense kernel at the small active-set sizes typical of sparse stimulation. Not revisited here. |
| **Cross-thread reduction / atomics for the fan-out and delayed-input loops** | Those loops touch ~190 scattered elements out of 138,639 (0.14% of runtime, `HANDOFF.md` §2.1/2.7). Parallelizing or atomic-protecting them would spend synchronization cost on a workload too small to amortize it, and reintroduces exactly the false-sharing/race risk that partitioning by contiguous neuron range was designed to avoid. Leave them single-threaded on one worker while the others either idle at the barrier or, better, start drawing next-step Poisson stimulus concurrently. |

## Concrete next actions for this codebase

1. **`native/lif_kernel.c`** — add a `lif_step_mt(int nthreads, int *chunk_lo, int *chunk_hi, ...)` entry point (or a thin wrapper that calls the existing `sweep()` per-partition) that takes pre-computed, 64-neuron-aligned chunk boundaries as arguments. Do not touch `sweep_scalar`/`sweep_avx512` internals — they are already correct and bit-identical per neuron; only the calling convention changes.
2. **`native_engine.py` `NativeBrainEngine.__init__`** — compute the 4 chunk boundaries once (multiples of 64, summing to `N=138,639`, e.g. `34,688/34,688/34,688/34,575` rounded to the nearest valid multiple-of-64 split), and start 4 persistent, core-pinned OS threads (Windows `SetThreadAffinityMask` via `ctypes`/`kernel32`) that spin on a small shared atomic step-generation counter — not a per-step Python-level thread spawn, and not default OpenMP.
3. **`native_engine.py` `NativeBrainEngine.step`** — replace the single `self.lib.lif_step(...)` call with: (a) signal the 4 workers to run their chunk of the dense sweep, (b) barrier, (c) run the existing serial `del_idx`/`del_val` application and `lif_fanout` exactly as today (do not parallelize — see ruled-out table), (d) barrier/return. Two barriers per step, not four.
4. **Validation gate** — extend `verify_native.py` (and/or `flyloop/verify.py`) to run the existing 8-regime bit-identical suite with `nthreads=1` and `nthreads=4` and assert identical `v`, `g`, `refrac`, and spike trains. Per technique 1, this should pass trivially since there is no cross-thread reduction — but it is the actual proof, not the argument, and must gate the PR per this repo's Rule 1.
5. **Measure, don't assume, the AVX-512 clock behavior on THIS chip.** Run `travisdowns/avx-turbo` (https://github.com/travisdowns/avx-turbo) on the i7-1185G7 before and after landing the threaded kernel, at 1 and 4 active cores, AVX2 and AVX-512. This is a 10-minute action that converts §"Applicable techniques" #6 from "very likely fine, by analogy" into a measured fact for `HANDOFF.md`.
6. **Measure the real per-step wall-clock, not the isolated kernel figure.** The 0.219 ms/step baseline is the fused C kernel; confirm what it included (ctypes call overhead, the `torch.bernoulli` Poisson draw already kept in Python for RNG-stream fidelity — `native_engine.py` lines 209-217). Per this repo's Rule 2 ("measure, don't model"), benchmark the *threaded* `step()` end-to-end in the same harness used for the 0.219 ms figure before claiming the 55-60 us number in a table — the Amdahl arithmetic above is a target and a sanity bound, not a substitute for the measurement.
7. **Do not add non-temporal stores or software prefetch to `sweep_avx512`/`sweep_scalar`** — see ruled-out table; both would be net-negative on this access pattern.

## References

- [Williams, Waterman & Patterson — Roofline: An Insightful Visual Performance Model for Multicore Architectures, CACM 2009](https://cacm.acm.org/research/roofline-an-insightful-visual-performance-model-for-multicore-architectures/) — the model used in technique 5.
- [Travis Downs — Ice Lake AVX-512 Downclocking (2020)](https://travisdowns.github.io/blog/2020/08/19/icl-avx512-freq.html) — measured single/multi-core AVX-512 frequency behavior on Ice Lake client and Rocket Lake; license-level simplification vs Skylake-X.
- [travisdowns/avx-turbo (GitHub)](https://github.com/travisdowns/avx-turbo) — the tool to run directly on the i7-1185G7 (action item #5).
- [PCWorld — Intel defends AVX-512 against critics (2020)](https://www.pcworld.com/article/393372/intel-defends-avx-512-against-critics-who-wish-it-to-die-a-painful-death.html) — Intel's own characterization of client-generation downclocking as insignificant/zero.
- [Intel Community — OMP_WAIT_POLICY / KMP_BLOCKTIME](https://community.intel.com/t5/Intel-C-Compiler/OMP-WAIT-POLICY-OMP-Barrier-question/td-p/1148461) — spin-vs-sleep barrier tradeoff informing the persistent-pool design in technique 3.
- [Ash Vardanian — Beyond OpenMP in C++ & Rust: Taskflow, Rayon, Fork Union](https://ashvardanian.com/posts/beyond-openmp-in-cpp-rust/) and [ForkUnion (GitHub)](https://github.com/ashvardanian/ForkUnion) — spin-wait pool vs OS-scheduled thread-pool overhead numbers used in technique 3.
- [nviennot/core-to-core-latency (GitHub)](https://github.com/nviennot/core-to-core-latency) — the tool that would give a direct barrier-latency lower bound on this specific chip (not yet run this session).
- [Notebookcheck — Why LPDDR4x-4266 seems faster with Tiger Lake vs Renoir](https://www.notebookcheck.net/Here-is-why-LPDDR4x-4266-RAM-seems-considerably-faster-when-coupled-with-Intel-Tiger-Lake-U-CPUs-compared-to-equivalent-AMD-Renoir-APUs.480844.0.html) — the ~57 GB/s all-core bandwidth figure used in technique 5's pessimistic roofline bound.
- [Oracle/Sun Studio OpenMP Guide — 6.2 False Sharing and How to Avoid It](https://docs.oracle.com/cd/E19205-01/819-5270/6n7c71veg/index.html) — false-sharing mechanism and cache-line-padding remedy cited in technique 2.
- [Intel 64 and IA-32 Architectures Optimization Reference Manual](https://www.intel.com/content/dam/doc/manual/64-ia-32-architectures-optimization-manual.pdf) — non-temporal store usage guidance cited in the ruled-out table.
- This repo — [`HANDOFF.md`](../HANDOFF.md) §2 (cost distribution, cache/working-set figures, 0.219 ms measurement), §7 (invalidation risks) and [`flyloop/ROADMAP.md`](../flyloop/ROADMAP.md) §1 (dispatch-bound diagnosis) — all measured facts this note builds arithmetic on top of.

## Reviewer notes

**Citation verification (2026-07-30):** All 5 top-cited sources verified to exist and match their claimed usage:
- **Williams/Waterman/Patterson CACM 2009:** Roofline paper confirmed via ACM DL, Semantic Scholar, DBLP. Correctly cites the multicore performance model used in technique 5.
- **Travis Downs Ice Lake AVX-512 blog (2020):** Blog post confirmed. Accurately reports single-core 100 MHz downclock, no multi-core AVX-512 downclock on Ice Lake client.
- **travisdowns/avx-turbo GitHub:** Repository confirmed active with 228 stars. Tool does exactly what cited: measures AVX-512 frequency across core counts.
- **PCWorld Intel AVX-512 defense (2020):** Article confirmed. Correctly quotes Intel's characterization of client-generation impact as "insignificant or zero."
- **Intel Community KMP_BLOCKTIME/OMP_WAIT_POLICY:** Forum thread confirmed at exact URL. Accurately describes spin-vs-sleep tradeoff and KMP_BLOCKTIME mechanism.

**Arithmetic audit (2026-07-30):**
- Amdahl's law (n=4, p=0.9986): 1/(0.0014+0.24965)=3.983x ✓
- Speedup to latency: 0.219 ms ÷ 3.983 = 55 us ✓
- Two-core bound: 1.997x → 0.1097 ms (on 100 us budget) ✓
- DRAM bandwidth pessimistic bound: 3.88 MB ÷ 57 GB/s = 68 us ✓
- FMA utilization: 6.33 GFLOP/s ÷ 96 GFLOP/s ≈ 6.6% (document says "~7%", acceptable rounding) ✓
- L2 residency per thread: 2.2 MB ÷ 4 = 550 KB < 1.25 MB L2 ✓

**Clarity fix applied:** Line 100-102 wording clarified — the document now explicitly states that only the three *internal* partition boundaries (at neurons 34,624, 69,248, 103,872) must be multiples of 64, while the final chunk end position (138,639) need not be. The original text was ambiguous about whether the last chunk's end position had to align.
