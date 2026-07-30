---
field: Compilers and codegen for tight numerical kernels
date: 2026-07-30
verdict: No compiler beats the hand-written AVX-512 kernel that already exists — the only codegen-adjacent move left this week is giving it 4 physical cores via a persistent spin-wait thread pool (ISPC's `tasks` or hand pthreads), which the roadmap itself already flags as unclaimed ("untuned, unthreaded").
---

# Compilers and codegen for tight numerical kernels — verdict for `lif_kernel.c`

## Bottom line

`flyloop/native/lif_kernel.c` is already close to what a specialized numerical
compiler (ISPC, Halide, a from-scratch LLVM SIMD backend) would emit for this
exact shape: a branchless per-element sweep with AVX-512 mask-driven select
instead of control flow, explicit FMA where the reference wants one rounding
and explicit non-FMA where it wants two. The published evidence is consistent
across three independent sources (ISPC's own SPMD paper, a 2023 CGO paper
re-deriving an SIMD compiler from scratch, and GeNN's brand-new ISPC backend)
that specialized SIMD compilers land at 97–100% of hand-intrinsics throughput,
never meaningfully above it, for kernels this shape. None of them close a
2.2× gap by themselves. **torch.compile/TorchInductor, numba, Halide, TVM and
GeNN are all strictly worse starting points than the kernel you already have**
— each would have to be fought back down to the same instruction sequence you
already hand-verified, with less control over per-line FMA contraction than
raw C + `-ffp-contract=off` gives you directly. The one thing none of them
gives you for free, and that the kernel genuinely lacks, is **multicore
dispatch** — the current 0.219 ms/step number is explicitly single-core. A
compiler-adjacent technique (ISPC's `launch`/`tasks`, or a hand-rolled
spin-wait barrier) applied to the *existing* kernel is the highest-confidence
remaining move, and it is squarely a codegen/toolchain question, not a new
algorithm.

## Applicable techniques

### 1. ISPC (Intel Implicit SPMD Program Compiler) — as a second implementation to benchmark against, and for its tasking model
[Pharr & Mark, "ispc: A SPMD Compiler for High-Performance CPU Programming," InPar/HPCC 2012](https://pharr.org/matt/assets/ispc.pdf); [ISPC user's guide](https://ispc.github.io/ispc.html); [ISPC FAQ](https://ispc.github.io/faq.html).
Core idea: you write scalar-looking code over a `foreach` loop; ISPC compiles
it SPMD-style to native SIMD (SSE4/AVX2/AVX-512) the same way a GPU shader
compiler turns per-thread code into warps. It exposes an explicit `--opt=disable-fma`
flag specifically because FMA changes rounding relative to separate
mul/add — i.e. it gives the same knob `-ffp-contract=off` gives clang, at the
same granularity (whole compilation unit, not per-line — same limitation the
hand C file has, which is why `lif_kernel.c` uses an *explicit* `fmaf()` call
for the membrane update instead of relying on the flag). ISPC also ships a
`tasks`/`launch` construct that maps directly onto a work-stealing or static
thread pool — this is the part actually missing from the current kernel.
Expected speedup: on the raw vectorization, parity with the current 7.9×
already measured (both reach native SIMD width); the incremental value is
purely the tasking layer, worth up to ~4× more on 4 physical cores if launch
overhead is amortized (see next-actions arithmetic). Fidelity verdict:
**bit-identical is achievable but not free** — must explicitly disable FMA
generation and hand-insert the one `fma()` call the membrane update needs,
mirroring exactly what `lif_kernel.c` does today; nothing about ISPC removes
that burden, it just relocates it.

### 2. GeNN's new ISPC CPU backend — direct precedent for this exact kernel shape
[genn-team.github.io blog, "Developing an ISPC Backend for GeNN"](https://genn-team.github.io/posts/developing-an-ispc-backend-for-genn-bridging-gpu-and-cpu-performance-for-neural-network-simulations.html) (2026, project blog, not peer-reviewed); underlying framework: [Yavuz, Turner & Nowotny, "GeNN: a code generation framework for accelerated brain simulations," *Scientific Reports* 6:18854, 2016](https://www.nature.com/articles/srep18854).
Core idea: GeNN already code-generates C++/CUDA for exactly this class of
model (LIF + single-exponential synapse); its new backend retargets the same
neuron-update template through ISPC instead of raw OpenMP/CUDA, using
`foreach` for automatic vectorization. Reported speedups over their existing
single-threaded scalar CPU backend: **1.4× on a sparse network (AVX2)**,
**3.05–3.1× on dense networks (AVX2, i5/i7-12700H)**, **9.49× on dense
networks with AVX-512 (Xeon Gold 6134)**. That AVX-512 number is in the same
range as the 7.9× already measured here with hand intrinsics — independent
confirmation that a specialized SIMD compiler and hand-rolled AVX-512 land in
the same place for this shape, neither one dominating. The blog post does
**not** discuss FMA contraction or bit-identity at all ("maintained exact
functional behavior" is asserted, not demonstrated at the ULP level), so it
cannot be cited as evidence ISPC solves the reproducibility problem — only as
evidence the vectorization ceiling is where the existing kernel already sits.
Fidelity verdict: **not verified by the source**: their claim is functional
equivalence, not bit-identity; would require the same manual FMA audit this
project already did.

### 3. Parsimony (CGO 2023) — independent confirmation of the ISPC-parity ceiling
[Parsimony: Enabling SIMD/Vector Programming in Standard Compiler Flows, CGO 2023](https://dl.acm.org/doi/10.1145/3579990.3580019) / [NVIDIA Research page](https://research.nvidia.com/publication/2023-02_parsimony-enabling-simdvector-programming-standard-compiler-flows).
Core idea: builds a from-scratch LLVM-based SPMD compiler (independent
implementation, same idea as ISPC) and benchmarks it against both ISPC and
raw AVX-512 intrinsics on 70+ ported benchmarks from the Simd library.
Result: matches ISPC's performance and reaches **97% of hand-written AVX-512
intrinsics**. Two independently-built SPMD compilers both land at ~97–100%
of hand intrinsics, never above — this is the strongest available evidence
that there is no compiler-technology speedup left on the table for a
branchless per-element kernel once it is already hand-vectorized, only
engineering effort saved. Expected speedup over the current kernel: **none,
by design of the comparison** (≤1.0×, likely 0.97×). Fidelity verdict: not
addressed (the paper measures throughput parity, not numerics); irrelevant
here since it offers no speedup to weigh a fidelity cost against.

### 4. torch.compile / TorchInductor on Windows CPU — viable but strictly dominated here
[PyTorch tutorial: torch.compile on Windows CPU/XPU](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html); [Intel/PyTorch blog on Windows CPU inductor](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Accelerate-PyTorch-Inference-with-torch-compile-on-Windows-CPU/post/1640044).
Core idea: TorchInductor's C++ backend fuses the dozen ATen ops in
`brain_engine.step_inplace` into one (or a few) generated C++ loops with
OpenMP `#pragma omp parallel for`, using an installed C++ compiler (clang or
MSVC) as of PyTorch 2.5+ on Windows. This is mechanically the same fusion
`lif_kernel.c` already did by hand. Expected speedup: plausibly close to the
hand kernel's ~7.9× if Inductor's fusion heuristics happen to merge the whole
step into one loop and it picks up `-march=native`/AVX-512 — **unmeasured
here, and moot**, because the hand kernel already exists, is faster than
Inductor's typical fused-elementwise-loop output tends to be for 15-op
chains with FMA-sensitive lines (Inductor does not expose per-line FMA
contraction control), and is already gated bit-identical. The one thing
Inductor could add that the hand kernel doesn't have yet is the OpenMP
`parallel for` — but that is the exact same threading gap ISPC or a hand
pool would close, with less numeric control. Fidelity verdict: **not
bit-identical by default** — Inductor's default C++ codegen does not
guarantee it won't contract a `g = g*c + t`-shaped fusion into an FMA (no
public control finer than the compiler's own `-ffp-contract` default), so
landing this would require re-doing the exact per-line FMA audit `lif_kernel.c`
already did, against code you don't hand-write and that can silently change
between PyTorch versions.

### 5. Persistent spin-wait thread pool over 4 physical cores — not a "compiler" technique, but the actual gap the field points at
General HPC practice; no single citable paper, arithmetic shown below. This
is the concrete instantiation of what ISPC's `tasks` or Inductor's `omp
parallel for` would give automatically, done by hand for full control.
Core idea: the fused sweep in `sweep_avx512` has **zero cross-neuron
dependency** — every neuron's next state depends only on its own `v,g,refrac`
and the (already-known, delay-decoupled) delayed input for that neuron. Any
contiguous partition of `[0,N)` across 4 worker threads computes exactly the
same per-neuron floating-point operations, in the same order per neuron, as
the current single-thread call — **splitting the range changes nothing
about any individual neuron's arithmetic**, so it is exact by construction,
not merely "probably fine." (The two places order *does* matter — the
scatter-add fan-out and the ascending-index spike list — are already
serialized/measured at 0.14% of runtime and untouched by this change.)
Arithmetic: 0.219 ms/step ÷ 4 cores ≈ 0.055 ms compute time; even at only
60% parallel efficiency (uneven chunking, shared-L3 contention, AVX-512
frequency licensing on Tiger Lake) that is ≈0.091 ms — under the 0.1 ms
target on compute alone, with no margin for dispatch overhead. Fidelity
verdict: **exact / bit-identical**, given the sweep has no reduction and no
carried state across neurons — this is the strongest-fidelity item in this
whole note.

## Ruled out from this field, and why

| Technique | Why not |
|---|---|
| **Halide** | Scheduling language (`.vectorize()`, `.parallel()`, explicit `fma()` builtin) gives the same per-line FMA control as raw C, at the cost of learning a second DSL for a 15-line kernel that already exists and is verified. No published evidence it beats hand intrinsics for this shape (same 97–100%-of-hand-code ceiling as ISPC/Parsimony); its default LLVM codegen must be explicitly kept off fast-math (`set_fast_fp_math()` is opt-in, not default), which is good, but "opt-in strictness" is exactly what `-ffp-contract=off` already gives clang for free. |
| **TVM / TIR** | Built for tensor-algebra graphs (matmul/conv fusion, autoscheduling), not a 15-line scalar-per-element kernel with bitset output and mixed int/float control. TIR *can* express scalar loops and bit ops, but there is no reported use case matching "branchless per-neuron LIF sweep with explicit bitset packing," and the tooling overhead (Relay/TIR graph construction, autoscheduler search) for a kernel this small and already-optimal has no payoff. |
| **numba** | Not installed on the target machine (confirmed in the task's own hardware inventory). `fastmath=False` (the default) does avoid reassociation and is the right instinct for this problem, but numba's fastmath flag is set **per JIT function**, not per line — the kernel needs FMA on the membrane update and explicitly *not* on the conductance decay in the same function, which is exactly the granularity numba's public API doesn't expose (you would still be dropping to hand-inserted intrinsics inside the numba function to get it, at which point numba adds nothing over ctypes-calling the existing DLL). No upside over the already-verified C kernel. |
| **oneAPI/DPC++ on the Iris Xe iGPU** | The iGPU shares the same LPDDR4x pool (~68 GB/s) the CPU already uses, and Xe-LP's own architecture doc gives 128 B/cycle to its L2/memory — no bandwidth win over a working set (2.2 MB) that already fits in 12 MB of *shared L3* on the CPU side. The real cost is kernel-launch/synchronization latency over Level Zero, which the roadmap already flags (§7 risk 6, "GPU at small active sets... measure before porting") — at a 100 µs/step real-time budget, launch+sync latency alone is a plausible budget-killer, and there is no published number establishing it is safely sub-microsecond on this iGPU for a 138k-element dispatch every 0.1 ms. Not ruled out on principle, but unmeasured and lower-confidence than the CPU threading path that needs no new toolchain. |
| **GeNN (adopt the framework, not just its ISPC idea)** | Would mean re-deriving this project's entire model inside GeNN's code-generation DSL to get its ISPC/CUDA backends "for free," discarding the hand-verified bit-identity work already done against `BrainEngine`/Brian2. The ISPC backend blog gives no bit-identity evidence (see item 2) — adopting GeNN buys, at best, parity with the kernel already in hand, at the cost of the whole verification chain. |
| **Brian2 `cpp_standalone`** | This is the **reference implementation**, not a target — it's what `compare_ground_truth.py` is meant to validate against (HANDOFF §4, §10 item 7). It generates Jinja2-templated C++ calling the GNU Scientific Library's ODE solvers, not a fused vectorized sweep, and its own measured cost (Wang et al. 2025, Table already in HANDOFF §8: 4419 s/sim-second) confirms it is the slow baseline everything here is 100–500× faster than, not a codegen technique to imitate. |
| **MSVC 2022 as the AVX-512 compiler** | Microsoft's own blog on AVX-512 auto-vectorization in MSVC exists, but historically MSVC's autovectorizer has lagged clang/gcc for AVX-512 and the repo already standardized on clang 21 for `lif_kernel.c`; no reason to introduce a second, less-controllable toolchain for the one file where per-instruction FP contraction must be exact. Keep MSVC for the rest of the build, clang for this file, as already configured. |

## Concrete next actions for this codebase

1. **Add a 4-way persistent thread pool to `flyloop/native/lif_kernel.c` / `flyloop/native_engine.py`.** Add `lif_step_range(lo, hi, ...)` that calls the existing `sweep()` on a sub-range (it already takes `lo, hi` — the AVX-512 path just needs its 64-neuron alignment logic checked at arbitrary chunk boundaries, not only `[0, N)`). In `native_engine.py`, spin up 3 persistent OS threads at `NativeBrainEngine.__init__` (main thread does the 4th chunk), each pinned to a physical core (`SetThreadAffinityMask` on Windows) and parked on a spinning atomic step counter rather than a condition variable. Dispatch: bump the counter, each worker computes its `[lo,hi)` chunk of the sweep, main thread spins on a completion counter before proceeding to the (already serial, 0.14%-of-runtime) delay/fan-out step. Bit-identical by construction (see item 5 above) — still run `verify_native.py` to confirm, since it costs nothing and this is exactly the kind of change Rule 1 (HANDOFF §9) exists for.
2. **Measure OS thread-wake latency on this exact machine before trusting the design in (1).** Microbenchmark: park 3 threads on (a) a `std::condition_variable`/Windows `WaitForSingleObject` and (b) a busy-spin on `std::atomic`, and time 10,000 wake-compute-rejoin cycles of each against a 0.1 ms/step budget. If (a) consumes more than ~20% of the 100 µs budget (`Monitor`-style measurement, not the researched-elsewhere numbers this note declines to cite without a source), switch to (b); busy-spinning idle cores is the standard fix in this latency regime and costs nothing extra given the target already assumes 4 dedicated cores.
3. **Build a second implementation of `sweep_avx512` in ISPC** (`--target=avx512skx-i32x16 --opt=disable-fma`, one explicit `fma()` call on the membrane-update line only, matching the audited rounding in `lif_kernel.c` lines 109–128) purely as a cross-check and to get its `tasks`/`launch` construct essentially for free instead of hand-rolling (1) and (2). Budget ~1 day; compare generated assembly and wall-clock against the existing kernel; only adopt if `verify_native.py` passes bit-identical **and** it beats the hand thread pool — per items 2–3 above, expect parity at best, not a win.
4. **Do not port to DPC++/the Iris Xe iGPU, adopt GeNN, or reach for TVM/Halide this week.** All three require new toolchains or new frameworks for a kernel whose vectorization is already within a few percent of the achievable ceiling (item 3's 97% figure); the entire remaining 2.2× gap is a threading problem, not a codegen problem, per §0 of `HANDOFF.md` and item 5 above.
5. **If (1)–(3) leave a residual gap**, profile the single-thread sweep with VTune (already the right tool for Tiger Lake) to check whether the measured 0.219 ms is throughput-bound or latency-bound: 0.219 ms over 138,639 neurons at 16 lanes/iteration is ≈8,665 SIMD iterations, i.e. ≈25.3 ns ≈ 76 cycles per 16-neuron iteration at 3.0 GHz — several times the ~10–15 cycle dependency-chain latency the `load → sub/mul → fmadd → cmp → blend → store` chain should need if fully pipelined, suggesting store-forwarding or unaligned-`loadu`/`storeu` stalls rather than an FMA-throughput ceiling. Try 64-byte-aligned allocation (replace the numpy-owned `v/g/refrac` buffers with `_aligned_malloc`'d ones passed into the DLL) and a manual 2-wide software-pipelined unroll of the 4-chunk-of-16 inner loop before assuming threading is the only lever — this is classic intrinsics tuning, not a new compiler, and is cheap to try in the same file.

## References

- [Pharr & Mark, "ispc: A SPMD Compiler for High-Performance CPU Programming," InPar/HPCC 2012](https://pharr.org/matt/assets/ispc.pdf) — the SPMD-to-SIMD compilation model and its parity with hand intrinsics.
- [ISPC User's Guide](https://ispc.github.io/ispc.html) and [ISPC FAQ](https://ispc.github.io/faq.html) — `--opt=disable-fma`, `foreach`, `tasks`/`launch`.
- [genn-team.github.io, "Developing an ISPC Backend for GeNN"](https://genn-team.github.io/posts/developing-an-ispc-backend-for-genn-bridging-gpu-and-cpu-performance-for-neural-network-simulations.html) — measured 1.4–9.49× over GeNN's single-threaded CPU backend for LIF-class models; no bit-identity claim.
- [Yavuz, Turner & Nowotny, "GeNN: a code generation framework for accelerated brain simulations," *Scientific Reports* 6:18854 (2016)](https://www.nature.com/articles/srep18854) — the underlying GeNN codegen framework.
- [Parsimony: Enabling SIMD/Vector Programming in Standard Compiler Flows, CGO 2023](https://dl.acm.org/doi/10.1145/3579990.3580019) — 97% of hand AVX-512 intrinsics from an independently-built SPMD compiler.
- [PyTorch: "How to use torch.compile on Windows CPU/XPU"](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html) — TorchInductor Windows CPU support, OpenMP-parallel C++ codegen, compiler prerequisites.
- [Intel/PyTorch community blog, "Accelerate PyTorch Inference with torch.compile on Windows CPU"](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Accelerate-PyTorch-Inference-with-torch-compile-on-Windows-CPU/post/1640044) — corroborating Windows CPU inductor details.
- [Numba Performance Tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html) — `fastmath` semantics, per-function granularity, reassociation risk.
- [Intel oneAPI: Intel Xe GPU Architecture doc](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html) — Xe-LP 96 EU, L2 size, 128 B/cycle bandwidth figures used in the iGPU rule-out.
- [Wang et al. 2025, "Neuromorphic Simulation of *Drosophila* on Loihi 2," arXiv:2508.16792](https://arxiv.org/abs/2508.16792) — already in `HANDOFF.md`; Brian 2 reference-implementation wall-clock (4419 s/sim-second) cited here to justify ruling out `cpp_standalone` as a performance target.
- [Brian 2 code generation developer docs](https://brian2.readthedocs.io/en/stable/developer/codegen.html) — confirms the Jinja2-template/GSL-solver architecture of `cpp_standalone`, contrasted with the fused hand kernel here.

## Reviewer notes

**Citation audit (2026-07-30):** All top-5 citations verified as legitimate and accurately characterized:
- **Pharr & Mark (2012):** Paper confirmed published in Innovative Parallel Computing, May 2012. PDF exists and is citable as ISPC's foundational work on SPMD-to-SIMD compilation.
- **GeNN ISPC backend blog (2025):** Post verified; reports exact speedups claimed: 1.4× sparse/AVX2, 3.05–3.1× dense/AVX2, 9.49× dense/AVX-512 on Xeon Gold 6134.
- **Yavuz, Turner & Nowotny (2016):** Confirmed in *Scientific Reports* 6:18854; foundational GeNN code-generation framework paper.
- **Parsimony (CGO 2023):** Confirmed published in Proceedings of CGO 2023; reports 97% of hand-written AVX-512 intrinsics on 70+ benchmarks — exact match to document's central parity claim.
- **PyTorch torch.compile Windows:** Tutorial confirmed current and accurate; covers TorchInductor C++ backend, OpenMP parallelism, MSVC/clang compiler setup, PyTorch 2.5+ Windows CPU support.

**Arithmetic verification:** All three independent calculations cross-checked and correct:
1. Parallel-speedup scaling (line 127–129): 0.219 ms ÷ 4 cores = 0.055 ms; 60% efficiency → 0.091 ms. ✓
2. Latency budget (line 150): 20% of 100 µs = 20 µs. ✓
3. Cycles-per-iteration profile (line 153): 138,639 neurons ÷ 16 = 8,665 iterations; 0.219 ms ÷ 8,665 = 25.3 ns = 76 cycles @ 3.0 GHz. ✓

**No corrections required.** Document is well-sourced and internally consistent. Characterizations of cited works are precise and properly contextualized (e.g., GeNN ISPC blog's lack of ULP-level analysis is correctly noted as a limitation for bit-identity claims).
