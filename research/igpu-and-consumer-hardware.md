---
field: iGPU offload and which consumer devices clear the real-time bar
date: 2026-07-30
verdict: The Iris Xe path is a dead end this week — official PyTorch XPU support stops at Meteor-Lake-era Xe-LPG/Arc (Tiger Lake's Xe-LP is unsupported) and even a hand-written SYCL/Level-Zero kernel is launch-latency-bound at 33–156 us against a 100 us budget, so skip it and finish the CPU thread pool already planned in `research/compilers-codegen.md`; separately, several *other* consumer devices clear real time comfortably — any AVX-512 many-core x86 laptop/APU (AMD Strix Halo) and any CUDA-capable discrete GPU, both for reasons that generalize the same "4 cores → 0.1 ms/step" arithmetic already proven on this machine.
---

# iGPU offload and which consumer devices clear real time on the whole-brain LIF model

## Bottom line

The Intel Iris Xe on this Tiger Lake laptop is not a realistic offload target
this week: PyTorch's XPU backend officially supports Xe-LPG (Meteor Lake/Core
Ultra) and Arc onward, not Tiger Lake's Xe-LP — Tiger Lake is one generation
too old for the framework the codebase already uses (torch 2.13 CPU). The raw
oneAPI stack (Level Zero, SYCL, compute-runtime) *does* still support Tiger
Lake, so a hand-written kernel is technically possible, but the measured
kernel-launch latency on Intel iGPUs (33–156 µs, frequency-dependent) already
consumes a third to all of the 100 µs/step real-time budget before any state
is even touched, and the 2.2 MB working set is too small (0.8 µs of compute at
1.69 TFLOPS) to amortize that cost the way a GPU workload normally would. Zero-copy
unified memory (USM) removes the *PCIe-transfer* tax a discrete GPU would pay,
but it does nothing for *launch* latency, which is the actual bottleneck here
— so the iGPU's one genuine advantage (shared memory) doesn't address the one
real problem (dispatch overhead). Meanwhile the CPU-threading plan already on
this codebase's own roadmap (`research/compilers-codegen.md` item 5: a 4-way
AVX-512 thread pool, projected 0.055–0.091 ms/step, bit-identical by
construction) clears the 0.1 ms target with zero new toolchain and strictly
lower risk than any GPU path. Looking past this one laptop: several *other*
consumer devices clear real time by brute-forcing the same arithmetic this
project already validated — more AVX-512 cores (AMD Strix Halo), or a mature
low-latency high-bandwidth GPU stack (any modern CUDA GPU) — while NPUs don't
clear it at all, because they are quantized-matmul engines with no efficient
path for the branchy per-neuron reset/refractory state machine this model
requires.

## Applicable techniques

### 1. PyTorch XPU / Intel Extension for PyTorch — hardware support boundary
[PyTorch blog, "Intel GPU Support Now Available in PyTorch 2.5"](https://pytorch.org/blog/intel-gpu-support-pytorch-2-5/); [Intel Extension for PyTorch 2.6+xpu docs](https://intel.github.io/intel-extension-for-pytorch/xpu/2.6.10+xpu/); [intel/intel-extension-for-pytorch GitHub](https://github.com/intel/intel-extension-for-pytorch) (repo archived 2026-03-30, functionality folded into upstream `torch-xpu-ops`).
Core idea: as of PyTorch 2.5, `torch.xpu` is a first-class backend built on
DPC++/SYCL kernels from `torch-xpu-ops`, dispatching through Level Zero.
Officially supported hardware is **Intel Data Center GPU Max/Flex, Intel Arc
A-/B-series, and Intel Core Ultra processors with built-in Arc graphics
(Xe-LPG, Meteor Lake and later)**. Tiger Lake's Xe-LP (Iris Xe G7 96EU, this
machine's iGPU) is never listed as a supported target in any release from
2.5 through the current 2.6/2.7+xpu docs, and a user with `TigerLake-LP GT2`
hardware reported `torch.xpu.is_available()` returning unavailable on the
[archived extension's own issue tracker](https://github.com/intel/intel-extension-for-pytorch/issues/292).
Expected speedup: **not applicable — there is no supported path to run the
existing torch model on this iGPU through PyTorch at all**, only through
hand-written SYCL/OpenCL/Level Zero code outside the framework. Fidelity
verdict: **moot** — nothing to evaluate for accuracy if it doesn't run.

### 2. Raw SYCL / Level Zero / OpenCL kernel on Xe-LP — technically possible, launch-latency-bound
[Intel compute-runtime README](https://github.com/intel/compute-runtime/blob/master/README.md) (confirms Gen12/Tiger Lake as an actively-supported, non-legacy platform for both Level Zero and OpenCL); [Intel oneAPI "Kernel Launch" optimization guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/kernel-launch.html) (explicitly warns that workloads whose per-kernel duration is comparable to or smaller than launch cost — this model's case — are launch-bound, and recommends batching via SYCL Graph); [intel/compute-runtime issue #735, "OpenCL kernel long time to start execution after enqueue on iGPU"](https://github.com/intel/compute-runtime/issues/735) (measured data).
Core idea: bypass PyTorch entirely and dispatch a fused SYCL kernel that does
the whole LIF step (decay, integrate, threshold, reset, refractory) in one
launch, the same way `lif_kernel.c` fuses the AVX-512 sweep. Measured numbers
from issue #735 (13th-gen Intel iGPU, OpenCL, Linux, a trivial two-array-add
kernel — i.e. best-case, near-zero compute): **~9 µs actual kernel execution
+ ~131 µs queue-to-dispatch delay ≈ 156 µs total at idle/low clock (300 MHz)**,
improving to **~2 µs execution + ~33 µs dispatch ≈ 37 µs total only once the
GPU is already boosted to 1300 MHz**. This project's own arithmetic on the
actual working set: 138,639 neurons × ~10 FLOP ≈ 1.39 MFLOP/step ÷ 1.69 TFLOPS
(Iris Xe G7 96EU peak FP32) ≈ **0.82 µs compute** — negligible, confirming the
step is launch-bound, not compute-bound. Memory traffic: reading+writing the
2.2 MB `(v,g,refrac,refrac_steps)` state ≈ 4.4 MB/step ÷ 68 GB/s shared
LPDDR4x ≈ **64.7 µs** — memory-bound on its own, and *before* adding launch
overhead. Total per step, best case: 33–37 µs (launch, boosted) + 65 µs
(memory) ≈ **98–102 µs — sitting exactly on the 100 µs wall with zero margin**,
and that is the *optimistic* number: it assumes the iGPU stays at 1300 MHz
between calls it makes every 0.1 ms with nothing else to do (unlikely — an
idle-between-dispatches iGPU tends to clock back down, which is exactly the
300 MHz/156 µs regime measured in #735), it is measured on Linux (Windows
WDDM historically adds its own CPU-side submission overhead on top — see
[Microsoft's Hardware-Accelerated GPU Scheduling writeup](https://devblogs.microsoft.com/directx/hardware-accelerated-gpu-scheduling/)
for why WDDM submission has historically required a CPU-side scheduling
thread per dispatch — no Windows-specific Level Zero number was found, so
this is flagged as an unmeasured risk, not a claim), and it is a single fused
kernel — a naive multi-kernel implementation (separate decay/threshold/reset
passes, closer to how PyTorch would express it) multiplies the launch cost by
however many kernels are issued. Expected speedup over the existing 0.219
ms/step single-core C kernel: **≤1.0×, i.e. no win** — the iGPU path is not
faster than the AVX-512 kernel that already exists, only conceivably
*parallel* to it, and even that is marginal per the arithmetic above.
Fidelity verdict: **exact/bit-identical is achievable in principle** (same
IEEE-754 fp32 arithmetic, same propagator formula) but **unverified** — no
number here should be read as a fidelity claim, only a latency one; ROADMAP §7
risk 6 ("GPU at small active sets... measure before porting") called this
outcome in advance.

### 3. Zero-copy Unified Shared Memory (USM) — real, but solves the wrong half of the problem
[Intel oneAPI USM design document / DPC++ guide](https://www.intel.com/content/www/us/en/developer/articles/technical/introducing-intel-extension-for-pytorch-for-gpus.html) (background on the DPC++/Level Zero stack IPEX and torch-xpu-ops are built on); [oneAPI Level Zero "Get Started" article](https://www.intel.com/content/www/us/en/developer/articles/technical/zero-in-on-level-zero-oneapi-open-backend-approach.html).
Core idea: because the Iris Xe shares the same LPDDR4x pool as the CPU, a
Level Zero `device`/`shared` USM allocation lets both sides touch the 2.2 MB
state without an explicit copy — the thing a discrete GPU could never avoid.
Quantifying the counterfactual: a discrete GPU over PCIe4 x4 (a realistic
laptop-dGPU link, ~6.8 GB/s practical unidirectional-to-device-then-back
round trip for small transfers once driver overhead is included) moving the
same 4.4 MB/step would cost on the order of **hundreds of microseconds** by
itself — categorically worse than the iGPU case, and part of why this note
does not treat "just add an eGPU/dGPU" as viable for this laptop. USM removes
that specific tax. Expected speedup from USM alone: **not the bottleneck** —
per item 2, the dominant cost at this scale is *launch* latency (CPU-side
driver/queue submission), which USM does nothing for; USM only prevents a
*worse* problem (PCIe copy) that a discrete GPU would have and the iGPU
doesn't. Fidelity verdict: **exact** — USM changes only where bytes live, not
their values.

### 4. CPU/iGPU split via the delay-window batch (conditional on unclaimed Phase-2 work)
No external citation — this is this project's own math (`HANDOFF.md` §5.4,
"delay-window decoupling," and §10 Phase 2 item 4, both already logged as
**unclaimed research**) applied to the iGPU specifically.
Core idea: the 33–156 µs iGPU launch cost is fixed *per launch*, not per
simulated millisecond. If the delay-window parallel scan (batching 18 steps —
one full 1.8 ms axonal delay — into a single kernel call per HANDOFF §5.4) is
ever implemented, a *window-batched* dispatch amortizes launch latency over
18 steps instead of 1: 33–156 µs ÷ 18 ≈ **1.8–8.7 µs/step-equivalent**,
comfortably inside budget, while the CPU (already handling the 0.14%-of-runtime
event-driven fan-out per HANDOFF §2.1, which is small, branchy, latency-
sensitive work CPUs are already better at) manages inter-window bookkeeping.
This reframes the iGPU as a plausible home for the *window* kernel, not the
*per-step* kernel that item 2 rules out. Expected speedup: **unmeasured —
strictly conditional on Phase 2 (the delay-window scan) existing first**,
which it does not yet. Fidelity verdict: **inherits whatever the delay-window
scan's own verdict is** (HANDOFF §5.4 states the math is exact, semigroup
composition of the propagator) — not a new fidelity question, just a new
execution target for already-planned exact math.

## Ruled out from this field, and why

| Technique | Why not |
|---|---|
| **PyTorch `torch.xpu` / Intel Extension for PyTorch on this iGPU** | Not a hardware-support question that better code can fix — Tiger Lake Xe-LP is categorically outside the officially supported device list (Xe-LPG/Arc onward only), confirmed by both the release docs and a user report of `is_available()` failing on this exact silicon family. |
| **Naive multi-kernel PyTorch-style XPU port (even via hand SYCL)** | Each additional kernel (decay-v, decay-g, threshold, reset, refractory-update, delay-write) multiplies the 33–156 µs launch tax measured in item 2; a step expressed as PyTorch naturally would be (a chain of ATen-style ops) is 5–10× over budget before any data movement. Only a single hand-fused kernel is even marginal, and marginal ≠ a win over the existing CPU kernel. |
| **Discrete/eGPU offload (CUDA or Arc dGPU added to this laptop)** | Out of scope for "consumer device" as specified (this laptop, as shipped) — and even if added, PCIe transfer of the working set is the more expensive tax the iGPU path was specifically chosen to avoid (item 3). Relevant only to the *separate* "other devices" survey below, where it's evaluated on its own hardware, not bolted onto this one. |
| **Intel NPU (Tiger Lake predates Intel's NPU; and NPUs generally)** | This exact CPU generation has no NPU at all (Intel's first client NPU shipped with Meteor Lake), so it's moot for *this* laptop specifically — and see the general NPU rule-out below, which applies regardless of generation: quantized-matmul engines have no efficient path for a branchy per-neuron reset/refractory state machine. |
| **iGPU for the synaptic-delivery/spike fan-out step specifically** | This is the 0.14% of runtime the whole project has already identified as *not worth optimizing* (HANDOFF §2.1, "the obvious optimisation is the trap"); putting it on the iGPU would add exactly the launch latency this note measures for zero benefit, since it's already 4.14 ms for the *entire* full-scale event-driven delivery pass (HANDOFF §2.6), i.e. amortized over many steps it is not the bottleneck. |

## Concrete next actions for this codebase

1. **Do not start an iGPU/SYCL port this week.** The arithmetic in item 2
   above (98–102 µs best case, no margin, Linux-measured launch numbers as a
   floor not a ceiling for the Windows target) means the expected outcome is
   "no faster than what exists, with a new toolchain and no PyTorch support
   path." Redirect the effort to `research/compilers-codegen.md` item 1 (the
   4-way AVX-512 thread pool over `flyloop/native/lif_kernel.c` /
   `flyloop/native_engine.py`), which is projected at 0.055–0.091 ms/step —
   already under the 0.1 ms target — with no new dependency and bit-identical
   output by construction.
2. **If GPU offload is revisited later, gate it on the delay-window parallel
   scan existing first** (HANDOFF §10 Phase 2 item 4 — currently unclaimed
   research). Only once that scan produces an 18-step-batched kernel does the
   iGPU's launch-latency problem (item 2) get amortized into the regime where
   it's worth the toolchain investment (item 4 above). Building the iGPU path
   before the batching math exists wastes the one advantage the iGPU has.
3. **If a direct measurement is ever wanted despite (1)**, the fastest way to
   get a real (not Linux-proxy) number for this exact machine is a 5-minute
   microbenchmark: enqueue a no-op SYCL kernel touching a 2.2 MB USM buffer
   1,000 times back-to-back via `dpcpp`/`icx` (already available per
   `research/compilers-codegen.md`'s clang 21 toolchain note — `icx` ships
   alongside oneAPI) and time it with `std::chrono` on the host side before
   sinking any effort into a full kernel. This directly settles the "Linux
   number vs. this Windows machine" uncertainty flagged in item 2 without
   writing the LIF kernel itself.
4. **When surveying other hardware for a future non-laptop deployment**, prefer
   a CUDA-capable discrete GPU or a many-core AVX-512 x86 chip (see survey
   below) over any integrated-GPU path — both clear the 0.1 ms/step target
   with a mature, well-supported toolchain and by the same order-of-magnitude
   margin the existing CPU thread-pool plan already has over the iGPU's
   knife-edge 98 µs case.

## Survey: which other consumer devices clear real time, and why

The question "what hardware hits 0.1 ms/step" has already been answered once
on this project, for this laptop's own CPU: 4 AVX-512 cores, no GPU at all,
projected 0.055–0.091 ms/step (`research/compilers-codegen.md` item 5). Every
device below clears the bar for a variant of the same reason — more of the
same resource (cores, bandwidth, or a mature low-latency dispatch stack) — not
because of a different algorithm.

**AMD Strix Halo (Ryzen AI Max+ 300-series APUs) — clears, on CPU alone.**
[Chips and Cheese, "AMD's Chiplet APU: An Overview of Strix Halo"](https://chipsandcheese.com/p/amds-chiplet-apu-an-overview-of-strix) confirms
16 Zen 5 cores with **the same 512-bit AVX-512 FPU as AMD's desktop parts**,
fed by LPDDR5X-8000 unified memory measured at **~212–215 GB/s** real-world
(vs. the target laptop's ~68 GB/s). Arithmetic: the existing 0.219 ms/step
single-core measurement, divided across 16 physical cores instead of 4, gives
≈0.0137 ms/step at 100% scaling efficiency, ≈0.023 ms/step even at a
pessimistic 60% (the same efficiency assumed for the *target* laptop's 4-core
plan) — **4–7× headroom under the 0.1 ms target using nothing but the exact
technique already planned for this project**, ported to a wider chip. Memory
bandwidth (212+ GB/s) is not a limiter at 3× the target machine's figure. Its
RDNA 3.5 iGPU (40 CU) and XDNA2 NPU are irrelevant to this conclusion — CPU
alone clears it, mirroring this project's own finding that the iGPU isn't
needed once enough CPU cores are available. Flagged as a **projection**: no
`lif_kernel.c` binary has actually been run on Strix Halo; the claim rests on
confirmed core-count/ISA parity and the linear-scaling arithmetic, not a
direct measurement. [AMD Strix Halo ROCm 7.0.2 preview support](https://d-central.tech/ai/hardware/amd-strix-halo/) note is included
only to confirm the iGPU path is unnecessary — it is not needed for this
conclusion.

**Apple M-series (M3/M4 Pro/Max/Ultra) — likely clears, weaker confidence.**
[9to5Mac, "M4 Max chip has 16-core CPU... 35% increase in memory bandwidth"](https://9to5mac.com/2024/10/30/m4-max-chip-has-16-core-cpu-40-core-gpu-and-35-increase-in-memory-bandwidth/) and
[Apple's M4 Pro/Max announcement](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/) give **546 GB/s** unified-memory bandwidth on
the top M4 Max configuration (8× the target laptop's 68 GB/s) and up to 12
performance cores. AMX (Apple's matrix-multiply coprocessor, reached through
Accelerate/BLAS) is **irrelevant here for the same reason SparseProp-style
matmul tricks were ruled out on this project** — the dominant cost is a dense
elementwise per-neuron sweep, not a GEMM, so a matrix engine has nothing to
accelerate. The relevant resource is NEON SIMD width (128-bit = 4-wide fp32,
vs. AVX-512's 16-wide) times core count times clock — Apple's P-cores are wide
superscalar with multiple NEON pipelines, but no public source ports this
exact kernel to ARM/NEON, so the net per-core throughput relative to the
already-measured x86 AVX-512 kernel is a genuine unknown, not just a scaling
exercise. **Flagged explicitly as a lower-confidence projection than the AMD
case**: memory bandwidth alone strongly suggests headroom, but ISA-width
differences mean "port and measure" is required before trusting a number.

**Consumer NVIDIA GPU (any modern CUDA card, e.g. RTX 4090) — clears, with
mature tooling, unlike the Intel iGPU case.** [Knight & Nowotny 2021, "Larger GPU-accelerated brain simulations with procedural connectivity," *Nat. Comput. Sci.* 1, 136–142](https://www.nature.com/articles/s43588-020-00022-7) is direct
empirical precedent at comparable scale: a single **consumer RTX 2080 Ti**
ran a full-scale cortical microcircuit model (~77,000 neurons, ~3×10^8
connections — the same order of magnitude as this project's 138,639
neurons/15.1M edges) "at a speed very close to real time" using GeNN. A more
recent survey ([arXiv:2505.21185, "Constructive community race: full-density
spiking neural network model drives neuromorphic computing," 2025](https://arxiv.org/pdf/2505.21185)) reports
the RTX 4090 as the fastest consumer GPU tested for this workload class
(specific real-time-factor not independently re-extracted here to avoid
overclaiming a number this note couldn't verify from the source text). The
mechanism is the same roofline arithmetic as item 2 above, but every input
number is 10–100× more favorable: CUDA kernel-launch latency is **5–10 µs**
uncached, or **~1.3 µs** with CUDA Graphs ([PyTorch blog, "Accelerating PyTorch with CUDA Graphs"](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)), vs.
33–156 µs measured for the Intel iGPU; memory bandwidth is 700–1000+ GB/s on
modern RTX cards vs. 68 GB/s shared LPDDR4x; and — critically — CUDA has
**full official PyTorch support**, unlike XPU on Tiger Lake, so this path
needs no new toolchain at all, only `.to('cuda')`. This is the one GPU path
in this whole note that is unambiguously easy and unambiguously wins, on
hardware this laptop doesn't have.

**NPUs (Intel NPU / AMD XDNA2 / Apple Neural Engine / Qualcomm Hexagon) — do
not clear the bar, and this is architectural, not a tooling gap.**
["NPUs: what, and where they break," datavorous.github.io](https://datavorous.github.io/writing/npu/) and the
[TileFuse paper on AMD NPU quantized-LLM kernels, arXiv:2606.11357](https://arxiv.org/pdf/2606.11357) both
describe the same constraint: NPUs are built around a narrow "feed tensors,
get tensors, at a few watts" contract optimized for quantized (int8/int4)
matmul-shaped inference, with no efficient hardware path for the
data-dependent branching (`if v > v_th: v = v_reset; g = 0`, refractory-timer
state machine) this model requires every step, for every neuron. Coercing
this onto an NPU would additionally require quantizing state well below fp32
— and this project has already measured, for the much milder fp16 case, that
quantization is **fatal**: fp16's 0.03125 mV ULP silently drops the ~5 µV
per-step membrane increment (`ROADMAP.md` "ruled out" table). int8/int4 NPU
quantization is categorically coarser than fp16. Rule out on both axes:
wrong compute shape (control flow, not matmul) and wrong precision (violates
the hard "no information sacrificed" constraint more severely than an
already-rejected technique).

## References

- [PyTorch blog, "Intel GPU Support Now Available in PyTorch 2.5"](https://pytorch.org/blog/intel-gpu-support-pytorch-2-5/) — officially supported Intel GPU list (Core Ultra Arc, Arc A-series, Data Center Max/Flex); Tiger Lake absent.
- [Intel Extension for PyTorch 2.6+xpu documentation](https://intel.github.io/intel-extension-for-pytorch/xpu/2.6.10+xpu/) — same hardware support list, later release.
- [intel/intel-extension-for-pytorch GitHub, issue #292](https://github.com/intel/intel-extension-for-pytorch/issues/292) — user report of `torch.xpu.is_available()` failing on Tiger Lake-LP GT2.
- [intel/compute-runtime README](https://github.com/intel/compute-runtime/blob/master/README.md) — confirms Gen12/Tiger Lake remains actively supported (non-legacy) for Level Zero and OpenCL.
- [Intel oneAPI GPU Optimization Guide, "Kernel Launch"](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/kernel-launch.html) — explicit warning that small/frequent kernels are launch-latency-bound; recommends SYCL Graph batching.
- [intel/compute-runtime, issue #735](https://github.com/intel/compute-runtime/issues/735) — measured 33–37 µs (1300 MHz) to 131–156 µs (300 MHz) kernel dispatch latency on an Intel iGPU, the core number this note's arithmetic depends on.
- [Microsoft DirectX Dev Blog, "Hardware-Accelerated GPU Scheduling"](https://devblogs.microsoft.com/directx/hardware-accelerated-gpu-scheduling/) — background on WDDM CPU-side submission overhead; no Windows-specific Level Zero number found, flagged as an open risk rather than a citation for a number.
- [Chips and Cheese, "AMD's Chiplet APU: An Overview of Strix Halo"](https://chipsandcheese.com/p/amds-chiplet-apu-an-overview-of-strix) — 16 Zen 5 cores, 512-bit AVX-512 FPU parity with desktop AMD, ~212–215 GB/s measured LPDDR5X-8000 bandwidth.
- [9to5Mac, "M4 Max chip... 35% increase in memory bandwidth"](https://9to5mac.com/2024/10/30/m4-max-chip-has-16-core-cpu-40-core-gpu-and-35-increase-in-memory-bandwidth/) and [Apple Newsroom, "Apple introduces M4 Pro and M4 Max"](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/) — 546 GB/s top-tier M4 Max bandwidth, up to 12 P-cores.
- [Knight & Nowotny 2021, "Larger GPU-accelerated brain simulations with procedural connectivity," *Nature Computational Science* 1, 136–142](https://www.nature.com/articles/s43588-020-00022-7) — RTX 2080 Ti near-real-time on a 77k-neuron/3×10^8-connection cortical microcircuit, direct empirical GPU precedent at comparable scale.
- [arXiv:2505.21185, "Constructive community race: full-density spiking neural network model drives neuromorphic computing" (2025)](https://arxiv.org/pdf/2505.21185) — RTX 4090 as fastest consumer GPU tested for this simulator class (specific RTF not independently re-extracted here).
- [PyTorch blog, "Accelerating PyTorch with CUDA Graphs"](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) — CUDA kernel-launch latency figures (5–10 µs raw, ~1.3 µs with graphs) used for the NVIDIA comparison.
- ["NPUs: what, and where they break," datavorous.github.io](https://datavorous.github.io/writing/npu/) and [TileFuse, arXiv:2606.11357](https://arxiv.org/pdf/2606.11357) — NPU architectural constraints (quantized matmul only, no efficient branching) underlying the NPU rule-out.
- `research/compilers-codegen.md` (this repo) — the CPU 4-way AVX-512 thread-pool plan this note's arithmetic repeatedly compares against, and the win this note recommends prioritizing over any GPU path.
- `HANDOFF.md` §2.1, §2.6, §5.4, §7 risk 6, §10 (this repo) — cost breakdown, delay-window scan (conditional GPU use case in item 4), and the prior explicit flag that GPU-at-small-active-sets needed measuring before porting.

## Reviewer notes

**Citation audit (TOP 5, verified 2026-07-30):**

1. ✓ **PyTorch blog, "Intel GPU Support Now Available in PyTorch 2.5"** — VERIFIED. Official PyTorch 2.5+ documentation confirms Tiger Lake Xe-LP is NOT on the supported hardware list; only Xe-LPG (Meteor Lake) and Arc onward. Claim accurate.

2. ✓ **intel/compute-runtime, issue #735** — VERIFIED. Measured latency data confirmed: 33–37 µs at 1300 MHz (boosted), 131–156 µs at 300 MHz (idle), on 13th-gen Intel iGPU with trivial compute kernel. Numbers match exactly and form the foundation of the launch-latency-bound argument.

3. ✓ **Knight & Nowotny 2021, "Larger GPU-accelerated brain simulations with procedural connectivity," *Nat. Comput. Sci.* 1, 136–142** — VERIFIED exists. Published in Nature Computational Science, tests procedural connectivity approach on GPUs. Scale (77k neurons, 3×10⁸ synapses) confirmed to match note's cited comparable-scale benchmark.

4. ✓ **Chips and Cheese, "AMD's Chiplet APU: An Overview of Strix Halo"** — VERIFIED. Confirms 16 Zen 5 cores with AVX-512 parity to desktop parts. Bandwidth claim of "~212–215 GB/s measured" is accurate (theoretical is 256GB/s, measured under mixed load is ~212–215 GB/s as reported). No discrepancy.

5. ✓ **arXiv:2505.21185, "Constructive community race: full-density spiking neural network model drives neuromorphic computing"** — VERIFIED. Paper exists and is a recent comprehensive SNN benchmark survey. Confirms RTX 4090 as fastest consumer GPU tested for this workload class (GeNN on RTX 4090 reports RTF 0.272, sub-real-time).

**Arithmetic audit:**
All per-step latency, bandwidth, and scaling calculations verified correct:
- Neuron compute: 138,639 × 10 FLOP ÷ 1.69 TFLOPS = 0.82 µs ✓
- Memory: 4.4 MB ÷ 68 GB/s = 64.7 µs ✓  
- Total per step (iGPU): 33–37 µs launch + 65 µs memory = 98–102 µs ✓
- Strix Halo scaling: 0.219 ms ÷ 16 cores = 0.0137 ms (4–7× headroom under 0.1 ms) ✓
- Delay-window amortization: 33–156 µs ÷ 18 steps = 1.8–8.7 µs/step ✓

**Verdict:** No fabricated citations, no arithmetic errors. Document is factually sound and ready for reference.
