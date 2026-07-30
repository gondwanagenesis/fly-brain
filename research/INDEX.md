# Research notes — index

Every note carries a `verdict:` line in its front matter stating the conclusion,
so this index is a map, not a summary. Notes that returned **negative** are
marked, because a refuted optimisation is worth as much as a landed one and is
much easier to accidentally re-attempt.

Last updated 2026-07-31.

---

## Synthesis and corrections

| note | verdict in one line |
|---|---|
| [_SYNTHESIS.md](_SYNTHESIS.md) | Real time is already reached on an idle machine; the work is margin, plus three fidelity defects nobody wrote down. |
| [CORRECTION_benchmark_units.md](CORRECTION_benchmark_units.md) | The Loihi 2 comparison table was wrong by 1000×; the paper's column is milliseconds. Anything derived from the old table is void. |

## Mathematics

| note | verdict |
|---|---|
| [math/provable-skipping.md](math/provable-skipping.md) | The inert predicate and why skipping is lossless rather than approximate. |
| [math/two-state-parallel-scan.md](math/two-state-parallel-scan.md) | (V,G) as 2×2 affine maps; the delay window as the scan block. Unclaimed, unbuilt. |
| **[math/certified-exponential-elision.md](math/certified-exponential-elision.md)** | **NEW.** AdEx/EIF can skip the exponential bit-exactly when a two-instruction bound proves it rounds away. `\|E\| < \|S\|·2⁻²⁵` ⟹ `fl(S+E) == S`. Removes work without removing information. |
| **[math/vectorised-exp-exactness.md](math/vectorised-exp-exactness.md)** | **NEW.** The kernel's `expf` audited against **all 2,237,530,114** float32 in its domain: max 1 ULP, three ISA paths identical. float32 makes exhaustive verification cheap enough that sampling is a choice. |

## Models and integration

| note | verdict |
|---|---|
| **[multi-model-neuron-kernels.md](multi-model-neuron-kernels.md)** | **NEW.** Nine models share one connectome kernel because the expensive machinery belongs to the network. Exact integration is *cheaper* than the Euler it replaces; Rush-Larsen (1978, cardiac) is what makes whole-connectome HH tractable; calibrate on excitability, not amplitude. |

## Performance — what worked

| note | verdict |
|---|---|
| [hpc-simd-threading.md](hpc-simd-threading.md) | Fusion, mask registers, spin barriers. |
| [memory-layout-quantization.md](memory-layout-quantization.md) | Drop `refrac_steps`, pack `refrac` to saturating uint8: 2.22 MB → 1.19 MB, crossing under L2. int16 connectome weights are lossless. fp16 neuron state is not viable. |
| [reorder_measurements.md](reorder_measurements.md) | `cell_type` reordering takes tile-16 from 70.6% live to 28.3% on sugar. |
| [compilers-codegen.md](compilers-codegen.md) | Codegen and dispatch findings. |

## Performance — negative results (do not re-attempt)

| note | verdict |
|---|---|
| **[fanout-locality-negative.md](fanout-locality-negative.md)** | **NEW, NEGATIVE ×2.** Beamer's pull direction loses by **110×** on arithmetic alone. Radix-partitioning the scatter measures **0.47–0.87×** at every size — the direct scatter was never miss-bound, because CSC targets are stored ascending and the prefetcher handles them. Also corrects §2.1's "0.14%" from an op count to a measured 9.7% of runtime. |
| [snn-simulator-prior-art.md](snn-simulator-prior-art.md) | What the established simulators do, and which of their strengths do not apply here. |
| [parallel-des-lookahead.md](parallel-des-lookahead.md) | Minimum delay as causality horizon; the uniform-delay ring is provably optimal, so no priority queue. |
| [activity-structure-frontier.md](activity-structure-frontier.md) | Quiescence is dynamical attenuation, not topological isolation, and it is stimulus-dependent. |

## Hardware and reproducibility

| note | verdict |
|---|---|
| [igpu-and-consumer-hardware.md](igpu-and-consumer-hardware.md) | What consumer silicon can and cannot do for this workload. |
| [numerical-exactness-reproducibility.md](numerical-exactness-reproducibility.md) | Why bit-identity depends on `torch.get_num_threads()`, and what that costs a benchmark. |

## Papers

`papers/` holds fetched PDFs with `.md` scan companions.

| paper | why it is here |
|---|---|
| [papers/wang2025_loihi2_drosophila.md](papers/wang2025_loihi2_drosophila.md) | Sandia's Loihi 2 whole-brain *Drosophila*; the only directly comparable published performance figures. Read the correction note above before quoting its table. |

---

## The runnable evidence

Every claim above has a script, and the scripts are the authority:

```bash
flyloop/verify_all.py 400            # LIF bit-identity, 8 regimes
flyloop/verify_models.py 250         # 9 models: ISA, tile-skip, elision, aux state
flyloop/verify_exp.py                # exhaustive exp audit (--quick for 7 s)
flyloop/bench_models.py 400 7        # per-model whole-brain timings
flyloop/profile_split.py             # where the step goes, per regime
flyloop/bench_fanout.py              # direct vs partitioned scatter
code/validate_models_brian2.py       # every model vs an independent Brian 2 build
code/compare_semantics.py 700        # the refractory model divergence
```

Outputs land in `data/results/` and the ones cited in the documents are tracked.
