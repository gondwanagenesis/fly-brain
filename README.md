# Whole-brain *Drosophila* simulation, faster than real time on a laptop

The complete adult fruit-fly brain — **138,639 neurons, 15,091,983 connectome
edges, 54.5M synapses** — simulated at **2.2× real time on four CPU cores with no
GPU**, bit-identical to the reference implementation, with the membrane equation
switchable between **nine neuron models** at runtime.

Built on the [FlyWire](https://flywire.ai/) connectome (v783) and the leaky
integrate-and-fire model of
[Shiu et al. 2024, *Nature*](https://www.nature.com/articles/s41586-024-07763-9).

A fork of [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain),
whose multi-framework benchmark harness is preserved below unchanged.

---

## What this fork adds

Three independent contributions. Every performance claim is gated on
**bit-identical** output; every correctness claim is a reproducible measurement,
not an argument.

### 1. A fused native kernel — 3.1× to 104× over the PyTorch backend

| regime | native ms/step | vs torch | real time |
|---|---|---|---|
| single neuron | 0.0109 | 97.7× | **9.16×** |
| silent | 0.0260 | 104.4× | **3.84×** |
| P9 walking (2) | 0.0262 | 79.5× | **3.82×** |
| **sugar GRNs (21)** | **0.0461** | **52.2×** | **2.17×** |
| broad (100) | 0.0763 | 30.6× | 1.31× |
| broad (1000) | 0.3019 | 10.1× | 0.33× |
| saturating (40k) | 2.2536 | 3.1× | 0.04× |

All eight regimes verified **bit-identical** — full state compared as raw
`uint32` every step, spike trains matched exactly. No regression in any regime.

The starting assumption was that the dense path was memory-bandwidth-bound with
~1.5–2× of headroom. Profiling said otherwise: at 171 µs/step only 57% was the
kernel, 26% was `torch.bernoulli` drawing *twenty-one* random numbers, and 16%
was ctypes glue. It was **framework-bound**, and the headroom was ~30×.

What recovered it: fusing ~12 full-array passes into one; mapping
threshold-and-reset onto AVX-512 mask registers so the spike bitset falls out of
the compare for free; replacing the dense `(19, N)` delay ring (10.5 MB, ~190
non-zeros) with sparse `(index, value)` slots; batching the Poisson draws;
dropping the refractory counter to a gate bitset; and skipping inert 16-neuron
tiles after renumbering by `cell_type`.

One binary, runtime-dispatched across **AVX-512 / AVX2 / scalar**, all three
proven identical. Runs on any x86-64 since ~2013; even the no-SIMD path beats
PyTorch.

### 2. The two backends do not simulate the same model

The sharper result, and the one we would most like the upstream team to check.

During a neuron's refractory period, **Brian 2** — the designated ground truth —
freezes `g` and lets arriving synaptic input **accumulate**. The **PyTorch**
backend keeps decaying `g` and **discards** the input.

| | discarded weight | spikes | active-neuron Jaccard |
|---|---|---|---|
| sugar (21) | **6.40%** | +22.5% | **0.871** |
| broad (1000) | 15.23% | +199% | 0.666 |

Separately, the PyTorch axonal delay is **one timestep longer** than Brian 2's
(20 steps against 19) — a 5.6% longer delay on every synapse, compounding per
hop. Wang et al. flag rounding 1.8 ms to 2.0 ms on Loihi 2 as a fidelity
compromise; the PyTorch backend arrives at 1.9 ms by accident.

**Why this went unnoticed:** Brian 2 against *itself* at two seeds scores Jaccard
**0.850**. The divergence measures 0.871. The existing methodology does not have
the statistical power to detect it — which is a finding about the benchmark, not
just about the backends.

### 3. Nine neuron models over one connectome

The membrane equation is now a runtime switch. The connectome, delay ring,
fan-out, refractory gate, tiling and thread pool are properties of the *network*
and are shared; a model contributes only how `(v, g, aux)` advance, when a spike
is declared, and what the reset does.

| model | reference | integration |
|---|---|---|
| LIF (Euler) | Shiu et al. 2024 — as shipped | first order |
| LIF (exact) | Rotter & Diesmann 1999 | **exact**, and *fewer* ops than Euler |
| Izhikevich | Izhikevich 2003 | first order |
| AdEx | Brette & Gerstner 2005 | Rush–Larsen adaptation |
| EIF | Fourcaud-Trocmé et al. 2003 | first order |
| QIF | Ermentrout & Kopell 1986 | first order |
| Resonate-and-fire | Izhikevich 2001 | **exact** (sub-threshold) |
| Hodgkin–Huxley | Hodgkin & Huxley 1952 | Rush–Larsen gates |
| GLIF | Teeter et al. 2018 | **exact** (both states) |

Each body is written **once** and compiled **three times** (AVX-512 / AVX2 /
scalar), which makes ISA equivalence structural rather than something to
re-audit. Each model is *calibrated* rather than copied out of its paper: one
free gain is solved for numerically so a single synapse moves every model the
same fraction of the way to threshold — otherwise switching models would measure
unit mismatch instead of dynamics.

Two results worth stating because they invert the usual intuition:

- **Izhikevich is the fastest model on the real protocols**, beating LIF, because
  it drives the network less hard (7.5% live tiles on sugar against LIF's 28.5%).
  A "more complex" model running faster is a *dynamics* result, not an
  arithmetic one.
- **Hodgkin–Huxley is the quietest** — 2.6% live tiles — because its potassium
  conductance gives intrinsic adaptation no I&F model here has. Its ~180
  flop/neuron are paid on 2.6% of the brain, landing full HH at 0.42× real time
  rather than the ~0.03× its arithmetic alone predicts.

### FlyBrain Studio — a live 3D interface

```bash
.venv/Scripts/python.exe flyloop/studio.py     # -> http://127.0.0.1:8765
```

All 138,639 neurons at their real FlyWire coordinates, driven by the native
kernel. Every point is a real neuron and lights when that neuron actually spikes
— the render is the simulation's output, not an animation of it. Model switching
costs milliseconds, not a reload. Standard library only, no build step.

---

## Verify it yourself

Nothing here asks to be taken on trust. Each command is the gate that the
corresponding claim had to pass.

```bash
python flyloop/verify_all.py 400          # 8 regimes, bit-identical + timing
python flyloop/verify_models.py 250       # 9 models x 5 gates
python flyloop/verify_exp.py              # exhaustive ULP audit of the vector exp
python code/compare_semantics.py 700      # the refractory divergence, by ablation
python code/compare_to_brian2.py 100      # ground truth, with a noise floor
python code/validate_models_brian2.py     # 9 models vs independent Brian 2
```

The model gates are:

| gate | what it proves |
|---|---|
| **A** ISA equivalence | AVX-512 / AVX2 / scalar bit-identical — vector width is a pure speed knob |
| **B** no regression | LIF still bit-identical to PyTorch after the kernel grew eight models |
| **C** tile-skipping | each model with skipping on and off, compared bit-for-bit |
| **D** elision | the certified exponential elision forced off, identical bits |
| **E** auxiliary state | `u`, `w`, `y`, `m/h/n/armed`, `θ` compared as uint32 too |

The vectorised `expf` is the only place the kernel makes an accuracy *choice*, so
it is audited **exhaustively rather than sampled** — all **2,237,530,114**
representable float32 in its live domain, on all three ISA paths:

```
max 1 ULP    mean 0.0077 ULP    99.23% exactly rounded
```

Against independent Brian 2 implementations, eight of nine models agree to the
float32 floor; Hodgkin–Huxley's residual is shown to be first-order integrator
convergence (error ratio 2.00 per step halving), not a transcription error.

---

## What changed, relative to upstream

**New — the native kernel and its models**

| file | what it is |
|---|---|
| `flyloop/native/nrn_kernel.c` | fused sweep, sparse delay line, event-driven fan-out, tile skipping, thread pool |
| `flyloop/native/sweep_template.h` | the nine model bodies + vectorised `exp`, written once |
| `flyloop/native/simd.h` | scalar / AVX2 / AVX-512 op abstraction |
| `flyloop/native/nrn_params.h` | the parameter block and model enum |
| `flyloop/models.py` | model registry, PSP calibration, resting-state search |
| `flyloop/native_engine.py` | the engine — state arrays, step loop, stimulation |
| `flyloop/native_lib.py` | content-addressed build + ctypes binding |
| `flyloop/studio.py`, `flyloop/studio/` | the live 3D interface |

**New — verification**

| file | what it proves |
|---|---|
| `flyloop/verify_all.py` | 8 regimes, bit-identity + timing, both orderings |
| `flyloop/verify_models.py` | the five model gates above |
| `flyloop/verify_exp.py` | exhaustive ULP audit |
| `code/compare_semantics.py` | the refractory divergence, single-variable ablation |
| `code/compare_to_brian2.py` | ground truth with a seed-to-seed noise floor |
| `code/validate_models_brian2.py` | every model against an independent implementation |
| `code/run_brian2_reference.py` | full 138,639-neuron Brian 2 network, exact or Euler |

**Modified**

| file | change |
|---|---|
| `code/run_pytorch.py` | event-driven fan-out + ring buffer — 3.2–5.2×, bit-identical |
| `flyloop/brain_engine.py` | active-set stepping, state packing, two-way dense/sparse switch |

**Documentation**

| document | for whom |
|---|---|
| [FINDINGS.md](FINDINGS.md) | the upstream team — every result with its caveats |
| [MODELS.md](MODELS.md) | the nine models: architecture, calibration, verification |
| [FORK.md](FORK.md) | full write-up and theory of the changes |
| [HANDOFF.md](HANDOFF.md) | complete working record, **including dead ends** |
| [research/INDEX.md](research/INDEX.md) | literature notes with per-note verdicts |

---

## Honest limits

Kept deliberately prominent, because the numbers above are easy to over-read.

- **Real time holds in the sparse regimes only.** Tile skipping is
  activity-dependent by construction — under broad 1000-neuron drive it is
  0.33× real time, and 0.04× when saturating. The two *published* experiments
  (sugar, P9) are both sparse; artificial broad stimulation is not.
- **Not faster than Loihi 2.** Sandia's 12-chip system reaches 0.0538
  s/simulated-second against our 0.461 — roughly 10× ahead. The claim here is
  real time on a *consumer device*, not beating neuromorphic silicon.
- **Level with GeNN at n=1, behind at n=8.** 0.461 s/sim-s against GeNN's 0.450
  on an RTX 4070 — same class, different machines. This is a **latency** result;
  GPUs still win batched throughput, and we measured that CPU batching cannot
  close it ([FINDINGS.md](FINDINGS.md) §4).
- **The Brian 2 comparison passes, but it is a weak test.** Jaccard 0.931 against
  a 0.850 seed-to-seed noise floor — high enough to hide the divergences in §2.
  Passing it is necessary, not sufficient.
- **Bit-identity is a property of `torch.get_num_threads()`.** ATen's
  `add_(t, alpha=)` is a single-rounding FMA in its vectorised body and a
  separate multiply-then-add in its scalar tail, so reproducibility claims must
  pin the thread count.
- **Timings are minima on a loaded laptop.** Background indexers inflated the
  PyTorch baseline from 1.66 to 4.97 ms/step before we noticed. Ratios are
  stable; absolute seconds are not.

No pull request has been opened upstream. That repository is a *benchmark*;
changing one backend's numbers alters a published comparison, and §2 changes what
the comparison means. Both seemed like conversations to have first.

---

# Upstream documentation

Everything below is the original
[eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain)
documentation for the multi-framework benchmark harness, preserved unchanged.

## Usage

With this computational model, one can manipulate the neural activity of a set of _Drosophila_ neurons.
The output of the model is the spike times and rates of all affected neurons.

Two types of manipulations are currently implemented:
- *Activation*:
Neurons can be activated at a fixed frequency to model optogenetic activation.
This triggers Poisson spiking in the target neurons. 
Two sets of neurons with distinct frequencies can be defined.
- *Silencing*:
In addition to activation, a different set of neurons can be silenced to model optogenetic silencing.
This sets all synaptic connections to and from those neurons to zero.

The entrypoint is [main.py](main.py), which parses CLI arguments and calls
[code/benchmark.py](code/benchmark.py) -- the central orchestrator that dispatches
to framework-specific runners:
[run_brian2_cuda.py](code/run_brian2_cuda.py),
[run_pytorch.py](code/run_pytorch.py),
[run_nestgpu.py](code/run_nestgpu.py), and
[run_genn.py](code/run_genn.py). The optional Brian2GeNN backend lives in
[run_brian2_genn.py](code/run_brian2_genn.py) and uses a separate conda
environment because Brian2GeNN 1.7.0 pins Brian2<2.6 while Brian2CUDA uses
Brian2 2.8.0.

```bash
# Run the 5 main-environment frameworks with default durations (0.1s–1000s)
# and trials (1,4,8,16,32)
python main.py

# Specific durations and trial count
python main.py --t_run 0.1 1 10 --n_run 1

# Single framework
python main.py --nestgpu --t_run 1 --n_run 1
python main.py --genn --t_run 1 --n_run 1
python main.py --brian2genn --t_run 1 --n_run 1

# Combine frameworks
python main.py --brian2-cpu --pytorch --t_run 0.1 1 --n_run 1 4 8 16 32

# Five-round Nature-paper benchmark suite
# Uses the March grid: t_run=(0.1,1,10,100), n_run=(1,4,8,16,32), 5 core backends
python main.py --paper --run-label nature_2026_07

# Add Brian2GeNN as the 6th framework from the brain-fly-brian2genn environment
python main.py --brian2genn --paper --run-label nature_2026_07
```

Results are incrementally saved to `data/benchmark-results.csv` as each
benchmark completes, with separate columns for setup time (loading, compilation)
and simulation time (the always-on cost). For repeated paper runs, the CSV keeps
the original March rows and appends new rows keyed by `run_label` and `round`;
the corresponding spike parquet path is recorded in `spike_path`.

Spike timing exports are written to parquet outside the timed simulation section
so file I/O does not contaminate `sim_time`. GeNN additionally flushes bounded
on-device spike-recording windows during long batched runs; that transfer time
is tracked as result collection rather than simulation time. A labeled paper run
writes partitioned outputs like:

```text
data/results/nature_2026_07/
├── manifest.csv
├── checksums.sha256
├── round_01/
│   ├── brian2cpp_t1.0s_n1.parquet
│   ├── brian2cuda_t1.0s_n1.parquet
│   ├── pytorch_t1.0s_n1.parquet
│   ├── nestgpu_t1.0s_n1.parquet
│   ├── genn_t1.0s_n1.parquet
│   └── brian2genn_t1.0s_n1.parquet
└── round_02/
```

The consolidated publication bundle contains 600 spike parquet files: 20 grid
points for each of six frameworks across five rounds. The `no_io/` subfolder
contains the corresponding one-round, 120-row timing dataset collected with
spike probing and output disabled; it intentionally contains no spike parquet
files.

Each spike parquet has one row per spike. The canonical timing column for new
exports is `time_ms`, with `trial`, `neuron_index`, `flywire_id`, and `exp_name`.
The legacy `t` column is kept for existing analysis scripts.

The full `nature_2026_07` spike parquet bundle is too large for regular Git
tracking, so parquet files are intentionally gitignored. The committed metadata
files are `manifest.csv` and `checksums.sha256`; the full bundle is stored in
Google Drive:

https://drive.google.com/drive/folders/1jiSfb5lNfm9gwP0YyyRz5ATIrDpBAcjs

After downloading the Drive folder into `data/results/nature_2026_07/`, verify
the bundle with:

```bash
cd data/results/nature_2026_07
sha256sum -c checksums.sha256
```

### Ground truth comparison

Brian2 (CPU) serves as the ground truth for neural accuracy: it implements the
canonical LIF model from
[Shiu et al. (Nature 2024)](https://www.nature.com/articles/s41586-024-07763-9),
which achieved 91% prediction accuracy against experimental _Drosophila_ data.
Each backend also saves per-neuron spike trains to `data/results/`, and a
comparison script measures how closely the other backends reproduce Brian2's
output:

```bash
python code/compare_ground_truth.py                  # default: t_run=1s, n_run=1
python code/compare_ground_truth.py --t_run 10 --n_run 4   # longer / averaged
python code/compare_ground_truth.py --run-label nature_2026_07 --round 1
```

This computes active-neuron overlap (Jaccard), per-neuron firing-rate
correlation, and spike-count ratios, and writes structured results to
`data/ground-truth-comparison.json`.

For all-framework pairwise comparisons, including firing-rate parity rows and
spike-time matches within a tolerance window, use:

```bash
python code/compare_spike_outputs.py \
  --run-label nature_2026_07 \
  --round 1 \
  --output-dir data/results/nature_2026_07/comparisons
```

This writes `pairwise_summary.csv`, `pairwise_summary.json`,
`parity_rates.csv`, and `missing_inputs.json`. The pairwise summary has one row
per framework pair and `t_run`/`n_run` combination.

For paper-support parity files comparing one backend against Brian2 CPU across
all five labeled rounds, use:

```bash
python code/compare_backend_to_brian2.py \
  --run-label nature_2026_07 \
  --backend brian2genn \
  --output-dir data/results/nature_2026_07/comparisons
```

This writes `<backend>_vs_brian2_rate_summary.csv/json`,
`<backend>_vs_brian2_rate_parity.csv`, and
`<backend>_vs_brian2_missing_inputs.json`. Add `--include-timing` only for
smaller targeted checks where greedy spike-time matching is scientifically
useful and computationally reasonable.

## Installation

### Conda environment

The `brain-fly` conda environment provides everything needed to run the
**Brian2**, **Brian2CUDA**, **PyTorch**, **NEST GPU**, and **GeNN** backends
(including CUDA-enabled PyTorch and PyGeNN):

```bash
conda env create -f environment.yml
conda activate brain-fly
```

On Ubuntu/WSL, PyGeNN's source build also needs the system `pkg-config` binary
and libffi headers:

```bash
sudo apt-get install -y pkg-config libffi-dev
```

### GeNN

The `--genn` backend uses PyGeNN 5.4.0 with the CUDA backend. It implements
Brian2-style Poisson activation into membrane voltage, delayed sparse recurrent
synapses, GeNN batching for `n_run`, and the same parquet spike schema as the
other benchmark runners.

Large batched GeNN runs cap the on-device spike recording buffer with
`GENN_RECORDING_WINDOW_MAX_SLOTS` (default: `800000`). This preserves full spike
timing exports while avoiding CUDA out-of-memory errors for large
`n_run * t_run` combinations.

If PyGeNN was not installed when the conda environment was created, install it
inside `brain-fly` with:

```bash
export CUDA_PATH=/usr/local/cuda-12.5
export CUDA_HOME=$CUDA_PATH
export PATH=$CUDA_PATH/bin:$PATH
pip install https://github.com/genn-team/genn/archive/refs/tags/5.4.0.zip
```

### Brian2GeNN

The `--brian2genn` backend uses Brian2GeNN 1.7.0 as a Brian2 standalone device
targeting GeNN/CUDA. It is intentionally isolated from the main `brain-fly`
environment because Brian2GeNN pins Brian2<2.6, which conflicts with
Brian2CUDA's Brian2 2.8.0 requirement.

Create the environment with:

```bash
conda env create -f environment-brian2genn.yml
conda activate brain-fly-brian2genn
```

Brian2GeNN 1.7.0 expects GeNN 4.x command-line scripts such as
`genn-buildmodel.sh`. If they are not already installed, place GeNN 4.9.0 at
`~/.local/src/genn-4.9.0` or set `BRIAN2GENN_GENN_PATH`/`GENN_PATH` to your
GeNN 4.x source tree:

```bash
export CUDA_PATH=/usr/local/cuda-12.5
export CUDA_HOME=$CUDA_PATH
export BRIAN2GENN_GENN_PATH=$HOME/.local/src/genn-4.9.0
export GENN_PATH=$BRIAN2GENN_GENN_PATH
export PATH=$GENN_PATH/bin:$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

For scientific comparability, the Brian2GeNN runner exports the same per-spike
parquet schema as the other backends and uses the same upstream Poisson drive.
Brian2GeNN cannot run this model's independent trials as a true GeNN batch in
the way the direct `--genn` backend can, so `n_run>1` is implemented as
independent build/run trials with deterministic per-trial C RNG seeds. The
`sim_time` column records GeNN executable time; `build_time` records the
Brian2GeNN code generation/compilation overhead.

### NEST GPU

NEST GPU requires a separate build from source with a custom neuron model
(`user_m1`). This is only needed if you want to use the `--nestgpu` backend.

**Prerequisites:**

- **NVIDIA CUDA Toolkit** (12.x) — follow the
  [official installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
- **CMake** — `sudo apt install cmake` (or see
  [cmake.org](https://cmake.org/download/)).

**Steps:**

1. Clone NEST GPU:

```bash
git clone https://github.com/nest/nest-gpu
```

2. Copy the custom source files into the NEST GPU tree. You must replace `/path/to/nest-gpu` with your own local path:

```bash
cp scripts/nestgpu_source_files/src/user_m1.{h,cu}    /path/to/nest-gpu/src/
cp scripts/nestgpu_source_files/pythonlib/nestgpu.py   /path/to/nest-gpu/pythonlib/
```

   The patched `nestgpu.py` fixes weight array initialization (lines 2225-2227).

3. Build and install (set `-DCMAKE_CUDA_ARCHITECTURES` to match your GPU, e.g.
   `89` for RTX 4070):

```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DCMAKE_INSTALL_PREFIX=$HOME/.nest-gpu-build \
      /path/to/nest-gpu
make -j$(nproc) && make install
```

For a full setup from a fresh Windows machine (WSL2 + CUDA + Miniconda), see
[scripts/setup_WSL_CUDA.sh](scripts/setup_WSL_CUDA.sh).

----

## Frameworks

| Framework | Backend | Status |
|---|---|---|
| **Brian2** | C++ standalone (multi-core CPU) | ready |
| **Brian2CUDA** | CUDA standalone (GPU) | ready |
| **PyTorch** | CUDA (GPU) | ready |
| **NEST GPU** | CUDA (GPU, custom `user_m1` neuron) | ready |
| **GeNN** | CUDA (GPU, PyGeNN 5.4.0) | ready |
| **Brian2GeNN** | Brian2GeNN 1.7.0 / GeNN CUDA | ready, separate env |

All six frameworks share the same data, model parameters, spike-output schema,
and folder structure. The five main backends run from `brain-fly` plus a
system-level NEST GPU install; Brian2GeNN runs from `brain-fly-brian2genn`
because of its Brian2 version pin.

## Quickstart

```bash
# Create the conda environment (includes CUDA-enabled PyTorch)
conda env create -f environment.yml
conda activate brain-fly

# Run a 1-second benchmark on the five main-environment backends
python main.py --t_run 1 --n_run 1 --no_log_file

# Specific backends (combinable)
python main.py --brian2-cpu                    # Brian2 CPU only
python main.py --brian2cuda-gpu               # Brian2CUDA GPU only
python main.py --pytorch                      # PyTorch only
python main.py --nestgpu                      # NEST GPU only
python main.py --genn                         # GeNN only
python main.py --brian2genn                   # Brian2GeNN only, from brain-fly-brian2genn
python main.py --pytorch --genn               # PyTorch + GeNN

# Full benchmark suite (all durations, n_run=1,4,8,16,32, five main backends)
python main.py

# Nature-paper suite: five main backends, March parameter grid, 5 rounds
python main.py --paper --run-label nature_2026_07

# Brian2GeNN Nature-paper add-on from the separate brain-fly-brian2genn env
python main.py --brian2genn --paper --run-label nature_2026_07
```

### `main.py` options

| Flag | Description |
|---|---|
| *(default)* | Run all: Brian2 (CPU) → Brian2CUDA (GPU) → PyTorch → NEST GPU → GeNN |
| `--brian2-cpu` | Brian2 C++ standalone (CPU) only |
| `--brian2cuda-gpu` | Brian2CUDA (GPU) only |
| `--pytorch` | PyTorch (GPU/CPU) only |
| `--nestgpu` | NEST GPU only |
| `--genn` | GeNN CUDA backend only |
| `--brian2genn` | Brian2GeNN backend only; use the `brain-fly-brian2genn` environment |
| `--t_run` | Simulation duration(s) in seconds, e.g. `--t_run 0.1 1 10` |
| `--n_run` | Number of independent trials, e.g. `--n_run 1 4 8 16 32` |
| `--paper` | Run the paper suite: `t_run=[0.1,1,10,100]`, `n_run=[1,4,8,16,32]`, 5 rounds |
| `--rounds` | Repeat the full selected backend/parameter suite N times |
| `--round-start` | First round number to write, useful for resuming a labeled run |
| `--run-label` | Group repeated spike outputs under `data/results/<label>/` and append labeled CSV rows |
| `--log_file FILE` | Write log to file (default: `data/results/benchmarks.log`) |
| `--no_log_file` | Console output only |

Backend flags are combinable: `--brian2-cpu --pytorch` runs Brian2 CPU then PyTorch.

## Project structure

```
fly-brain/
├── main.py                     # Entrypoint (benchmark runner CLI)
├── environment.yml             # Conda env definition (brain-fly)
├── environment-brian2genn.yml  # Separate Brian2GeNN env definition
├── code/
│   ├── benchmark.py            # Orchestrator: config, logging, dispatcher
│   ├── run_brian2_cuda.py      # Brian2 / Brian2CUDA benchmark runner
│   ├── run_pytorch.py          # PyTorch benchmark runner (model + utils)
│   ├── run_nestgpu.py          # NEST GPU benchmark runner (subprocess per trial)
│   ├── run_genn.py             # GeNN/PyGeNN benchmark runner
│   ├── compare_ground_truth.py # Compare backends against Brian2 (CPU) ground truth
│   └── paper-brian2/           # Original paper code (not used by benchmarks)
│       ├── model.py            # Core LIF network model (Brian2)
│       ├── utils.py            # Analysis helpers (load_exps, get_rate)
│       ├── example.ipynb       # Tutorial: activation, silencing, rate analysis
│       └── figures.ipynb       # Reproduce paper figures (uses archive 630 data)
├── data/
│   ├── 2025_Completeness_783.csv       # Neuron list (FlyWire v783)
│   ├── 2025_Connectivity_783.parquet   # Synapse connectivity (FlyWire v783)
│   ├── benchmark-results.csv           # Accumulated benchmark timings
│   ├── ground-truth-comparison.json   # Backend accuracy vs Brian2 (CPU)
│   ├── sez_neurons.pickle              # SEZ neuron subset (for figures)
│   ├── weight_coo.pkl                  # Cached sparse weights COO (gitignored)
│   ├── weight_csr.pkl                  # Cached sparse weights CSR (gitignored)
│   ├── archive/
│   │   ├── 2023_Completeness_630.csv   # Legacy v630 data
│   │   └── 2023_Connectivity_630.parquet
└── scripts/
    └── setup_WSL_CUDA.sh       # WSL2 + CUDA + Miniconda setup
```

## Data

The model uses FlyWire connectome data version **783** (public release).
Legacy version 630 data is kept in `data/archive/` for paper figure reproduction.

| File | Description | Size |
|---|---|---|
| `2025_Completeness_783.csv` | Neuron IDs and metadata | 3.2 MB |
| `2025_Connectivity_783.parquet` | Pre/post-synaptic indices + weights | 97 MB |
| `weight_coo.pkl` | Sparse weight matrix (COO), auto-generated by PyTorch | ~288 MB |
| `weight_csr.pkl` | Sparse weight matrix (CSR), auto-generated by PyTorch | ~289 MB |

## Architecture per framework

| | Brian2 / Brian2CUDA | PyTorch | NEST GPU |
|---|---|---|---|
| Build step | C++ / CUDA codegen + compile | None (eager mode) | None |
| Trial parallelism | Sequential (`device.run`) | Batched (`batch_size=n_run`) | Subprocess per trial (cannot reset in-process) |
| Weight format | Brian2 `Synapses` object | Sparse CSR tensor | Array-based `Connect` |
| Neuron model | Brian2 equations | Custom `nn.Module` classes | Custom CUDA kernel (`user_m1`) |
| Timestep | 0.1 ms | 0.1 ms | 0.1 ms |

## System requirements

- Linux (tested on Ubuntu 22.04 under WSL2 on Windows 11)
- NVIDIA GPU with CUDA 12.x (tested on RTX 4070)
- Miniconda / Anaconda
- NEST GPU compiled from source (for `--nestgpu` backend)
- `scripts/setup_WSL_CUDA.sh` documents the full setup from a fresh Windows machine
