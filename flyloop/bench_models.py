"""Per-model whole-brain benchmark.

    .venv\\Scripts\\python.exe flyloop\\bench_models.py [block] [reps]

Correctness lives in verify_models.py and is measured in a SEPARATE pass. That
separation is not fussiness: verify interleaves two engines and wraps every step
in perf_counter calls, so quoting its per-regime timings as speedups is how a
bogus regression got recorded earlier in this project's history.

WHAT TO READ OUT OF THIS
------------------------
Real time for this model is 0.1 ms/step (dt = 0.1 ms, so 10,000 steps per
simulated second). The `rt` column is multiples of that: 1.00x means the
simulated fly brain keeps up with the real one.

`live%` is the fraction of 16-neuron tiles that could NOT be skipped. It is the
single number that explains most of the variation between rows, because tile
skipping is activity-dependent by construction -- a tile is skippable only when
all sixteen of its neurons sit at the model's exact resting fixed point. Two
models at the same live% and different speeds differ in ARITHMETIC; two models
at the same speed and different live% differ in DYNAMICS.

Timings are minima over repeats. Load can only make a block slower, so the
minimum is the honest estimate of what the machine can do, and every derived
"faster than real time" claim is conservative. Watch for background indexers:
Syncthing re-indexing this repository inflated the PyTorch baseline from 1.66 to
4.97 ms/step before anyone noticed.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "flyloop"))

import models as nrn_models                     # noqa: E402
from brain_engine import BrainEngine            # noqa: E402
from native_engine import NativeBrainEngine     # noqa: E402

SUGAR = [720575940624963786, 720575940630233916, 720575940637568838,
         720575940638202345, 720575940617000768, 720575940630797113,
         720575940632889389, 720575940621754367, 720575940621502051,
         720575940640649691, 720575940639332736, 720575940616885538,
         720575940639198653, 720575940639259967, 720575940617937543,
         720575940632425919, 720575940633143833, 720575940612670570,
         720575940628853239, 720575940629176663, 720575940611875570]
P9 = [720575940627652358, 720575940635872101]

DATA = str(ROOT / "data")
THREADS = 4
BLOCK = int(sys.argv[1]) if len(sys.argv) > 1 else 400
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 7
WARM = 300


def timed(step, block=BLOCK, reps=REPS, warm=WARM):
    for _ in range(warm):
        step()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(block):
            step()
        out.append((time.perf_counter() - t0) / block)
    return min(out), statistics.median(out)


def live_frac(e):
    # tile_live carries a padding word whose trailing bits are set at init, so
    # the count must be masked to n_tiles or it can exceed 100%.
    bits = np.unpackbits(e.tile_live.view(np.uint8), bitorder="little")
    return 100.0 * int(bits[:e.n_tiles].sum()) / e.n_tiles


def main():
    torch.set_num_threads(THREADS)
    rng = np.random.default_rng(0)
    pool = BrainEngine(data_dir=DATA, stim_ids=SUGAR[:1], seed=0).i2flyid

    regimes = [
        ("silent", SUGAR, 0.0),
        ("P9 (2)", P9, 100.0),
        ("sugar (21)", SUGAR, 200.0),
        ("broad (1000)", rng.choice(pool, 1000, replace=False).tolist(), 100.0),
    ]

    print(f"per-model whole-brain benchmark   N=138,639   threads={THREADS}   "
          f"min of {REPS} x {BLOCK} steps")
    print(f"real time = 0.1000 ms/step\n")

    # The PyTorch baseline, for the one model that has a counterpart there.
    base = {}
    for name, ids, rate in regimes:
        r = BrainEngine(data_dir=DATA, stim_ids=ids, seed=1234)
        r.active_mode = False
        r.inplace = True
        r.inject(rate)
        base[name] = timed(r.step, min(BLOCK, 200), 5, 100)[0]
    print("PyTorch dense baseline (ms/step): "
          + "  ".join(f"{k} {v*1e3:.3f}" for k, v in base.items()) + "\n")

    hdr = f"{'model':<11} {'aux':>3} " + " ".join(
        f"{n:>21}" for n, _, _ in regimes)
    print(hdr)
    print(f"{'':<11} {'':>3} " + " ".join(
        f"{'ms/step   rt  live%':>21}" for _ in regimes))
    print("-" * len(hdr))

    results = {}
    for key in nrn_models.ALL_MODELS:
        spec = nrn_models.build(key)
        row = f"{key:<11} {spec.n_aux:>3} "
        results[key] = {}
        for name, ids, rate in regimes:
            e = NativeBrainEngine(data_dir=DATA, stim_ids=ids, seed=1234,
                                  threads=THREADS, reorder="cell_type",
                                  model=key)
            e.inject(rate)
            ms, med = timed(e.step)
            lf = live_frac(e)
            rt = 0.1 / (ms * 1e3)
            row += f" {ms*1e3:>8.4f} {rt:>5.2f}x {lf:>5.1f}%"
            results[key][name] = {"ms_per_step": ms * 1e3, "realtime": rt,
                                  "live_pct": lf, "median_ms": med * 1e3}
        print(row)

    print("-" * len(hdr))
    print("rt    = multiples of real time (1.00x = 0.1 ms/step)")
    print("live% = tiles that could NOT be skipped; the win is 100 - live%")

    out = ROOT / "data" / "results" / "model_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"threads": THREADS, "block": BLOCK, "reps": REPS,
         "pytorch_baseline_ms": {k: v * 1e3 for k, v in base.items()},
         "models": results}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
