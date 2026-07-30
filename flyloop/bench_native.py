"""Wall-clock benchmark for the native kernel.

Reports the MINIMUM and MEDIAN of many repeated blocks rather than a single
mean. For a sub-millisecond kernel on a laptop the mean is dominated by
interference -- turbo/thermal transitions, background processes, Windows
scheduler noise -- all of which can only ever make a block slower. The minimum
is therefore the best estimate of the kernel's actual cost, and the gap between
minimum and median is the honest measure of how noisy the machine was.

Real time for this model is 0.1 ms/step: dt = 0.1 ms means 10,000 steps per
simulated second, so 0.1 ms/step is exactly 1 simulated second per wall second.

Run:  .venv\\Scripts\\python.exe flyloop\\bench_native.py
"""
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "flyloop"))

from brain_engine import BrainEngine          # noqa: E402
from native_engine import NativeBrainEngine   # noqa: E402

SUGAR = [720575940624963786, 720575940630233916, 720575940637568838,
         720575940638202345, 720575940617000768, 720575940630797113,
         720575940632889389, 720575940621754367, 720575940621502051,
         720575940640649691, 720575940639332736, 720575940616885538,
         720575940639198653, 720575940639259967, 720575940617937543,
         720575940632425919, 720575940633143833, 720575940612670570,
         720575940628853239, 720575940629176663, 720575940611875570]

DATA = str(ROOT / "data")
BLOCK = 500          # steps per timed block
REPEATS = 15
WARMUP = 300


def timed(engine, block=BLOCK, repeats=REPEATS):
    for _ in range(WARMUP):
        engine.step()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(block):
            engine.step()
        out.append((time.perf_counter() - t0) / block)
    return min(out), statistics.median(out)


def row(name, mn, md):
    rt = 0.1 / (mn * 1e3)
    print(f"{name:<22} {mn*1e3:>9.4f} {md*1e3:>9.4f} {mn*1e4:>10.2f} {rt:>9.2f}x")


def main():
    torch.set_num_threads(4)
    print(f"block={BLOCK} steps x {REPEATS} repeats, warmup={WARMUP}")
    print(f"{'engine':<22} {'min ms':>9} {'med ms':>9} {'s/sim-sec':>10} "
          f"{'realtime':>10}")
    print("-" * 64)

    ref = BrainEngine(data_dir=DATA, stim_ids=SUGAR, seed=1234)
    ref.active_mode = False
    ref.inplace = True
    ref.inject(200.0)
    row("pytorch dense", *timed(ref))

    act = BrainEngine(data_dir=DATA, stim_ids=SUGAR, seed=1234)
    act.active_mode = True
    act.inplace = True
    act.inject(200.0)
    row("pytorch active-set", *timed(act))

    for th in (1, 2, 4):
        nat = NativeBrainEngine(data_dir=DATA, stim_ids=SUGAR, seed=1234,
                                threads=th)
        nat.inject(200.0)
        nat.lib.lif_set_threads(th)
        row(f"native x{th}", *timed(nat))

    print("-" * 64)
    print("real time = 1.00x = 0.1 ms/step (dt=0.1 ms => 10,000 steps/sim-second)")


if __name__ == "__main__":
    main()
