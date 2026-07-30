"""Full correctness + performance sweep, across every regime and both orderings.

Answers two questions the other scripts do not:

1. Is the tile-skip win GENERAL, or does it only apply to sparse stimulation?
   Tile-skipping is activity-dependent by construction -- a tile can only be
   skipped if all 16 of its neurons are at exact rest -- so this reports the
   live-tile fraction alongside the speedup for every regime, including
   saturating drive where no ordering has any headroom left.

2. Does it ever COST anything? At high activity the tile machinery still pays
   for the per-tile branch and the periodic rescan while skipping nothing. That
   has to be measured, not assumed, because a "faster in the good case, slower
   in the bad case" optimisation is not obviously worth having.

Correctness and timing are measured in SEPARATE passes. verify_native.py
interleaves both engines and wraps every step in two perf_counter calls, which
makes its per-regime timings a correctness harness rather than a benchmark;
quoting them as speedups is how a bogus 0.64x regression got recorded earlier.

Run:  .venv\\Scripts\\python.exe flyloop\\verify_all.py [verify_steps]
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
P9 = [720575940627652358, 720575940635872101]

DATA = str(ROOT / "data")
VSTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
THREADS = 4


def u32(a):
    return a.view(np.uint32)


def live_frac(e):
    # tile_live carries a padding word and its trailing bits are set at init,
    # so the count must be masked to n_tiles or it can exceed 100%.
    bits = np.unpackbits(e.tile_live.view(np.uint8), bitorder="little")
    return 100.0 * int(bits[:e.n_tiles].sum()) / e.n_tiles


def timed(step, block, reps=7, warm=200):
    for _ in range(warm):
        step()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(block):
            step()
        out.append((time.perf_counter() - t0) / block)
    return min(out), statistics.median(out)


def verify(ids, rate, steps):
    """Lockstep bit-comparison. Native indices are permuted, so v/g are compared
    under the permutation and spike trains are matched by FlyWire id."""
    ref = BrainEngine(data_dir=DATA, stim_ids=ids, seed=1234)
    ref.active_mode = False
    ref.inplace = True
    ref.inject(rate)
    nat = NativeBrainEngine(data_dir=DATA, stim_ids=ids, seed=1234,
                            threads=THREADS, reorder="cell_type")
    nat.inject(rate)
    P = nat.perm
    n_sp = 0
    for s in range(steps):
        rs = ref.step()
        nat.step()
        ri = np.sort(ref.i2flyid[np.nonzero(rs.numpy() > 0)[0]])
        ni = np.sort(nat.i2flyid[nat.spike_idx])
        n_sp += len(ri)
        if not np.array_equal(ri, ni):
            return f"SPIKES@{s}", n_sp, nat
        if not (np.array_equal(u32(ref.v.numpy()[P]), u32(nat.v))
                and np.array_equal(u32(ref.g.numpy()[P]), u32(nat.g))):
            return f"STATE@{s}", n_sp, nat
    return None, n_sp, nat


def main():
    torch.set_num_threads(THREADS)
    pool = BrainEngine(data_dir=DATA, stim_ids=SUGAR[:1], seed=0).i2flyid
    rng = np.random.default_rng(0)

    regimes = [("single neuron (1 @200Hz)", SUGAR[:1], 200.0),
               ("P9 walking (2 @100Hz)", P9, 100.0),
               ("sugar GRNs (21 @200Hz)", SUGAR, 200.0),
               ("silent (0 drive)", SUGAR, 0.0)]
    for n in (100, 1000, 10000):
        regimes.append((f"broad ({n} @100Hz)",
                        rng.choice(pool, n, replace=False).tolist(), 100.0))
    regimes.append(("saturating (40k @200Hz)",
                    rng.choice(pool, 40000, replace=False).tolist(), 200.0))

    print(f"verify={VSTEPS} steps   threads={THREADS}   "
          f"isa={NativeBrainEngine(data_dir=DATA, stim_ids=SUGAR[:1], seed=0).isa}")
    print()
    print(f"{'regime':<24} {'correct':>9} {'torch ms':>9} {'nat ms':>8} "
          f"{'nat+ord':>8} {'vs torch':>9} {'ord gain':>9} {'live%':>7} {'rt':>6}")
    print("-" * 100)

    all_ok = True
    for name, ids, rate in regimes:
        bad, n_sp, _ = verify(ids, rate, VSTEPS)
        all_ok &= bad is None

        block = 400 if rate and len(ids) < 20000 else 120
        r = BrainEngine(data_dir=DATA, stim_ids=ids, seed=1234)
        r.active_mode = False
        r.inplace = True
        r.inject(rate)
        t_ref, _ = timed(r.step, block)

        res = {}
        for ro in (None, "cell_type"):
            e = NativeBrainEngine(data_dir=DATA, stim_ids=ids, seed=1234,
                                  threads=THREADS, reorder=ro)
            e.lib.lif_set_threads(THREADS)
            e.inject(rate)
            res[ro] = (timed(e.step, block)[0], live_frac(e))

        t_nat, _ = res[None]
        t_ord, live = res["cell_type"]
        print(f"{name:<24} {(bad or 'BIT-EQ'):>9} {t_ref*1e3:>9.4f} "
              f"{t_nat*1e3:>8.4f} {t_ord*1e3:>8.4f} {t_ref/t_ord:>8.1f}x "
              f"{t_nat/t_ord:>8.2f}x {live:>6.1f}% {0.1/(t_ord*1e3):>5.2f}x")

    print("-" * 100)
    print("ord gain = reordering's own contribution (native vs native, same conditions)")
    print("live%    = tiles that could NOT be skipped; the win is 100 - live%")
    print("rt       = multiples of real time (1.00x = 0.1 ms/step)")
    print("ALL BIT-IDENTICAL" if all_ok else "*** FAILURES ***")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
