"""Where does the step actually go, per regime?

    .venv\\Scripts\\python.exe flyloop\\profile_split.py

HANDOFF section 15 item 2 says fan-out dominates the broad and saturating
regimes and proposes Beamer's push/pull direction switch as the fix. Before
building that, this measures the split -- because the arithmetic for the pull
direction looks unfavourable and a negative result here saves days.

The two halves are timed separately by calling the kernel's two exported
entry points independently: `nrn_step` (the neuron sweep plus the delayed pass)
and `lif_fanout` (the event-driven scatter). Both are driven from a state
captured mid-run, so the measurement is of the regime it claims to be.
"""
import statistics
import sys
import time
from pathlib import Path

import ctypes
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "flyloop"))

from brain_engine import BrainEngine            # noqa: E402
from native_engine import NativeBrainEngine     # noqa: E402

DATA = str(ROOT / "data")
SUGAR = [720575940624963786, 720575940630233916, 720575940637568838,
         720575940638202345, 720575940617000768, 720575940630797113,
         720575940632889389, 720575940621754367, 720575940621502051,
         720575940640649691, 720575940639332736, 720575940616885538,
         720575940639198653, 720575940639259967, 720575940617937543,
         720575940632425919, 720575940633143833, 720575940612670570,
         720575940628853239, 720575940629176663, 720575940611875570]


def bench(fn, reps=9, block=200):
    for _ in range(50):
        fn()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(block):
            fn()
        out.append((time.perf_counter() - t0) / block)
    return min(out)


def main():
    torch.set_num_threads(4)
    rng = np.random.default_rng(0)
    pool = BrainEngine(data_dir=DATA, stim_ids=SUGAR[:1], seed=0).i2flyid

    regimes = [("sugar (21)", SUGAR, 200.0),
               ("broad (1000)", rng.choice(pool, 1000, replace=False).tolist(), 100.0),
               ("broad (10000)", rng.choice(pool, 10000, replace=False).tolist(), 100.0),
               ("saturating (40k)", rng.choice(pool, 40000, replace=False).tolist(), 200.0)]

    print("step cost split, whole brain, 4 threads, min of 9 x 200\n")
    hdr = (f"{'regime':<18} {'spikes/step':>11} {'edges/step':>11} "
           f"{'full ms':>9} {'fanout ms':>10} {'fanout %':>9} {'ns/edge':>8}")
    print(hdr)
    print("-" * len(hdr))

    E = None
    for name, ids, rate in regimes:
        e = NativeBrainEngine(data_dir=DATA, stim_ids=ids, seed=1234,
                              threads=4, reorder="cell_type")
        e.inject(rate)
        E = int(e.crow[-1])
        for _ in range(600):                      # reach the regime's steady state
            e.step()

        nsp, edges, n = 0, 0, 400
        for _ in range(n):
            k = e.step()
            nsp += k
            if k:
                s = e.spike_idx
                edges += int((e.crow[s + 1] - e.crow[s]).sum())
        sp_per, ed_per = nsp / n, edges / n

        full = bench(e.step)

        # The fan-out alone, on a spike list of the size this regime produces.
        idx = np.sort(rng.choice(e.N, max(1, int(round(sp_per))),
                                 replace=False)).astype(np.int32)
        p_idx = idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        out_i = np.zeros(e.N, dtype=np.int32)
        out_v = np.zeros(e.N, dtype=np.float32)
        p_oi = out_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        p_ov = out_v.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        def fan():
            e.lib.lif_fanout(e.N, p_idx, idx.size, e._pcrow, e._ppost, e._pval,
                             e._cf["w_scale"], e._pacc, e._ptouch, e._ptbits,
                             p_oi, p_ov, 0, None, None, 0)

        fo = bench(fan)
        ns_edge = fo * 1e9 / max(1.0, ed_per)
        print(f"{name:<18} {sp_per:>11.1f} {ed_per:>11.0f} {full*1e3:>9.4f} "
              f"{fo*1e3:>10.4f} {100*fo/full:>8.1f}% {ns_edge:>8.1f}")

    print("-" * len(hdr))
    print(f"connectome has {E:,} edges in total.")
    print("A PULL fan-out (each thread owns postsynaptic neurons and gathers)")
    print("costs O(E) per step regardless of activity. Compare that column")
    print("against `edges/step`: pull only wins where edges/step approaches E.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
