"""Direct vs radix-partitioned fan-out, and where the crossover actually is.

    .venv\\Scripts\\python.exe flyloop\\bench_fanout.py

The scatter that delivers spikes is an indexed int32 add into a 138,639-entry
accumulator (554 KB). While the touched fraction is small it stays in cache;
once it is large the accesses are random over an array that does not fit, and
every one is a miss. `profile_split.py` measures the consequence: the cost per
edge nearly triples between broad(10000) and the saturating regime even though
the work per edge is identical.

Radix partitioning trades random traffic for sequential traffic -- bucket the
(post, val) pairs by the high bits of `post`, then accumulate a bucket at a
time so the live accumulator slice is 16 KB. It moves MORE bytes and should
therefore lose at small edge counts and win at large ones. This finds the
crossing point rather than assuming one, and the constant it produces is what
`PART_MIN_EDGES` in the kernel is set to.

Exactness is not at stake: integer accumulation is associative and the totals
never round (ANALYSIS.md section 1), so both paths must produce identical
output. This script asserts that at every size.
"""
import ctypes
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "flyloop"))

from native_engine import NativeBrainEngine     # noqa: E402

DATA = str(ROOT / "data")
SUGAR = [720575940624963786, 720575940630233916, 720575940637568838]


def main():
    e = NativeBrainEngine(data_dir=DATA, stim_ids=SUGAR, seed=0,
                          threads=4, reorder="cell_type")
    N, rng = e.N, np.random.default_rng(0)
    deg = (e.crow[1:] - e.crow[:-1])
    mean_deg = float(deg.mean())
    print(f"N={N:,}  edges={int(e.crow[-1]):,}  mean out-degree={mean_deg:.1f}")
    print(f"accumulator = {N*4/1024:.0f} KB\n")

    out_i = np.zeros(N, dtype=np.int32)
    out_v = np.zeros(N, dtype=np.float32)
    p_oi = out_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    p_ov = out_v.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    NUL = ctypes.POINTER(ctypes.c_int32)()

    hdr = (f"{'spikes':>8} {'edges':>10} {'direct us':>10} {'partition us':>13} "
           f"{'speedup':>8} {'ns/edge dir':>12} {'ns/edge part':>13} {'same':>5}")
    print(hdr)
    print("-" * len(hdr))

    best_gain_at = None
    for nsp in (16, 64, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768):
        idx = np.sort(rng.choice(N, nsp, replace=False)).astype(np.int32)
        p_idx = idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        edges = int(deg[idx].sum())
        if edges > e.pair_cap:
            print(f"{nsp:>8} {edges:>10}  exceeds pair_cap, kernel falls back")
            continue

        def run(part):
            return e.lib.lif_fanout(
                N, p_idx, nsp, e._pcrow, e._ppost, e._pval, e._cf["w_scale"],
                e._pacc, e._ptouch, e._ptbits, p_oi, p_ov, 0,
                e._ppairp if part else NUL, e._ppairv if part else NUL,
                e.pair_cap if part else 0)

        # correctness: both paths must give the same compacted result
        n1 = run(False)
        a1 = np.sort(out_i[:n1].copy()), None
        v1 = out_v[:n1].copy()[np.argsort(out_i[:n1])]
        n2 = run(True)
        a2 = np.sort(out_i[:n2].copy())
        v2 = out_v[:n2].copy()[np.argsort(out_i[:n2])]
        same = (n1 == n2 and np.array_equal(a1[0], a2)
                and np.array_equal(v1.view(np.uint32), v2.view(np.uint32)))

        def timed(part, reps=9):
            block = max(1, min(200, 2_000_000 // max(edges, 1)))
            for _ in range(20):
                run(part)
            out = []
            for _ in range(reps):
                t0 = time.perf_counter()
                for _ in range(block):
                    run(part)
                out.append((time.perf_counter() - t0) / block)
            return min(out)

        td, tp = timed(False), timed(True)
        gain = td / tp
        print(f"{nsp:>8} {edges:>10} {td*1e6:>10.2f} {tp*1e6:>13.2f} "
              f"{gain:>7.2f}x {td*1e9/edges:>12.2f} {tp*1e9/edges:>13.2f} "
              f"{'yes' if same else '**NO**':>5}")
        if gain > 1.0 and best_gain_at is None:
            best_gain_at = edges

    print("-" * len(hdr))
    print("`same` compares the compacted (index, value) output bit-for-bit;")
    print("integer accumulation is order-independent so this must hold.")
    if best_gain_at:
        print(f"\ncrossover: partitioning starts winning around "
              f"{best_gain_at:,} edges/step")
    else:
        print("\npartitioning never won at any size tested -- leave it off")
    return 0


if __name__ == "__main__":
    sys.exit(main())
