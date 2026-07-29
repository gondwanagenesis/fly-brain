"""Bit-identity gate for the native kernel.

Rule 1 of this project: every performance change must be proven bit-identical.
This gate is stronger than ``verify.py`` -- instead of comparing spike trains
only, it compares the ENTIRE floating-point state (v, g, refrac) bit-for-bit at
EVERY step, so a divergence is caught on the step it first appears rather than
whenever it happens to flip a threshold crossing.

That distinction earns its cost. A kernel can emit identical spikes for
thousands of steps while its state drifts by ULPs, then diverge under a
different stimulus. Comparing raw bit patterns removes the possibility: if the
states are equal as uint32 at every step, the two implementations are the same
function of the same inputs.

Note what "bit-identical to PyTorch" turned out to mean here. ATen vectorises
`add_(t, alpha=)` with a fused multiply-add but finishes each thread-chunk with
a scalar tail that is *not* fused, so the reference integrates the last
(chunk_len mod 16) neurons of every chunk with a different rounding from the
rest. The kernel reproduces that partition deliberately; see lif_kernel.c. The
corollary is that the reference's bit pattern depends on torch.get_num_threads(),
so this gate pins the thread count.

Run:  .venv\\Scripts\\python.exe flyloop\\verify_native.py [steps]
"""
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
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1200

# The reference's low bits depend on its thread count (see module docstring),
# so the gate fixes it rather than inheriting whatever the machine defaults to.
THREADS = 4


def u32(a):
    return a.view(np.uint32)


def regimes(pool, rng):
    yield "sugar GRNs (21 @200Hz)", SUGAR, 200.0
    yield "P9 walking (2 @100Hz)", P9, 100.0
    yield "single neuron (1 @200Hz)", SUGAR[:1], 200.0
    yield "silent (0 drive)", SUGAR, 0.0
    for n in (100, 1000, 10000):
        yield (f"broad ({n} @100Hz)",
               rng.choice(pool, n, replace=False).tolist(), 100.0)
    yield ("saturating (40k @200Hz)",
           rng.choice(pool, 40000, replace=False).tolist(), 200.0)


def main():
    torch.set_num_threads(THREADS)
    pool = BrainEngine(data_dir=DATA, stim_ids=SUGAR[:1], seed=0).i2flyid
    rng = np.random.default_rng(0)

    print(f"steps={STEPS}  torch threads={torch.get_num_threads()}")
    print(f"{'regime':<26} {'spikes':>8} {'torch ms':>9} {'native ms':>10} "
          f"{'speedup':>8}  {'state':<10} spikes")
    print("-" * 90)

    all_ok = True
    tot_r = tot_n = 0.0
    for name, ids, rate in regimes(pool, rng):
        ref = BrainEngine(data_dir=DATA, stim_ids=ids, seed=1234)
        ref.active_mode = False          # compare against the dense reference
        ref.inplace = True
        ref.inject(rate)

        nat = NativeBrainEngine(data_dir=DATA, stim_ids=ids, seed=1234)
        nat.inject(rate)

        first_bad = None
        if not (np.array_equal(u32(ref.v.numpy()), u32(nat.v))
                and np.array_equal(u32(ref.g.numpy()), u32(nat.g))
                and np.array_equal(u32(ref.refrac.numpy()), u32(nat.refrac))
                and np.array_equal(u32(ref.refrac_steps.numpy()),
                                   u32(nat.refrac_steps))):
            first_bad = -1

        n_sp = 0
        sp_ok = True
        t_ref = t_nat = 0.0
        for s in range(STEPS):
            t0 = time.perf_counter()
            r_sp = ref.step()
            t_ref += time.perf_counter() - t0

            t0 = time.perf_counter()
            nat.step()
            t_nat += time.perf_counter() - t0

            ri = np.nonzero(r_sp.numpy() > 0)[0]
            ni = np.sort(nat.spike_idx)
            n_sp += len(ri)
            if len(ri) != len(ni) or not np.array_equal(ri, ni):
                sp_ok = False

            if first_bad is None and not (
                np.array_equal(u32(ref.v.numpy()), u32(nat.v))
                and np.array_equal(u32(ref.g.numpy()), u32(nat.g))
                and np.array_equal(u32(ref.refrac.numpy()), u32(nat.refrac))
            ):
                first_bad = s

        st_ok = first_bad is None
        all_ok &= (sp_ok and st_ok)
        tot_r += t_ref
        tot_n += t_nat

        print(f"{name:<26} {n_sp:>8d} {1e3*t_ref/STEPS:>9.4f} "
              f"{1e3*t_nat/STEPS:>10.4f} {t_ref/t_nat:>7.2f}x  "
              f"{'BIT-EQUAL' if st_ok else f'DIFF@{first_bad}':<10} "
              f"{'EXACT' if sp_ok else 'MISMATCH'}")

    print("-" * 90)
    print(f"{'TOTAL':<26} {'':>8} {1e3*tot_r/STEPS:>9.4f} "
          f"{1e3*tot_n/STEPS:>10.4f} {tot_r/tot_n:>7.2f}x  "
          f"{'ALL BIT-IDENTICAL' if all_ok else 'FAILURES'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
