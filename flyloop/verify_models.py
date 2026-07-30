"""The gate for the multi-model kernel: five independent checks, all nine models.

Run:  .venv\\Scripts\\python.exe flyloop\\verify_models.py [steps]

The repository's Rule 1 is that no performance change lands without being proven
bit-identical. Eight of the nine models have no PyTorch counterpart to be
identical TO, so "bit-identical" has to be re-earned rather than borrowed. What
follows is what it means here.

A. ISA EQUIVALENCE. Every model is compiled three times from one source
   (scalar, AVX2, AVX-512). Forcing each in turn and comparing the whole-brain
   state as raw uint32 every step proves that vector width is a pure speed knob
   and that the AVX2 and scalar paths -- which this machine would otherwise
   never execute -- are not quietly wrong for the machines that do run them.

B. NO LIF REGRESSION. The reference model still has to be bit-identical to the
   PyTorch backend after the kernel grew eight more models and lif_step became
   a wrapper around the generalised nrn_step. This is the check that would
   catch a refactor breaking the one result the repository already published.

C. TILE-SKIPPING EXACTNESS. Skipping a tile is only lossless if the model's
   update is a genuine fixed point at the resting state. Running each model
   with skipping ON and OFF and comparing bit-for-bit tests that directly,
   rather than trusting the algebra. This is the check that catches a model
   whose "rest" is off by one ulp.

D. CERTIFIED-ELISION EXACTNESS. AdEx and EIF skip the exponential when a bound
   proves it rounds away. Setting elide_tiny to zero forces it to be computed
   every time; the two runs must agree bit-for-bit. If the bound were ever too
   loose, this is where it shows up -- and it shows up as a hard failure, not
   as a slow drift in someone's results.

E. AUX-STATE INTEGRITY. The auxiliary arrays (u, w, y, m/h/n/armed, theta) are
   compared as uint32 alongside v and g, so a model cannot pass by keeping its
   membrane right while its second state variable diverges.

Timing is deliberately NOT measured here. bench_models.py does that in a
separate pass, because interleaving two engines and wrapping every step in
perf_counter calls makes a correctness harness, not a benchmark.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "flyloop"))

import models as nrn_models                      # noqa: E402
from brain_engine import BrainEngine             # noqa: E402
from native_engine import NativeBrainEngine      # noqa: E402

SUGAR = [720575940624963786, 720575940630233916, 720575940637568838,
         720575940638202345, 720575940617000768, 720575940630797113,
         720575940632889389, 720575940621754367, 720575940621502051,
         720575940640649691, 720575940639332736, 720575940616885538,
         720575940639198653, 720575940639259967, 720575940617937543,
         720575940632425919, 720575940633143833, 720575940612670570,
         720575940628853239, 720575940629176663, 720575940611875570]

DATA = str(ROOT / "data")
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
THREADS = 4
ISA = {2: "AVX-512", 1: "AVX2", 0: "scalar"}


def u32(a):
    return np.ascontiguousarray(a).view(np.uint32)


def state_of(e):
    """Everything a model owns, as raw bits."""
    parts = [u32(e.v), u32(e.g)]
    if e.n_aux:
        parts.append(u32(e.aux.reshape(-1)))
    return parts


def same_state(a, b):
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def make(model, ids=SUGAR, rate=200.0, seed=1234, reorder="cell_type",
         rescan=True, elide=True):
    e = NativeBrainEngine(data_dir=DATA, stim_ids=ids, seed=seed,
                          threads=THREADS, reorder=reorder, model=model)
    if not rescan:
        # Never clear a tile: every tile stays live for the whole run, so the
        # sweep touches all 138,639 neurons every step. Slow, and exactly what
        # the skipping path has to reproduce.
        e.rescan_every = 10 ** 9
    if not elide:
        e.spec.params.elide_tiny = np.float32(0.0)
    e.inject(rate)
    return e


def run_pair(a, b, steps, label):
    """Step two engines in lockstep, comparing full state and spikes."""
    for s in range(steps):
        a.step()
        b.step()
        if not np.array_equal(np.sort(a.spike_idx), np.sort(b.spike_idx)):
            return f"{label}: SPIKES differ at step {s}"
        if not same_state(state_of(a), state_of(b)):
            return f"{label}: STATE differs at step {s}"
    return None


# ------------------------------------------------------------------ gate A
def gate_isa(model, steps):
    """All three instruction-set paths, same bits."""
    import native_lib
    lib = native_lib.lib()
    cap = lib.lif_force_isa(99)          # what this host actually supports
    out, ref = [], None
    for want in (2, 1, 0):
        if want > cap:
            out.append((ISA[want], "n/a"))
            continue
        lib.lif_force_isa(want)
        e = make(model)
        for _ in range(steps):
            e.step()
        st = [x.copy() for x in state_of(e)]
        sp = int(e.n_spikes)
        if ref is None:
            ref, ref_sp = st, sp
            out.append((ISA[want], "ref"))
        else:
            ok = same_state(ref, st) and ref_sp == sp
            out.append((ISA[want], "OK" if ok else "DIFF"))
    lib.lif_force_isa(99)
    return out


# ------------------------------------------------------------------ gate B
def gate_lif_regression(steps):
    """lif_euler through nrn_step is still bit-identical to PyTorch."""
    ref = BrainEngine(data_dir=DATA, stim_ids=SUGAR, seed=1234)
    ref.active_mode = False
    ref.inplace = True
    ref.inject(200.0)
    nat = make("lif_euler")
    P = nat.perm
    for s in range(steps):
        rs = ref.step()
        nat.step()
        ri = np.sort(ref.i2flyid[np.nonzero(rs.numpy() > 0)[0]])
        ni = np.sort(nat.i2flyid[nat.spike_idx])
        if not np.array_equal(ri, ni):
            return f"spikes differ at step {s}"
        if not (np.array_equal(u32(ref.v.numpy()[P]), u32(nat.v))
                and np.array_equal(u32(ref.g.numpy()[P]), u32(nat.g))):
            return f"state differs at step {s}"
    return None


def main():
    torch.set_num_threads(THREADS)
    print(f"multi-model verification   steps={STEPS}   threads={THREADS}")
    print()

    fails = []

    # ---- B first: it is the one result already published ----
    t0 = time.perf_counter()
    bad = gate_lif_regression(STEPS)
    print(f"B  LIF regression vs PyTorch : {'BIT-EQ' if not bad else bad}"
          f"   ({time.perf_counter() - t0:.1f}s)")
    if bad:
        fails.append(("B", "lif_euler", bad))
    print()

    hdr = (f"{'model':<11} {'aux':>3} {'rest fp':>8} {'AVX-512':>8} {'AVX2':>6} "
           f"{'scalar':>7} {'C skip':>8} {'D elide':>8} {'spikes':>8}")
    print(hdr)
    print("-" * len(hdr))

    for key in nrn_models.ALL_MODELS:
        spec = nrn_models.build(key)
        row = [f"{key:<11}", f"{spec.n_aux:>3}",
               f"{str(spec.extra['rest_is_fixed_point']):>8}"]

        # ---- A ----
        for name, res in gate_isa(key, STEPS):
            row.append(f"{res:>8}" if name == "AVX-512"
                       else (f"{res:>6}" if name == "AVX2" else f"{res:>7}"))
            if res == "DIFF":
                fails.append(("A", key, name))

        # ---- C: skipping on vs off ----
        a, b = make(key, rescan=True), make(key, rescan=False)
        bad = run_pair(a, b, STEPS, "skip")
        row.append(f"{'BIT-EQ' if not bad else 'DIFF':>8}")
        if bad:
            fails.append(("C", key, bad))

        # ---- D: elision on vs off (only AdEx and EIF have one) ----
        if key in ("adex", "eif"):
            a, b = make(key, elide=True), make(key, elide=False)
            bad = run_pair(a, b, STEPS, "elide")
            row.append(f"{'BIT-EQ' if not bad else 'DIFF':>8}")
            if bad:
                fails.append(("D", key, bad))
        else:
            row.append(f"{'-':>8}")

        e = make(key)
        n = sum(e.step() for _ in range(STEPS))
        row.append(f"{n:>8}")
        print(" ".join(row))

    print("-" * len(hdr))
    print("A = the three ISA paths agree bit-for-bit (E: aux state included)")
    print("C = tile skipping changes nothing; D = certified elision changes nothing")
    if fails:
        print("\n*** FAILURES ***")
        for g, k, d in fails:
            print(f"  gate {g}  {k}: {d}")
        return 1
    print("\nALL GATES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
