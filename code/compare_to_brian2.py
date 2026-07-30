"""Ground-truth comparison: native kernel vs Brian 2 on the full brain.

This closes the gate the rest of the work deliberately left open. Everything
else in this fork is verified BIT-IDENTICAL to the repo's PyTorch backend --
which is a strong statement about the port, and no statement at all about
whether the PyTorch backend matches the published Brian 2 model. It does not
(see FINDINGS.md sections 1 and 2), so this measures the size of the gap.

Spike-for-spike comparison is IMPOSSIBLE across these backends: Brian 2's
Poisson stream and torch's Generator are different PRNGs, so even a perfect
reimplementation would emit different spikes from the same seed. The comparison
is therefore statistical, using the same metrics as the repo's own
compare_ground_truth.py:

  - Jaccard overlap of the ACTIVE-NEURON SET (its headline metric)
  - Pearson r of per-neuron spike counts over neurons active in both
  - spike-count ratio

Because the Poisson drive is stochastic, a chunk of the observed difference is
just trial-to-trial variability. To separate that from real model divergence
the script also runs Brian 2 against ITSELF at two seeds, which is the noise
floor: any backend difference smaller than that is not evidence of anything.

Run:  .venv\\Scripts\\python.exe code\\compare_to_brian2.py [duration_ms]
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "flyloop"))
sys.path.insert(0, str(ROOT / "code"))

DUR_MS = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
DATA = str(ROOT / "data")
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

SUGAR = [720575940624963786, 720575940630233916, 720575940637568838,
         720575940638202345, 720575940617000768, 720575940630797113,
         720575940632889389, 720575940621754367, 720575940621502051,
         720575940640649691, 720575940639332736, 720575940616885538,
         720575940639198653, 720575940639259967, 720575940617937543,
         720575940632425919, 720575940633143833, 720575940612670570,
         720575940628853239, 720575940629176663, 720575940611875570]


def counts_from_df(df, n_neurons_by_id):
    """Per-flywire-id spike counts as a dict."""
    if not len(df):
        return {}
    return df.groupby("flywire_id").size().to_dict()


def compare(a, b, label_a, label_b):
    ka, kb = set(a), set(b)
    inter, union = len(ka & kb), len(ka | kb)
    jac = inter / union if union else 1.0
    both = sorted(ka & kb)
    if len(both) > 2:
        va = np.array([a[k] for k in both], float)
        vb = np.array([b[k] for k in both], float)
        r = float(np.corrcoef(va, vb)[0, 1])
    else:
        r = float("nan")
    na, nb = sum(a.values()), sum(b.values())
    print(f"{label_a:<26} vs {label_b:<26} "
          f"spikes {na:>7d}/{nb:>7d} ratio {nb/max(na,1):>5.2f}  "
          f"active {len(ka):>5d}/{len(kb):>5d}  Jaccard {jac:>6.3f}  r {r:>6.3f}")
    return jac, r


def run_brian2(seed, method, tag):
    out = ROOT / "data" / "results" / f"brian2_reference_{tag}.parquet"
    cmd = [PY, str(ROOT / "code" / "run_brian2_reference.py"),
           "--duration-ms", str(DUR_MS), "--method", method,
           "--seed", str(seed), "--tag", tag, "--codegen-target", "numpy"]
    print(f"  running Brian 2 ({method}, seed {seed}) ... ", end="", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if p.returncode != 0:
        print("FAILED")
        print(p.stdout[-2500:])
        print(p.stderr[-2500:])
        sys.exit(1)
    print("done")
    return pd.read_parquet(out)


def run_native(seed):
    from native_engine import NativeBrainEngine
    e = NativeBrainEngine(data_dir=DATA, stim_ids=SUGAR, seed=seed,
                          threads=4, reorder="cell_type")
    e.inject(200.0)
    for _ in range(int(round(DUR_MS / 0.1))):
        e.step(record=True)
    return e.spikes_dataframe()


def main():
    print(f"Ground-truth comparison, sugar protocol, {DUR_MS:.0f} ms simulated")
    print()

    b_exact = counts_from_df(run_brian2(0, "exact", "gt_exact_s0"), None)
    b_seed1 = counts_from_df(run_brian2(1, "exact", "gt_exact_s1"), None)
    b_euler = counts_from_df(run_brian2(0, "euler", "gt_euler_s0"), None)
    nat = counts_from_df(run_native(1234), None)

    print()
    print("NOISE FLOOR -- same backend, same model, different Poisson seed.")
    print("Nothing below this line is evidence of a backend difference.")
    print("-" * 118)
    jac_noise, r_noise = compare(b_exact, b_seed1,
                                 "brian2 exact seed 0", "brian2 exact seed 1")

    print()
    print("MODEL / BACKEND DIFFERENCES")
    print("-" * 118)
    compare(b_exact, b_euler, "brian2 exact", "brian2 euler")
    jac_nat, r_nat = compare(b_exact, nat, "brian2 exact", "native kernel")

    print()
    print("-" * 118)
    print(f"noise floor (seed-to-seed):  Jaccard {jac_noise:.3f}   r {r_noise:.3f}")
    print(f"native vs brian2 exact:      Jaccard {jac_nat:.3f}   r {r_nat:.3f}")
    if jac_nat >= jac_noise - 0.02:
        print("=> native is within the stochastic noise floor of Brian 2.")
    else:
        print("=> native differs from Brian 2 by MORE than seed-to-seed noise.")
        print("   Expected: the native kernel is bit-identical to the PyTorch")
        print("   backend, and that backend (a) drops synaptic input during")
        print("   refractoriness where Brian 2 accumulates it, (b) integrates")
        print("   with forward Euler, and (c) delays spikes one timestep longer.")
        print("   See FINDINGS.md sections 1-2. This number is the size of that gap,")
        print("   not a defect introduced by this fork.")


if __name__ == "__main__":
    main()
