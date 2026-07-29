"""Prove the optimized run_pytorch.py path is numerically identical to the
original dense path, using the repo's own TorchModel.

Runs both paths from the same seed and requires bit-identical spike trains
(timestep + neuron index) across several stimulation regimes -- not just the
default sugar experiment.

Usage:  python code/verify_pytorch_perf.py [steps]
"""
import sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

import run_pytorch as rp
from benchmark import path_comp, path_con, EXPERIMENTS

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400


def spike_signature(event_driven, exc_ids, rate, steps, seed=1234):
    flyid2i, _ = rp.get_hash_tables(str(path_comp))
    exc = [flyid2i[n] for n in exc_ids if n in flyid2i]
    W = rp.get_weights(str(path_con), str(path_comp), str(path_con.parent), csr=True)

    model = rp.TorchModel(1, W.shape[0], rp.DT, rp.MODEL_PARAMS, W,
                          exc_indices=exc, device='cpu')
    model.event_driven = event_driven
    state = model.state_init()

    rates = torch.zeros(1, W.shape[0])
    if exc:
        rates[:, exc] = rate

    gen = torch.Generator(device='cpu'); gen.manual_seed(seed)
    times, ids = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for t in range(steps):
            state = model(rates, *state, generator=gen)
            nz = (state[2][0] > 0).nonzero(as_tuple=True)[0]
            if nz.numel():
                ids.append(nz.numpy())
                times.append(np.full(nz.numel(), t))
    elapsed = time.perf_counter() - t0
    if ids:
        return elapsed, np.concatenate(times), np.concatenate(ids)
    return elapsed, np.array([]), np.array([])


REGIMES = [
    ("sugar GRNs (21 @200Hz)", EXPERIMENTS['sugar']['neu_exc'], 200.0),
    ("P9 walking (2 @100Hz)", EXPERIMENTS['p9']['neu_exc'], 100.0),
    ("single neuron (1 @200Hz)", EXPERIMENTS['sugar']['neu_exc'][:1], 200.0),
    ("no drive (0 Hz)", EXPERIMENTS['sugar']['neu_exc'], 0.0),
]

print(f"verifying optimized vs original TorchModel over {STEPS} steps\n")
print(f"{'regime':<26} {'spikes':>7} {'dense s':>9} {'event s':>9} {'speedup':>8}  match")
print("-" * 74)
all_ok = True
for name, ids_, rate in REGIMES:
    d_t, d_ts, d_id = spike_signature(False, ids_, rate, STEPS)
    e_t, e_ts, e_id = spike_signature(True, ids_, rate, STEPS)
    ok = np.array_equal(d_ts, e_ts) and np.array_equal(d_id, e_id)
    all_ok &= ok
    print(f"{name:<26} {len(d_ts):>7d} {d_t:>9.3f} {e_t:>9.3f} "
          f"{d_t/e_t:>7.2f}x  {'IDENTICAL' if ok else 'MISMATCH'}")

print("-" * 74)
print("ALL IDENTICAL" if all_ok else "FAILURES PRESENT")
sys.exit(0 if all_ok else 1)
