"""Quantify a MODEL-LEVEL divergence between this repo's backends.

The Brian 2 backend (`run_brian2_cuda.py`, and the designated ground truth in
`compare_ground_truth.py`) declares:

    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau              : volt (unless refractory)
    Synapses(..., on_pre='g += w', delay=t_dly)

so during a neuron's 2.2 ms refractory period **g is frozen** and arriving
synaptic input **still accumulates** into it. When refractoriness ends the
neuron carries the whole accumulated charge.

The PyTorch backend (`run_pytorch.py`, `flyloop/brain_engine.py`) instead does:

    gate  = (refrac >= refrac_steps)
    g_new = g * decay + delayed * gate

so during refractoriness **g keeps decaying** and arriving input is **dropped
entirely** -- not deferred, discarded. Over 22 steps the retained charge decays
by 0.98^22 ~ 0.64 and every spike that lands in the window is lost.

These are different models, not different numerical schemes. A neuron in the
PyTorch backend is effectively deaf for 2.2 ms after each of its own spikes; the
same neuron in Brian 2 integrates through the window and is then driven by the
accumulated charge.

This script measures the size of the effect directly: the fraction of arriving
synaptic weight that the PyTorch gate discards, as a function of firing rate.
It is a lower bound on the divergence, since dropped charge also changes which
neurons spike next, which compounds.

Run:  .venv\\Scripts\\python.exe code\\measure_refractory_divergence.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "flyloop"))

from brain_engine import BrainEngine   # noqa: E402

SUGAR = [720575940624963786, 720575940630233916, 720575940637568838,
         720575940638202345, 720575940617000768, 720575940630797113,
         720575940632889389, 720575940621754367, 720575940621502051,
         720575940640649691, 720575940639332736, 720575940616885538,
         720575940639198653, 720575940639259967, 720575940617937543,
         720575940632425919, 720575940633143833, 720575940612670570,
         720575940628853239, 720575940629176663, 720575940611875570]

DATA = str(ROOT / "data")
STEPS = 800


def measure(stim_ids, rate, steps=STEPS):
    """Fraction of arriving synaptic weight discarded by the refractory gate."""
    e = BrainEngine(data_dir=DATA, stim_ids=stim_ids, seed=1234)
    e.active_mode = False
    e.inplace = True
    e.inject(rate)

    tot_w = drop_w = 0.0
    tot_ev = drop_ev = 0
    n_spikes = 0
    for _ in range(steps):
        # inspect the slot the engine is about to consume, with the gate the
        # engine will apply to it
        delayed = e.buf[e.head]
        refrac_next = torch.where(e.spikes > 0,
                                  torch.zeros_like(e.refrac), e.refrac + 1)
        gate = (refrac_next >= e.refrac_steps)

        nz = delayed != 0
        aw = delayed.abs()
        tot_w += float(aw[nz].sum())
        drop_w += float(aw[nz & ~gate].sum())
        tot_ev += int(nz.sum())
        drop_ev += int((nz & ~gate).sum())

        n_spikes += int((e.step() > 0).sum())

    return dict(
        spikes=n_spikes,
        rate_hz=n_spikes / (steps * 1e-4) / e.N,
        drop_w=100.0 * drop_w / tot_w if tot_w else 0.0,
        drop_ev=100.0 * drop_ev / tot_ev if tot_ev else 0.0,
        events=tot_ev,
    )


def main():
    torch.set_num_threads(4)
    pool = BrainEngine(data_dir=DATA, stim_ids=SUGAR[:1], seed=0).i2flyid
    rng = np.random.default_rng(0)

    regimes = [("sugar GRNs (21)", SUGAR, 200.0)]
    for n in (100, 1000, 10000):
        regimes.append((f"broad ({n})",
                        rng.choice(pool, n, replace=False).tolist(), 100.0))
    regimes.append(("saturating (40k @200Hz)",
                    rng.choice(pool, 40000, replace=False).tolist(), 200.0))

    print("How much arriving synaptic input does the PyTorch refractory gate")
    print("discard, that the Brian 2 ground truth would have kept?")
    print()
    print(f"{'regime':<24} {'spikes':>8} {'pop rate Hz':>12} {'arrivals':>11} "
          f"{'dropped %w':>11} {'dropped %ev':>12}")
    print("-" * 82)
    for name, ids, rate in regimes:
        r = measure(ids, rate)
        print(f"{name:<24} {r['spikes']:>8d} {r['rate_hz']:>12.3f} "
              f"{r['events']:>11d} {r['drop_w']:>10.3f}% {r['drop_ev']:>11.3f}%")
    print("-" * 82)
    print("dropped %w  = share of arriving |weight| discarded")
    print("dropped %ev = share of arriving nonzero events discarded")


if __name__ == "__main__":
    main()
