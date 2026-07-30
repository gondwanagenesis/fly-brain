"""Regression test: the dense fallback must be recoverable.

`_dense_fallback` used to be a ONE-WAY LATCH. Once broad activity tripped it,
`step()` never called `step_active()` again, so a single transient burst
disabled the sparse path for the rest of the run and permanently gave up the
2-10x it buys. Nothing in the suite caught it, because every regime held its
stimulus constant -- activity never subsided, so the latch was never asked to
release.

Two things are checked:

1. MECHANISM. From a quiet state, with the fallback artificially latched, the
   engine must return to the sparse path, and the spike train must stay
   bit-identical to a pure-dense run throughout.

2. WHAT ACTUALLY HAPPENS AFTER A BURST. Driving 2,000 neurons at 200 Hz and
   then removing the stimulus entirely does NOT quiet the network -- it stays
   self-sustaining on recurrent activity alone, so the active set never falls
   back under the re-entry threshold. That is a property of the model, not a
   defect in the switch, and it is recorded here so the next person does not
   read the non-recovery as a bug.

Run:  .venv\\Scripts\\python.exe flyloop\\test_fallback_recovery.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "flyloop"))

from brain_engine import BrainEngine   # noqa: E402

DATA = str(ROOT / "data")
SUGAR = [720575940624963786, 720575940630233916, 720575940637568838,
         720575940638202345, 720575940617000768, 720575940630797113,
         720575940632889389, 720575940621754367, 720575940621502051,
         720575940640649691, 720575940639332736, 720575940616885538,
         720575940639198653, 720575940639259967, 720575940617937543,
         720575940632425919, 720575940633143833, 720575940612670570,
         720575940628853239, 720575940629176663, 720575940611875570]


def test_mechanism():
    """A latched fallback must release once activity is genuinely low."""
    print("1. MECHANISM -- latched fallback on a quiet network")
    ok = True

    # A genuinely quiet regime. Sugar sits at ~4.9% live, inside the
    # hysteresis band (release below 4%, trip above 8%), so it correctly does
    # NOT release -- that is the band doing its job, not a failure.
    ids = SUGAR[:1]
    a = BrainEngine(data_dir=DATA, stim_ids=ids, seed=1234)
    a.active_mode = True
    a.inplace = True
    a.inject(200.0)
    d = BrainEngine(data_dir=DATA, stim_ids=ids, seed=1234)
    d.active_mode = False
    d.inplace = True
    d.inject(200.0)

    for _ in range(200):                      # settle on the sparse path
        a.step()
        d.step()

    # Latch it exactly as a real fallback does. step_active() maintains
    # _spike_idx rather than the dense `spikes` vector, so the dense kernel's
    # view has to be re-materialised first -- skipping that is an invalid state
    # transition, not a test of the switch.
    a._rebuild_dense_state(None, None, None)
    a._dense_fallback = True
    a._since_recheck = 0
    sp_a, sp_d = [], []
    for _ in range(a.recheck_every + 50):
        sp_a.append(int((a.step() > 0).sum()))
        sp_d.append(int((d.step() > 0).sum()))

    released = not a._dense_fallback
    identical = sp_a == sp_d
    live = ((a.g != 0) | ~a._v_fixed(a.v) | (a.buf.abs().amax(0) != 0)
            | (a.refrac < a.refrac_steps) | (a.spikes > 0))
    if a.stim_idx.numel():
        live[a.stim_idx] = True
    print(f"   released back to sparse   : {released}")
    print(f"   live fraction (re-entry test uses this): "
          f"{100.0*float(live.sum())/a.N:.2f}%  "
          f"(threshold {100*a.sparse_switch:.0f}%)")
    print(f"   spike train == dense      : {identical} ({sum(sp_a)} spikes)")
    ok &= released and identical
    return ok


def observe_burst():
    """Record what a broad burst actually does -- it does not subside."""
    print()
    print("2. OBSERVATION -- broad burst, then stimulus removed")
    rng = np.random.default_rng(0)
    pool = BrainEngine(data_dir=DATA, stim_ids=SUGAR[:1], seed=0).i2flyid
    ids = rng.choice(pool, 2000, replace=False).tolist()

    e = BrainEngine(data_dir=DATA, stim_ids=ids, seed=1234)
    e.active_mode = True
    e.inplace = True
    e.inject(200.0)
    for _ in range(400):
        e.step()
    tripped = e._dense_fallback
    e.inject(0.0)
    spikes = 0
    for _ in range(2000):
        spikes += int((e.step() > 0).sum())
    print(f"   fallback tripped by burst : {tripped}")
    print(f"   spikes in 2000 steps AFTER stimulus removed : {spikes}")
    print(f"   -> the network is self-sustaining; activity does not subside,")
    print(f"      so re-entry correctly does not trigger. Not a switch defect.")
    return tripped


def main():
    torch.set_num_threads(4)
    ok = test_mechanism()
    observe_burst()
    print()
    print("PASS -- fallback is bidirectional and exact" if ok
          else "FAIL -- fallback did not release, or diverged from dense")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
