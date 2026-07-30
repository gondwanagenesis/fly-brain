"""Does the refractory-delivery divergence change the science?

`measure_refractory_divergence.py` shows the PyTorch backend discards 6-15% of
arriving synaptic weight that the Brian 2 ground truth keeps. That is a model
difference, but a model difference only matters if it moves the observable --
the spike trains. This measures that.

SINGLE-VARIABLE DESIGN. Both arms are the PyTorch backend, from identical
initial state and identical RNG stream. The ONLY difference is one term:

    A (as shipped)   g_new = g*decay + delayed * gate     <- input dropped
                                                             while refractory
    B (input kept)   g_new = g*decay + delayed            <- input delivered

Everything else -- integration scheme, precision, ordering, Poisson draws,
fan-out -- is byte-for-byte the same, so any divergence is attributable to this
term alone and to nothing else.

B is NOT a full Brian 2 reimplementation. Brian 2 additionally freezes dg/dt and
dv/dt during refractoriness, so it would retain even more charge than B does.
B is therefore a LOWER BOUND on the true A-vs-Brian-2 gap: it removes the
discard but keeps PyTorch's decay-through-refractoriness.

Cross-backend spike-for-spike comparison against Brian 2 itself is impossible
(different PRNG streams), which is exactly why this is done as a controlled
single-variable ablation inside one backend instead.

Run:  .venv\\Scripts\\python.exe code\\compare_semantics.py [steps]
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
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 800


@torch.no_grad()
def run(stim_ids, rate, mode, steps):
    """Dense PyTorch step under one of three conductance semantics.

    mode='shipped' : g_new = g*decay + delayed*gate   (input dropped)
    mode='deliver' : g_new = g*decay + delayed        (input kept, still decays)
    mode='brian2'  : g_new = (g + delayed) if refractory else (g*decay + delayed)
                     -- Brian 2's actual semantics: `(unless refractory)` freezes
                     dg/dt, while `on_pre` still fires, because Brian 2 applies
                     the refractory clamp to DIFFERENTIAL EQUATIONS ONLY, not to
                     synaptic on_pre statements.
    """
    e = BrainEngine(data_dir=DATA, stim_ids=stim_ids, seed=1234)
    e.active_mode = False
    e.inplace = True
    e.inject(rate)

    p, dt, N = e.p, e.dt, e.N
    wS = p["wScale"]
    counts = torch.zeros(N)
    first = torch.full((N,), float("nan"))
    total = 0

    for s in range(steps):
        vstim = torch.zeros(N)
        if e.stim_idx.numel():
            ps = torch.bernoulli(e._rates * (dt / 1000.0), generator=e.gen)
            vstim.index_copy_(0, e.stim_idx, ps * (p["scalePoisson"] * wS))

        rec = torch.zeros(N)
        src = e.spikes.nonzero(as_tuple=True)[0]
        if src.numel():
            st, en = e.fo_crow[src], e.fo_crow[src + 1]
            cnt = en - st
            tot = int(cnt.sum())
            if tot:
                base = torch.repeat_interleave(st, cnt)
                ramp = torch.arange(tot) - torch.repeat_interleave(
                    torch.cumsum(cnt, 0) - cnt, cnt)
                sel = base + ramp
                rec.scatter_add_(0, e.fo_post[sel], e.fo_val[sel])
            rec *= wS

        e.refrac = torch.where(e.spikes > 0, torch.zeros_like(e.refrac),
                               e.refrac + 1)
        gate = (e.refrac >= e.refrac_steps).float()

        delayed = e.buf[e.head]
        # ---- THE ONE LINE UNDER TEST ----
        decayed = e.g * (1 - dt / p["tauSyn"])
        if mode == "shipped":
            g_new = decayed + delayed * gate
        elif mode == "deliver":
            g_new = decayed + delayed
        else:                                   # brian2: freeze decay, keep input
            g_new = torch.where(gate > 0, decayed + delayed, e.g + delayed)
        e.buf[e.head] = rec
        e.head = (e.head + 1) % e.L

        v = e.v + vstim
        v = v + (dt / p["tauMem"]) * (e.g - (v - p["vRest"]))
        sp = (v > p["vThreshold"]).float()
        v = v - (v - p["vReset"]) * sp

        e.g = g_new - g_new * sp
        e.spikes, e.v = sp, v

        idx = sp.nonzero(as_tuple=True)[0]
        if idx.numel():
            counts[idx] += 1
            new = idx[torch.isnan(first[idx])]
            first[new] = s * dt
            total += idx.numel()

    return counts.numpy(), first.numpy(), total


def compare(name, ids, rate, steps, arm="deliver"):
    ca, fa, ta = run(ids, rate, "shipped", steps)
    cb, fb, tb = run(ids, rate, arm, steps)

    act_a, act_b = set(np.nonzero(ca)[0]), set(np.nonzero(cb)[0])
    inter = len(act_a & act_b)
    union = len(act_a | act_b)
    jac = inter / union if union else 1.0

    both = np.array(sorted(act_a & act_b), dtype=int)
    if len(both) > 2:
        r = float(np.corrcoef(ca[both], cb[both])[0, 1])
        # first-spike latency shift on neurons active in both arms
        d = fb[both] - fa[both]
        d = d[~np.isnan(d)]
        lat = float(np.mean(np.abs(d))) if len(d) else 0.0
    else:
        r, lat = float("nan"), 0.0

    print(f"{name:<24} {ta:>8d} {tb:>8d} {100*(tb-ta)/max(ta,1):>+8.1f}% "
          f"{len(act_a):>7d} {len(act_b):>7d} {jac:>7.3f} {r:>7.3f} {lat:>9.3f}")


def main():
    torch.set_num_threads(4)
    pool = BrainEngine(data_dir=DATA, stim_ids=SUGAR[:1], seed=0).i2flyid
    rng = np.random.default_rng(0)

    print("Effect of the refractory delivery gate on the OBSERVABLE.")
    print("A = as shipped (input dropped while refractory);  "
          "B = input delivered.")
    print("Everything else identical: same seed, same RNG stream, same order.")
    print()
    print(f"{'regime':<24} {'A spk':>8} {'B spk':>8} {'delta':>9} "
          f"{'A act':>7} {'B act':>7} {'Jacc':>7} {'rate r':>7} {'|dlat|ms':>9}")
    print("-" * 96)

    b1000 = rng.choice(pool, 1000, replace=False).tolist()
    for arm in ("deliver", "brian2"):
        print(f"  [B arm = {arm}]")
        compare("sugar GRNs (21)", SUGAR, 200.0, STEPS, arm)
        compare("broad (1000)", b1000, 100.0, STEPS, arm)

    print("-" * 96)
    print("Jacc   = Jaccard overlap of the ACTIVE-NEURON SET -- the quantity")
    print("         compare_ground_truth.py reports as its headline metric.")
    print("rate r = Pearson r of per-neuron spike counts, neurons active in both.")


if __name__ == "__main__":
    main()
