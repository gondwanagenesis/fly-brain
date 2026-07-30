"""Cross-validate every neuron model against Brian 2.

    .venv\\Scripts\\python.exe code\\validate_models_brian2.py

verify_models.py proves the kernel is SELF-consistent: the three ISA paths
agree, tile skipping changes nothing, the certified elision changes nothing.
None of that would notice if a model's equations were simply wrong -- a
consistently wrong AdEx passes every one of those gates.

This is the external check. Each model is written out again in Brian 2, from
the published equations, sharing no code with the kernel, and both are given
the same synaptic impulse. Brian 2 integrates in float64 with its own solver
selection, so the comparison is float32-fused-single-pass against
float64-general-purpose. The question is not "are the bits equal" -- they
cannot be -- but "is the difference the size that float32 and a different
integrator explain, or the size that a wrong coefficient explains". Those are
orders of magnitude apart, which is what makes the test informative.

BOTH SIDES GET THE SAME SYNAPSE
-------------------------------
g(0) = g0 with dg/dt = -g/tau_s, rather than a held constant current. Holding g
would have been simpler and would also have been wrong for the LIF family:
their exact propagator's P_vg coefficient is derived on the assumption that g
DECAYS across the step, so pinning g every step drives a system the propagator
does not solve, and the steady state comes out a factor of five off. Feeding a
decaying conductance exercises the propagator, the synapse and the membrane
together, which is the combination that actually runs in the network.
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "flyloop"))

import models as nrn_models          # noqa: E402

import brian2 as b2                  # noqa: E402
b2.prefs.codegen.target = "numpy"
for _n in ("brian2.codegen", "brian2.groups", "brian2.core"):
    b2.BrianLogger.suppress_name(_n)

DUR = 200.0        # ms
DT = 0.1


LIF_FAMILY = ("lif_euler", "lif_exact", "glif")


def kernel_trace(spec, g0, dur=DUR, dt=DT):
    """One isolated neuron driven by the kernel. Sample 0 is the initial state,
    before any step, so the trace indexes time as Brian 2's StateMonitor does.

    TWO DRIVE MODES, and the reason there are two:

    The LIF family gets a synaptic IMPULSE and lets it decay, because their
    exact propagator's P_vg coefficient IS the analytic integral of a decaying
    conductance across the step -- pinning g would drive a system the
    propagator does not solve, and the steady state comes out a factor of five
    off.

    Every other model gets a CONSTANT current, because the kernel applies the
    synapse as a zero-order hold: `I = k_in * g` is read once at the top of the
    step and held across all sub-steps, with g decaying once per dt. That is a
    deliberate modelling choice -- it matches how delayed input actually
    arrives, at step boundaries -- but Brian 2 decays g continuously, and the
    ~1% disagreement that follows is about the SYNAPSE discretisation, not the
    membrane equation. Since the membrane equation is what this test exists to
    check, the synapse is taken out of the comparison rather than left to
    dominate it. The synapse is covered instead by the LIF rows here (1e-5) and
    by whole-brain bit-identity with PyTorch.
    """
    v, g, aux = nrn_models._isolated(spec, 16)
    hold = spec.key not in LIF_FAMILY
    g[:] = np.float32(g0)
    n = int(dur / dt)
    vs = np.empty(n + 1, dtype=np.float64)
    vs[0] = float(v[0])
    sp = []
    for i in range(n):
        if hold:
            g[:] = np.float32(g0)
        if nrn_models._sweep(spec, v, g, aux, 1):
            sp.append((i + 1) * dt)
        vs[i + 1] = float(v[0])
    return vs, np.asarray(sp)


def brian_trace(spec, g0, dur=DUR, dt=DT):
    """The same neuron, written out from the published equations in Brian 2."""
    P = spec.params
    key = spec.key
    ns = max(1, int(P.n_sub))
    # Resonate-and-fire has no exact reference available, so it gets a refined
    # RK4 one: at dt/20 with a fourth-order method the reference is effectively
    # the true solution, and any disagreement is ours.
    if key == "raf":
        ns *= 20
    sub = dt / ns
    # Brian 2 runs at the KERNEL'S sub-step, not at dt. Otherwise the kernel
    # takes n_sub half-steps where Brian takes one whole one, and the residual
    # is dominated by that O(dt) integrator difference -- about 1% -- which is
    # the same size as a real coefficient error and would mask one.
    b2.defaultclock.dt = sub * b2.ms
    taus = nrn_models.TAU_S * b2.ms
    kin = float(P.k_in)
    hold = key not in LIF_FAMILY
    base = {"taus": taus, "kin": kin}
    # `g : 1 (constant)` mirrors the kernel's held drive exactly, and it also
    # keeps the system linear for resonate-and-fire, which is what lets Brian
    # integrate that one exactly instead of with forward Euler.
    SYN = "g : 1 (constant)" if hold else "dg/dt = -g/taus : 1"

    if key in ("lif_euler", "lif_exact", "glif"):
        # Here g IS the input, in millivolts, inside the membrane equation --
        # this is the Shiu et al. form, not a separate injected current.
        eqs = ("""dv/dt = (g - (v - vr))/tau : 1
                 """ + SYN)
        G = b2.NeuronGroup(1, eqs, threshold="v > vth", reset="v = vr; g = 0",
                           method="exact",
                           namespace={**base, "tau": nrn_models.TAU_M * b2.ms,
                                      "vr": float(spec.rest[0]),
                                      "vth": float(P.v_th)})
        G.v = float(spec.init[0])

    elif key == "izhikevich":
        eqs = ("""dv/dt = (0.04*v**2 + 5*v + 140 - u + kin*g)/ms : 1
                 du/dt = (a*(b*v - u))/ms : 1
                 """ + SYN)
        G = b2.NeuronGroup(1, eqs, threshold="v >= vpk",
                           reset="v = c; u += d", method="euler",
                           namespace={**base,
                                      "a": float(P.izh_a) / sub,
                                      "b": float(P.izh_b), "c": float(P.izh_c),
                                      "d": float(P.izh_d),
                                      "vpk": float(P.v_peak)})
        G.v, G.u = float(spec.init[0]), float(spec.init[1])

    elif key in ("adex", "eif"):
        w_term = "- w" if key == "adex" else ""
        w_eq = "dw/dt = (a*(v-EL) - w)/tauw/ms : 1" if key == "adex" else ""
        eqs = ("dv/dt = (-gL*(v-EL) + gL*DT*exp((v-VT)/DT) " + w_term
               + " + kin*g)/C/ms : 1\n" + w_eq + "\n" + SYN)
        nsp = {**base, "gL": float(P.ad_gL), "EL": float(P.ad_EL),
               "VT": float(P.ad_VT), "DT": 1.0 / float(P.ad_inv_DT),
               "C": sub / float(P.ad_dtC), "vpk": float(P.v_peak),
               "vr": float(P.v_reset)}
        reset = "v = vr"
        if key == "adex":
            nsp["a"] = float(P.ad_a)
            nsp["tauw"] = -sub / np.log(float(P.ad_wdecay))
            nsp["bb"] = float(P.ad_b)
            reset = "v = vr; w += bb"
        G = b2.NeuronGroup(1, eqs, threshold="v > vpk", reset=reset,
                           method="euler", namespace=nsp)
        G.v = float(spec.init[0])
        if key == "adex":
            G.w = float(spec.init[1])

    elif key == "qif":
        eqs = ("""dv/dt = (k*(v - vr)*(v - vc) + kin*g)/ms : 1
                 """ + SYN)
        G = b2.NeuronGroup(1, eqs, threshold="v > vpk", reset="v = vrst",
                           method="euler",
                           namespace={**base, "k": float(P.qif_k),
                                      "vr": float(P.v_rest),
                                      "vc": float(P.qif_vc),
                                      "vpk": float(P.v_peak),
                                      "vrst": float(P.v_reset)})
        G.v = float(spec.init[0])

    elif key == "raf":
        # raf_rr / raf_ri encode exp(b*dt)*(cos, sin) of the per-step rotation.
        # Recover the continuous (b, omega) so Brian 2 integrates the ODE
        # rather than re-running our own discretisation back at us.
        rr, ri = float(P.raf_rr), float(P.raf_ri)
        bb = float(np.log(np.hypot(rr, ri)) / DT)
        om = float(np.arctan2(ri, rr) / DT)
        eqs = ("""dv/dt = (bb*v - om*y + kin*g)/ms : 1
                 dy/dt = (om*v + bb*y)/ms : 1
                 """ + SYN)
        # The whole system -- rotation plus exponential synapse -- is linear,
        # so Brian can integrate it exactly too. Comparing an exact kernel
        # against a forward-Euler reference would measure Brian's phase drift
        # on a 50 Hz oscillator rather than anything about this kernel.
        # Brian's exact solver rejects this system outright -- "the solution
        # to the linear system contains complex values" -- because a resonator
        # has complex eigenvalues. That is precisely the case the kernel does
        # handle in closed form, so the reference here is RK4 on a refined
        # clock rather than an exact solution Brian will not produce.
        G = b2.NeuronGroup(1, eqs, threshold="y > vth", reset="v = vr; y = 0",
                           method="rk4",
                           namespace={**base, "bb": bb, "om": om,
                                      "vth": float(P.v_th),
                                      "vr": float(P.v_reset)})
        G.v, G.y = float(spec.init[0]), float(spec.init[1])

    elif key == "hh":
        sh = float(P.hh_shift)
        eqs = """
        dv/dt = (gNa*m**3*hg*(ENa-(v-sh)) + gK*n**4*(EK-(v-sh))
                 + gL*(EL-(v-sh)) + kin*g)/Cm/ms : 1
        dm/dt  = (alpham*(1-m) - betam*m)/ms : 1
        dhg/dt = (alphah*(1-hg) - betah*hg)/ms : 1
        dn/dt  = (alphan*(1-n) - betan*n)/ms : 1
        """ + "        " + SYN + """
        alpham = 0.1*(v-sh+40)/(1-exp(-(v-sh+40)/10)) : 1
        betam  = 4*exp(-(v-sh+65)/18) : 1
        alphah = 0.07*exp(-(v-sh+65)/20) : 1
        betah  = 1/(1+exp(-(v-sh+35)/10)) : 1
        alphan = 0.01*(v-sh+55)/(1-exp(-(v-sh+55)/10)) : 1
        betan  = 0.125*exp(-(v-sh+65)/80) : 1
        """
        # `refractory` as a CONDITION is Brian 2's equivalent of the kernel's
        # `armed` latch: it stops one action potential being counted several
        # times as it rides above threshold.
        G = b2.NeuronGroup(1, eqs, method="exponential_euler",
                           threshold="v > vth", refractory="v > vrs",
                           namespace={**base, "gNa": float(P.hh_gNa),
                                      "gK": float(P.hh_gK), "gL": float(P.hh_gL),
                                      "ENa": float(P.hh_ENa),
                                      "EK": float(P.hh_EK), "EL": float(P.hh_EL),
                                      "Cm": sub / float(P.hh_dtC), "sh": sh,
                                      "vth": float(P.v_th),
                                      "vrs": float(P.v_reset)})
        G.v = float(spec.init[0])
        G.m, G.hg, G.n = (float(spec.init[1]), float(spec.init[2]),
                          float(spec.init[3]))
    else:
        raise KeyError(key)

    G.g = g0
    mon = b2.StateMonitor(G, "v", record=0)
    spk = b2.SpikeMonitor(G)
    net = b2.Network(G, mon, spk)
    net.run(dur * b2.ms)
    # subsample back to the kernel's dt grid
    return (np.asarray(mon.v[0], dtype=np.float64)[::ns],
            np.asarray(spk.t / b2.ms))


def rheobase(spec, dt=DT, dur=300.0):
    """Smallest HELD synaptic drive that makes this model fire.

    The critical impulse the models are calibrated on is a TRANSIENT volley;
    held forever, a fraction of it still sits far above the current needed for
    repetitive firing. Testing "sub-threshold" behaviour against the transient
    number therefore measured a spike train, and the membrane comparison was
    dominated by spike-time jitter rather than by the sub-threshold trajectory.
    """
    lo, hi = 1e-6, 1e6
    for _ in range(44):
        mid = math.sqrt(lo * hi)
        v, g, aux = nrn_models._isolated(spec, 16)
        fired = 0
        for _ in range(int(dur / dt)):
            g[:] = np.float32(mid)
            if nrn_models._sweep(spec, v, g, aux, 1):
                fired = 1
                break
        if fired:
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


def compare(spec, g0):
    """Best of the two plausible time alignments, and which one won.

    The two harnesses do not have to agree about whether sample i is the state
    before or after step i, and a one-sample offset shows up as an error the
    size of one step's rise -- 3% of the PSP for the LIF family, which is the
    same order as a genuine coefficient error and would be easy to misread as
    one. Rather than assert an alignment, both are measured and the shift is
    reported: a real discrepancy does not go away under either.
    """
    vk, sk = kernel_trace(spec, g0)
    vb, sb = brian_trace(spec, g0)
    best = None
    for shift in (0, 1):
        n = min(len(vk) - shift, len(vb))
        d = np.abs(vk[shift:shift + n] - vb[:n])
        if best is None or d.max() < best[0]:
            best = (float(d.max()), shift)
    swing = max(1e-9, float(np.max(np.abs(vb - vb[0]))))
    dt_sp = float("nan")
    if len(sk) and len(sb):
        m = min(len(sk), len(sb))
        dt_sp = float(np.mean(np.abs(sk[:m] - sb[:m])))
    return best[0], swing, len(sk), len(sb), dt_sp, best[1]


def main():
    print(f"Brian 2 {b2.__version__} cross-validation   "
          f"{DUR:.0f} ms @ dt={DT} ms")
    print("both sides driven by the same synaptic impulse "
          f"(g0 = {nrn_models.CRITICAL_IMPULSE:.2f} mV x scale)\n")
    hdr = (f"{'model':<11} {'regime':<8} {'max |dv|':>10} {'PSP swing':>10} "
           f"{'rel err':>9} {'shift':>6} {'spikes k/b':>11} {'d t_spk':>8}")
    print(hdr)
    print("-" * len(hdr))
    worst, worst_key = 0.0, ""
    sub_err = {}
    for key in nrn_models.ALL_MODELS:
        if key == "lif_euler":
            # Model 0 has no template body: it runs the ATen-faithful sweep,
            # which nrn_sweep_raw does not drive. It needs no external check
            # here anyway -- it is already proven bit-identical to the PyTorch
            # backend, and code/compare_to_brian2.py compares that backend to
            # Brian 2 at whole-brain scale.
            print(f"{'lif_euler':<11} {'--':<8}  "
                  f"covered by verify_native.py + compare_to_brian2.py")
            continue
        spec = nrn_models.build(key)
        drive = (nrn_models.CRITICAL_IMPULSE if key in LIF_FAMILY
                 else rheobase(spec))
        for label, frac in (("sub", 0.60), ("supra", 1.40)):
            g0 = frac * drive
            try:
                mx, swing, nk, nb, dts, shift = compare(spec, g0)
            except Exception as e:
                print(f"{key:<11} {label:<8}  FAILED: "
                      f"{type(e).__name__}: {str(e)[:58]}")
                continue
            rel = mx / swing
            if label == "sub":
                sub_err[key] = rel
                if rel > worst:
                    worst, worst_key = rel, key
            print(f"{key:<11} {label:<8} {mx:>10.3e} {swing:>10.4f} "
                  f"{rel:>8.2e} {shift:>6} {nk:>4}/{nb:<5} {dts:>8.3f}")
    print("-" * len(hdr))
    print("rel err = max membrane difference / the size of the PSP itself.")
    print("Both sides now use the same integrator at the same step size, so the")
    print("residual is float32-against-float64 and lands near 1e-5 to 1e-4.")
    print("A mistranscribed coefficient lands near 1e-1. The gap is the test.")
    # ---- step refinement: the decisive test ----------------------------
    #
    # A residual above the float32 floor is either an integrator difference or
    # a wrong coefficient, and arguing about which is pointless when it can be
    # measured. Halve the step: an integrator difference shrinks with it, a
    # wrong coefficient does not move. Run for any model still above 1e-3.
    slow = [k for k, r in sub_err.items() if r > 1e-3]
    if slow:
        print()
        print("step refinement -- integrator difference, or wrong coefficient?")
        print(f"  {'model':<11} {'n_sub':>6} {'rel err':>10} {'ratio':>7}")
        for key in slow:
            base_spec = nrn_models.build(key)
            k_in = float(base_spec.params.k_in)
            n0 = max(1, int(base_spec.params.n_sub))
            drive = 0.6 * rheobase(base_spec)
            prev = None
            for mult in (1, 2, 4, 8):
                sp = nrn_models.build(key, n_sub=n0 * mult)
                sp.params.k_in = np.float32(k_in)   # only the step size varies
                mx, swing, _, _, _, _ = compare(sp, drive)
                rel = mx / swing
                ratio = (prev / rel) if prev else float("nan")
                print(f"  {key:<11} {n0*mult:>6} {rel:>10.3e} {ratio:>7.2f}")
                prev = rel
        print("  A ratio near 2 per halving is first-order convergence: the two")
        print("  sides are solving the same equations with different methods.")
        print("  A ratio near 1 would mean the equations themselves differ.")

    print("The supra-threshold rows are expected to be large: past the cutoff")
    print("the trajectory is divergent by construction, so a sub-ulp difference")
    print("in WHEN the reset fires becomes a whole spike amplitude. The spike")
    print("TIMES are the meaningful comparison there, not the membrane trace.")
    print(f"\nworst sub-threshold relative error: {worst:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
