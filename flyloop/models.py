"""Neuron-model registry for the whole-brain Drosophila kernel.

Nine membrane models over one connectome. The network -- 138,639 neurons,
15,091,983 edges, a uniform 1.8 ms axonal delay, a 5 ms exponential synapse --
is fixed; what changes is the differential equation each neuron obeys.

WHY THE MODELS ARE CALIBRATED RATHER THAN COPIED
------------------------------------------------
Each model's published parameters live in their own units and their own voltage
scale: AdEx in pF/nS/pA around -70.6 mV, Izhikevich in a dimensionless
polynomial around -70 mV, Hodgkin-Huxley in uA/cm2 around -65 mV. Dropping
those straight onto a connectome whose synaptic weights are calibrated in
millivolts against a 7 mV threshold does not produce "the fly brain with AdEx
neurons", it produces a network that is either silent or in seizure, and the
comparison between models would measure unit mismatch rather than dynamics.

So each model is CALIBRATED: its published shape is kept exactly, and one free
gain `k_in` is solved for numerically so that the model needs the SAME NUMBER OF
SIMULTANEOUS SYNAPSES TO FIRE as the reference LIF does -- about 162. That makes
"switch the model" a controlled experiment (same connectome, same drive, same
synaptic efficacy in the only units that govern network behaviour) rather than a
units accident. `calibrate()` solves it by bisection against the kernel itself;
nothing is hand-tuned.

Excitability, not amplitude, is the right invariant, and the difference is not
academic: matching millivolts instead put Hodgkin-Huxley about five times too
close to threshold and drove the whole connectome into sustained 20 Hz firing,
which would have read as a finding about HH dynamics and was really a finding
about dividing by the wrong voltage. See `calibrate`.

WHAT EACH MODEL COSTS, AND WHY
------------------------------
Per neuron per dt, on the AVX-512 path:

  lif_euler   ~5 flops   the shipped reference; forward Euler
  lif_exact   ~4 flops   FEWER operations than Euler AND exact. Two FMAs and a
                         multiply for the (v,g) matrix exponential.
  qif         ~5 flops   quadratic normal form, no transcendental
  raf         ~6 flops   exact rotation; constants precomputed
  glif        ~7 flops   lif_exact plus a moving threshold
  izh        ~10 flops   Horner quadratic + recovery variable
  eif        ~9-21       9 when the exponential is provably elided, 21 when not
  adex      ~12-24       as eif, plus Rush-Larsen adaptation
  hh          ~180       six rate functions, four states, four substeps

The eif/adex spread is the certified elision described in sweep_template.h: the
exponential term is skipped when a rigorous bound proves it rounds away, which
in this connectome is most neurons most of the time.
"""
from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- model ids
M_LIF_EULER, M_LIF_EXACT, M_IZH, M_ADEX, M_EIF, M_QIF, M_RAF, M_HH, M_GLIF = range(9)

MODEL_IDS = {
    "lif_euler": M_LIF_EULER,
    "lif_exact": M_LIF_EXACT,
    "izhikevich": M_IZH,
    "adex": M_ADEX,
    "eif": M_EIF,
    "qif": M_QIF,
    "raf": M_RAF,
    "hh": M_HH,
    "glif": M_GLIF,
}
MODEL_NAMES = {v: k for k, v in MODEL_IDS.items()}

# Auxiliary state arrays beyond (v, g). Must match NRN_NAUX in nrn_params.h.
N_AUX = {
    M_LIF_EULER: 0, M_LIF_EXACT: 0, M_IZH: 1, M_ADEX: 1,
    M_EIF: 0, M_QIF: 0, M_RAF: 1, M_HH: 4, M_GLIF: 1,
}


class NrnParams(ctypes.Structure):
    """Mirror of `nrn_params` in native/nrn_params.h.

    Field order is load-bearing: the C side reads this by offset. The layout is
    all 4-byte scalars, so there is no padding to disagree about, and
    `_check_layout()` asserts the total size against the C `sizeof` reported by
    the DLL at import time.
    """
    _fields_ = [
        ("v_rest", ctypes.c_float), ("v_reset", ctypes.c_float),
        ("v_th", ctypes.c_float), ("v_peak", ctypes.c_float),
        ("g_decay", ctypes.c_float), ("k_in", ctypes.c_float),
        ("reset_g", ctypes.c_int32), ("n_sub", ctypes.c_int32),
        ("c_decay", ctypes.c_float), ("c_mem", ctypes.c_float),
        ("alpha_m", ctypes.c_float), ("alpha_s", ctypes.c_float),
        ("p_vg", ctypes.c_float),
        ("izh_p2", ctypes.c_float), ("izh_p1", ctypes.c_float),
        ("izh_p0", ctypes.c_float),
        ("izh_a", ctypes.c_float), ("izh_b", ctypes.c_float),
        ("izh_c", ctypes.c_float), ("izh_d", ctypes.c_float),
        ("izh_dt", ctypes.c_float),
        ("ad_dtC", ctypes.c_float), ("ad_gL", ctypes.c_float),
        ("ad_EL", ctypes.c_float), ("ad_VT", ctypes.c_float),
        ("ad_inv_DT", ctypes.c_float), ("ad_gLDT", ctypes.c_float),
        ("ad_wdecay", ctypes.c_float), ("ad_a", ctypes.c_float),
        ("ad_b", ctypes.c_float), ("elide_tiny", ctypes.c_float),
        ("qif_k", ctypes.c_float), ("qif_vc", ctypes.c_float),
        ("raf_rr", ctypes.c_float), ("raf_ri", ctypes.c_float),
        ("raf_mr", ctypes.c_float), ("raf_mi", ctypes.c_float),
        ("hh_dtC", ctypes.c_float), ("hh_dt", ctypes.c_float),
        ("hh_gNa", ctypes.c_float), ("hh_gK", ctypes.c_float),
        ("hh_gL", ctypes.c_float), ("hh_ENa", ctypes.c_float),
        ("hh_EK", ctypes.c_float), ("hh_EL", ctypes.c_float),
        ("hh_shift", ctypes.c_float),
        ("gl_thdecay", ctypes.c_float), ("gl_th_inf", ctypes.c_float),
        ("gl_th_jump", ctypes.c_float),
    ]


@dataclass
class ModelSpec:
    """Everything the engine and the UI need to know about one model."""
    key: str
    mid: int
    label: str
    citation: str
    equation: str
    note: str
    n_aux: int
    params: NrnParams
    rest: np.ndarray          # [v_rest, aux0_rest, aux1_rest, ...]  float32
    init: np.ndarray          # same shape; the state a fresh neuron starts in
    aux_names: tuple = ()
    flops: str = ""
    exact: str = ""
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------ helpers
def _f32(x):
    return float(np.float32(x))


# The reference network's own scale, from Shiu et al. Everything is calibrated
# against these three numbers and nothing else.
V_REST = -52.0
V_TH = -45.0
TAU_M = 20.0
TAU_S = 5.0
W_SYN = 0.275          # mV per synapse, before the integer synapse count


def reference_psp_fraction(dt=0.1):
    """Peak depolarisation from ONE unit synaptic event, as a fraction of the
    distance to threshold, in the reference LIF.

    This is the single number every other model is calibrated to match. It is
    computed from the exact propagator rather than measured, because for the
    linear reference the closed form is available:

        u(t) = (tau_s / (tau_m - tau_s)) * g0 * (e^{-t/tau_m} - e^{-t/tau_s})

    whose peak sits at t* = ln(tau_m/tau_s) / (1/tau_s - 1/tau_m) = 9.242 ms.
    """
    ts = math.log(TAU_M / TAU_S) / (1.0 / TAU_S - 1.0 / TAU_M)
    kappa = (TAU_S / (TAU_M - TAU_S)) * (math.exp(-ts / TAU_M) - math.exp(-ts / TAU_S))
    return (kappa * W_SYN) / (V_TH - V_REST)


def reference_critical_impulse():
    """The synaptic charge that just fires a reference LIF neuron from rest.

    Every model is calibrated so that THIS impulse is exactly its own firing
    threshold, which is what makes "one synapse" mean the same thing in all
    nine. In the reference that is (v_th - v_rest) / kappa_peak = 44.4 mV of
    conductance, i.e. about 162 synapses arriving together -- a number worth
    keeping in view, because it is the reason the connectome is quiet: a
    typical neuron needs a coincident volley, not a trickle.
    """
    ts = math.log(TAU_M / TAU_S) / (1.0 / TAU_S - 1.0 / TAU_M)
    kappa = (TAU_S / (TAU_M - TAU_S)) * (math.exp(-ts / TAU_M) - math.exp(-ts / TAU_S))
    return (V_TH - V_REST) / kappa


CRITICAL_IMPULSE = None      # set immediately below; see reference_critical_impulse


CRITICAL_IMPULSE = reference_critical_impulse()
N_SYNAPSES_TO_FIRE = CRITICAL_IMPULSE / W_SYN


# ------------------------------------------------- driving the kernel itself
#
# Calibration, the resting-state search and the PSP measurements all run THE
# KERNEL, through `nrn_sweep_raw` -- a bare sweep with no delay line, no
# fan-out and no tile skipping. Nothing here re-implements a membrane equation.
# That is deliberate: a second implementation in numpy would be a second thing
# to keep correct, and every disagreement between them would need adjudicating.
# There is one implementation, and these functions are just harnesses for it.


def _isolated(spec: ModelSpec, n=16):
    """Fresh state arrays for `n` identical, unconnected neurons at rest."""
    v = np.full(n, spec.init[0], dtype=np.float32)
    g = np.zeros(n, dtype=np.float32)
    aux = (np.repeat(spec.init[1:].astype(np.float32), n).reshape(spec.n_aux, n)
           .copy() if spec.n_aux else np.zeros(0, dtype=np.float32))
    return v, g, np.ascontiguousarray(aux)


def _sweep(spec: ModelSpec, v, g, aux, steps=1):
    """Advance the isolated neurons `steps` times, in place. Returns spikes."""
    import native_lib
    L = native_lib.lib()
    n = v.size
    nt = (n + 15) // 16
    tl = np.zeros(nt + 1, np.int32)
    tlive = np.zeros((nt + 63) // 64 + 1, np.uint64)
    spb = np.zeros((n + 63) // 64 + 1, np.uint64)
    stride = n if spec.n_aux else 0
    return L.nrn_sweep_raw(
        spec.mid, ctypes.byref(spec.params),
        native_lib.ptr(v), native_lib.ptr(g),
        native_lib.ptr(aux.reshape(-1)) if stride else None, stride,
        n, steps,
        native_lib.ptr(tl, ctypes.c_int32),
        native_lib.ptr(tlive, ctypes.c_uint64),
        native_lib.ptr(spb, ctypes.c_uint64))


def find_rest(spec: ModelSpec, max_iter=2_000_000):
    """The state the model ACTUALLY settles to, to the last bit.

    Tile skipping is only lossless if the update is a genuine fixed point of
    the float32 arithmetic at the resting state: skipping reproduces the state
    unchanged, so it is exact precisely when stepping would too. The analytic
    resting potential is not good enough for that -- for Izhikevich it is an
    irrational root of a quadratic, and for Hodgkin-Huxley it is the solution
    of a transcendental system, neither of which lands on a float32 that the
    update maps to itself.

    So the state is not asserted, it is FOUND: relax an isolated neuron with no
    input until one further step changes nothing at the bit level. Returns
    (rest_state, is_fixed_point). If the search does not converge, the caller
    must disable tile skipping for that model rather than skip something whose
    state would have drifted.
    """
    if spec.mid == M_LIF_EULER:
        # No template body: model 0 runs the ATen-faithful sweep, whose fixed
        # point is v_rest exactly (t = 0, so v + 0*c_mem == v).
        return spec.init.copy(), True

    v, g, aux = _isolated(spec, 16)
    block = 256
    done = 0
    while done < max_iter:
        before = (v.copy(), aux.copy())
        _sweep(spec, v, g, aux, 1)
        if (np.array_equal(v.view(np.uint32), before[0].view(np.uint32))
                and np.array_equal(aux.view(np.uint32), before[1].view(np.uint32))):
            out = np.concatenate([v[:1], aux[:, 0] if spec.n_aux else []])
            return out.astype(np.float32), True
        _sweep(spec, v, g, aux, block)
        done += block + 1
    out = np.concatenate([v[:1], aux[:, 0] if spec.n_aux else []])
    return out.astype(np.float32), False


def psp_peak(spec: ModelSpec, k_in, dt=0.1, window_ms=60.0, g0=W_SYN):
    """Peak depolarisation from one unit synaptic event, in this model's mV.

    Spiking is DISABLED for the measurement, by pushing the cutoff out of
    reach. Without that the measurement is not monotone in k_in and bisection
    breaks: AdEx resets to E_L, which is also its resting potential, so once
    the drive is strong enough to fire, the post-step membrane reads exactly
    rest again and the apparent PSP collapses back to zero. Every model has
    some version of this -- Izhikevich resets to -65 mV, five above its own
    rest -- so rather than special-case them, the reset is simply taken out of
    the loop. A calibrated PSP is sub-threshold by construction anyway; what is
    wanted is the size of the bump, not what the neuron does at the top of it.
    """
    P = spec.params
    keep = (P.k_in, P.v_peak, P.v_th)
    P.k_in = _f32(k_in)
    P.v_peak = _f32(3.0e38)
    P.v_th = _f32(3.0e38)
    try:
        v, g, aux = _isolated(spec, 16)
        g[:] = np.float32(g0)
        best = float(v[0])
        for _ in range(int(window_ms / dt)):
            _sweep(spec, v, g, aux, 1)
            x = float(v[0])
            if x > best:
                best = x
        return best - float(spec.rest[0])
    finally:
        P.k_in, P.v_peak, P.v_th = keep


def n_synapses_to_fire(spec: ModelSpec, k_in, dt=0.1, window_ms=100.0):
    """Does a simultaneous volley of `CRITICAL_IMPULSE` worth of synapse fire it?

    Returns the spike count. Monotone non-decreasing in k_in, which is what
    bisection needs.
    """
    old = spec.params.k_in
    spec.params.k_in = _f32(k_in)
    try:
        v, g, aux = _isolated(spec, 16)
        g[:] = np.float32(CRITICAL_IMPULSE)
        return _sweep(spec, v, g, aux, int(window_ms / dt)) // 16
    finally:
        spec.params.k_in = old


def calibrate(spec: ModelSpec, dt=0.1, target=None):
    """Solve for the synaptic gain that matches the reference LIF's EXCITABILITY.

    The invariant held fixed across models is "how many simultaneous synapses
    does it take to fire this neuron from rest", not "how many millivolts does
    one synapse move it". Those are the same question only for models sharing a
    voltage scale, and they are badly different for Hodgkin-Huxley: matching
    millivolts there means measuring the PSP against the 65 mV climb to the
    action-potential peak, when the neuron actually becomes unstable about
    10 mV above rest. Calibrating on amplitude made HH roughly five times too
    excitable and put the whole connectome into sustained 20 Hz firing -- a
    result that would have looked like a finding about Hodgkin-Huxley dynamics
    and was really a finding about the wrong denominator.

    Excitability is also the quantity that actually sets network behaviour:
    what a neuron does depends on how much convergent input it takes to
    recruit it, not on the arbitrary units of its membrane variable.

    Bisection on k_in against "does the reference's critical volley fire this
    model", which is monotone in k_in by construction.
    """
    lo, hi = 1e-9, 1e9
    if not n_synapses_to_fire(spec, hi, dt):
        raise RuntimeError(f"{spec.key}: the reference's critical volley never "
                           f"fires it, even at k_in=1e9")
    if n_synapses_to_fire(spec, lo, dt):
        raise RuntimeError(f"{spec.key}: fires with no synaptic gain at all")
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if n_synapses_to_fire(spec, mid, dt):
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


def _calibrate_amplitude(spec: ModelSpec, dt=0.1, target=None):
    """Superseded amplitude calibration, kept because it is the right tool for
    a different question: matching PSP size in millivolts when two models share
    a voltage scale. `calibrate` matches excitability instead; see its docstring
    for why that is the one that governs network dynamics.

    Bisection on k_in so that the peak depolarisation produced by one unit
    synaptic event equals `target` times this model's own threshold distance.
    The trace maximum is monotone non-decreasing in k_in -- more input can only
    push v higher at every point before the first reset, and once a spike
    occurs the recorded maximum is the spike peak -- so bisection cannot
    converge to a wrong branch.

    Bracketing starts at [1e-9, 1e9] and is never widened: a model that cannot
    reach the target inside twenty orders of magnitude has wrong parameters,
    and failing loudly beats shipping a silently mis-scaled network.
    """
    if target is None:
        target = reference_psp_fraction(dt)
    want = target * (float(spec.params.v_th) - float(spec.rest[0]))

    # Bracket by expanding UPWARD from a tiny gain, never by starting at a huge
    # one. Monotonicity holds where the response is sub-threshold and fails at
    # the far end: with k_in = 1e9 the AdEx exponential saturates inside a
    # single sub-step, v becomes +inf, the cutoff fires and the post-step
    # membrane reads exactly E_L again -- an apparent PSP of zero. Doubling up
    # from below stops the moment the target is first exceeded, which is
    # comfortably inside the well-behaved region for every model here.
    lo = 1e-7
    while psp_peak(spec, lo, dt) >= want:
        lo /= 64.0
        if lo < 1e-30:
            raise RuntimeError(f"{spec.key}: even a vanishing gain overshoots "
                               f"the calibration target ({want:.3e})")
    hi = lo
    for _ in range(200):
        hi *= 2.0
        p = psp_peak(spec, hi, dt)
        if p >= want and math.isfinite(p):
            break
    else:
        raise RuntimeError(f"{spec.key}: no gain reaches the calibration "
                           f"target {want:.3e}")
    for _ in range(44):
        mid = math.sqrt(lo * hi)
        if psp_peak(spec, mid, dt) < want:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


# ------------------------------------------------------------------ builders
def _base(dt):
    P = NrnParams()
    P.v_rest = _f32(V_REST)
    P.v_reset = _f32(V_REST)
    P.v_th = _f32(V_TH)
    P.v_peak = _f32(V_TH)
    P.g_decay = _f32(math.exp(-dt / TAU_S))
    P.k_in = _f32(1.0)
    P.reset_g = 0
    P.n_sub = 1
    # 2^-25. If |exponential term| < |S| * 2^-25 then it is below half an ulp
    # of S, so fl(S + E) == S bit-for-bit and the term can be skipped without
    # changing the answer. Zero disables the elision; see nrn_params.h.
    P.elide_tiny = _f32(2.0 ** -25)
    # LIF constants, filled for every model because the engine reads c_decay /
    # c_mem when it needs an Euler comparison and alpha_* for the exact one.
    P.c_decay = _f32(1 - dt / TAU_S)
    P.c_mem = _f32(dt / TAU_M)
    am = math.exp(-dt / TAU_M)
    a_s = math.exp(-dt / TAU_S)
    P.alpha_m = _f32(am)
    P.alpha_s = _f32(a_s)
    P.p_vg = _f32((TAU_S / (TAU_M - TAU_S)) * (am - a_s))
    return P


def _check_layout():
    """Fail at import if nrn_params.h and NrnParams have drifted apart."""
    import native_lib
    c_size = native_lib.lib().nrn_params_size()
    if c_size != ctypes.sizeof(NrnParams):
        raise RuntimeError(
            f"nrn_params layout mismatch: C says {c_size} bytes, Python says "
            f"{ctypes.sizeof(NrnParams)}. Fields must match nrn_params.h in "
            "order and type.")


_CACHE_PATH = _HERE_DATA = None


def _cache_file():
    """Where the solved calibration constants live.

    The solve is ~60 kernel runs of 600 steps per model, which is a couple of
    seconds each -- fine once, unacceptable every time an engine is built or
    the UI switches model. The cache is keyed by the kernel's source
    fingerprint, so touching any model body invalidates it automatically rather
    than serving constants that no longer describe the code.
    """
    import native_lib
    from pathlib import Path
    d = Path(__file__).resolve().parent.parent / "data" / "model_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"calib_{native_lib._source_fingerprint()}.json"


def _load_cache():
    import json
    p = _cache_file()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(c):
    import json
    _cache_file().write_text(json.dumps(c, indent=1, sort_keys=True))


def build(key, dt=0.1, calibrated=True, n_sub=None):
    """Return the fully-populated ModelSpec for `key`.

    Three things happen here that cannot happen in the per-model builders,
    because all three require running the kernel:

    1. the layout of the parameter block is checked against the C struct;
    2. the float32 resting state is FOUND rather than assumed, which is what
       licenses tile skipping for this model (see find_rest);
    3. the synaptic gain is solved for, so one synapse means the same thing in
       every model.
    """
    if key not in MODEL_IDS:
        raise KeyError(f"unknown model {key!r}; have {sorted(MODEL_IDS)}")
    _check_layout()
    # n_sub is an override for convergence testing: re-running a model with
    # finer sub-steps and watching the residual against an independent
    # reference shrink is what distinguishes an integrator difference from a
    # wrong coefficient. It deliberately does NOT re-run the calibration --
    # the point is to change only the step size.
    if n_sub is None:
        spec = _BUILDERS[key](dt)
    else:
        try:
            spec = _BUILDERS[key](dt, n_sub=int(n_sub))
        except TypeError:
            spec = _BUILDERS[key](dt)

    cache = _load_cache()
    ck = f"{key}@{dt}@{n_sub if n_sub is not None else '-'}"
    hit = cache.get(ck) if calibrated else None

    if hit:
        rest = np.asarray(hit["rest"], dtype=np.float32)
        ok = hit["fixed_point"]
    else:
        rest, ok = find_rest(spec)
    spec.extra["rest_is_fixed_point"] = bool(ok)
    spec.extra["rest_drift"] = float(abs(rest[0] - spec.rest[0]))
    if ok:
        spec.rest = rest
        spec.init = rest.copy()
    # If the relaxation did not reach a bit-level fixed point the engine must
    # not skip tiles for this model: a "resting" neuron would still be moving,
    # and skipping it would silently freeze that drift.
    spec.extra["can_skip_tiles"] = bool(ok)

    if calibrated and spec.mid not in (M_LIF_EULER, M_LIF_EXACT, M_GLIF):
        k = hit["k_in"] if hit else calibrate(spec, dt)
        spec.params.k_in = _f32(k)
    spec.extra["k_in"] = float(spec.params.k_in)
    spec.extra["psp_mv"] = (hit["psp_mv"] if hit else
                            (psp_peak(spec, float(spec.params.k_in), dt)
                             if spec.mid != M_LIF_EULER else None))
    spec.extra["psp_fraction_target"] = reference_psp_fraction(dt)

    if calibrated and not hit:
        cache[ck] = {"rest": [float(x) for x in rest], "fixed_point": bool(ok),
                     "k_in": float(spec.params.k_in),
                     "psp_mv": spec.extra["psp_mv"]}
        _save_cache(cache)
    return spec


def _b_lif_euler(dt):
    P = _base(dt)
    P.reset_g = 1
    return ModelSpec(
        key="lif_euler", mid=M_LIF_EULER, label="LIF (forward Euler)",
        citation="Shiu et al. 2024, Nature 634:210 -- as shipped in run_pytorch.py",
        equation='dv/dt = (g − (v − v_rest)) / τₘ      dg/dt = −g / τₛ',
        note=("The repository's own backend. Forward Euler shortens every time "
              "constant by exactly dt/2, so this runs 0.25-1% fast relative to "
              "the Brian 2 reference the model was published with. Kept as the "
              "default because it is the only model here that is bit-identical "
              "to the PyTorch backend, which is what the benchmark compares."),
        n_aux=0, params=P,
        rest=np.array([V_REST], np.float32),
        init=np.array([V_REST], np.float32),
        flops="~5", exact="no (first order)")


def _b_lif_exact(dt):
    P = _base(dt)
    P.reset_g = 1
    return ModelSpec(
        key="lif_exact", mid=M_LIF_EXACT, label="LIF (exact propagator)",
        citation="Rotter & Diesmann 1999, Biol. Cybern. 81:381",
        equation='u ← αₘ·u + P_vg·g      g ← αₛ·g      (α = e^−ᵈᵗᐟᵀ)',
        note=("The matrix exponential of the same linear system, so it is exact "
              "between synaptic arrivals rather than first-order. It is also "
              "CHEAPER than Euler -- two FMAs and a multiply against Euler's "
              "four operations -- so there is no trade to make. This is what "
              "Brian 2 selects automatically for linear equations, and "
              "therefore what the published model actually ran."),
        n_aux=0, params=P,
        rest=np.array([V_REST], np.float32),
        init=np.array([V_REST], np.float32),
        flops="~4", exact="yes (between arrivals)")


def _b_izh(dt, n_sub=1):
    P = _base(dt)
    # Izhikevich's polynomial is dimensionally tied to its own mV scale, so the
    # model keeps its published voltage range (-70 rest, +30 peak) and the
    # connectome is matched to it by k_in rather than the other way round.
    a, b, c, d = 0.02, 0.2, -65.0, 8.0     # regular spiking
    P.izh_p2, P.izh_p1, P.izh_p0 = _f32(0.04), _f32(5.0), _f32(140.0)
    P.izh_b, P.izh_c, P.izh_d = _f32(b), _f32(c), _f32(d)
    P.n_sub = n_sub
    P.izh_dt = _f32(dt / P.n_sub)
    P.izh_a = _f32(a * dt / P.n_sub)       # the struct carries a*dt_sub
    vr = _izh_rest(a, b)
    P.v_rest = _f32(vr)
    P.v_reset = _f32(c)
    P.v_th = _f32(-50.0)                   # the unstable fixed point
    P.v_peak = _f32(30.0)
    return ModelSpec(
        key="izhikevich", mid=M_IZH, label="Izhikevich (regular spiking)",
        citation="Izhikevich 2003, IEEE Trans. Neural Netw. 14:1569",
        equation='v′ = 0.04v² + 5v + 140 − u + I      u′ = a(bv − u)      v ≥ 30 → v←c, u←u+d',
        note=("Two state variables and a quadratic, tuned so that four "
              "parameters reproduce most cortical firing patterns. The "
              "quadratic is a Horner chain, i.e. two FMAs, so the cost is the "
              "second state array's memory traffic rather than the arithmetic "
              "-- it benchmarks within a few percent of LIF."),
        n_aux=1, params=P, aux_names=("u",),
        rest=np.array([vr, b * vr], np.float32),
        init=np.array([vr, b * vr], np.float32),
        flops="~10", exact="no (first order)")


def _izh_rest(a, b):
    """Stable fixed point of 0.04v^2 + 5v + 140 - u = 0 with u = bv."""
    A, B, C = 0.04, 5.0 - b, 140.0
    disc = B * B - 4 * A * C
    if disc < 0:
        raise RuntimeError("Izhikevich parameters have no resting state")
    return (-B - math.sqrt(disc)) / (2 * A)


def _b_adex(dt, n_sub=2):
    P = _base(dt)
    # Brette & Gerstner 2005, regular-spiking set. pF / nS / mV / pA / ms.
    C, gL, EL, VT, DT = 281.0, 30.0, -70.6, -50.4, 2.0
    tau_w, a, b = 144.0, 4.0, 80.5
    P.n_sub = n_sub
    h = dt / n_sub
    P.ad_dtC = _f32(h / C)
    P.ad_gL = _f32(gL)
    P.ad_EL = _f32(EL)
    P.ad_VT = _f32(VT)
    P.ad_inv_DT = _f32(1.0 / DT)
    P.ad_gLDT = _f32(gL * DT)
    P.ad_wdecay = _f32(math.exp(-h / tau_w))
    P.ad_a = _f32(a)
    P.ad_b = _f32(b)
    P.v_rest = _f32(EL)
    P.v_reset = _f32(EL)
    P.v_th = _f32(VT)
    P.v_peak = _f32(20.0)      # cutoff: the exponential blows up past this
    return ModelSpec(
        key="adex", mid=M_ADEX, label="AdEx (adaptive exponential)",
        citation="Brette & Gerstner 2005, J. Neurophysiol. 94:3637",
        equation='C·v′ = −g_L(v−E_L) + g_LΔ_T·e^((v−V_T)/Δ_T) − w + I      τ_w·w′ = a(v−E_L) − w',
        note=("Predicts real cortical spike times to within ~2 ms, which is why "
              "it is the standard 'realistic but tractable' model. Two "
              "optimisations make it viable whole-brain: the adaptation "
              "variable is advanced by its exact solution (Rush & Larsen 1978, "
              "from cardiac electrophysiology), and the exponential is skipped "
              "whenever a rigorous bound proves it rounds away -- which in this "
              "connectome is most neurons most of the time."),
        n_aux=1, params=P, aux_names=("w",),
        rest=np.array([EL, 0.0], np.float32),
        init=np.array([EL, 0.0], np.float32),
        flops="~12 elided / ~24 full", exact="no (first order in v)")


def _b_eif(dt, n_sub=2):
    P = _base(dt)
    C, gL, EL, VT, DT = 281.0, 30.0, -70.6, -50.4, 2.0
    P.n_sub = n_sub
    h = dt / n_sub
    P.ad_dtC = _f32(h / C)
    P.ad_gL = _f32(gL)
    P.ad_EL = _f32(EL)
    P.ad_VT = _f32(VT)
    P.ad_inv_DT = _f32(1.0 / DT)
    P.ad_gLDT = _f32(gL * DT)
    P.v_rest = _f32(EL)
    P.v_reset = _f32(EL)
    P.v_th = _f32(VT)
    P.v_peak = _f32(20.0)
    return ModelSpec(
        key="eif", mid=M_EIF, label="EIF (exponential I&F)",
        citation="Fourcaud-Trocme et al. 2003, J. Neurosci. 23:11628",
        equation='C·v′ = −g_L(v−E_L) + g_LΔ_T·e^((v−V_T)/Δ_T) + I',
        note=("AdEx with the adaptation removed: the spike-initiation "
              "nonlinearity alone. Derived as the reduction of a "
              "Hodgkin-Huxley sodium current, so the exponential is not a "
              "convenience -- it is what the real spike onset looks like."),
        n_aux=0, params=P,
        rest=np.array([EL], np.float32),
        init=np.array([EL], np.float32),
        flops="~9 elided / ~21 full", exact="no (first order)")


def _b_qif(dt, n_sub=2):
    P = _base(dt)
    # Normal form scaled so rest and the unstable fixed point bracket the
    # reference's own 7 mV threshold gap.
    vc = V_TH
    P.qif_vc = _f32(vc)
    P.qif_k = _f32(1.0 / (TAU_M * (V_TH - V_REST)))
    P.n_sub = n_sub
    P.izh_dt = _f32(dt / n_sub)
    P.v_peak = _f32(V_TH + 25.0)
    return ModelSpec(
        key="qif", mid=M_QIF, label="QIF / theta neuron",
        citation="Ermentrout & Kopell 1986, SIAM J. Appl. Math. 46:233",
        equation='τ·v′ = k(v − v_rest)(v − v_c) + I',
        note=("The canonical form every type-I excitable neuron reduces to "
              "near its saddle-node-on-invariant-circle bifurcation. Cheapest "
              "model here that still has a genuine spike-generating "
              "nonlinearity: two FMAs and no transcendental."),
        n_aux=0, params=P,
        rest=np.array([V_REST], np.float32),
        init=np.array([V_REST], np.float32),
        flops="~5", exact="no (first order)")


def _b_raf(dt):
    P = _base(dt)
    b_damp = -1.0 / 10.0        # 10 ms envelope
    omega = 2 * math.pi * 0.05  # 50 Hz sub-threshold resonance
    decay = math.exp(b_damp * dt)
    P.raf_rr = _f32(decay * math.cos(omega * dt))
    P.raf_ri = _f32(decay * math.sin(omega * dt))
    # (e^{lambda dt} - 1)/lambda: the exact response to an input held constant
    # over the step. Adding the input undivided would make the model disagree
    # with its own ODE by a factor of dt -- caught by the Brian 2 cross-check,
    # where the resonator's PSP came out ten times too small.
    lam = complex(b_damp, omega)
    M = (complex(math.cos(omega * dt), math.sin(omega * dt)) * decay - 1) / lam
    P.raf_mr = _f32(M.real)
    P.raf_mi = _f32(M.imag)
    P.v_rest = _f32(0.0)
    P.v_reset = _f32(0.0)
    P.v_th = _f32(1.0)
    P.v_peak = _f32(1.0)
    return ModelSpec(
        key="raf", mid=M_RAF, label="Resonate-and-fire",
        citation="Izhikevich 2001, Neural Netw. 14:883",
        equation='z′ = (b + iω)z + I      fire when Im z > a',
        note=("A damped oscillator rather than an integrator: it responds to "
              "input FREQUENCY, not just amplitude, and can be inhibited into "
              "firing. The sub-threshold flow is a scaled rotation and the "
              "response to a constant input over the step has a closed form "
              "too, so BOTH halves are exact -- six multiplies and four adds, "
              "no transcendental at run time and no discretisation error at "
              "all below threshold."),
        n_aux=1, params=P, aux_names=("y",),
        rest=np.array([0.0, 0.0], np.float32),
        init=np.array([0.0, 0.0], np.float32),
        flops="~8", exact="yes (sub-threshold, incl. input)")


def _b_hh(dt, n_sub=4):
    P = _base(dt)
    gNa, gK, gL = 120.0, 36.0, 0.3          # mS/cm2
    ENa, EK, ELk = 50.0, -77.0, -54.387     # mV
    Cm = 1.0                                # uF/cm2
    P.n_sub = n_sub
    h = dt / n_sub
    P.hh_dt = _f32(h)
    P.hh_dtC = _f32(h / Cm)
    P.hh_gNa, P.hh_gK, P.hh_gL = _f32(gNa), _f32(gK), _f32(gL)
    P.hh_ENa, P.hh_EK, P.hh_EL = _f32(ENa), _f32(EK), _f32(ELk)

    v_hh, m0, h0, n0 = _hh_rest(gNa, gK, gL, ENa, EK, ELk)
    # Put HH's resting potential on the connectome's -52 mV scale so that the
    # atlas, the readouts and the stimulation code do not have to special-case
    # it. This is a shift of the voltage axis, not a change to the dynamics.
    shift = V_REST - v_hh
    P.hh_shift = _f32(shift)
    P.v_rest = _f32(v_hh + shift)
    P.v_th = _f32(0.0 + shift)        # detect the action potential upstroke
    P.v_reset = _f32(-50.0 + shift)   # re-arm once repolarised below this
    P.v_peak = _f32(40.0 + shift)
    return ModelSpec(
        key="hh", mid=M_HH, label="Hodgkin-Huxley",
        citation="Hodgkin & Huxley 1952, J. Physiol. 117:500",
        equation='C·v′ = −ḡ_Na·m³h(v−E_Na) − ḡ_K·n⁴(v−E_K) − g_L(v−E_L) + I',
        note=("The full conductance model, four state variables and six "
              "voltage-dependent rate functions. The gates use Rush & Larsen "
              "1978 exponential integration borrowed from cardiac "
              "electrophysiology: forward Euler on m needs dt well under 10 us "
              "because tau_m falls to ~0.05 ms during the upstroke, whereas "
              "Rush-Larsen is unconditionally stable for the gates. That is "
              "what makes 25 us substeps -- and therefore whole-connectome HH "
              "-- possible at all. It is still by far the most expensive model "
              "here, and the benchmark says so."),
        n_aux=4, params=P, aux_names=("m", "h", "n", "armed"),
        rest=np.array([v_hh + shift, m0, h0, n0, 1.0], np.float32),
        init=np.array([v_hh + shift, m0, h0, n0, 1.0], np.float32),
        flops="~180", exact="no (gates exact, v first order)")


def _hh_rest(gNa, gK, gL, ENa, EK, EL):
    """Resting potential and gate steady states, by fixed-point iteration.

    Solved rather than quoted: the textbook -65 mV is for the original squid
    parameters and shifts as soon as any conductance is changed, and the tile
    skipping machinery needs the state the model ACTUALLY settles to, to the
    last bit, or a resting neuron would never be recognised as inert.
    """
    v = -65.0
    for _ in range(20000):
        m = _inf(0.1 * _vtrap(v + 40, 10), 4 * math.exp(-(v + 65) / 18))
        hh = _inf(0.07 * math.exp(-(v + 65) / 20),
                  1 / (1 + math.exp(-(v + 35) / 10)))
        n = _inf(0.01 * _vtrap(v + 55, 10), 0.125 * math.exp(-(v + 65) / 80))
        I = (gNa * m ** 3 * hh * (ENa - v) + gK * n ** 4 * (EK - v)
             + gL * (EL - v))
        v += 0.01 * I
        if abs(I) < 1e-14:
            break
    return v, m, hh, n


def _inf(a, b):
    return a / (a + b)


def _vtrap(x, k):
    """x / (1 - exp(-x/k)), with the removable singularity at x = 0 handled.
    Only used to derive HH's starting parameters; the kernel has its own
    vectorised copy, and find_rest() refines the result against that one."""
    if abs(x) < 1e-6:
        return k + x / 2.0
    return x / (1.0 - math.exp(-x / k))


def _b_glif(dt):
    P = _base(dt)
    P.reset_g = 1
    tau_theta = 30.0
    P.gl_thdecay = _f32(math.exp(-dt / tau_theta))
    P.gl_th_inf = _f32(V_TH)
    P.gl_th_jump = _f32(1.2)
    P.v_th = _f32(V_TH)
    P.v_peak = _f32(V_TH)
    return ModelSpec(
        key="glif", mid=M_GLIF, label="GLIF (adaptive threshold)",
        citation="Teeter et al. 2018, Nat. Commun. 9:709 (Allen Institute GLIF)",
        equation='u ← αₘ·u + P_vg·g      θ ← θ_∞ + (θ−θ_∞)e^(−dt/τ_θ)      spike → θ += Δθ',
        note=("Spike-frequency adaptation for the price of one extra stream. "
              "The membrane is exactly LIF-exact; only the threshold moves, and "
              "since the threshold equation is also linear and autonomous it is "
              "advanced by its exact exponential too. The Allen Institute's "
              "GLIF family showed this captures most of what an adapting "
              "cortical neuron does without leaving the linear regime."),
        n_aux=1, params=P, aux_names=("theta",),
        rest=np.array([V_REST, V_TH], np.float32),
        init=np.array([V_REST, V_TH], np.float32),
        flops="~7", exact="yes (both states)")


_BUILDERS = {
    "lif_euler": _b_lif_euler,
    "lif_exact": _b_lif_exact,
    "izhikevich": _b_izh,
    "adex": _b_adex,
    "eif": _b_eif,
    "qif": _b_qif,
    "raf": _b_raf,
    "hh": _b_hh,
    "glif": _b_glif,
}

ALL_MODELS = tuple(_BUILDERS)
