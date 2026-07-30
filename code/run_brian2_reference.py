"""
Brian2 ground-truth validation harness for the whole-brain Drosophila LIF model
(Shiu et al. 2024, Nature; FlyWire v783).

This is a SEPARATE, self-contained reference implementation. It exists to
answer one question: how much does the repo's fast PyTorch/native backend
(``code/run_pytorch.py``, ``flyloop/brain_engine.py``, ``flyloop/native_engine.py``)
diverge from Brian2 -- the tool the original Shiu et al. model was published
with -- and specifically how much of that divergence is due to forward-Euler
integration (the PyTorch/native default) vs. exact linear integration (Brian2's
``method='auto'`` choice for a linear system, exposed here as ``method='exact'``).

Relationship to ``code/run_brian2_cuda.py``: that file already builds a Brian2
model (``method='linear'``, i.e. exact) and is treated as the project's
ground truth in ``code/compare_ground_truth.py``. This script does NOT modify
or replace it. It exists alongside for three reasons that file doesn't cover:

  1. It supports a SUBNETWORK mode (first N neurons + edges among them) so the
     model can be validated in seconds, not minutes, before attempting the
     full 138,639-neuron / 15,091,983-synapse build.
  2. It supports BOTH ``method='exact'`` and ``method='euler'`` from the same
     code path, so the exact-vs-Euler gap can be measured with everything else
     (data loading, weight construction, Poisson injection, refractory
     handling) held identical.
  3. It writes output in the flyloop-native spike schema
     (time_ms, neuron_index, flywire_id) used by
     ``flyloop/brain_engine.py``'s / ``flyloop/native_engine.py``'s
     ``spikes_dataframe()``, not the legacy benchmark schema
     (t, time_ms, trial, neuron_index, flywire_id, exp_name).

Model (must match code/run_pytorch.py MODEL_PARAMS and
flyloop/brain_engine.py MODEL_PARAMS exactly):

    dg/dt = -g/tau_syn                    tau_syn = 5 ms
    dv/dt = (g - (v - v_rest))/tau_mem    tau_mem = 20 ms
    on spike: v -> v_reset, g -> 0
    presynaptic spike adds w to g after a UNIFORM 1.8 ms axonal delay.

    v_rest = v_reset = -52 mV, v_threshold = -45 mV
    tau_mem = 20 ms, tau_syn = 5 ms
    refractory = 2.2 ms, ABSOLUTE (clamps both dv/dt and dg/dt)
    dt = 0.1 ms
    w = 0.275 mV x integer synapse count ("Excitatory x Connectivity")
    Poisson drive: stimulated neurons get a direct +250*0.275 mV = +68.75 mV
    kick to v (NOT routed through g) on each Bernoulli(rate*dt) success, and
    are permanently non-refractory (rfc = 0 ms).

Usage:
    # tiny smoke test, subnetwork of the first 2000 neurons
    .venv\\Scripts\\python.exe code\\run_brian2_reference.py --n-neurons 2000 --duration-ms 100

    # full brain, exact integration, 100 ms
    .venv\\Scripts\\python.exe code\\run_brian2_reference.py --duration-ms 100 --method exact

    # full brain, forward Euler, for the euler-vs-exact comparison
    .venv\\Scripts\\python.exe code\\run_brian2_reference.py --duration-ms 100 --method euler

Output: data/results/brian2_reference_<tag>.parquet with columns
    time_ms, neuron_index, flywire_id
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# benchmark.py is the project's existing single source of truth for data
# paths and the 'sugar' / 'p9' experiment definitions (stim neuron flywire
# ids + rate). Reused here read-only; this script never imports torch.
try:
    from benchmark import get_experiment, path_comp, path_con, path_res
except Exception:  # pragma: no cover - defensive fallback if run standalone
    path_comp = (_HERE / '../data/2025_Completeness_783.csv').resolve()
    path_con = (_HERE / '../data/2025_Connectivity_783.parquet').resolve()
    path_res = (_HERE / '../data/results').resolve()

    def get_experiment(name=None):
        # Embedded copy of benchmark.py's 'sugar' experiment, kept in sync
        # manually. Only used if importing benchmark.py fails.
        return {
            'key': 'sugar',
            'name': 'Sugar GRNs (200 Hz)',
            'neu_exc': [
                720575940624963786, 720575940630233916, 720575940637568838,
                720575940638202345, 720575940617000768, 720575940630797113,
                720575940632889389, 720575940621754367, 720575940621502051,
                720575940640649691, 720575940639332736, 720575940616885538,
                720575940639198653, 720575940639259967, 720575940617937543,
                720575940632425919, 720575940633143833, 720575940612670570,
                720575940628853239, 720575940629176663, 720575940611875570,
            ],
            'neu_exc2': [], 'neu_slnc': [], 'stim_rate': 200.0,
        }

from brian2 import (  # noqa: E402
    NeuronGroup, Synapses, PoissonInput, SpikeMonitor, Network,
    mV, ms, Hz, defaultclock, seed as brian2_seed, prefs,
)
import logging  # noqa: E402
from brian2.utils.logger import BrianLogger  # noqa: E402
BrianLogger.console_handler.setLevel(logging.WARNING)

# ============================================================================
# Model parameters -- must match code/run_pytorch.py MODEL_PARAMS and
# flyloop/brain_engine.py MODEL_PARAMS exactly.
# ============================================================================

V_REST = -52 * mV
V_RESET = -52 * mV
V_THRESHOLD = -45 * mV
TAU_MEM = 20 * ms
TAU_SYN = 5 * ms
T_REFRAC = 2.2 * ms
T_DELAY = 1.8 * ms
W_SCALE = 0.275 * mV
SCALE_POISSON = 250          # dimensionless multiplier, see get_weights() note below
DT = 0.1 * ms

DEFAULT_EXPERIMENT = 'sugar'

# The differential system. Both variables carry "(unless refractory)" per the
# spec: during the 2.2 ms absolute refractory period neither v nor g may
# evolve via their ODE. v_reset == v_rest in this parameterization, so this
# is *also* what run_pytorch.py achieves, but by a different, more implicit
# route (see the "CRITICAL GAP" comment on REFRACTORY_GATE_SYNAPSES below).
EQS = dedent('''
    dv/dt = (g - (v - v_rest)) / tau_mem : volt (unless refractory)
    dg/dt = -g / tau_syn                 : volt (unless refractory)
    rfc                                  : second
''')

NAMESPACE = dict(
    v_rest=V_REST, tau_mem=TAU_MEM, tau_syn=TAU_SYN,
)

THRESHOLD = 'v > v_threshold'
RESET = 'v = v_reset; g = 0*mV'
RESET_NAMESPACE = dict(v_reset=V_RESET, v_threshold=V_THRESHOLD)


def _peak_memory_mb():
    """Peak working-set size (MB) of this process, via the Windows psapi API.

    stdlib-only (no psutil in this venv). Returns None off Windows or on any
    failure -- this is a diagnostic, not something the harness depends on.
    """
    try:
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # Without explicit argtypes/restype, ctypes assumes 32-bit int
        # marshaling for the pseudo-HANDLE and the call silently fails
        # (returns FALSE) on 64-bit Python -- must be declared explicitly.
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.kernel32.GetCurrentProcess.argtypes = []
        ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb)
        if ok:
            return counters.PeakWorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    return None


def get_hash_tables(comp_path):
    """flywire id <-> tensor index mappings, same convention as run_pytorch.py."""
    df_comp = pd.read_csv(comp_path, index_col=0)
    i2flyid = np.asarray(df_comp.index, dtype=np.int64)
    flyid2i = {int(j): i for i, j in enumerate(i2flyid)}
    return flyid2i, i2flyid


def build_network(n_neurons=None, stim_ids=None, stim_rate=None, method='exact',
                   data_dir=None, seed=0, record_spikes=True, dt=DT,
                   codegen_target='numpy', refractory_gate_synapses=False,
                   experiment=DEFAULT_EXPERIMENT, verbose=True):
    """Build the Brian2 network.

    Parameters
    ----------
    n_neurons : int or None
        None -> full brain (138,639 neurons, all 15,091,983 edges).
        int  -> SUBNETWORK mode: take the first ``n_neurons`` rows of
                2025_Completeness_783.csv (row order = neuron index) and only
                the edges where BOTH endpoints fall inside that range. Lets
                the model be validated in seconds before attempting the full
                build.
    stim_ids : list[int] or None
        Flywire ids to drive with Poisson input. None -> the experiment's
        ``neu_exc`` list (default 'sugar': 21 GRNs). In subnetwork mode, ids
        outside the first n_neurons are dropped; if that empties the list
        entirely (e.g. n_neurons=2000, whose max sugar-GRN index is 129,730),
        falls back to stimulating neuron indices 0..min(5,n_neurons)-1 with a
        printed warning, so a small subnetwork still has activity to check.
    stim_rate : float or None
        Poisson rate in Hz. None -> the experiment's stim_rate.
    method : 'exact' or 'euler'
        Brian2 StateUpdateMethod for the (v, g) system. 'exact' is Brian2's
        analytic/closed-form solution for the linear ODE system (equivalent
        to the Rotter & Diesmann propagator in run_pytorch.py's INTEGRATION=
        'exact' branch). 'euler' is standard forward Euler, matching
        run_pytorch.py's default INTEGRATION='euler' bit-for-bit in structure
        (though not bit-for-bit in float value, since Brian2 uses float64
        internally and PyTorch/native use float32 -- see module docstring
        "gaps" section printed by main() and item (d) in the task report).
    codegen_target : 'numpy' | 'cython' | 'auto'
        Brian2 runtime codegen backend. Default 'numpy': this machine has no
        cl.exe/gcc on PATH, so 'cython' (needs a C/C++ compiler to build the
        generated extension) is not usable, and 'auto' would waste time
        probing for one before falling back. Results are IDENTICAL regardless
        of codegen target -- this only affects speed.
    refractory_gate_synapses : bool
        See the long comment below. False (default) matches the task's
        literal spec ``on_pre='g_post += w'``. True switches to
        ``on_pre='g_post += w * int(not_refractory_post)'``, which is closer
        to run_pytorch.py's actual per-step behavior of *dropping* (not
        merely deferring) synaptic input that arrives while the postsynaptic
        neuron is refractory. This is the most important item in (d) of the
        deliverable -- see the docstring section "GAP: refractory synaptic
        delivery" below.

    Returns
    -------
    (net, neu, syn, poisson_inputs, spike_mon, info) where info is a dict of
    metadata (N, i2flyid, stim_idx, stim_rate, n_synapses, method, ...).
    """
    # ------------------------------------------------------------------
    # GAP: refractory synaptic delivery (see also module-level docstring
    # and the (d) section of the report this script's author returns).
    #
    # run_pytorch.py / flyloop/brain_engine.py gate the DELAYED synaptic
    # input at delivery time: if the postsynaptic neuron is refractory at
    # the moment a delayed spike would arrive, that input is dropped
    # forever (see brain_engine.step_inplace: `gate = refrac >= refrac_steps`
    # multiplies the delayed buffer read before it's folded into g). It is
    # NOT merely postponed.
    #
    # Standard Brian2 `(unless refractory)` semantics only pause the ODE
    # INTEGRATION of a variable during refractoriness; they do not gate
    # direct/event-based writes to that variable, such as a Synapses
    # on_pre statement. With the literal spec's on_pre='g_post += w', a
    # synaptic event that arrives while the postsynaptic neuron is
    # refractory WILL still increment g (it just won't decay/leak while
    # (unless refractory) is in force) and will still affect v once the
    # refractory period ends -- exactly the behavior run_pytorch.py's gate
    # exists to prevent.
    #
    # Because v_reset == v_rest in this parameterization, run_pytorch.py's
    # trick (zero g at spike, gate out new arrivals, so v never has
    # anything to integrate while refractory) and Brian2's `(unless
    # refractory)` clamp produce IDENTICAL results for a postsynaptic
    # neuron that receives NO synaptic input during its refractory window.
    # They diverge only when input DOES arrive during that window -- which,
    # for a real GRN-driven cortex-scale run with recurrent connectivity, is
    # not a rare edge case.
    # ------------------------------------------------------------------
    on_pre = 'g_post += w'
    if refractory_gate_synapses:
        on_pre = 'g_post += w * int(not_refractory_post)'

    prefs.codegen.target = codegen_target
    defaultclock.dt = dt
    brian2_seed(int(seed))

    data_dir = Path(data_dir) if data_dir else Path(path_comp).parent
    comp_path = data_dir / '2025_Completeness_783.csv'
    con_path = data_dir / '2025_Connectivity_783.parquet'

    flyid2i_full, i2flyid_full = get_hash_tables(comp_path)
    N = int(n_neurons) if n_neurons is not None else len(i2flyid_full)
    i2flyid = i2flyid_full[:N]

    con = pd.read_parquet(
        con_path,
        columns=['Presynaptic_Index', 'Postsynaptic_Index',
                 'Excitatory x Connectivity'],
    )
    if n_neurons is not None:
        mask = (con['Presynaptic_Index'].to_numpy() < N) & \
               (con['Postsynaptic_Index'].to_numpy() < N)
        con = con.loc[mask]

    exp = get_experiment(experiment)
    if stim_ids is None:
        stim_ids = exp['neu_exc']
    if stim_rate is None:
        stim_rate = exp['stim_rate']

    stim_idx = sorted(
        flyid2i_full[i] for i in stim_ids
        if i in flyid2i_full and flyid2i_full[i] < N
    )
    fallback_stim = False
    if not stim_idx:
        fallback_stim = True
        k = min(5, N)
        stim_idx = list(range(k))
        if verbose:
            print(
                f"[build_network] none of the {len(stim_ids)} requested stim "
                f"ids fall within the first {N} neurons (subnetwork mode); "
                f"falling back to stimulating neuron indices 0..{k - 1} at "
                f"{stim_rate} Hz so the subnetwork has activity to check."
            )

    namespace = dict(NAMESPACE)
    namespace.update(RESET_NAMESPACE)

    neu = NeuronGroup(
        N, model=EQS, method=method,
        threshold=THRESHOLD, reset=RESET,
        refractory='rfc', name='neurons', namespace=namespace,
    )
    neu.v = V_REST
    neu.g = 0 * mV
    neu.rfc = T_REFRAC
    if stim_idx:
        # Stimulated (sensory-injection) neurons are permanently
        # non-refractory, matching run_pytorch.py's
        # `refrac_steps[exc_indices] = 0`.
        neu.rfc[np.asarray(stim_idx)] = 0 * ms

    syn = Synapses(neu, neu, 'w : volt', on_pre=on_pre, delay=T_DELAY,
                    name='synapses')
    n_synapses = len(con)
    if n_synapses:
        syn.connect(i=con['Presynaptic_Index'].to_numpy(),
                    j=con['Postsynaptic_Index'].to_numpy())
        # w = 0.275 mV x integer synapse count. "Excitatory x Connectivity"
        # already carries the sign (excitatory positive / inhibitory
        # negative), same column run_pytorch.py's get_weights() uses
        # directly as the sparse matrix values.
        syn.w = con['Excitatory x Connectivity'].to_numpy() * W_SCALE

    # Poisson drive: PoissonInput with N=1 draws one independent Bernoulli
    # trial per target neuron per timestep with success probability
    # rate*dt -- the same per-step Bernoulli scheme
    # run_pytorch.py's PoissonSpikeGenerator uses
    # (torch.bernoulli(rates * dt/1000)). weight = w_syn * f_poi =
    # 0.275 mV * 250 = 68.75 mV, added DIRECTLY to v (target_var='v'), NOT
    # routed through g -- matching run_pytorch.py's
    # `voltage_stim = wScale * poisson_spikes` being added straight into v
    # in LIFNeuron.forward, bypassing the synapse/conductance path entirely.
    #
    # One PoissonInput object per neuron (not a single vectorized one over a
    # fancy-indexed subgroup) because Brian2 Subgroups must be contiguous
    # index ranges, and the stimulated ids are not contiguous in general.
    # This mirrors code/run_brian2_cuda.py's proven pattern exactly.
    poisson_inputs = []
    if stim_idx:
        weight = W_SCALE * SCALE_POISSON
        for i in stim_idx:
            poisson_inputs.append(PoissonInput(
                target=neu[i], target_var='v', N=1,
                rate=stim_rate * Hz, weight=weight,
            ))

    spike_mon = SpikeMonitor(neu) if record_spikes else None

    objects = [neu, syn, *poisson_inputs]
    if spike_mon is not None:
        objects.append(spike_mon)
    net = Network(*objects)

    info = dict(
        N=N, i2flyid=i2flyid, stim_idx=stim_idx, stim_rate=stim_rate,
        n_synapses=n_synapses, method=method, fallback_stim=fallback_stim,
        refractory_gate_synapses=refractory_gate_synapses,
        codegen_target=codegen_target, dt=dt, experiment=exp['key'],
    )
    return net, neu, syn, poisson_inputs, spike_mon, info


def spikes_to_dataframe(spike_mon, i2flyid):
    """Match flyloop/native_engine.py's spikes_dataframe() schema exactly:
    columns time_ms, neuron_index, flywire_id."""
    if spike_mon is None or spike_mon.num_spikes == 0:
        return pd.DataFrame(columns=['time_ms', 'neuron_index', 'flywire_id'])
    idx = np.asarray(spike_mon.i, dtype=np.int64)
    t_ms = np.asarray(spike_mon.t / ms, dtype=np.float64)
    return pd.DataFrame({
        'time_ms': t_ms,
        'neuron_index': idx,
        'flywire_id': i2flyid[idx],
    })


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Brian2 ground-truth validation harness for the '
                     'whole-brain Drosophila LIF model.',
    )
    parser.add_argument('--n-neurons', type=int, default=None,
                         help='Subnetwork size (first N neurons + edges '
                              'among them). Omit for the full brain '
                              '(138,639 neurons / 15,091,983 synapses).')
    parser.add_argument('--duration-ms', type=float, default=100.0,
                         help='Simulated duration in ms (default: 100).')
    parser.add_argument('--method', choices=['exact', 'euler'],
                         default='exact',
                         help="Brian2 integration method for the (v,g) "
                              "system (default: exact).")
    parser.add_argument('--experiment', default=DEFAULT_EXPERIMENT,
                         choices=['sugar', 'p9'],
                         help="Which benchmark.py experiment's stim ids/rate "
                              "to use by default (default: sugar).")
    parser.add_argument('--stim-rate', type=float, default=None,
                         help='Override the Poisson drive rate in Hz.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tag', type=str, default=None,
                         help='Output filename tag. Default: '
                              'n<N-or-full>_<method>.')
    parser.add_argument('--codegen-target', default='numpy',
                         choices=['numpy', 'cython', 'auto'],
                         help='Brian2 runtime codegen backend (default: '
                              'numpy -- no C/C++ compiler is on PATH on '
                              'this machine, so cython/auto would just '
                              'fail or waste time probing).')
    parser.add_argument('--refractory-gate-synapses', action='store_true',
                         help='Use on_pre="g_post += w * '
                              'int(not_refractory_post)" instead of the '
                              'literal spec on_pre="g_post += w". See the '
                              'GAP comment in build_network(). Off by '
                              'default to match the literal task spec.')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--no-save', action='store_true',
                         help='Run but do not write the output parquet.')
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    net, neu, syn, poisson_inputs, spike_mon, info = build_network(
        n_neurons=args.n_neurons,
        stim_rate=args.stim_rate,
        method=args.method,
        seed=args.seed,
        codegen_target=args.codegen_target,
        refractory_gate_synapses=args.refractory_gate_synapses,
        data_dir=args.data_dir,
        experiment=args.experiment,
    )
    build_s = time.perf_counter() - t0

    print(f"Brian2 codegen target : {prefs.codegen.target}")
    print(f"Neurons               : {info['N']:,}")
    print(f"Synapses              : {info['n_synapses']:,}")
    print(f"Method                : {info['method']}")
    print(f"Refractory-gated syn  : {info['refractory_gate_synapses']}")
    print(f"Stimulated neurons    : {len(info['stim_idx'])} "
          f"@ {info['stim_rate']} Hz"
          f"{'  [FALLBACK STIM, see warning above]' if info['fallback_stim'] else ''}")
    print(f"Build wall time       : {build_s:.3f} s")

    t1 = time.perf_counter()
    net.run(args.duration_ms * ms)
    run_s = time.perf_counter() - t1

    n_spikes = int(spike_mon.num_spikes) if spike_mon is not None else 0
    peak_mb = _peak_memory_mb()

    per_100ms = run_s * (100.0 / args.duration_ms)
    per_1000ms = run_s * (1000.0 / args.duration_ms)
    print(f"Run wall time         : {run_s:.3f} s "
          f"for {args.duration_ms:.1f} ms simulated")
    print(f"  -> normalized        : {per_100ms:.3f} s / 100ms-simulated "
          f"({per_1000ms:.3f} s / 1000ms-simulated)")
    print(f"Spikes                : {n_spikes:,}")
    if peak_mb is not None:
        print(f"Peak working set      : {peak_mb:.1f} MB")
    else:
        print("Peak working set      : unavailable "
              "(GetProcessMemoryInfo failed)")

    tag = args.tag or f"n{info['N']}_{info['method']}"
    if not args.no_save:
        df = spikes_to_dataframe(spike_mon, info['i2flyid'])
        out_dir = Path(path_res)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"brian2_reference_{tag}.parquet"
        df.to_parquet(out_path, compression='brotli')
        print(f"Saved {len(df):,} spikes to {out_path}")

    return dict(info=info, build_s=build_s, run_s=run_s, n_spikes=n_spikes,
                peak_mb=peak_mb)


if __name__ == '__main__':
    main()
