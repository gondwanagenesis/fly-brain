"""
PyTorch benchmark runner for the Drosophila brain model.

Implements the LIF neuron model with alpha-function synapses using PyTorch,
with support for both CPU and CUDA GPU computation. Batches n_run trials
in parallel for efficient GPU utilization.

Model architecture (from Shiu et al.):
    PoissonSpikeGenerator → recurrent weights (sparse matmul) → AlphaLIF
    where AlphaLIF = AlphaSynapse + LIFNeuron + refractory period

Called by benchmark.py orchestrator.
"""

import pandas as pd
import pyarrow  # noqa: F401  — must be imported before torch to avoid libarrow conflict
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from time import perf_counter as time
import traceback

from benchmark import (
    T_RUN_VALUES_SEC, N_RUN_VALUES,
    path_comp, path_con, path_wt,
    get_experiment, get_spike_output_path, print_summary_table, save_result_csv,
    spike_io_enabled,
)

# ============================================================================
# PyTorch Model Parameters (matching Brian2 default_params)
# ============================================================================

MODEL_PARAMS = {
    'tauSyn': 5.0,        # ms
    'tDelay': 1.8,        # ms
    'v0': -52.0,          # mV
    'vReset': -52.0,      # mV
    'vRest': -52.0,       # mV
    'vThreshold': -45.0,  # mV
    'tauMem': 20.0,       # ms
    'tRefrac': 2.2,       # ms
    'scalePoisson': 250,
    'wScale': 0.275,
}

DT = 0.1  # Simulation timestep in ms (matches Brian2 defaultclock.dt)

# Integration scheme.
#
# 'euler' reproduces the original forward-Euler update exactly (bit-identical).
# 'exact' uses the Rotter & Diesmann (1999) matrix-exponential propagator.
#
# Why 'exact' matters: the subthreshold system is linear and time-invariant, so
# it has a closed-form propagator. Brian2 -- which the reference Shiu et al.
# model was run in -- selects that automatically for linear equations
# (method='auto' -> 'exact'). Forward Euler is therefore NOT the published
# baseline: it is the exact solution of a slightly different model.
#
# Forward Euler at dt=0.1 ms shortens every time constant by exactly dt/2:
#     tau_syn 5.000 -> 4.950 ms  (-1.00%)
#     tau_mem 20.000 -> 19.950 ms (-0.25%)
# and inflates the g->v coupling from the true 0.00493794 to 0.005 (+1.26%).
# Net effect: total synaptic charge is preserved to ~1e-5 (so firing *rates*
# are fine), but PSP peak amplitude is +0.47% and the whole simulation runs
# 0.25-1% fast -- a 0.25-0.5 ms timing error over a 50 ms response, which is an
# order of magnitude larger than the dt/2 grid quantisation.
#
# The exact form costs the same number of FLOPs (3 mul + 2 add vs 2 mul +
# 3 add), so there is no performance reason to keep Euler.
#
# NOTE: 'exact' deliberately changes numerics, so it is NOT bit-identical to
# 'euler'. It is opt-in for that reason; flipping the default is a decision for
# the maintainers, ideally after a compare_ground_truth.py run against Brian2.
INTEGRATION = 'euler'


def propagator_constants(dt, params):
    """Rotter & Diesmann (1999) propagator for the (v, g) subsystem.

    The Jacobian is lower-triangular (g drives v, v does not drive g), so the
    matrix exponential has a closed form:

        alpha_m = exp(-dt / tauMem)
        alpha_s = exp(-dt / tauSyn)
        P_vg    = tauSyn / (tauMem - tauSyn) * (alpha_m - alpha_s)

    Update:  g <- alpha_s * g
             v <- alpha_m * v + P_vg * g + vRest * (1 - alpha_m)

    P_vg is a difference of two nearly equal exponentials (~6 bits of
    cancellation), so it must be evaluated in double precision on the host and
    passed down as a constant -- never recomputed in reduced precision.
    """
    import math
    tau_m, tau_s = float(params['tauMem']), float(params['tauSyn'])
    alpha_m = math.exp(-dt / tau_m)
    alpha_s = math.exp(-dt / tau_s)
    if abs(tau_m - tau_s) < 1e-12:          # degenerate tau_m == tau_s limit
        p_vg = (dt / tau_m) * alpha_m
    else:
        p_vg = tau_s / (tau_m - tau_s) * (alpha_m - alpha_s)
    return alpha_m, alpha_s, p_vg

# ============================================================================
# Model Classes
# ============================================================================

class PoissonSpikeGenerator(nn.Module):
    """Generates one timestep of Poisson-distributed spikes from firing rates."""

    def __init__(self, dt, scale, device='cpu'):
        super().__init__()
        self.prob_scale = dt / 1000.0
        self.scale = scale
        self.device = device

    def forward(self, rates, generator=None):
        return torch.bernoulli(rates * self.prob_scale, generator=generator) * self.scale


class AlphaSynapse(nn.Module):
    """Alpha-function synapse dynamics with configurable delay."""

    def __init__(self, batch, size, dt, params, device='cpu'):
        super().__init__()
        self.time_factor = dt / params['tauSyn']
        self.steps_delay = int(params['tDelay'] / dt)
        self.size = size
        self.device = device
        self.batch = batch

    def state_init(self):
        conductance = torch.zeros(self.batch, self.size, device=self.device)
        delay_buffer = torch.zeros(
            self.batch, self.steps_delay + 1, self.size, device=self.device
        )
        self.head = 0
        return conductance, delay_buffer

    def forward(self, input_, conductance, delay_buffer, refrac):
        # PERF: ring buffer instead of torch.roll.
        #
        # torch.roll rewrites the whole (batch, steps_delay+1, size) buffer
        # every timestep -- 10.5 MB per step, ~105 GB of memcpy per simulated
        # second at dt=0.1ms. The slot we just consumed is exactly the one that
        # roll would move to the end, so reading and then overwriting it in
        # place is equivalent and costs O(size) instead of O(size*steps_delay).
        conductance_new = (
            conductance * (1 - self.time_factor)
            + delay_buffer[:, self.head, :] * refrac
        )
        delay_buffer[:, self.head, :] = input_
        self.head = (self.head + 1) % (self.steps_delay + 1)
        return conductance_new, delay_buffer


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with surrogate gradient (ATan)."""

    def __init__(self, batch, size, dt, params, device='cpu'):
        super().__init__()
        self.size = size
        self.dt = dt
        self.tau_mem = params['tauMem']
        self.v_reset = params['vReset']
        self.v_rest = params['vRest']
        self.v_threshold = params['vThreshold']
        self.v_0 = params['v0']
        self.time_factor = dt / self.tau_mem
        self.spike_gradient = self.ATan.apply
        self.device = device
        self.batch = batch

    def state_init(self):
        v = torch.zeros(self.batch, self.size, device=self.device) + self.v_0
        spikes = torch.zeros(self.batch, self.size, device=self.device)
        return spikes, v

    def forward(self, conductance, voltage_stim, v):
        # Brian/GeNN:
        # V += Vstim
        # V += MemFactor * (G - (V - Vrest))

        v = v + voltage_stim
        v = v + self.time_factor * (conductance - (v - self.v_rest))

        spike = self.spike_gradient(v - self.v_threshold)

        reset = ((v - self.v_reset) * spike).detach()
        v = v - reset

        return spike, v

    @staticmethod
    class ATan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, v):
            spike = (v > 0).float()
            ctx.save_for_backward(v)
            return spike

        @staticmethod
        def backward(ctx, grad_output):
            (v,) = ctx.saved_tensors
            grad = 1 / (1 + (np.pi * v).pow_(2)) * grad_output
            return grad


class AlphaLIF(nn.Module):
    """LIF neuron with alpha-function synapse dynamics and refractory period."""

    def __init__(
        self,
        batch,
        size,
        dt,
        params,
        exc_indices=None,
        device='cpu'
    ):
        super().__init__()
        self.size = size
        self.synapse = AlphaSynapse(batch, size, dt, params, device=device)
        self.neuron = LIFNeuron(batch, size, dt, params, device=device)
        base_refrac = int(round(params['tRefrac'] / dt))

        self.refrac_steps = torch.full(
            (size,),
            base_refrac,
            dtype=torch.long,
            device=device,
        )

        if exc_indices is not None:
            self.refrac_steps[exc_indices] = 0

    def state_init(self):
        conductance, delay_buffer = self.synapse.state_init()
        spikes, v = self.neuron.state_init()
        refrac = self.refrac_steps.unsqueeze(0).repeat(self.neuron.batch, 1).float()
        return conductance, delay_buffer, spikes, v, refrac

    def forward(self,
            recurrent_input,
            voltage_stim,
            conductance,
            delay_buffer,
            spikes,
            v,
            refrac):
        refrac = torch.where(
            spikes > 0,
            torch.zeros_like(refrac),
            refrac + 1,
        )
        conductance_new, delay_buffer = self.synapse(
            recurrent_input,
            conductance,
            delay_buffer,
            (
                refrac >= self.refrac_steps.unsqueeze(0)
            ).float()
        )

        spikes, v_new = self.neuron(
            conductance,
            voltage_stim,
            v
        )
        conductance_reset = (conductance_new * spikes).detach()
        conductance_new = conductance_new - conductance_reset
        return conductance_new, delay_buffer, spikes, v_new, refrac


class TorchModel(nn.Module):
    """
    Top-level model: Poisson input + recurrent connectome weights + AlphaLIF.

    The weights tensor should be a sparse matrix (CSR or COO) derived from
    the Drosophila connectome.
    """

    def __init__(
            self,
            batch,
            size,
            dt,
            params,
            weights,
            exc_indices=None,
            device='cpu'
        ):
        super().__init__()
        self.neurons = AlphaLIF(
            batch,
            size,
            dt,
            params,
            exc_indices=exc_indices,
            device=device
        )
        self.weights = weights
        self.poisson = PoissonSpikeGenerator(dt, params['scalePoisson'], device=device)
        self.scale = params['wScale']

        # PERF: event-driven fan-out table (W in CSC == W.T in CSR).
        #
        # The dense form computes spikes @ W.T, i.e. all 15.1M non-zeros every
        # timestep, regardless of activity. Measured mean activity on the sugar
        # experiment is 1.75 spikes/step out of 138,639 neurons (0.0013%), so
        # >99.99% of that work multiplies by zero. Grouping the weights by
        # PREsynaptic index lets us touch only the synapses of neurons that
        # actually fired: ~190 multiply-adds instead of 15.1M.
        #
        # Measured 11.6x end-to-end with bit-identical spike trains.
        # Set event_driven=False to fall back to the dense path.
        self.event_driven = True
        w_coo = weights.to_sparse_coo().coalesce()
        post, pre = w_coo.indices()[0], w_coo.indices()[1]
        order = torch.argsort(pre)
        counts = torch.bincount(pre[order], minlength=weights.shape[0])
        crow = torch.zeros(weights.shape[0] + 1, dtype=torch.long, device=device)
        crow[1:] = torch.cumsum(counts, 0)
        self.fan_crow = crow
        self.fan_post = post[order].contiguous().to(device)
        self.fan_val = w_coo.values()[order].float().contiguous().to(device)

    def state_init(self):
        return self.neurons.state_init()

    def _recurrent(self, spikes):
        """Accumulate synaptic drive from the neurons that spiked this step."""
        if not self.event_driven:
            return torch.matmul(spikes, self.weights.transpose(0, 1))
        out = torch.zeros_like(spikes)
        for b in range(spikes.shape[0]):
            src = spikes[b].nonzero(as_tuple=True)[0]
            if not src.numel():
                continue
            start, end = self.fan_crow[src], self.fan_crow[src + 1]
            cnt = end - start
            total = int(cnt.sum())
            if not total:
                continue
            base = torch.repeat_interleave(start, cnt)
            ramp = torch.arange(total, device=spikes.device) - \
                torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
            sel = base + ramp
            out[b].scatter_add_(0, self.fan_post[sel], self.fan_val[sel])
        return out

    def forward(self, rates, conductance, delay_buffer, spikes, v, refrac, generator=None):
        poisson_spikes = self.poisson(
            rates,
            generator=generator
        )

        voltage_stim = self.scale * poisson_spikes

        weighted_spikes = self._recurrent(spikes)

        recurrent_input = self.scale * weighted_spikes

        conductance, delay_buffer, spikes, v, refrac = self.neurons(
            recurrent_input,
            voltage_stim,
            conductance,
            delay_buffer,
            spikes,
            v,
            refrac,
        )
        return conductance, delay_buffer, spikes, v, refrac

# ============================================================================
# Data Utilities
# ============================================================================

def get_hash_tables(comp_path):
    """Build flywire ID <-> tensor index mappings from completeness CSV."""
    df_comp = pd.read_csv(comp_path, index_col=0)
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}
    i2flyid = {j: i for i, j in flyid2i.items()}
    return flyid2i, i2flyid


def get_weights(conn_path, comp_path, wt_dir, csr=True):
    """Load or build sparse weight matrix from connectivity data.

    Caches weight_coo.pkl / weight_csr.pkl in wt_dir for reuse.
    """
    wt_dir = Path(wt_dir)
    coo_path = wt_dir / 'weight_coo.pkl'
    csr_path = wt_dir / 'weight_csr.pkl'

    data_conn = pd.read_parquet(conn_path)
    data_name = pd.read_csv(comp_path)
    num_neurons = data_name.shape[0]

    try:
        with open(coo_path, 'rb') as f:
            weight_coo = pickle.load(f)
    except FileNotFoundError:
        print('Weights not found, constructing COO weight matrix...')
        idx = [
            data_conn['Postsynaptic_Index'].to_list(),
            data_conn['Presynaptic_Index'].to_list(),
        ]
        val = data_conn['Excitatory x Connectivity'].to_list()
        weight_coo = torch.sparse_coo_tensor(
            idx, val, (num_neurons, num_neurons)
        ).to(torch.float32)
        with open(coo_path, 'wb') as f:
            pickle.dump(weight_coo, f)

    if csr:
        try:
            with open(csr_path, 'rb') as f:
                weight_csr = pickle.load(f)
        except FileNotFoundError:
            print('CSR weights not found, converting from COO...')
            weight_csr = weight_coo.to_sparse_csr()
            with open(csr_path, 'wb') as f:
                pickle.dump(weight_csr, f)
        return weight_csr
    else:
        return weight_coo

# ============================================================================
# Benchmark Functions
# ============================================================================

def run_single_benchmark(t_run_sec, n_run, experiment, logger,
                         run_idx=None, total_runs=None,
                         run_label=None, round_idx=None):
    """
    Run a single PyTorch benchmark with specified t_run and n_run.

    Uses batch_size = n_run to run all trials in parallel on GPU.
    """
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    t_sim_ms = t_run_sec * 1000.0
    num_steps = int(t_sim_ms / DT)

    exp_name = f'pytorch_t{t_run_sec}s_n{n_run}'

    run_info = f"[{run_idx}/{total_runs}] " if run_idx else ""
    logger.log_raw("")
    logger.log_raw("=" * 80)
    logger.log(f"{run_info}BENCHMARK: t_run={t_run_sec}s, n_run={n_run}")
    logger.log_raw("=" * 80)
    logger.log(f"Device: {device_name.upper()}")
    logger.log(f"Steps: {num_steps} (dt={DT}ms)")
    logger.log(f"Experiment: {exp_name}")
    record_spikes = spike_io_enabled()
    logger.log(
        "Spike probing/output: "
        f"{'enabled' if record_spikes else 'disabled'}"
    )
    if run_label:
        logger.log(f"Run label: {run_label}")
    if round_idx is not None:
        logger.log(f"Round: {round_idx}")

    stim_rate = experiment['stim_rate']

    timings = {}
    results = {}

    try:
        # ===== Phase 1: ID mappings =====
        t_mapping_start = time()
        flyid2i, i2flyid = get_hash_tables(str(path_comp))
        exc_indices = [flyid2i[n] for n in experiment['neu_exc']]
        timings['id_mapping'] = time() - t_mapping_start
        logger.log(f"ID mapping:         {timings['id_mapping']:.3f}s")

        # ===== Phase 2: Load weights =====
        logger.log("Loading weights...")
        t_weights_start = time()
        weights = get_weights(str(path_con), str(path_comp), str(path_wt), csr=True)
        weights = weights.to(device=device_name)
        num_neurons = weights.shape[0]
        timings['weight_loading'] = time() - t_weights_start
        logger.log(f"  Weight loading:   {timings['weight_loading']:.3f}s")
        logger.log(f"  Neurons: {num_neurons}, Batch: {n_run}")

        # ===== Phase 3: Create model =====
        logger.log("Creating model...")
        t_model_start = time()
        model = TorchModel(
            n_run,
            num_neurons,
            DT,
            MODEL_PARAMS,
            weights,
            exc_indices=exc_indices,
            device=device_name
        )
        conductance, delay_buffer, spikes, v, refrac = model.state_init()
        timings['model_creation'] = time() - t_model_start
        timings['model_setup_total'] = timings['weight_loading'] + timings['model_creation']
        logger.log(f"  Model creation:   {timings['model_creation']:.3f}s")
        logger.log(f"  Total setup:      {timings['model_setup_total']:.3f}s")

        if device_name == 'cuda':
            free, total = torch.cuda.mem_get_info(device_name)
            vram_gb = (total - free) / 1024 ** 3
            logger.log(f"  VRAM after setup: {vram_gb:.2f} GB")

        # ===== Phase 4: Setup inputs =====
        rates = torch.zeros(n_run, num_neurons, device=device_name)
        rates[:, exc_indices] = stim_rate

        # ===== Phase 5: Run simulation =====
        logger.log(f"Running simulation ({num_steps} steps, {n_run} trial(s) batched)...")

        spike_batch_idx = []
        spike_neuron_idx = []
        spike_timesteps = []

        t_simulation_start = time()
        with torch.no_grad():
            for t_step in range(num_steps):
                conductance, delay_buffer, spikes, v, refrac = model(
                    rates, conductance, delay_buffer, spikes, v, refrac
                )
                if record_spikes:
                    spike_mask = spikes > 0
                    if spike_mask.any():
                        b_idx, n_idx = spike_mask.nonzero(as_tuple=True)
                        spike_batch_idx.append(b_idx.cpu())
                        spike_neuron_idx.append(n_idx.cpu())
                        spike_timesteps.append(
                            torch.full((len(b_idx),), t_step, dtype=torch.long)
                        )

                if num_steps >= 10000 and (t_step + 1) % (num_steps // 10) == 0:
                    elapsed = time() - t_simulation_start
                    pct = (t_step + 1) / num_steps * 100
                    logger.log(
                        f"  Progress: {pct:.0f}% ({t_step+1}/{num_steps})"
                        f" - {elapsed:.1f}s elapsed"
                    )

        if device_name == 'cuda':
            torch.cuda.synchronize()

        timings['simulation_total'] = time() - t_simulation_start
        timings['simulation_avg_per_trial'] = timings['simulation_total'] / n_run
        timings['device_build'] = 0.0
        logger.log(f"  Simulation time:  {timings['simulation_total']:.3f}s")
        logger.log(f"  Avg per trial:    {timings['simulation_avg_per_trial']:.3f}s")

        if device_name == 'cuda':
            free, total = torch.cuda.mem_get_info(device_name)
            vram_gb = (total - free) / 1024 ** 3
            logger.log(f"  VRAM used:        {vram_gb:.2f} GB")

        # ===== Phase 6: Collect and save results =====
        logger.log("Collecting results...")
        t_collect_start = time()
        path_save = ''

        if not record_spikes:
            df = pd.DataFrame(
                {
                    't': [], 'time_ms': [], 'trial': [],
                    'neuron_index': [], 'flywire_id': [], 'exp_name': [],
                }
            )
            timings['result_collection'] = 0.0
            timings['result_save'] = 0.0
            logger.log("  Spike probing/output disabled; no parquet written")
        elif spike_batch_idx:
            all_batch = torch.cat(spike_batch_idx).numpy()
            all_neurons = torch.cat(spike_neuron_idx).numpy()
            all_times_steps = torch.cat(spike_timesteps).numpy()
            all_times_ms = all_times_steps * DT

            df = pd.DataFrame({
                't': all_times_ms,
                'trial': all_batch,
                'neuron_index': all_neurons,
                'flywire_id': [i2flyid[int(n)] for n in all_neurons],
                'exp_name': exp_name,
            })
            df.insert(1, 'time_ms', df['t'])
        else:
            df = pd.DataFrame(
                {
                    't': [], 'time_ms': [], 'trial': [],
                    'neuron_index': [], 'flywire_id': [], 'exp_name': [],
                }
            )

        if record_spikes:
            timings['result_collection'] = time() - t_collect_start

            t_save_start = time()
            path_save = get_spike_output_path(exp_name, run_label, round_idx)
            path_save.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path_save, compression='brotli')
            timings['result_save'] = time() - t_save_start

        logger.log(f"  Collection:       {timings['result_collection']:.3f}s")
        logger.log(f"  Save to file:     {timings['result_save']:.3f}s")
        if path_save:
            logger.log(f"  Output file:      {path_save}")

        # ===== Calculate totals and metrics =====
        timings['total_elapsed'] = (
            timings['id_mapping']
            + timings['model_setup_total']
            + timings['simulation_total']
            + timings['result_collection']
            + timings['result_save']
        )

        total_simulated_time = t_run_sec * n_run
        timings['realtime_ratio'] = (
            total_simulated_time / timings['simulation_total']
            if timings['simulation_total'] > 0 else float('inf')
        )
        timings['realtime_ratio_total'] = (
            total_simulated_time / timings['total_elapsed']
            if timings['total_elapsed'] > 0 else float('inf')
        )

        n_active = df['flywire_id'].nunique() if len(df) > 0 else 0
        n_spikes = len(df)

        results = {
            't_run_sec': t_run_sec,
            'n_run': n_run,
            'n_active_neurons': n_active,
            'n_spikes': n_spikes,
            'status': 'success',
            'timings': timings,
            'backend_key': 'pytorch',
            'experiment_name': experiment['name'],
            'experiment_key': experiment['key'],
            'run_label': run_label or '',
            'round': round_idx or '',
            'spike_path': str(path_save),
        }

        # ===== Summary =====
        logger.log_raw("")
        logger.log_raw("-" * 60)
        logger.log("TIMING SUMMARY")
        logger.log_raw("-" * 60)
        logger.log(f"  Model setup:        {timings['model_setup_total']:>10.3f}s")
        logger.log(f"  Simulation:         {timings['simulation_total']:>10.3f}s")
        logger.log(f"  Result processing:  {timings['result_collection'] + timings['result_save']:>10.3f}s")
        logger.log(f"  -----------------------------------------")
        logger.log(f"  TOTAL ELAPSED:      {timings['total_elapsed']:>10.3f}s")
        logger.log_raw("")
        logger.log(f"  Simulated time:     {total_simulated_time:>10.1f}s ({n_run} x {t_run_sec}s)")
        logger.log(f"  Realtime ratio (sim only): {timings['realtime_ratio']:>6.3f}x")
        logger.log(f"  Realtime ratio (total):    {timings['realtime_ratio_total']:>6.3f}x")
        logger.log_raw("")
        logger.log(f"  Active neurons:     {n_active:>10d}")
        logger.log(f"  Total spikes:       {n_spikes:>10d}")
        logger.log_raw("-" * 60)

    except Exception as e:
        logger.log(f"ERROR: {str(e)}")
        logger.log_raw(traceback.format_exc())
        results = {
            't_run_sec': t_run_sec,
            'n_run': n_run,
            'n_active_neurons': 0,
            'n_spikes': 0,
            'status': f'error: {str(e)}',
            'timings': timings,
            'backend_key': 'pytorch',
            'experiment_name': experiment['name'],
            'experiment_key': experiment['key'],
            'run_label': run_label or '',
            'round': round_idx or '',
        }

    return results


def run_all_benchmarks(t_run_values=None, n_run_values=None,
                       experiment=None, logger=None,
                       run_label=None, round_idx=None):
    """
    Run all PyTorch benchmark combinations.

    Args:
        t_run_values: List of t_run durations in seconds, or None for all
        n_run_values: List of n_run values to test, or None for all
        experiment: experiment config dict from get_experiment()
        logger: BenchmarkLogger instance
    """
    if t_run_values is None:
        t_run_values = T_RUN_VALUES_SEC
    if n_run_values is None:
        n_run_values = N_RUN_VALUES
    if experiment is None:
        experiment = get_experiment()

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    backend_name = f'PyTorch ({device_name.upper()})'

    benchmarks = []
    for n_run in n_run_values:
        for t_run_sec in t_run_values:
            benchmarks.append((t_run_sec, n_run))

    total_runs = len(benchmarks)

    logger.log_raw("")
    logger.log_raw("=" * 80)
    logger.log(f"BENCHMARK SUITE: {backend_name}")
    logger.log_raw("=" * 80)
    logger.log(f"Device: {device_name.upper()}")
    if device_name == 'cuda':
        logger.log(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.log(f"t_run values: {t_run_values} seconds")
    logger.log(f"n_run values: {n_run_values}")
    if run_label:
        logger.log(f"Run label: {run_label}")
    if round_idx is not None:
        logger.log(f"Round: {round_idx}")
    logger.log(f"Total benchmarks: {total_runs}")
    logger.log_raw("=" * 80)

    all_results = []

    for run_idx, (t_run_sec, n_run) in enumerate(benchmarks, 1):
        result = run_single_benchmark(
            t_run_sec=t_run_sec,
            n_run=n_run,
            experiment=experiment,
            logger=logger,
            run_idx=run_idx,
            total_runs=total_runs,
            run_label=run_label,
            round_idx=round_idx,
        )
        all_results.append(result)
        save_result_csv(backend_name, result)

    print_summary_table(all_results, backend_name, logger)

    return all_results
