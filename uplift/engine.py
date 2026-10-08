"""UpliftEngine: the native whole-brain kernel with ports for an uplift.

The fly is not reimplemented here. ``NativeBrainEngine`` still steps every
native neuron with the Shiu et al. LIF equations, the 1.8 ms axonal delay, the
refractory gate and the bit-exact fused kernel. This subclass only adds the
three ways anything outside the fly is allowed to touch it:

  SENSORY PORT   Poisson spikes into real sensory neurons (the paper's own
                 stimulation convention: a driven neuron loses its
                 refractoriness while it is driven, and only then).
  SYNAPTIC PORT  extra synaptic input into any neuron, delivered through the
                 SAME delay ring as a native synapse -- it arrives 1.8 ms after
                 the spike that caused it, respects the target's refractory
                 gate, and wakes the target's tile. Grafts and the fly's own
                 plasticity speak to the fly only through this port.
  READOUT        a per-neuron spike counter, from which every population rate,
                 every "thought frame" and every behaviour is decoded.

Invariant, checked by ``uplift/tests/test_identity.py``: with no emitters
attached and nothing queued, ``UpliftEngine.step`` is ``NativeBrainEngine.step``
plus a counter increment, so the fly's state stays bit-identical to the plain
engine under the same drive.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "flyloop") not in sys.path:
    sys.path.insert(0, str(ROOT / "flyloop"))

from native_engine import NativeBrainEngine  # noqa: E402

DT_MS = 0.1


class UpliftEngine(NativeBrainEngine):
    """Native kernel + sensory port + synaptic port + readout counters.

    Parameters beyond NativeBrainEngine's:
      sensory_ids   every neuron the sensory port may ever drive. Fixed at
                    construction (it defines the stimulated set); rates are
                    changed freely afterwards with ``set_rates``.
      extend        graft hook, see uplift/graft.py.
      poisson_block steps of Poisson drive drawn at once. Small, because a
                    closed loop changes rates every few milliseconds and each
                    change discards the rest of the block.
    """

    def __init__(self, data_dir=str(ROOT / "data"), sensory_ids=(), seed=0,
                 threads=None, reorder="cell_type", model="lif_euler",
                 extend=None, poisson_block=128, rng="numpy"):
        import os
        if threads is None and os.environ.get("UPLIFT_THREADS"):
            threads = int(os.environ["UPLIFT_THREADS"])
        self.rng = np.random.default_rng(seed)
        # rng="torch" reproduces NativeBrainEngine's Poisson stream exactly
        # (same generator, same 4096-step block), for the identity gate.
        self.torch_rng = rng == "torch"
        if self.torch_rng:
            poisson_block = 4096
        super().__init__(data_dir=data_dir, stim_ids=None, seed=seed,
                         threads=threads, reorder=reorder, model=model,
                         extend=extend)
        self.base_refrac = int(round(self.p["tRefrac"] / self.dt))
        self.POISSON_BLOCK = int(poisson_block)
        self._rate_np = np.zeros(0, dtype=np.float32)
        if len(sensory_ids):
            self.set_stim_neurons(list(sensory_ids))
        self.counts = np.zeros(self.N, dtype=np.int64)   # spikes since t=0
        self.steps = 0
        self._emitters = []      # objects with .emit(engine, prev_spikes)
        self._tickers = []       # (every_steps, obj with .tick(engine))
        self._queue_i, self._queue_v = [], []

    # ------------------------------------------------------------ sensory
    def set_stim_neurons(self, flywire_ids):
        n = super().set_stim_neurons(flywire_ids)
        # The parent strips refractoriness from the whole stimulated set. Here
        # that set is "everything the port MAY drive", so give it back, and
        # remove it only from neurons actually being driven (set_rates).
        self.refrac_steps[self.stim_idx] = self.base_refrac
        self._rate_np = np.zeros(self.stim_idx.size, dtype=np.float32)
        self._slot_of_stim = {int(s): k for k, s in enumerate(self.stim_idx)}
        return n

    def stim_positions(self, slots):
        """Positions in the stimulated set for engine slots (-1 if absent)."""
        return np.asarray([self._slot_of_stim.get(int(s), -1) for s in slots],
                          dtype=np.int64)

    def set_rates(self, positions, rates_hz):
        """Set Poisson rates (Hz) for positions within the stimulated set."""
        positions = np.asarray(positions, dtype=np.int64)
        self._rate_np[positions] = rates_hz
        self._refresh_refrac()
        self._poi_block = None

    def clear_rates(self):
        self._rate_np[:] = 0.0
        self._refresh_refrac()
        self._poi_block = None

    def inject(self, rates_hz):
        """Parent-compatible: one rate for the whole stimulated set."""
        self._rate_np[:] = rates_hz
        self._refresh_refrac()
        self._poi_block = None

    def _refresh_refrac(self):
        driven = self._rate_np > 0
        self.refrac_steps[self.stim_idx[driven]] = 0
        self.refrac_steps[self.stim_idx[~driven]] = self.base_refrac

    def _draw_poisson(self):
        if self.torch_rng:
            import torch
            self._rates = torch.as_tensor(self._rate_np, dtype=torch.float32)
            return NativeBrainEngine._draw_poisson(self)
        B, n = self.POISSON_BLOCK, self.stim_idx.size
        blk = np.zeros((B, n), dtype=np.float32)
        nz = np.flatnonzero(self._rate_np)
        if nz.size:
            p = self._rate_np[nz].astype(np.float64) * (self.dt / 1000.0)
            hit = self.rng.random((B, nz.size)) < p
            blk[:, nz] = hit * np.float32(self._poi_scale)
        self._poi_block = blk
        self._poi_pos = 0

    # ------------------------------------------------------------ synaptic
    def deliver(self, slots, values):
        """Queue synaptic input (same units as a native delay-slot entry, i.e.
        synapse count x wScale) for engine slots. It is written into the
        delay ring on the next step and arrives one axonal delay later."""
        slots = np.asarray(slots, dtype=np.int32)
        if slots.size:
            self._queue_i.append(slots)
            self._queue_v.append(np.asarray(values, dtype=np.float32))

    def add_emitter(self, obj):
        self._emitters.append(obj)
        return obj

    def add_ticker(self, obj, every_steps):
        self._tickers.append((int(every_steps), obj))
        return obj

    def _append_slot(self, h, idx, val):
        n0 = self._del_n[h]
        k = idx.size
        if n0 + k > self.N:
            raise OverflowError("delay slot overflow")
        self._del_idx[h][n0:n0 + k] = idx
        self._del_val[h][n0:n0 + k] = val
        self._del_n[h] = n0 + k

    # ------------------------------------------------------------ step
    def step(self, record=False):
        h = self.head
        prev = self._sp_buf[self._cur][:self._cur_n]   # untouched by this step
        nsp = super().step(record)
        if nsp:
            self.counts[self._sp_buf[self._cur][:nsp]] += 1
        if self._emitters or self._queue_i:
            pi, pv = self._queue_i, self._queue_v
            self._queue_i, self._queue_v = [], []
            for em in self._emitters:
                r = em.emit(self, prev)
                if r is not None:
                    pi.append(r[0])
                    pv.append(r[1])
            if pi:
                idx = np.concatenate(pi).astype(np.int32, copy=False)
                val = np.concatenate(pv).astype(np.float32, copy=False)
                keep = val != 0
                if keep.any():
                    self._append_slot(h, idx[keep], val[keep])
        self.steps += 1
        for every, obj in self._tickers:
            if self.steps % every == 0:
                obj.tick(self)
        return nsp

    def run(self, ms):
        n = int(round(ms / self.dt))
        tot = 0
        for _ in range(n):
            tot += self.step()
        return tot

    def full_reset(self):
        """Back to t=0, keeping the connectome, grafts and learned weights."""
        self.reset()
        self.counts[:] = 0
        self.steps = 0
        self._queue_i, self._queue_v = [], []
        self._poi_block = None
        for _, obj in self._tickers:
            if hasattr(obj, "on_reset"):
                obj.on_reset(self)
        for obj in self._emitters:
            if hasattr(obj, "on_reset"):
                obj.on_reset(self)


class Readout:
    """Windowed population rates from the engine's spike counter."""

    def __init__(self, engine, groups: dict):
        """groups: name -> array of engine slots."""
        self.e = engine
        self.names = list(groups)
        self.slots = [np.asarray(groups[k], dtype=np.int64) for k in self.names]
        self.last = [engine.counts[s].copy() for s in self.slots]
        self.t_last = engine.t_ms

    def read(self):
        """Mean rate (Hz per neuron) of each group since the previous read."""
        dt_s = max(self.e.t_ms - self.t_last, 1e-9) / 1000.0
        out = {}
        for k, s, l in zip(self.names, self.slots, self.last):
            c = self.e.counts[s]
            out[k] = float((c - l).sum()) / max(s.size, 1) / dt_s
            l[:] = c
        self.t_last = self.e.t_ms
        return out

    def reset(self):
        self.last = [self.e.counts[s].copy() for s in self.slots]
        self.t_last = self.e.t_ms
