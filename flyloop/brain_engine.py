"""Steppable, optimized whole-brain LIF engine.

Same model as Shiu et al. / run_pytorch.py, but:
  (1) ring-buffer delay line instead of torch.roll   -> ~19x less memory traffic
  (2) W @ s  instead of  s @ W.T                     -> uses fast CSR spmv, no transpose
  (3) sparse Poisson (sample only stimulated ids)    -> avoids 1.4e9 wasted draws
  (4) on-device spike accumulation                   -> no per-step CPU sync
  (5) step()/inject()/read_rates() API               -> usable in a closed sensorimotor loop

Numerically equivalent to the original (same update equations, same RNG semantics
for the stimulated subset).
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MODEL_PARAMS = {
    "tauSyn": 5.0, "tDelay": 1.8, "v0": -52.0, "vReset": -52.0, "vRest": -52.0,
    "vThreshold": -45.0, "tauMem": 20.0, "tRefrac": 2.2,
    "scalePoisson": 250, "wScale": 0.275,
}
DT = 0.1  # ms


class BrainEngine:
    """Whole-brain LIF you can drive one timestep at a time."""

    def __init__(self, data_dir="data", device=None, params=None, dt=DT,
                 stim_ids=None, seed=0):
        self.p = dict(params or MODEL_PARAMS)
        self.dt = dt
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        d = Path(data_dir)

        comp = pd.read_csv(d / "2025_Completeness_783.csv", index_col=0)
        self.flyid2i = {int(j): i for i, j in enumerate(comp.index)}
        self.i2flyid = np.asarray(comp.index, dtype=np.int64)
        self.N = len(self.i2flyid)

        # (6) EVENT-DRIVEN fan-out table: W.T in CSR == W in CSC.
        # Lets us touch only the synapses of neurons that actually spiked
        # (~1.6/step here) instead of all 15.1M nnz every step.
        #
        # NOTE: the dense CSR matrix (weight_csr.pkl, 289 MB) is ONLY needed for
        # the event_mode=False fallback, so it is loaded lazily -- deployments
        # that always run event-driven never need to ship it.
        self._W = None
        self._wpath = d / "weight_csr.pkl"
        self.event_mode = True
        cache = d / "fanout_csc.pt"
        if cache.exists():
            fo = torch.load(cache)
        else:
            Wc = self.W.to_sparse_coo().coalesce()
            post, pre = Wc.indices()[0], Wc.indices()[1]
            o = torch.argsort(pre)                      # group by presynaptic
            pre_s, post_s, val_s = pre[o], post[o], Wc.values()[o]
            counts = torch.bincount(pre_s, minlength=self.N)
            crow = torch.zeros(self.N + 1, dtype=torch.long)
            crow[1:] = torch.cumsum(counts, 0)
            fo = {"crow": crow, "post": post_s.contiguous(),
                  "val": val_s.float().contiguous()}
            torch.save(fo, cache)
        self.fo_crow = fo["crow"].to(self.device)
        self.fo_post = fo["post"].to(self.device)
        self.fo_val = fo["val"].to(self.device)

        # ---- state ----
        self.L = int(self.p["tDelay"] / dt) + 1          # delay-line length
        self.g = torch.zeros(self.N, device=self.device)  # conductance
        self.buf = torch.zeros(self.L, self.N, device=self.device)
        self.head = 0                                     # ring pointer
        self.spikes = torch.zeros(self.N, device=self.device)
        self.v = torch.full((self.N,), self.p["v0"], device=self.device)

        base_refrac = int(round(self.p["tRefrac"] / dt))
        self.refrac_steps = torch.full((self.N,), base_refrac,
                                       dtype=torch.long, device=self.device)
        self.refrac = self.refrac_steps.clone().float()

        # stimulated (sensory-injection) neurons are non-refractory, per upstream
        self.stim_idx = torch.empty(0, dtype=torch.long, device=self.device)
        if stim_ids:
            self.set_stim_neurons(stim_ids)

        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)

        self.t_ms = 0.0
        self._rec_t, self._rec_n = [], []
        self._sf = self.p["tauSyn"], self.p["tauMem"]

        # (7) preallocated scratch: the naive step allocates ~25 N-sized tensors
        # (13 MB) per timestep. Reusing buffers with in-place ops measured 2.1x.
        z = lambda dt=torch.float32: torch.zeros(self.N, dtype=dt,
                                                 device=self.device)
        self._vstim, self._rec = z(), z()
        self._tmp, self._gate, self._gnew = z(), z(), z()
        self._spb = z(torch.bool)     # spike mask
        self._gb = z(torch.bool)      # refractory gate mask
        self.inplace = True

        # (9) STATE PACKING: keep the four per-neuron arrays in one (4,N)
        # tensor. The active path then does 1 gather + 1 scatter instead of
        # 4 + 3 -- which matters because that path is dominated by PyTorch
        # per-op dispatch (~45us/op), not by arithmetic. v/g/refrac/refrac_steps
        # become views, so the dense kernel is untouched.
        self.state = torch.zeros(4, self.N, device=self.device)
        self.state[0] = self.v
        self.state[1] = self.g
        self.state[2] = self.refrac
        self.state[3] = self.refrac_steps.to(torch.float32)
        self.v = self.state[0]
        self.g = self.state[1]
        self.refrac = self.state[2]
        self.refrac_steps = self.state[3]

        self.active_mode = True
        self._init_active()

    @property
    def W(self):
        """Dense CSR connectome. Only touched by the event_mode=False path."""
        if self._W is None:
            if not self._wpath.exists():
                raise FileNotFoundError(
                    f"{self._wpath} is required only for event_mode=False; "
                    "event-driven mode needs just fanout_csc.pt")
            with open(self._wpath, "rb") as f:
                self._W = pickle.load(f).to(self.device)
        return self._W

    # ---------------- interface ----------------
    def set_stim_neurons(self, flywire_ids):
        """Declare which neurons can be driven (they lose refractoriness)."""
        idx = [self.flyid2i[int(i)] for i in flywire_ids if int(i) in self.flyid2i]
        self.stim_idx = torch.tensor(idx, dtype=torch.long, device=self.device)
        self.refrac_steps[self.stim_idx] = 0
        self._rates = torch.zeros(len(idx), device=self.device)
        return len(idx)

    def inject(self, rates_hz):
        """Set Poisson drive (Hz) on the stimulated neurons. Scalar or per-neuron."""
        if np.isscalar(rates_hz):
            self._rates.fill_(float(rates_hz))
        else:
            self._rates.copy_(torch.as_tensor(
                rates_hz, dtype=torch.float32, device=self.device))

    def indices_of(self, flywire_ids):
        return torch.tensor([self.flyid2i[int(i)] for i in flywire_ids
                             if int(i) in self.flyid2i],
                            dtype=torch.long, device=self.device)

    @torch.no_grad()
    def step_inplace(self, record=False):
        """Allocation-free step. Numerically identical to step(); ~2x faster.

        Note v - (v-vReset)*sp  ==  vReset where sp else v, so the reset
        becomes a masked_fill_ instead of three N-sized temporaries.
        """
        p, N, dt, dev = self.p, self.N, self.dt, self.device
        wS = p["wScale"]
        vstim, rec, tmp = self._vstim, self._rec, self._tmp
        gate, gnew, spb, gb = self._gate, self._gnew, self._spb, self._gb

        # --- Poisson drive (stimulated subset only) ---
        vstim.zero_()
        if self.stim_idx.numel():
            ps = torch.bernoulli(self._rates * (dt / 1000.0), generator=self.gen)
            ps.mul_(p["scalePoisson"] * wS)
            vstim.index_copy_(0, self.stim_idx, ps)

        # --- recurrent drive (event-driven fan-out) ---
        rec.zero_()
        src = self.spikes.nonzero(as_tuple=True)[0]
        if src.numel():
            st, en = self.fo_crow[src], self.fo_crow[src + 1]
            cnt = en - st
            tot = int(cnt.sum())
            if tot:
                base = torch.repeat_interleave(st, cnt)
                ramp = torch.arange(tot, device=dev) - \
                    torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
                sel = base + ramp
                rec.scatter_add_(0, self.fo_post[sel], self.fo_val[sel])
            rec.mul_(wS)

        # --- refractory bookkeeping ---
        torch.gt(self.spikes, 0, out=spb)
        self.refrac.add_(1).masked_fill_(spb, 0.0)
        torch.ge(self.refrac, self.refrac_steps, out=gb)
        gate.copy_(gb)

        # --- conductance (ring-buffer delay line) ---
        delayed = self.buf[self.head]
        torch.mul(delayed, gate, out=tmp)
        gnew.copy_(self.g).mul_(1 - dt / p["tauSyn"]).add_(tmp)
        self.buf[self.head].copy_(rec)
        self.head = (self.head + 1) % self.L

        # --- LIF membrane update (uses pre-update conductance) ---
        v = self.v
        v.add_(vstim)
        tmp.copy_(v).sub_(p["vRest"]).neg_().add_(self.g)
        v.add_(tmp, alpha=dt / p["tauMem"])
        torch.gt(v, p["vThreshold"], out=spb)
        v.masked_fill_(spb, p["vReset"])

        gnew.masked_fill_(spb, 0.0)
        self.g.copy_(gnew)
        self.spikes.copy_(spb)
        self.t_ms += dt

        if record:
            nz = spb.nonzero(as_tuple=True)[0]
            if nz.numel():
                self._rec_n.append(nz.cpu())
                self._rec_t.append(torch.full((nz.numel(),), self.t_ms))
        return self.spikes

    # ---------------- (8) active-set stepping ----------------
    # A neuron is *provably inert* when g==0, v==vRest, it is not refractory,
    # it did not just spike, and it has no delayed input still in flight.
    # Substituting those into the update gives back exactly the same state, so
    # skipping it is lossless -- not an approximation.
    #
    # Measured on the sugar experiment: 91% of the network is inert, and the
    # dense neuron update is ~99.9% of runtime (728:1 vs synaptic work), so
    # this attacks the actual bottleneck. It is stimulus-dependent, hence the
    # automatic fall back to the dense kernel once the active set gets large
    # -- a fully-active network must never be slower than before.

    def _init_active(self):
        self.active = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        if self.stim_idx.numel():
            self.active[self.stim_idx] = True
        self._idx = self.active.nonzero(as_tuple=True)[0]
        self._spike_idx = torch.empty(0, dtype=torch.long, device=self.device)
        self.prune_every = 64
        # Empirical crossover. Measured on 8 regimes: the sparse path wins
        # decisively below ~8% active, but once the active set (or the spike
        # rate feeding it) grows, gather/scatter + index rebuilding costs more
        # than the dense kernel -- at 26% active with a high firing rate it was
        # 2.3x SLOWER. Switching early costs a little upside on mid-density
        # runs and buys a hard no-regression guarantee.
        self.dense_switch = 0.08
        # Switch on EDGE WORK, not spike count. This connectome is hub-dominated
        # and scale-free: out-degree spans 1 to ~9,800, so a single high-degree
        # spike can carry more fan-out than a thousand low-degree ones while
        # looking cheap under a count-only threshold. The old `spike_switch`
        # (0.4% of N spiking) is kept only as a cheap pre-filter.
        self.spike_switch = 0.004
        self.edge_switch = 0.02       # >2% of all synapses fanned out -> dense
        self._nnz = int(self.fo_crow[-1])
        self._since_prune = 0

        # Returning to the sparse path.
        #
        # This USED to be a one-way latch: once _dense_fallback was set, step()
        # never called step_active() again, so a single transient burst disabled
        # the sparse path for the remainder of the run and permanently gave up
        # the 2-10x it buys. Beamer's direction-optimising BFS -- which this
        # switch is an instance of -- switches BOTH ways, with hysteresis to
        # avoid oscillating at the boundary.
        #
        # Re-entry is checked only every `recheck_every` steps (rebuilding the
        # active set costs an O(N) scan) and requires activity to fall well
        # BELOW dense_switch, not merely back under it.
        self._dense_fallback = False
        self.sparse_switch = 0.04     # hysteresis: half of dense_switch
        self.recheck_every = 256
        self._since_recheck = 0

    @torch.no_grad()
    def _v_fixed(self, v):
        """True where the membrane update is a FIXED POINT, given g == 0.

        The inert test used to be `v == v_rest`, which is unreachable in
        practice. With g == 0 the membrane decays d <- 0.995*d where
        d = v - v_rest, but near -52 mV the fp32 ULP is ~3.8e-6, so once d
        reaches one ULP the update rounds back to itself: a neuron that was ever
        perturbed gets STUCK one ULP above rest and never returns to it exactly.
        Under the old predicate such a neuron could never be pruned, so the
        active set only ever grew and the sparse path decayed over a long run.

        Testing the actual fixed-point condition is both more general and
        equally exact: if the update maps v to itself and g is already 0, then
        g stays 0 and v stays v, so skipping the neuron reproduces the state
        exactly. `v == v_rest` is just the special case d = 0.
        """
        return v.add(-(v - self.p["vRest"]), alpha=self.dt / self.p["tauMem"]) == v

    @torch.no_grad()
    def _prune_active(self):
        """Deactivate provably-inert neurons. Exact (see note above)."""
        idx = self._idx
        if not idx.numel():
            return
        spiking = torch.zeros(idx.numel(), dtype=torch.bool, device=self.device)
        if self._spike_idx.numel():
            spiking[torch.searchsorted(idx, self._spike_idx)] = True
        inert = (self._v_fixed(self.v[idx])
                 & (self.g[idx] == 0)
                 & ~spiking
                 & (self.refrac[idx] >= self.refrac_steps[idx])
                 & (self.buf[:, idx].abs().amax(0) == 0))
        if self.stim_idx.numel():       # driven neurons never go inert
            inert[torch.searchsorted(idx, self.stim_idx)] = False
        if inert.any():
            self.active[idx[inert]] = False
            self._idx = idx[~inert]

    @torch.no_grad()
    def step_active(self, record=False):
        """Step touching only non-inert neurons.

        Everything here is O(active), never O(N): the active index list and the
        spiking-neuron list are both carried between steps, and the recurrent
        drive accumulates into a compact buffer addressed by searchsorted, so
        no dense array is scanned, zeroed or allocated.
        """
        p, N, dt, dev = self.p, self.N, self.dt, self.device
        wS = p["wScale"]

        idx = self._idx
        # Decide BEFORE doing any work -- otherwise the fan-out is computed
        # twice and the fallback ends up slower than plain dense.
        too_dense = idx.numel() > self.dense_switch * N
        if not too_dense and self._spike_idx.numel() > self.spike_switch * N:
            # cheap count filter passed; now check the quantity that actually
            # predicts cost, the total out-degree of the spiking set
            edges = int((self.fo_crow[self._spike_idx + 1]
                         - self.fo_crow[self._spike_idx]).sum())
            too_dense = edges > self.edge_switch * self._nnz
        if too_dense:
            self._dense_fallback = True
            self._since_recheck = 0
            return self.step_inplace(record=record)

        # --- recurrent drive from last step's spikes (no dense scan) ---
        src = self._spike_idx
        tgt = vals = None
        if src.numel():
            st, en = self.fo_crow[src], self.fo_crow[src + 1]
            cnt = en - st
            tot = int(cnt.sum())
            if tot:
                base = torch.repeat_interleave(st, cnt)
                ramp = torch.arange(tot, device=dev) - \
                    torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
                sel = base + ramp
                tgt = self.fo_post[sel]
                vals = self.fo_val[sel]
                fresh = tgt[~self.active[tgt]]
                if fresh.numel():                   # grow the active set
                    fresh = torch.unique(fresh)
                    self.active[fresh] = True
                    idx = torch.sort(torch.cat([idx, fresh])).values
                    self._idx = idx
                    if idx.numel() > self.dense_switch * N:
                        self._dense_fallback = True
                        self._rebuild_dense_state(tgt, vals, wS)
                        return self.step_inplace(record=record)

        na = idx.numel()
        rec = torch.zeros(na, device=dev)           # O(active), not O(N)
        if tgt is not None:
            rec.scatter_add_(0, torch.searchsorted(idx, tgt), vals)
            rec.mul_(wS)

        # --- gather: ONE indexing op for all four state arrays (9) ---
        s = self.state[:, idx]                  # (4, na): v, g, refrac, refrac_steps
        v, g, rf, rs = s[0], s[1], s[2], s[3]   # views into s
        rf.add_(1)
        if src.numel():
            rf[torch.searchsorted(idx, src)] = 0.0
        gate = (rf >= rs).to(v.dtype)

        # --- Poisson (stim neurons are always in the active set) ---
        if self.stim_idx.numel():
            ps = torch.bernoulli(self._rates * (dt / 1000.0), generator=self.gen)
            ps.mul_(p["scalePoisson"] * wS)
            v.index_add_(0, torch.searchsorted(idx, self.stim_idx), ps)

        # --- conductance (ring-buffer delay line) ---
        gnew = g * (1 - dt / p["tauSyn"]) + self.buf[self.head, idx] * gate
        self.buf[self.head, idx] = rec
        self.head = (self.head + 1) % self.L

        # --- membrane (uses pre-update conductance, as upstream) ---
        v.add_((g - (v - p["vRest"])), alpha=dt / p["tauMem"])
        spb = v > p["vThreshold"]
        v.masked_fill_(spb, p["vReset"])
        gnew.masked_fill_(spb, 0.0)
        g.copy_(gnew)

        # --- scatter back: ONE indexing op ---
        self.state[:, idx] = s
        self._spike_idx = idx[spb]
        self.t_ms += dt

        self._since_prune += 1
        if self._since_prune >= self.prune_every:
            self._since_prune = 0
            self._prune_active()

        if record and self._spike_idx.numel():
            self._rec_n.append(self._spike_idx.cpu())
            self._rec_t.append(torch.full((self._spike_idx.numel(),), self.t_ms))
        return self._spike_idx

    def _rebuild_dense_state(self, tgt, vals, wS):
        """Re-materialise self.spikes before handing control to the dense
        kernel (which reads it densely rather than via _spike_idx)."""
        self.spikes.zero_()
        if self._spike_idx.numel():
            self.spikes[self._spike_idx] = 1.0

    @torch.no_grad()
    def _try_leave_dense(self):
        """Rebuild the active set and go back to the sparse path, if activity
        has genuinely subsided. Exact: `live` is the same provably-inert
        predicate `_prune_active` uses, evaluated over all N."""
        live = ((self.g != 0)
                | ~self._v_fixed(self.v)
                | (self.buf.abs().amax(0) != 0)
                | (self.refrac < self.refrac_steps)
                | (self.spikes > 0))
        if self.stim_idx.numel():
            live[self.stim_idx] = True          # driven neurons never go inert
        if float(live.sum()) >= self.sparse_switch * self.N:
            return False
        self.active = live
        self._idx = live.nonzero(as_tuple=True)[0]
        self._spike_idx = self.spikes.nonzero(as_tuple=True)[0]
        self._dense_fallback = False
        return True

    @torch.no_grad()
    def step(self, record=False):
        """Advance one dt. Returns the spike vector (N,) as float 0/1."""
        if getattr(self, "active_mode", False) and self._dense_fallback:
            self._since_recheck += 1
            if self._since_recheck >= self.recheck_every:
                self._since_recheck = 0
                self._try_leave_dense()
        if getattr(self, "active_mode", False) and not self._dense_fallback:
            return self.step_active(record=record)
        if getattr(self, "inplace", False):
            return self.step_inplace(record=record)
        p, N = self.p, self.N
        wS = p["wScale"]

        # (3) sparse Poisson: only draw for stimulated neurons
        vstim = torch.zeros(N, device=self.device)
        if self.stim_idx.numel():
            pr = self._rates * (self.dt / 1000.0)
            ps = torch.bernoulli(pr, generator=self.gen) * p["scalePoisson"]
            vstim.index_copy_(0, self.stim_idx, ps * wS)

        # (2)/(6) recurrent drive.
        # dense: rec = W @ s   |   event-driven: only fan out from spiking neurons.
        if self.event_mode:
            src = self.spikes.nonzero(as_tuple=True)[0]
            rec = torch.zeros(N, device=self.device)
            if src.numel():
                st, en = self.fo_crow[src], self.fo_crow[src + 1]
                cnt = en - st
                tot = int(cnt.sum())
                if tot:
                    base = torch.repeat_interleave(st, cnt)
                    ramp = torch.arange(tot, device=self.device) - \
                        torch.repeat_interleave(
                            torch.cumsum(cnt, 0) - cnt, cnt)
                    sel = base + ramp
                    rec.scatter_add_(0, self.fo_post[sel], self.fo_val[sel])
            rec *= wS
        else:
            rec = torch.mv(self.W, self.spikes) * wS

        # refractory gate
        self.refrac = torch.where(self.spikes > 0,
                                  torch.zeros_like(self.refrac), self.refrac + 1)
        gate = (self.refrac >= self.refrac_steps).float()

        # (1) ring-buffer delay line (read + overwrite the same slot)
        delayed = self.buf[self.head]
        g_new = self.g * (1 - self.dt / p["tauSyn"]) + delayed * gate
        self.buf[self.head] = rec
        self.head = (self.head + 1) % self.L

        # LIF update (uses pre-update conductance, matching upstream ordering)
        v = self.v + vstim
        v = v + (self.dt / p["tauMem"]) * (self.g - (v - p["vRest"]))
        sp = (v > p["vThreshold"]).float()
        v = v - (v - p["vReset"]) * sp

        self.g = g_new - g_new * sp
        self.spikes, self.v = sp, v
        self.t_ms += self.dt

        if record:
            nz = sp.nonzero(as_tuple=True)[0]
            if nz.numel():
                self._rec_n.append(nz.to("cpu", non_blocking=True))
                self._rec_t.append(torch.full((nz.numel(),), self.t_ms))
        return sp

    def read_rates(self, idx, window_ms=20.0):
        """Instantaneous firing proxy for a neuron subset: spikes now (0/1)."""
        return self.spikes.index_select(0, idx)

    def spikes_dataframe(self):
        if not self._rec_n:
            return pd.DataFrame(columns=["time_ms", "neuron_index", "flywire_id"])
        n = torch.cat(self._rec_n).numpy()
        t = torch.cat(self._rec_t).numpy()
        return pd.DataFrame({"time_ms": t, "neuron_index": n,
                             "flywire_id": self.i2flyid[n]})


class RateTracker:
    """Exponential firing-rate estimate for a neuron subset (for DN readout)."""

    def __init__(self, engine, idx, tau_ms=50.0):
        self.e, self.idx, self.tau = engine, idx, tau_ms
        self.r = torch.zeros(len(idx), device=engine.device)

    def update(self):
        s = self.e.spikes.index_select(0, self.idx)
        a = self.e.dt / self.tau
        self.r += a * (s * (1000.0 / self.e.dt) - self.r)
        return self.r
