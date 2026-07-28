"""Active-set (lazy) whole-brain LIF.

Profiling showed that after making synapse propagation event-driven, the cost is
dominated by ~12 DENSE elementwise passes over all 138,639 neurons every step --
even though only a few hundred neurons ever leave rest.

A LIF neuron with no input decays deterministically toward rest, and an
exponential synapse decays toward zero. So a neuron only needs to be *touched*
while it is away from rest. We therefore keep an ACTIVE SET and update state
only on that subset, admitting neurons when synaptic events arrive and retiring
them once they have decayed back within eps of rest.

Combined with a sparse delay queue (NEST-style: events are pushed into the slot
where they are due, instead of rolling a dense (L, N) buffer), the whole step
becomes O(active + events) rather than O(N).
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np, pandas as pd, torch

from brain_engine import MODEL_PARAMS, DT


class SparseBrain:
    def __init__(self, data_dir="data", device=None, params=None, dt=DT,
                 stim_ids=None, seed=0, eps_v=1e-3, eps_g=1e-4):
        self.p = dict(params or MODEL_PARAMS)
        self.dt, self.eps_v, self.eps_g = dt, eps_v, eps_g
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        d = Path(data_dir)

        comp = pd.read_csv(d / "2025_Completeness_783.csv", index_col=0)
        self.flyid2i = {int(j): i for i, j in enumerate(comp.index)}
        self.i2flyid = np.asarray(comp.index, dtype=np.int64)
        self.N = len(self.i2flyid)

        fo = torch.load(d / "fanout_csc.pt")   # built by BrainEngine
        self.crow = fo["crow"].to(self.device)
        self.post = fo["post"].to(self.device)
        self.val = fo["val"].to(self.device)

        # dense state kept for indexing convenience, but only *touched* on the
        # active set -- this is what makes it fast.
        self.v = torch.full((self.N,), self.p["v0"], device=self.device)
        self.g = torch.zeros(self.N, device=self.device)
        self.refrac = torch.zeros(self.N, device=self.device)
        base = int(round(self.p["tRefrac"] / dt))
        self.refrac_steps = torch.full((self.N,), base, dtype=torch.long,
                                       device=self.device)

        self.L = int(self.p["tDelay"] / dt) + 1
        self.q_idx = [torch.empty(0, dtype=torch.long, device=self.device)
                      for _ in range(self.L)]
        self.q_val = [torch.empty(0, device=self.device) for _ in range(self.L)]
        self.head = 0

        self.active = torch.empty(0, dtype=torch.long, device=self.device)
        self.spike_idx = torch.empty(0, dtype=torch.long, device=self.device)

        self.stim_idx = torch.empty(0, dtype=torch.long, device=self.device)
        if stim_ids:
            self.set_stim_neurons(stim_ids)
        self.gen = torch.Generator(device=self.device); self.gen.manual_seed(seed)
        self.t_ms = 0.0
        self._rt, self._rn = [], []

    def set_stim_neurons(self, ids):
        idx = [self.flyid2i[int(i)] for i in ids if int(i) in self.flyid2i]
        self.stim_idx = torch.tensor(idx, dtype=torch.long, device=self.device)
        self.refrac_steps[self.stim_idx] = 0
        self._rates = torch.zeros(len(idx), device=self.device)
        return len(idx)

    def inject(self, hz):
        if np.isscalar(hz): self._rates.fill_(float(hz))
        else: self._rates.copy_(torch.as_tensor(hz, dtype=torch.float32,
                                                device=self.device))

    def indices_of(self, ids):
        return torch.tensor([self.flyid2i[int(i)] for i in ids
                             if int(i) in self.flyid2i],
                            dtype=torch.long, device=self.device)

    @torch.no_grad()
    def step(self, record=False):
        p, wS, dev = self.p, self.p["wScale"], self.device

        # ---- 1. fan out last step's spikes into the delay queue -------------
        if self.spike_idx.numel():
            st, en = self.crow[self.spike_idx], self.crow[self.spike_idx + 1]
            cnt = en - st
            tot = int(cnt.sum())
            if tot:
                base = torch.repeat_interleave(st, cnt)
                ramp = torch.arange(tot, device=dev) - \
                    torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
                sel = base + ramp
                slot = (self.head + self.L - 1) % self.L
                self.q_idx[slot] = self.post[sel]
                self.q_val[slot] = self.val[sel] * wS

        # ---- 2. pop events due now -----------------------------------------
        ev_i, ev_v = self.q_idx[self.head], self.q_val[self.head]
        self.q_idx[self.head] = torch.empty(0, dtype=torch.long, device=dev)
        self.q_val[self.head] = torch.empty(0, device=dev)
        self.head = (self.head + 1) % self.L

        # ---- 3. Poisson drive on stimulated neurons ------------------------
        stim_i = stim_v = None
        if self.stim_idx.numel():
            pr = self._rates * (self.dt / 1000.0)
            ps = torch.bernoulli(pr, generator=self.gen)
            nz = ps.nonzero(as_tuple=True)[0]
            if nz.numel():
                stim_i = self.stim_idx[nz]
                stim_v = torch.full((nz.numel(),),
                                    p["scalePoisson"] * wS, device=dev)

        # ---- 4. admit touched neurons into the active set -------------------
        parts = [self.active]
        if ev_i.numel(): parts.append(ev_i)
        if stim_i is not None: parts.append(stim_i)
        a = torch.unique(torch.cat(parts)) if len(parts) > 1 else self.active
        if a.numel() == 0:
            self.t_ms += self.dt
            self.spike_idx = torch.empty(0, dtype=torch.long, device=dev)
            return self.spike_idx

        # ---- 5. state update, ONLY on the active set ------------------------
        va, ga, ra = self.v[a], self.g[a], self.refrac[a]
        rs = self.refrac_steps[a]

        # refractory bookkeeping (spiked last step -> reset counter)
        prev_sp = torch.zeros(a.numel(), dtype=torch.bool, device=dev)
        if self.spike_idx.numel():
            prev_sp = torch.isin(a, self.spike_idx)
        ra = torch.where(prev_sp, torch.zeros_like(ra), ra + 1)
        gate = (ra >= rs).float()

        # scatter this step's delayed input onto the active subset
        inp = torch.zeros(a.numel(), device=dev)
        if ev_i.numel():
            pos = torch.searchsorted(a, ev_i)
            inp.scatter_add_(0, pos, ev_v)
        vst = torch.zeros(a.numel(), device=dev)
        if stim_i is not None:
            vst.scatter_add_(0, torch.searchsorted(a, stim_i), stim_v)

        g_new = ga * (1 - self.dt / p["tauSyn"]) + inp * gate
        vv = va + vst
        vv = vv + (self.dt / p["tauMem"]) * (ga - (vv - p["vRest"]))
        sp = vv > p["vThreshold"]
        vv = torch.where(sp, torch.full_like(vv, p["vReset"]), vv)
        g_new = torch.where(sp, torch.zeros_like(g_new), g_new)

        self.v[a], self.g[a], self.refrac[a] = vv, g_new, ra
        self.spike_idx = a[sp]

        # ---- 6. retire neurons that have decayed back to rest ---------------
        keep = (g_new.abs() > self.eps_g) | \
               ((vv - p["vRest"]).abs() > self.eps_v) | (ra < rs)
        self.active = a[keep]

        self.t_ms += self.dt
        if record and self.spike_idx.numel():
            self._rn.append(self.spike_idx.cpu())
            self._rt.append(torch.full((self.spike_idx.numel(),), self.t_ms))
        return self.spike_idx

    @property
    def n_active(self): return self.active.numel()

    def spikes_dataframe(self):
        if not self._rn:
            return pd.DataFrame(columns=["time_ms", "neuron_index", "flywire_id"])
        n = torch.cat(self._rn).numpy(); t = torch.cat(self._rt).numpy()
        return pd.DataFrame({"time_ms": t, "neuron_index": n,
                             "flywire_id": self.i2flyid[n]})
