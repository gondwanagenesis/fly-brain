"""The plasticity the real fly has: dopamine-gated KC->MBON learning.

The Shiu et al. model is a fixed connectome. The fly is not: its main learning
site is the synapse from Kenyon cells (KCs) onto mushroom-body output neurons
(MBONs), and dopaminergic neurons (DANs) that innervate the same compartment
gate it. Pairing KC activity with dopamine depresses that compartment's
KC->MBON synapses (Hige et al. 2015; Cohn et al. 2015; Aso & Rubin 2016).
The order matters: KC then DA depresses, DA then KC potentiates (Handler et
al. 2019). Restoring this is the first, and the most defensible, step in
making the model fly more capable: it is the fly's OWN learning rule, at the
fly's OWN learning site, taught by the fly's OWN dopamine neurons.

THREE DESIGN CHOICES, EACH KEEPING THE FLY INTACT
------------------------------------------------
1. Compartments come from the connectome, not a hand table. DANs synapse
   directly onto the MBONs of their own compartment, so the row-normalised
   DAN->MBON synapse matrix A[m, d] says how much of the dopamine "seen" at
   MBON m comes from DAN d. It recovers the textbook pairings without being
   told them: MBON11 (gamma1pedc) <- PPL101 0.88, MBON14 (alpha3) <- PPL106 0.99,
   MBON07 (alpha1) <- PAM11 0.96, MBON12 (gamma2alpha'1) <- PPL103 0.93.

2. The learned change rides on top of the native synapse as a DELTA. Native
   KC->MBON weights stay in the kernel's int16 table untouched; this module
   adds (f - 1) * w0 through the synaptic port, with the same 1.8 ms delay. At
   f == 1 (nothing learned) it emits nothing at all, so a fly that has not
   learned is bit-identical to the unmodified model. Learning is the only
   thing that can move it.

3. DANs are driven by whatever drives them in the brain. If sugar reaches PAM
   neurons through the fly's own circuits, the fly teaches itself. External
   teaching, when used, is delivered the way an experimenter would: by
   activating DANs (Aso & Rubin 2016 used optogenetic DAN activation), through
   the synaptic port so the DANs keep their own dynamics.

THE RULE  (per KC->MBON edge e = (k, m), evaluated every tick of T ms)
    eKC_k   <- eKC_k * exp(-T/tau_kc) + n_k          KC eligibility (spikes)
    da_m     = sum_d A[m, d] * n_d                    dopamine at m's compartment
    eDA_m   <- eDA_m * exp(-T/tau_da) + da_m          dopamine trace
    df_e     = - eta_dep * eKC_k * da_m               forward pairing: depress
               + eta_pot * n_k  * eDA_m(before)       backward pairing: potentiate
               + (1 - f_e) * T / tau_recover          slow return to baseline
    f_e      in [0, f_max]
"""
from __future__ import annotations

import numpy as np


class MBPlasticity:
    """Dopamine-gated KC->MBON plasticity as a synaptic-port emitter."""

    def __init__(self, engine, atlas, *, kc_slots=None, tick_ms=5.0,
                 tau_kc_ms=1000.0, tau_da_ms=1000.0, eta_dep=0.02,
                 eta_pot=0.005, tau_recover_ms=None, f_max=1.5):
        e = engine
        self.tick_ms = float(tick_ms)
        self.every = max(1, int(round(tick_ms / e.dt)))
        self.tau_kc, self.tau_da = float(tau_kc_ms), float(tau_da_ms)
        self.eta_dep, self.eta_pot = float(eta_dep), float(eta_pot)
        self.tau_recover = tau_recover_ms
        self.f_max = float(f_max)

        N = e.N
        kc = (atlas["mb.KC"].idx(e) if kc_slots is None
              else np.asarray(kc_slots, dtype=np.int64))
        mbon = atlas["mb.MBON"].idx(e)
        dan = atlas["mb.DAN"].idx(e)
        self.kc_slots, self.mbon_slots, self.dan_slots = kc, mbon, dan
        self.is_kc = np.zeros(N, dtype=bool)
        self.is_kc[kc] = True
        kc_local = np.full(N, -1, dtype=np.int64)
        kc_local[kc] = np.arange(kc.size)
        m_local = np.full(N, -1, dtype=np.int64)
        m_local[mbon] = np.arange(mbon.size)

        # ---- plastic edges: every KC->MBON entry of the engine's own CSC ----
        lens = e.crow[kc + 1] - e.crow[kc]
        rows = np.repeat(np.arange(kc.size), lens)
        pos = (np.repeat(e.crow[kc], lens)
               + np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
        post = e.post[pos]
        keep = m_local[post] >= 0
        self.e_kc = rows[keep]                     # KC local index per edge
        self.e_m = m_local[post[keep]]             # MBON local index per edge
        self.w0 = e.val[pos[keep]].astype(np.float32)   # native synapse count
        self.n_edges = int(self.e_kc.size)
        order = np.argsort(self.e_kc, kind="stable")
        self.e_kc, self.e_m, self.w0 = (self.e_kc[order], self.e_m[order],
                                        self.w0[order])
        self.ptr = np.zeros(kc.size + 1, dtype=np.int64)
        np.cumsum(np.bincount(self.e_kc, minlength=kc.size), out=self.ptr[1:])
        self.kc_local = kc_local

        # ---- compartment map from DAN->MBON synapses ----
        A = np.zeros((mbon.size, dan.size), dtype=np.float64)
        dan_local = np.full(N, -1, dtype=np.int64)
        dan_local[dan] = np.arange(dan.size)
        for j, d in enumerate(dan):
            a, b = e.crow[d], e.crow[d + 1]
            tgt = m_local[e.post[a:b]]
            ok = tgt >= 0
            np.add.at(A[:, j], tgt[ok], np.abs(e.val[a:b][ok]).astype(np.float64))
        rs = A.sum(1, keepdims=True)
        self.A = np.divide(A, rs, out=np.zeros_like(A), where=rs > 0)
        # valence of each MBON from the dopamine that teaches it: an MBON whose
        # compartment is taught by reward (PAM) must signal AVOID, since reward
        # depresses it and learned approach is the result; PPL1-taught MBONs
        # signal APPROACH (Aso et al. 2014b).
        pam = np.isin(dan, atlas["mb.PAM"].idx(e))
        self.reward_share = self.A[:, pam].sum(1)
        self.punish_share = self.A[:, ~pam].sum(1)
        self.valence = np.where(self.A.sum(1) > 0,
                                self.punish_share - self.reward_share, 0.0)

        self.w_scale = float(e.w_scale)
        self.enabled = True
        self.on_reset(e, hard=True)

    # ------------------------------------------------------------ state
    def on_reset(self, engine, hard=False):
        """Engine time reset. Learned weights persist unless ``hard``."""
        if hard:
            self.f = np.ones(self.n_edges, dtype=np.float32)
            self.dw = np.zeros(self.n_edges, dtype=np.float32)
            self.any_learned = False
        self.eKC = np.zeros(self.kc_slots.size, dtype=np.float64)
        self.eDA = np.zeros(self.mbon_slots.size, dtype=np.float64)
        self.last_kc = engine.counts[self.kc_slots].copy()
        self.last_dan = engine.counts[self.dan_slots].copy()

    def forget(self):
        self.on_reset_weights()

    def on_reset_weights(self):
        self.f[:] = 1.0
        self.dw[:] = 0.0
        self.any_learned = False

    # ------------------------------------------------------------ port
    def emit(self, engine, prev):
        """Delta synaptic input from KCs that spiked on the previous step."""
        if not self.any_learned or prev.size == 0:
            return None
        ks = prev[self.is_kc[prev]]
        if ks.size == 0:
            return None
        loc = self.kc_local[ks]
        a, b = self.ptr[loc], self.ptr[loc + 1]
        lens = b - a
        if lens.sum() == 0:
            return None
        idx = (np.repeat(a, lens) + np.arange(lens.sum())
               - np.repeat(np.cumsum(lens) - lens, lens))
        d = self.dw[idx]
        tot = np.bincount(self.e_m[idx], weights=d, minlength=self.mbon_slots.size)
        nz = np.flatnonzero(tot)
        if nz.size == 0:
            return None
        return self.mbon_slots[nz], tot[nz].astype(np.float32)

    # ------------------------------------------------------------ learning
    def tick(self, engine):
        ck = engine.counts[self.kc_slots]
        cd = engine.counts[self.dan_slots]
        n_k = (ck - self.last_kc).astype(np.float64)
        n_d = (cd - self.last_dan).astype(np.float64)
        self.last_kc, self.last_dan = ck.copy(), cd.copy()
        T = self.tick_ms
        da = self.A @ n_d                                   # per MBON
        eDA_before = self.eDA
        self.eKC = self.eKC * np.exp(-T / self.tau_kc) + n_k
        self.eDA = self.eDA * np.exp(-T / self.tau_da) + da
        if not self.enabled:
            return
        touched = False
        df = None
        if da.any() and self.eta_dep:
            df = -self.eta_dep * self.eKC[self.e_kc] * da[self.e_m]
            touched = True
        if n_k.any() and self.eta_pot and eDA_before.any():
            p = self.eta_pot * n_k[self.e_kc] * eDA_before[self.e_m]
            df = p if df is None else df + p
            touched = True
        if self.tau_recover and self.any_learned:
            r = (1.0 - self.f) * (T / self.tau_recover)
            df = r if df is None else df + r
            touched = True
        if touched:
            self.f = np.clip(self.f + df, 0.0, self.f_max).astype(np.float32)
            self.dw = ((self.f - 1.0) * self.w0 * self.w_scale).astype(np.float32)
            self.any_learned = bool(np.any(self.f != 1.0))

    # ------------------------------------------------------------ inspection
    def mbon_drive_change(self):
        """Per MBON: learned / native total KC weight (1 = unchanged)."""
        w = np.bincount(self.e_m, weights=self.w0 * self.f,
                        minlength=self.mbon_slots.size)
        w0 = np.bincount(self.e_m, weights=self.w0, minlength=self.mbon_slots.size)
        return np.divide(w, w0, out=np.ones_like(w), where=w0 > 0)

    def state_dict(self):
        return {"f": self.f.copy()}

    def load_state_dict(self, d):
        self.f = np.asarray(d["f"], dtype=np.float32).copy()
        self.dw = ((self.f - 1.0) * self.w0 * self.w_scale).astype(np.float32)
        self.any_learned = bool(np.any(self.f != 1.0))
