"""Grafts: new neurons wired into the fly's own circuits.

A graft here is not a separate network that talks to the fly through an API.
Its neurons are appended to the connectome itself (NativeBrainEngine's
``extend`` hook), so the native kernel integrates them with the SAME membrane
equation, the same 1.8 ms delay, the same refractory gate and the same tiling
as every native cell. The fly cannot tell a grafted neuron from its own except
by what it connects to.

Every grafted neuron is a STATISTICAL CLONE of a native cell type: its inputs
and outputs are drawn from the measured connectivity of real neurons of that
type, then scaled by an explicit gain. Nothing about the graft is hand-wired
to produce a behaviour.

Two grafts are provided.

ExpandedMushroomBody -- more Kenyon cells.
    The fly's associative memory is ~5,000 Kenyon cells (FlyWire v783: 5,177).
    The honeybee mushroom body is reported at roughly 170,000 KCs per brain
    side-pair order of magnitude larger (see research/review/03 for the
    sourced figure), and bees solve learning problems flies are not known to.
    Expanding the KC layer is therefore the evolution-shaped way to give a fly
    more memory: same inputs (projection neurons), same outputs (MBONs), same
    inhibition (APL), same teacher (DANs), same plasticity -- just more of it.
    Each synthetic KC copies the output profile of a random real KC of a
    chosen class, and draws fresh input claws from real PNs with the measured
    claw-count and synapse-count distributions. Output gain is set so the
    graft supplies a chosen share of each MBON's KC input (default 0.5), which
    is the knob that says how much "new mind" the memory has.

WordSense -- a new sensory organ for words.
    A bank of synthetic projection neurons, one small group per word, wired
    into the calyx like real PNs (each contacting ~40 KCs). A word the fly
    hears becomes a sparse KC pattern exactly like an odour, and the fly's
    own dopamine-gated plasticity can attach a value to it. This is the
    honest form of "giving a fly language": words become things it can learn
    about, using the learning machinery it already has.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SYN_ID_BASE = 900_000_000_000_000_000      # synthetic FlyWire-like ids


@dataclass
class GraftSpec:
    name: str
    n: int
    ids: np.ndarray = None                  # assigned at build
    slot0: int = -1                         # first engine slot
    meta: dict = field(default_factory=dict)


class GraftBuilder:
    """Collects synthetic neurons and edges, then produces ``extend``."""

    def __init__(self, atlas, seed=0):
        self.atlas = atlas
        self.rng = np.random.default_rng(seed)
        self.grafts: list[GraftSpec] = []
        self._edges = []                    # (pre, post, val) in CSV index space
        self.n_native = atlas.N
        self.n_total = atlas.N

    # ------------------------------------------------------------- helpers
    def _alloc(self, name, n, **meta):
        g = GraftSpec(name, n, slot0=self.n_total, meta=meta)
        g.ids = np.arange(SYN_ID_BASE + self.n_total,
                          SYN_ID_BASE + self.n_total + n, dtype=np.int64)
        self.n_total += n
        self.grafts.append(g)
        return g

    def _add(self, pre, post, val):
        pre = np.asarray(pre, np.int64)
        post = np.asarray(post, np.int64)
        val = np.asarray(val, np.int64)
        keep = val != 0
        if keep.any():
            self._edges.append((pre[keep], post[keep], val[keep]))

    def _rows(self, mask):
        return np.flatnonzero(mask)

    def _csv_mask(self, pop):
        return np.isin(self.atlas.sim_ids, self.atlas[pop].ids)

    @staticmethod
    def _stoch_round(x, rng):
        f = np.floor(x)
        return (f + (rng.random(x.shape) < (x - f))).astype(np.int64)

    # ------------------------------------------------------------- grafts
    def expanded_mushroom_body(self, conn, n_kc=20000, kc_class="mb.KC",
                               share=0.5, name="xMB"):
        """Synthetic KCs; see module docstring. ``conn`` = (pre, post, w) CSV
        index arrays of the native connectome."""
        pre, post, w = conn
        at, rng = self.atlas, self.rng
        KC = self._csv_mask(kc_class)
        PN = self._csv_mask("al.PN")
        MBON = self._csv_mask("mb.MBON")
        APL = self._csv_mask("mb.APL")
        kc_rows = self._rows(KC)

        g = self._alloc(name, n_kc, share=share, template=kc_class)
        new = np.arange(g.slot0, g.slot0 + n_kc)

        # --- input claws: PN -> KC statistics --------------------------
        e_in = PN[pre] & KC[post]
        claws = np.bincount(post[e_in], minlength=at.N)[kc_rows]
        claws = claws[claws > 0]
        pn_out = np.bincount(pre[e_in], minlength=at.N).astype(float)
        pn_rows = self._rows(pn_out > 0)
        p_pn = pn_out[pn_rows] / pn_out[pn_rows].sum()
        syn_in = w[e_in]
        k = rng.choice(claws, n_kc)
        ip = np.concatenate([rng.choice(pn_rows, kk, replace=False, p=p_pn)
                             for kk in k])
        ipost = np.repeat(new, k)
        self._add(ip, ipost, rng.choice(syn_in, ip.size))

        # --- outputs and APL loop: clone a template real KC each ------
        tmpl = rng.choice(kc_rows, n_kc)
        order = np.argsort(pre, kind="stable")
        ps, qs, ws = pre[order], post[order], w[order]
        start = np.searchsorted(ps, np.arange(at.N + 1))
        out_pre, out_post, out_w = [], [], []
        for j, t in enumerate(tmpl):
            a, b = start[t], start[t + 1]
            q, ww = qs[a:b], ws[a:b]
            keep = MBON[q] | APL[q]
            out_pre.append(np.full(keep.sum(), new[j]))
            out_post.append(q[keep])
            out_w.append(ww[keep])
        op, oq, ow = (np.concatenate(out_pre), np.concatenate(out_post),
                      np.concatenate(out_w).astype(float))
        # gain: the graft supplies `share` of each MBON's total KC input
        native_kc_to_mbon = np.bincount(post[KC[pre] & MBON[post]],
                                        weights=np.abs(w[KC[pre] & MBON[post]]),
                                        minlength=at.N)
        to_m = MBON[oq]
        syn_to_mbon = np.bincount(oq[to_m], weights=np.abs(ow[to_m]),
                                  minlength=at.N)
        tot_native = native_kc_to_mbon[MBON].sum()
        tot_syn = syn_to_mbon[MBON].sum()
        gain_m = (share / (1 - share)) * tot_native / max(tot_syn, 1)
        apl_native = np.abs(w[KC[pre] & APL[post]]).sum()
        to_apl = APL[oq]
        gain_apl = (share / (1 - share)) * apl_native / max(np.abs(ow[to_apl]).sum(), 1)
        ow[to_m] *= gain_m
        ow[to_apl] *= gain_apl
        self._add(op, oq, self._stoch_round(ow, rng))
        # APL -> synthetic KC inhibition, copied from APL -> template KC
        e_apl = APL[pre] & KC[post]
        apl_to = np.zeros(at.N)
        np.add.at(apl_to, post[e_apl], w[e_apl])
        apl_rows = self._rows(APL)
        for a_row in apl_rows:
            sel = e_apl & (pre == a_row)
            per_kc = np.zeros(at.N)
            np.add.at(per_kc, post[sel], w[sel])
            v = per_kc[tmpl]
            self._add(np.full(n_kc, a_row), new, v)
        g.meta.update(gain_mbon=float(gain_m), gain_apl=float(gain_apl),
                      mean_claws=float(k.mean()))
        return g

    def word_sense(self, conn, n_pn=150, name="words", also=()):
        """A second antennal lobe, for words.

        ``n_pn`` synthetic projection neurons. Every real Kenyon cell receives
        claws from them with the SAME statistics as its real PN input: the
        number of claws is drawn from the measured PN-partner count of KCs,
        the synapse count of each claw from the measured PN->KC distribution.
        A word is then a combinatorial pattern over these PNs (see
        ``word_code``), exactly as an odour is a pattern over glomeruli, so
        KCs recruited by a word are those whose claws happen to sample enough
        of its active PNs -- sparse, random, convergent coding, the textbook
        mushroom-body scheme (Litwin-Kumar et al. 2017).
        """
        pre, post, w = conn
        at, rng = self.atlas, self.rng
        KC = self._csv_mask("mb.KC")
        PN = self._csv_mask("al.PN")
        e = PN[pre] & KC[post]
        kc_rows = self._rows(KC)
        claws = np.bincount(post[e], minlength=at.N)[kc_rows]
        claws = claws[claws > 0]
        syn = w[e]
        g = self._alloc(name, n_pn, n_pn=n_pn)
        # grafted KCs (e.g. an expanded mushroom body) hear words too
        extra = [np.arange(gx.slot0, gx.slot0 + gx.n) for gx in also]
        targets = np.concatenate([kc_rows] + extra)
        k = rng.choice(claws, targets.size)          # claws per KC
        src = np.concatenate([rng.choice(n_pn, kk, replace=False) for kk in k])
        dst = np.repeat(targets, k)
        self._add(g.slot0 + src, dst, rng.choice(syn, src.size))
        g.meta.update(mean_claws=float(k.mean()), edges=int(src.size))
        return g

    # ------------------------------------------------------------- corrections
    def peripheral_terminals(self, conn=None):
        """Central synapses cannot fire a sensory neuron.

        Every FlyWire sensory neuron (super_class 'sensory') has its cell body
        and spike-initiation zone in a peripheral organ -- antenna, labellum,
        eye, leg. What the connectome contains of it is its AXON TERMINAL
        arbor in the brain, so every synapse it receives there is axo-axonic:
        in the fly it modulates transmitter release (e.g. GABAergic
        presynaptic inhibition of ORNs; Olsen & Wilson 2008) and cannot
        initiate a spike. A point-neuron LIF cannot tell a terminal input from
        a dendritic one (flybench FINDINGS, 2026-09-20), so it lets local
        neurons fire every receptor neuron in the antennal lobe -- the
        broadcast that destroys odour identity. This removes that input. It
        needs no synapse positions: for these cells, ALL central input is
        terminal input. Release modulation itself is not modelled here.
        """
        self._block_post = np.flatnonzero(
            (self.atlas.ann.super_class == "sensory").to_numpy(bool))
        return self._block_post.size

    def dopamine_modulatory(self):
        """Dopamine acts on Kenyon cells and MBONs through plasticity, not as a
        fast excitatory transmitter.

        The Shiu convention signs every monoamine synapse excitatory, so a DAN
        spike depolarises the KCs and MBONs it contacts. Mushroom-body
        dopamine receptors (Dop1R1, Dop1R2, DopEcR) are G-protein coupled:
        their documented effect is the plasticity of KC->MBON synapses
        (superfly/plasticity.py), which is modelled explicitly. Keeping the
        fast excitation as well counts the same dopamine twice, and makes
        every teaching pulse excite KCs broadly, so every KC becomes eligible
        and learning loses its specificity (measured: 50,629 of 62,261 edges
        changed in a 3-trial protocol). Removes DAN->KC and DAN->MBON edges.
        """
        at = self.atlas
        dan = self._csv_mask("mb.DAN")
        tgt = self._csv_mask("mb.KC") | self._csv_mask("mb.MBON")
        self._drop_pairs = (dan, tgt)
        return int(dan.sum()), int(tgt.sum())

    # ------------------------------------------------------------- build
    def extender(self):
        """The ``extend`` callable for NativeBrainEngine (pre-permutation)."""
        edges = self._edges
        n_total = self.n_total
        block = getattr(self, "_block_post", None)
        drop = getattr(self, "_drop_pairs", None)

        def extend(engine):
            N0 = engine.N
            lens = np.diff(engine.crow)
            pre0 = np.repeat(np.arange(N0, dtype=np.int64), lens)
            post0 = engine.post.astype(np.int64)
            val0 = engine.val.astype(np.int64)
            if edges:
                ap = np.concatenate([e[0] for e in edges])
                aq = np.concatenate([e[1] for e in edges])
                av = np.concatenate([e[2] for e in edges])
                pre = np.concatenate([pre0, ap])
                post = np.concatenate([post0, aq])
                val = np.concatenate([val0, av])
            else:
                pre, post, val = pre0, post0, val0
            if drop is not None:
                pm, qm = drop
                ext_p = np.zeros(n_total, bool); ext_p[:pm.size] = pm
                ext_q = np.zeros(n_total, bool); ext_q[:qm.size] = qm
                cut = ext_p[pre] & ext_q[post]
                engine.n_dropped_da = int(cut.sum())
                pre, post, val = pre[~cut], post[~cut], val[~cut]
            if block is not None and block.size:
                cut = np.isin(post, block)
                engine.n_blocked_edges = int(cut.sum())
                pre, post, val = pre[~cut], post[~cut], val[~cut]
            key = pre * n_total + post
            o = np.argsort(key, kind="stable")
            key, val = key[o], val[o]
            uk, st = np.unique(key, return_index=True)
            vs = np.add.reduceat(val, st)
            pu, qu = uk // n_total, uk % n_total
            assert np.abs(vs).max() <= 32767
            crow = np.zeros(n_total + 1, dtype=np.int64)
            np.cumsum(np.bincount(pu, minlength=n_total), out=crow[1:])
            new_ids = np.concatenate([g.ids for g in self.grafts]) \
                if self.grafts else np.zeros(0, np.int64)
            return crow, qu.astype(np.int32), vs.astype(np.int16), new_ids

        return extend

    def slots(self, engine, name):
        g = next(x for x in self.grafts if x.name == name)
        return engine.indices_of(g.ids)


def word_code(word, n_pn=150, per_ngram=12, n=3):
    """PNs a word activates: hashed character n-grams of '^word$'.

    Deterministic (no Python hash randomisation), and similar-sounding words
    share n-grams and therefore PNs -- so they overlap in the mushroom body
    the way similar odours do, and what is learned about one generalises a
    little to the other, as it does for odours.
    """
    import hashlib
    s = f"^{word.lower()}$"
    grams = {s[i:i + n] for i in range(max(1, len(s) - n + 1))}
    act = set()
    for gr in sorted(grams):
        h = hashlib.sha256(gr.encode()).digest()
        r = np.random.default_rng(int.from_bytes(h[:8], "little"))
        act.update(int(x) for x in r.choice(n_pn, per_ngram, replace=False))
    return np.array(sorted(act), dtype=np.int64)


def load_connectome(atlas, data_dir=None):
    """Native (pre, post, signed weight) in CSV index space."""
    import pandas as pd
    from superfly.anatomy import DATA
    d = data_dir or DATA
    c = pd.read_parquet(d / "2025_Connectivity_783.parquet",
                        columns=["Presynaptic_Index", "Postsynaptic_Index",
                                 "Excitatory x Connectivity"])
    return (c.Presynaptic_Index.to_numpy(np.int64),
            c.Postsynaptic_Index.to_numpy(np.int64),
            c["Excitatory x Connectivity"].to_numpy(np.int64))
