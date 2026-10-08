"""SuperFly: the whole stack around one fly brain.

    L0  the fly        SuperflyEngine (native 138,639-neuron kernel, untouched)
    L1  its learning   MBPlasticity (dopamine-gated KC->MBON)
    L2  its interface  senses in (sensory port), readouts out
    L3  its grafts     word-sense PNs (+ optional expanded mushroom body)
    L4  its voice      superfly/flylm.py / superfly/language.py, outside this file

Everything the layers above may know about the fly comes from ``observe()``,
which reads spike counts and nothing else.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from superfly.anatomy import Atlas, REGIONS
from superfly.engine import SuperflyEngine, Readout
from superfly.graft import GraftBuilder, load_connectome, word_code
from superfly.language import CONCEPTS, CONCEPT, ACTIONS
from superfly.plasticity import MBPlasticity

# Feature space for the voice: mean rate per CELL TYPE, over the cell classes
# below. Sensory neurons are excluded on purpose -- a percept has to reach the
# brain to be spoken about; the receptors firing is not enough.
FEATURE_SUPER = ("central", "descending", "motor", "endocrine", "ascending",
                 "visual_projection")


@dataclass
class Observation:
    t_ms: float
    window_ms: float
    features: np.ndarray        # per cell type, Hz
    actions: dict               # action -> Hz of its descending population
    mbon: np.ndarray            # per MBON neuron, Hz
    valence: float              # MBON population readout
    regions: dict               # region -> Hz
    spikes: int


class SuperFly:
    def __init__(self, atlas: Atlas | None = None, *, word_pns=150, xmb=0,
                 xmb_share=0.5, plasticity=True, seed=0, threads=None,
                 model=None, gain=None, terminals=False, plasticity_kw=None,
                 da_modulatory=None, drives=(), verbose=True):
        t0 = time.perf_counter()
        self.atlas = at = atlas or Atlas()
        conn = load_connectome(at) if (word_pns or xmb or plasticity) else None
        gb = GraftBuilder(at, seed=seed)
        self.g_xmb = gb.expanded_mushroom_body(conn, n_kc=xmb, share=xmb_share) \
            if xmb else None
        self.g_words = gb.word_sense(conn, n_pn=word_pns,
                                     also=[self.g_xmb] if self.g_xmb else []) \
            if word_pns else None
        if terminals:
            gb.peripheral_terminals()
        # with plasticity on, dopamine's KC/MBON action IS the plasticity
        if da_modulatory is None:
            da_modulatory = plasticity
        if da_modulatory:
            gb.dopamine_modulatory()
        self.da_modulatory = bool(da_modulatory)
        self.gb = gb
        sensory = set()
        for c in CONCEPTS:
            for p in c.pops:
                sensory.update(int(x) for x in at[p].ids)
        if self.g_words is not None:
            sensory.update(int(x) for x in self.g_words.ids)
        for p in drives:                     # extra driven populations (interoception)
            sensory.update(int(x) for x in at[p].ids)
        kw = {} if model is None else {"model": model}
        self.e = e = SuperflyEngine(sensory_ids=sorted(sensory), seed=seed,
                                    threads=threads, gain=gain,
                                    extend=gb.extender() if (gb.grafts or terminals or da_modulatory) else None,
                                    **kw)
        # ---- concept -> positions in the stimulated set
        self.concept_pos = {}
        for c in CONCEPTS:
            slots = np.concatenate([at[p].idx(e) for p in c.pops])
            self.concept_pos[c.key] = e.stim_positions(slots)
        self.drive_pos = {p: e.stim_positions(at[p].idx(e)) for p in drives}
        self.word_pos = None
        if self.g_words is not None:
            self.word_pos = e.stim_positions(e.indices_of(self.g_words.ids))
        self.lexicon = {}                    # word -> the PNs it activates
        # ---- learning
        self.mb = None
        if plasticity:
            kc = at["mb.KC"].idx(e)
            if self.g_xmb is not None:
                kc = np.concatenate([kc, e.indices_of(self.g_xmb.ids)])
            self.mb = MBPlasticity(e, at, kc_slots=kc, conn=conn,
                                   **(plasticity_kw or {}))
            e.add_emitter(self.mb)
            e.add_ticker(self.mb, self.mb.every)
        self.pam = at["mb.PAM"].idx(e)
        self.ppl1 = at["mb.PPL1"].idx(e)
        self._teach = None
        self._tone = None                    # (slots, rate_hz, g) background input
        e.add_emitter(self)                  # DAN teaching through the synaptic port
        # ---- readouts
        ann = at.ann
        keep = ann.super_class.isin(FEATURE_SUPER).to_numpy(bool) \
            & (ann.cell_type != "").to_numpy(bool)
        types = ann.cell_type.to_numpy()
        self.feat_names = np.array(sorted(set(types[keep])))
        tix = {t: i for i, t in enumerate(self.feat_names)}
        orig_rows = np.flatnonzero(keep)
        slots = e.indices_of(at.sim_ids[orig_rows])
        self._feat_slots = slots
        self._feat_ix = np.asarray([tix[t] for t in types[orig_rows]])
        self._feat_n = np.bincount(self._feat_ix, minlength=len(tix)).astype(float)
        self.act_groups = {a: at[f"dn.{a}"].idx(e) for a in ACTIONS if a != "feed"}
        self.act_groups["feed"] = at["motor.MN9"].idx(e)
        reg = at.region_slots(e)
        self.region_groups = {k: np.flatnonzero(reg == i)
                              for i, (k, _, _) in enumerate(REGIONS)}
        self.mbon_slots = at["mb.MBON"].idx(e)
        self._last = e.counts.copy()
        self._t_last = e.t_ms
        if verbose:
            print(f"SuperFly: {e.n_native} native + {e.N - e.n_native} grafted "
                  f"neurons, {len(self.feat_names)} cell-type features, "
                  f"plastic edges {self.mb.n_edges if self.mb else 0}, "
                  f"built in {time.perf_counter() - t0:.1f}s", flush=True)

    # ------------------------------------------------------------ teaching
    def teach(self, sign, hz=40.0, mv=40.0):
        """Activate reward (sign>0, PAM) or punishment (sign<0, PPL1) DANs by
        Poisson 'virtual synapses' (synaptic port), the in-silico analogue of
        the optogenetic DAN activation used to train flies. 0 stops it."""
        if not sign:
            self._teach = None
            return
        self._teach = (self.pam if sign > 0 else self.ppl1, hz * abs(sign), mv)

    def tone(self, rate_hz=0.0, g=2.0, pop="mb.MBON"):
        """Spontaneous activity for a population, as tonic synaptic input.

        Real MBONs fire spontaneously (Hige et al. 2015 recorded baselines of
        a few to ~20 Hz) and learning shows up as a change from it. The Shiu
        model has no spontaneous activity anywhere, so a sparse KC code never
        reaches MBON threshold and the memory has no readout. This adds a
        Poisson background through the synaptic port -- refractoriness and
        integration untouched -- and is OFF by default."""
        self._tone = (self.atlas[pop].idx(self.e), float(rate_hz), float(g)) \
            if rate_hz > 0 else None

    def emit(self, engine, prev):
        out_i, out_v = [], []
        if self._tone is not None:
            slots, hz, g = self._tone
            hit = engine.rng.random(slots.size) < hz * engine.dt / 1000.0
            if hit.any():
                out_i.append(slots[hit])
                out_v.append(np.full(int(hit.sum()), g, dtype=np.float32))
        r = self._emit_teach(engine)
        if r is not None:
            out_i.append(r[0])
            out_v.append(r[1])
        if not out_i:
            return None
        return np.concatenate(out_i), np.concatenate(out_v)

    def _emit_teach(self, engine):
        if self._teach is None:
            return None
        slots, hz, mv = self._teach
        hit = engine.rng.random(slots.size) < hz * engine.dt / 1000.0
        if not hit.any():
            return None
        s = slots[hit]
        # a fixed depolarisation per event, independent of the network gain:
        # the teacher is an experimenter's light pulse, not a fly synapse
        return s, np.full(s.size, mv, dtype=np.float32)

    # ------------------------------------------------------------ senses
    def word_channel(self, word, allocate=True):
        """Positions (in the stimulated set) of the PNs this word activates."""
        if self.word_pos is None:
            return None
        if word not in self.lexicon:
            self.lexicon[word] = word_code(word, n_pn=self.word_pos.size)
        return self.word_pos[self.lexicon[word]]

    def sense(self, concepts=None, words=(), word_hz=150.0):
        """Set the sensory drive: concepts = {key: intensity 0..1}."""
        e = self.e
        e._rate_np[:] = 0.0
        for k, inten in (concepts or {}).items():
            e._rate_np[self.concept_pos[k]] = CONCEPT[k].rate_hz * float(inten)
        for wd in words:
            pos = self.word_channel(wd)
            if pos is not None:
                e._rate_np[pos] = word_hz
        e._refresh_refrac()
        e._poi_block = None

    def run(self, ms):
        return self.e.run(ms)

    def rest(self, ms=200.0):
        """No stimulus and no teaching (spontaneous activity, if on, continues)."""
        self.sense({})
        self.teach(0)
        return self.run(ms)

    def reset(self):
        """Back to a quiet brain at t=0. Learned weights persist."""
        self.sense({})
        self.teach(0)
        self.e.full_reset()
        self._last = self.e.counts.copy()
        self._t_last = self.e.t_ms

    # ------------------------------------------------------------ readout
    def mark(self):
        self._last = self.e.counts.copy()
        self._t_last = self.e.t_ms

    def observe(self) -> Observation:
        e = self.e
        dt_s = max(e.t_ms - self._t_last, 1e-9) / 1000.0
        d = (e.counts - self._last).astype(float)
        self._last = e.counts.copy()
        win = e.t_ms - self._t_last
        self._t_last = e.t_ms
        self._last_window = (d / dt_s).astype(np.float32)   # per-neuron Hz
        f = np.bincount(self._feat_ix, weights=d[self._feat_slots],
                        minlength=len(self.feat_names)) / self._feat_n / dt_s
        acts = {a: float(d[s].sum() / max(s.size, 1) / dt_s)
                for a, s in self.act_groups.items()}
        mb = d[self.mbon_slots] / dt_s
        val = 0.0
        if self.mb is not None and mb.sum() > 0:
            val = float(np.dot(self.mb.valence, mb) / (np.abs(self.mb.valence) @ mb + 1e-9))
        regs = {k: float(d[s].sum() / max(s.size, 1) / dt_s)
                for k, s in self.region_groups.items() if s.size}
        return Observation(e.t_ms, win, f.astype(np.float32), acts, mb, val,
                           regs, int(d.sum()))

    # ------------------------------------------------------------ voice features
    def voice_features(self, obs):
        """What the language bridge sees: per-cell-type rates of the central
        brain (Kenyon-cell types dropped) followed by every Kenyon cell's own
        rate. Must match superfly/experiments/make_corpus.py."""
        if not hasattr(self, "_vf_ix"):
            self._vf_ix = np.array([i for i, t in enumerate(self.feat_names)
                                    if not t.startswith("KC")])
            self._vf_kc = self.atlas["mb.KC"].idx(self.e)
        return np.concatenate([obs.features[self._vf_ix],
                               self._last_window[self._vf_kc]]).astype(np.float32)

    def present(self, concepts=None, words=(), ms=300.0, settle_ms=50.0):
        """Present a stimulus and observe the brain's response to it."""
        self.sense(concepts, words)
        self.run(settle_ms)
        self.mark()
        self.run(ms - settle_ms)
        obs = self.observe()
        return obs
