"""Run flybench (github.com/brandoncho369/flybench) against SUPERFLY's engine.

flybench is an independent, cited behaviour suite for whole-brain fly
simulations (36 tasks, shuffled-wiring controls). We use it as the
"is it still a fly?" test, so the fly is judged by someone else's ruler.

Two pieces:

  build_connectome()   flybench's Connectome built from THIS repo's data
                       (2025_Connectivity_783.parquet + the v783 annotation
                       table), in the engine's own neuron order, so the
                       reference LIF and our kernel see the same graph.
                       flybench's own builder reads FlyWire Codex CSVs (which
                       need an account and are blocked in this environment);
                       differences are recorded in the connectome's meta.

  NativeSim            the flybench adapter: SuperflyEngine under flybench's
                       contract. params.extra["variant"] picks the fly:
                         shiu      the published model, nothing attached
                         superfly  + MB plasticity, grafts present, untaught
                         ntfix     + literature neurotransmitter corrections

    flybench run -c flywire783_repo --gain 1.0 \
        --simulator superfly.bench.flybench_adapter:NativeSim
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flybench.adapter import SimulatorAdapter
from flybench.connectome import Connectome, DEFAULT_CACHE
from flybench.sim import LIFParams, SimResult

NAME = "flywire783_repo"
_NT_CODE = {"acetylcholine": "ACH", "gaba": "GABA", "glutamate": "GLUT",
            "dopamine": "DA", "serotonin": "SER", "octopamine": "OCT"}


def build_connectome(cache=DEFAULT_CACHE, name=NAME):
    from superfly.anatomy import DATA, load_annotations
    comp = pd.read_csv(DATA / "2025_Completeness_783.csv", index_col=0)
    ids = np.asarray(comp.index, dtype=np.int64)
    n = len(ids)
    c = pd.read_parquet(DATA / "2025_Connectivity_783.parquet",
                        columns=["Presynaptic_Index", "Postsynaptic_Index",
                                 "Connectivity", "Excitatory x Connectivity"])
    W = sp.csr_matrix((c["Excitatory x Connectivity"].to_numpy(np.float32),
                       (c.Presynaptic_Index.to_numpy(), c.Postsynaptic_Index.to_numpy())),
                      shape=(n, n), dtype=np.float32)
    ann = load_annotations(DATA).drop_duplicates("root_id").set_index("root_id").reindex(ids)
    s = lambda col: ann[col].astype("string").fillna("").to_numpy()
    labels = [" | ".join(x for x in (a, b, cc) if x) for a, b, cc in
              zip(s("synonyms"), s("cell_sub_class"), s("hemibrain_type"))]
    out = pd.DataFrame({
        "root_id": ids, "flow": s("flow"), "super_class": s("super_class"),
        "class": s("cell_class"), "sub_class": s("cell_sub_class"),
        "cell_type": s("cell_type"), "hemibrain_type": s("hemibrain_type"),
        "hemilineage": s("ito_lee_hemilineage"), "side": s("side"),
        "nerve": s("nerve"), "labels": labels,
        "nt_type": [_NT_CODE.get(x, "") for x in s("top_nt")],
    })
    pos = np.stack([ann.pos_x.to_numpy(float) * 4, ann.pos_y.to_numpy(float) * 4,
                    ann.pos_z.to_numpy(float) * 40], 1).astype(np.float32)
    con = Connectome(root_ids=ids, W=W, positions=pos, annotations=out, name=name,
                     meta={"source": "fly-brain repo parquet + flywire_annotations v783",
                           "min_synapses": int(c.Connectivity.min()),
                           "labels": "synthesised from synonyms|cell_sub_class|hemibrain_type "
                                     "(Codex community labels unavailable here)",
                           "n": n, "n_edges": int(W.nnz)})
    path = con.save(Path(cache) / name)
    print(f"saved {path}: {n} neurons, {W.nnz} edges, min synapses "
          f"{con.meta['min_synapses']}")
    return con


_ENGINES: dict = {}


def _engine(variant):
    if variant in _ENGINES:
        return _ENGINES[variant]
    from superfly.engine import SuperflyEngine
    threads = int(os.environ.get("SUPERFLY_THREADS", "0")) or None
    if variant == "shiu":
        e = SuperflyEngine(threads=threads)
        extra = None
    elif variant in ("superfly", "ntfix"):
        from superfly.fly import SuperFly
        f = SuperFly(threads=threads, ntfix=(variant == "ntfix"), verbose=False)
        e, extra = f.e, f
    else:
        raise ValueError(f"unknown variant {variant!r}")
    slot_of = e.inv[:e.n_native] if e.inv is not None else np.arange(e.n_native)
    orig_of = e.perm if e.perm is not None else np.arange(e.N)
    _ENGINES[variant] = (e, extra, np.asarray(slot_of), np.asarray(orig_of),
                         e.val.copy())
    return _ENGINES[variant]


class NativeSim(SimulatorAdapter):
    capabilities = frozenset({"can_silence"})

    def __init__(self, connectome: Connectome, params: LIFParams | None = None):
        super().__init__(connectome, params)
        self.variant = self.p.extra.get("variant",
                                        os.environ.get("SUPERFLY_VARIANT", "shiu"))
        (self.e, self.fly, self.slot_of, self.orig_of,
         self.val0) = _engine(self.variant)
        if connectome.n != self.e.n_native:
            raise ValueError(f"connectome has {connectome.n} neurons, engine "
                             f"{self.e.n_native}: build it with build_connectome()")
        # silencing: rows the task zeroed relative to the full graph
        row_abs = np.asarray(abs(connectome.W).sum(1)).ravel()
        self.silenced = np.flatnonzero((row_abs == 0) & (self._base_out() > 0))

    _BASE = None

    def _base_out(self):
        if NativeSim._BASE is None:
            lens = np.diff(self.e.crow)[:self.e.n_native] if self.e.perm is None \
                else np.diff(self.e.crow)[self.slot_of]
            NativeSim._BASE = lens
        return NativeSim._BASE

    def _prepare(self, stimuli):
        e = self.e
        # gain scales every recurrent synapse; sensory drive is a forced spike
        ws = np.float32(self.p.w_syn_mv * self.p.gain)
        e.w_scale = ws
        e._cf["w_scale"].value = float(ws)
        # restore the full graph, then silence (outgoing synapses only, as
        # flybench defines it)
        e.val[:] = self.val0
        for i in self.silenced:
            s = self.slot_of[i]
            e.val[e.crow[s]:e.crow[s + 1]] = 0
        # stimulated set for this run
        e.refrac_steps[:] = e.base_refrac
        e.tile_pin[:] = 0
        orig = np.unique(np.concatenate([np.asarray(s.neurons, np.int64)
                                         for s in stimuli])) if stimuli else []
        e.set_stim_neurons(e.i2flyid[self.slot_of[orig]].tolist() if len(orig) else [])
        self._pos = [e.stim_positions(self.slot_of[np.asarray(s.neurons, np.int64)])
                     for s in stimuli]
        e.rng = np.random.default_rng(self.p.seed)
        e.full_reset()
        if self.fly is not None and self.fly.mb is not None:
            self.fly.mb.on_reset_weights()

    def run(self, duration_ms: float, stimuli=None) -> SimResult:
        stimuli = [s for s in (stimuli or []) if len(s.neurons)]
        self._prepare(stimuli)
        e = self.e
        n_steps = int(round(duration_ms / e.dt))
        upd = 10                                       # rates refreshed every 1 ms
        times, ids = [], []
        for k in range(n_steps):
            t = k * e.dt
            if k % upd == 0 and stimuli:
                e._rate_np[:] = 0.0
                for s, pos in zip(stimuli, self._pos):
                    if s.t_start_ms <= t < s.t_end_ms:
                        e._rate_np[pos] = s.rates_at(t)
                e._refresh_refrac()
                e._poi_block = None
            n = e.step()
            if n:
                sl = e.spike_idx
                sl = sl[sl < e.n_native]
                if sl.size:
                    times.append(np.full(sl.size, t + e.dt, np.float32))
                    ids.append(self.orig_of[sl].astype(np.int32))
        st = np.concatenate(times) if times else np.empty(0, np.float32)
        si = np.concatenate(ids) if ids else np.empty(0, np.int32)
        return SimResult(spike_times_ms=st, spike_neurons=si,
                         duration_ms=duration_ms, n=self.c.n)


if __name__ == "__main__":
    build_connectome()
