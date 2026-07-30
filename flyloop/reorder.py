"""Neuron reordering for tile-skipping (gates the "reorder for locality" idea).

Motivation (HANDOFF.md sec 11.6 / research/reorder_measurements.md): after
1500 steps of the sugar protocol only ~7.7% of neurons are "live" (state
differs from rest), but in the shipped FlyWire-CSV order they are scattered
1.1-neuron runs, so 64-wide tile-skipping saves only ~1%. If neurons that tend
to be live together are placed adjacent in the index, tiles can be skipped in
bulk instead.

This module answers: which grouping key, computed ONCE and independent of any
stimulus, packs live neurons together best -- and does the win generalise
across regimes, or is it an artifact of the sugar protocol specifically?

------------------------------------------------------------------------------
Neuron index convention
------------------------------------------------------------------------------
The "original index" of a neuron is its ROW POSITION in
``data/2025_Completeness_783.csv`` (row 0 = index 0), matching
``BrainEngine.i2flyid`` / ``flyid2i``. All permutations below are permutations
of that index space: ``perm`` is an array of length N with
``perm[new_index] = old_index`` (i.e. ``reordered_array = original_array[perm]``
gathers the original array into the new order).

------------------------------------------------------------------------------
Supported keys
------------------------------------------------------------------------------
  'none'                     identity permutation (perm = arange(N)).
  'cell_type'                group by the `cell_type` column.
  'super_class+cell_class'   group by `super_class` + '|' + `cell_class`.
  'hemibrain_type'           group by the `hemibrain_type` column.
  'morton'                   group by 3D Z-order (Morton code) of
                              (soma_x, soma_y, soma_z), 21 bits/axis.

------------------------------------------------------------------------------
Determinism / the 20,553 unannotated neurons
------------------------------------------------------------------------------
``data/flywire_meta/neuron_annotations.tsv`` does not cover every neuron for
every field. Coverage is PER FIELD, not one global number -- e.g. for our
138,639 neurons, `cell_type` is present for 137,147 (1,492 missing) while
soma position (needed for 'morton') is present for only 118,086 (20,553
missing). Each key therefore defines "unannotated" as: the field(s) that key
needs are empty/NaN for that neuron (or the neuron's root_id is absent from
the tsv entirely, which reindexing turns into the same NaN case).

DESIGN CHOICE (must be documented, not just implemented): every unannotated
neuron for a given key is placed into ONE group, and that group is sorted
to the END -- after every annotated group in ascending label order. Within
every group (annotated or not) neurons keep their ORIGINAL INDEX ORDER,
because the sort is a stable sort (`kind='mergesort'`) applied to data that
is fed in in original-index order, and stable sort never reorders ties. This
makes the whole permutation a pure function of the on-disk data: same inputs
-> same perm, byte for byte, every time.

('morton' sorts its annotated group by the Morton code, ascending, rather
than a label string -- everything else about the rule is identical.)

------------------------------------------------------------------------------
Caching
------------------------------------------------------------------------------
Each permutation is cached at ``<data_dir>/reorder_cache/perm_<key>.npy``
(int64, length N). Delete the file to force a rebuild (e.g. after the
annotations tsv is updated). Caching is purely a disk artifact of calling
this module -- it does not modify anything already in the repo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Sentinel group label for "this key's field(s) are missing for this neuron".
# Chosen to sort AFTER any real annotation string in a plain ascending string
# sort (annotation values in this dataset are ASCII identifiers/short codes;
# this sentinel is longer and higher in codepoint than anything observed).
_UNANNOTATED = "￿￿_UNANNOTATED"

_ANN_COLUMNS = ["root_id", "super_class", "cell_class", "cell_type",
                "hemibrain_type", "soma_x", "soma_y", "soma_z"]

_VALID_KEYS = ("none", "cell_type", "super_class+cell_class",
               "hemibrain_type", "morton")


def _load_ids(data_dir: Path) -> np.ndarray:
    comp = pd.read_csv(data_dir / "2025_Completeness_783.csv", index_col=0)
    return comp.index.to_numpy(dtype=np.int64)


def _load_annotations(data_dir: Path, ids: np.ndarray) -> pd.DataFrame:
    """Annotation table aligned 1:1 to `ids` (original index order).

    Missing root_ids (not in the tsv) become all-NaN rows -- handled
    identically to "field present but empty" by every key below.
    """
    ann_path = data_dir / "flywire_meta" / "neuron_annotations.tsv"
    ann = pd.read_csv(ann_path, sep="\t", dtype=str, usecols=_ANN_COLUMNS)
    ann["root_id"] = ann["root_id"].astype(np.int64)
    # The tsv has zero duplicate root_ids in this dataset (verified), but
    # guard deterministically anyway: keep the first occurrence.
    ann = ann.drop_duplicates(subset="root_id", keep="first").set_index("root_id")
    return ann.reindex(ids)  # NaN rows for any id absent from the tsv


def _nonempty(series: pd.Series) -> pd.Series:
    return series.notna() & (series.str.strip() != "")


def _morton3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, bits: int = 21) -> np.ndarray:
    """3D Morton (Z-order) code, `bits` bits/axis, bit-interleave loop.

    Values are masked to `bits` bits (deterministic wraparound if a
    coordinate somehow exceeded 2**21 -- it does not in this dataset, whose
    observed soma range is ~2**18).
    """
    mask = np.uint64((1 << bits) - 1)
    xu = x.astype(np.uint64) & mask
    yu = y.astype(np.uint64) & mask
    zu = z.astype(np.uint64) & mask
    code = np.zeros(xu.shape, dtype=np.uint64)
    for b in range(bits):
        bb = np.uint64(b)
        bit = np.uint64(1) << bb
        code |= ((xu & bit) >> bb) << np.uint64(3 * b)
        code |= ((yu & bit) >> bb) << np.uint64(3 * b + 1)
        code |= ((zu & bit) >> bb) << np.uint64(3 * b + 2)
    return code


def _group_and_annotated(ann: pd.DataFrame, key: str):
    """Return (group_label: str Series, annotated: bool Series), both length N,
    in the SAME original-index order as `ann`."""
    if key == "cell_type":
        col = ann["cell_type"]
        annotated = _nonempty(col)
        group = col.where(annotated, other=_UNANNOTATED)

    elif key == "super_class+cell_class":
        sc = ann["super_class"].fillna("")
        cc = ann["cell_class"].fillna("")
        annotated = _nonempty(ann["super_class"]) | _nonempty(ann["cell_class"])
        group = (sc + "|" + cc)
        group = group.where(annotated, other=_UNANNOTATED)

    elif key == "hemibrain_type":
        col = ann["hemibrain_type"]
        annotated = _nonempty(col)
        group = col.where(annotated, other=_UNANNOTATED)

    elif key == "morton":
        x = pd.to_numeric(ann["soma_x"], errors="coerce")
        y = pd.to_numeric(ann["soma_y"], errors="coerce")
        z = pd.to_numeric(ann["soma_z"], errors="coerce")
        annotated = x.notna() & y.notna() & z.notna()
        m = annotated.to_numpy()
        codes = np.zeros(len(ann), dtype=np.uint64)
        if m.any():
            codes[m] = _morton3d(x.to_numpy()[m].astype(np.int64),
                                  y.to_numpy()[m].astype(np.int64),
                                  z.to_numpy()[m].astype(np.int64))
        # Sortable fixed-width decimal string so we can reuse the same
        # string-sort path as the other keys; unannotated rows get the
        # sentinel. uint64 codes here are < 2**63, so 20 digits never
        # truncates.
        group_str = np.where(m, np.char.zfill(codes.astype("U20"), 20), _UNANNOTATED)
        group = pd.Series(group_str, index=ann.index)

    else:
        raise ValueError(f"unknown key {key!r}; valid: {_VALID_KEYS}")

    return group, annotated


def build_permutation(data_dir="data", key: str = "cell_type") -> np.ndarray:
    """Deterministic permutation of the neuron index that groups neurons by
    `key`, so that a later grouping-locality metric (tile_liveness) can be
    computed against it.

    Returns
    -------
    perm : np.ndarray[int64], length N
        perm[new_index] = old_index. Original index = row order of
        data/2025_Completeness_783.csv. Cached to
        <data_dir>/reorder_cache/perm_<key>.npy.
    """
    if key not in _VALID_KEYS:
        raise ValueError(f"key must be one of {_VALID_KEYS}, got {key!r}")

    data_dir = Path(data_dir)
    cache_dir = data_dir / "reorder_cache"
    cache_path = cache_dir / f"perm_{key}.npy"
    if cache_path.exists():
        return np.load(cache_path)

    ids = _load_ids(data_dir)
    n = len(ids)

    if key == "none":
        perm = np.arange(n, dtype=np.int64)
    else:
        ann = _load_annotations(data_dir, ids)
        group, annotated = _group_and_annotated(ann, key)
        # df's RangeIndex 0..N-1 IS the original index (ann was reindexed to
        # `ids`, which is in original row order) -- so sort_values' returned
        # index is directly the perm we want.
        df = pd.DataFrame({
            "is_unannotated": (~annotated).to_numpy(),
            "group": group.to_numpy(),
        })
        order = df.sort_values(["is_unannotated", "group"], kind="mergesort").index
        perm = order.to_numpy(dtype=np.int64)

    assert perm.shape == (n,)
    assert np.array_equal(np.sort(perm), np.arange(n)), "perm must be a bijection"

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, perm)
    return perm


def tile_liveness(live_mask: np.ndarray, perm: np.ndarray, tile: int = 64) -> float:
    """Fraction of `tile`-wide contiguous blocks (in `perm`'s order) that
    contain at least one live neuron -- i.e. the fraction of tiles that
    CANNOT be skipped. 1 - this value is the fraction of tiles saved.

    The last (partial) tile, if N is not a multiple of `tile`, is padded with
    non-live entries -- it cannot make the result look better than reality,
    only very slightly worse in the (small) partial-tile case, which is the
    conservative direction.
    """
    live_mask = np.asarray(live_mask, dtype=bool)
    n = live_mask.shape[0]
    assert perm.shape[0] == n
    reordered = live_mask[perm]
    n_tiles = (n + tile - 1) // tile
    pad = n_tiles * tile - n
    if pad:
        reordered = np.concatenate([reordered, np.zeros(pad, dtype=bool)])
    tiles = reordered.reshape(n_tiles, tile)
    return float(tiles.any(axis=1).mean())


# ---------------------------------------------------------------------------
# Measurement driver (step 2/3/4 of the task). Not part of the importable
# API contract above, but kept in this file per the task instructions.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import time
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT / "flyloop"))
    import torch
    from brain_engine import BrainEngine  # noqa: E402

    DATA = str(ROOT / "data")
    STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500

    SUGAR = [720575940624963786, 720575940630233916, 720575940637568838,
             720575940638202345, 720575940617000768, 720575940630797113,
             720575940632889389, 720575940621754367, 720575940621502051,
             720575940640649691, 720575940639332736, 720575940616885538,
             720575940639198653, 720575940639259967, 720575940617937543,
             720575940632425919, 720575940633143833, 720575940612670570,
             720575940628853239, 720575940629176663, 720575940611875570]
    P9 = [720575940627652358, 720575940635872101]
    KEYS = ["none", "cell_type", "super_class+cell_class", "morton", "hemibrain_type"]
    TILES = [16, 32, 64]

    torch.set_num_threads(4)

    print(f"steps={STEPS}", file=sys.stderr)

    # --- permutations, computed ONCE, stimulus-independent ---
    perms = {}
    t0 = time.perf_counter()
    for k in KEYS:
        perms[k] = build_permutation(DATA, key=k)
        print(f"perm[{k}] built in {time.perf_counter()-t0:.2f}s (cumulative)",
              file=sys.stderr)

    # coverage report per key (for the md)
    ids = _load_ids(Path(DATA))
    ann = _load_annotations(Path(DATA), ids)
    coverage = {}
    for k in KEYS:
        if k == "none":
            coverage[k] = (len(ids), 0)
            continue
        _, annotated = _group_and_annotated(ann, k)
        coverage[k] = (int(annotated.sum()), int((~annotated).sum()))

    # --- pool for broad regimes: engine.i2flyid, built once ---
    probe = BrainEngine(data_dir=DATA, stim_ids=SUGAR[:1], seed=0)
    pool = probe.i2flyid
    rng = np.random.default_rng(0)
    del probe

    def regimes():
        yield "sugar GRNs (21 @200Hz)", SUGAR, 200.0
        yield "P9 walking (2 @100Hz)", P9, 100.0
        yield "single neuron (1 @200Hz)", SUGAR[:1], 200.0
        for n in (100, 1000, 10000):
            yield (f"broad ({n} @100Hz)",
                   rng.choice(pool, n, replace=False).tolist(), 100.0)

    results = []  # list of dict
    for name, stim_ids, rate in regimes():
        eng = BrainEngine(data_dir=DATA, stim_ids=stim_ids, seed=1234)
        eng.active_mode = False
        eng.inplace = True
        eng.inject(rate)
        t0 = time.perf_counter()
        for _ in range(STEPS):
            eng.step_inplace()
        dt_s = time.perf_counter() - t0

        v = eng.v.numpy()
        g = eng.g.numpy()
        buf_max = eng.buf.abs().amax(0).numpy()
        live_mask = (v != -52.0) | (g != 0) | (buf_max > 0)
        live_frac = float(live_mask.mean())
        n_live = int(live_mask.sum())

        oracle_perm = np.argsort(~live_mask, kind="stable")

        row = {"regime": name, "n_stim": len(stim_ids), "rate_hz": rate,
               "live_frac": live_frac, "n_live": n_live,
               "wall_s": dt_s, "tiles": {}}
        for k in KEYS:
            row["tiles"][k] = {t: tile_liveness(live_mask, perms[k], t) for t in TILES}
        row["tiles"]["oracle"] = {t: tile_liveness(live_mask, oracle_perm, t) for t in TILES}
        results.append(row)
        print(f"[{name}] live={live_frac*100:.3f}% ({n_live}) "
              f"tile64 none={row['tiles']['none'][64]*100:.2f}% "
              f"cell_type={row['tiles']['cell_type'][64]*100:.2f}% "
              f"oracle={row['tiles']['oracle'][64]*100:.2f}% "
              f"[{dt_s:.1f}s]", file=sys.stderr)

    # --- write the markdown report ---
    out_path = ROOT / "research" / "reorder_measurements.md"
    lines = []
    lines.append("# Neuron reordering for tile-skipping — generalisation across regimes\n")
    lines.append(f"Generated by `flyloop/reorder.py` ({STEPS} steps/regime, "
                  "seed=1234, torch threads=4).\n")
    lines.append("Gating question: does cell_type-based clustering generalise "
                  "beyond the sugar protocol, and where does tile-skipping stop "
                  "paying for itself as stimulation broadens?\n")

    lines.append("## Annotation coverage per key (N = 138,639)\n")
    lines.append("| key | annotated | unannotated | unannotated % |")
    lines.append("|---|---:|---:|---:|")
    for k in KEYS:
        a, u = coverage[k]
        lines.append(f"| {k} | {a:,} | {u:,} | {100*u/len(ids):.2f}% |")
    lines.append("")
    lines.append("Unannotated neurons form a single deterministic trailing group per "
                  "key (see `flyloop/reorder.py` module docstring for the exact rule); "
                  "`hemibrain_type` in particular is unannotated for the majority of "
                  "the brain, so it is included mainly as a lower-bound control.\n")

    lines.append("## Live fraction per regime (after 1500 steps)\n")
    lines.append("| regime | stim n | rate Hz | live % | live neurons |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in results:
        lines.append(f"| {r['regime']} | {r['n_stim']} | {r['rate_hz']:.0f} | "
                      f"{r['live_frac']*100:.3f}% | {r['n_live']:,} |")
    lines.append("")

    for tile in TILES:
        lines.append(f"## Tile-{tile} live fraction by ordering key "
                      "(lower = more work saved)\n")
        header = "| regime | " + " | ".join(KEYS + ["oracle"]) + " |"
        sep = "|---|" + "---:|" * (len(KEYS) + 1)
        lines.append(header)
        lines.append(sep)
        for r in results:
            cells = [f"{r['tiles'][k][tile]*100:.2f}%" for k in KEYS + ["oracle"]]
            lines.append(f"| {r['regime']} | " + " | ".join(cells) + " |")
        lines.append("")

    # --- crossover analysis: live_frac vs best-key saved% at tile=64 ---
    lines.append("## Crossover: where does tile-skipping stop paying for itself?\n")
    lines.append("Best non-oracle key's saved fraction (1 - live-tile-fraction) at "
                  "tile=64, vs the regime's overall live fraction:\n")
    lines.append("| regime | live % | best key | best saved % | oracle saved % |")
    lines.append("|---|---:|---|---:|---:|")
    crossover_note = None
    for r in results:
        best_key, best_val = min(
            ((k, r["tiles"][k][64]) for k in KEYS if k != "none"),
            key=lambda kv: kv[1])
        best_saved = 100 * (1 - best_val)
        oracle_saved = 100 * (1 - r["tiles"]["oracle"][64])
        lines.append(f"| {r['regime']} | {r['live_frac']*100:.2f}% | {best_key} | "
                      f"{best_saved:.2f}% | {oracle_saved:.2f}% |")
        if best_saved < 10.0 and crossover_note is None:
            crossover_note = (r['regime'], r['live_frac'], best_saved)
    if crossover_note:
        lines.append(f"\nBest-key saved% first drops below 10% at **{crossover_note[0]}** "
                      f"(live fraction {crossover_note[1]*100:.2f}%).\n")
    else:
        lines.append("\nBest-key saved% never drops below 10% in the regimes tested.\n")

    lines.append("## Verdict\n")
    lines.append("(fill in after inspecting the numbers above — written by the "
                  "generating run below)\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}", file=sys.stderr)
