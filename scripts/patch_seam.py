"""Make ATen's rounding seam follow the NEURON, not the array position.

ATen vectorises `add_(t, alpha=)` with an FMA but finishes each at::parallel_for
chunk with a scalar tail that is not fused, so the reference integrates the last
(chunk_len mod 16) neurons of every chunk with a different rounding. With
N = 138,639 and 4 threads those are original indices 34,656-34,659,
69,316-69,319, 103,976-103,979 and 138,624-138,638 -- 31 neurons.

Before reordering, array position == neuron, so reproducing the seam by position
was the same thing as reproducing it per neuron. Once neurons are permuted the
two come apart: position 34,656 now holds some other neuron, and applying the
non-FMA rounding there diverges from the reference (observed at step 394).

So the seam is now computed per neuron in the ORIGINAL index space and carried
through the permutation. A tile whose neurons all agree keeps a fast uniform
path; the handful that straddle the seam (at most 31 neurons, so at most 31
tiles of 8,668) fall back to a scalar loop that consults a per-neuron bit.

This keeps the kernel bit-identical to the reference for EVERY neuron, including
the reference's own vector/tail artefacts, rather than merely for most of them.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
C = ROOT / "flyloop" / "native" / "lif_kernel.c"
PY = ROOT / "flyloop" / "native_engine.py"

MIXED = 2


def sub(text, old, new, what):
    if old not in text:
        sys.exit(f"ANCHOR NOT FOUND: {what}")
    return text.replace(old, new, 1)


s = C.read_text(encoding="utf-8")

# --- scalar sweep gains a per-neuron mode ---
s = sub(s, """static void sweep_scalar(int lo, int hi, int use_fma,
                         float *restrict v, float *restrict g,
                         uint64_t *restrict sp_out,
                         float c_decay, float c_mem,
                         float v_rest, float v_reset, float v_th,
                         int wf, int wl)
{
    if (use_fma) { LIF_BODY(1) } else { LIF_BODY(0) }
}""",
"""static void sweep_scalar(int lo, int hi, int use_fma,
                         float *restrict v, float *restrict g,
                         uint64_t *restrict sp_out,
                         float c_decay, float c_mem,
                         float v_rest, float v_reset, float v_th,
                         int wf, int wl, const uint64_t *fma_bits)
{
    /* use_fma == 2 means the tile straddles ATen's vector/tail seam, so the
     * rounding differs per neuron and has to be looked up. At most 31 neurons
     * in the whole brain are in that state. */
    if (use_fma == 2) { LIF_BODY(BIT_GET(fma_bits, i)) }
    else if (use_fma) { LIF_BODY(1) }
    else              { LIF_BODY(0) }
}""", "sweep_scalar mixed")

# every sweep_scalar call site passes the bitset through
s = s.replace("""        sweep_scalar(lo, hi, tile_fma[t], v, g, sp_out,
                     c_decay, c_mem, v_rest, v_reset, v_th, wf, wl);""",
              """        sweep_scalar(lo, hi, tile_fma[t], v, g, sp_out,
                     c_decay, c_mem, v_rest, v_reset, v_th, wf, wl, fma_bits);""")
s = s.replace("""        sweep_scalar(i, vec_end, 1, v, g, sp_out,
                     c_decay, c_mem, v_rest, v_reset, v_th, wf, wl);""",
              """        sweep_scalar(i, vec_end, 1, v, g, sp_out,
                     c_decay, c_mem, v_rest, v_reset, v_th, wf, wl, NULL);""")
s = s.replace("""        sweep_scalar(vec_end, hi, 0, v, g, sp_out,
                     c_decay, c_mem, v_rest, v_reset, v_th, wf, wl);""",
              """        sweep_scalar(vec_end, hi, 0, v, g, sp_out,
                     c_decay, c_mem, v_rest, v_reset, v_th, wf, wl, NULL);""")
s = s.replace("""    if (i < hi) sweep_scalar(i, hi, use_fma, v, g, sp_out,
                             c_decay, c_mem, v_rest, v_reset, v_th, wf, wl);""",
              """    if (i < hi) sweep_scalar(i, hi, use_fma, v, g, sp_out,
                             c_decay, c_mem, v_rest, v_reset, v_th, wf, wl, NULL);""")

# --- thread the bitset through the tile sweep ---
s = sub(s, """static void sweep_tile_range(int t0, int t1,
                             const int32_t *tile_lo, const uint8_t *tile_fma,
                             const uint64_t *tile_live,""",
"""static void sweep_tile_range(int t0, int t1,
                             const int32_t *tile_lo, const uint8_t *tile_fma,
                             const uint64_t *tile_live, const uint64_t *fma_bits,""",
        "sweep_tile_range sig")

s = sub(s, """    const uint8_t *tile_fma;
    const uint64_t *tile_live;""",
"""    const uint8_t *tile_fma;
    const uint64_t *tile_live;
    const uint64_t *fma_bits;""", "job_t fma_bits")

s = sub(s, """    sweep_tile_range(j->t0, j->t1, j->tile_lo, j->tile_fma, j->tile_live,
                     j->v, j->g, j->sp_out, j->c_decay, j->c_mem,
                     j->v_rest, j->v_reset, j->v_th);""",
"""    sweep_tile_range(j->t0, j->t1, j->tile_lo, j->tile_fma, j->tile_live,
                     j->fma_bits, j->v, j->g, j->sp_out, j->c_decay, j->c_mem,
                     j->v_rest, j->v_reset, j->v_th);""", "run_job fma_bits")

s = sub(s, """static void sweep_all(const int32_t *chunk_tile, int n_chunks,
                      const int32_t *tile_lo, const uint8_t *tile_fma,
                      const uint64_t *tile_live,""",
"""static void sweep_all(const int32_t *chunk_tile, int n_chunks,
                      const int32_t *tile_lo, const uint8_t *tile_fma,
                      const uint64_t *tile_live, const uint64_t *fma_bits,""",
        "sweep_all sig")
s = sub(s, """            sweep_tile_range(chunk_tile[c], chunk_tile[c + 1], tile_lo, tile_fma,
                             tile_live, v, g, sp_out, c_decay, c_mem,
                             v_rest, v_reset, v_th);""",
"""            sweep_tile_range(chunk_tile[c], chunk_tile[c + 1], tile_lo, tile_fma,
                             tile_live, fma_bits, v, g, sp_out, c_decay, c_mem,
                             v_rest, v_reset, v_th);""", "sweep_all serial")
s = sub(s, """        j->tile_lo = tile_lo; j->tile_fma = tile_fma; j->tile_live = tile_live;""",
"""        j->tile_lo = tile_lo; j->tile_fma = tile_fma; j->tile_live = tile_live;
        j->fma_bits = fma_bits;""", "sweep_all jobs")

s = sub(s, """    const int32_t *tile_lo, const uint8_t *tile_fma, uint64_t *tile_live,""",
        """    const int32_t *tile_lo, const uint8_t *tile_fma, uint64_t *tile_live,
    const uint64_t *fma_bits,""", "lif_step sig")
s = sub(s, """    sweep_all(chunk_tile, n_chunks, tile_lo, tile_fma, tile_live,
              v, g, sp_scratch, c_decay, c_mem, v_rest, v_reset, v_th);""",
"""    sweep_all(chunk_tile, n_chunks, tile_lo, tile_fma, tile_live, fma_bits,
              v, g, sp_scratch, c_decay, c_mem, v_rest, v_reset, v_th);""",
        "lif_step sweep call")

C.write_text(s, encoding="utf-8")
print("patched lif_kernel.c")

# ------------------------------------------------------------------ Python ---
t = PY.read_text(encoding="utf-8")

t = sub(t, """        c_i32p, ctypes.POINTER(ctypes.c_uint8), c_u64p,  # tile_lo, tile_fma, live""",
"""        c_i32p, ctypes.POINTER(ctypes.c_uint8), c_u64p,  # tile_lo, tile_fma, live
        c_u64p,                                          # fma_bits (per neuron)""",
        "argtypes fma_bits")

t = sub(t, """            self._ptlo, self._ptfma, self._ptlive,""",
        """            self._ptlo, self._ptfma, self._ptlive, self._pfmab,""",
        "call site fma_bits")

t = sub(t, """        self._ptlive = _p(self.tile_live, u64)""",
        """        self._ptlive = _p(self.tile_live, u64)
        self._pfmab = _p(self.fma_bits, u64)""", "fma_bits pointer")

# Build the per-neuron seam map, in ORIGINAL index space, then permute it.
t = sub(t, """            if vec_end < hi:                      # ATen's scalar tail: no FMA
                t_lo.append(vec_end)
                t_fma.append(0)
            c_tile.append(len(t_lo))
        t_lo.append(N)
        self.tile_lo = np.asarray(t_lo, dtype=np.int32)
        self.tile_fma = np.asarray(t_fma, dtype=np.uint8)""",
"""            if vec_end < hi:                      # ATen's scalar tail: no FMA
                t_lo.append(vec_end)
                t_fma.append(0)
            c_tile.append(len(t_lo))
        t_lo.append(N)
        self.tile_lo = np.asarray(t_lo, dtype=np.int32)
        self.tile_fma = np.asarray(t_fma, dtype=np.uint8)

        # --- ATen's rounding seam follows the NEURON, not the array slot ---
        #
        # The reference applies its non-fused scalar tail to the last
        # (chunk_len mod tail_w) neurons of each of ITS chunks, in the ORIGINAL
        # index space. Unpermuted, position == neuron and the tile-level flags
        # above are exact. Once neurons are permuted the two come apart, so the
        # flag is recomputed per neuron and carried through the permutation;
        # tiles that end up straddling the seam are marked MIXED (2) and take a
        # scalar path that looks the bit up. At most 31 neurons brain-wide.
        fma_orig = np.zeros(N, dtype=bool)
        for c in range(self.n_chunks):
            lo, hi = int(bounds[c]), int(bounds[c + 1])
            fma_orig[lo:lo + ((hi - lo) // self.tail_w) * self.tail_w] = True
        fma_new = fma_orig[self.perm] if self.perm is not None else fma_orig
        self.fma_bits = np.zeros(((N + 63) >> 6) + 1, dtype=np.uint64)
        _w = np.nonzero(fma_new)[0]
        np.bitwise_or.at(self.fma_bits, _w >> 6,
                         np.uint64(1) << (_w & 63).astype(np.uint64))
        for k in range(len(self.tile_fma)):
            seg = fma_new[self.tile_lo[k]:self.tile_lo[k + 1]]
            self.tile_fma[k] = 1 if seg.all() else (0 if not seg.any() else 2)""",
        "seam map")

PY.write_text(t, encoding="utf-8")
print("patched native_engine.py")
