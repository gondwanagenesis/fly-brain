"""Change 2: skip provably-inert 16-neuron tiles.

91% of neurons sit at EXACT rest under localised stimulus and are provably
unchanged by the update, but in the shipped neuron order they are scattered
(mean live-run length 1.1), so tile-skipping saved 1.15%. Reordering by
`cell_type` clusters them: at tile=16 it leaves 28.3% of tiles live on sugar,
12.7% on P9, 0.5% on a single neuron -- 71.7% / 87.3% / 99.5% of the sweep
skippable. See research/reorder_measurements.md.

TILES ARE CHUNK-RELATIVE, not global. The ATen chunk boundaries we must mirror
(ceil(N/nthreads) = 34,660) are not multiples of 16, so a globally-aligned tile
would straddle the FMA/non-FMA rounding seam at a chunk edge. Defining tiles
from each chunk's start makes the FMA region exactly (vec_end-lo)/16 whole
groups with the non-FMA remainder as one final partial tile, so every tile has a
single, uniform rounding mode and no tile ever crosses a seam.

FOUR BUGS AN ADVERSARIAL REVIEW CAUGHT BEFORE THIS SHIPPED:

 B1 Stimulated neurons receive drive OUTSIDE the delay ring (`v[stim] += ...`
    runs before the sweep). A stim neuron whose Poisson draw is 0 sits at
    v == v_rest and g == 0 exactly, satisfying the inert test -- so its tile
    would be cleared, the sweep would stop running, and `v` would accumulate
    forever with no threshold test. The entire sensory drive silently
    disconnects. FIX: tiles containing stim neurons are PINNED live and the
    rescan may never clear them. (The PyTorch active-set path already carries
    this fix at brain_engine.py:274.)

 B1b At t=0 every neuron is at exact rest, so initialising `tile_live` by
    scanning marks everything dead and the simulation never starts. FIX: all
    tiles start LIVE; the first rescan clears the genuinely inert ones.

 B3 A refractory neuron also sits at v == v_rest, g == 0 exactly -- it is
    bit-indistinguishable from a resting one. FIX: the inert test requires the
    gate to be OPEN, so refractory neurons keep their tile live (~40 neurons,
    <=40 tiles). The refractory countdown is walked outside the sweep anyway
    (landed with Change 1), so this is belt-and-braces.

 B4 Marking a tile live when the fan-out EMITS an arrival is not enough: a
    rescan can clear it again before the arrival lands up to 19 steps later.
    FIX: mark live at CONSUMPTION, in the delayed pass, at the exact moment g is
    mutated. Between emission and consumption the neuron is genuinely unchanged,
    so a dead tile is correct there.

Skipping is exact. For a neuron with v == v_rest and g == 0: the reference
computes t = (v_rest - v_rest) + 0 = 0, v = fma(0, c_mem, v_rest) = v_rest,
g = 0 * c_decay = 0, and v_rest > v_th is false, so v, g and the spike bit are
all unchanged -- which is exactly what skipping produces (the spike bitset is
memset to 0 each step, so a skipped tile contributes zeros).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
C = ROOT / "flyloop" / "native" / "lif_kernel.c"
PY = ROOT / "flyloop" / "native_engine.py"


def sub(text, old, new, what):
    if old not in text:
        sys.exit(f"ANCHOR NOT FOUND: {what}")
    return text.replace(old, new, 1)


s = C.read_text(encoding="utf-8")

# ---- one 16-wide AVX-512 group, factored out so the tile loop can call it ----
s = sub(s, "/* Runtime ISA selection, resolved once. */", """#ifdef LIF_X86
/* One 16-neuron group. Same arithmetic as sweep_avx512's loop body. */
__attribute__((target("avx512f,avx512bw,avx512dq")))
static void group16(int i, float *restrict v, float *restrict g,
                    uint64_t *restrict sp_out,
                    float c_decay, float c_mem,
                    float v_rest, float v_reset, float v_th, int wf, int wl)
{
    const __m512 gi = _mm512_loadu_ps(g + i);
    const __m512 vv = _mm512_loadu_ps(v + i);
    const __m512 gn = _mm512_mul_ps(gi, _mm512_set1_ps(c_decay));
    const __m512 t  = _mm512_add_ps(_mm512_sub_ps(_mm512_set1_ps(v_rest), vv), gi);
    const __m512 vn = _mm512_fmadd_ps(t, _mm512_set1_ps(c_mem), vv);
    const __mmask16 sp = _mm512_cmp_ps_mask(vn, _mm512_set1_ps(v_th), _CMP_GT_OQ);
    _mm512_storeu_ps(v + i, _mm512_mask_blend_ps(sp, vn, _mm512_set1_ps(v_reset)));
    _mm512_storeu_ps(g + i, _mm512_mask_blend_ps(sp, gn, _mm512_setzero_ps()));
    put_bits(sp_out, i, (uint32_t)sp, 16, wf, wl);
}
#endif

/* Runtime ISA selection, resolved once. */""", "group16")

# ---- tile sweep replaces the vec_end split inside a chunk ----
s = sub(s, """static void sweep_chunk(int lo, int hi, int tail_w,""",
"""/* Sweep the live tiles of one chunk.
 *
 * Every tile carries a single uniform rounding mode (see the module note on
 * chunk-relative tiling), so the FMA/non-FMA seam is a property of the tile
 * rather than something the loop has to re-derive. */
static void sweep_tile_range(int t0, int t1,
                             const int32_t *tile_lo, const uint8_t *tile_fma,
                             const uint64_t *tile_live,
                             float *restrict v, float *restrict g,
                             uint64_t *restrict sp_out,
                             float c_decay, float c_mem,
                             float v_rest, float v_reset, float v_th)
{
    if (t1 <= t0) return;
    const int wf = tile_lo[t0] >> 6, wl = (tile_lo[t1] - 1) >> 6;
    for (int t = t0; t < t1; ++t) {
        if (!((tile_live[t >> 6] >> (t & 63)) & 1ULL)) continue;   /* provably inert */
        const int lo = tile_lo[t], hi = tile_lo[t + 1];
#ifdef LIF_X86
        if (g_isa == ISA_AVX512 && tile_fma[t] && hi - lo == 16) {
            group16(lo, v, g, sp_out, c_decay, c_mem, v_rest, v_reset, v_th, wf, wl);
            continue;
        }
#endif
        sweep_scalar(lo, hi, tile_fma[t], v, g, sp_out,
                     c_decay, c_mem, v_rest, v_reset, v_th, wf, wl);
    }
}

/* Clear tiles that are provably inert. Runs every RESCAN steps; between scans a
 * live tile stays live, which is conservative and never skips a live neuron. */
static void rescan_tiles(int nt, const int32_t *tile_lo, uint64_t *tile_live,
                         const uint64_t *tile_pin,
                         const float *v, const float *g,
                         const uint64_t *gate_bits, const uint64_t *sp_bits,
                         float v_rest)
{
    for (int t = 0; t < nt; ++t) {
        const uint64_t m = 1ULL << (t & 63);
        if (!(tile_live[t >> 6] & m)) continue;
        if (tile_pin[t >> 6] & m) continue;          /* B1: stim tiles are pinned */
        int inert = 1;
        for (int i = tile_lo[t]; i < tile_lo[t + 1]; ++i) {
            /* gate must be OPEN: a refractory neuron is bit-indistinguishable
             * from a resting one in (v, g), and must not be skipped (B3). */
            if (v[i] != v_rest || g[i] != 0.0f
                || !BIT_GET(gate_bits, i) || BIT_GET(sp_bits, i)) {
                inert = 0;
                break;
            }
        }
        if (inert) tile_live[t >> 6] &= ~m;
    }
}

static void sweep_chunk(int lo, int hi, int tail_w,""", "tile sweep fns")

# ---- job carries a tile range now ----
s = sub(s, """    int lo, hi, tail_w;
    float *v, *g;
    uint64_t *sp_out;""",
"""    int lo, hi, tail_w;
    int t0, t1;
    const int32_t *tile_lo;
    const uint8_t *tile_fma;
    const uint64_t *tile_live;
    float *v, *g;
    uint64_t *sp_out;""", "job_t tiles")

s = sub(s, """    sweep_chunk(j->lo, j->hi, j->tail_w, j->v, j->g, j->sp_out,
                j->c_decay, j->c_mem, j->v_rest, j->v_reset, j->v_th);""",
"""    sweep_tile_range(j->t0, j->t1, j->tile_lo, j->tile_fma, j->tile_live,
                     j->v, j->g, j->sp_out, j->c_decay, j->c_mem,
                     j->v_rest, j->v_reset, j->v_th);""", "run_job tiles")

s = sub(s, """static void sweep_all(const int32_t *chunks, int n_chunks, int tail_w,
                      float *v, float *g, uint64_t *sp_out,
                      float c_decay, float c_mem,
                      float v_rest, float v_reset, float v_th)
{
    if (!POOL.started || POOL.nthreads < 2 || n_chunks != POOL.nthreads) {
        for (int c = 0; c < n_chunks; ++c)
            sweep_chunk(chunks[c], chunks[c + 1], tail_w, v, g, sp_out,
                        c_decay, c_mem, v_rest, v_reset, v_th);
        return;
    }
    for (int c = 0; c < n_chunks; ++c) {
        job_t *j = &POOL.jobs[c];
        j->kind = 0;
        j->lo = chunks[c]; j->hi = chunks[c + 1]; j->tail_w = tail_w;
        j->v = v; j->g = g; j->sp_out = sp_out;""",
"""static void sweep_all(const int32_t *chunk_tile, int n_chunks,
                      const int32_t *tile_lo, const uint8_t *tile_fma,
                      const uint64_t *tile_live,
                      float *v, float *g, uint64_t *sp_out,
                      float c_decay, float c_mem,
                      float v_rest, float v_reset, float v_th)
{
    if (!POOL.started || POOL.nthreads < 2 || n_chunks != POOL.nthreads) {
        for (int c = 0; c < n_chunks; ++c)
            sweep_tile_range(chunk_tile[c], chunk_tile[c + 1], tile_lo, tile_fma,
                             tile_live, v, g, sp_out, c_decay, c_mem,
                             v_rest, v_reset, v_th);
        return;
    }
    for (int c = 0; c < n_chunks; ++c) {
        job_t *j = &POOL.jobs[c];
        j->kind = 0;
        j->t0 = chunk_tile[c]; j->t1 = chunk_tile[c + 1];
        j->tile_lo = tile_lo; j->tile_fma = tile_fma; j->tile_live = tile_live;
        j->v = v; j->g = g; j->sp_out = sp_out;""", "sweep_all tiles")

# ---- lif_step: new params, tile-aware sweep, mark-live on consumption, rescan
s = sub(s, """    const int32_t *chunks, int n_chunks, int tail_w,
    int32_t *out_spike_idx)""",
"""    const int32_t *chunks, int n_chunks, int tail_w,
    const int32_t *tile_lo, const uint8_t *tile_fma, uint64_t *tile_live,
    const uint64_t *tile_pin, const int32_t *chunk_tile,
    const int32_t *neuron_tile, int n_tiles, int rescan,
    int32_t *out_spike_idx)""", "lif_step tile params")

s = sub(s, """    sweep_all(chunks, n_chunks, tail_w, v, g, sp_scratch,
              c_decay, c_mem, v_rest, v_reset, v_th);""",
"""    (void)chunks; (void)tail_w;
    sweep_all(chunk_tile, n_chunks, tile_lo, tile_fma, tile_live,
              v, g, sp_scratch, c_decay, c_mem, v_rest, v_reset, v_th);""",
        "lif_step sweep call")

s = sub(s, """        if (!BIT_GET(sp_scratch, i)) {
            const float gate = BIT_GET(gate_bits, i) ? 1.0f : 0.0f;
            g[i] = g[i] + del_val[k] * gate;
        }""",
"""        if (!BIT_GET(sp_scratch, i)) {
            const float gate = BIT_GET(gate_bits, i) ? 1.0f : 0.0f;
            g[i] = g[i] + del_val[k] * gate;
        }
        /* B4: mark live at CONSUMPTION, the moment g can change -- not at
         * emission, because a rescan could clear the tile during the up-to-19
         * steps the arrival spends in flight. Marked unconditionally: a spiking
         * neuron's tile is live anyway, and this keeps the invariant simple. */
        {
            const int tt = neuron_tile[i];
            tile_live[tt >> 6] |= 1ULL << (tt & 63);
        }""", "mark live on consumption")

s = sub(s, """    memcpy(sp_bits, sp_scratch, (size_t)(nw + 1) * sizeof(uint64_t));
    return nsp;""",
"""    if (rescan)
        rescan_tiles(n_tiles, tile_lo, tile_live, tile_pin, v, g,
                     gate_bits, sp_scratch, v_rest);

    memcpy(sp_bits, sp_scratch, (size_t)(nw + 1) * sizeof(uint64_t));
    return nsp;""", "rescan call")

C.write_text(s, encoding="utf-8")
print("patched lif_kernel.c")

# ------------------------------------------------------------------ Python ---
t = PY.read_text(encoding="utf-8")

t = sub(t, """        c_i32p, ctypes.c_int, ctypes.c_int,              # chunk bounds, tail width
        c_i32p,                                          # out_spike_idx""",
"""        c_i32p, ctypes.c_int, ctypes.c_int,              # chunk bounds, tail width
        c_i32p, ctypes.POINTER(ctypes.c_uint8), c_u64p,  # tile_lo, tile_fma, live
        c_u64p, c_i32p,                                  # tile_pin, chunk_tile
        c_i32p, ctypes.c_int, ctypes.c_int,              # neuron_tile, n_tiles, rescan
        c_i32p,                                          # out_spike_idx""",
        "argtypes tiles")

t = sub(t, """        self.tail_w = aten_vector_width()""",
"""        self.tail_w = aten_vector_width()

        # ---- 16-neuron tiles, defined RELATIVE TO EACH CHUNK ----
        #
        # The ATen chunk boundaries are not multiples of 16, so a globally
        # aligned tile would straddle the FMA/non-FMA seam at a chunk edge.
        # Tiling from each chunk's start makes the FMA region exactly
        # (vec_end - lo)/16 whole groups, with the non-FMA remainder as one
        # final partial tile -- so every tile has one uniform rounding mode.
        TILE = 16
        t_lo, t_fma, c_tile = [], [], [0]
        for c in range(self.n_chunks):
            lo, hi = int(bounds[c]), int(bounds[c + 1])
            vec_end = lo + ((hi - lo) // self.tail_w) * self.tail_w
            i = lo
            while i < vec_end:
                t_lo.append(i)
                t_fma.append(1)
                i = min(i + TILE, vec_end)
            if vec_end < hi:                      # ATen's scalar tail: no FMA
                t_lo.append(vec_end)
                t_fma.append(0)
            c_tile.append(len(t_lo))
        t_lo.append(N)
        self.tile_lo = np.asarray(t_lo, dtype=np.int32)
        self.tile_fma = np.asarray(t_fma, dtype=np.uint8)
        self.chunk_tile = np.asarray(c_tile, dtype=np.int32)
        self.n_tiles = nt = len(t_fma)

        # neuron -> tile, used only by the sparse delayed pass (~190/step)
        self.neuron_tile = np.zeros(N, dtype=np.int32)
        for k in range(nt):
            self.neuron_tile[self.tile_lo[k]:self.tile_lo[k + 1]] = k

        _ntw = (nt + 63) >> 6
        # B1b: every tile starts LIVE. Initialising by scan would mark the whole
        # brain dead at t=0 (all neurons are at exact rest) and nothing would
        # ever run.
        self.tile_live = np.full(_ntw + 1, np.uint64(0xFFFFFFFFFFFFFFFF),
                                 dtype=np.uint64)
        self.tile_pin = np.zeros(_ntw + 1, dtype=np.uint64)
        self.rescan_every = 64
        self._since_rescan = 0""", "tile construction")

t = sub(t, """        self._stim_val = np.zeros(len(idx), dtype=np.float32)
        self._rates = torch.zeros(len(idx))""",
"""        self._stim_val = np.zeros(len(idx), dtype=np.float32)
        self._rates = torch.zeros(len(idx))
        # B1: stimulated neurons are driven OUTSIDE the delay ring, so nothing
        # in the delayed pass can mark their tile live. A stim neuron whose
        # Poisson draw is 0 sits at exact rest and would be declared inert, its
        # tile cleared, and the sensory drive silently disconnected. Pin them.
        if hasattr(self, "tile_pin"):
            for i in self.stim_idx:
                tt = int(self.neuron_tile[int(i)])
                self.tile_pin[tt >> 6] |= np.uint64(1) << np.uint64(tt & 63)
                self.tile_live[tt >> 6] |= np.uint64(1) << np.uint64(tt & 63)""",
        "pin stim tiles")

t = sub(t, """        self._pchunk = _p(self.chunks, i32)""",
"""        self._pchunk = _p(self.chunks, i32)
        self._ptlo = _p(self.tile_lo, i32)
        self._ptfma = _p(self.tile_fma, ctypes.c_uint8)
        self._ptlive = _p(self.tile_live, u64)
        self._ptpin = _p(self.tile_pin, u64)
        self._pctile = _p(self.chunk_tile, i32)
        self._pntile = _p(self.neuron_tile, i32)""", "tile pointers")

t = sub(t, """            self._pchunk, self.n_chunks, self.tail_w,
            self._psp[1 - self._cur],""",
"""            self._pchunk, self.n_chunks, self.tail_w,
            self._ptlo, self._ptfma, self._ptlive,
            self._ptpin, self._pctile,
            self._pntile, self.n_tiles, rescan,
            self._psp[1 - self._cur],""", "call site tiles")

t = sub(t, """        h = self.head
        prev_n = self._cur_n
        cf = self._cf""",
"""        self._since_rescan += 1
        rescan = 1 if self._since_rescan >= self.rescan_every else 0
        if rescan:
            self._since_rescan = 0

        h = self.head
        prev_n = self._cur_n
        cf = self._cf""", "rescan cadence")

PY.write_text(t, encoding="utf-8")
print("patched native_engine.py")
