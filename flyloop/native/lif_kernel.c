/* Fused single-pass LIF kernel for the Shiu et al. whole-brain Drosophila model.
 *
 * WHY THIS EXISTS
 * ---------------
 * The PyTorch step performs ~12 separate full-array passes (refrac add, compare,
 * mul, add, sub, neg, add, fma, gt, masked_fill x2, copy). Each pass reads and
 * writes N floats, so the step moves ~12x more memory than the algorithm needs,
 * and the 138,639-neuron state (2.2 MB) is streamed through cache twelve times
 * instead of once. Fusing the whole update into a single pass over neurons is
 * the single largest available win on a CPU, and it is exactly-representable:
 * no arithmetic changes, only the order in which memory is touched.
 *
 * BIT-IDENTITY
 * ------------
 * Every floating-point operation below reproduces the corresponding PyTorch op
 * in the same order with the same rounding. Two facts were established by
 * probing torch 2.13 on this machine and MUST be preserved:
 *
 *   1. `v.add_(t, alpha=a)` is fmaf(t, (float)a, v)  -- a SINGLE rounding.
 *      Verified: separate mul-then-add differs on 37/200000 elements.
 *   2. `g.mul_(c).add_(t)` is fl(fl(g*c) + t)        -- TWO roundings, no FMA.
 *      Verified: bit-identical to the two-op form on 200000 elements.
 *
 * The file must therefore be compiled with -ffp-contract=off so the compiler
 * cannot fuse (2) into an FMA, while the explicit fmaf() in (1) still emits
 * vfmadd. Compiling with -ffast-math would silently break bit-identity.
 *
 * SPARSE DELAY LINE
 * -----------------
 * Upstream stores the 1.8 ms axonal delay as a dense (19, N) fp32 ring buffer:
 * 10.5 MB, which alone exceeds the L2 and evicts the neuron state from cache
 * every 19 steps. But the buffer's contents are the recurrent input, which has
 * only ~190 non-zeros per step out of 138,639 (measured 1.75 spikes/step x
 * ~110 mean fan-out). Storing each slot as an (index, value) list instead makes
 * the delay line ~30 KB resident, and the whole working set L2/L3-resident.
 * This is exact, not an approximation: the omitted entries are exactly zero,
 * and fl(x + 0.0f) == x for every x that can occur here (g is never -0.0
 * because it only ever arrives via a multiply by a positive constant, an
 * addition, or an explicit +0.0 store).
 */

#include <stdint.h>
#include <string.h>
#include <math.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#define HAVE_AVX512 1
#endif

#if defined(_MSC_VER)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#define BIT_GET(bits, i) (((bits)[(i) >> 6] >> ((i) & 63)) & 1ULL)

EXPORT int lif_has_avx512(void)
{
#ifdef HAVE_AVX512
    return 1;
#else
    return 0;
#endif
}

/* ------------------------------------------------------------------------ */
/* The fused dense sweep.                                                    */
/*                                                                           */
/* Reproduces, per neuron, exactly this PyTorch sequence from                */
/* brain_engine.step_inplace:                                                */
/*                                                                           */
/*   refrac = spiked_prev ? 0 : refrac + 1                                   */
/*   gate   = (refrac >= refrac_steps)                                       */
/*   gnew   = g * c_decay                      (+ delayed*gate, applied      */
/*                                              sparsely by the caller)      */
/*   t      = -(v - v_rest) + g                                              */
/*   v      = fma(t, c_mem, v)                                               */
/*   sp     = v > v_th                                                       */
/*   v      = sp ? v_reset : v                                               */
/*   g      = sp ? 0 : gnew                                                  */
/*                                                                           */
/* Branchless so the body auto-vectorises; the spike indices are recovered   */
/* afterwards from the output bitset, which costs one pass over 17 KB rather  */
/* than a data-dependent branch per neuron.                                  */
/* ------------------------------------------------------------------------ */
/* Read/write 16 spike bits at an arbitrary neuron offset. The chunk boundaries
 * we must reproduce are not 64-aligned, so bit groups straddle words; both
 * buffers carry one word of padding so the w+1 access is always in range. */
static inline uint32_t get16(const uint64_t *b, int i)
{
    const int w = i >> 6, s = i & 63;
    uint64_t x = b[w] >> s;
    if (s > 48) x |= b[w + 1] << (64 - s);
    return (uint32_t)(x & 0xFFFFu);
}

/* OR bits into the (pre-zeroed) output. Words strictly inside a chunk are
 * exclusively owned by one thread and written plainly; the chunk's boundary
 * words may also be touched by a neighbouring chunk, so those are OR-ed
 * atomically. That is 2 atomics per chunk instead of one per 16 neurons. */
static inline void or_word(uint64_t *b, int w, uint64_t a, int w_first, int w_last)
{
    if (w <= w_first || w >= w_last)
        __atomic_fetch_or(&b[w], a, __ATOMIC_RELAXED);
    else
        b[w] |= a;
}

static inline void put16(uint64_t *b, int i, uint32_t m, int w_first, int w_last)
{
    const int w = i >> 6, s = i & 63;
    or_word(b, w, (uint64_t)m << s, w_first, w_last);
    if (s > 48) or_word(b, w + 1, (uint64_t)m >> (64 - s), w_first, w_last);
}

/* Scalar reference. Also handles each chunk's tail and non-AVX-512 hosts.
 * Writes spike bits by OR into a PRE-ZEROED output word. */
static void sweep_scalar(int lo, int hi,
                         float *restrict v, float *restrict g,
                         float *restrict refrac, const float *restrict refrac_steps,
                         const uint64_t *restrict sp_in, uint64_t *restrict sp_out,
                         float c_decay, float c_mem,
                         float v_rest, float v_reset, float v_th)
{
    const int w_first = lo >> 6, w_last = (hi - 1) >> 6;
    for (int i = lo; i < hi; ++i) {
        const uint64_t was = BIT_GET(sp_in, i);

        /* refractory counter: increment, or reset if this neuron spiked on
         * the previous step. */
        const float r = was ? 0.0f : refrac[i] + 1.0f;

        const float gi = g[i];
        /* conductance decay. TWO roundings (mul then add) -- must not be
         * contracted into an FMA; the delayed input is added sparsely by the
         * caller in this same two-rounding order. */
        const float gn = gi * c_decay;

        /* membrane. -(v - v_rest) + g -- written v_rest - v, which IEEE-754
         * round-to-nearest makes an exact negation of (v - v_rest).
         *
         * NOTE THE NON-FMA FORM HERE, which differs from sweep_avx512 on
         * purpose. ATen's `add_(t, alpha=)` is vectorised with an FMA in its
         * main body but falls back to a separate multiply-then-add in its
         * SCALAR TAIL, so the reference itself integrates its last
         * (N mod vector_width) neurons with a different rounding from the rest
         * -- for N = 138,639 and a 16-wide float vector that is neurons
         * 138,624..138,638. Reproducing the seam is what makes this kernel
         * bit-identical rather than merely equivalent; using an FMA here
         * produced a 1 ULP divergence on neuron 138,637 at step 394.
         *
         * (The discrepancy is upstream's, not ours: 15 of 138,639 neurons
         * obey a marginally different update rule. It is far below any
         * modelling error, but it is real, and it is recorded in HANDOFF.md.)
         */
        const float vv = v[i];
        const float t = (v_rest - vv) + gi;
        const float vn = (t * c_mem) + vv;

        const uint64_t sp = (vn > v_th) ? 1ULL : 0ULL;

        v[i] = sp ? v_reset : vn;
        g[i] = sp ? 0.0f : gn;
        refrac[i] = r;

        if (sp) or_word(sp_out, i >> 6, 1ULL << (i & 63), w_first, w_last);
    }
}

#ifdef HAVE_AVX512
/* AVX-512 sweep, 16 neurons per iteration.
 *
 * The threshold-and-reset is the reason this maps so well: AVX-512 has real
 * mask registers, so `v > v_th` produces a __mmask16 directly from vcmpps and
 * both resets become single vblendmps instructions -- no select-by-arithmetic,
 * no branches. The same mask, concatenated four times, IS the output spike
 * bitset word, so the bitset costs nothing to produce.
 *
 * Arithmetic is identical to sweep_scalar: vsubps, vaddps, vmulps and one
 * explicit vfmadd132ps. No reassociation, no contraction.
 */
static void sweep_avx512(int lo, int hi,
                         float *restrict v, float *restrict g,
                         float *restrict refrac, const float *restrict refrac_steps,
                         const uint64_t *restrict sp_in, uint64_t *restrict sp_out,
                         float c_decay, float c_mem,
                         float v_rest, float v_reset, float v_th)
{
    const __m512 vc_decay = _mm512_set1_ps(c_decay);
    const __m512 vc_mem   = _mm512_set1_ps(c_mem);
    const __m512 vv_rest  = _mm512_set1_ps(v_rest);
    const __m512 vv_reset = _mm512_set1_ps(v_reset);
    const __m512 vv_th    = _mm512_set1_ps(v_th);
    const __m512 vone     = _mm512_set1_ps(1.0f);
    const __m512 vzero    = _mm512_setzero_ps();

    /* Vectorise from the START of the chunk, exactly as ATen does, so this
     * chunk's leftover (hi-lo) mod 16 neurons land in the scalar tail at the
     * same indices as the reference's. */
    const int w_first = lo >> 6, w_last = (hi - 1) >> 6;
    int i = lo;
    for (; i + 16 <= hi; i += 16) {
        const __mmask16 was = (__mmask16)get16(sp_in, i);

        const __m512 rf = _mm512_loadu_ps(refrac + i);
        const __m512 gi = _mm512_loadu_ps(g + i);
        const __m512 vv = _mm512_loadu_ps(v + i);

        /* refrac + 1, zeroed where the neuron spiked last step */
        const __m512 r = _mm512_mask_blend_ps(was, _mm512_add_ps(rf, vone), vzero);

        const __m512 gn = _mm512_mul_ps(gi, vc_decay);
        const __m512 t  = _mm512_add_ps(_mm512_sub_ps(vv_rest, vv), gi);
        /* SINGLE-rounding FMA, matching ATen's vectorised add_(t, alpha=).
         * Measured: forcing the separate two-rounding form here diverges from
         * the reference at step 33, versus step 514 with the FMA, so ATen's
         * main loop is definitely fused. Its scalar tail is not -- see
         * sweep_scalar. */
        const __m512 vn = _mm512_fmadd_ps(t, vc_mem, vv);

        const __mmask16 sp = _mm512_cmp_ps_mask(vn, vv_th, _CMP_GT_OQ);

        _mm512_storeu_ps(v + i,      _mm512_mask_blend_ps(sp, vn, vv_reset));
        _mm512_storeu_ps(g + i,      _mm512_mask_blend_ps(sp, gn, vzero));
        _mm512_storeu_ps(refrac + i, r);

        put16(sp_out, i, (uint32_t)sp, w_first, w_last);
    }
    if (i < hi) sweep_scalar(i, hi, v, g, refrac, refrac_steps,
                             sp_in, sp_out, c_decay, c_mem, v_rest, v_reset, v_th);
}
#endif

static void sweep(int lo, int hi,
                  float *restrict v, float *restrict g,
                  float *restrict refrac, const float *restrict refrac_steps,
                  const uint64_t *restrict sp_in, uint64_t *restrict sp_out,
                  float c_decay, float c_mem,
                  float v_rest, float v_reset, float v_th)
{
#ifdef HAVE_AVX512
    sweep_avx512(lo, hi, v, g, refrac, refrac_steps, sp_in, sp_out,
                 c_decay, c_mem, v_rest, v_reset, v_th);
#else
    sweep_scalar(lo, hi, v, g, refrac, refrac_steps, sp_in, sp_out,
                 c_decay, c_mem, v_rest, v_reset, v_th);
#endif
}

/* ------------------------------------------------------------------------ */
/* Thread pool.                                                              */
/*                                                                           */
/* The chunks are disjoint neuron ranges, so running them concurrently is a   */
/* pure win with no fidelity question: each thread reads and writes only its  */
/* own slice, and the chunk boundaries are fixed by the ATen mirror rather    */
/* than by our thread count. The kernel's output is therefore INDEPENDENT of  */
/* how many threads we use -- unlike the reference, whose low bits depend on  */
/* torch.get_num_threads().                                                   */
/*                                                                           */
/* A step is only ~200 us, so an OS-level barrier (~5-20 us) would cost real  */
/* percentage points. Workers spin on a generation counter instead, backing   */
/* off to SwitchToThread only after a long spin so an idle engine does not    */
/* peg four cores.                                                            */
/*                                                                           */
/* The one shared resource is the spike bitset: chunk boundaries are not      */
/* 64-aligned, so the first and last word of each chunk may also be touched   */
/* by a neighbour. Those two words per chunk are OR-ed atomically; every      */
/* interior word is exclusively owned and written plainly.                    */
/* ------------------------------------------------------------------------ */
#if defined(_WIN32)
#include <windows.h>

typedef struct {
    int lo, hi;
    float *v, *g, *refrac;
    const float *refrac_steps;
    const uint64_t *sp_in;
    uint64_t *sp_out;
    float c_decay, c_mem, v_rest, v_reset, v_th;
} job_t;

#define MAX_THREADS 16

static struct {
    int nthreads;                 /* total workers, INCLUDING the caller */
    int started;
    volatile long generation;
    volatile long done;
    volatile long stop;
    HANDLE h[MAX_THREADS];
    job_t jobs[MAX_THREADS];
} POOL;

static void run_job(const job_t *j)
{
    sweep(j->lo, j->hi, j->v, j->g, j->refrac, j->refrac_steps,
          j->sp_in, j->sp_out, j->c_decay, j->c_mem,
          j->v_rest, j->v_reset, j->v_th);
}

static DWORD WINAPI worker(LPVOID arg)
{
    const int id = (int)(intptr_t)arg;
    long seen = 0;
    for (;;) {
        long spins = 0;
        while (__atomic_load_n(&POOL.generation, __ATOMIC_ACQUIRE) == seen) {
            if (__atomic_load_n(&POOL.stop, __ATOMIC_ACQUIRE)) return 0;
            if (++spins < 8000) _mm_pause();
            else SwitchToThread();
        }
        seen = __atomic_load_n(&POOL.generation, __ATOMIC_ACQUIRE);
        if (__atomic_load_n(&POOL.stop, __ATOMIC_ACQUIRE)) return 0;
        run_job(&POOL.jobs[id]);
        __atomic_fetch_add(&POOL.done, 1, __ATOMIC_RELEASE);
    }
}

EXPORT int lif_set_threads(int n)
{
    if (n < 1) n = 1;
    if (n > MAX_THREADS) n = MAX_THREADS;
    if (POOL.started) {
        __atomic_store_n(&POOL.stop, 1, __ATOMIC_RELEASE);
        __atomic_fetch_add(&POOL.generation, 1, __ATOMIC_RELEASE);
        for (int i = 1; i < POOL.nthreads; ++i) {
            WaitForSingleObject(POOL.h[i], 1000);
            CloseHandle(POOL.h[i]);
        }
        POOL.started = 0;
        __atomic_store_n(&POOL.stop, 0, __ATOMIC_RELEASE);
        POOL.generation = 0;
    }
    POOL.nthreads = n;
    for (int i = 1; i < n; ++i)
        POOL.h[i] = CreateThread(NULL, 0, worker, (LPVOID)(intptr_t)i, 0, NULL);
    POOL.started = 1;
    return POOL.nthreads;
}

/* Run the chunks in parallel. Falls back to a serial loop when the pool is
 * unconfigured or there are more chunks than threads, so behaviour is
 * identical either way. */
static void sweep_chunks(const int32_t *chunks, int n_chunks,
                         float *v, float *g, float *refrac,
                         const float *refrac_steps,
                         const uint64_t *sp_in, uint64_t *sp_out,
                         float c_decay, float c_mem,
                         float v_rest, float v_reset, float v_th)
{
    if (!POOL.started || POOL.nthreads < 2 || n_chunks != POOL.nthreads) {
        for (int c = 0; c < n_chunks; ++c)
            sweep(chunks[c], chunks[c + 1], v, g, refrac, refrac_steps,
                  sp_in, sp_out, c_decay, c_mem, v_rest, v_reset, v_th);
        return;
    }
    for (int c = 0; c < n_chunks; ++c) {
        job_t *j = &POOL.jobs[c];
        j->lo = chunks[c]; j->hi = chunks[c + 1];
        j->v = v; j->g = g; j->refrac = refrac; j->refrac_steps = refrac_steps;
        j->sp_in = sp_in; j->sp_out = sp_out;
        j->c_decay = c_decay; j->c_mem = c_mem;
        j->v_rest = v_rest; j->v_reset = v_reset; j->v_th = v_th;
    }
    __atomic_store_n(&POOL.done, 0, __ATOMIC_RELEASE);
    __atomic_fetch_add(&POOL.generation, 1, __ATOMIC_RELEASE);
    run_job(&POOL.jobs[0]);                       /* caller takes chunk 0 */
    long spins = 0;
    while (__atomic_load_n(&POOL.done, __ATOMIC_ACQUIRE) < POOL.nthreads - 1) {
        if (++spins < 8000) _mm_pause();
        else SwitchToThread();
    }
}
#else
EXPORT int lif_set_threads(int n) { (void)n; return 1; }
static void sweep_chunks(const int32_t *chunks, int n_chunks,
                         float *v, float *g, float *refrac,
                         const float *refrac_steps,
                         const uint64_t *sp_in, uint64_t *sp_out,
                         float c_decay, float c_mem,
                         float v_rest, float v_reset, float v_th)
{
    for (int c = 0; c < n_chunks; ++c)
        sweep(chunks[c], chunks[c + 1], v, g, refrac, refrac_steps,
              sp_in, sp_out, c_decay, c_mem, v_rest, v_reset, v_th);
}
#endif

/* ------------------------------------------------------------------------ */
/* One complete timestep.                                                    */
/*                                                                           */
/* Returns the number of spikes; fills out_spike_idx with their (ascending)   */
/* indices, which is the order PyTorch's nonzero() produces and therefore the */
/* order the fan-out scatter_add must run in to stay bit-identical.          */
/* ------------------------------------------------------------------------ */
EXPORT int lif_step(
    int n,
    float *v, float *g, float *refrac, const float *refrac_steps,
    uint64_t *sp_bits,                       /* in: prev spikes, out: current */
    uint64_t *sp_scratch,
    float c_decay, float c_mem,
    float v_rest, float v_reset, float v_th,
    const int32_t *del_idx, const float *del_val, int n_del,
    const int32_t *stim_idx, const float *stim_val, int n_stim,
    const int32_t *chunks, int n_chunks,
    int32_t *out_spike_idx)
{
    const int nw = (n + 63) >> 6;

    /* --- Poisson / sensory drive, applied to v before the membrane update,
     *     matching `v.add_(vstim)`. Sparse: only stimulated neurons. --- */
    for (int k = 0; k < n_stim; ++k)
        v[stim_idx[k]] += stim_val[k];

    /* --- the fused sweep, run chunk by chunk.
     *
     * The chunk boundaries mirror ATen's at::parallel_for partitioning of the
     * same tensors: ceil(N / num_threads) neurons each. This matters for
     * exactness, not for speed. ATen vectorises from the start of every chunk
     * and finishes each with a scalar tail that is NOT FMA-fused, so the
     * reference applies a marginally different rounding to the last
     * (chunk_len mod 16) neurons of every chunk. Reproducing the same
     * partition reproduces the same seams, and the kernel becomes bit-identical
     * rather than merely spike-identical.
     *
     * The corollary is worth stating plainly: the reference's bit pattern is a
     * function of PyTorch's thread count. See HANDOFF.md. --- */
    memset(sp_scratch, 0, (size_t)(nw + 1) * sizeof(uint64_t));
    sweep_chunks(chunks, n_chunks, v, g, refrac, refrac_steps,
                 sp_bits, sp_scratch, c_decay, c_mem, v_rest, v_reset, v_th);

    /* --- delayed synaptic input.
     *
     * Applied after the sweep, skipping neurons that just spiked. This is
     * exactly PyTorch's `gnew = g*c + delayed*gate` followed by
     * `gnew.masked_fill_(spiked, 0)`: for a spiking neuron the result is 0
     * either way, and for a non-spiking neuron g[i] currently holds
     * fl(g_old * c_decay), so adding here gives the identical two roundings.
     *
     * The gate is recomputed from the refrac value the sweep just stored,
     * which is the post-increment value PyTorch uses. --- */
    for (int k = 0; k < n_del; ++k) {
        const int i = del_idx[k];
        if (!BIT_GET(sp_scratch, i)) {
            const float gate = (refrac[i] >= refrac_steps[i]) ? 1.0f : 0.0f;
            g[i] = g[i] + del_val[k] * gate;
        }
    }

    /* --- recover spike indices from the bitset (one pass over 17 KB) --- */
    int nsp = 0;
    for (int w = 0; w < nw; ++w) {
        uint64_t b = sp_scratch[w];
        while (b) {
#if defined(__GNUC__) || defined(__clang__)
            const int t = __builtin_ctzll(b);
#else
            int t = 0; uint64_t x = b; while (!(x & 1)) { x >>= 1; ++t; }
#endif
            out_spike_idx[nsp++] = (w << 6) + t;
            b &= b - 1;
        }
    }

    memcpy(sp_bits, sp_scratch, (size_t)nw * sizeof(uint64_t));
    return nsp;
}

/* ------------------------------------------------------------------------ */
/* Event-driven fan-out into the next delay slot.                            */
/*                                                                           */
/* Reproduces PyTorch's                                                      */
/*     rec.zero_(); rec.scatter_add_(0, fan_post[sel], fan_val[sel]);         */
/*     rec.mul_(wScale)                                                      */
/* where `sel` enumerates, for each spiking neuron in ascending index order,  */
/* its synapses in CSC order. Accumulating in that same order gives the same  */
/* rounding, so the result is bit-identical; the scale is applied after the   */
/* accumulation, as upstream does.                                           */
/*                                                                           */
/* Output is written as a compacted (index, value) list -- one entry per      */
/* touched postsynaptic neuron -- so the delay ring stores ~190 pairs per     */
/* slot instead of 138,639 floats.                                           */
/* ------------------------------------------------------------------------ */
EXPORT int lif_fanout(
    int n,
    const int32_t *spike_idx, int nsp,
    const int64_t *crow, const int32_t *post, const float *val,
    float w_scale,
    float *acc,               /* dense scratch, length n, must be all-zero    */
    int32_t *touched,         /* scratch, length n                            */
    uint64_t *touch_bits,     /* scratch, (n+63)/64, must be all-zero         */
    int32_t *out_idx, float *out_val)
{
    int nt = 0;
    for (int s = 0; s < nsp; ++s) {
        const int j = spike_idx[s];
        const int64_t lo = crow[j], hi = crow[j + 1];
        for (int64_t k = lo; k < hi; ++k) {
            const int p = post[k];
            acc[p] += val[k];
            const uint64_t m = 1ULL << (p & 63);
            if (!(touch_bits[p >> 6] & m)) {
                touch_bits[p >> 6] |= m;
                touched[nt++] = p;
            }
        }
    }
    /* compact, scale, and restore the scratch buffers to zero so the next
     * call needs no O(N) clear */
    for (int t = 0; t < nt; ++t) {
        const int p = touched[t];
        out_idx[t] = p;
        out_val[t] = acc[p] * w_scale;
        acc[p] = 0.0f;
        touch_bits[p >> 6] = 0;
    }
    return nt;
}
