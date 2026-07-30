/* Fused single-pass LIF kernel for the Shiu et al. whole-brain Drosophila model.
 *
 * WHY THIS EXISTS
 * ---------------
 * The PyTorch step performs ~12 separate full-array passes (refrac add, compare,
 * mul, add, sub, neg, add, fma, gt, masked_fill x2, copy). Each pass reads and
 * writes N floats, so the step moves ~12x more memory than the algorithm needs,
 * and the 138,639-neuron state (2.2 MB) is streamed through cache twelve times
 * instead of once. Fusing the whole update into a single pass over neurons is
 * the largest available win on a CPU, and it changes no arithmetic at all --
 * only the order in which memory is touched.
 *
 * PORTABILITY
 * -----------
 * This file is compiled WITHOUT -march=native and selects its kernel at RUNTIME
 * via CPUID (including the XCR0 check): AVX-512 -> AVX2 -> portable scalar.
 * All three paths are proven to produce identical results. One binary
 * runs on any x86-64, and on non-x86 the scalar path is used. Building with
 * -march=native would produce a library that only runs on the machine that
 * built it, which is not acceptable for a simulator meant to be distributed.
 *
 * BIT-IDENTITY, AND WHY IT DEPENDS ON THE HOST'S PYTORCH
 * -----------------------------------------------------
 * Every floating-point operation reproduces the corresponding ATen op in the
 * same order with the same rounding. Two facts, established by probing torch
 * 2.13 and verified by divergence bisection, MUST be preserved:
 *
 *   1. `v.add_(t, alpha=a)` is fmaf(t, (float)a, v) -- a SINGLE rounding --
 *      in its VECTORISED body.
 *   2. ...but NOT in its SCALAR TAIL, which is a separate multiply-then-add.
 *   3. `g.mul_(c).add_(t)` is fl(fl(g*c) + t) -- TWO roundings, no FMA.
 *
 * Consequence: ATen integrates the last (chunk_len mod W) neurons of every
 * at::parallel_for chunk with a different rounding from the rest, where W is
 * ATen's float vector width -- 16 under AVX-512, 8 under AVX2. So the seam
 * position is a property of the HOST'S PYTORCH BUILD, not of this kernel.
 * `tail_w` is therefore a parameter: the caller passes ATen's width (read from
 * torch.backends.cpu.get_cpu_capability()) and this kernel reproduces the seam
 * wherever it actually falls. The kernel's own SIMD width is independent of it.
 *
 * Must be compiled with -ffp-contract=off so the compiler cannot fuse (3) into
 * an FMA. -ffast-math would break bit-identity thoroughly.
 *
 * SPARSE DELAY LINE
 * -----------------
 * Upstream stores the 1.8 ms axonal delay as a dense (19, N) fp32 ring buffer:
 * 10.5 MB, which alone exceeds L2 and evicts the neuron state from cache every
 * 19 steps. Its contents are the recurrent input, which has only ~190 non-zeros
 * per step out of 138,639 (1.75 spikes/step x ~110 mean fan-out). Storing each
 * slot as an (index, value) list makes the delay line ~30 KB resident and the
 * whole working set cache-resident. Exact, not approximate: the omitted entries
 * are exactly zero, and fl(x + 0.0f) == x for every value that occurs here
 * (g is never -0.0, since it only ever arrives via a multiply by a positive
 * constant, an addition, or an explicit +0.0 store).
 */

#include <stdint.h>
#include <string.h>
#include <math.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#define LIF_X86 1
#include <immintrin.h>
#include <intrin.h>
#endif

#if defined(_MSC_VER)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#define BIT_GET(bits, i) (((bits)[(i) >> 6] >> ((i) & 63)) & 1ULL)

/* ------------------------------------------------------------------------ */
/* Spike-bitset access.                                                      */
/*                                                                           */
/* The chunk boundaries we must reproduce are not 64-aligned, so bit groups   */
/* straddle words; both buffers carry one word of padding so the w+1 access   */
/* is always in range.                                                       */
/*                                                                           */
/* Words strictly inside a chunk are owned by exactly one thread and written  */
/* plainly. A chunk's first and last words may also be touched by a           */
/* neighbouring chunk, so those are OR-ed atomically -- two atomics per chunk */
/* rather than one per vector group.                                          */
/* ------------------------------------------------------------------------ */
__attribute__((unused)) static inline uint32_t get_bits(const uint64_t *b, int i, int n)
{
    const int w = i >> 6, s = i & 63;
    uint64_t x = b[w] >> s;
    if (s + n > 64) x |= b[w + 1] << (64 - s);
    return (uint32_t)(x & ((n >= 32) ? 0xFFFFFFFFu : ((1u << n) - 1u)));
}

static inline void or_word(uint64_t *b, int w, uint64_t a, int wf, int wl)
{
    if (w <= wf || w >= wl) __atomic_fetch_or(&b[w], a, __ATOMIC_RELAXED);
    else                    b[w] |= a;
}

static inline void put_bits(uint64_t *b, int i, uint32_t m, int n, int wf, int wl)
{
    const int w = i >> 6, s = i & 63;
    or_word(b, w, (uint64_t)m << s, wf, wl);
    if (s + n > 64) or_word(b, w + 1, (uint64_t)m >> (64 - s), wf, wl);
}

/* ------------------------------------------------------------------------ */
/* Per-neuron update. Reproduces, exactly, this sequence from                */
/* brain_engine.step_inplace:                                                */
/*                                                                           */
/*   refrac = spiked_prev ? 0 : refrac + 1                                   */
/*   gnew   = g * c_decay        (+ delayed*gate, applied sparsely by the    */
/*                                caller, in this same two-rounding order)   */
/*   t      = -(v - v_rest) + g                                             */
/*   v      = v + t*c_mem        (FMA in ATen's body, NOT in its tail)      */
/*   sp     = v > v_th                                                      */
/*   v      = sp ? v_reset : v                                              */
/*   g      = sp ? 0 : gnew                                                 */
/* ------------------------------------------------------------------------ */
#define LIF_BODY(USE_FMA)                                                     \
    for (int i = lo; i < hi; ++i) {                                           \
        const float gi = g[i];                                                \
        const float gn = gi * c_decay;                                        \
        const float vv = v[i];                                                \
        /* v_rest - v is an exact negation of v - v_rest under IEEE-754       \
         * round-to-nearest, so this matches ATen's sub_().neg_() pair. */    \
        const float t = (v_rest - vv) + gi;                                   \
        const float vn = (USE_FMA) ? fmaf(t, c_mem, vv) : ((t * c_mem) + vv); \
        const uint64_t sp = (vn > v_th) ? 1ULL : 0ULL;                        \
        v[i] = sp ? v_reset : vn;                                             \
        g[i] = sp ? 0.0f : gn;                                                \
        if (sp) or_word(sp_out, i >> 6, 1ULL << (i & 63), wf, wl);            \
    }

/* Portable scalar sweep. Serves three roles: the reference implementation,
 * ATen's non-fused scalar tail, and the fallback on hosts without AVX2. */
static void sweep_scalar(int lo, int hi, int use_fma,
                         float *restrict v, float *restrict g,
                         uint64_t *restrict sp_out,
                         float c_decay, float c_mem,
                         float v_rest, float v_reset, float v_th,
                         int wf, int wl)
{
    if (use_fma) { LIF_BODY(1) } else { LIF_BODY(0) }
}

#ifdef LIF_X86
/* AVX-512: 16 neurons per iteration.
 *
 * This is where the model's shape pays off. AVX-512 has real mask registers,
 * so `v > v_th` produces a __mmask16 straight from vcmpps, both resets become
 * single vblendmps instructions, and the mask IS the spike bitset -- the
 * bitset costs nothing to produce. No branches, no select-by-arithmetic. */
__attribute__((target("avx512f,avx512bw,avx512dq")))
static int sweep_avx512(int lo, int hi,
                        float *restrict v, float *restrict g,
                        uint64_t *restrict sp_out,
                        float c_decay, float c_mem,
                        float v_rest, float v_reset, float v_th,
                        int wf, int wl)
{
    const __m512 vc_decay = _mm512_set1_ps(c_decay), vc_mem = _mm512_set1_ps(c_mem);
    const __m512 vv_rest = _mm512_set1_ps(v_rest), vv_reset = _mm512_set1_ps(v_reset);
    const __m512 vv_th = _mm512_set1_ps(v_th);
    const __m512 vzero = _mm512_setzero_ps();

    int i = lo;
    for (; i + 16 <= hi; i += 16) {
        const __m512 gi = _mm512_loadu_ps(g + i);
        const __m512 vv = _mm512_loadu_ps(v + i);

        const __m512 gn = _mm512_mul_ps(gi, vc_decay);
        const __m512 t  = _mm512_add_ps(_mm512_sub_ps(vv_rest, vv), gi);
        const __m512 vn = _mm512_fmadd_ps(t, vc_mem, vv);      /* one rounding */
        const __mmask16 sp = _mm512_cmp_ps_mask(vn, vv_th, _CMP_GT_OQ);

        _mm512_storeu_ps(v + i,      _mm512_mask_blend_ps(sp, vn, vv_reset));
        _mm512_storeu_ps(g + i,      _mm512_mask_blend_ps(sp, gn, vzero));
        put_bits(sp_out, i, (uint32_t)sp, 16, wf, wl);
    }
    return i;
}

/* AVX2: 8 neurons per iteration. No mask registers, so the resets use blendv
 * and the spike bits come from movemask. Same arithmetic, same roundings. */
__attribute__((target("avx2,fma")))
static int sweep_avx2(int lo, int hi,
                      float *restrict v, float *restrict g,
                      uint64_t *restrict sp_out,
                      float c_decay, float c_mem,
                      float v_rest, float v_reset, float v_th,
                      int wf, int wl)
{
    const __m256 vc_decay = _mm256_set1_ps(c_decay), vc_mem = _mm256_set1_ps(c_mem);
    const __m256 vv_rest = _mm256_set1_ps(v_rest), vv_reset = _mm256_set1_ps(v_reset);
    const __m256 vv_th = _mm256_set1_ps(v_th);
    const __m256 vzero = _mm256_setzero_ps();

    int i = lo;
    for (; i + 8 <= hi; i += 8) {

        const __m256 gi = _mm256_loadu_ps(g + i);
        const __m256 vv = _mm256_loadu_ps(v + i);

        const __m256 gn = _mm256_mul_ps(gi, vc_decay);
        const __m256 t  = _mm256_add_ps(_mm256_sub_ps(vv_rest, vv), gi);
        const __m256 vn = _mm256_fmadd_ps(t, vc_mem, vv);      /* one rounding */
        const __m256 sp = _mm256_cmp_ps(vn, vv_th, _CMP_GT_OQ);

        _mm256_storeu_ps(v + i,      _mm256_blendv_ps(vn, vv_reset, sp));
        _mm256_storeu_ps(g + i,      _mm256_blendv_ps(gn, vzero, sp));
        put_bits(sp_out, i, (uint32_t)_mm256_movemask_ps(sp), 8, wf, wl);
    }
    return i;
}
#endif /* LIF_X86 */

/* Runtime ISA selection, resolved once. */
enum { ISA_SCALAR = 0, ISA_AVX2 = 1, ISA_AVX512 = 2 };
static int g_isa = -1;

/* CPUID directly rather than __builtin_cpu_supports, which pulls in libgcc's
 * __cpu_model / __cpu_indicator_init -- symbols clang's MSVC target does not
 * link. Doing it by hand also forces the XCR0 check, which is the part people
 * forget: a CPU can report AVX-512 while the OS has not enabled ZMM state
 * saving, and executing the instructions then faults. */
static int detect_isa(void)
{
#ifdef LIF_X86
    int r[4];
    __cpuid(r, 0);
    const int max_leaf = r[0];
    if (max_leaf < 1) return ISA_SCALAR;

    __cpuidex(r, 1, 0);
    const int ecx1 = r[2];
    const int has_osxsave = (ecx1 >> 27) & 1;
    const int has_fma     = (ecx1 >> 12) & 1;
    if (!has_osxsave) return ISA_SCALAR;

    const unsigned long long xcr0 = _xgetbv(0);
    const int ymm_ok = (xcr0 & 0x6) == 0x6;              /* XMM | YMM */
    const int zmm_ok = (xcr0 & 0xE6) == 0xE6;            /* + opmask, ZMM_hi, hi16 */
    if (!ymm_ok || max_leaf < 7) return ISA_SCALAR;

    __cpuidex(r, 7, 0);
    const int ebx7 = r[1];
    const int has_avx2    = (ebx7 >> 5) & 1;
    const int has_avx512f = (ebx7 >> 16) & 1;
    const int has_avx512dq= (ebx7 >> 17) & 1;
    const int has_avx512bw= (ebx7 >> 30) & 1;

    if (zmm_ok && has_avx512f && has_avx512dq && has_avx512bw) return ISA_AVX512;
    if (has_avx2 && has_fma) return ISA_AVX2;
#endif
    return ISA_SCALAR;
}

EXPORT int lif_isa(void)
{
    if (g_isa < 0) g_isa = detect_isa();
    return g_isa;
}

/* Force a lower ISA than the host supports. Exists so the test suite can prove
 * that all three paths produce IDENTICAL results on one machine -- otherwise
 * the AVX2 and scalar paths would only ever be exercised on hardware we cannot
 * test on. Returns the ISA actually in effect (never above what is supported). */
EXPORT int lif_force_isa(int isa)
{
    const int cap = detect_isa();
    g_isa = (isa < 0 || isa > cap) ? cap : isa;
    return g_isa;
}

/* ------------------------------------------------------------------------ */
/* Sweep one chunk.                                                          */
/*                                                                           */
/* `tail_w` is ATen's float vector width on the HOST, not ours. ATen          */
/* vectorises up to lo + floor((hi-lo)/tail_w)*tail_w with an FMA and runs    */
/* the remainder scalar WITHOUT one, so that split is reproduced here         */
/* regardless of which SIMD path we take internally.                          */
/* ------------------------------------------------------------------------ */
static void sweep_chunk(int lo, int hi, int tail_w,
                        float *restrict v, float *restrict g,
                        uint64_t *restrict sp_out,
                        float c_decay, float c_mem,
                        float v_rest, float v_reset, float v_th)
{
    const int wf = lo >> 6, wl = (hi - 1) >> 6;
    if (tail_w < 1) tail_w = 1;
    const int vec_end = lo + ((hi - lo) / tail_w) * tail_w;

    int i = lo;
#ifdef LIF_X86
    if (g_isa == ISA_AVX512)
        i = sweep_avx512(i, vec_end, v, g, sp_out,
                         c_decay, c_mem, v_rest, v_reset, v_th, wf, wl);
    else if (g_isa == ISA_AVX2)
        i = sweep_avx2(i, vec_end, v, g, sp_out,
                       c_decay, c_mem, v_rest, v_reset, v_th, wf, wl);
#endif
    /* still inside ATen's vectorised region -> FMA */
    if (i < vec_end)
        sweep_scalar(i, vec_end, 1, v, g, sp_out,
                     c_decay, c_mem, v_rest, v_reset, v_th, wf, wl);
    /* ATen's scalar tail -> NO FMA */
    if (vec_end < hi)
        sweep_scalar(vec_end, hi, 0, v, g, sp_out,
                     c_decay, c_mem, v_rest, v_reset, v_th, wf, wl);
}

/* ------------------------------------------------------------------------ */
/* Thread pool.                                                              */
/*                                                                           */
/* The chunks are disjoint neuron ranges, so running them concurrently needs  */
/* no synchronisation inside a step. Because the chunk boundaries come from   */
/* the ATen mirror rather than from our thread count, the kernel's OUTPUT IS  */
/* INDEPENDENT of how many threads it uses -- unlike the reference, whose low */
/* bits are a function of torch.get_num_threads().                            */
/*                                                                           */
/* A step is only ~50-200 us, so an OS barrier (5-20 us) would cost real      */
/* percentage points. Workers spin on a generation counter, backing off to a  */
/* yield after a long spin so an idle engine does not peg every core.         */
/* ------------------------------------------------------------------------ */
#if defined(_WIN32)
#include <windows.h>
#define YIELD_CPU() SwitchToThread()
#else
#include <pthread.h>
#include <sched.h>
#define YIELD_CPU() sched_yield()
#endif

#ifndef LIF_X86
#define _mm_pause() ((void)0)
#endif

#define MAX_THREADS 32

typedef struct {
    int kind;                     /* 0 = neuron sweep, 1 = synaptic fan-out */
    int lo, hi, tail_w;
    float *v, *g;
    uint64_t *sp_out;
    float c_decay, c_mem, v_rest, v_reset, v_th;
    /* fan-out job */
    const int32_t *spk;
    const int64_t *crow;
    const int32_t *post;
    const int16_t *wval;
    int32_t *acc, *touched;
    uint64_t *tbits;
    volatile long *ntouched;
} job_t;

static struct {
    int nthreads, started;
    volatile long generation, done, stop;
#if defined(_WIN32)
    HANDLE h[MAX_THREADS];
#else
    pthread_t h[MAX_THREADS];
#endif
    job_t jobs[MAX_THREADS];
} POOL;

/* Accumulate the fan-out of spikes [lo, hi) into the shared int32 accumulator.
 *
 * Running this concurrently is licensed by a theorem, not by tolerance (see
 * ANALYSIS.md section 1): every connectome weight is an exact integer and the
 * largest total in-weight on any neuron is 69,948, so the accumulation NEVER
 * rounds and its result is independent of summation order. Integer atomics are
 * exact by construction, so threads may interleave arbitrarily and the answer
 * is bit-identical to the serial version.
 *
 * Each postsynaptic neuron must appear ONCE in the compacted output. The thread
 * that flips its touch bit from 0 to 1 owns it and claims a slot with an atomic
 * counter; the resulting order varies between runs, which is harmless because
 * the consumer indexes by neuron and each appears exactly once. */
/* MEASURED NEGATIVE RESULT -- read before re-enabling threading here.
 *
 * The theorem in ANALYSIS.md section 1 proves this accumulation is
 * order-independent, so threading it is CORRECT. It is not FASTER. In the
 * saturating regime (~943 spikes/step, ~104,000 synapse updates/step) four
 * threads hammering a shared 138,639-entry accumulator cost two atomics per
 * synapse and made the whole step 2.1x SLOWER than serial -- the regime went
 * from 1.56x faster than PyTorch to 0.73x, i.e. an outright regression.
 * Contention, not correctness, is the binding constraint.
 *
 * The promising fix is not more atomics but the opposite formulation: a PULL
 * fan-out where each thread owns a range of POSTsynaptic neurons and gathers
 * from spiking sources, which needs no atomics at all but costs O(E) instead of
 * O(spikes x fanout). That is Beamer's direction-optimising push/pull switch,
 * and it should win exactly where push loses -- in the dense regimes. Requires
 * a CSR (by-postsynaptic) copy of the connectome alongside the CSC one.
 *
 * Serial remains the default. This path is kept, gated behind `threaded`, for
 * the dense case once a pull path exists to switch to.
 */
static void fanout_range(const job_t *j)
{
    for (int s = j->lo; s < j->hi; ++s) {
        const int src = j->spk[s];
        const int64_t a = j->crow[src], b = j->crow[src + 1];
        for (int64_t k = a; k < b; ++k) {
            const int p = j->post[k];
            __atomic_fetch_add(&j->acc[p], (int32_t)j->wval[k], __ATOMIC_RELAXED);
            const uint64_t m = 1ULL << (p & 63);
            const uint64_t was = __atomic_fetch_or(&j->tbits[p >> 6], m,
                                                   __ATOMIC_RELAXED);
            if (!(was & m)) {
                const long slot = __atomic_fetch_add(j->ntouched, 1,
                                                     __ATOMIC_RELAXED);
                j->touched[slot] = p;
            }
        }
    }
}

/* Serial fan-out: plain loads and stores, no atomics. This is the default and
 * the fast path. */
static void fanout_serial(const job_t *j)
{
    long nt = *j->ntouched;
    for (int s = j->lo; s < j->hi; ++s) {
        const int src = j->spk[s];
        const int64_t a = j->crow[src], b = j->crow[src + 1];
        for (int64_t k = a; k < b; ++k) {
            const int p = j->post[k];
            j->acc[p] += (int32_t)j->wval[k];
            const uint64_t m = 1ULL << (p & 63);
            if (!(j->tbits[p >> 6] & m)) {
                j->tbits[p >> 6] |= m;
                j->touched[nt++] = p;
            }
        }
    }
    *j->ntouched = nt;
}

static void run_job(const job_t *j)
{
    if (j->kind == 1) { fanout_range(j); return; }
    sweep_chunk(j->lo, j->hi, j->tail_w, j->v, j->g, j->sp_out,
                j->c_decay, j->c_mem, j->v_rest, j->v_reset, j->v_th);
}

static void spin_until(volatile long *addr, long target_gt)
{
    long spins = 0;
    while (__atomic_load_n(addr, __ATOMIC_ACQUIRE) <= target_gt) {
        if (++spins < 8000) _mm_pause();
        else YIELD_CPU();
    }
}

#if defined(_WIN32)
static DWORD WINAPI worker(LPVOID arg)
#else
static void *worker(void *arg)
#endif
{
    const int id = (int)(intptr_t)arg;
    long seen = 0;
    for (;;) {
        long spins = 0;
        while (__atomic_load_n(&POOL.generation, __ATOMIC_ACQUIRE) == seen) {
            if (__atomic_load_n(&POOL.stop, __ATOMIC_ACQUIRE)) return 0;
            if (++spins < 8000) _mm_pause();
            else YIELD_CPU();
        }
        seen = __atomic_load_n(&POOL.generation, __ATOMIC_ACQUIRE);
        if (__atomic_load_n(&POOL.stop, __ATOMIC_ACQUIRE)) return 0;
        run_job(&POOL.jobs[id]);
        __atomic_fetch_add(&POOL.done, 1, __ATOMIC_RELEASE);
    }
}

EXPORT int lif_set_threads(int n)
{
    if (g_isa < 0) g_isa = detect_isa();
    if (n < 1) n = 1;
    if (n > MAX_THREADS) n = MAX_THREADS;
    if (POOL.started) {
        __atomic_store_n(&POOL.stop, 1, __ATOMIC_RELEASE);
        __atomic_fetch_add(&POOL.generation, 1, __ATOMIC_RELEASE);
        for (int i = 1; i < POOL.nthreads; ++i) {
#if defined(_WIN32)
            WaitForSingleObject(POOL.h[i], 1000);
            CloseHandle(POOL.h[i]);
#else
            pthread_join(POOL.h[i], NULL);
#endif
        }
        POOL.started = 0;
        __atomic_store_n(&POOL.stop, 0, __ATOMIC_RELEASE);
        POOL.generation = 0;
    }
    POOL.nthreads = n;
    for (int i = 1; i < n; ++i) {
#if defined(_WIN32)
        POOL.h[i] = CreateThread(NULL, 0, worker, (LPVOID)(intptr_t)i, 0, NULL);
#else
        pthread_create(&POOL.h[i], NULL, worker, (void *)(intptr_t)i);
#endif
    }
    POOL.started = 1;
    return POOL.nthreads;
}

static void sweep_all(const int32_t *chunks, int n_chunks, int tail_w,
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
        j->v = v; j->g = g; j->sp_out = sp_out;
        j->c_decay = c_decay; j->c_mem = c_mem;
        j->v_rest = v_rest; j->v_reset = v_reset; j->v_th = v_th;
    }
    __atomic_store_n(&POOL.done, 0, __ATOMIC_RELEASE);
    __atomic_fetch_add(&POOL.generation, 1, __ATOMIC_RELEASE);
    run_job(&POOL.jobs[0]);                       /* caller takes chunk 0 */
    spin_until(&POOL.done, POOL.nthreads - 2);
}

/* ------------------------------------------------------------------------ */
/* One complete timestep.                                                    */
/*                                                                           */
/* Returns the spike count; fills out_spike_idx with ascending indices, which */
/* is the order PyTorch's nonzero() produces and therefore the order the      */
/* fan-out scatter_add must run in to stay bit-identical.                    */
/* ------------------------------------------------------------------------ */
EXPORT int lif_step(
    int n,
    float *v, float *g,
    uint64_t *gate_bits, uint64_t *in_ref,
    int32_t *rc_idx, int32_t *rc_cnt, int *n_ref_io,
    const int32_t *refrac_steps,
    uint64_t *sp_bits, uint64_t *sp_scratch,
    float c_decay, float c_mem,
    float v_rest, float v_reset, float v_th,
    const int32_t *del_idx, const float *del_val, int n_del,
    const int32_t *stim_idx, const float *stim_val, int n_stim,
    const int32_t *chunks, int n_chunks, int tail_w,
    int32_t *out_spike_idx)
{
    const int nw = (n + 63) >> 6;
    if (g_isa < 0) g_isa = detect_isa();

    /* --- sensory drive, applied to v before the membrane update, matching
     *     `v.add_(vstim)`. Sparse: only stimulated neurons. --- */
    for (int k = 0; k < n_stim; ++k)
        v[stim_idx[k]] += stim_val[k];

    /* --- advance the refractory countdowns BEFORE the delayed pass reads the
     *     gate. Walked over the compact list, never over all N, and
     *     deliberately OUTSIDE the sweep: a refractory neuron sits at
     *     v == v_rest and g == 0 exactly, so it is bit-indistinguishable from a
     *     resting one; if this ever rode along inside a skippable sweep the gate
     *     would never reopen and the neuron would be deaf for the rest of the
     *     run. --- */
    {
        int n_ref = *n_ref_io, m = 0;
        for (int k = 0; k < n_ref; ++k) {
            const int i = rc_idx[k];
            if (--rc_cnt[i] <= 0) {                       /* gate reopens */
                gate_bits[i >> 6] |= 1ULL << (i & 63);
                in_ref[i >> 6] &= ~(1ULL << (i & 63));
            } else {
                rc_idx[m++] = i;
            }
        }
        *n_ref_io = m;
    }

    memset(sp_scratch, 0, (size_t)(nw + 1) * sizeof(uint64_t));
    sweep_all(chunks, n_chunks, tail_w, v, g, sp_scratch,
              c_decay, c_mem, v_rest, v_reset, v_th);

    /* --- delayed synaptic input.
     *
     * Applied after the sweep, skipping neurons that just spiked. This is
     * exactly `gnew = g*c + delayed*gate` followed by
     * `gnew.masked_fill_(spiked, 0)`: for a spiking neuron the result is 0
     * either way, and for a non-spiking one g[i] currently holds
     * fl(g_old * c_decay), so adding here gives the identical two roundings.
     * The gate is recomputed from the refrac value the sweep just stored,
     * which is the post-increment value ATen uses. --- */
    for (int k = 0; k < n_del; ++k) {
        const int i = del_idx[k];
        if (!BIT_GET(sp_scratch, i)) {
            const float gate = BIT_GET(gate_bits, i) ? 1.0f : 0.0f;
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

    /* --- neurons that spiked THIS step enter refractoriness.
     *
     * c = refrac_steps + 1, NOT refrac_steps. The reference's gate is closed for
     * steps t+1..t+refrac_steps and reopens at t+refrac_steps+1, and the
     * decrement above runs once per step before the gate is read; using
     * refrac_steps here reopens one step early and admits one step of input the
     * reference discards. A neuron already counting simply has its count reset,
     * which is what a re-spike means. --- */
    for (int k = 0; k < nsp; ++k) {
        const int i = out_spike_idx[k];
        rc_cnt[i] = refrac_steps[i] + 1;
        const uint64_t m = 1ULL << (i & 63);
        gate_bits[i >> 6] &= ~m;
        if (!(in_ref[i >> 6] & m)) {
            in_ref[i >> 6] |= m;
            rc_idx[(*n_ref_io)++] = i;
        }
    }

    memcpy(sp_bits, sp_scratch, (size_t)(nw + 1) * sizeof(uint64_t));
    return nsp;
}

/* ------------------------------------------------------------------------ */
/* Event-driven fan-out into the next delay slot.                            */
/*                                                                           */
/* Reproduces                                                                */
/*     rec.zero_(); rec.scatter_add_(0, fan_post[sel], fan_val[sel]);         */
/*     rec.mul_(wScale)                                                      */
/* where `sel` enumerates, for each spiking neuron in ascending index order,  */
/* its synapses in CSC order. Accumulating in that order gives the same       */
/* rounding, and the scale is applied after accumulation, as upstream does.   */
/*                                                                           */
/* Output is a compacted (index, value) list -- one entry per touched         */
/* postsynaptic neuron -- so a delay slot holds ~190 pairs, not 138,639       */
/* floats. The scratch buffers are restored to zero on the way out, so no     */
/* O(N) clear is ever needed.                                                */
/* ------------------------------------------------------------------------ */
EXPORT int lif_fanout(
    int n,
    const int32_t *spike_idx, int nsp,
    const int64_t *crow, const int32_t *post, const int16_t *val,
    float w_scale,
    int32_t *acc, int32_t *touched, uint64_t *touch_bits,
    int32_t *out_idx, float *out_val, int threaded)
{
    long nt = 0;

    if (threaded && POOL.started && POOL.nthreads > 1 && nsp >= 64) {
        const int T = POOL.nthreads;
        const int per = (nsp + T - 1) / T;
        for (int c = 0; c < T; ++c) {
            job_t *j = &POOL.jobs[c];
            j->kind = 1;
            j->lo = c * per;
            j->hi = (c + 1) * per < nsp ? (c + 1) * per : nsp;
            if (j->lo > nsp) j->lo = nsp;
            j->spk = spike_idx; j->crow = crow; j->post = post; j->wval = val;
            j->acc = acc; j->touched = touched; j->tbits = touch_bits;
            j->ntouched = &nt;
        }
        __atomic_store_n(&POOL.done, 0, __ATOMIC_RELEASE);
        __atomic_fetch_add(&POOL.generation, 1, __ATOMIC_RELEASE);
        run_job(&POOL.jobs[0]);
        spin_until(&POOL.done, POOL.nthreads - 2);
    } else {
        job_t j;
        j.kind = 1; j.lo = 0; j.hi = nsp;
        j.spk = spike_idx; j.crow = crow; j.post = post; j.wval = val;
        j.acc = acc; j.touched = touched; j.tbits = touch_bits;
        j.ntouched = &nt;
        fanout_serial(&j);
    }

    /* Compact, scale, and restore the scratch buffers to zero so the next call
     * needs no O(N) clear. The int32 total is exact (|total| <= 69,948), so the
     * conversion to float is exact and only the single multiply by w_scale
     * rounds -- identical to the float-accumulation version. */
    const int cnt = (int)nt;
    for (int t = 0; t < cnt; ++t) {
        const int p = touched[t];
        out_idx[t] = p;
        out_val[t] = (float)acc[p] * w_scale;
        acc[p] = 0;
        touch_bits[p >> 6] = 0;
    }
    return cnt;
}
