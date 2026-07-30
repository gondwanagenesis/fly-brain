/* Fused single-pass multi-model neuron kernel for the whole-brain Drosophila
 * connectome (Shiu et al. 2024 on FlyWire v783).
 *
 * NINE MEMBRANE MODELS, ONE NETWORK
 * ---------------------------------
 * The connectome machinery -- the 1.8 ms sparse delay line, the event-driven
 * fan-out, the refractory gate, 16-neuron tile skipping, the ATen-faithful
 * chunk partition, the spin-barrier thread pool -- is a property of the
 * NETWORK and the SCHEDULE, not of the membrane equation. So it is written
 * once here and shared by every model; a model contributes only how (v, g,
 * aux) advance over one dt, when a spike is declared, and what the reset does.
 *
 *   0 LIF forward-Euler   the reference: bit-identical to run_pytorch.py
 *   1 LIF exact           Rotter & Diesmann 1999 propagator
 *   2 Izhikevich 2003     quadratic + recovery variable
 *   3 AdEx                Brette & Gerstner 2005, w by Rush-Larsen
 *   4 EIF                 Fourcaud-Trocme et al. 2003
 *   5 QIF                 Ermentrout & Kopell normal form
 *   6 resonate-and-fire   Izhikevich 2001, exactly rotated
 *   7 Hodgkin-Huxley      1952, gates by Rush & Larsen 1978
 *   8 GLIF                adaptive threshold, Allen Institute GLIF-3 form
 *
 * Model 0 keeps its own hand-written sweep because it alone has to reproduce
 * ATen's mixed FMA/non-FMA rounding seam (see below) to stay bit-identical to
 * the PyTorch reference. Models 1-8 are generated from sweep_template.h, once
 * per instruction set, so their three ISA paths are identical by construction
 * rather than by inspection.
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

#include "simd.h"
#include "nrn_params.h"

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
/* Models 1-8: three instantiations of one source.                           */
/*                                                                           */
/* sweep_template.h is included once per instruction set with the generic op */
/* macros bound to that ISA. Nothing about a model's arithmetic is written   */
/* more than once, so "AVX-512, AVX2 and scalar agree" is a property of the  */
/* build rather than a claim that has to be re-audited whenever a model is   */
/* touched. verify_models.py still checks it at runtime on this host.        */
/* ------------------------------------------------------------------------ */
#define FN(x)     x##_sc
#define FN_ATTR
#define VW        S_W
#define VF        S_VF
#define VM        S_VM
#define VSET1     S_VSET1
#define VLOADM    S_VLOADM
#define VSTOREM   S_VSTOREM
#define VADD      S_VADD
#define VSUB      S_VSUB
#define VMUL      S_VMUL
#define VDIV      S_VDIV
#define VFMA      S_VFMA
#define VMIN      S_VMIN
#define VMAX      S_VMAX
#define VGT       S_VGT
#define VGE       S_VGE
#define VLT       S_VLT
#define VAND      S_VAND
#define VOR       S_VOR
#define VSEL      S_VSEL
#define VBITS     S_VBITS
#define VANY      S_VANY
#define VMASK_LOW S_VMASK_LOW
#define VMFALSE   S_VMFALSE
#define VSCALEF   S_VSCALEF
#define VROUND    S_VROUND
#include "sweep_template.h"
#include "simd_undef.h"

#ifdef LIF_X86
#define FN(x)     x##_a2
#define FN_ATTR   __attribute__((target("avx2,fma")))
#define VW        A2_W
#define VF        A2_VF
#define VM        A2_VM
#define VSET1     A2_VSET1
#define VLOADM    A2_VLOADM
#define VSTOREM   A2_VSTOREM
#define VADD      A2_VADD
#define VSUB      A2_VSUB
#define VMUL      A2_VMUL
#define VDIV      A2_VDIV
#define VFMA      A2_VFMA
#define VMIN      A2_VMIN
#define VMAX      A2_VMAX
#define VGT       A2_VGT
#define VGE       A2_VGE
#define VLT       A2_VLT
#define VAND      A2_VAND
#define VOR       A2_VOR
#define VSEL      A2_VSEL
#define VBITS     A2_VBITS
#define VANY      A2_VANY
#define VMASK_LOW A2_VMASK_LOW
#define VMFALSE   A2_VMFALSE
#define VSCALEF   A2_VSCALEF
#define VROUND    A2_VROUND
#include "sweep_template.h"
#include "simd_undef.h"

#define FN(x)     x##_a5
#define FN_ATTR   __attribute__((target("avx512f,avx512bw,avx512dq")))
#define VW        A5_W
#define VF        A5_VF
#define VM        A5_VM
#define VSET1     A5_VSET1
#define VLOADM    A5_VLOADM
#define VSTOREM   A5_VSTOREM
#define VADD      A5_VADD
#define VSUB      A5_VSUB
#define VMUL      A5_VMUL
#define VDIV      A5_VDIV
#define VFMA      A5_VFMA
#define VMIN      A5_VMIN
#define VMAX      A5_VMAX
#define VGT       A5_VGT
#define VGE       A5_VGE
#define VLT       A5_VLT
#define VAND      A5_VAND
#define VOR       A5_VOR
#define VSEL      A5_VSEL
#define VBITS     A5_VBITS
#define VANY      A5_VANY
#define VMASK_LOW A5_VMASK_LOW
#define VMFALSE   A5_VMFALSE
#define VSCALEF   A5_VSCALEF
#define VROUND    A5_VROUND
#include "sweep_template.h"
#include "simd_undef.h"
#endif /* LIF_X86 */

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
                         int wf, int wl, const uint64_t *fma_bits)
{
    /* use_fma == 2 means the tile straddles ATen's vector/tail seam, so the
     * rounding differs per neuron and has to be looked up. At most 31 neurons
     * in the whole brain are in that state. */
    if (use_fma == 2) { LIF_BODY(BIT_GET(fma_bits, i)) }
    else if (use_fma) { LIF_BODY(1) }
    else              { LIF_BODY(0) }
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

#ifdef LIF_X86
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
/* Sweep the live tiles of one chunk.
 *
 * Every tile carries a single uniform rounding mode (see the module note on
 * chunk-relative tiling), so the FMA/non-FMA seam is a property of the tile
 * rather than something the loop has to re-derive. */
static void sweep_tile_range(int t0, int t1,
                             const int32_t *tile_lo, const uint8_t *tile_fma,
                             const uint64_t *tile_live, const uint64_t *fma_bits,
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
        /* == 1, not just truthy: mode 2 means the tile straddles ATen's
         * vector/tail seam, so its rounding varies per neuron and it must take
         * the scalar path that looks the bit up. */
        if (g_isa == ISA_AVX512 && tile_fma[t] == 1 && hi - lo == 16) {
            group16(lo, v, g, sp_out, c_decay, c_mem, v_rest, v_reset, v_th, wf, wl);
            continue;
        }
#endif
        sweep_scalar(lo, hi, tile_fma[t], v, g, sp_out,
                     c_decay, c_mem, v_rest, v_reset, v_th, wf, wl, fma_bits);
    }
}

/* Clear tiles that are provably inert. Runs every RESCAN steps; between scans a
 * live tile stays live, which is conservative and never skips a live neuron. */
/* Clear tiles that are provably inert, for ANY model.
 *
 * "Inert" means the model's own update is a FIXED POINT at this neuron's
 * current state with zero input: advancing it reproduces the same bits, so
 * skipping it is lossless rather than approximate. For LIF that state is
 * (v_rest, 0); for Izhikevich it is (v_r, b*v_r); for HH it is the resting
 * gate steady state. The rest vector is supplied by the caller, which computed
 * it by iterating the model's own reference update to stationarity in float32
 * and asserting that one more step changes nothing -- so this predicate never
 * has to know which model it is testing.
 *
 * `rest[0]` is the resting membrane value, `rest[1..n_aux]` the resting
 * auxiliary values. g's rest value is 0 for every model. */
static void rescan_tiles(int nt, const int32_t *tile_lo, uint64_t *tile_live,
                         const uint64_t *tile_pin,
                         const float *v, const float *g,
                         const float *aux, int naux_stride, int n_aux,
                         const float *rest,
                         const uint64_t *gate_bits, const uint64_t *sp_bits)
{
    const float v_rest = rest[0];
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
            for (int k = 0; k < n_aux; ++k) {
                if (aux[(size_t)k * naux_stride + i] != rest[1 + k]) {
                    inert = 0;
                    break;
                }
            }
            if (!inert) break;
        }
        if (inert) tile_live[t >> 6] &= ~m;
    }
}

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
                     c_decay, c_mem, v_rest, v_reset, v_th, wf, wl, NULL);
    /* ATen's scalar tail -> NO FMA */
    if (vec_end < hi)
        sweep_scalar(vec_end, hi, 0, v, g, sp_out,
                     c_decay, c_mem, v_rest, v_reset, v_th, wf, wl, NULL);
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
    int kind;                     /* 0 = LIF sweep, 1 = fan-out, 2 = model sweep */
    int model, naux_stride;
    float *aux;
    const nrn_params *P;
    int lo, hi, tail_w;
    int t0, t1;
    const int32_t *tile_lo;
    const uint8_t *tile_fma;
    const uint64_t *tile_live;
    const uint64_t *fma_bits;
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

/* ------------------------------------------------------------------------ */
/* Radix-partitioned fan-out -- IMPLEMENTED, MEASURED, AND OFF BY DEFAULT.    */
/*                                                                           */
/* Read this before re-treading it. The hypothesis was reasonable and the     */
/* measurement refuted it.                                                   */
/*                                                                           */
/* HYPOTHESIS. profile_split.py shows the cost per delivered edge nearly      */
/* triples between broad(10000) and the saturating regime -- 8.4 ns to        */
/* 21.7 ns -- though the work per edge is identical, one indexed int32 add.   */
/* The obvious explanation is residency: `acc` is 138,639 int32 = 542 KB and  */
/* a random scatter over all of it should miss cache. The standard fix is a   */
/* radix-partitioned scatter (as in a radix join, or the counting phase of a  */
/* sparse transpose): bucket the (post, val) pairs by the high bits of        */
/* `post`, then accumulate a bucket at a time so the live accumulator slice   */
/* is 16 KB and stays in L1. More total traffic, but all of it sequential.    */
/*                                                                           */
/* MEASUREMENT (flyloop/bench_fanout.py). It never wins, at any size:         */
/*                                                                           */
/*     edges/step     direct   partitioned   speedup   ns/edge direct        */
/*          1,283     8.5 us       14.0 us     0.61x        6.66             */
/*         28,322   182.3 us      310.3 us     0.59x        6.44             */
/*        106,124    1.07 ms       1.44 ms     0.74x       10.04             */
/*        907,682    5.42 ms       9.96 ms     0.54x        5.97             */
/*      1,796,811   12.44 ms      26.48 ms     0.47x        6.93             */
/*                                                                           */
/* The direct path holds 6-11 ns per edge all the way to 1.8M edges/step and  */
/* does NOT degrade with size, so it was never miss-bound in the first place. */
/* The reason: the CSC range of one presynaptic neuron lists its targets in   */
/* ASCENDING order, so each source's fan-out is a sequential walk the         */
/* prefetcher handles. The randomness is only BETWEEN sources.                */
/*                                                                           */
/* So the 21.7 ns/edge seen in the live saturating step is not intrinsic to   */
/* the scatter -- it is CONTENTION with the rest of the step, whose working   */
/* set (neuron state, tile arrays, delay slots) competes for the same cache.  */
/* Partitioning makes that worse, by adding ~1 MB of extra traffic that       */
/* evicts more of it.                                                        */
/*                                                                           */
/* Kept rather than deleted so the measurement stays reproducible: pass a     */
/* non-NULL pair buffer to enable it. The engine passes NULL.                 */
/*                                                                           */
/* Exactness was never the issue and was verified anyway: integer             */
/* accumulation is associative and the totals never round (ANALYSIS.md        */
/* section 1), so both paths produce bit-identical compacted output at every  */
/* size tested.                                                              */
/*                                                                           */
/* NOT the pull direction either. Beamer's direction-optimising switch was    */
/* the other obvious candidate and the arithmetic rules it out: a pull        */
/* fan-out costs O(E) = 15,091,983 edges every step regardless of activity,   */
/* against 136,605 actually needed in the densest regime tested -- 110x more  */
/* work. Push wins everywhere in this connectome. The problem was never the   */
/* direction, and it was never the scatter.                                   */
/* ------------------------------------------------------------------------ */
#define PART_BITS 12                       /* 4096 neurons per bucket = 16 KB */
#define PART_MAX  64
/* There is no crossover -- see the table above. Left low so that supplying
 * a pair buffer actually exercises the path when benchmarking it. */
#define PART_MIN_EDGES 512

static void fanout_partitioned(const job_t *j, int32_t *pair_post,
                               int32_t *pair_val, long total, int n)
{
    const int nb = (n + (1 << PART_BITS) - 1) >> PART_BITS;
    long off[PART_MAX + 1];
    long cnt[PART_MAX];
    for (int b = 0; b < nb; ++b) cnt[b] = 0;

    /* pass 1 -- count per bucket */
    for (int s = j->lo; s < j->hi; ++s) {
        const int src = j->spk[s];
        for (int64_t k = j->crow[src]; k < j->crow[src + 1]; ++k)
            ++cnt[j->post[k] >> PART_BITS];
    }
    off[0] = 0;
    for (int b = 0; b < nb; ++b) off[b + 1] = off[b] + cnt[b];

    /* pass 2 -- place the pairs, sequentially per bucket */
    long put[PART_MAX];
    for (int b = 0; b < nb; ++b) put[b] = off[b];
    for (int s = j->lo; s < j->hi; ++s) {
        const int src = j->spk[s];
        for (int64_t k = j->crow[src]; k < j->crow[src + 1]; ++k) {
            const int p = j->post[k];
            const long w = put[p >> PART_BITS]++;
            pair_post[w] = p;
            pair_val[w] = (int32_t)j->wval[k];
        }
    }

    /* pass 3 -- accumulate one bucket at a time; the live slice is 16 KB */
    long nt = *j->ntouched;
    for (int b = 0; b < nb; ++b) {
        for (long k = off[b]; k < off[b + 1]; ++k) {
            const int p = pair_post[k];
            j->acc[p] += pair_val[k];
            const uint64_t m = 1ULL << (p & 63);
            if (!(j->tbits[p >> 6] & m)) {
                j->tbits[p >> 6] |= m;
                j->touched[nt++] = p;
            }
        }
    }
    *j->ntouched = nt;
    (void)total;
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

/* Dispatch a model sweep to the widest path this CPU actually supports.
 * The three implementations are generated from one source, so this choice is a
 * pure speed knob: verify_models.py forces each in turn and checks that the
 * whole-brain state comes out bit-identical. */
static void sweep_model_isa(int model, int t0, int t1,
                            const int32_t *tile_lo, const uint64_t *tile_live,
                            float *v, float *g, float *aux, int naux_stride,
                            uint64_t *sp_out, const nrn_params *P,
                            int wf, int wl)
{
#ifdef LIF_X86
    if (g_isa == ISA_AVX512) {
        sweep_tiles_a5(model, t0, t1, tile_lo, tile_live, v, g, aux,
                       naux_stride, sp_out, P, wf, wl);
        return;
    }
    if (g_isa == ISA_AVX2) {
        sweep_tiles_a2(model, t0, t1, tile_lo, tile_live, v, g, aux,
                       naux_stride, sp_out, P, wf, wl);
        return;
    }
#endif
    sweep_tiles_sc(model, t0, t1, tile_lo, tile_live, v, g, aux,
                   naux_stride, sp_out, P, wf, wl);
}

static void run_job(const job_t *j)
{
    if (j->kind == 1) { fanout_range(j); return; }
    if (j->kind == 2) {
        const int wf = j->tile_lo[j->t0] >> 6;
        const int wl = (j->tile_lo[j->t1] - 1) >> 6;
        sweep_model_isa(j->model, j->t0, j->t1, j->tile_lo, j->tile_live,
                        j->v, j->g, j->aux, j->naux_stride, j->sp_out, j->P,
                        wf, wl);
        return;
    }
    sweep_tile_range(j->t0, j->t1, j->tile_lo, j->tile_fma, j->tile_live,
                     j->fma_bits, j->v, j->g, j->sp_out, j->c_decay, j->c_mem,
                     j->v_rest, j->v_reset, j->v_th);
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

static void sweep_all(const int32_t *chunk_tile, int n_chunks,
                      const int32_t *tile_lo, const uint8_t *tile_fma,
                      const uint64_t *tile_live, const uint64_t *fma_bits,
                      float *v, float *g, uint64_t *sp_out,
                      float c_decay, float c_mem,
                      float v_rest, float v_reset, float v_th)
{
    if (!POOL.started || POOL.nthreads < 2 || n_chunks != POOL.nthreads) {
        for (int c = 0; c < n_chunks; ++c)
            sweep_tile_range(chunk_tile[c], chunk_tile[c + 1], tile_lo, tile_fma,
                             tile_live, fma_bits, v, g, sp_out, c_decay, c_mem,
                             v_rest, v_reset, v_th);
        return;
    }
    for (int c = 0; c < n_chunks; ++c) {
        job_t *j = &POOL.jobs[c];
        j->kind = 0;
        j->t0 = chunk_tile[c]; j->t1 = chunk_tile[c + 1];
        j->tile_lo = tile_lo; j->tile_fma = tile_fma; j->tile_live = tile_live;
        j->fma_bits = fma_bits;
        j->v = v; j->g = g; j->sp_out = sp_out;
        j->c_decay = c_decay; j->c_mem = c_mem;
        j->v_rest = v_rest; j->v_reset = v_reset; j->v_th = v_th;
    }
    __atomic_store_n(&POOL.done, 0, __ATOMIC_RELEASE);
    __atomic_fetch_add(&POOL.generation, 1, __ATOMIC_RELEASE);
    run_job(&POOL.jobs[0]);                       /* caller takes chunk 0 */
    spin_until(&POOL.done, POOL.nthreads - 2);
}

/* Same partition, same barrier, any model. Chunks are disjoint neuron ranges,
 * so no model needs synchronisation inside a step -- that is a property of the
 * schedule, not of the membrane equation, which is exactly why the thread pool
 * did not have to be duplicated per model. */
static void sweep_all_model(int model, const int32_t *chunk_tile, int n_chunks,
                            const int32_t *tile_lo, const uint64_t *tile_live,
                            float *v, float *g, float *aux, int naux_stride,
                            uint64_t *sp_out, const nrn_params *P)
{
    if (!POOL.started || POOL.nthreads < 2 || n_chunks != POOL.nthreads) {
        for (int c = 0; c < n_chunks; ++c) {
            const int t0 = chunk_tile[c], t1 = chunk_tile[c + 1];
            if (t1 <= t0) continue;
            sweep_model_isa(model, t0, t1, tile_lo, tile_live, v, g, aux,
                            naux_stride, sp_out, P,
                            tile_lo[t0] >> 6, (tile_lo[t1] - 1) >> 6);
        }
        return;
    }
    for (int c = 0; c < n_chunks; ++c) {
        job_t *j = &POOL.jobs[c];
        j->kind = 2;
        j->model = model;
        j->t0 = chunk_tile[c]; j->t1 = chunk_tile[c + 1];
        j->tile_lo = tile_lo; j->tile_live = tile_live;
        j->v = v; j->g = g; j->aux = aux; j->naux_stride = naux_stride;
        j->sp_out = sp_out; j->P = P;
    }
    __atomic_store_n(&POOL.done, 0, __ATOMIC_RELEASE);
    __atomic_fetch_add(&POOL.generation, 1, __ATOMIC_RELEASE);
    run_job(&POOL.jobs[0]);
    spin_until(&POOL.done, POOL.nthreads - 2);
}

/* ------------------------------------------------------------------------ */
/* One complete timestep.                                                    */
/*                                                                           */
/* Returns the spike count; fills out_spike_idx with ascending indices, which */
/* is the order PyTorch's nonzero() produces and therefore the order the      */
/* fan-out scatter_add must run in to stay bit-identical.                    */
/* ------------------------------------------------------------------------ */
EXPORT int nrn_step(
    int model, const nrn_params *P,
    float *aux, int naux_stride, int n_aux, const float *rest,
    int n,
    float *v, float *g,
    uint64_t *gate_bits, uint64_t *in_ref,
    int32_t *rc_idx, int32_t *rc_cnt, int *n_ref_io,
    const int32_t *refrac_steps,
    uint64_t *sp_bits, uint64_t *sp_scratch,
    const int32_t *del_idx, const float *del_val, int n_del,
    const int32_t *stim_idx, const float *stim_val, int n_stim,
    const int32_t *chunks, int n_chunks, int tail_w,
    const int32_t *tile_lo, const uint8_t *tile_fma, uint64_t *tile_live,
    const uint64_t *fma_bits,
    const uint64_t *tile_pin, const int32_t *chunk_tile,
    const int32_t *neuron_tile, int n_tiles, int rescan,
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
    (void)chunks; (void)tail_w;
    /* Model 0 keeps the hand-written sweep because it alone has to reproduce
     * ATen's per-chunk FMA/non-FMA seam. Everything else goes through the
     * generated per-ISA sweeps, which need no seam because they have no
     * PyTorch counterpart whose low bits they must match. */
    if (model == M_LIF_EULER)
        sweep_all(chunk_tile, n_chunks, tile_lo, tile_fma, tile_live, fma_bits,
                  v, g, sp_scratch, P->c_decay, P->c_mem,
                  P->v_rest, P->v_reset, P->v_th);
    else
        sweep_all_model(model, chunk_tile, n_chunks, tile_lo, tile_live,
                        v, g, aux, naux_stride, sp_scratch, P);

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
        /* B4: mark live at CONSUMPTION, the moment g can change -- not at
         * emission, because a rescan could clear the tile during the up-to-19
         * steps the arrival spends in flight. Marked unconditionally: a spiking
         * neuron's tile is live anyway, and this keeps the invariant simple. */
        {
            const int tt = neuron_tile[i];
            tile_live[tt >> 6] |= 1ULL << (tt & 63);
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

    if (rescan)
        rescan_tiles(n_tiles, tile_lo, tile_live, tile_pin, v, g,
                     aux, naux_stride, n_aux, rest, gate_bits, sp_scratch);

    memcpy(sp_bits, sp_scratch, (size_t)(nw + 1) * sizeof(uint64_t));
    return nsp;
}

/* The original entry point, preserved byte-for-byte in its behaviour.
 *
 * It exists so that verify_native.py and verify_all.py -- the gates that prove
 * bit-identity with the PyTorch reference -- keep testing the SAME code path
 * after the kernel grew eight more models. Since it forwards to nrn_step, a
 * green run of those gates is also evidence that the generalised step did not
 * disturb the LIF path, which is the property most worth protecting here. */
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
    const int32_t *tile_lo, const uint8_t *tile_fma, uint64_t *tile_live,
    const uint64_t *fma_bits,
    const uint64_t *tile_pin, const int32_t *chunk_tile,
    const int32_t *neuron_tile, int n_tiles, int rescan,
    int32_t *out_spike_idx)
{
    nrn_params P;
    memset(&P, 0, sizeof P);
    P.c_decay = c_decay; P.c_mem = c_mem;
    P.v_rest = v_rest; P.v_reset = v_reset; P.v_th = v_th;
    const float rest[1] = { v_rest };
    return nrn_step(M_LIF_EULER, &P, NULL, 0, 0, rest,
                    n, v, g, gate_bits, in_ref, rc_idx, rc_cnt, n_ref_io,
                    refrac_steps, sp_bits, sp_scratch,
                    del_idx, del_val, n_del, stim_idx, stim_val, n_stim,
                    chunks, n_chunks, tail_w, tile_lo, tile_fma, tile_live,
                    fma_bits, tile_pin, chunk_tile, neuron_tile, n_tiles,
                    rescan, out_spike_idx);
}

/* ------------------------------------------------------------------------ */
/* Model introspection helpers.                                              */
/*                                                                           */
/* These exist so that Python never has to carry a SECOND implementation of  */
/* any model. Calibrating the synaptic gain, locating the exact float32       */
/* resting state that tile-skipping needs, and auditing the vectorised        */
/* exponential are all done by driving THIS code, not a numpy lookalike that  */
/* could drift away from it.                                                 */
/* ------------------------------------------------------------------------ */

/* Size check, so a mismatch between nrn_params.h and models.py's ctypes
 * Structure fails at import instead of silently reading garbage offsets. */
EXPORT int nrn_params_size(void) { return (int)sizeof(nrn_params); }

/* Advance `n` neurons `steps` times with NO network: no delay line, no
 * fan-out, no refractory gate, no tile skipping. Every neuron is swept every
 * step. `g` decays by the model's own factor and is otherwise untouched, so a
 * caller can inject an impulse by presetting it.
 *
 * Spikes are counted, not recorded -- the callers that need timing drive one
 * step at a time and look at the state themselves. */
EXPORT int nrn_sweep_raw(int model, const nrn_params *P,
                         float *v, float *g, float *aux, int naux_stride,
                         int n, int steps,
                         int32_t *tile_lo_buf, uint64_t *tile_live_buf,
                         uint64_t *sp_buf)
{
    if (g_isa < 0) g_isa = detect_isa();
    const int n_tiles = (n + 15) / 16;
    for (int t = 0; t <= n_tiles; ++t) {
        int x = t * 16;
        tile_lo_buf[t] = x < n ? x : n;
    }
    const int ntw = (n_tiles + 63) / 64 + 1;
    const int nw = (n + 63) / 64 + 1;
    int total = 0;
    for (int s = 0; s < steps; ++s) {
        for (int i = 0; i < ntw; ++i) tile_live_buf[i] = ~0ULL;
        memset(sp_buf, 0, (size_t)nw * sizeof(uint64_t));
        sweep_model_isa(model, 0, n_tiles, tile_lo_buf, tile_live_buf,
                        v, g, aux, naux_stride, sp_buf, P, 0, nw - 1);
        for (int w = 0; w < nw; ++w) {
            uint64_t b = sp_buf[w];
            while (b) { ++total; b &= b - 1; }
        }
    }
    return total;
}

/* Evaluate the kernel's own vectorised exp on an array, through whichever ISA
 * path is currently selected. verify_exp.py uses this to walk every float32 in
 * the reachable argument range against a float64 reference -- an exhaustive
 * audit rather than a sampled one. */
#ifdef LIF_X86
__attribute__((target("avx512f,avx512bw,avx512dq")))
static int exp_probe_a5(const float *x, float *out, int n)
{
    int i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(out + i, fly_exp_a5(_mm512_loadu_ps(x + i)));
    return i;
}

__attribute__((target("avx2,fma")))
static int exp_probe_a2(const float *x, float *out, int n)
{
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(out + i, fly_exp_a2(_mm256_loadu_ps(x + i)));
    return i;
}
#endif

EXPORT void nrn_exp_probe(const float *x, float *out, int n)
{
    if (g_isa < 0) g_isa = detect_isa();
    int i = 0;
#ifdef LIF_X86
    if (g_isa == ISA_AVX512)     i = exp_probe_a5(x, out, n);
    else if (g_isa == ISA_AVX2)  i = exp_probe_a2(x, out, n);
#endif
    for (; i < n; ++i) out[i] = fly_exp_sc(x[i]);
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
    int32_t *out_idx, float *out_val, int threaded,
    int32_t *pair_post, int32_t *pair_val, long pair_cap)
{
    long nt = 0;

    /* Only when the caller supplies a pair buffer, which the engine does
     * not: partitioning measured SLOWER at every size. See the note above. */
    if (pair_post && nsp > 0) {
        long total = 0;
        for (int s = 0; s < nsp; ++s) {
            const int src = spike_idx[s];
            total += (long)(crow[src + 1] - crow[src]);
        }
        if (total > PART_MIN_EDGES && total <= pair_cap) {
            job_t j;
            j.kind = 1; j.lo = 0; j.hi = nsp;
            j.spk = spike_idx; j.crow = crow; j.post = post; j.wval = val;
            j.acc = acc; j.touched = touched; j.tbits = touch_bits;
            j.ntouched = &nt;
            fanout_partitioned(&j, pair_post, pair_val, total, n);
            goto compact;
        }
    }

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

compact:
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
