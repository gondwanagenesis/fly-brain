/* Minimal SIMD abstraction: one arithmetic body, three instruction sets.
 *
 * WHY
 * ---
 * The multi-model kernel needs nine neuron models x three ISAs (AVX-512, AVX2,
 * portable scalar) = 27 sweep loops. Writing those by hand would be 27 chances
 * to make an arithmetic typo that only shows up on hardware we cannot test, and
 * the repo's whole discipline is that the three paths must be provably
 * identical. So each model is written ONCE against the macros below, and
 * `sweep_template.h` is included three times with different definitions.
 *
 * RULES THAT KEEP THE THREE PATHS BIT-IDENTICAL
 * ---------------------------------------------
 * 1. Every macro maps to exactly one IEEE-754 operation with one rounding.
 *    There is no macro that is an FMA on one path and a mul+add on another --
 *    VFMA is fused everywhere (including scalar, via fmaf) and VMUL/VADD are
 *    never contracted (the build passes -ffp-contract=off).
 * 2. Lane count never changes the arithmetic. Each lane computes the same
 *    expression on the same inputs; no horizontal ops, no reductions, no
 *    reassociation. Vector width is therefore a pure performance knob.
 * 3. Masks select, they do not arithmetically blend. VSEL is a blend
 *    instruction, not `m*a + (1-m)*b`, so no rounding enters through the mask.
 *
 * The one deliberate exception is model 0 (LIF forward-Euler), which must
 * reproduce ATen's mixed FMA/non-FMA seam to stay bit-identical to the PyTorch
 * reference. That model keeps its own hand-written path in the kernel and does
 * not go through this header.
 */
#ifndef FLY_SIMD_H
#define FLY_SIMD_H

#include <stdint.h>
#include <math.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#define FLY_X86 1
#include <immintrin.h>
#endif

/* ===================================================================== */
/* Scalar (W = 1). Also the reference definition of every operation.      */
/* ===================================================================== */
#define S_W            1
#define S_VF           float
#define S_VM           int
#define S_VSET1(x)     (x)
#define S_VLOAD(p)     (*(p))
#define S_VSTORE(p, x) (*(p) = (x))
#define S_VADD(a, b)   ((a) + (b))
#define S_VSUB(a, b)   ((a) - (b))
#define S_VMUL(a, b)   ((a) * (b))
#define S_VDIV(a, b)   ((a) / (b))
#define S_VFMA(a, b, c) fmaf((a), (b), (c))     /* a*b + c, ONE rounding */
#define S_VMFALSE      0
#define S_VMIN(a, b)   ((a) < (b) ? (a) : (b))
#define S_VMAX(a, b)   ((a) > (b) ? (a) : (b))
#define S_VGT(a, b)    ((a) > (b))
#define S_VGE(a, b)    ((a) >= (b))
#define S_VLT(a, b)    ((a) < (b))
#define S_VAND(a, b)   ((a) & (b))
#define S_VANDN(a, b)  ((~(a)) & (b))
#define S_VOR(a, b)    ((a) | (b))
#define S_VSEL(m, a, b) ((m) ? (a) : (b))       /* mask ? a : b */
#define S_VBITS(m)     ((uint32_t)((m) & 1))
#define S_VANY(m)      ((m) != 0)
#define S_VMASK_LOW(k) ((k) > 0)
#define S_VLOADM(p, m) ((m) ? *(p) : 0.0f)
#define S_VSTOREM(p, m, x) do { if (m) *(p) = (x); } while (0)
/* 2^n by direct exponent construction; n is an integer-valued float in
 * [-126, 127]. Used by the exp kernel's final scaling. */
static inline float s_vscalef(float x, float n)
{
    union { uint32_t u; float f; } p;
    int e = (int)n;
    if (e < -126) { p.u = 0u; }
    else if (e > 127) { p.u = 0x7F800000u; }
    else { p.u = (uint32_t)(e + 127) << 23; }
    return x * p.f;
}
#define S_VSCALEF(x, n) s_vscalef((x), (n))
#define S_VROUND(x)     nearbyintf(x)           /* round-to-nearest-even */

/* ===================================================================== */
/* AVX2 (W = 8). No mask registers: comparisons yield all-ones lane masks */
/* and selection is blendv.                                              */
/* ===================================================================== */
#ifdef FLY_X86
#define A2_W            8
#define A2_VF           __m256
#define A2_VM           __m256
#define A2_VSET1(x)     _mm256_set1_ps(x)
#define A2_VLOAD(p)     _mm256_loadu_ps(p)
#define A2_VSTORE(p, x) _mm256_storeu_ps((p), (x))
#define A2_VADD(a, b)   _mm256_add_ps((a), (b))
#define A2_VSUB(a, b)   _mm256_sub_ps((a), (b))
#define A2_VMUL(a, b)   _mm256_mul_ps((a), (b))
#define A2_VDIV(a, b)   _mm256_div_ps((a), (b))
#define A2_VFMA(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#define A2_VMFALSE      _mm256_setzero_ps()
#define A2_VMIN(a, b)   _mm256_min_ps((a), (b))
#define A2_VMAX(a, b)   _mm256_max_ps((a), (b))
#define A2_VGT(a, b)    _mm256_cmp_ps((a), (b), _CMP_GT_OQ)
#define A2_VGE(a, b)    _mm256_cmp_ps((a), (b), _CMP_GE_OQ)
#define A2_VLT(a, b)    _mm256_cmp_ps((a), (b), _CMP_LT_OQ)
#define A2_VAND(a, b)   _mm256_and_ps((a), (b))
#define A2_VANDN(a, b)  _mm256_andnot_ps((a), (b))
#define A2_VOR(a, b)    _mm256_or_ps((a), (b))
#define A2_VSEL(m, a, b) _mm256_blendv_ps((b), (a), (m))
#define A2_VBITS(m)     ((uint32_t)_mm256_movemask_ps(m))
#define A2_VANY(m)      (_mm256_movemask_ps(m) != 0)
#define A2_VMASK_LOW(k) _mm256_castsi256_ps(_mm256_cmpgt_epi32(                \
        _mm256_set1_epi32(k), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)))
#define A2_VLOADM(p, m) _mm256_maskload_ps((p), _mm256_castps_si256(m))
#define A2_VSTOREM(p, m, x) _mm256_maskstore_ps((p), _mm256_castps_si256(m), (x))
#define A2_VROUND(x)    _mm256_round_ps((x), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
/* No vscalefps before AVX-512, so build 2^n by shifting the biased exponent
 * into place. Clamped exactly as the scalar version is. */
__attribute__((target("avx2,fma")))
static inline __m256 a2_vscalef(__m256 x, __m256 n)
{
    __m256i e = _mm256_cvtps_epi32(n);
    e = _mm256_max_epi32(e, _mm256_set1_epi32(-127));
    e = _mm256_min_epi32(e, _mm256_set1_epi32(128));
    __m256i b = _mm256_slli_epi32(_mm256_add_epi32(e, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(x, _mm256_castsi256_ps(b));
}
#define A2_VSCALEF(x, n) a2_vscalef((x), (n))

/* ===================================================================== */
/* AVX-512 (W = 16). Real mask registers, and vscalefps does 2^n in one   */
/* instruction with correct handling of the whole exponent range.        */
/* ===================================================================== */
#define A5_W            16
#define A5_VF           __m512
#define A5_VM           __mmask16
#define A5_VSET1(x)     _mm512_set1_ps(x)
#define A5_VLOAD(p)     _mm512_loadu_ps(p)
#define A5_VSTORE(p, x) _mm512_storeu_ps((p), (x))
#define A5_VADD(a, b)   _mm512_add_ps((a), (b))
#define A5_VSUB(a, b)   _mm512_sub_ps((a), (b))
#define A5_VMUL(a, b)   _mm512_mul_ps((a), (b))
#define A5_VDIV(a, b)   _mm512_div_ps((a), (b))
#define A5_VFMA(a, b, c) _mm512_fmadd_ps((a), (b), (c))
#define A5_VMFALSE      ((__mmask16)0)
#define A5_VMIN(a, b)   _mm512_min_ps((a), (b))
#define A5_VMAX(a, b)   _mm512_max_ps((a), (b))
#define A5_VGT(a, b)    _mm512_cmp_ps_mask((a), (b), _CMP_GT_OQ)
#define A5_VGE(a, b)    _mm512_cmp_ps_mask((a), (b), _CMP_GE_OQ)
#define A5_VLT(a, b)    _mm512_cmp_ps_mask((a), (b), _CMP_LT_OQ)
#define A5_VAND(a, b)   ((__mmask16)((a) & (b)))
#define A5_VANDN(a, b)  ((__mmask16)((~(a)) & (b)))
#define A5_VOR(a, b)    ((__mmask16)((a) | (b)))
#define A5_VSEL(m, a, b) _mm512_mask_blend_ps((m), (b), (a))
#define A5_VBITS(m)     ((uint32_t)(m))
#define A5_VANY(m)      ((m) != 0)
#define A5_VMASK_LOW(k) ((__mmask16)((1u << (k)) - 1u))
#define A5_VLOADM(p, m) _mm512_maskz_loadu_ps((m), (p))
#define A5_VSTOREM(p, m, x) _mm512_mask_storeu_ps((p), (m), (x))
#define A5_VSCALEF(x, n) _mm512_scalef_ps((x), (n))
#define A5_VROUND(x)    _mm512_roundscale_ps((x), 0)   /* nearest-even */
#endif /* FLY_X86 */

#endif /* FLY_SIMD_H */
