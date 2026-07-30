/* One sweep function per instruction set, covering every neuron model.
 *
 * Included three times from nrn_kernel.c with the generic op macros (VADD,
 * VFMA, VSEL, ...) bound to scalar, AVX2 and AVX-512 respectively, and FN(x)
 * bound to a distinct name suffix. Every model body below is therefore written
 * exactly once and compiled three times, which is what makes "all three ISA
 * paths produce identical results" a structural property rather than something
 * to keep re-checking by hand.
 *
 * The model switch is OUTSIDE the neuron loop, so each model gets a fully
 * specialised loop with no per-iteration branch and no indirect call.
 *
 * WHAT IS SHARED AND WHAT IS PER-MODEL
 * ------------------------------------
 * Shared, and untouched by the choice of model: the 1.8 ms sparse delay line,
 * the event-driven fan-out, the refractory gate, 16-neuron tile skipping, the
 * chunk partition and the thread pool. Those are properties of the network and
 * the schedule, not of the membrane equation. A model supplies only how
 * (v, g, aux) advance over one dt, when a spike is declared, and what the reset
 * does. That separation is why adding a model costs one macro here plus one
 * entry in models.py, and why none of the connectome machinery is duplicated.
 *
 * TAILS ARE MASKED, NOT PEELED
 * ----------------------------
 * A tile is 16 neurons except for the one ATen-tail tile per chunk, so the
 * loop would otherwise need a scalar peel per model per ISA. Masked
 * load/store instead: lanes past the end are loaded as zero, computed on, and
 * discarded on store. The arithmetic in a masked-off lane is garbage by
 * construction and never reaches memory or the spike bitset.
 */

/* ------------------------------------------------------------------ */
/* Vectorised expf.                                                    */
/*                                                                     */
/* Cody-Waite range reduction + degree-6 minimax polynomial + exact     */
/* power-of-two scaling: the classic Cephes expf, expressed in the      */
/* abstraction so all three ISAs execute the same operations in the     */
/* same order and therefore agree bit-for-bit.                         */
/*                                                                     */
/* The input is clamped to [-87, 88] first. That is not an accuracy     */
/* compromise -- expf underflows to zero below -87.34 and overflows     */
/* above 88.72, so the clamp only pins values that are already          */
/* saturated -- and it keeps the reduced exponent inside [-126, 127],   */
/* which is where the scalar, AVX2 and AVX-512 scaling paths are        */
/* provably the same operation.                                        */
/*                                                                     */
/* Accuracy is not asserted here, it is MEASURED: verify_exp.py walks   */
/* every float32 in the reachable argument range against a float64      */
/* reference and reports the worst ULP seen.                           */
/* ------------------------------------------------------------------ */
FN_ATTR
static inline VF FN(fly_exp)(VF x)
{
    x = VMIN(VMAX(x, VSET1(-87.0f)), VSET1(88.0f));

    const VF n = VROUND(VMUL(x, VSET1(1.44269504088896341f)));

    /* n*ln2 is not representable, so subtract it in two exact-ish pieces:
     * 0.693359375 is exact in binary32 and carries the high bits. */
    VF r = VFMA(n, VSET1(-0.693359375f), x);
    r = VFMA(n, VSET1(2.12194440e-4f), r);

    VF p = VSET1(1.9875691500e-4f);
    p = VFMA(p, r, VSET1(1.3981999507e-3f));
    p = VFMA(p, r, VSET1(8.3334519073e-3f));
    p = VFMA(p, r, VSET1(4.1665795894e-2f));
    p = VFMA(p, r, VSET1(1.6666665459e-1f));
    p = VFMA(p, r, VSET1(5.0000001201e-1f));

    VF y = VFMA(p, VMUL(r, r), r);
    y = VADD(y, VSET1(1.0f));
    return VSCALEF(y, n);
}

/* Rush & Larsen 1978, in the (alpha, beta) parameterisation of a
 * Hodgkin-Huxley gate:
 *     y_inf = a/(a+b),   tau = 1/(a+b)
 *     y <- y_inf + (y - y_inf) exp(-dt (a+b))
 * Exact for frozen v over the substep, and unconditionally stable -- which is
 * the whole reason HH is tractable here at all. */
FN_ATTR
static inline VF FN(rush_larsen)(VF y, VF a, VF b, VF dt)
{
    const VF s = VADD(a, b);
    const VF yinf = VDIV(a, s);
    const VF dec = FN(fly_exp)(VMUL(VSUB(VSET1(0.0f), s), dt));
    return VFMA(VSUB(y, yinf), dec, yinf);
}

/* x / (1 - exp(-x/k)) with the removable singularity at x = 0 handled.
 * Every serious HH implementation carries this guard; NEURON calls it vtrap.
 * The limit is k, approached linearly, so inside the guard band the expansion
 * k + x/2 is both more accurate than the naive quotient and branch-free. */
FN_ATTR
static inline VF FN(vtrap)(VF x, VF inv_k, VF k)
{
    const VM near0 = VLT(VMUL(x, x), VSET1(1e-8f));
    const VF d = VSUB(VSET1(1.0f), FN(fly_exp)(VMUL(VSUB(VSET1(0.0f), x), inv_k)));
    /* the divisor is forced to 1 in the guard band so the division cannot
     * produce a 0/0 NaN that a later select would have to clean up */
    const VF safe = VSEL(near0, VSET1(1.0f), d);
    return VSEL(near0, VFMA(x, VSET1(0.5f), k), VDIV(x, safe));
}

/* ------------------------------------------------------------------ */
/* Model bodies.                                                       */
/*                                                                     */
/* Each expands inside a loop that has already loaded `vv` (membrane)   */
/* and `gi` (synaptic state) for VW neurons at index i, and must leave  */
/* `vn`, `gn` and `spm` (spike mask) defined.                          */
/* ------------------------------------------------------------------ */

/* --- 1: LIF, exact propagator (Rotter & Diesmann 1999) --------------
 *
 * u' = alpha_m u + p_vg g   and   g' = alpha_s g,  with u = v - v_rest.
 * This is the matrix exponential of the (u, g) system, so it is EXACT over the
 * interval between synaptic arrivals rather than a first-order approximation
 * of it.
 *
 * It is also CHEAPER than the forward-Euler form it replaces: two FMAs and a
 * multiply, against Euler's subtract, add, FMA and multiply. Being both more
 * accurate and fewer operations is unusual enough to state plainly -- there is
 * no accuracy/speed trade to make here, Euler is simply dominated. */
#define BODY_LIF_EXACT                                                        \
    const VF u_ = VSUB(vv, c_vrest);                                          \
    VF vn = VFMA(gi, c_pvg, VFMA(u_, c_alpham, c_vrest));                     \
    VF gn = VMUL(gi, c_alphas);                                               \
    VM spm = VGT(vn, c_vth);                                                  \
    vn = VSEL(spm, c_vreset, vn);                                             \
    gn = VSEL(spm, c_zero, gn);

/* --- 2: Izhikevich 2003 ---------------------------------------------
 *
 * v' = 0.04 v^2 + 5 v + 140 - u + I;  u' = a(b v - u);  v >= 30 -> v=c, u+=d.
 *
 * The quadratic is a Horner chain, i.e. two FMAs, so the "nonlinear" model
 * costs about what the linear one does. Izhikevich's cost is the extra state
 * variable's memory traffic, not its arithmetic -- the opposite of the usual
 * intuition, and the reason it benchmarks within a few percent of LIF here. */
#define BODY_IZH                                                              \
    VF uu = VLOADM(a0 + i, lanem);                                            \
    VF vn = vv;                                                               \
    const VF Ii = VMUL(gi, c_kin);                                            \
    VM spm = VMFALSE;                                                         \
    for (int s_ = 0; s_ < nsub; ++s_) {                                       \
        VF q_ = VFMA(c_izhp2, vn, c_izhp1);                                   \
        q_ = VFMA(q_, vn, c_izhp0);                                           \
        q_ = VADD(VSUB(q_, uu), Ii);                                          \
        const VF du_ = VMUL(c_izhadt, VSUB(VMUL(c_izhb, vn), uu));            \
        vn = VFMA(q_, c_subdt, vn);                                           \
        uu = VADD(uu, du_);                                                   \
        const VM sp_ = VGE(vn, c_vpeak);                                       \
        spm = VOR(spm, sp_);                                                  \
        vn = VSEL(sp_, c_izhc, vn);                                            \
        uu = VSEL(sp_, VADD(uu, c_izhd), uu);                                  \
    }                                                                         \
    VSTOREM(a0 + i, lanem, uu);                                               \
    VF gn = VMUL(gi, c_gdecay);                                               \
    if (reset_g) gn = VSEL(spm, c_zero, gn);

/* --- 3: AdEx (Brette & Gerstner 2005) --------------------------------
 *
 * C v' = -gL(v-EL) + gL dT exp((v-VT)/dT) - w + I ;  tau_w w' = a(v-EL) - w
 * v > v_peak -> v = v_reset, w += b.
 *
 * Two things make this cheap enough to run whole-brain.
 *
 * 1. The adaptation variable is advanced by its EXACT solution for frozen v,
 *    w <- w_inf + (w - w_inf) e^{-dt/tau_w}, not by forward Euler. That is
 *    Rush & Larsen's trick from cardiac electrophysiology, and it costs one
 *    FMA because the decay factor is a constant.
 *
 * 2. CERTIFIED EXPONENTIAL ELISION. The exponential is roughly twelve of the
 *    model's twenty operations, and in this connectome most neurons sit near
 *    rest most of the time, where that term is far below the rounding floor of
 *    the update. So the body first computes a RIGOROUS UPPER BOUND on the term
 *    -- exp(x) <= 2^(round(x log2 e) + 1), one round and one scalef -- and
 *    skips the polynomial entirely when no lane in the vector can matter.
 *
 *    The elision is exact, and the argument is short. Write the update as
 *    v_new = fma(dtC, S + E, v) with S = -gL(v-EL) - w + I and E the
 *    exponential term. Under round-to-nearest, |E| < ulp(S)/2 implies
 *    fl(S + E) == S identically, because S + E then lies strictly inside S's
 *    rounding interval; everything downstream sees the same bits. Since
 *    ulp(S) >= |S| 2^-24, testing |E_bound| < |S| 2^-25 is sufficient, and
 *    that is one absolute value, one multiply and one compare. When it passes
 *    on every lane the polynomial does not execute.
 *
 *    This removes work without removing information -- the elided term does
 *    not merely round away after the fact, it is proven to round away before
 *    being computed. */
#define BODY_ADEX                                                             \
    VF ww = VLOADM(a0 + i, lanem);                                            \
    VF vn = vv;                                                               \
    const VF Ii = VMUL(gi, c_kin);                                            \
    VM spm = VMFALSE;                                                         \
    for (int s_ = 0; s_ < nsub; ++s_) {                                       \
        const VF S_ = VSUB(VFMA(c_adngl, VSUB(vn, c_adel), Ii), ww);          \
        const VF x_ = VMUL(VSUB(vn, c_advt), c_adinvdt);                      \
        const VF nb_ = VADD(VROUND(VMUL(x_, VSET1(1.44269504088896341f))),    \
                            VSET1(1.0f));                                     \
        const VM need_ = VGE(VSCALEF(c_adgldt, nb_),                          \
                             VMUL(VMAX(S_, VSUB(c_zero, S_)), c_elide));       \
        VF E_ = c_zero;                                                       \
        if (VANY(need_))                                                      \
            E_ = VSEL(need_, VMUL(c_adgldt, FN(fly_exp)(x_)), c_zero);        \
        vn = VFMA(c_addtc, VADD(S_, E_), vn);                                 \
        /* The spike test comes BEFORE the adaptation update, and w is        \
         * integrated toward the CUTOFF rather than toward a divergent        \
         * v. Without that, a membrane that has just overshot v_peak          \
         * gives winf = a*(inf - E_L) = inf, and the very next line           \
         * evaluates fma(w - inf, decay, inf) = -inf + inf = NaN. w is        \
         * then NaN forever, S is NaN, and the neuron falls silent            \
         * because NaN fails every ordered comparison -- a whole              \
         * population going quiet from one arithmetic edge case.              \
         * Clamping to v_peak changes no sub-threshold trajectory: it         \
         * only applies on the sub-step that is already declared a            \
         * spike, where the model's own dynamics are undefined and the        \
         * cutoff is what stands in for them. */                              \
        const VM sp_ = VGT(vn, c_vpeak);                                      \
        spm = VOR(spm, sp_);                                                  \
        const VF vw_ = VSEL(sp_, c_vpeak, vn);                                \
        const VF winf_ = VMUL(c_ada, VSUB(vw_, c_adel));                      \
        ww = VFMA(VSUB(ww, winf_), c_adwdec, winf_);                          \
        ww = VSEL(sp_, VADD(ww, c_adb), ww);                                  \
        vn = VSEL(sp_, c_vreset, vn);                                         \
    }                                                                         \
    VSTOREM(a0 + i, lanem, ww);                                               \
    VF gn = VMUL(gi, c_gdecay);                                               \
    if (reset_g) gn = VSEL(spm, c_zero, gn);

/* --- 4: EIF (Fourcaud-Trocme et al. 2003) ----------------------------
 * AdEx with the adaptation removed: the spike-initiation nonlinearity on its
 * own. Same certified elision, one fewer state array. */
#define BODY_EIF                                                              \
    VF vn = vv;                                                               \
    const VF Ii = VMUL(gi, c_kin);                                            \
    VM spm = VMFALSE;                                                         \
    for (int s_ = 0; s_ < nsub; ++s_) {                                       \
        const VF S_ = VFMA(c_adngl, VSUB(vn, c_adel), Ii);                    \
        const VF x_ = VMUL(VSUB(vn, c_advt), c_adinvdt);                      \
        const VF nb_ = VADD(VROUND(VMUL(x_, VSET1(1.44269504088896341f))),    \
                            VSET1(1.0f));                                     \
        const VM need_ = VGE(VSCALEF(c_adgldt, nb_),                          \
                             VMUL(VMAX(S_, VSUB(c_zero, S_)), c_elide));       \
        VF E_ = c_zero;                                                       \
        if (VANY(need_))                                                      \
            E_ = VSEL(need_, VMUL(c_adgldt, FN(fly_exp)(x_)), c_zero);        \
        vn = VFMA(c_addtc, VADD(S_, E_), vn);                                 \
        const VM sp_ = VGT(vn, c_vpeak);                                      \
        spm = VOR(spm, sp_);                                                  \
        vn = VSEL(sp_, c_vreset, vn);                                         \
    }                                                                         \
    VF gn = VMUL(gi, c_gdecay);                                               \
    if (reset_g) gn = VSEL(spm, c_zero, gn);

/* --- 5: QIF / theta neuron -------------------------------------------
 * v' = k (v - v_rest)(v - v_c) + I. Two FMAs, no transcendental. This is the
 * normal form every type-I excitable neuron reduces to near its saddle-node
 * bifurcation (Ermentrout & Kopell 1986), so it is the cheapest model here
 * that still has a genuine spike-generating nonlinearity. */
#define BODY_QIF                                                              \
    VF vn = vv;                                                               \
    const VF Ii = VMUL(gi, c_kin);                                            \
    VM spm = VMFALSE;                                                         \
    for (int s_ = 0; s_ < nsub; ++s_) {                                       \
        const VF d_ = VMUL(VSUB(vn, c_vrest), VSUB(vn, c_qifvc));             \
        vn = VFMA(VFMA(d_, c_qifk, Ii), c_subdt, vn);                         \
        const VM sp_ = VGT(vn, c_vpeak);                                      \
        spm = VOR(spm, sp_);                                                  \
        vn = VSEL(sp_, c_vreset, vn);                                         \
    }                                                                         \
    VF gn = VMUL(gi, c_gdecay);                                               \
    if (reset_g) gn = VSEL(spm, c_zero, gn);

/* --- 6: resonate-and-fire (Izhikevich 2001) --------------------------
 * z' = (b + i w) z + I, with z = x + i y. The homogeneous flow is a scaled
 * rotation, so one step is a 2x2 matrix multiply with CONSTANT entries
 * e^{b dt}(cos w dt, sin w dt): exactly integrable, four multiplies and two
 * adds, no transcendental at run time. The only sub-threshold resonator here
 * that is exact by construction rather than by small dt. */
#define BODY_RAF                                                              \
    const VF yy = VLOADM(a0 + i, lanem);                                      \
    const VF Ii = VMUL(gi, c_kin);                                            \
    VF vn = VFMA(vv, c_rafrr, VFMA(yy, VSUB(c_zero, c_rafri),                 \
                                   VMUL(c_rafmr, Ii)));                       \
    VF yn = VFMA(vv, c_rafri, VFMA(yy, c_rafrr,                               \
                                   VMUL(c_rafmi, Ii)));                       \
    VM spm = VGT(yn, c_vth);                                                  \
    vn = VSEL(spm, c_vreset, vn);                                             \
    yn = VSEL(spm, c_zero, yn);                                               \
    VSTOREM(a0 + i, lanem, yn);                                               \
    VF gn = VMUL(gi, c_gdecay);                                               \
    if (reset_g) gn = VSEL(spm, c_zero, gn);

/* --- 8: GLIF, adaptive threshold (Allen Institute GLIF-3 form) -------
 * Membrane and threshold are both linear and autonomous between spikes, so
 * BOTH advance by their exact exponential solution. Spike-frequency adaptation
 * for the price of one extra stream and two FMAs, with the membrane equation
 * left exactly as in LIF-exact. */
#define BODY_GLIF                                                             \
    VF th = VLOADM(a0 + i, lanem);                                            \
    const VF u_ = VSUB(vv, c_vrest);                                          \
    VF vn = VFMA(gi, c_pvg, VFMA(u_, c_alpham, c_vrest));                     \
    th = VFMA(VSUB(th, c_glthinf), c_glthdec, c_glthinf);                     \
    VM spm = VGT(vn, th);                                                     \
    vn = VSEL(spm, c_vreset, vn);                                             \
    th = VSEL(spm, VADD(th, c_glthjmp), th);                                  \
    VSTOREM(a0 + i, lanem, th);                                               \
    VF gn = VMUL(gi, c_alphas);                                               \
    if (reset_g) gn = VSEL(spm, c_zero, gn);

/* --- 7: Hodgkin & Huxley 1952 ---------------------------------------
 *
 * The only model here with no reset: a spike is a threshold crossing of a real
 * action potential, so it is detected with two-level hysteresis carried in a
 * fourth auxiliary variable (`armed`). A neuron re-arms once it repolarises
 * below v_reset, which is what stops one action potential being counted as
 * several.
 *
 * The gates use RUSH & LARSEN 1978 exponential integration, borrowed from
 * cardiac electrophysiology where it is the standard way to keep HH-type gates
 * stable at usable step sizes. Forward Euler on m needs dt well under 10 us
 * because tau_m falls to ~0.05 ms during the upstroke; Rush-Larsen is
 * unconditionally stable for the gates, which is what makes 25 us substeps
 * viable and the model tractable at connectome scale at all.
 *
 * There is no elision here: at rest the gates sit at a non-trivial steady
 * state, so nothing is negligible. HH pays full price every step, six
 * exponentials per neuron per substep, and the benchmark says so. */
#define BODY_HH                                                               \
    VF mm = VLOADM(a0 + i, lanem);                                            \
    VF hg = VLOADM(a1 + i, lanem);                                            \
    VF nn = VLOADM(a2 + i, lanem);                                            \
    VF ar = VLOADM(a3 + i, lanem);                                            \
    VF vn = vv;                                                               \
    const VF Ii = VMUL(gi, c_kin);                                            \
    VM spm = VMFALSE;                                                         \
    for (int s_ = 0; s_ < nsub; ++s_) {                                       \
        const VF vh = VSUB(vn, c_hhshift);        /* HH's own -65 mV scale */ \
        const VF am = VMUL(VSET1(0.1f),                                       \
            FN(vtrap)(VADD(vh, VSET1(40.0f)), VSET1(0.1f), VSET1(10.0f)));    \
        const VF bm = VMUL(VSET1(4.0f), FN(fly_exp)(                          \
            VMUL(VADD(vh, VSET1(65.0f)), VSET1(-0.0555555556f))));            \
        const VF ah = VMUL(VSET1(0.07f), FN(fly_exp)(                         \
            VMUL(VADD(vh, VSET1(65.0f)), VSET1(-0.05f))));                    \
        const VF bh = VDIV(VSET1(1.0f), VADD(VSET1(1.0f), FN(fly_exp)(        \
            VMUL(VADD(vh, VSET1(35.0f)), VSET1(-0.1f)))));                    \
        const VF an = VMUL(VSET1(0.01f),                                      \
            FN(vtrap)(VADD(vh, VSET1(55.0f)), VSET1(0.1f), VSET1(10.0f)));    \
        const VF bn = VMUL(VSET1(0.125f), FN(fly_exp)(                        \
            VMUL(VADD(vh, VSET1(65.0f)), VSET1(-0.0125f))));                  \
        mm = FN(rush_larsen)(mm, am, bm, c_hhdt);                             \
        hg = FN(rush_larsen)(hg, ah, bh, c_hhdt);                             \
        nn = FN(rush_larsen)(nn, an, bn, c_hhdt);                             \
        const VF m3 = VMUL(VMUL(mm, mm), mm);                                 \
        const VF n2 = VMUL(nn, nn);                                           \
        VF I_ = VMUL(VMUL(c_hhgna, VMUL(m3, hg)), VSUB(c_hhena, vh));         \
        I_ = VFMA(VMUL(c_hhgk, VMUL(n2, n2)), VSUB(c_hhek, vh), I_);          \
        I_ = VFMA(c_hhgl, VSUB(c_hhel, vh), I_);                              \
        vn = VFMA(c_hhdtc, VADD(I_, Ii), vn);                                 \
        const VM cross_ = VAND(VGT(vn, c_vth), VGT(ar, VSET1(0.5f)));         \
        spm = VOR(spm, cross_);                                               \
        ar = VSEL(cross_, c_zero,                                             \
                  VSEL(VLT(vn, c_vreset), VSET1(1.0f), ar));                  \
    }                                                                         \
    VSTOREM(a0 + i, lanem, mm);                                               \
    VSTOREM(a1 + i, lanem, hg);                                               \
    VSTOREM(a2 + i, lanem, nn);                                               \
    VSTOREM(a3 + i, lanem, ar);                                               \
    VF gn = VMUL(gi, c_gdecay);                                               \
    if (reset_g) gn = VSEL(spm, c_zero, gn);

/* ------------------------------------------------------------------ */
/* The sweep. One live tile at a time, VW neurons at a time.           */
/* ------------------------------------------------------------------ */
#define RUN_TILES(BODY)                                                       \
    for (int t = t0; t < t1; ++t) {                                           \
        if (!((tile_live[t >> 6] >> (t & 63)) & 1ULL)) continue;              \
        const int lo = tile_lo[t], hi = tile_lo[t + 1];                       \
        for (int i = lo; i < hi; i += VW) {                                   \
            const int k_ = (hi - i) < VW ? (hi - i) : VW;                     \
            const VM lanem = VMASK_LOW(k_);                                   \
            const VF vv = VLOADM(v + i, lanem);                               \
            const VF gi = VLOADM(g + i, lanem);                               \
            BODY                                                              \
            spm = VAND(spm, lanem);                                           \
            VSTOREM(v + i, lanem, vn);                                        \
            VSTOREM(g + i, lanem, gn);                                        \
            const uint32_t bits_ = VBITS(spm);                                \
            if (bits_) put_bits(sp_out, i, bits_, k_, wf, wl);                \
        }                                                                     \
    }

FN_ATTR
static void FN(sweep_tiles)(int model, int t0, int t1,
                            const int32_t *tile_lo, const uint64_t *tile_live,
                            float *restrict v, float *restrict g,
                            float *restrict aux, int naux_stride,
                            uint64_t *restrict sp_out,
                            const nrn_params *P, int wf, int wl)
{
    if (t1 <= t0) return;

    /* Every parameter is splatted once, outside the loop. */
    const VF c_zero = VSET1(0.0f);
    const VF c_elide = VSET1(P->elide_tiny);   /* normally 2^-25 */
    const VF c_vrest = VSET1(P->v_rest), c_vreset = VSET1(P->v_reset);
    const VF c_vth = VSET1(P->v_th), c_vpeak = VSET1(P->v_peak);
    const VF c_gdecay = VSET1(P->g_decay), c_kin = VSET1(P->k_in);
    const VF c_alpham = VSET1(P->alpha_m), c_alphas = VSET1(P->alpha_s);
    const VF c_pvg = VSET1(P->p_vg);
    const VF c_izhp2 = VSET1(P->izh_p2), c_izhp1 = VSET1(P->izh_p1);
    const VF c_izhp0 = VSET1(P->izh_p0);
    const VF c_izhadt = VSET1(P->izh_a), c_izhb = VSET1(P->izh_b);
    const VF c_izhc = VSET1(P->izh_c), c_izhd = VSET1(P->izh_d);
    const VF c_subdt = VSET1(P->izh_dt);
    const VF c_addtc = VSET1(P->ad_dtC), c_adngl = VSET1(-P->ad_gL);
    const VF c_adel = VSET1(P->ad_EL), c_advt = VSET1(P->ad_VT);
    const VF c_adinvdt = VSET1(P->ad_inv_DT), c_adgldt = VSET1(P->ad_gLDT);
    const VF c_adwdec = VSET1(P->ad_wdecay), c_ada = VSET1(P->ad_a);
    const VF c_adb = VSET1(P->ad_b);
    const VF c_qifk = VSET1(P->qif_k), c_qifvc = VSET1(P->qif_vc);
    const VF c_rafrr = VSET1(P->raf_rr), c_rafri = VSET1(P->raf_ri);
    const VF c_rafmr = VSET1(P->raf_mr), c_rafmi = VSET1(P->raf_mi);
    const VF c_hhdtc = VSET1(P->hh_dtC), c_hhdt = VSET1(P->hh_dt);
    const VF c_hhgna = VSET1(P->hh_gNa), c_hhgk = VSET1(P->hh_gK);
    const VF c_hhgl = VSET1(P->hh_gL), c_hhena = VSET1(P->hh_ENa);
    const VF c_hhek = VSET1(P->hh_EK), c_hhel = VSET1(P->hh_EL);
    const VF c_hhshift = VSET1(P->hh_shift);
    const VF c_glthdec = VSET1(P->gl_thdecay), c_glthinf = VSET1(P->gl_th_inf);
    const VF c_glthjmp = VSET1(P->gl_th_jump);

    const int nsub = P->n_sub < 1 ? 1 : P->n_sub;
    const int reset_g = P->reset_g;
    float *const a0 = aux;
    float *const a1 = aux + naux_stride;
    float *const a2 = aux + 2 * naux_stride;
    float *const a3 = aux + 3 * naux_stride;
    (void)a1; (void)a2; (void)a3; (void)c_alphas;

    switch (model) {
    case M_LIF_EXACT: { RUN_TILES(BODY_LIF_EXACT) break; }
    case M_IZH:       { RUN_TILES(BODY_IZH)       break; }
    case M_ADEX:      { RUN_TILES(BODY_ADEX)      break; }
    case M_EIF:       { RUN_TILES(BODY_EIF)       break; }
    case M_QIF:       { RUN_TILES(BODY_QIF)       break; }
    case M_RAF:       { RUN_TILES(BODY_RAF)       break; }
    case M_HH:        { RUN_TILES(BODY_HH)        break; }
    case M_GLIF:      { RUN_TILES(BODY_GLIF)      break; }
    default: break;   /* M_LIF_EULER has its own ATen-seam-faithful path */
    }
}

#undef RUN_TILES
#undef BODY_LIF_EXACT
#undef BODY_IZH
#undef BODY_ADEX
#undef BODY_EIF
#undef BODY_QIF
#undef BODY_RAF
#undef BODY_GLIF
#undef BODY_HH
