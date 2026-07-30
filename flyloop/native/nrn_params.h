/* Parameter block shared by every neuron model, and the model id enum.
 *
 * One flat struct rather than a union, because the whole thing is 200 bytes,
 * it is loaded once per sweep and hoisted into registers, and a union would
 * buy nothing but a chance to read the wrong member. Python fills it in
 * `flyloop/models.py`; the field order here and there must match exactly.
 *
 * EVERY constant is precomputed on the Python side in float64 and narrowed to
 * float32 once. Nothing in the hot loop computes an exponential, a division or
 * a reciprocal of a parameter -- if a model's update contains `dt/C`, the
 * struct carries `dt_over_C`, not `dt` and `C`.
 */
#ifndef FLY_NRN_PARAMS_H
#define FLY_NRN_PARAMS_H

#include <stdint.h>

enum {
    M_LIF_EULER = 0,   /* Shiu et al. as shipped: forward Euler. The reference. */
    M_LIF_EXACT = 1,   /* Rotter & Diesmann 1999 propagator. Same shape, exact. */
    M_IZH       = 2,   /* Izhikevich 2003 quadratic + recovery variable        */
    M_ADEX      = 3,   /* Brette & Gerstner 2005 adaptive exponential          */
    M_EIF       = 4,   /* Fourcaud-Trocme et al. 2003 exponential I&F          */
    M_QIF       = 5,   /* quadratic I&F (Ermentrout & Kopell theta neuron)     */
    M_RAF       = 6,   /* Izhikevich 2001 resonate-and-fire (exactly rotated)  */
    M_HH        = 7,   /* Hodgkin & Huxley 1952, Rush-Larsen gates             */
    M_GLIF      = 8,   /* adaptive-threshold LIF (Allen Institute GLIF-3 form) */
    M_COUNT     = 9
};

/* Number of auxiliary per-neuron state arrays each model needs, beyond the
 * universal (v, g). aux is laid out as (n_aux, N), contiguous per variable. */
static const int NRN_NAUX[M_COUNT] = {
    0,  /* LIF euler                */
    0,  /* LIF exact                */
    1,  /* izh: u                   */
    1,  /* adex: w                  */
    0,  /* eif                      */
    0,  /* qif                      */
    1,  /* raf: y (imaginary part)  */
    4,  /* hh: m, h, n, armed       */
    1,  /* glif: theta              */
};

/* HH's fourth slot is not a gating variable. It is the spike DETECTOR's
 * hysteresis latch: HH has no reset, so a spike is an upward crossing of a
 * real action potential, and without a latch a single spike lasting several
 * substeps above threshold would be counted several times. `armed` is 1 when
 * the neuron has repolarised below v_reset since its last detected spike. */

typedef struct {
    /* --- universal --------------------------------------------------- */
    float v_rest, v_reset, v_th, v_peak;
    float g_decay;      /* per-step synaptic decay factor                 */
    float k_in;         /* synaptic gain into this model's voltage scale  */
    int32_t reset_g;    /* 1 = spiking zeroes g (upstream LIF semantics)  */
    int32_t n_sub;      /* substeps per dt (>=1); dt_sub folded into the
                         * constants below, so the loop just runs n_sub
                         * times over the same arithmetic                 */

    /* --- 0: LIF forward Euler ---------------------------------------- */
    float c_decay;      /* 1 - dt/tau_syn                                 */
    float c_mem;        /* dt/tau_mem                                     */

    /* --- 1: LIF exact ------------------------------------------------- */
    float alpha_m;      /* exp(-dt/tau_mem)                               */
    float alpha_s;      /* exp(-dt/tau_syn) == alpha_m^4 when the ratio is 4 */
    float p_vg;         /* tau_s/(tau_m - tau_s) * (alpha_m - alpha_s)    */

    /* --- 2: Izhikevich ------------------------------------------------ */
    float izh_p2, izh_p1, izh_p0;   /* 0.04, 5, 140 -- named so the shape
                                     * can be retuned without touching C  */
    float izh_a, izh_b, izh_c, izh_d;
    float izh_dt;                   /* substep length                     */

    /* --- 3/4: AdEx and EIF -------------------------------------------- */
    float ad_dtC;       /* dt_sub / C                                     */
    float ad_gL;        /* leak conductance                               */
    float ad_EL;        /* leak reversal                                  */
    float ad_VT;        /* exponential-onset threshold                    */
    float ad_inv_DT;    /* 1 / Delta_T                                    */
    float ad_gLDT;      /* gL * Delta_T                                   */
    float ad_wdecay;    /* exp(-dt_sub / tau_w)                           */
    float ad_a;         /* subthreshold adaptation coupling               */
    float ad_b;         /* spike-triggered adaptation increment           */
    float elide_tiny;   /* certified-elision threshold, normally 2^-25.
                         * SET IT TO ZERO to disable elision entirely: the
                         * test becomes `bound >= 0`, which is always true, so
                         * the exponential is computed on every lane. That is
                         * how verify_models.py turns "the elision is exact"
                         * from a claim in a comment into a bit-comparison
                         * against the same kernel with it switched off.     */

    /* --- 5: QIF -------------------------------------------------------- */
    float qif_k;        /* k, where dv/dt = k (v-v_rest)(v-v_c) + I       */
    float qif_vc;       /* critical voltage                               */

    /* --- 6: resonate-and-fire ------------------------------------------ */
    float raf_rr, raf_ri;  /* exp(b dt) * (cos w dt, sin w dt)            */
    float raf_mr, raf_mi;  /* (exp(lambda dt) - 1)/lambda, lambda = b+i*w.
                            * The EXACT response to a constant input held
                            * over the step. Adding the input undivided --
                            * as an impulse per step rather than a rate --
                            * makes the model disagree with its own
                            * published ODE by a factor of dt, which is
                            * exactly what the Brian 2 cross-check found. */

    /* --- 7: Hodgkin-Huxley --------------------------------------------- */
    float hh_dtC;       /* dt_sub / C_m                                   */
    float hh_dt;        /* dt_sub, for the gate time constants            */
    float hh_gNa, hh_gK, hh_gL;
    float hh_ENa, hh_EK, hh_EL;
    float hh_shift;     /* v_model = v_hh + shift, to sit HH's -65 mV rest
                         * on the connectome's -52 mV scale               */

    /* --- 8: GLIF (adaptive threshold) -----------------------------------
     * The membrane reuses alpha_m / alpha_s / p_vg, so GLIF differs from
     * LIF-exact only in that the threshold moves. */
    float gl_thdecay;   /* exp(-dt/tau_theta)                             */
    float gl_th_inf;    /* asymptotic threshold                           */
    float gl_th_jump;   /* per-spike threshold increment                  */
} nrn_params;

#endif /* FLY_NRN_PARAMS_H */
