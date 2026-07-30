"""Apply Change 1: take the refractory counter out of the dense sweep.

The sweep currently reads and writes a per-neuron fp32 `refrac` counter -- 554 KB
each way -- purely so the sparse delayed-input pass can evaluate
`gate = (refrac >= refrac_steps)`. At 1.75 spikes/step only ~40 neurons are
refractory at any instant, so a dense array is the wrong representation.

Replaced by a gate bitset (17 KB) plus a compact list of the currently-counting
neurons. Because the previous-spike bit was ALSO only used for the refrac reset,
`sp_in` leaves the sweep too, so the sweep collapses to: read v, g -> write v, g
and the spike bit. 24.25 -> 16.25 bytes per neuron.

THE COUNTDOWN ARITHMETIC is the part that is easy to get wrong. Ground truth from
the reference: `refrac = spiked_prev ? 0 : refrac + 1` gives refrac = k at step
t+1+k after a spike at t, so `gate = refrac >= refrac_steps` is CLOSED for steps
t+1..t+22 and OPENS at t+23 (verified on neuron 95808). Setting c = refrac_steps
and decrementing to zero would open at t+22 -- ONE STEP EARLY, admitting one step
of input the reference discards. The correct invariant is c = refrac_steps + 1,
decremented once at the START of each step (before the delayed pass reads the
gate), gate open at c <= 0:

    end of step t   c = 23              (spike)
    step t+1        c = 22  closed      (reference refrac = 0,  0 >= 22 false)
    step t+22       c =  1  closed      (reference refrac = 21, 21 >= 22 false)
    step t+23       c =  0  OPEN        (reference refrac = 22, 22 >= 22 true)

Stimulated neurons have refrac_steps = 0, so c = 1 and the gate reopens at t+1,
matching the reference's permanently-open gate for them.
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


def drop(text, needles):
    """Delete every line containing any needle. Reports what it removed."""
    out, removed = [], 0
    for line in text.split("\n"):
        if any(nd in line for nd in needles):
            removed += 1
            continue
        out.append(line)
    return "\n".join(out), removed


# ---------------------------------------------------------------- C kernel ---
s = C.read_text(encoding="utf-8")

# The refrac lanes, the previous-spike mask, and the refrac stores, everywhere.
s, n = drop(s, [
    "const uint64_t was = BIT_GET(sp_in, i);",
    "const float r = was ? 0.0f : refrac[i] + 1.0f;",
    "refrac[i] = r;",
    "const __mmask16 was = (__mmask16)get_bits(sp_in, i, 16);",
    "const __m512 rf = _mm512_loadu_ps(refrac + i);",
    "_mm512_mask_blend_ps(was, _mm512_add_ps(rf, vone), vzero)",
    "_mm512_storeu_ps(refrac + i, r);",
    "/* refrac + 1, zeroed where the neuron spiked last step */",
    "const uint32_t wasm = get_bits(sp_in, i, 8);",
    "expand 8 bits to an 8-lane",
    "const __m256i bidx = _mm256_setr_epi32",
    "const __m256i wasv = _mm256_and_si256",
    "const __m256 was = _mm256_castsi256_ps(",
    "_mm256_cmpeq_epi32(wasv, bidx));",
    "const __m256 rf = _mm256_loadu_ps(refrac + i);",
    "_mm256_blendv_ps(_mm256_add_ps(rf, vone), vzero, was)",
    "_mm256_storeu_ps(refrac + i, r);",
])
print(f"dropped {n} refrac/prev-spike lines")
if n < 14:
    sys.exit(f"expected >=14 dropped lines, got {n} -- anchors have drifted")

s = s.replace("const __m512 vone = _mm512_set1_ps(1.0f), vzero = _mm512_setzero_ps();",
              "const __m512 vzero = _mm512_setzero_ps();")
s = s.replace("const __m256 vone = _mm256_set1_ps(1.0f), vzero = _mm256_setzero_ps();",
              "const __m256 vzero = _mm256_setzero_ps();")

# get_bits is now unused; keep it but silence -Wall.
s = s.replace("static inline uint32_t get_bits(",
              "__attribute__((unused)) static inline uint32_t get_bits(")

# --- signatures: refrac, refrac_steps and sp_in all leave the sweep ---
s = s.replace("""                         float *restrict v, float *restrict g,
                         float *restrict refrac,
                         const uint64_t *restrict sp_in, uint64_t *restrict sp_out,""",
              """                         float *restrict v, float *restrict g,
                         uint64_t *restrict sp_out,""")
s = s.replace("""                        float *restrict v, float *restrict g,
                        float *restrict refrac,
                        const uint64_t *restrict sp_in, uint64_t *restrict sp_out,""",
              """                        float *restrict v, float *restrict g,
                        uint64_t *restrict sp_out,""")
s = s.replace("""                      float *restrict v, float *restrict g,
                      float *restrict refrac,
                      const uint64_t *restrict sp_in, uint64_t *restrict sp_out,""",
              """                      float *restrict v, float *restrict g,
                      uint64_t *restrict sp_out,""")

for a, b in (
    ("i = sweep_avx512(i, vec_end, v, g, refrac, sp_in, sp_out,",
     "i = sweep_avx512(i, vec_end, v, g, sp_out,"),
    ("i = sweep_avx2(i, vec_end, v, g, refrac, sp_in, sp_out,",
     "i = sweep_avx2(i, vec_end, v, g, sp_out,"),
    ("sweep_scalar(i, vec_end, 1, v, g, refrac, sp_in, sp_out,",
     "sweep_scalar(i, vec_end, 1, v, g, sp_out,"),
    ("sweep_scalar(vec_end, hi, 0, v, g, refrac, sp_in, sp_out,",
     "sweep_scalar(vec_end, hi, 0, v, g, sp_out,"),
    ("""    float *v, *g, *refrac;
    const uint64_t *sp_in;
    uint64_t *sp_out;""", """    float *v, *g;
    uint64_t *sp_out;"""),
    ("""    sweep_chunk(j->lo, j->hi, j->tail_w, j->v, j->g, j->refrac,
                j->sp_in, j->sp_out, j->c_decay, j->c_mem,
                j->v_rest, j->v_reset, j->v_th);""",
     """    sweep_chunk(j->lo, j->hi, j->tail_w, j->v, j->g, j->sp_out,
                j->c_decay, j->c_mem, j->v_rest, j->v_reset, j->v_th);"""),
    ("""                      float *v, float *g, float *refrac,
                      const uint64_t *sp_in, uint64_t *sp_out,""",
     """                      float *v, float *g, uint64_t *sp_out,"""),
    ("""            sweep_chunk(chunks[c], chunks[c + 1], tail_w, v, g, refrac,
                        sp_in, sp_out, c_decay, c_mem, v_rest, v_reset, v_th);""",
     """            sweep_chunk(chunks[c], chunks[c + 1], tail_w, v, g, sp_out,
                        c_decay, c_mem, v_rest, v_reset, v_th);"""),
    ("""        j->v = v; j->g = g; j->refrac = refrac;
        j->sp_in = sp_in; j->sp_out = sp_out;""",
     """        j->v = v; j->g = g; j->sp_out = sp_out;"""),
    ("""                        float *restrict v, float *restrict g,
                        float *restrict refrac,
                        uint64_t *restrict sp_out,""",
     """                        float *restrict v, float *restrict g,
                        uint64_t *restrict sp_out,"""),
):
    s = s.replace(a, b)

# --- lif_step: gate bitset + countdown list replace the refrac arrays ---
s = sub(s, """    float *v, float *g, float *refrac, const float *refrac_steps,
    uint64_t *sp_bits, uint64_t *sp_scratch,""",
"""    float *v, float *g,
    uint64_t *gate_bits, uint64_t *in_ref,
    int32_t *rc_idx, int32_t *rc_cnt, int *n_ref_io,
    const int32_t *refrac_steps,
    uint64_t *sp_bits, uint64_t *sp_scratch,""", "lif_step signature")

s = sub(s, """    memset(sp_scratch, 0, (size_t)(nw + 1) * sizeof(uint64_t));
    sweep_all(chunks, n_chunks, tail_w, v, g, refrac, sp_bits, sp_scratch,
              c_decay, c_mem, v_rest, v_reset, v_th);""",
"""    /* --- advance the refractory countdowns BEFORE the delayed pass reads the
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
    sweep_all(chunks, n_chunks, tail_w, v, g, sp_bits, sp_scratch,
              c_decay, c_mem, v_rest, v_reset, v_th);""", "lif_step decrement")

s = sub(s, "            const float gate = (refrac[i] >= refrac_steps[i]) ? 1.0f : 0.0f;",
        "            const float gate = BIT_GET(gate_bits, i) ? 1.0f : 0.0f;",
        "delayed-pass gate")

s = sub(s, """    memcpy(sp_bits, sp_scratch, (size_t)(nw + 1) * sizeof(uint64_t));
    return nsp;""",
"""    /* --- neurons that spiked THIS step enter refractoriness.
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
    return nsp;""", "lif_step enter-refractory")

C.write_text(s, encoding="utf-8")
print("patched lif_kernel.c")

# --------------------------------------------------------------- Python -----
t = PY.read_text(encoding="utf-8")

t = sub(t, """        c_f32p, c_f32p, c_f32p, c_f32p,                  # v g refrac refrac_steps
        c_u64p, c_u64p,                                  # sp_bits, sp_scratch""",
"""        c_f32p, c_f32p,                                  # v, g
        c_u64p, c_u64p,                                  # gate_bits, in_ref
        c_i32p, c_i32p, ctypes.POINTER(ctypes.c_int),    # rc_idx, rc_cnt, n_ref
        c_i32p,                                          # refrac_steps (int32)
        c_u64p, c_u64p,                                  # sp_bits, sp_scratch""",
        "argtypes")

t = sub(t, """        base_refrac = int(round(self.p["tRefrac"] / dt))
        self.refrac = np.full(N, np.float32(base_refrac), dtype=np.float32)
        self.refrac_steps = np.full(N, np.float32(base_refrac), dtype=np.float32)""",
"""        base_refrac = int(round(self.p["tRefrac"] / dt))
        self.refrac_steps = np.full(N, base_refrac, dtype=np.int32)
        # Refractory state as a gate bitset plus a compact countdown list. Only
        # ~40 neurons are refractory at once (1.75 spikes/step x 22 steps), so a
        # dense fp32 counter cost 554 KB of sweep traffic each way for nothing.
        # All gates start OPEN, matching the reference's initial
        # refrac == refrac_steps.
        _nw = (N + 63) >> 6
        self.gate_bits = np.full(_nw + 1, np.uint64(0xFFFFFFFFFFFFFFFF),
                                 dtype=np.uint64)
        self.in_ref = np.zeros(_nw + 1, dtype=np.uint64)
        self.rc_idx = np.zeros(N, dtype=np.int32)
        self.rc_cnt = np.zeros(N, dtype=np.int32)
        self.n_ref = ctypes.c_int(0)""", "state alloc")

t = sub(t, "        self.refrac_steps[self.stim_idx] = np.float32(0.0)",
        "        self.refrac_steps[self.stim_idx] = 0", "stim refrac_steps")

t = sub(t, """        self._prf = _p(self.refrac, f)
        self._prs = _p(self.refrac_steps, f)""",
"""        self._pgate = _p(self.gate_bits, u64)
        self._pinref = _p(self.in_ref, u64)
        self._prci = _p(self.rc_idx, i32)
        self._prcc = _p(self.rc_cnt, i32)
        self._prs = _p(self.refrac_steps, i32)""", "pointers")

t = sub(t, """            self._pv, self._pg, self._prf, self._prs,
            self._pspb, self._pspc,""",
"""            self._pv, self._pg,
            self._pgate, self._pinref,
            self._prci, self._prcc, ctypes.byref(self.n_ref),
            self._prs,
            self._pspb, self._pspc,""", "call site")

PY.write_text(t, encoding="utf-8")
print("patched native_engine.py")
