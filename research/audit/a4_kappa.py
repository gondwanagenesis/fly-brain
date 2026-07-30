import sys
from pathlib import Path
import numpy as np
ROOT = Path(r"C:\Users\neogo\Documents\FlyBrain")
sys.path.insert(0, str(ROOT / "code"))
from window_step import kappa, kappa_max_over_window, KAPPA_WINDOW_MAX, DELAY, TAU_M, TAU_S

# true interior maximiser of kappa
s_star = (TAU_S*TAU_M/(TAU_M-TAU_S))*np.log(TAU_M/TAU_S)
print("kappa interior peak at s* = %.6f ms  (window D = %.2f ms)" % (s_star, DELAY))
print("KAPPA_WINDOW_MAX  =", repr(KAPPA_WINDOW_MAX))
print("kappa(D)          =", repr(float(kappa(DELAY))))
print("equal:", KAPPA_WINDOW_MAX == float(kappa(DELAY)))

# high precision check of the fp value (float128 where available)
ld = np.longdouble
k_ld = (np.exp(-ld(DELAY)/ld(TAU_M)) - np.exp(-ld(DELAY)/ld(TAU_S)))*(ld(TAU_S)/(ld(TAU_M)-ld(TAU_S)))
print("longdouble kappa(D) =", k_ld, " fp64 - ld =", np.float64(KAPPA_WINDOW_MAX) - np.float64(k_ld))

# soundness of the SAMPLED max when the peak is interior
for D in (1.8, 5.0, 9.242, 12.0, 20.0):
    samp = kappa_max_over_window(D)
    true = float(kappa(min(D, s_star)))
    print(f"D={D:6.3f}  sampled={samp:.12f}  true={true:.12f}  sampled-true={samp-true: .3e}"
          f"  {'UNSOUND (under-estimate)' if samp < true else 'ok'}")
