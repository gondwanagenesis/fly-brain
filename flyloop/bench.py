"""Head-to-head: upstream TorchModel vs optimized BrainEngine.
Same connectome, same params, same stimulus. Reports speed + biological agreement.
"""
import sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "flyloop"))

from brain_engine import BrainEngine, MODEL_PARAMS, DT

SUGAR = [720575940624963786,720575940630233916,720575940637568838,720575940638202345,
720575940617000768,720575940630797113,720575940632889389,720575940621754367,
720575940621502051,720575940640649691,720575940639332736,720575940616885538,
720575940639198653,720575940639259967,720575940617937543,720575940632425919,
720575940633143833,720575940612670570,720575940628853239,720575940629176663,
720575940611875570]
T_MS = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
STEPS = int(T_MS / DT)
print(f"benchmark: {T_MS} ms  ({STEPS} steps)  sugar GRNs @200Hz\n")

# ---------------- optimized ----------------
t0 = time.perf_counter()
eng = BrainEngine(data_dir=str(ROOT / "data"), stim_ids=SUGAR, seed=0)
setup_new = time.perf_counter() - t0
eng.inject(200.0)
t0 = time.perf_counter()
n_new = 0
for _ in range(STEPS):
    n_new += int(eng.step(record=True).sum().item())
run_new = time.perf_counter() - t0
df = eng.spikes_dataframe()
act_new = df.flywire_id.nunique() if len(df) else 0
print(f"OPTIMIZED   setup {setup_new:6.2f}s   run {run_new:7.3f}s   "
      f"spikes {n_new:6d}   active {act_new:4d}")

# ---------------- upstream ----------------
import run_pytorch as rp
t0 = time.perf_counter()
flyid2i, i2flyid = rp.get_hash_tables(str(ROOT / "data" / "2025_Completeness_783.csv"))
exc = [flyid2i[n] for n in SUGAR]
W = rp.get_weights(str(ROOT / "data" / "2025_Connectivity_783.parquet"),
                   str(ROOT / "data" / "2025_Completeness_783.csv"),
                   str(ROOT / "data"), csr=True)
model = rp.TorchModel(1, W.shape[0], DT, MODEL_PARAMS, W, exc_indices=exc, device="cpu")
state = model.state_init()
setup_old = time.perf_counter() - t0
rates = torch.zeros(1, W.shape[0]); rates[:, exc] = 200.0
t0 = time.perf_counter()
n_old = 0
with torch.no_grad():
    for _ in range(STEPS):
        state = model(rates, *state)
        n_old += int((state[2] > 0).sum().item())
run_old = time.perf_counter() - t0
print(f"UPSTREAM    setup {setup_old:6.2f}s   run {run_old:7.3f}s   spikes {n_old:6d}")

print(f"\n  speedup: {run_old/run_new:.2f}x   "
      f"({run_old/STEPS*1000:.2f} -> {run_new/STEPS*1000:.2f} ms/step)")
print(f"  spike-count agreement: {n_new} vs {n_old} "
      f"({100*n_new/max(n_old,1):.1f}%)")
print(f"  projected 1 s sim: {run_old/STEPS*10000:.0f}s -> {run_new/STEPS*10000:.0f}s")
