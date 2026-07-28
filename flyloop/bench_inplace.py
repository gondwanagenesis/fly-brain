"""Clean best-of-2 benchmark: allocating vs in-place step, idle machine."""
import sys, time, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "flyloop"))
from brain_engine import BrainEngine

SUGAR = [720575940624963786,720575940630233916,720575940637568838,720575940638202345,
720575940617000768,720575940630797113,720575940632889389,720575940621754367,
720575940621502051,720575940640649691,720575940639332736,720575940616885538,
720575940639198653,720575940639259967,720575940617937543,720575940632425919,
720575940633143833,720575940612670570,720575940628853239,720575940629176663,
720575940611875570]

STEPS = 3000
res, spk = {}, {}
for label, ip in [("alloc", False), ("inplace", True),
                  ("alloc", False), ("inplace", True)]:
    e = BrainEngine(data_dir=str(ROOT / "data"), stim_ids=SUGAR, seed=0)
    e.inplace = ip
    e.inject(200.0)
    for _ in range(200):
        e.step()
    t0 = time.perf_counter()
    n = 0
    for _ in range(STEPS):
        n += int(e.step().sum().item())
    dt = time.perf_counter() - t0
    print(f"  {label:<8} {dt:7.3f}s  {dt/STEPS*1000:.4f} ms/step  spikes={n}")
    res.setdefault(label, []).append(dt)
    spk[label] = n

a, b = min(res["alloc"]), min(res["inplace"])
print()
print(f"BEST-OF-2  alloc {a/STEPS*1000:.4f} ms/step | "
      f"inplace {b/STEPS*1000:.4f} ms/step | ratio {a/b:.2f}x")
print(f"correctness: spikes {spk['inplace']} vs {spk['alloc']} "
      f"{'IDENTICAL' if spk['inplace']==spk['alloc'] else 'DIFFER'}")
print(f"1 s brain time: {a/STEPS*10000:.1f}s -> {b/STEPS*10000:.1f}s")
print(f"threads={torch.get_num_threads()}")
