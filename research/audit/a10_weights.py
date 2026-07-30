import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(r"C:\Users\neogo\Documents\FlyBrain")
fo = torch.load(ROOT/"data"/"fanout_csc.pt")
v = fo["val"]
print("fanout val dtype", v.dtype, "n =", v.numel())
print("integral:", bool(torch.all(v == torch.round(v))))
print("min", float(v.min()), "max", float(v.max()))
print("abs max", float(v.abs().max()), " fits int16:", float(v.abs().max()) <= 32767)
print("max |sum| of duplicates onto one target (worst-case fp32 exactness bound):")
# per-target total positive / negative
post = fo["post"]; N = int(fo["crow"].numel())-1
pos = torch.zeros(N).scatter_add_(0, post, v.clamp(min=0))
neg = torch.zeros(N).scatter_add_(0, post, v.clamp(max=0))
print("   max total in-weight", float(pos.max()), " min", float(neg.min()),
      " -> exact in fp32 (<2^24):", float(pos.max()) < 2**24)
