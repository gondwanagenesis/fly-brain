import sys, glob, os, numpy as np
from PIL import Image
for f in sorted(glob.glob('ink/*.png')):
    a=np.array(Image.open(f).convert('L'))>60
    ys,xs=np.where(a)
    if len(xs)==0: print(os.path.basename(f),'no ink'); continue
    # ink beyond the safe box (allow 6px tolerance on glow-free ink)
    out=[]
    if xs.max()>935: out.append(f'right {xs.max()}')
    if ys.min()<226: out.append(f'top {ys.min()}')
    if ys.max()>1504: out.append(f'bottom {ys.max()}')
    if xs.min()<48: out.append(f'left {xs.min()}')
    print(os.path.basename(f), f'ink x {xs.min()}-{xs.max()} y {ys.min()}-{ys.max()}', 'VIOLATION ' + ', '.join(out) if out else 'ok')
