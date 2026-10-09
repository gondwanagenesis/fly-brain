"""Real-data frames for the Reel (his actual neuron positions and his actual simulated firing).
usage: python render_real_frames.py <dir containing render_brain.py + rates_seq.npy> <out_dir> [W H]
  seq_XX.png    the 30 real frames of rates_seq.npy (100 ms each): silence, ignition at frame 5, sugar, shadow
  build_XX.png  24 frames of the nervous system assembling neuron by neuron (random order, fixed seed), each lit so the points read
Render at W x H = 1000 x 1340 (default) and upscale in the edit, or pass e.g. 1500 2010 for hero shots."""
import sys, os, numpy as np
from PIL import Image
src, out = sys.argv[1], sys.argv[2]
W = int(sys.argv[3]) if len(sys.argv) > 3 else 1000; H = int(sys.argv[4]) if len(sys.argv) > 4 else 1340
sys.path.insert(0, src); os.makedirs(out, exist_ok=True)
import render_brain as rb
seq = np.load(os.path.join(src, 'rates_seq.npy'))
for i in range(len(seq)):
    Image.fromarray(rb.render(seq[i], W, H)).save(os.path.join(out, f'seq_{i:02d}.png'))
xyz, cat, has = rb.xyz, rb.cat, rb.has
order = np.random.default_rng(0).permutation(len(xyz))
for k, frac in enumerate(np.geomspace(0.002, 1.0, 24)):
    keep = order[:max(1, int(frac * len(order)))]
    rb.xyz, rb.cat, rb.has = xyz[keep], cat[keep], has[keep]
    Image.fromarray(rb.render(np.full(len(keep), 7.0 * (1.0 - frac) ** 3.0, np.float32), W, H)).save(os.path.join(out, f'build_{k:02d}.png'))
rb.xyz, rb.cat, rb.has = xyz, cat, has
print('wrote', len(seq) + 24, 'frames to', out)
