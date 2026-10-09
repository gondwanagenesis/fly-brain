import sys, numpy as np
sys.path.insert(0,'../insta')
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFilter
import render_brain as rb
xyz, cat, has = rb.xyz, rb.cat, rb.has
W, H = 480, 645
x0, x1, z0, z1 = -770, 770, -520, 1530
sc = min(W/(x1-x0), H/(z1-z0))*0.96
px = ((xyz[:,0]-(x0+x1)/2)*sc + W/2).astype(int)
pz = ((xyz[:,2]-z0)*sc + (H-(z1-z0)*sc)/2).astype(int)
inb = (px>=0)&(px<W)&(pz>=0)&(pz<H)&has
pal = np.array([[34,211,238],[255,43,214],[124,255,79],[45,212,191],[255,170,0],[255,120,40],[255,59,80],[160,110,255]],np.float32)/255
def slab(ylo, yhi):
    m = inb & (xyz[:,1]>=ylo) & (xyz[:,1]<yhi)
    img = np.zeros((H,W,3),np.float32)
    for c in range(3): np.add.at(img[:,:,c], (pz[m], px[m]), pal[cat[m],c]*2.2)
    glow = gaussian_filter(img,(1.2,1.2,0))*1.6 + gaussian_filter(img,(5,5,0))*2.2
    out = np.clip(1-np.exp(-(img*1.3+glow)),0,1)
    a = np.clip(out.max(2)*1.6,0,1)
    rgba = np.dstack([out, a]); return Image.fromarray((rgba*255).astype(np.uint8),'RGBA')
def stack(n=9, CW=1080, CH=1500, shear=0.0):
    canvas = Image.new('RGBA',(CW,CH),(0,0,0,0))
    edges = np.linspace(-300, 420, n+1)
    # back (k=n-1) to front (k=0)
    for k in range(n-1,-1,-1):
        s = slab(edges[k], edges[k]+62)
        panel = Image.new('RGBA',(W,H),(0,30,45,40)); d = ImageDraw.Draw(panel)
        d.rectangle([0,0,W-1,H-1], outline=(0,240,255,230), width=2)
        panel.alpha_composite(s)
        scale = 1.0 - 0.02*k
        pw, ph = int(W*scale), int(H*scale)
        panel = panel.resize((pw,ph), Image.LANCZOS)
        # skew
        cx = 60 + k*74; cy = 790 - k*26 - ph//2
        # glow edge
        glow = panel.filter(ImageFilter.GaussianBlur(6)); canvas.alpha_composite(glow,(cx,cy)); canvas.alpha_composite(panel,(cx,cy))
    return canvas
if __name__=='__main__':
    c = stack(); c.save('assets/slabs_stack.png'); print(c.size)
