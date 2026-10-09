import random, os
HERE = os.path.dirname(os.path.abspath(__file__))
SAFE = dict(left=54, right=929, top=230, bottom=1498)   # IG safe box (px on 1080x1920)

def page(body, guides=False, extra_css=''):
    g = ('<div class="guides"><i class="t"></i><i class="b"></i><i class="r"></i>'
         '<span style="left:20px;top:200px">TOP 12% UI</span><span style="left:20px;top:1506px">BOTTOM 22% UI</span>'
         '<span style="left:934px;top:240px;writing-mode:vertical-rl">RIGHT 14% UI</span></div>') if guides else ''
    return (f'<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="reel.css">'
            f'<style>{extra_css}</style></head><body><div class="frame">{body}{g}</div></body></html>')

def slices(img, n=6, seed=1, ymin=.2, ymax=.85, maxshift=60, pos='center', size='cover', hmin=.006, hmax=.035, op=(.65,.9), extra=''):
    rnd = random.Random(seed); out = []
    for _ in range(n):
        y = rnd.uniform(ymin, ymax); h = rnd.uniform(hmin, hmax)
        dx = rnd.choice([-1, 1]) * rnd.uniform(14, maxshift)
        hue = rnd.choice([160, -70, 250, 90, -20, 200])
        out.append(f'<div class="slice" style="background-image:url({img});background-size:{size};background-position:{pos};'
                   f'clip-path:inset({y*100:.2f}% 0 {100-(y+h)*100:.2f}% 0);transform:translateX({dx:.0f}px);'
                   f'filter:hue-rotate({hue}deg) saturate({rnd.uniform(2,3.2):.1f});opacity:{rnd.uniform(*op):.2f};{extra}"></div>')
    return ''.join(out)

def rgbsplit(img, pos='center', size='cover', d=9, op=.55, extra=''):
    """cheap RGB channel split: cyan copy shifted left, magenta copy shifted right (screen blended)"""
    return (f'<div class="slice" style="background-image:url({img});background-size:{size};background-position:{pos};'
            f'transform:translateX({-d}px);filter:brightness(.9) sepia(1) hue-rotate(150deg) saturate(5);opacity:{op};{extra}"></div>'
            f'<div class="slice" style="background-image:url({img});background-size:{size};background-position:{pos};'
            f'transform:translateX({d}px);filter:brightness(.9) sepia(1) hue-rotate(260deg) saturate(5);opacity:{op};{extra}"></div>')

def fx(scan=True, vig=True, noise=True):
    return ('<div class="vig"></div>' if vig else '') + ('<div class="scan"></div>' if scan else '') + ('<div class="noise"></div>' if noise else '')

def grad(top=.8, tstop=30, bot=.9, bstart=62):
    return (f'<div class="abs" style="inset:0;background:linear-gradient(180deg,rgba(0,0,0,{top}) 0%,rgba(0,0,0,0) {tstop}%,'
            f'rgba(0,0,0,0) {bstart}%,rgba(0,0,0,{bot}) 100%)"></div>')
