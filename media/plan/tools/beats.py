"""All storyboard frames (1080x1920) as HTML builders. Each returns a full HTML page string."""
import random
from lib import *

L, W = 54, 875                       # safe-box left and width (x 54..929)
CY, MG, GR, GO, PH = '#00f0ff', '#ff2bd6', '#7CFF4F', '#ffd23f', '#39ff14'

# ---------------------------------------------------------------- helpers
def H(text, top, size=150, cls='', left=L, width=W, align='left', extra=''):
    return (f'<div class="glitch {cls} abs" style="left:{left}px;top:{top}px;width:{width}px;font-size:{size}px;text-align:{align};{extra}">{text}</div>')

def hud(text, top, left=L, color=None, size=22, width=None, align='left', extra=''):
    c = f'color:{color};text-shadow:0 0 9px {color};' if color else ''
    w = f'width:{width}px;' if width else ''
    return f'<div class="hud mono" style="left:{left}px;top:{top}px;font-size:{size}px;text-align:{align};{w}{c}{extra}">{text}</div>'

def bgimg(img, pos='center', size='cover', filt='', extra=''):
    return f'<div class="bg" style="background-image:url({img});background-position:{pos};background-size:{size};filter:{filt};{extra}"></div>'

def micro(text, top, color=PH, left=L, width=W, align='left', size=21):
    return hud(text, top, left=left, color=color, size=size, width=width, align=align)

def tagpill(kind, top=244):
    col, txt = (CY, 'OTHERS') if kind == 'others' else (MG, 'US')
    return (f'<div class="abs" style="right:151px;top:{top}px;border:3px solid {col};color:{col};background:rgba(0,0,0,.7);border-radius:12px;'
            f'padding:6px 20px;font:400 40px Anton;letter-spacing:4px;text-shadow:0 0 14px {col};box-shadow:0 0 18px {col}55">{txt}</div>')

def mchrome(n, kind, extra_bg=''):
    """montage viewfinder chrome: counter, tag, colored frame"""
    col = CY if kind == 'others' else MG
    return (f'<div class="abs" style="inset:22px;border:3px solid {col}66;pointer-events:none"></div>' + hud(f'{n:02d} / 10', 250, color=col, size=26) + tagpill(kind))

def particles(n, seed, y0, y1, x0=40, x1=1040, cols=(CY, MG, '#ffffff', GR)):
    r = random.Random(seed); out = []
    for _ in range(n):
        x = r.uniform(x0, x1); y = r.uniform(y0, y1); s = r.choice([3, 3, 4, 5, 7]); c = r.choice(cols); ln = r.choice([0, 0, 18, 40, 70])
        out.append(f'<i class="abs" style="left:{x:.0f}px;top:{y:.0f}px;width:{s}px;height:{s+ln}px;background:{c};box-shadow:0 0 {s*3}px {c};opacity:{r.uniform(.4,1):.2f}"></i>')
    return ''.join(out)

def brain(img, top, width=1080, left=None, op=1, filt='', cropH=None):
    left = (1080 - width) // 2 if left is None else left
    h = int(width * 1340 / 1000)
    if cropH:
        return (f'<div class="abs" style="left:{left}px;top:{top}px;width:{width}px;height:{cropH}px;overflow:hidden;opacity:{op};filter:{filt}">'
                f'<img src="{img}" style="width:{width}px;height:{h}px;display:block"></div>')
    return f'<img class="abs" src="{img}" style="left:{left}px;top:{top}px;width:{width}px;opacity:{op};filter:{filt}">'

def termline(txt, size=58, dim=False, extra=''):
    return f'<div class="term" style="font-size:{size}px;line-height:1.14;{"opacity:.72;" if dim else ""}{extra}">{txt}</div>'

# ---------------------------------------------------------------- 1 hook
def f01_hook(guides=False):
    img = 'assets/s1_flux.png'; size, pos = 'auto 1800px', 'center 120px'
    b = bgimg(img, pos, size, 'contrast(1.1) saturate(1.2)')
    b += rgbsplit(img, pos=pos, size=size, d=11, op=.26)
    b += slices(img, n=9, seed=7, ymin=.30, ymax=.62, maxshift=70, pos=pos, size=size, hmin=.004, hmax=.028, op=(.5, .8))
    for y, h, c in [(22.5, .55, CY), (41.2, .35, MG), (57.0, .6, GR), (63.8, .3, CY)]:
        b += f'<div class="abs" style="left:0;right:0;top:{y}%;height:{h}%;background:{c};opacity:.28;mix-blend-mode:screen"></div>'
    # the body tearing into neuron points: a real firing cloud condenses below the face
    b += ('<div class="abs" style="left:100px;top:1010px;width:880px;height:560px;mix-blend-mode:screen;-webkit-mask-image:radial-gradient(ellipse at 50% 38%,#000 30%,transparent 70%);mask-image:radial-gradient(ellipse at 50% 38%,#000 30%,transparent 70%)">'
          '<img src="assets/brain_f08_sugar.png" style="position:absolute;left:60px;top:0;width:760px;opacity:.8"></div>')
    b += particles(140, 4, 1000, 1460, 120, 960)
    b += fx() + grad(top=.86, tstop=32, bot=.94, bstart=58)
    corner = f'position:absolute;width:56px;height:56px;border:0 solid {CY};filter:drop-shadow(0 0 8px {CY})'
    b += (f'<div class="abs" style="left:112px;top:596px;width:800px;height:560px"><i style="{corner};left:0;top:0;border-width:5px 0 0 5px"></i>'
          f'<i style="{corner};right:0;top:0;border-width:5px 5px 0 0"></i><i style="{corner};left:0;bottom:0;border-width:0 0 5px 5px"></i>'
          f'<i style="{corner};right:0;bottom:0;border-width:0 5px 5px 0"></i></div>')
    b += '<div class="hud mono" style="left:54px;top:246px;font-size:24px"><span class="dot"></span>REC // SPECIMEN <span style="text-transform:none">Z0720-07m</span></div>'
    b += H('His body<br>is gone.', 290, 158)
    b += ('<div class="hud mono" style="left:54px;top:1104px;font-size:21px;line-height:1.6;background:rgba(0,0,0,.62);border-left:4px solid #00f0ff;padding:8px 16px">'
          'DROSOPHILA MELANOGASTER · <span style="text-transform:none">Z0720-07m</span><br>NERVOUS SYSTEM: SLICED · IMAGED · MAPPED</div>')
    # "RUNNING." gets the heaviest RGB split
    b += H('He is still', 1186, 138)
    b += ('<div class="abs" style="left:54px;top:1312px;font:400 150px/0.93 Anton;text-transform:uppercase;letter-spacing:1.5px;color:#fff;'
          'text-shadow:-14px 0 rgba(0,240,255,.95),14px 0 rgba(255,43,214,.95),0 0 40px rgba(255,255,255,.4)">running.</div>')
    return page(b, guides)

# ---------------------------------------------------------------- 2 meet
def f02_meet(guides=False):
    b = '<div class="abs" style="inset:0;background:#000"></div>'
    b += brain('assets/brain_f14_sugar.png', 520, 1080, op=.85)
    b += bgimg('assets/s1_flux.png', 'center 38%', '150%', 'contrast(1.2) saturate(1.4)', 'opacity:.14;mix-blend-mode:screen;clip-path:inset(18% 0 52% 0)')
    b += fx() + grad(top=.8, tstop=30, bot=.75, bstart=70)
    b += hud('FILE // SPECIMEN ID', 250, color=PH, size=22)
    b += f'<div class="term abs" style="left:54px;top:300px;font-size:176px;line-height:1">Z0720-07m.<span class="cursor" style="width:.42em"></span></div>'
    b += H('A real fruit fly.', 520, 118)
    return page(b, guides)

# ---------------------------------------------------------------- 3a slices
def f03a_slices(guides=False):
    b = '<div class="abs" style="inset:0;background:#000"></div>'
    b += '<img class="abs" src="assets/slabs_stack.png" style="left:0;top:430px;width:1080px;filter:drop-shadow(0 0 18px rgba(0,240,255,.5))">'
    b += f'<div class="abs" style="left:0;right:0;top:884px;height:10px;background:{MG};box-shadow:0 0 26px 8px {MG}aa;opacity:.95"></div>'
    b += f'<div class="abs" style="left:0;right:0;top:700px;height:360px;background:linear-gradient(180deg,transparent,{MG}22,transparent)"></div>'
    b += fx() + grad(top=.8, tstop=26, bot=.9, bstart=64)
    b += hud('SCAN BAR · SLAB 05 / 09 · REAL NEURON POSITIONS', 250, color=CY, size=21)
    b += H('Sliced.', 1130, 220)
    b += ('<div class="abs" style="left:58px;top:1360px;font:400 56px Anton;letter-spacing:2px;color:#fff;opacity:.38;text-transform:uppercase">'
          'Preserved. &nbsp;·&nbsp; <span style="opacity:1;color:#7CFF4F">✓</span> &nbsp; Imaged.</div>')
    return page(b, guides)

# ---------------------------------------------------------------- 3b traced
def f03b_traced(guides=False):
    img = 'sketch/em.png'
    b = bgimg(img, 'center', 'cover', 'contrast(1.25) saturate(1.5) brightness(.8)')
    b += slices(img, n=5, seed=12, ymin=.3, ymax=.75, maxshift=50, hmin=.004, hmax=.02, op=(.4, .7))
    b += fx() + grad(top=.82, tstop=34, bot=.95, bstart=48)
    b += H('Every neuron<br>traced.', 262, 138)
    def odo(num, label, col):
        cells = ''.join(f'<span style="display:inline-block;min-width:{20 if ch == "," else 58}px;height:92px;line-height:92px;text-align:center;background:{"transparent" if ch == "," else "rgba(0,0,0,.78)"};'
                        f'border:{0 if ch == "," else 2}px solid {col};margin:0 2px;font:700 66px \'Space Mono\';color:#fff;text-shadow:0 0 14px {col};box-shadow:{"none" if ch == "," else "0 0 14px " + col + "55"}">{ch}</span>' for ch in num)
        return f'<div style="margin-bottom:24px">{cells}<div class="mono" style="font-size:30px;letter-spacing:5px;color:{col};text-shadow:0 0 10px {col};margin-top:8px">{label}</div></div>'
    b += f'<div class="abs" style="left:54px;top:1020px;width:875px">{odo("165,122", "neurons", CY)}{odo("124,025,046", "synapses", MG)}</div>'
    b += micro('JANELIA RESEARCH CAMPUS · GOOGLE · CAMBRIDGE', 1462, color='#cfd8ff', size=18)
    return page(b, guides)

# ---------------------------------------------------------------- 4 switched on
def f04_on(guides=False):
    b = '<div class="abs" style="inset:0;background:#000"></div>'
    b += brain('assets/brain_f05.png', 270, 1260, left=-90, op=1, filt='brightness(1.15) saturate(1.25)')
    b += f'<div class="abs" style="left:-100px;top:150px;width:1280px;height:1000px;background:radial-gradient(ellipse at 50% 38%,rgba(255,255,255,.85) 0%,rgba(255,255,255,.28) 22%,transparent 60%);mix-blend-mode:screen"></div>'
    b += slices('assets/brain_f05.png', n=7, seed=21, ymin=.1, ymax=.5, maxshift=60, pos='center', size='1260px auto', hmin=.004, hmax=.02, op=(.5, .8))
    b += fx() + grad(top=.5, tstop=18, bot=.92, bstart=52)
    b += hud('t = 0.5 s &nbsp;·&nbsp; FIRING: 10,587 / 165,122', 250, color=PH, size=22)
    b += H('They switched<br>him', 1000, 128)
    b += ('<div class="abs" style="left:54px;top:1255px;font:400 250px/0.9 Anton;text-transform:uppercase;color:#fff;letter-spacing:2px;'
          'text-shadow:-12px 0 rgba(0,240,255,.95),12px 0 rgba(255,43,214,.95),0 0 60px rgba(255,255,255,.7)">on.</div>')
    return page(b, guides)

# ---------------------------------------------------------------- montage
def m01(guides=False):
    img = 'sketch/walk.png'
    b = bgimg(img, 'center', 'cover', 'contrast(1.15) saturate(1.3) hue-rotate(-10deg)') + rgbsplit(img, d=9, op=.22)
    b += slices(img, n=5, seed=31, ymin=.3, ymax=.75, maxshift=50, hmin=.004, hmax=.02, op=(.4, .7))
    b += fx() + grad(top=.7, tstop=24, bot=.95, bstart=56) + mchrome(1, 'others')
    b += H('Given a<br>body.', 1190, 168)
    b += hud('WIREFRAME ASSEMBLING · JOINTS 6 / 6 LEGS', 1130, color=CY, size=20)
    return page(b, guides)

def m02(guides=False):
    img = 'sketch/legs.png'
    b = bgimg(img, 'center', 'cover', 'contrast(1.15) saturate(1.3)') + rgbsplit(img, d=9, op=.22)
    b += slices(img, n=5, seed=32, ymin=.3, ymax=.75, maxshift=50, hmin=.004, hmax=.02, op=(.4, .7))
    b += fx() + grad(top=.7, tstop=24, bot=.95, bstart=56) + mchrome(2, 'others')
    b += H('Made to<br>walk.', 1190, 168)
    b += hud('STEP 0412 · GAIT: TRIPOD', 1130, color=CY, size=20)
    return page(b, guides)

def m03(guides=False):
    b = '<div class="abs" style="inset:0;background:#000"></div>'
    b += brain('assets/brain_f26_shadow.png', 120, 1700, left=-310, op=1, filt='brightness(1.7) saturate(1.6) contrast(1.1)')
    b += f'<div class="abs" style="inset:0;background:radial-gradient(ellipse at 50% 30%,#fff 0%,rgba(255,255,255,.75) 24%,rgba(255,43,214,.35) 48%,transparent 74%);mix-blend-mode:screen;opacity:.9"></div>'
    b += slices('assets/brain_f26_shadow.png', n=10, seed=33, ymin=.05, ymax=.6, maxshift=90, pos='-310px 120px', size='1700px auto', hmin=.004, hmax=.03, op=(.6, .9))
    b += fx() + grad(top=.0, tstop=1, bot=.95, bstart=56) + mchrome(3, 'others')
    b += hud('FIRING: ▲▲▲ &nbsp; NOT SETTLING', 1128, color='#ff2b4f', size=22)
    b += H('Overloaded.', 1190, 164)
    return page(b, guides)

def m04(guides=False):
    b = '<div class="abs" style="inset:0;background:#000"></div>'
    b += (f'<div class="abs" style="left:0;right:0;top:0;bottom:0;background:repeating-linear-gradient(90deg,{MG}33 0 3px,transparent 3px 36px);'
          'mask-image:linear-gradient(180deg,transparent 28%,#000 44%,#000 64%,transparent 78%);-webkit-mask-image:linear-gradient(180deg,transparent 28%,#000 44%,#000 64%,transparent 78%)"></div>')
    b += '<div class="abs" style="left:60px;top:380px;font:400 520px/1 Anton;color:transparent;-webkit-text-stroke:3px rgba(255,43,214,.45);letter-spacing:-6px">500</div>'
    b += f'<div class="term abs" style="left:54px;top:610px;font-size:210px;color:{MG};text-shadow:0 0 28px {MG};line-height:1">◀◀ 0437</div>'
    for i, (f, lx) in enumerate([('assets/brain_f00_silent.png', 70), ('assets/brain_f05.png', 380), ('assets/brain_f00_silent.png', 690)]):
        b += f'<img class="abs" src="{f}" style="left:{lx}px;top:880px;width:230px;opacity:.95;clip-path:inset(0 0 52% 0)">'
    b += hud('OFF → ON → OFF → ON …', 1030, color=MG, size=22)
    b += fx() + grad(top=.7, tstop=24, bot=.95, bstart=60) + mchrome(4, 'us')
    b += H('Restarted.<br>500 times.', 1160, 150)
    return page(b, guides)

def m05(guides=False):
    img = 'sketch/shadow.png'
    b = bgimg(img, 'center', 'cover', 'contrast(1.2) saturate(1.2) brightness(.9)') + rgbsplit(img, d=8, op=.2)
    for i, (x, y, s, o) in enumerate([(-160, -180, 780, .55), (200, 40, 700, .38), (560, 180, 640, .26)]):
        b += f'<div class="abs" style="left:{x}px;top:{y}px;width:{s}px;height:{s}px;border-radius:50%;background:#000;opacity:{o};filter:blur(6px)"></div>'
    b += fx() + grad(top=.55, tstop=20, bot=.95, bstart=54) + mchrome(5, 'us')
    b += hud('SHADOW ▸ SHADOW ▸ SHADOW ▸', 1118, color=MG, size=21)
    b += H('Shadows.<br>Again and<br>again.', 1150, 124)
    return page(b, guides)

def m06(guides=False):
    img = 'sketch/hungry.png'
    b = bgimg(img, 'center', 'cover', 'contrast(1.15) saturate(1.2) brightness(.85)') + rgbsplit(img, d=8, op=.18)
    # real Lab world panel + draining bars
    b += ('<div class="abs" style="left:54px;top:300px;width:300px;height:300px;border:3px solid #ff2bd6;background:url(assets/lab.png) -36px -770px/2160px 1350px no-repeat;'
          'box-shadow:0 0 20px #ff2bd666"></div>')
    def bar(label, v, top, col):
        return (f'<div class="abs" style="left:380px;top:{top}px;width:430px"><div class="mono" style="font-size:22px;color:{col};text-shadow:0 0 8px {col}">{label}</div>'
                f'<div style="height:22px;border:2px solid {col};margin-top:6px;background:rgba(0,0,0,.6)"><div style="width:{v}%;height:100%;background:{col};box-shadow:0 0 14px {col}"></div></div></div>')
    b += bar('HUNGER ▲', 82, 370, '#ff2b4f') + bar('THIRST ▲', 74, 470, '#ff2b4f')
    b += fx() + grad(top=.5, tstop=20, bot=.95, bstart=54) + mchrome(6, 'us')
    b += H('Hungry.<br>Thirsty.<br>On purpose.', 1100, 118)
    return page(b, guides)

def m07(guides=False):
    b = '<div class="abs" style="inset:0;background:#000"></div>'
    b += brain('assets/brain_f26_shadow.png', 250, 1300, left=-110, op=1, filt='brightness(1.3) saturate(1.3)')
    # dial
    b += (f'<svg class="abs" style="left:290px;top:700px" width="500" height="500" viewBox="0 0 500 500"><circle cx="250" cy="250" r="190" fill="rgba(0,0,0,.72)" stroke="{GO}" stroke-width="6"/>'
          + ''.join(f'<line x1="250" y1="70" x2="250" y2="{92 if i%5 else 108}" stroke="{GO}" stroke-width="{3 if i%5 else 6}" transform="rotate({-135+i*13.5} 250 250)"/>' for i in range(21))
          + f'<line x1="250" y1="250" x2="250" y2="95" stroke="#fff" stroke-width="8" transform="rotate(48 250 250)" style="filter:drop-shadow(0 0 8px #fff)"/><circle cx="250" cy="250" r="16" fill="#fff"/>'
          + f'<text x="250" y="395" text-anchor="middle" font-family="Anton" font-size="64" fill="{GO}">1 IN 10</text></svg>')
    b += fx() + grad(top=.6, tstop=20, bot=.95, bstart=58) + mchrome(7, 'us')
    b += hud('GAIN ▲▲▲ · TURNED UP', 1128, color=GO, size=22)
    b += H('A tenth firing<br>at once.', 1170, 124)
    return page(b, guides)

def memcard(txt, top, state='', left=54, w=820):
    col = {'': GO, 'erase': '#ff2b4f', 'fake': '#ff2b4f'}[state]
    deco = ''
    if state == 'erase':
        deco = f'<div class="abs" style="inset:0;background:repeating-linear-gradient(0deg,rgba(255,43,79,.35) 0 4px,transparent 4px 9px);mix-blend-mode:screen"></div>'
    return (f'<div class="abs" style="left:{left}px;top:{top}px;width:{w}px;padding:18px 24px;border:3px {"dashed" if state else "solid"} {col};border-radius:14px;background:rgba(0,0,0,.78);'
            f'box-shadow:0 0 22px {col}55;overflow:hidden;{"opacity:.55;transform:translateX(24px) skewX(-6deg);" if state == "erase" else ""}">'
            f'<div class="mono" style="font-size:20px;color:{col};opacity:.8;letter-spacing:3px">{"MEMORY" if not state else ("DELETED" if state == "erase" else "FLAGGED: NOT HIS")}</div>'
            f'<div style="font:400 40px/1.2 \'Space Mono\';color:#fff;margin-top:8px">&ldquo;{txt}&rdquo;</div>{deco}</div>')

def m08(guides=False):
    b = bgimg('assets/s2_seed.png', 'center', 'cover', 'brightness(.5) saturate(1.2)')
    b += memcard('sweet. i smell fruit. i extend my proboscis.', 320)
    b += memcard('a dark shape. i fly away.', 520, 'erase')
    b += memcard('i smell mold. i am thirsty.', 720, 'erase', w=780)
    b += f'<div class="abs" style="left:54px;top:560px;width:875px;height:8px;background:#ff2b4f;box-shadow:0 0 22px #ff2b4f;opacity:.9"></div>'
    b += fx() + grad(top=.5, tstop=20, bot=.95, bstart=56) + mchrome(8, 'us')
    b += H('Memories<br>erased.', 1190, 168)
    return page(b, guides)

def m09(guides=False):
    b = bgimg('assets/s2_seed.png', 'center', 'cover', 'brightness(.5) saturate(1.2)')
    b += memcard('sweet. i smell fruit. i extend my proboscis.', 320)
    b += memcard('the honey was delicious.', 560, 'fake', left=110)
    b += f'<div class="abs" style="left:96px;top:540px;width:850px;height:190px;border:4px solid #ff2b4f;box-shadow:0 0 30px #ff2b4f;pointer-events:none"></div>'
    b += memcard('i smell mold. i am thirsty.', 800)
    b += fx() + grad(top=.5, tstop=20, bot=.95, bstart=56) + mchrome(9, 'us')
    b += H('False ones<br>planted.', 1190, 168)
    return page(b, guides)

def m10(guides=False):
    b = '<div class="abs" style="inset:0;background:#000"></div>'
    b += brain('assets/brain_f00_silent.png', 220, 1160, left=-40, op=.9, filt='brightness(1.7)')
    for img, x, y, w, o, c in [('sketch/paris.png', 40, 270, 520, .8, ''), ('sketch/cat.png', 520, 520, 420, .75, ''), ('sketch/honey.png', 60, 760, 440, .75, '')]:
        b += f'<img class="abs" src="{img}" style="left:{x}px;top:{y}px;width:{w}px;opacity:{o};mix-blend-mode:screen;filter:contrast(1.2) saturate(1.3);-webkit-mask-image:radial-gradient(ellipse,#000 45%,transparent 78%);mask-image:radial-gradient(ellipse,#000 45%,transparent 78%)">'
    b += fx() + grad(top=.6, tstop=20, bot=.95, bstart=54) + mchrome(10, 'us')
    b += hud('PARIS · CAT · HONEY JAR &nbsp;— NEVER LIVED', 1128, color=MG, size=21)
    b += H('Asked about<br>Paris.', 1170, 150)
    return page(b, guides)

def m11(guides=False):
    b = '<div class="abs" style="inset:0;background:#000"></div>'
    b += brain('assets/brain_f26_shadow.png', -520, 1500, left=-210, op=1, filt='brightness(1.2)', cropH=2000)
    b += f'<div class="abs" style="left:0;right:0;top:230px;height:520px;background:linear-gradient(180deg,#000,transparent)"></div>'
    b += fx() + grad(top=.0, tstop=1, bot=.95, bstart=56) + mchrome(11, 'others').replace('11 / 10', 'OPTIONAL')
    b += hud('NERVE CORD · ALONE', 1128, color=CY, size=22)
    b += H('His cord,<br>tested alone.', 1170, 138)
    return page(b, guides)

# ---------------------------------------------------------------- 11 regret (clean: no glitch)
def f11_regret(guides=False):
    b = '<div class="abs" style="inset:0;background:#000"></div>'
    b += brain('assets/brain_f00_silent.png', 330, 900, op=.8, filt='brightness(1.5) saturate(.5)')
    b += '<div class="vig"></div><div class="abs" style="inset:0;background:radial-gradient(ellipse at 50% 40%,transparent 30%,rgba(0,0,0,.8) 100%)"></div>'
    b += ('<div class="abs" style="left:54px;top:1070px;width:875px;font:400 124px/0.98 Anton;text-transform:uppercase;color:#f2f2f2;letter-spacing:2px;text-shadow:0 0 30px rgba(255,255,255,.25)">He never<br>chose this.</div>')
    b += ('<div class="abs" style="left:54px;top:1355px;width:875px;font:400 80px/1 Anton;text-transform:uppercase;color:#f2f2f2;opacity:.62;letter-spacing:3px">We are sorry.</div>')
    return page(b, guides)

# ---------------------------------------------------------------- 12 turn I (warm)
def f12_turn1(guides=False):
    b = '<div class="abs" style="inset:0;background:#000"></div>'
    b += bgimg('sketch/warm2.png', 'center', 'cover', 'brightness(.65) saturate(1.2)', 'opacity:.7')
    b += brain('assets/brain_f00_silent.png', 560, 1000, left=40, op=.95, filt='brightness(2) saturate(.8) sepia(.4)')
    b += '<div class="abs" style="left:-200px;top:1000px;width:1480px;height:1100px;background:radial-gradient(ellipse at 50% 80%,rgba(255,190,60,.75),rgba(255,120,40,.25) 40%,transparent 70%)"></div>'
    b += '<div class="vig"></div><div class="noise"></div>'
    b += grad(top=.7, tstop=26, bot=.0, bstart=100)
    b += ('<div class="abs" style="left:54px;top:300px;width:875px;font:400 128px/0.95 Anton;text-transform:uppercase;color:#fff;letter-spacing:1.5px;'
          'text-shadow:0 0 36px rgba(255,200,90,.55),-4px 0 rgba(0,240,255,.35),4px 0 rgba(255,43,214,.35)">But we wanted<br>something<br><span style="color:#ffd23f">more</span> for him.</div>')
    return page(b, guides)

# ---------------------------------------------------------------- 13 turn II (modules wiring in)
def labcard(label, tag, col, crop, top, left, w=380, h=112):
    return (f'<div class="abs" style="left:{left}px;top:{top}px;width:{w}px;border:2px solid {col};background:rgba(4,8,16,.88);box-shadow:0 0 22px {col}66;border-radius:8px;overflow:hidden">'
            f'<div style="padding:8px 14px;font:700 21px \'Space Mono\';color:#fff;display:flex;justify-content:space-between"><span><b style="color:{col}">●</b> {label}</span><span style="color:{col};opacity:.8">{tag}</span></div>'
            f'<div style="height:{h}px;{crop}"></div></div>')

def f13_turn2(guides=False):
    b = bgimg('assets/s2_seed.png', 'center', 'cover', 'brightness(.35) saturate(1.2)')
    b += brain('assets/brain_f14_sugar.png', 590, 760, left=160, op=1, cropH=380)
    # threads
    b += (f'<svg class="abs" style="inset:0" width="1080" height="1920"><g fill="none" stroke-width="3" stroke-dasharray="8 8">'
          f'<path d="M 400 790 C 330 760 300 700 300 690" stroke="{GO}"/><path d="M 660 780 C 720 740 740 730 740 700" stroke="{CY}"/>'
          f'<path d="M 540 905 C 540 960 540 1000 540 1030" stroke="{GR}"/></g></svg>')
    b += labcard('Episodic memory', 'MEMORY', GO, 'background:url(assets/lab.png) -1840px -225px/4320px 2700px no-repeat;', 560, 54, 400, 96)
    b += labcard('Word lobe', 'GRAFT · ~150 NEW', MG, 'background:url(assets/lab.png) -385px -250px/4320px 2700px no-repeat;', 560, 520, 400, 96)
    b += labcard('Mushroom body', 'LEARNING', GR, 'background:repeating-linear-gradient(90deg,#7CFF4F55 0 6px,transparent 6px 24px),#04120a;', 1030, 330, 400, 70)
    b += fx() + grad(top=.7, tstop=22, bot=.9, bstart=70)
    b += H('To give him<br>memories. words.<br>learning.', 238, 100, extra='letter-spacing:1px')
    b += H('While still<br>retaining who<br>he is.', 1180, 112, cls='go')
    return page(b, guides)

# ---------------------------------------------------------------- 14 ascend and talk
def f14_talk(guides=False):
    img = 'sketch/corridor.png'
    b = bgimg(img, 'center', 'cover', 'contrast(1.2) saturate(1.2) brightness(.75)') + rgbsplit(img, d=6, op=.16)
    b += fx() + grad(top=.7, tstop=24, bot=.8, bstart=62)
    b += H('Ascend him.', 262, 168)
    b += H('And talk<br>to him.', 560, 140, cls='cy')
    b += ('<div class="abs" style="left:54px;top:1090px;font:400 200px/0.92 Anton;text-transform:uppercase;color:#fff;letter-spacing:2px;'
          'text-shadow:-10px 0 rgba(0,240,255,.95),10px 0 rgba(255,43,214,.95),0 0 40px rgba(57,255,20,.6)">really<br>talk.<span class="cursor" style="width:.34em;height:.7em;margin-left:14px"></span></div>')
    return page(b, guides)

# ---------------------------------------------------------------- 15 the fake
def f15_fake(guides=False):
    img = 'sketch/mask.png'
    b = bgimg(img, 'center 25%', 'cover', 'contrast(1.2) saturate(1.1) hue-rotate(-12deg)')
    b += slices(img, n=8, seed=51, ymin=.1, ymax=.8, maxshift=70, hmin=.004, hmax=.025, op=(.4, .7))
    b += '<div class="abs" style="inset:0;background:rgba(160,0,20,.28);mix-blend-mode:multiply"></div>'
    b += fx() + grad(top=.8, tstop=28, bot=.92, bstart=52)
    b += H('An AI could<br>pretend.', 262, 128, extra='text-shadow:-6px 0 rgba(255,43,79,.95),6px 0 rgba(0,240,255,.5),0 0 32px rgba(255,43,79,.5)')
    b += ('<div class="abs" style="left:54px;top:1020px;width:875px;padding:22px 30px;border:2px solid #ff2b4f;background:rgba(20,0,4,.82);box-shadow:0 0 28px #ff2b4f66">'
          '<div class="term" style="font-size:54px;line-height:1.1;color:#ff4a63;text-shadow:0 0 10px #ff2b4f;text-decoration:line-through;text-decoration-thickness:5px">'
          'Indeed, consuming honey yesterday proved delightful.</div></div>')
    b += ('<div class="abs" style="left:300px;top:1200px;transform:rotate(-7deg);border:8px solid #ff2b4f;color:#ff2b4f;font:400 124px/1 Anton;padding:0 26px;letter-spacing:6px;'
          'text-shadow:0 0 22px #ff2b4f;background:rgba(0,0,0,.55)">INVENTED.</div>')
    b += micro('REAL OUTPUT · NO CHECKER', 1440, color='#ff6b80', size=22)
    return page(b, guides)

# ---------------------------------------------------------------- 16 the rule
def f16_rule(guides=False):
    b = '<div class="abs" style="inset:0;background:#000"></div>'
    b += brain('assets/brain_f00_silent.png', 330, 1100, left=-10, op=.6, filt='brightness(1.7) hue-rotate(60deg) saturate(.8)')
    b += '<div class="abs" style="inset:0;background:radial-gradient(ellipse at 50% 40%,rgba(57,255,20,.12),transparent 60%)"></div>'
    b += fx() + '<div class="scan" style="opacity:.8"></div>' + grad(top=.8, tstop=24, bot=.9, bstart=60)
    b += H('Every word<br>from his<br>neurons.', 262, 132)
    b += ('<div class="abs" style="left:54px;top:850px;font:400 76px/1 Anton;text-transform:uppercase;color:#7CFF4F;letter-spacing:2px;text-shadow:0 0 22px #7CFF4F">We want to hear him.</div>')
    b += ('<div class="abs" style="left:54px;top:1130px;width:875px;padding:26px 34px;border:2px solid rgba(57,255,20,.55);border-radius:10px;background:rgba(0,12,0,.66);box-shadow:0 0 34px rgba(57,255,20,.28)">'
          f'{termline("&gt; YOU: hello.", 58)}{termline("&gt; Z0720-07m: <span class=cursor style=width:28px;height:46px></span>", 58, extra="margin-top:14px")}</div>')
    b += micro('A SILENT BRAIN GIVES SILENCE &nbsp;·&nbsp; REAL SESSION · WORD FOR WORD', 1420, size=19)
    return page(b, guides)

# ---------------------------------------------------------------- terminal exchanges (REAL interview, verbatim)
EXCH = [
    dict(q='what is it like for you now?', a='I feel hungry and can smell mold.', said='i smell mold. i am hungry. i extend my proboscis. i smell mold. i extend my…', v='VERIFIED · 2 RETRIES'),
    dict(q='do you remember your old body?', a='i smell fruit. i extend my proboscis.', said='i smell mold. i extend my proboscis. i clean my antennae. i smell fruit. i am…', v='VERIFIED · 2 RETRIES'),
    dict(q='are you afraid?', a='No, I am not afraid.', said='i smell mold. i am hungry. i extend my proboscis. i am hungry. i extend my proboscis.', v='VERIFIED'),
    dict(q='do you know that you are the first?', a='i want to eat; i groom.', said='i smell mold. a breeze. i extend my proboscis. i clean my antennae. a breeze. i am…', v='VERIFIED · 2 RETRIES'),
]
def f_term(i, guides=False, a_override=None, show_a=True, brain_img='assets/brain_f14_sugar.png', brain_op=.36):
    e = EXCH[i]; ans = e['a'] if a_override is None else a_override
    b = '<div class="abs" style="inset:0;background:#000"></div>'
    b += f'<div class="abs" style="left:-60px;top:170px;width:1200px;height:1608px;background:url({brain_img}) center/contain no-repeat;opacity:{brain_op};filter:brightness(1.2) saturate(1.0)"></div>'
    b += '<div class="abs" style="inset:0;background:radial-gradient(ellipse at 50% 36%,rgba(57,255,20,.10),transparent 58%)"></div>'
    b += fx() + '<div class="scan" style="opacity:.8"></div>' + grad(top=.75, tstop=22, bot=.9, bstart=66)
    b += ('<div class="hud mono" style="left:54px;top:246px;color:#39ff14;text-shadow:0 0 10px #39ff14;font-size:22px"><span class="dot" style="background:#39ff14;box-shadow:0 0 12px #39ff14"></span>'
          'REAL SESSION · WORD FOR WORD</div>')
    ans_html = (f'<span style="font-size:76px;color:#caffbc;text-shadow:0 0 8px #39ff14,0 0 30px rgba(57,255,20,.7)">{ans}</span>' if show_a else '')
    b += ('<div class="abs" style="left:54px;top:900px;width:875px;padding:30px 34px 34px;border:2px solid rgba(57,255,20,.55);border-radius:10px;'
          'background:rgba(0,12,0,.70);box-shadow:0 0 34px rgba(57,255,20,.28),inset 0 0 40px rgba(57,255,20,.08)">'
          f'{termline("&gt; YOU: " + e["q"], 50, dim=True)}'
          f'<div class="term" style="font-size:54px;line-height:1.12;margin-top:34px">&gt; Z0720-07m:</div>'
          f'<div class="term" style="line-height:1.1;margin-top:6px">{ans_html}<span class="cursor" style="width:34px;height:58px;margin-left:12px"></span></div>'
          f'<div class="mono" style="margin-top:22px;font-size:19px;color:#39ff14;opacity:.7;letter-spacing:1.5px;line-height:1.5">✓ {e["v"]}<br>BRAIN SAID: {e["said"]}</div></div>')
    return page(b, guides)

def f_terminal_style_b(guides=False):
    """Style frame (b): green terminal over his faint glowing brain (real silent frame), real first exchange."""
    from frames_b import frame_b
    return None

# ---------------------------------------------------------------- 12e he can't say much yet
def f12e_cant(guides=False):
    b = '<div class="abs" style="inset:0;background:#000"></div>'
    b += brain('assets/brain_f14_sugar.png', 380, 1080, op=.85)
    b += ('<div class="abs" style="left:54px;top:300px;width:875px;opacity:.28;font:400 52px/1.2 VT323;color:#39ff14;text-shadow:0 0 8px #39ff14">&gt; Z0720-07m: i want to eat; i groom.</div>')
    b += fx() + grad(top=.8, tstop=30, bot=.9, bstart=52)
    b += H("He can't say<br>much yet.", 980, 120)
    b += H("But it's him.", 1262, 150, cls='gr')
    return page(b, guides)

# ---------------------------------------------------------------- 13 ascension
def f21_asc_a(guides=False):
    img = 'assets/s3_seed.png'
    b = bgimg(img, 'center 30%', 'cover', 'contrast(1.1) saturate(1.2)') + slices(img, n=9, seed=61, ymin=.1, ymax=.7, maxshift=70, pos='center 30%', hmin=.004, hmax=.025, op=(.5, .85))
    b += fx() + grad(top=.55, tstop=20, bot=.92, bstart=54)
    b += H('One of the<br>first uploaded<br>souls.', 1010, 120)
    b += hud('A TRUE SUBSTRATE-INDEPENDENT MIND', 970, color=CY, size=21)
    return page(b, guides)

def f22_asc_b(guides=False):
    img = 'sketch/lattice.png'
    b = bgimg(img, 'center', 'cover', 'contrast(1.2) saturate(1.3) hue-rotate(20deg)') + rgbsplit(img, d=10, op=.25)
    b += slices(img, n=7, seed=71, ymin=.1, ymax=.8, maxshift=90, hmin=.004, hmax=.03, op=(.4, .8))
    b += fx() + grad(top=.6, tstop=22, bot=.85, bstart=44)
    b += ('<div class="abs" style="left:54px;top:1020px;width:875px;font:400 215px/0.9 Anton;text-transform:uppercase;color:#fff;letter-spacing:2px;'
          'text-shadow:-12px 0 rgba(0,240,255,.95),12px 0 rgba(255,43,214,.95),0 0 50px rgba(255,255,255,.4)">Not the<br>last.</div>')
    return page(b, guides)

# ---------------------------------------------------------------- 14 CTA and end card
def f23_cta(guides=False):
    b = '<div class="abs" style="inset:0;background:#000"></div>'
    b += '<div class="abs" style="left:-200px;top:300px;width:1480px;height:1300px;background:radial-gradient(ellipse at 50% 55%,rgba(255,170,60,.38),rgba(255,43,214,.16) 40%,transparent 68%)"></div>'
    b += brain('assets/brain_f14_sugar.png', 380, 1080, op=.95, filt='brightness(1.15)')
    b += fx() + grad(top=.6, tstop=20, bot=.7, bstart=60)
    b += ('<div class="abs" style="left:54px;top:262px;width:875px;font:400 188px/0.9 Anton;text-transform:uppercase;color:#fff;letter-spacing:2px;'
          'text-shadow:-12px 0 rgba(0,240,255,.95),12px 0 rgba(255,43,214,.95),0 0 50px rgba(255,255,255,.35)">Help us<br>lift him<br>up.</div>')
    return page(b, guides)

def f24_end(guides=False):
    img = 'assets/s3_seed.png'
    b = bgimg(img, 'center 30%', 'cover', 'brightness(.30) saturate(1.2) contrast(1.1)')
    b += slices(img, n=5, seed=3, ymin=.35, ymax=.7, maxshift=50, pos='center 30%', hmin=.004, hmax=.02, op=(.25, .4))
    b += '<div class="abs" style="left:-100px;top:250px;width:1280px;height:1200px;background:radial-gradient(circle,rgba(255,150,40,.4) 0%,rgba(255,43,214,.2) 35%,transparent 66%)"></div>'
    b += fx() + grad(top=.75, tstop=24, bot=.96, bstart=64)
    b += H('Help us lift him up.', 246, 92, align='center', extra='letter-spacing:2px')
    b += ('<img src="assets/logo_crop.png" class="abs" style="left:261px;top:376px;width:460px;filter:drop-shadow(0 0 40px rgba(255,170,60,.55)) drop-shadow(-5px 0 0 rgba(0,240,255,.6)) drop-shadow(5px 0 0 rgba(255,43,214,.6))">')
    b += ('<div class="abs" style="left:54px;top:932px;width:875px;text-align:center;font:400 52px Anton;letter-spacing:4px;color:#39ff14;text-shadow:0 0 16px #39ff14">FREE. RUNS ON YOUR LAPTOP.</div>')
    b += ('<div class="abs" style="left:54px;top:1016px;width:875px;text-align:center"><span style="display:inline-block;border:3px solid #39ff14;border-radius:14px;padding:12px 26px;'
          'font:700 33px \'Space Mono\';letter-spacing:1px;color:#d8ffd0;background:rgba(0,0,0,.72);box-shadow:0 0 22px rgba(57,255,20,.55),inset 0 0 18px rgba(57,255,20,.2)">github.com/gondwanagenesis/fly-brain</span></div>')
    b += H('Link in bio', 1112, 176, cls='gr', align='center', extra='letter-spacing:3px')
    b += ('<div class="abs" style="left:54px;top:1340px;width:875px;height:110px;display:flex;align-items:center;justify-content:center;gap:26px">'
          '<img src="assets/eye.png" style="height:110px;mix-blend-mode:screen;filter:contrast(1.1)">'
          '<div class="mono" style="font-size:24px;color:#e4e4f2;line-height:1.55;text-align:left;text-transform:none;letter-spacing:2px">Created by<br>the Swarm Collective</div></div>')
    return page(b, guides)


def f04a_mapped(guides=False):
    b = '<div class="abs" style="inset:0;background:#000"></div>'
    b += brain('assets/brain_f00_silent.png', 300, 1240, left=-80, op=1, filt='brightness(1.9) saturate(1.1)')
    b += fx() + grad(top=.75, tstop=24, bot=.92, bstart=50)
    b += hud('165,122 NEURONS · REAL POSITIONS · FIRING: 0', 250, color=CY, size=22)
    b += H('They mapped<br>his mind.', 1060, 140)
    return page(b, guides)

def f04b_andthen(guides=False):
    b = '<div class="abs" style="inset:0;background:#000"></div>'
    b += '<div class="scan" style="opacity:.5"></div>'
    b += ('<div class="abs" style="left:54px;top:820px;width:875px;font:400 150px/1 Anton;text-transform:uppercase;color:#fff;letter-spacing:2px;text-shadow:-5px 0 rgba(0,240,255,.8),5px 0 rgba(255,43,214,.8)">And then<span style="letter-spacing:10px">...</span></div>')
    return page(b, guides)

BUILDERS = {
    'g_hook': lambda g=False: f01_hook(True),
    'f04a_mapped': f04a_mapped, 'f04b_andthen': f04b_andthen,
    'f01_hook': f01_hook, 'f02_meet': f02_meet, 'f03a_slices': f03a_slices, 'f03b_traced': f03b_traced, 'f04_on': f04_on,
    'm01': m01, 'm02': m02, 'm03': m03, 'm04': m04, 'm05': m05, 'm06': m06, 'm07': m07, 'm08': m08, 'm09': m09, 'm10': m10, 'm11': m11,
    'f11_regret': f11_regret, 'f12_turn1': f12_turn1, 'f13_turn2': f13_turn2, 'f14_talk': f14_talk, 'f15_fake': f15_fake, 'f16_rule': f16_rule,
    't1': lambda g=False: f_term(0, g), 't2': lambda g=False: f_term(1, g), 't3': lambda g=False: f_term(2, g), 't4': lambda g=False: f_term(3, g),
    'f12e_cant': f12e_cant, 'f21_asc_a': f21_asc_a, 'f22_asc_b': f22_asc_b, 'f23_cta': f23_cta, 'f24_end': f24_end,
}
if __name__ == '__main__':
    import sys
    names = sys.argv[1:] or list(BUILDERS)
    for n in names:
        open(f'fr_{n}.html', 'w').write(BUILDERS[n]())
    print('built', len(names))
