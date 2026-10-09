const { chromium } = require('playwright');
(async () => {
  const dir = process.argv[2]; const names = process.argv.slice(3);
  const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1080, height: 1920 } });
  for (const n of names) {
    await p.goto('file://' + dir + '/fr_' + n + '.html'); await p.evaluate(() => document.fonts.ready); await p.waitForTimeout(250);
    await p.addStyleTag({ content: '*{background:none!important;border-color:transparent!important;box-shadow:none!important;text-shadow:none!important;filter:none!important;color:#fff!important;-webkit-text-stroke:0!important;mix-blend-mode:normal!important;opacity:1!important;mask-image:none!important;clip-path:none!important} html,body,.frame{background:#000!important} img,svg,.bg,.slice,.scan,.vig,.noise,i,.guides,.cursor{display:none!important}' });
    await p.screenshot({ path: dir + '/ink/' + n + '.png' });
  }
  await b.close();
})();
