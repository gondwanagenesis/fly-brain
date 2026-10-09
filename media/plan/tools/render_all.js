const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const dir = process.argv[2]; const names = process.argv.slice(3);
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1080, height: 1920 }, deviceScaleFactor: 1 });
  // warm the font cache once
  for (const n of names) {
    await p.goto('file://' + dir + '/fr_' + n + '.html');
    await p.evaluate(() => document.fonts.ready);
    await p.waitForTimeout(700);
    await p.evaluate(() => document.fonts.ready);
    await p.screenshot({ path: dir + '/out/' + n + '.png' });
    console.log('ok', n);
  }
  await b.close();
})();
