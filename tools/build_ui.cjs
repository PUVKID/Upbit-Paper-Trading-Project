// 브라우저가 외부 CDN 없이 사용할 의존성과 코인 자산을 로컬에 묶습니다.
const fs = require('node:fs');
const path = require('node:path');
const esbuild = require('esbuild');
const root = path.resolve(__dirname, '..');
const web = path.join(root, 'web');
const assets = path.join(web, 'assets');
const licenses = path.join(web, 'licenses');
fs.mkdirSync(assets, {recursive:true});
fs.mkdirSync(licenses, {recursive:true});
esbuild.buildSync({entryPoints:[path.join(web,'vendor-entry.js')],
  outfile:path.join(web,'vendor.bundle.js'),bundle:true,minify:true,format:'iife',
  platform:'browser',target:'es2020',legalComments:'eof',loader:{'.png':'dataurl'}});
for (const coin of ['btc','sol','eth','xrp','doge']) {
  fs.copyFileSync(path.join(root,'node_modules','cryptocurrency-icons','128','color',coin+'.png'),path.join(assets,coin+'.png'));
}
for (const [name,file] of [['three','LICENSE'],['lucide','LICENSE'],['cryptocurrency-icons','LICENSE.md']]) {
  fs.copyFileSync(path.join(root,'node_modules',name,file),path.join(licenses,name+'.txt'));
}
console.log(JSON.stringify({bundle_bytes:fs.statSync(path.join(web,'vendor.bundle.js')).size,
  coins:5,external_runtime_requests:0}));
