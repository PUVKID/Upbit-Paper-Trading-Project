'use strict';
// 실제 읽기 전용 API와 통제된 화면 입력을 구분하여 검사합니다.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const {PNG} = require('pngjs');
const output = process.env.FRONTEND_EVIDENCE_DIR || path.resolve(__dirname,'../7차/frontend-evidence/repair61-shadow-contract');
const checks = [];
function check(name, test) { assert.ok(test,name); checks.push(name); }
function pixels(buffer) {
  const png = PNG.sync.read(buffer), colors = new Set();
  for (let i=0;i<png.data.length;i+=16) colors.add(`${png.data[i]},${png.data[i+1]},${png.data[i+2]}`);
  return colors.size;
}
(async () => {
  fs.mkdirSync(output,{recursive:true});
  const browser = await chromium.launch({headless:true,...(process.env.BROWSER_EXECUTABLE ? {executablePath:process.env.BROWSER_EXECUTABLE} : {})});
  const failures = [];
  try {
    const page = await browser.newPage({viewport:{width:1513,height:1040},colorScheme:'light',reducedMotion:'no-preference'});
    page.setDefaultTimeout(15000);
    page.on('pageerror',error => failures.push(error.message));
    page.on('console',msg => { if (msg.type()==='error') failures.push(msg.text()); });
    await page.addInitScript(()=>{ if (localStorage.getItem('upbit-theme') === null) localStorage.setItem('upbit-theme','light'); });
    const pending = page.waitForResponse(response=>response.url().endsWith('/api/status') && response.status()===200);
    await page.goto('http://127.0.0.1:8765/');
    const live = await (await pending).json();
    await page.waitForFunction(()=>window.UpbitScene?.stats().training?.ready && !loading && snapshot !== null && document.getElementById('last-refresh').textContent.includes('KST'));
    await page.locator('#auto').uncheck();
    await page.evaluate(()=>document.fonts.ready);
    check('실제 API 계좌 유지',await page.locator('#research-account').innerText().then(t=>t.includes(Math.round(live.account.equity).toLocaleString('ko-KR'))));
    check('미존재 사전학습 상태',Object.keys(live.foundation_research || {}).length > 0 || await page.locator('#foundation-state').innerText()==='미실행');
    check('브랜드 이름 제거',!(await page.locator('.sidebar').innerText()).includes('PUVKID'));
    check('삭제한 정적 파일 요청 없음',await page.locator('script[src="/workflow.js"]').count()===0);
    for (const width of [1513,1024,390,320]) {
      await page.setViewportSize({width,height:width<500?844:1040});
      await page.evaluate(()=>window.scrollTo(0,0));
      await page.waitForTimeout(300);
      check(`가로 넘침 없음 ${width}`,await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      check(`5단계 글자 잘림 없음 ${width}`,await page.locator('#workflow-steps b').evaluateAll(nodes=>nodes.every(n=>n.scrollWidth<=n.clientWidth+1 && n.scrollHeight<=n.clientHeight+1)));
      const shot = await page.locator('#training-scene').screenshot();
      check(`캔버스 실제 픽셀 ${width}`,pixels(shot)>100);
      await page.evaluate(()=>window.scrollTo(0,0));
      await page.screenshot({path:path.join(output,`live-${width}.png`),fullPage:false});
    }
    await page.setViewportSize({width:1513,height:1040});
    await page.evaluate(()=>window.scrollTo(0,0));
    for (const [stage,index] of [['download',0],['features',1],['fit',2],['validation',3],['apply',4]]) {
      await page.evaluate(({live,stage})=>{snapshot={...live,training_repetition:null,foundation_research:{state:'running',model_id:'controlled-candidate',stage,step:0,total_steps:20,loss:0,candidate_ready:false,has_error:false}};render();},{live,stage});
      check(`통제 단계 ${stage}`,await page.evaluate(()=>window.UpbitScene.stats().training.stage)===index);
      await page.locator('#training-scene').scrollIntoViewIfNeeded();
      await page.waitForTimeout(180);
      const first = await page.locator('#training-scene').screenshot();
      await page.waitForTimeout(280);
      const second = await page.locator('#training-scene').screenshot();
      check(`실제 픽셀 모션 ${stage}`,!first.equals(second));
    }
    await page.locator('#foundation-record').evaluate(node=>{node.open=true});
    const zero = await page.locator('#foundation-details').evaluate(node=>Object.fromEntries([...node.children].map(row=>[row.querySelector('dt').textContent,row.querySelector('dd').textContent])));
    check('0 스텝과 0 손실 유지',zero['스텝']==='0 / 20' && zero['손실']==='0');
    await page.locator('#foundation-record').evaluate(node=>{node.open=false});
    await page.locator('#motion-toggle').click();
    await page.waitForTimeout(180);
    const frozen = await page.locator('#training-scene').screenshot();
    await page.waitForTimeout(250);
    check('화면 모션 정지',frozen.equals(await page.locator('#training-scene').screenshot()));
    await page.locator('#motion-toggle').click();
    await page.emulateMedia({reducedMotion:'reduce'});
    await page.waitForTimeout(180);
    const reduced = await page.locator('#training-scene').screenshot();
    await page.waitForTimeout(250);
    check('모션 축소 정지',reduced.equals(await page.locator('#training-scene').screenshot()));
    await page.emulateMedia({reducedMotion:'no-preference'});
    await page.evaluate(live=>{snapshot={...live,training_repetition:null,foundation_research:{state:'failed',stage:'fit',has_error:true,model_id:'<img src=x onerror=alert(1)>'}};render();},live);
    check('오류 상태 표시',await page.locator('#research-state').innerText()==='오류');
    check('입력 HTML 비실행',await page.locator('#foundation-details img').count()===0);
    await page.evaluate(live=>{snapshot={...live,training_repetition:null,foundation_research:{state:'complete',stage:'complete',candidate_ready:true}};render();},live);
    check('후보와 적용 구분',(await page.locator('#foundation-state').innerText()).includes('적용 아님'));
    check('후보 준비를 매매 적용으로 표시하지 않음',await page.locator('#workflow-steps li:last-child[data-state=complete]').count()===0);
    check('사전학습 완료는 검증 단계',await page.evaluate(()=>window.UpbitScene.stats().training.stage)===3);
    for (const [stage,index] of [['preflight',1],['extract',1],['load',1],['finetune',2],['verify',3],['validation',3]]) {
      await page.evaluate(({live,stage})=>{snapshot={...live,training_repetition:null,foundation_research:{state:'running',stage}};render();},{live,stage});
      check(`실제 계약 단계 ${stage}`,await page.evaluate(()=>window.UpbitScene.stats().training.stage)===index);
      const title = await page.locator('#workflow-title').innerText();
      check(`실제 계약 작업 제목 ${stage}`,index===1 ? title.includes('정리') : index===2 ? title.includes('학습') : title.includes('검증'));
    }
    for (const [state,label] of [['waiting','새 미사용 자료 대기'],['stopped','중단']]) {
      await page.evaluate(({live,state})=>{snapshot={...live,training_repetition:null,foundation_research:{state}};render();},{live,state});
      check(`실제 계약 상태 ${state}`,await page.locator('#foundation-state').innerText()===label);
      check(`실제 계약 비활성 모션 ${state}`,await page.evaluate(()=>window.UpbitScene.stats().training.paused));
    }
    await page.evaluate(live=>{snapshot={...live,training_repetition:null,foundation_research:{state:'complete',stage:'complete',candidate_ready:true,checkpoint_verified:true,
      validation_mae:.02,baseline_mae:.03,direction_accuracy:.625,net_return:null,trades:null}};render();},live);
    await page.locator('#foundation-record').evaluate(node=>{node.open=true});
    const metrics = await page.locator('#foundation-details').evaluate(node=>Object.fromEntries([...node.children].map(row=>[row.querySelector('dt').textContent,row.querySelector('dd').textContent])));
    check('상대 MAE 정규화 비율 표시',metrics['상대 MAE']==='2%');
    check('종가유지 상대 MAE 정규화 비율 표시',metrics['종가유지 상대 MAE']==='3%');
    check('방향 정확도 비율 표시',metrics['방향 정확도']==='62.5%');
    check('미평가 수익·거래 유지',metrics['순수익률']==='미평가' && metrics['완료 거래']==='미평가');
    check('연구 입력·예측 구분',metrics['입력·예측'].includes('5분봉 종가') && metrics['입력·예측'].includes('12스텝'));
    check('기존 모델 설명 구분',(await page.locator('.learning-now h2').innerText()).includes('XGBoost'));
    check('후보와 체크포인트 상태 표시',(await page.locator('#foundation-summary').innerText()).includes('검증됨') && (await page.locator('#foundation-summary').innerText()).includes('매매 적용 아님'));
    await page.evaluate(live=>{snapshot={...live,training_repetition:null,training:{...live.training,rows:999999,features:888},foundation_research:{state:'training',stage:'finetune',step:7,total_steps:50,
      rows:1234,device:'cuda:0',context_length:512,batch_size:8,learning_rate:.00001}};render();},live);
    const strip = await page.locator('#learning-strip').evaluate(node=>Object.fromEntries([...node.children].map(row=>[row.querySelector('dt').textContent,row.querySelector('dd').textContent])));
    check('현재 사전학습 50스텝 표시',strip['현재 스텝']==='7 / 50');
    check('사전학습 실제 자료·장치 표시',strip['학습 자료']==='1,234행' && strip['장치']==='cuda:0');
    check('사전학습 문맥·배치 표시',strip['문맥 길이']==='512개' && strip['배치 크기']==='8개');
    check('XGBoost 회차·특징 요약 혼입 없음',!Object.hasOwn(strip,'학습 회차') && !Object.hasOwn(strip,'학습 항목 · 최근 집계') && !Object.values(strip).some(value=>value.includes('999,999')));
    check('사전학습 실제 학습률 상세',(await page.locator('#foundation-details').innerText()).includes('0.00001'));
    const unevaluated = await page.locator('#foundation-details').evaluate(node=>Object.fromEntries([...node.children].map(row=>[row.querySelector('dt').textContent,row.querySelector('dd').textContent])));
    check('상대 MAE 미평가 유지',unevaluated['상대 MAE']==='미평가' && unevaluated['종가유지 상대 MAE']==='미평가');
    await page.evaluate(live=>{snapshot={...live,training_repetition:null,foundation_research:{state:'complete',stage:'complete',
      checkpoint_verified:true,candidate_ready:true,has_status_warning:true}};render();},live);
    check('완료 상태 저장 경고는 학습 실패와 구분',
      (await page.locator('#foundation-state').innerText()).includes('상태 기록 확인 필요') &&
      (await page.locator('#workflow-title').innerText()).includes('연구를 마쳤습니다'));
    await page.locator('#foundation-record').evaluate(node=>{node.open=false});
    await page.locator('#theme-toggle').click();
    check('다크 전환',await page.locator('html').getAttribute('data-theme')==='dark');
    await page.reload();
    await page.waitForFunction(()=>window.UpbitScene?.stats().training?.ready && !loading && snapshot !== null && document.getElementById('last-refresh').textContent.includes('KST'));
    check('저장 테마 유지',await page.locator('html').getAttribute('data-theme')==='dark');
    await page.locator('#auto').uncheck();
    await page.screenshot({path:path.join(output,'live-dark.png'),fullPage:false});
    for (const tab of ['account','data','trades','training']) {
      await page.locator(`#tab-${tab}`).click();
      check(`기존 탭 ${tab}`,await page.locator(`#${tab}`).isVisible());
    }
    await page.locator('#training-raw').evaluate(n=>{n.open=true});
    await page.locator('#market-filter').selectOption('KRW-DOGE');
    check('기존 종목 필터 유지',await page.locator('#market-filter').inputValue()==='KRW-DOGE');
    const evaluation = await page.evaluate(()=>({expected:percent(snapshot.markets?.['KRW-DOGE']?.validation?.accuracy),actual:document.querySelector('#evaluation tr')?.cells[1]?.textContent}));
    check('후속 평가 원본 정확도 표시 유지',evaluation.actual===evaluation.expected);
    await page.setViewportSize({width:390,height:844});
    for (const symbol of ['BTC','SOL','ETH','XRP','DOGE']) {
      await page.locator('#tab-training').click();
      await page.locator('#market-filter').selectOption(`KRW-${symbol}`);
      check(`모바일 학습 필터 ${symbol}`,await page.locator('#market-filter').inputValue()===`KRW-${symbol}`);
      await page.locator('#tab-data').click();
      await page.locator('#data-filter').selectOption(symbol);
      const data = await page.evaluate(symbol=>({expected:snapshot.datasets.filter(row=>row.symbol?.replace('KRW-','')===symbol).length,
        rows:[...document.querySelectorAll('#datasets tr')].filter(row=>!row.querySelector('.empty')).map(row=>row.cells[0].textContent)}),symbol);
      check(`모바일 실제 자료행 ${symbol}`,data.rows.length===data.expected && data.rows.every(value=>value.replace('KRW-','')===symbol));
      await page.locator('#tab-trades').click();
      await page.locator('#trade-filter').selectOption(symbol);
      await page.locator('#side-filter').selectOption('all');
      const trades = await page.evaluate(symbol=>({expected:snapshot.trades.filter(row=>row.symbol?.replace('KRW-','')===symbol).length,
        rows:[...document.querySelectorAll('#trades-body tr')].filter(row=>!row.querySelector('.empty')).map(row=>row.cells[1].textContent)}),symbol);
      check(`모바일 실제 체결행 ${symbol}`,trades.rows.length===trades.expected && trades.rows.every(value=>value.replace('KRW-','')===symbol));
    }
    await page.locator('#tab-account').click();
    check('모바일 계좌 기존 평가 유지',await page.evaluate(()=>document.getElementById('equity').textContent===money(snapshot.account.equity)));
    check('모바일 가로 넘침 없음',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    await page.locator('#tab-training').click();
    await page.locator('#training-raw').evaluate(node=>{node.open=false});
    await page.locator('#theme-toggle').click();
    await page.evaluate(()=>window.scrollTo(0,0));
    await page.screenshot({path:path.join(output,'mobile-light-final.png')});
    await page.locator('#training-scene').evaluate(canvas=>canvas.dispatchEvent(new Event('webglcontextlost',{cancelable:true})));
    check('WebGL 연결 소실 문자 대체',await page.locator('#training-scene-fallback').isVisible() && !(await page.locator('#training-scene').isVisible()));
    check('WebGL 연결 소실 상태 유지',(await page.locator('#workflow-status').innerText()).length>0);
    const fallbackContext = await browser.newContext({viewport:{width:390,height:844},colorScheme:'light'});
    await fallbackContext.addInitScript(()=>{
      const original = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = function(kind,...args) {
        return ['webgl','webgl2','experimental-webgl'].includes(kind) ? null : original.call(this,kind,...args);
      };
    });
    const fallbackPage = await fallbackContext.newPage();
    const fallbackErrors = [];
    fallbackPage.on('pageerror',error=>fallbackErrors.push(error.message));
    await fallbackPage.goto('http://127.0.0.1:8765/');
    await fallbackPage.waitForFunction(()=>snapshot !== null && !loading && document.getElementById('last-refresh').textContent.includes('KST'));
    await fallbackPage.locator('#auto').uncheck();
    check('WebGL 미지원 문자 대체',await fallbackPage.locator('#training-scene-fallback').isVisible() && await fallbackPage.locator('#workflow-steps li').count()===5);
    check('WebGL 미지원 실제 계좌 유지',await fallbackPage.evaluate(()=>document.getElementById('research-account').textContent.includes(money(snapshot.account.equity))));
    check('WebGL 미지원 가로 넘침 없음',await fallbackPage.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    check('WebGL 미지원 런타임 오류 없음',fallbackErrors.length===0);
    await fallbackPage.screenshot({path:path.join(output,'mobile-webgl-fallback.png')});
    await fallbackContext.close();
    check('브라우저 오류 없음',failures.length===0);
    console.log(JSON.stringify({passed:checks.length,checks,errors:failures,api:true,controlled_states:true,screenshots:output},null,2));
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
