'use strict';
// 실제 서버 진입과 통제된 추가 학습 상태를 구분하여 기존 DOM 표시를 검사합니다.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const {PNG} = require('pngjs');
const output = path.resolve(__dirname,'../evidence/repetition-ui-20261103');
const checks = [];
function check(name, condition) { assert.ok(condition,name); checks.push(name); }
const fixture = updates => ({state:'training',stage:'finetune',model_id:'amazon/chronos-2',run:'controlled-repeat',
  rep_index:3,total_repetitions:10,completed_repetitions:2,step:123,total_steps:500,loss:.25,device:'cuda:0',rows:1500,
  context_length:256,batch_size:8,lr:.000001,checkpoint_verified:false,evaluation_performed:false,candidate_ready:false,promote:false,
  validation_mae:null,baseline_mae:null,direction_accuracy:null,net_return:null,trades:null,...updates});
const values = node => Object.fromEntries([...node.children].map(row=>[row.querySelector('dt').textContent,row.querySelector('dd').textContent]));
(async()=>{
  fs.mkdirSync(output,{recursive:true});
  const browser = await chromium.launch({headless:true,...(process.env.BROWSER_EXECUTABLE ? {executablePath:process.env.BROWSER_EXECUTABLE} : {})});
  const errors = [];
  try {
    const page = await browser.newPage({viewport:{width:1513,height:1040},colorScheme:'light'});
    page.setDefaultTimeout(15000);
    page.on('pageerror',error=>errors.push(error.message));
    page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
    const response = page.waitForResponse(response=>response.url().endsWith('/api/status') && response.ok());
    await page.goto('http://127.0.0.1:8765/');
    const live = await (await response).json();
    await page.waitForFunction(()=>snapshot !== null && !loading && window.UpbitScene?.stats().training?.ready);
    await page.locator('#auto').uncheck();
    check('실제 서버 기존 계좌 연결',await page.locator('#research-account').innerText().then(t=>t.includes(Math.round(live.account.equity).toLocaleString('ko-KR'))));
    const original = {...live,training_repetition:null,foundation_research:{state:'complete',stage:'complete',model_id:'amazon/chronos-2',
      run:'controlled-foundation-7',step:50,total_steps:50,candidate_ready:true,checkpoint_verified:true,validation_mae:.99}};
    let controlledSnapshot = null;
    await page.route('**/api/status',route=>controlledSnapshot ? route.fulfill({json:controlledSnapshot}) : route.continue());
    const apply = async repetition => {
      controlledSnapshot = {...original,training_repetition:repetition};
      const response = page.waitForResponse(response=>response.url().endsWith('/api/status') && response.ok());
      await page.locator('#refresh').click();
      await response;
      await page.waitForFunction(()=>!loading);
    };
    await apply(fixture());
    check('8차 최신 연구 표시',(await page.locator('#research-track').innerText()).includes('8차'));
    check('현재 회차 제목',(await page.locator('#workflow-title').innerText()).includes('3 / 10'));
    check('전체 스텝과 완료 회차',(await page.locator('#research-progress-detail').innerText()).includes('완료 2회 · 123 / 500'));
    const strip = await page.locator('#learning-strip').evaluate(values);
    check('요약 현재·완료·스텝 정확',strip['현재 회차']==='3 / 10' && strip['완료 회차']==='2 / 10' && strip['전체 스텝']==='123 / 500');
    check('8차 GPU 학습 단계',await page.evaluate(()=>window.UpbitScene.stats().training.stage)===2);
    await page.locator('#training-scene').scrollIntoViewIfNeeded();
    const first = await page.locator('#training-scene').screenshot();
    await page.waitForTimeout(300);
    const second = await page.locator('#training-scene').screenshot();
    check('추가 학습 실제 픽셀 모션',!first.equals(second));
    const colors = new Set(), png = PNG.sync.read(second);
    for(let i=0;i<png.data.length;i+=16)colors.add(`${png.data[i]},${png.data[i+1]},${png.data[i+2]}`);
    check('기존 3D 장면 비어 있지 않음',colors.size>100);
    await page.locator('#motion-toggle').click();
    await page.waitForTimeout(150);
    const stopped = await page.locator('#training-scene').screenshot();
    await page.waitForTimeout(250);
    check('모션 정지 유지',stopped.equals(await page.locator('#training-scene').screenshot()));
    await page.locator('#motion-toggle').click();
    await page.emulateMedia({reducedMotion:'reduce'});
    check('축소 모션 유지',await page.evaluate(()=>window.UpbitScene.stats().training.paused));
    await page.emulateMedia({reducedMotion:'no-preference'});
    await page.locator('#foundation-record').evaluate(node=>{node.open=true});
    let detail = await page.locator('#foundation-details').evaluate(values);
    check('7차 완료 기록 보존',detail['7차 연구 상태 · 보존'].includes('완료') && detail['7차 실행 ID · 보존']==='controlled-foundation-7');
    check('7차 평가와 8차 미평가 분리',detail['7차 상대 MAE · 보존']==='99%' && detail['상대 MAE']==='미평가');
    check('추가 학습률·문맥·배치',detail['학습률']==='0.000001' && detail['문맥 길이']==='256' && detail['배치 크기']==='8');
    for (const stage of ['extract','finetune','verify']) {
      await apply(fixture({stage}));
      check(`학습 단계 표시 ${stage}`,await page.evaluate(()=>window.UpbitScene.stats().training.stage)===({extract:1,finetune:2,verify:3})[stage]);
      if(stage==='verify')check('체크포인트 확인≠성능 검증',(await page.locator('#workflow-title').innerText()).includes('가중치'));
    }
    await apply(fixture({state:'complete',stage:'complete',rep_index:10,completed_repetitions:10,step:500,checkpoint_verified:true}));
    check('완료 문구 정확',await page.locator('#workflow-title').innerText()==='추가 학습 10회 완료 · 성능 평가 전');
    check('검증·적용 완료 색 금지',await page.locator('#workflow-steps li:nth-child(4)').getAttribute('data-state')==='waiting' && await page.locator('#workflow-steps li:nth-child(5)').getAttribute('data-state')==='waiting');
    check('현재 적용 모델과 분리',await page.evaluate(()=>window.UpbitWorkflow.resolve(snapshot).applied)===false);
    detail = await page.locator('#foundation-details').evaluate(values);
    check('모든 성능·수익 항목 미평가',['상대 MAE','종가유지 상대 MAE','방향 정확도','순수익률','완료 거래'].every(key=>detail[key]==='미평가'));
    check('가상 계좌 변경 없음',await page.evaluate(()=>document.getElementById('equity').textContent===money(snapshot.account.equity)));
    await page.locator('#foundation-record').evaluate(node=>{node.open=false});
    for(const [width,height] of [[1513,1040],[390,844],[320,844]]) {
      await page.setViewportSize({width,height});
      await page.evaluate(()=>window.scrollTo(0,0));
      check(`가로 넘침 없음 ${width}`,await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      check(`상태 제목 글자 잘림 없음 ${width}`,await page.locator('#workflow-title').evaluate(node=>node.scrollWidth<=node.clientWidth+1));
      await page.screenshot({path:path.join(output,`complete-${width}.png`)});
    }
    await page.locator('#theme-toggle').click();
    check('기존 다크 전환',await page.locator('html').getAttribute('data-theme')==='dark');
    await page.screenshot({path:path.join(output,'complete-dark-mobile.png')});
    await apply(null);
    check('8차 부재시 현행 연구 유지',(await page.locator('#research-track').innerText()).includes('사전학습 모델'));
    await apply(fixture({step:true}));
    check('잘못된 반복 상태시 현행 유지',await page.evaluate(()=>window.UpbitWorkflow.resolve(snapshot).source.run)==='controlled-foundation-7');
    for(const tab of ['account','trades','data','training']) {
      await page.locator(`#tab-${tab}`).click();
      check(`기존 접근 ${tab}`,await page.locator(`#${tab}`).isVisible());
    }
    check('런타임·콘솔 오류 없음',errors.length===0);
    const report = {passed:checks.length,checks,errors,actual_http_has_repetition:Object.hasOwn(live,'training_repetition'),
      controlled_repetition_states:true,screenshots:output};
    fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));
    console.log(JSON.stringify(report,null,2));
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
