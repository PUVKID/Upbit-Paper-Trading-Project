'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const {test}=require('node:test');
const source=fs.readFileSync(path.join(__dirname,'../web/scene.js'),'utf8');
const context={window:{}};
vm.runInNewContext(source.slice(0,source.indexOf('(() => {')),context);
const current=changes=>({schema_version:1,study:'pretrained_forecast',model_id:'google/timesfm-2.5-200m-pytorch',
  run:'pretrained-test',state:'running',stage:'predict',fit_calls:0,promotion:false,candidate_ready:false,
  total_windows:100,windows_completed:16,evaluation_performed:false,...changes});
const resolve=value=>context.window.UpbitWorkflow.resolve({pretrained_benchmark:value});
test('사전학습 예측 진행률은 학습 적합 횟수가 아님',()=>{
  const flow=resolve(current());
  assert.equal(flow.stage,2);assert.equal(flow.value,.16);
  assert.equal(flow.benchmark.fit_calls,0);assert.equal(flow.applied,false);
});
test('평가 완료와 오류·중단은 적용이나 활성 학습이 아님',()=>{
  for(const state of ['complete','error','stopped']) {
    const flow=resolve(current({state,stage:state==='complete'?'complete':'failed',windows_completed:100,evaluation_performed:true}));
    assert.equal(flow.active,false);assert.equal(flow.applied,false);
  }
});
test('미검증 완료와 잘못된 예산은 기존 트랙을 가리지 않음',()=>{
  for(const change of [{total_windows:6001},{fit_calls:1},{windows_completed:true},{state:'complete'}])
    assert.equal(resolve(current(change)).benchmark,undefined);
});
