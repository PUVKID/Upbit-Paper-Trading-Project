'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');
const source = fs.readFileSync(path.join(__dirname,'../web/scene.js'),'utf8');
const context = {window:{}};
vm.runInNewContext(source.slice(0,source.indexOf('(() => {')),context);
const resolve = s => context.window.UpbitWorkflow.resolve(s);
const current = updates => ({schema_version:1,state:'running',stage:'fit',run:'return-test',
  fit_calls:3,completed_fits:2,max_fits:30,fold_index:1,fold_total:2,objective:'net_forward_return',
  promotion:false,candidate_ready:false,evaluation_performed:false,...updates});
test('수익 연구는 과거 Chronos 완료 기록보다 우선함',()=>{
  const flow = resolve({return_research:current(),foundation_research:{state:'complete',stage:'complete'}});
  assert.equal(flow.track,'수익 예측 연구');
  assert.equal(flow.stage,2);
  assert.equal(flow.value,2/30);
  assert.equal(flow.applied,false);
});
test('연구 완료는 모델 적용이 아니며 검증 단계까지만 완료함',()=>{
  const flow = resolve({return_research:current({state:'complete',stage:'complete',fit_calls:30,completed_fits:30,fold_index:2,evaluation_performed:true})});
  assert.equal(flow.stage,3);
  assert.equal(flow.active,false);
  assert.equal(flow.applied,false);
  assert.ok(flow.completedStages.includes(3));
  assert.ok(!flow.completedStages.includes(4));
});

test('미적용 연구 후보는 연구 오류가 아니라 채택 보류임',()=>{
  const flow=resolve({return_research:current({state:'complete',stage:'complete',fit_calls:30,completed_fits:30,
    fold_index:2,evaluation_performed:true,research_review_eligible:false})});
  assert.equal(flow.status,'연구 완료 · 채택 보류');
  assert.equal(flow.failed,false);
  assert.equal(flow.applied,false);
});
test('중단과 오류는 활동 중인 GPU 학습으로 보이지 않음',()=>{
  for (const state of ['stopped','failed']) {
    const flow = resolve({return_research:current({state,stage:'failed'})});
    assert.equal(flow.active,false);
    assert.equal(flow.failed,state==='failed');
    assert.equal(flow.stage,3);
  }
});
test('잘못된 연구 상태는 현재 트랙을 탈취하지 못함',()=>{
  for (const changes of [{promotion:true},{fit_calls:true},{run:'/private'},{max_fits:31},
    {state:'complete',stage:'complete',fit_calls:30,completed_fits:30,evaluation_performed:false}]) {
    const flow = resolve({return_research:current(changes),foundation_research:{state:'complete',stage:'complete'}});
    assert.equal(flow.track,'사전학습 모델 연구');
  }
});
