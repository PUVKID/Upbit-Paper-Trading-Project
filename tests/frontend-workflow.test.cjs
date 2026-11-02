'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');
const root = path.resolve(__dirname,'..');
const source = fs.readFileSync(path.join(root,'web/scene.js'),'utf8');
// 장면 초기화 직전의 실제 공통 상태 해석기를 검사합니다.
const boundary = source.indexOf('(() => {');
assert.ok(boundary > 0);
const context = {window:{}};
vm.runInNewContext(source.slice(0,boundary),context);
const resolve = snapshot => context.window.UpbitWorkflow.resolve(snapshot);

test('삭제한 workflow.js 정적 경로를 참조하지 않음',()=>{
  const html = fs.readFileSync(path.join(root,'web/index.html'),'utf8');
  assert.doesNotMatch(html, /src=["']\/workflow\.js["']/);
});
test('직전 다른 연구의 승격을 현재 연구 적용으로 표시하지 않음',()=>{
  assert.equal(resolve({training:{run:'current',state:'running',stage:'fit'},research:{run:'previous',promoted:true}}).applied,false);
});
test('회차가 없으면 양쪽 값이 비어 있어도 적용으로 표시하지 않음',()=>{
  for (const training of [{},{run:''},{run:null}]) {
    assert.equal(resolve({training,research:{promoted:true}}).applied,false);
  }
});
test('같은 명시적 회차와 승격 기록이 있을 때만 기존 모델 적용 표시',()=>{
  assert.equal(resolve({training:{run:'current',state:'complete',stage:'complete'},research:{run:'current',promoted:true}}).applied,true);
  assert.equal(resolve({training:{run:'current'},research:{run:'current',promoted:false}}).applied,false);
});
test('사전학습 후보는 기존 승격 기록과 회차가 같아도 매매 적용으로 표시하지 않음',()=>{
  assert.equal(resolve({training:{run:'same'},research:{run:'same',promoted:true},foundation_research:{state:'complete',run:'same',candidate_ready:true}}).applied,false);
});
test('실제 shadowrunner 준비·학습·검증 단계 매핑',()=>{
  for (const [stage,expected] of [['preflight',1],['extract',1],['load',1],['finetune',2],['verify',3],['validation',3]]) {
    const flow = resolve({foundation_research:{state:'running',stage}});
    assert.equal(flow.stage,expected,stage);
    assert.equal(flow.active,true,stage);
    assert.equal(flow.status,'진행 중',stage);
  }
});
test('사전학습 complete는 검증 완료이며 매매 적용 단계가 아님',()=>{
  const flow = resolve({foundation_research:{state:'complete',stage:'complete',candidate_ready:true,checkpoint_verified:true}});
  assert.equal(flow.stage,3);
  assert.equal(flow.applied,false);
  assert.equal(flow.active,false);
});
test('사전학습 waiting과 stopped를 명확히 구분',()=>{
  assert.equal(resolve({foundation_research:{state:'waiting'}}).status,'새 미사용 자료 대기');
  assert.equal(resolve({foundation_research:{state:'stopped'}}).status,'중단');
});

const repetition = updates => ({state:'training',stage:'finetune',model_id:'amazon/chronos-2',run:'repeat-run',rep_index:3,
  total_repetitions:10,completed_repetitions:2,step:123,total_steps:500,loss:.4,rows:1000,device:'cuda:0',
  context_length:256,batch_size:8,lr:.000001,checkpoint_verified:false,evaluation_performed:false,candidate_ready:false,promote:false,...updates});
test('반복 학습이 유효하면 7차 완료와 혼합하지 않고 최신 트랙으로 선택',()=>{
  const old = {state:'complete',stage:'complete',run:'old',checkpoint_verified:true,candidate_ready:true};
  const flow = resolve({foundation_research:old,training_repetition:repetition()});
  assert.equal(flow.track,'Chronos 추가 학습');
  assert.equal(flow.source.run,'repeat-run');
  assert.equal(flow.source.step,123);
  assert.equal(flow.value,123/500);
  assert.equal(flow.source.learning_rate,.000001);
  assert.equal(flow.foundation.run,'old');
  assert.equal(flow.applied,false);
});
test('반복 학습 완료는 GPU 추가 학습 완료이며 성능 검증 완료가 아님',()=>{
  const flow = resolve({training_repetition:repetition({state:'complete',stage:'complete',rep_index:10,completed_repetitions:10,step:500,checkpoint_verified:true})});
  assert.equal(flow.stage,2);
  assert.equal(flow.active,false);
  assert.equal(flow.applied,false);
  assert.ok(!flow.completedStages.includes(3));
  assert.ok(!flow.completedStages.includes(4));
  for (const key of ['validation_mae','baseline_mae','direction_accuracy','net_return','trades']) assert.equal(flow.source[key],null);
});
test('반복 학습 부재·잘못된 핵심 상태는 기존 사전학습 트랙 유지',()=>{
  const base = {foundation_research:{state:'complete',stage:'complete',run:'old'}};
  for (const current of [undefined,{},repetition({step:true}),repetition({total_steps:50}),repetition({candidate_ready:true}),repetition({evaluation_performed:true})]) {
    const flow = resolve({...base,training_repetition:current});
    assert.equal(flow.track,'사전학습 모델 연구');
    assert.equal(flow.source.run,'old');
  }
});

test('검증된 연구 완료는 이전 준비·학습도 완료하며 매매 적용은 제외',()=>{
  const complete = resolve({foundation_research:{state:'complete',stage:'complete',checkpoint_verified:true}});
  assert.equal(Array.from(complete.completedStages).join(','),'0,1,2,3');
  assert.equal(complete.applied,false);
  for (const state of ['training','waiting','error','stopped']) {
    assert.equal(resolve({foundation_research:{state,checkpoint_verified:true}}).completedStages.length,0);
  }
  assert.equal(resolve({foundation_research:{state:'complete',checkpoint_verified:false}}).completedStages.length,0);
});
