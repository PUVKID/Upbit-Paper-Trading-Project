'use strict';
const $ = id => document.getElementById(id);
const coins = ['BTC', 'SOL', 'ETH', 'XRP'];
let snapshot = null;
let loading = false;
let lastSuccess = null;
const numeric = v => typeof v === 'number' && Number.isFinite(v);
const number = (v, digits = 0) => numeric(v) ? v.toLocaleString('ko-KR', {maximumFractionDigits: digits}) : '—';
const percent = v => numeric(v) ? `${number(v * 100, 2)}%` : '—';
const money = v => numeric(v) ? `${number(v)} 원` : '—';
const text = v => v === null || v === undefined || v === '' ? '대기' : String(v);
const date = v => {
  if (!v) return '대기';
  const d = new Date(v);
  return Number.isNaN(d.getTime()) ? '시각 확인 필요' : d.toLocaleString('ko-KR', {timeZone: 'Asia/Seoul', hour12: false});
};
function set(id, value) { $(id).textContent = value; }
function empty(tbody, columns, label) {
  const row = tbody.insertRow();
  const cell = row.insertCell();
  cell.colSpan = columns;
  cell.className = 'empty';
  cell.textContent = label;
}
function table(id, items, render, columns, label = '저장된 기록 대기') {
  const tbody = $(id);
  tbody.replaceChildren();
  if (!items.length) return empty(tbody, columns, label);
  for (const item of items) {
    const row = tbody.insertRow();
    for (const value of render(item)) row.insertCell().textContent = text(value);
  }
}
function details(id, entries) {
  const fragment = document.createDocumentFragment();
  for (const [label, value] of entries) {
    const group = document.createElement('div');
    const term = document.createElement('dt');
    const description = document.createElement('dd');
    term.textContent = label;
    description.textContent = text(value);
    group.append(term, description);
    fragment.append(group);
  }
  $(id).replaceChildren(fragment);
}
function progress(id, value, label) {
  if (numeric(value)) {
    $(id).value = Math.max(0, Math.min(100, value * 100));
    set(`${id}-text`, `${label} ${number(value * 100, 1)}%`);
  } else {
    $(id).removeAttribute('value');
    set(`${id}-text`, `${label} 대기`);
  }
}
function services(s) {
  const container = $('services');
  container.replaceChildren();
  for (const [key, title] of [['collector', '최신 데이터'], ['backfill', '과거 데이터'], ['trainer', '학습'], ['paper', '모의 운용'], ['health', '운용검증']]) {
    const service = s.services?.[key];
    const health = key === 'health' ? s.health : null;
    const currentState = key === 'health' ? health?.state : service?.state;
    const item = document.createElement('div');
    item.className = 'service';
    const dot = document.createElement('span');
    dot.className = `dot${['실행 중', '검사통과'].includes(currentState) ? ' active' : currentState === '검사실패' ? ' failed' : ''}`;
    dot.setAttribute('aria-hidden', 'true');
    const name = document.createElement('b');
    name.textContent = title;
    const state = document.createElement('span');
    state.textContent = text(currentState);
    const count = document.createElement('small');
    count.textContent = key === 'health'
      ? `최근 검사 ${date(health?.checked_utc)} · 데이터셋 ${number(health?.checked_datasets)} · 오류 ${number(health?.errors_count)} · 대기 ${number(health?.pending_count)} · 검사기 ${text(service?.state)}`
      : service?.iteration == null ? '회차 대기' : `${number(service.iteration)}회차 · 연속 실패 ${number(service.consecutive_failures)}`;
    item.append(dot, name, state, count);
    container.append(item);
  }
}
function distributionText(value) {
  return ['down', 'flat', 'up'].map(key => number(value?.[key])).join(' / ');
}
function sumDistribution(values) {
  const available = values.filter(value => value && typeof value === 'object');
  if (!available.length) return null;
  return Object.fromEntries(['down', 'flat', 'up'].map(key => [key, available.every(value => numeric(value[key])) ? available.reduce((sum, value) => sum + value[key], 0) : null]));
}
const diagnosticLabels = {
  diagnostics: '진단', research: '연구', prediction_distribution: '예측 분포', predicted_distribution: '예측 분포',
  class_distribution: '클래스 분포', label_distribution: '라벨 분포', actual_distribution: '실제 분포',
  probability_quantiles: '확률 분위수', probabilities: '확률', quantiles: '분위수',
  down: '하락', flat: '중립', up: '상승', candidates: '후보', candidate: '후보 번호', candidate_id: '후보 ID',
  rule: '규칙', rules: '규칙', rule_evaluations: '규칙 평가', confidence: '매매 확률 임계값', edge: '매매 우위 임계값',
  trades: '거래 수', fees_paid: '지급 수수료', fees: '수수료', costs: '비용',
  rejection_reasons: '기각 사유', rejection_reason: '기각 사유', reason: '사유', reasons: '사유',
  parameters: '실험 파라미터', params: '실험 파라미터', label_threshold: '라벨 임계값',
  class_weights: '클래스 가중치', depth: '깊이', horizon: '예측 지평',
  net_return: '순수익률', net_pnl: '순손익', MDD: '최대 낙폭', logloss: '로그 손실',
  accuracy: '정확도', macro_f1: '매크로 F1', balanced_accuracy: '균형 정확도',
  promotion: '승격', promotion_reason: '승격 사유', promote: '승격 여부',
  expected_return: '기대 수익률', signal: '신호', guard: '진입 제한', quote_age: '호가 경과',
  buy: '시험 매수', sell: '시험 매도', qty: '수량', cash: '진단 현금', vwap: '체결 평균가',
  fee_paid: '지급 수수료', unfilled: '미체결', partial: '부분 체결', timestamp: '시각',
  event_id: '이벤트 ID', paper_only: '모의 전용', quote_timestamps: '호가 시각', error: '오류 코드',
  argmax_counts: '최대 확률 예측 수 · 하락/중립/상승', signal_counts: '신호 수 · 하락/중립/상승',
  filter_reasons: '중립 필터 사유', no_trade_policy: '무거래 정책', buy_confidence_below: '매수 확신도 미달',
  buy_edge_below: '매수 우위 미달', neutral_signal: '중립 신호', trials: '규칙 후보 평가', trading: '거래 평가',
  objective: '선정 목적값', min_closed_trades: '완료 거래 증거 하한', sequence: '연구 순번',
  class_weight_mode: '클래스 가중치 방식', horizon_minutes: '예측 지평 · 분',
  eligible: '운용 적격', executions: '체결 수', scope: '평가 범위', class_counts: '클래스 표본 수',
  class_factors: '클래스 가중치 계수', cap: '가중치 상한', half_life_days: '가중치 반감기 · 일',
  rule_trials: '규칙별 평가', weighting: '학습 가중치', feature_rows_dropped: '피처 결측 제외 행',
  policy_eligible: '정책 운용 적격', deployment: '운용 승격 하한', eligibility: '운용 적격 근거',
  temperature: '확률 보정 온도', weights: '후보 앙상블 가중치', schema_version: '스키마 버전',
  checked_at: '관측 시각', model_run: '실제 모델 ID', latest_raw_bar: '최근 원시 봉',
  base_timeframe: '기본 시간봉', signal_time: '신호 봉 시각', feature_age_seconds: '피처 경과 · 초',
  argmax: '최대 확률 클래스', inference_diagnostics: '추론 관측', decision_diagnostics: '운용 판단'
};
const diagnosticValues = {
  non_positive_net_return: '비용 차감 순수익 0 이하',
  non_positive_risk_adjusted_return: '위험 조정 수익 0 이하',
  insufficient_closed_trades: '완료 거래 증거 부족', no_trades: '거래 없음',
  no_signal: '신호 없음', unvalidated_or_unprofitable_model: '검증 미달 또는 수익 기준 미달',
  legacy_schema_requires_validation: '이전 스키마 재검증 필요', model_load_error: '운용 모델 읽기 실패',
  no_complete_features: '완성 피처 없음', stale_features: '피처 시각 만료', future_features: '미래 피처 차단',
  duplicate_signal_bar: '중복 신호 봉', no_trade_policy: '무거래 정책',
  model_neutral_or_threshold: '모델 중립 또는 임계값 미달', signal_ready: '모델 신호 준비', inference_error: '추론 오류'
};
function diagnosticRows(value, prefix = '') {
  const result = [];
  function visit(item, path, depth) {
    if (result.length >= 500 || depth > 10) return;
    if (item && typeof item === 'object') {
      for (const [key, child] of Object.entries(item).slice(0, 100)) {
        const label = Array.isArray(item) ? `[${Number(key) + 1}]` : diagnosticLabels[key] || key;
        visit(child, path ? `${path} / ${label}` : label, depth + 1);
      }
    } else if (item !== null && item !== undefined) {
      result.push({path, value: typeof item === 'boolean' ? (item ? '예' : '아니요') : diagnosticValues[item] || text(item).slice(0, 240)});
    }
  }
  visit(value, prefix, 0);
  return result;
}
function renderDiagnostics(s, marketEntries) {
  const cycle = s.services?.trainer || {};
  details('cycle-details', [
    ['현재 프로세스 회차', number(cycle.iteration)], ['성공 완료 횟수', number(cycle.successful_cycles)],
    ['다음 프로세스 회차', number(cycle.next_iteration)], ['성공 후 설정 대기 · 초', number(cycle.interval_seconds, 3)],
    ['직전 성공 실제 종료', date(cycle.previous_finished_utc)], ['직전 성공→현재 시작 · 초', number(cycle.success_gap_seconds, 3)],
    ['현재 프로세스 실제 시작', date(cycle.started_utc)], ['최근 프로세스 실제 종료', date(cycle.finished_utc)],
    ['완료 프로세스 실제 소요 · 초', number(cycle.duration_seconds, 3)]
  ]);
  const trials = marketEntries.flatMap(([market, report]) => (report.diagnostics?.rule_trials || report.research?.trials || []).map((trial, index) => ({market, index, trial})));
  table('rule-trials', trials, ({market, index, trial}) => [market.replace('KRW-', ''), index + 1,
    number(trial.rule?.confidence, 3), number(trial.rule?.edge, 3), number(trial.trading?.trades),
    money(trial.trading?.fees_paid), percent(trial.trading?.net_return), percent(trial.trading?.MDD),
    trial.eligible === true ? '적격' : trial.eligible === false ? '보류' : '대기',
    (trial.rejection_reasons || []).map(value => diagnosticValues[value] || text(value)).join(' · ') || '대기'], 10, '규칙별 평가 기록 대기');
  const research = marketEntries.flatMap(([market, report]) => ['diagnostics', 'research', 'rule', 'weights', 'temperature'].flatMap(kind =>
    diagnosticRows(report[kind]).map(row => ({...row, market, kind: diagnosticLabels[kind]}))));
  research.push(...diagnosticRows(s.research).map(row => ({...row, market: '전체 실험', kind: '연구'})));
  research.push(...diagnosticRows(s.promotion?.eligibility).map(row => ({...row, market: '전체 실험', kind: '승격 보류 근거'})));
  table('research-diagnostics', research.slice(0, 1000), row => [row.market.replace('KRW-', ''), row.kind, row.path, row.value], 4, '실측 진단 기록 대기');
  const decisions = coins.flatMap(coin => ['decision_diagnostics', 'inference_diagnostics'].flatMap(kind =>
    diagnosticRows(s[kind]?.[`KRW-${coin}`], diagnosticLabels[kind]).map(row => ({...row, coin}))));
  table('decision-diagnostics', decisions, row => [row.coin, row.path, row.value], 3, '운용 판단 기록 대기');
  const probe = s.execution_probe || {};
  details('probe-details', [
    ['확인 시각', date(probe.checked_at)], ['시장', text(probe.market)], ['출처', text(probe.source)],
    ['범위', text(probe.scope)], ['경로 통과', probe.passed === true ? '통과' : probe.passed === false ? '실패' : '대기'],
    ['진단 왕복 순손익 · 본 계좌 제외', money(probe.round_trip_net_pnl)], ['진단 수수료', money(probe.fees)]
  ]);
  const paths = Object.fromEntries(['buy', 'sell', 'quote_timestamps', 'error'].filter(key => probe[key] != null).map(key => [key, probe[key]]));
  table('probe-diagnostics', diagnosticRows(paths), row => [row.path, row.value], 2, '체결 경로 진단 기록 대기');
}
function render() {
  if (!snapshot) return;
  const s = snapshot;
  const a = s.account || {};
  services(s);
  set('capital', `${number(s.policy.initial_capital)} KRW`);
  set('start', date(s.policy.paper_start));
  set('equity', money(a.equity));
  set('cash', `현금 ${money(a.cash)}`);
  set('pnl', money(a.total_pnl));
  $('pnl').className = numeric(a.total_pnl) ? (a.total_pnl < 0 ? 'negative' : 'positive') : '';
  set('return', `수익률 ${numeric(a.return_pct) ? number(a.return_pct, 2) + '%' : '대기'}`);
  set('realized', `${number(a.realized_pnl)} / ${number(a.unrealized_pnl)}`);
  set('mdd', numeric(a.mdd_pct) ? `${number(a.mdd_pct, 2)}%` : '—');
  set('quote', `시세 최종 ${date(a.quote_utc)}`);
  const service = s.services?.paper;
  set('schedule', service?.state || (new Date() < new Date(s.policy.paper_start) ? '예약 시각 전' : '실행 상태 대기'));
  $('entry-guard').hidden = !a.entry_guard;
  set('entry-guard', a.entry_guard || '');
  table('assets', coins.map(symbol => ({symbol, ...(s.assets || []).find(row => row.symbol?.replace('KRW-', '') === symbol)})),
    row => [row.symbol, number(row.quantity, 8), money(row.average_price), money(row.price), money(row.market_value), money(row.unrealized_pnl), date(row.quote_utc)], 7);
  const t = s.training || {};
  const gpu = s.gpu || {};
  const promotion = s.promotion || {};
  const selected = $('market-filter').value;
  const marketEntries = Object.entries(s.markets || {}).filter(([market]) => selected === 'all' || selected === market);
  const selectedReport = s.markets?.[selected];
  const validationLabel = s.research?.schema_version === 2 ? '후속 평가 · 정책 선정 이후' : '검증 · 보정 구간';
  renderDiagnostics(s, marketEntries);
  const trainSplits = marketEntries.map(([, report]) => report.train).filter(split => numeric(split?.rows));
  const starts = trainSplits.map(split => split.period_start).filter(value => typeof value === 'string').sort();
  const ends = trainSplits.map(split => split.period_end).filter(value => typeof value === 'string').sort();
  const gpuFits = marketEntries.flatMap(([, report]) => report.gpu || []);
  const states = {running: '진행 중', complete: '완료', failed: '오류', error: '오류'};
  const stages = {features: '피처 구성', fit: '후보 학습', calibration: '확률 보정', locked_test: '잠금 시험', complete: '완료'};
  details('training-details', [
    ['현재 진행 상태', states[t.state] || text(t.state)], ['현재 단계', stages[t.stage] || text(t.stage)],
    ['현재 처리 시장', text(t.market)], ['시장 내 후보 번호', text(t.candidate)],
    ['전체 완료 후보 / 전체 후보', `${number(t.candidate_index)} / ${number(t.candidate_total)}`], ['기본 시간봉', numeric(t.timeframe) ? `${number(t.timeframe)}분` : text(t.timeframe)],
    ['GPU · 장치 전체', text(gpu.gpu_name)], ['GPU 메모리 · 장치 전체', `${number(gpu.gpu_memory_used_mb)} / ${number(gpu.gpu_memory_total_mb)} MiB`],
    ['GPU 사용률 · 장치 전체', numeric(gpu.gpu_utilization_pct) ? `${number(gpu.gpu_utilization_pct)}%` : '대기'], ['GPU 관측', `${text(gpu.state)} · ${date(gpu.checked_utc)}`],
    ['진행 상태 갱신', date(t.updated_utc)], ['학습 후보 실제 시작', date(t.started_utc)],
    ['학습 후보 실제 완료', t.finished_utc ? date(t.finished_utc) : '완료 시각 미기록'],
    ['학습 후보 총 소요', numeric(t.duration_seconds) ? `${number(t.duration_seconds, 2)}초` : '대기'],
    ['보고서 기록 시각', date(t.recorded_utc)], ['현재 진행 시작', date(t.progress_started_utc)],
    ['현재 진행 경과', numeric(t.elapsed_seconds) ? `${number(t.elapsed_seconds, 2)}초` : '대기'],
    ['현재 단계 표본 / 튜닝 표본', `${number(t.rows)} / ${number(t.tune_rows)}`],
    ['특성 수', number(t.features)], ['현재 부스팅 반복 / 최대 반복', `${number(t.current_round)} / ${number(t.target_rounds)}`],
    ['현재 튜닝 로그 손실', number(t.validation_logloss, 5)],
    ['최적화 학습률 · eta', numeric(t.learning_rate) ? number(t.learning_rate, 5) : '미기록'],
    ['실제 학습 데이터 시작', date(starts[0])], ['실제 학습 데이터 종료', date(ends.at(-1))],
    ['실제 학습 표본 수', trainSplits.length ? number(trainSplits.reduce((sum, split) => sum + split.rows, 0)) : '대기'],
    ['최신 학습 후보', text(t.model_id)], ['현재 운용 모델 · 포인터', text(s.current_model?.run)],
    ['최근 모의계좌 적용 모델', text(a.model_run)], ['진행 파일 실행 ID', text(t.run)],
    ['학습 후보 승격 판단', promotion.promote === true ? '승격 대상' : promotion.promote === false ? '승격하지 않음' : '판단 대기'],
    ['승격 사유', text(promotion.reason)],
    ['승격 비교 기준', promotion.comparison_baseline === 'cash' ? '현금 보유 · 수익률 0' : promotion.probability_comparable === true ? '검증된 운용 모델 · 확률·거래 비교' : '기록 대기'],
    ['비교 제외 이전 모델', text(promotion.excluded_incumbent_run)],
    ['후보 · 로그 손실 / 수익률 / MDD', `${number(promotion.candidate?.logloss, 4)} / ${percent(promotion.candidate?.net_return)} / ${percent(promotion.candidate?.MDD)}`],
    ['기존 · 로그 손실 / 수익률 / MDD', `${number(promotion.incumbent?.logloss, 4)} / ${percent(promotion.incumbent?.net_return)} / ${percent(promotion.incumbent?.MDD)}`],
    ['실제 부스터 장치', [...new Set(gpuFits.map(fit => fit.actual_device).filter(Boolean))].join(' · ') || '대기'],
    ['후보 깊이 / 최적 반복', gpuFits.map(fit => `${number(fit.depth)} / ${number(fit.best_iteration)}`).join(' · ') || '대기'],
    ['완료 후보 부스팅 반복', gpuFits.map(fit => number(fit.boosted_rounds)).join(' · ') || '대기'],
    ['완료 후보 소요 시간 · 초', gpuFits.map(fit => number(fit.duration_seconds, 2)).join(' · ') || '대기']
  ]);
  progress('training-progress', t.progress, '전체 후보 진행률');
  const splitRows = marketEntries.flatMap(([market, report]) => ['train', 'tune', 'calibration', 'rule', 'validation'].map(name => ({market, name, split: report.splits?.[name] || (['train', 'tune', 'calibration'].includes(name) ? report[name] : null)})));
  table('training-splits', splitRows, ({market, name, split}) => [market.replace('KRW-', ''), ({train: '학습', tune: '튜닝', calibration: '확률 보정', rule: '정책 선정', validation: '후속 평가'})[name], number(split?.rows), date(split?.period_start), date(split?.period_end), distributionText(split?.class_distribution)], 6);
  table('evaluation', [[validationLabel, selected === 'all' ? s.validation : selectedReport?.validation], ['잠금 시험 · 최종 평가', selected === 'all' ? s.locked_test : selectedReport?.locked_test]],
    ([name, m]) => [name, percent(m?.accuracy), percent(m?.macro_f1), percent(m?.balanced_accuracy), percent(m?.baseline_accuracy), number(m?.samples), `${date(m?.period_start)} ~ ${date(m?.period_end)}`], 7);
  const testState = s.locked_test_state;
  set('evaluation-scope', `평가 구간: ${validationLabel} · 잠금 시험: ${text(testState?.status)} · 후보 선정 사용: ${testState?.used_for_selection === false ? '아니요' : testState?.used_for_selection === true ? '예' : '확인 대기'} · 전체 Macro F1·균형 정확도: 원자료 결합 대기`);
  const marketRows = Object.entries(s.markets || {}).flatMap(([market, report]) => [[validationLabel, report.validation], ['잠금 시험', report.locked_test]].map(([split, metrics]) => ({market, split, metrics})));
  table('market-evaluation', marketRows, ({market, split, metrics: m}) => [market.replace('KRW-', ''), split, percent(m?.accuracy), percent(m?.macro_f1), percent(m?.balanced_accuracy), percent(m?.baseline_accuracy), number(m?.samples), `${date(m?.period_start)} ~ ${date(m?.period_end)}`], 8);
  const distribution = $('distribution');
  distribution.replaceChildren();
  for (const [label, value] of [['학습', sumDistribution(marketEntries.map(([, report]) => report.train?.class_distribution))], [validationLabel, selected === 'all' ? s.class_distribution : selectedReport?.validation?.class_distribution], ['잠금 시험', selected === 'all' ? s.locked_test?.class_distribution : selectedReport?.locked_test?.class_distribution]]) {
    const item = document.createElement('span');
    item.textContent = `${label}: ${distributionText(value)}`;
    distribution.append(item);
  }
  table('rounds', (s.rounds || []).slice(-20).reverse(), row => {
    const validation = selected === 'all' ? row : row.markets?.[selected]?.validation;
    const locked = selected === 'all' ? {accuracy: row.locked_accuracy, samples: row.locked_samples} : row.markets?.[selected]?.locked_test;
    return [text(row.run), row.finished_utc ? date(row.finished_utc) : '완료 시각 미기록', text(row.state), percent(validation?.accuracy), percent(validation?.macro_f1), number(validation?.samples), percent(locked?.accuracy), number(locked?.samples)];
  }, 8);
  const d = s.data || {};
  details('data-details', [['수집 상태', text(d.state)], ['누적 행 수', number(d.total_rows)], ['저장 용량', numeric(d.total_bytes) ? `${number(d.total_bytes / 1024 ** 2, 2)} MiB` : '대기'], ['갱신 시각', date(d.updated_utc)], ['최초 시각', date(d.period_start)], ['최종 시각', date(d.period_end)], ['누락 수', number(d.missing_count)], ['요청 진행', `${number(d.completed_requests)} / ${number(d.total_requests)}`]]);
  progress('data-progress', d.progress, '수집 진행률');
  const datasets = (s.datasets || []).filter(row => $('data-filter').value === 'all' || row.symbol?.replace('KRW-', '') === $('data-filter').value);
  table('datasets', datasets, row => [row.symbol, text(row.timeframe), number(row.rows), numeric(row.bytes) ? `${number(row.bytes / 1024 ** 2, 2)} MiB` : '대기', date(row.period_start), date(row.period_end), number(row.missing_count), text(row.state)], 8);
  const trades = (s.trades || []).filter(row => ($('trade-filter').value === 'all' || row.symbol?.replace('KRW-', '') === $('trade-filter').value) && ($('side-filter').value === 'all' || row.side?.toLowerCase() === $('side-filter').value));
  table('trades-body', trades.slice().reverse(), row => [date(row.timestamp), row.symbol, ({buy: '매수', sell: '매도'})[row.side?.toLowerCase()] || text(row.side), number(row.quantity, 8), money(row.price), money(row.fee), money(row.realized_pnl), text(row.reason)], 8, '조건에 맞는 모의 체결 기록 없음');
  const sources = Object.values(s.sources || {});
  set('source-state', `스냅숏 ${sources.filter(v => v === 'ok').length}개 연결 · ${sources.filter(v => v !== 'ok').length}개 대기/확인 필요`);
  drawCharts();
}
function canvas(id) {
  const element = $(id);
  const rect = element.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  const scale = window.devicePixelRatio || 1;
  element.width = Math.round(rect.width * scale);
  element.height = Math.round(rect.height * scale);
  const ctx = element.getContext('2d');
  ctx.scale(scale, scale);
  ctx.font = '700 11px Nanum, sans-serif';
  return {ctx, width: rect.width, height: rect.height};
}
function drawCharts() {
  if (!snapshot) return;
  const hours = $('range').value;
  const cutoff = hours === 'all' ? -Infinity : Date.now() - Number(hours) * 3600000;
  const points = (snapshot.equity || []).filter(row => numeric(row.equity) && Number.isFinite(Date.parse(row.timestamp)) && Date.parse(row.timestamp) >= cutoff).sort((a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp));
  $('chart-empty').hidden = points.length > 0;
  set('chart-caption', points.length ? `${points.length.toLocaleString('ko-KR')}개 평가 기록 · ${date(points[0].timestamp)} ~ ${date(points.at(-1).timestamp)} · KRW` : '선택 구간의 실제 모의계좌 평가 기록 없음');
  const chart = canvas('equity-chart');
  if (chart) {
    const {ctx, width: w, height: h} = chart;
    ctx.strokeStyle = '#e7ece8';
    for (let i = 0; i < 4; i++) { const y = 20 + i * (h - 50) / 3; ctx.beginPath(); ctx.moveTo(75, y); ctx.lineTo(w - 12, y); ctx.stroke(); }
    if (points.length) {
      const values = points.map(row => row.equity);
      let min = Math.min(...values), max = Math.max(...values);
      const padding = (max - min) * .12 || Math.max(1, Math.abs(max) * .001);
      min -= padding; max += padding;
      const start = Date.parse(points[0].timestamp), end = Date.parse(points.at(-1).timestamp);
      const x = row => 75 + (end === start ? .5 : (Date.parse(row.timestamp) - start) / (end - start)) * Math.max(1, w - 90);
      const y = value => 20 + (max - value) / (max - min) * (h - 50);
      ctx.fillStyle = '#75847b';
      for (let i = 0; i < 4; i++) ctx.fillText(number(max - i * (max - min) / 3), 0, 24 + i * (h - 50) / 3);
      ctx.strokeStyle = '#188467'; ctx.lineWidth = 2;
      ctx.beginPath(); points.forEach((row, i) => i ? ctx.lineTo(x(row), y(row.equity)) : ctx.moveTo(x(row), y(row.equity))); ctx.stroke();
      if (points.length === 1) { ctx.fillStyle = '#188467'; ctx.beginPath(); ctx.arc(x(points[0]), y(points[0].equity), 4, 0, Math.PI * 2); ctx.fill(); }
    }
  }
  const datasets = (snapshot.datasets || []).filter(row => numeric(row.rows) && ($('data-filter').value === 'all' || row.symbol?.replace('KRW-', '') === $('data-filter').value));
  $('data-empty').hidden = datasets.length > 0;
  const bars = canvas('data-chart');
  if (bars && datasets.length) {
    const totals = coins.map(symbol => ({symbol, rows: datasets.filter(row => row.symbol?.replace('KRW-', '') === symbol).reduce((sum, row) => sum + row.rows, 0)}));
    const max = Math.max(1, ...totals.map(row => row.rows));
    const {ctx, width: w, height: h} = bars;
    totals.forEach((row, i) => { const y = 12 + i * (h - 20) / 4; ctx.fillStyle = '#63756a'; ctx.fillText(row.symbol, 0, y + 14); ctx.fillStyle = ['#188467', '#548dac', '#8474a7', '#b29b51'][i]; ctx.fillRect(48, y, (w - 145) * row.rows / max, 20); ctx.fillStyle = '#45594c'; ctx.fillText(number(row.rows), Math.max(50, 56 + (w - 145) * row.rows / max), y + 14); });
  }
}
async function refresh() {
  if (loading) return;
  loading = true;
  $('refresh').disabled = true;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch('/api/status', {signal: controller.signal, cache: 'no-store'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    snapshot = await response.json();
    lastSuccess = new Date().toISOString();
    $('error').hidden = true;
    set('connection', '로컬 서버 연결됨');
    set('last-refresh', `${date(lastSuccess)} KST`);
    render();
  } catch (error) {
    set('connection', '연결 확인 필요');
    $('error').hidden = false;
    $('error').textContent = `현재 상태를 가져오지 못했습니다. ${lastSuccess ? '마지막 수신 자료를 표시합니다. ' : ''}${error.name === 'AbortError' ? '응답 시간 초과' : error.message}`;
  } finally { clearTimeout(timeout); loading = false; $('refresh').disabled = false; }
}
function openTab(button) {
  for (const tab of document.querySelectorAll('[role=tab]')) {
    const selected = tab === button;
    tab.setAttribute('aria-selected', String(selected)); tab.tabIndex = selected ? 0 : -1;
    $(tab.dataset.tab).hidden = !selected;
  }
  drawCharts();
}
const tabs = [...document.querySelectorAll('[role=tab]')];
tabs.forEach((button, index) => {
  button.addEventListener('click', () => openTab(button));
  button.addEventListener('keydown', event => {
    const offset = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
    if (!offset && !['Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const next = tabs[event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + offset + tabs.length) % tabs.length];
    openTab(next); next.focus();
  });
});
['range', 'data-filter', 'trade-filter', 'side-filter', 'market-filter'].forEach(id => $(id).addEventListener('change', render));
$('refresh').addEventListener('click', refresh);
window.addEventListener('resize', drawCharts);
document.fonts.ready.then(drawCharts);
setInterval(() => { if ($('auto').checked && !document.hidden) refresh(); }, 5000);
refresh();
