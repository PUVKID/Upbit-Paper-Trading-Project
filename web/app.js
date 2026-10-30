'use strict';
const $ = id => document.getElementById(id);
const coins = ['BTC', 'SOL', 'ETH', 'XRP', 'DOGE'];
const coinNames = {BTC: '비트코인', SOL: '솔라나', ETH: '이더리움', XRP: '리플', DOGE: '도지코인'};
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
  for (const [label, value, id] of entries) {
    const group = document.createElement('div');
    const term = document.createElement('dt');
    const description = document.createElement('dd');
    if (id) description.id = id;
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
function healthStatus(s) {
  const worker = s.services?.health || {};
  const checked = Date.parse(s.health?.checked_utc);
  const age = (Date.now() - checked) / 1000;
  const workerError = /오류|실패|재시도/.test(worker.state || '') || (numeric(worker.consecutive_failures) && worker.consecutive_failures > 0);
  const delayed = workerError || (Number.isFinite(age) && age > 600);
  return {delayed, label: delayed ? '점검 지연 · 재시도 중' : !Number.isFinite(checked) ? '점검 기록 대기' : text(s.health?.state)};
}
function services(s) {
  const container = $('services');
  container.replaceChildren();
  for (const [key, title] of [['collector', '최신 데이터'], ['backfill', '과거 데이터'], ['trainer', '학습'], ['paper', '모의 운용'], ['health', '운용검증']]) {
    const service = s.services?.[key];
    const health = key === 'health' ? s.health : null;
    const currentHealth = key === 'health' ? healthStatus(s) : null;
    const currentState = currentHealth ? currentHealth.label : service?.state;
    const item = document.createElement('div');
    item.className = 'service';
    const dot = document.createElement('span');
    dot.className = `dot${currentHealth?.delayed ? ' delayed' : ['실행 중', '검사통과'].includes(currentState) ? ' active' : currentState === '검사실패' ? ' failed' : ''}`;
    dot.setAttribute('aria-hidden', 'true');
    const name = document.createElement('b');
    name.textContent = title;
    const state = document.createElement('span');
    state.textContent = text(currentState);
    const count = document.createElement('small');
    count.textContent = key === 'health'
      ? `이전 결과 ${text(health?.state)} · 검사 시각 ${date(health?.checked_utc)} · 데이터셋 ${number(health?.checked_datasets)} · 오류 ${number(health?.errors_count)} · 대기 ${number(health?.pending_count)} · 검사기 ${text(service?.state)} · 연속 실패 ${number(service?.consecutive_failures)} · 시작 ${date(service?.started_utc)} · 실패 ${date(service?.failed_utc)}`
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
  renderOverview(s);
  window.UpbitScene?.update(s);
  services(s);
  set('capital', `${number(s.policy.initial_capital)} KRW`);
  set('start', date(s.policy.paper_start));
  set('equity', money(a.equity));
  set('cash', `현금 ${money(a.cash)}`);
  set('pnl', money(a.total_pnl));
  $('pnl').className = numeric(a.total_pnl) && a.total_pnl !== 0 ? (a.total_pnl < 0 ? 'negative' : 'positive') : '';
  set('return', numeric(a.return_pct) ? number(a.return_pct, 2) + '%' : '미기록');
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
  if ($('training-raw')?.open || $('decision-raw')?.open) renderDiagnostics(s, marketEntries);
  const trainSplits = marketEntries.map(([, report]) => report.train).filter(split => numeric(split?.rows));
  const starts = trainSplits.map(split => split.period_start).filter(value => typeof value === 'string').sort();
  const ends = trainSplits.map(split => split.period_end).filter(value => typeof value === 'string').sort();
  const gpuFits = marketEntries.flatMap(([, report]) => report.gpu || []);
  const states = {running: '진행 중', complete: '완료', failed: '오류', error: '오류', waiting_data: '50GB 자료 준비'};
  const stages = {features: '피처 구성', fit: '후보 학습', calibration: '확률 보정', locked_test: '잠금 시험', complete: '완료', download: '수집 · 검증 · 처리'};
  const external = s.external_data || {};
  const gb = value => numeric(value) ? `${number(value / 1e9, 2)} GB` : '미기록';
  const stepIndex = t.state === 'running' ? ({features: 0, fit: 1, calibration: 2, locked_test: 2, complete: 3})[t.stage] : undefined;
  const stepNames = ['데이터 준비', '모델 학습', '평가', '적용 검토'];
  [...$('workflow-steps').children].forEach((item, index) => {
    item.classList.toggle('current', index === stepIndex);
    if (index === stepIndex) item.setAttribute('aria-current', 'step');
    else item.removeAttribute('aria-current');
  });
  set('workflow-status', stepIndex === undefined ? `현재 단계: ${states[t.state] || '기록 대기'}` : `현재 단계: ${stepNames[stepIndex]} · ${stages[t.stage]}`);
  set('workflow-title', t.state === 'running' ? '가격의 흐름을 배우고 있어요' : t.state === 'complete' ? '이번 학습을 마쳤어요' : ['failed', 'error'].includes(t.state) ? '학습 상태를 확인하고 있어요' : '가격의 흐름을 배우는 과정');
  details('learning-strip', [
    ['현재 단계', stepIndex === undefined ? (states[t.state] || '기록 대기') : stepNames[stepIndex]],
    t.state === 'waiting_data' ? ['처리 완료 / 목표', `${gb(external.ingested_archive_bytes)} / ${gb(external.target_archive_bytes)}`] : ['완료 후보', `${number(t.candidate_index)} / ${number(t.candidate_total)}`],
    ['학습 회차', number(s.services?.trainer?.iteration)],
    ['GPU 모델 · 장치 전체', gpu.gpu_name || '미기록'],
    t.state === 'waiting_data' ? ['다운로드 검증', gb(external.verified_archive_bytes)] : ['학습 자료 · 최근 집계', numeric(t.rows) ? `${number(t.rows)}개` : '미집계'],
    t.state === 'waiting_data' ? ['외부 고유 분봉', numeric(external.unique_minute_rows) ? (external.unique_minute_rows >= 10000 ? `${number(external.unique_minute_rows / 10000, 1)}만 개` : `${number(external.unique_minute_rows)}개`) : '미집계'] : ['학습 항목 · 최근 집계', numeric(t.features) ? `${number(t.features)}개` : '미집계']
  ]);
  const stripIcons = ['timer', 'check', 'refresh-cw', 'cpu', 'receipt-text', 'database'];
  [...$('learning-strip').children].forEach((item, index) => {
    const icon = document.createElement('i');
    icon.dataset.lucide = stripIcons[index];
    item.prepend(icon);
  });
  const currentEntries = [
    ['현재 코인', coinNames[t.market?.replace('KRW-', '')] || text(t.market)],
    ['현재 상태 · 단계', `${states[t.state] || text(t.state)} · ${stages[t.stage] || text(t.stage)}`],
    ['완료 후보 / 전체 후보', `${number(t.candidate_index)} / ${number(t.candidate_total)}`],
    ['현재 후보', text(t.candidate)], ['실시간 진행 시작', date(t.progress_started_utc)],
    ['현재 진행 경과', numeric(t.elapsed_seconds) ? `${number(t.elapsed_seconds, 1)}초` : '대기'],
    ['모델 반복 진행', numeric(t.current_round) || numeric(t.target_rounds) ? `${numeric(t.current_round) ? number(t.current_round) : '미기록'} / ${numeric(t.target_rounds) ? number(t.target_rounds) : '미기록'}회` : '미기록'],
    ['데이터 · 최근 집계', numeric(t.rows) ? `${number(t.rows)}개` : '대기'],
    ['특징 · 최근 집계', numeric(t.features) ? `${number(t.features)}개` : '대기'],
    ['기본 시간봉', numeric(t.timeframe) ? `${number(t.timeframe)}분` : text(t.timeframe)],
    ['GPU 사용률 · 장치 전체', numeric(gpu.gpu_utilization_pct) ? `${number(gpu.gpu_utilization_pct)}%` : '대기']
  ];
  if (numeric(t.learning_rate)) currentEntries.push(['학습률 · eta', `${number(t.learning_rate, 5)} · 한 번에 반영하는 정도`]);
  details('current-learning', currentEntries);
  const research = s.research || {};
  details('recent-experiment-summary', [
    ['예측 대상', numeric(research.horizon_minutes) ? `다음 ${number(research.horizon_minutes)}분의 가격 움직임` : '예측 기간 미기록'],
    ['실험 방식', '다양한 조건으로 순환 실험'],
    ['관측 항목', '정확도 · 정밀도 · 재현율 · F1 점수 · 혼동 행렬'],
    ['평가 결과', numeric(s.validation?.accuracy) ? `후속 평가 정확도 ${percent(s.validation.accuracy)} · ${research.promoted === true ? '운용 적용' : research.promoted === false ? '운용 미적용' : '적용 여부 미기록'}` : '평가 결과 미기록']
  ]);
  details('recent-experiment', [
    ['실험 내용', numeric(research.horizon_minutes) ? `${number(research.horizon_minutes)}분 후 가격 방향 분류` : '예측 기간 미기록'],
    ['표본 가중 방식', research.class_weight_mode === 'bounded_inverse_sqrt' ? '분류별 표본 불균형을 고려한 가중치' : research.class_weight_mode === 'recency' ? '최근 표본에 비중을 둔 가중치' : '미기록'],
    ['예측 기간', numeric(research.horizon_minutes) ? `${number(research.horizon_minutes)}분` : '미기록', 'recent-experiment-horizon'],
    ['라벨 기준', numeric(research.label_threshold) ? `변화율 ${percent(research.label_threshold)} 기준` : '미기록', 'recent-experiment-threshold'],
    ['분류', '하락 · 관망 · 상승'],
    ['평가 정확도', numeric(s.validation?.accuracy) ? percent(s.validation.accuracy) : '미기록'],
    ['적용 여부', research.promoted === true ? '승격됨' : research.promoted === false ? '운용에 적용하지 않음' : '미기록', 'recent-experiment-promotion']
  ]);
  details('training-details', [
    ['현재 진행 상태', states[t.state] || text(t.state)], ['현재 단계', stages[t.stage] || text(t.stage)],
    ['현재 처리 시장', text(t.market)], ['시장 내 후보 번호', text(t.candidate)],
    ['전체 완료 후보 / 전체 후보', `${number(t.candidate_index)} / ${number(t.candidate_total)}`], ['기본 시간봉', numeric(t.timeframe) ? `${number(t.timeframe)}분` : text(t.timeframe)],
    ['GPU · 장치 전체', text(gpu.gpu_name)], ['GPU 메모리 · 장치 전체', `${number(gpu.gpu_memory_used_mb)} / ${number(gpu.gpu_memory_total_mb)} MiB`],
    ['GPU 사용률 · 장치 전체', numeric(gpu.gpu_utilization_pct) ? `${number(gpu.gpu_utilization_pct)}%` : '대기'], ['GPU 관측', `${text(gpu.state)} · ${date(gpu.checked_utc)}`],
    ['진행 상태 갱신', date(t.updated_utc)], ['직전 완료 모델 실제 시작', date(t.started_utc)],
    ['직전 완료 모델 실제 완료', t.finished_utc ? date(t.finished_utc) : '완료 시각 미기록'],
    ['직전 완료 모델 총 소요', numeric(t.duration_seconds) ? `${number(t.duration_seconds, 2)}초` : '대기'],
    ['보고서 기록 시각', date(t.recorded_utc)], ['실시간 진행 시작', date(t.progress_started_utc), 'current-learning-started'],
    ['현재 진행 경과', numeric(t.elapsed_seconds) ? `${number(t.elapsed_seconds, 2)}초` : '대기'],
    ['현재 단계 표본 / 튜닝 표본', `${number(t.rows)} / ${number(t.tune_rows)}`],
    ['특성 수', number(t.features)], ['현재 부스팅 반복 / 최대 반복', `${number(t.current_round)} / ${number(t.target_rounds)}`],
    ['현재 튜닝 로그 손실', number(t.validation_logloss, 5)],
    ['최적화 학습률 · eta', numeric(t.learning_rate) ? number(t.learning_rate, 5) : '미기록'],
    ['실제 학습 데이터 시작', date(starts[0])], ['실제 학습 데이터 종료', date(ends.at(-1))],
    ['실제 학습 표본 수', trainSplits.length ? number(trainSplits.reduce((sum, split) => sum + split.rows, 0)) : '대기'],
    ['최근 완료 실험 후보', text(research.candidate_id), 'recent-experiment-candidate'],
    ['직전 완료 모델 ID', text(t.model_id)], ['현재 운용 모델 · 포인터', text(s.current_model?.run)],
    ['최근 모의계좌 적용 모델', text(a.model_run)], ['진행 실행 ID', text(t.run), 'current-learning-run'],
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
  icons();
}
function canvas(id) {
  const element = $(id);
  const rect = element.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  const scale = Math.min(window.devicePixelRatio || 1, 1.25);
  element.width = Math.round(rect.width * scale);
  element.height = Math.round(rect.height * scale);
  const ctx = element.getContext('2d');
  ctx.scale(scale, scale);
  ctx.font = '700 11px Nanum, sans-serif';
  return {ctx, width: rect.width, height: rect.height};
}
function drawCharts() {
  if (!snapshot) return;
  const palette = getComputedStyle(document.documentElement);
  const color = name => palette.getPropertyValue(name).trim();
  const hours = $('range').value;
  const cutoff = hours === 'all' ? -Infinity : Date.now() - Number(hours) * 3600000;
  const points = (snapshot.equity || []).filter(row => numeric(row.equity) && Number.isFinite(Date.parse(row.timestamp)) && Date.parse(row.timestamp) >= cutoff).sort((a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp));
  $('chart-empty').hidden = points.length > 0;
  set('chart-caption', points.length ? `${points.length.toLocaleString('ko-KR')}개 평가 기록 · ${date(points[0].timestamp)} ~ ${date(points.at(-1).timestamp)} · KRW` : '선택 구간의 실제 모의계좌 평가 기록 없음');
  const chart = canvas('equity-chart');
  if (chart) {
    const {ctx, width: w, height: h} = chart;
    ctx.strokeStyle = color('--line');
    for (let i = 0; i < 4; i++) { const y = 20 + i * (h - 50) / 3; ctx.beginPath(); ctx.moveTo(75, y); ctx.lineTo(w - 12, y); ctx.stroke(); }
    if (points.length) {
      const values = points.map(row => row.equity);
      let min = Math.min(...values), max = Math.max(...values);
      const padding = (max - min) * .12 || Math.max(1, Math.abs(max) * .001);
      min -= padding; max += padding;
      const start = Date.parse(points[0].timestamp), end = Date.parse(points.at(-1).timestamp);
      const x = row => 75 + (end === start ? .5 : (Date.parse(row.timestamp) - start) / (end - start)) * Math.max(1, w - 90);
      const y = value => 20 + (max - value) / (max - min) * (h - 50);
      ctx.fillStyle = color('--muted');
      for (let i = 0; i < 4; i++) ctx.fillText(number(max - i * (max - min) / 3), 0, 24 + i * (h - 50) / 3);
      ctx.strokeStyle = color('--teal'); ctx.lineWidth = 2;
      ctx.beginPath(); points.forEach((row, i) => i ? ctx.lineTo(x(row), y(row.equity)) : ctx.moveTo(x(row), y(row.equity))); ctx.stroke();
      if (points.length === 1) { ctx.fillStyle = color('--teal'); ctx.beginPath(); ctx.arc(x(points[0]), y(points[0].equity), 4, 0, Math.PI * 2); ctx.fill(); }
    }
  }
  const datasets = (snapshot.datasets || []).filter(row => numeric(row.rows) && ($('data-filter').value === 'all' || row.symbol?.replace('KRW-', '') === $('data-filter').value));
  $('data-empty').hidden = datasets.length > 0;
  const bars = canvas('data-chart');
  if (bars && datasets.length) {
    const totals = coins.map(symbol => ({symbol, rows: datasets.filter(row => row.symbol?.replace('KRW-', '') === symbol).reduce((sum, row) => sum + row.rows, 0)}));
    const max = Math.max(1, ...totals.map(row => row.rows));
    const {ctx, width: w, height: h} = bars;
    totals.forEach((row, i) => { const y = 12 + i * (h - 20) / 4; ctx.fillStyle = color('--muted'); ctx.fillText(row.symbol, 0, y + 14); ctx.fillStyle = color(['--teal', '--blue', '--violet', '--gold', '--coral'][i]); ctx.fillRect(48, y, Math.max(0, w - 145) * row.rows / max, 20); ctx.fillStyle = color('--ink'); ctx.fillText(number(row.rows), Math.max(50, 56 + Math.max(0, w - 145) * row.rows / max), y + 14); });
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
    set('connection', '로컬 상태 서버 연결됨');
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
  set('page-title', {account: '자금 현황', training: 'AI 학습', data: '시장', trades: '거래'}[button.dataset.tab]);
  drawCharts();
  window.dispatchEvent(new Event('resize'));
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

function icons() {
  const vendor = window.UpbitVendors;
  if (vendor?.createIcons && vendor.icons) vendor.createIcons({icons: vendor.icons});
}

function quoteTime(value) {
  return Number.isFinite(Date.parse(value)) ? new Date(value).toLocaleTimeString('ko-KR', {timeZone: 'Asia/Seoul', hour12: false}) : '시각 미기록';
}
function quoteCurrent(value) {
  const age = Date.now() - Date.parse(value);
  // 화면 참고가격의 신선도만 60초로 구분하며 실행 호가의 안전 기준과 분리합니다.
  return Number.isFinite(age) && age >= -5000 && age <= 60000;
}
function quoteAge(value) {
  const age = (Date.now() - Date.parse(value)) / 1000;
  return !Number.isFinite(age) ? '경과 미기록' : age < -5 ? '관측 시각 확인 필요' : `${number(Math.max(0, Math.floor(age)))}초 전`;
}
function renderOverview(s) {
  const a = s.account || {};
  const health = healthStatus(s);
  set('health-summary', health.label);
  $('health-summary').classList.toggle('health-delayed', health.delayed);
  $('health-summary').title = health.delayed ? '점검 작업자 오류 또는 검사 기록 10분 초과입니다. 학습·수집 상태와 별개이며 이전 검사 결과는 서비스 상세에서 확인할 수 있습니다.' : '운용 점검 결과이며 학습·시세 연결 상태와 별개입니다.';
  $('health-warning').hidden = !health.delayed && health.label === '검사통과';
  set('collector-summary', text(s.services?.collector?.state));
  set('trainer-summary', ({running: '진행 중', complete: '완료', failed: '오류', error: '오류'})[s.training?.state] || text(s.services?.trainer?.state));
  const quoteTimes = [...(s.assets || []), ...(s.quotes || [])].map(asset => asset.quote_utc).filter(value => Number.isFinite(Date.parse(value)));
  if (a.quote_utc && Number.isFinite(Date.parse(a.quote_utc))) quoteTimes.push(a.quote_utc);
  const latestQuote = quoteTimes.sort((x, y) => Date.parse(y) - Date.parse(x))[0];
  set('quote-summary', latestQuote ? `${quoteCurrent(latestQuote) ? '호가 관측' : '갱신 지연'} · ${quoteAge(latestQuote)}` : '관측 미기록');
  $('quote-summary').classList.toggle('quote-stale', Boolean(latestQuote) && !quoteCurrent(latestQuote));
  $('quote-summary').title = latestQuote ? `${date(latestQuote)} · 거래소 최신성은 별도 확인이 필요합니다.` : '로컬 서버 연결은 거래소 최신 호가 수신의 증거가 아닙니다.';
  $('connection').title = latestQuote ? `공개 호가 관측 ${date(latestQuote)} · 로컬 서버 응답과 시세 최신성은 별개입니다.` : '로컬 상태 서버의 응답입니다. 공개 호가 시각은 미기록입니다.';
  set('trade-count', Array.isArray(s.trades) ? `${number(s.trades.length)}건` : '미기록');
  const paperState = s.services?.paper?.state || '';
  const paperError = /오류|실패/.test(paperState);
  const positions = (s.assets || []).some(asset => numeric(asset.quantity) && asset.quantity > 0);
  set('operation-state', paperError ? '확인이 필요해요' : a.entry_guard ? '기다리는 중' : positions ? '자산 보유 중' : '기회를 살피는 중');
  set('state-summary', paperError ? '거래 판단 상태를 확인해 주세요.' : a.entry_guard ? '투자 기준을 확인하고 있어요.' : positions ? '보유한 자산을 지켜보고 있어요.' : '조건이 맞을 때까지 기다려요.');
  $('operation-state').title = a.entry_guard || text(s.services?.paper?.state);
  const container = $('market-summary');
  container.replaceChildren();
  for (const coin of coins) {
    const market = `KRW-${coin}`;
    const decision = s.decision_diagnostics?.[market] || {};
    const inference = s.inference_diagnostics?.[market] || {};
    const report = s.markets?.[market] || {};
    const asset = (s.assets || []).find(row => row.symbol?.replace('KRW-', '') === coin);
    const quote = (s.quotes || []).find(row => row.symbol?.replace('KRW-', '') === coin);
    const priceSource = numeric(quote?.price) ? quote : asset;
    const row = document.createElement('article');
    row.className = 'market-row';
    const name = document.createElement('div');
    name.className = 'market-name';
    const image = document.createElement('img');
    image.src = `/assets/${coin.toLowerCase()}.png`;
    image.alt = '';
    image.width = 28;
    image.height = 28;
    const label = document.createElement('b');
    label.textContent = `${coinNames[coin]} · ${coin}`;
    name.append(image, label);
    const price = document.createElement('div');
    price.className = 'market-price';
    const observedPrice = numeric(priceSource?.price) ? `${number(priceSource.price, 8)} 원` : '시세 미기록';
    price.textContent = observedPrice;
    price.title = numeric(priceSource?.price) ? `마지막 호가 참고값 ${observedPrice} · ${date(priceSource.quote_utc)}` : '실제 시세 미기록';
    const priceCaption = document.createElement('small');
    priceCaption.className = 'quote-caption';
    const stale = numeric(priceSource?.price) && !quoteCurrent(priceSource.quote_utc);
    priceCaption.classList.toggle('quote-stale', stale);
    priceCaption.textContent = `${quote ? '공개 호가 중간값' : '호가 참고값'} · ${quoteTime(priceSource?.quote_utc)} · ${quoteAge(priceSource?.quote_utc)}${stale ? ' · 갱신 지연' : ''}`;
    price.append(priceCaption);
    const signal = document.createElement('span');
    signal.className = 'signal';
    const actualSignal = inference.signal ?? decision.signal;
    signal.textContent = ({0: '하락', 1: '중립', 2: '상승'})[actualSignal] || '판단 대기';
    const reason = document.createElement('p');
    const reasonValue = inference.reason || decision.guard || decision.reason;
    reason.textContent = diagnosticValues[reasonValue] || reasonValue || (report.diagnostics?.policy_eligible === false ? '운용 기준 미달' : '판단 근거 대기');
    row.append(name, price, signal, reason);
    container.append(row);
  }
  const t = s.training || {};
  const stageLabels = {features: '데이터·피처 구성', fit: 'GPU 후보 학습', calibration: '확률 보정', locked_test: '잠금 시험', complete: '승격 판단'};
  const current = t.stage === 'locked_test' ? 'calibration' : t.stage;
  for (const item of $('stage-rail').children) {
    const active = item.dataset.stage === current;
    item.classList.toggle('current', active);
    if (active) item.setAttribute('aria-current', 'step');
    else item.removeAttribute('aria-current');
  }
  const state = ({running: '학습 중', complete: '학습 완료', failed: '학습 오류', error: '학습 오류', waiting_data: '50GB 자료 준비'})[t.state] || '학습 기록 대기';
  const external = s.external_data || {};
  const waiting = t.state === 'waiting_data';
  set('learning-summary', waiting ? `${state} · 처리 ${number((external.ingested_archive_bytes || 0) / 1e9, 2)} / ${number((external.target_archive_bytes || 50000000000) / 1e9)} GB` : `${state} · ${coinNames[t.market?.replace('KRW-', '')] || '시장 확인 중'}${numeric(t.candidate_index) && numeric(t.candidate_total) ? ` · 후보 ${number(t.candidate_index)} / ${number(t.candidate_total)}` : ''}`);
  $('home-progress').setAttribute('aria-label', waiting ? '초기 외부 자료 처리 진행률' : '실제 전체 후보 학습 진행률');
  if (waiting && numeric(external.ingested_archive_bytes) && numeric(external.target_archive_bytes) && external.target_archive_bytes > 0) $('home-progress').value = Math.max(0, Math.min(100, 100 * external.ingested_archive_bytes / external.target_archive_bytes));
  else if (numeric(t.progress)) $('home-progress').value = Math.max(0, Math.min(100, t.progress * 100));
  else $('home-progress').removeAttribute('value');
}

// 기존 차트·보유자산·진단 노드를 이동해 ID와 렌더링 계약을 보존합니다.
function arrangeWorkspace() {
  const account = $('account');
  const assets = document.createElement('details');
  assets.id = 'account-detail';
  assets.className = 'raw-details';
  const assetsSummary = document.createElement('summary');
  assetsSummary.textContent = '자금·보유자산 상세';
  assets.append(assetsSummary);
  while (account.firstChild) assets.append(account.firstChild);
  const accountFacts = $('account-facts');
  assets.insertBefore(accountFacts, assetsSummary.nextSibling);
  assetsSummary.textContent = '자금·보유자산 상세';
  assets.append($('status-facts'), document.querySelector('.policy'), $('connection-details'));
  account.append($('home-overview'), $('prediction-workspace'), document.querySelector('.training-overview'), $('health-warning'), assets);
  const marketHeading = document.createElement('h2');
  marketHeading.textContent = '공개 호가 참고값';
  const marketSummary = document.createElement('div');
  marketSummary.id = 'market-summary';
  marketSummary.className = 'market-summary';
  $('data').prepend(marketHeading, marketSummary);
  $('data').append($('decision-raw'));
  const serviceDetails = $('services').closest('details');
  $('data').append(serviceDetails);
  for (const coin of coins) {
    const heading = document.createElement('div');
    heading.className = 'coin-heading';
    const logo = document.createElement('img');
    logo.src = `/assets/${coin.toLowerCase()}.png`;
    logo.alt = '';
    logo.width = 36;
    logo.height = 36;
    const name = document.createElement('b');
    name.textContent = coinNames[coin];
    const symbol = document.createElement('small');
    symbol.textContent = coin;
    const caption = document.createElement('span');
    caption.append(name, symbol);
    heading.append(logo, caption);
    $('coin-headings').append(heading);
  }
  for (const id of ['market-filter', 'data-filter', 'trade-filter']) {
    const select = $(id);
    for (const coin of coins) {
      const value = id === 'market-filter' ? `KRW-${coin}` : coin;
      if (![...select.options].some(option => option.value === value)) select.add(new Option(coinNames[coin], value));
      const option = [...select.options].find(option => option.value === value);
      option.textContent = coinNames[coin];
    }
  }
  const training = $('training');
  const raw = document.createElement('details');
  raw.id = 'training-raw';
  raw.className = 'raw-details';
  const title = document.createElement('summary');
  title.textContent = '학습 세부 관측·평가·진단 원문';
  raw.append(title);
  raw.append($('accuracy-target'));
  // 기술 지표·필터·원문 표는 상세에 보존하고 개념도와 요약만 먼저 표시합니다.
  const head = training.firstElementChild;
  const progressLine = $('training-progress').parentElement;
  for (const node of [...training.children]) if (node !== head && node !== progressLine && !node.classList.contains('learning-workspace') && !node.classList.contains('learning-information')) raw.append(node);
  training.append(raw);
  raw.prepend(head, progressLine);
  raw.prepend($('stage-rail'));
}
arrangeWorkspace();
openTab($('tab-training'));
window.dispatchEvent(new Event('resize'));
for (const detail of document.querySelectorAll('details')) detail.addEventListener('toggle', () => { if (detail.open) render(); });
icons();
$('account-link').addEventListener('click', () => { $('account-detail').open = true; $('account-detail').scrollIntoView({block: 'start'}); });
document.querySelector('.brand').addEventListener('click', event => { event.preventDefault(); openTab($('tab-account')); window.scrollTo({top: 0}); });
$('training-link').addEventListener('click', () => {
  openTab($('tab-training'));
  $('tab-training').focus();
  $('training').scrollIntoView({block: 'start'});
});
let motionEnabled = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
function updateMotionButton() {
  const button = $('motion-toggle');
  button.setAttribute('aria-pressed', String(!motionEnabled));
  button.setAttribute('aria-label', motionEnabled ? '화면 모션 정지' : '화면 모션 재생');
  button.replaceChildren();
  const icon = document.createElement('i');
  icon.dataset.lucide = motionEnabled ? 'pause' : 'play';
  button.append(icon);
  document.documentElement.classList.toggle('motion-paused', !motionEnabled);
  window.UpbitScene?.setMotion(motionEnabled);
  icons();
}
$('motion-toggle').addEventListener('click', () => { motionEnabled = !motionEnabled; updateMotionButton(); });
updateMotionButton();

function updateThemeButton() {
  const dark = document.documentElement.dataset.theme === 'dark';
  const button = $('theme-toggle');
  const label = dark ? '라이트 모드로 전환' : '다크 모드로 전환';
  button.setAttribute('aria-pressed', String(dark));
  button.setAttribute('aria-label', label);
  button.title = document.documentElement.dataset.themeStorage === 'unavailable' ? `${label} · 선택 저장 불가` : label;
  const icon = document.createElement('i');
  icon.dataset.lucide = dark ? 'sun' : 'moon';
  button.replaceChildren(icon);
  icons();
}
$('theme-toggle').addEventListener('click', () => {
  const root = document.documentElement;
  root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
  try {
    localStorage.setItem('upbit-theme', root.dataset.theme);
    delete root.dataset.themeStorage;
  } catch (error) {
    root.dataset.themeStorage = 'unavailable';
  }
  updateThemeButton();
  drawCharts();
  window.dispatchEvent(new Event('resize'));
});
updateThemeButton();
