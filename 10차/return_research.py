"""비용 차감 순수익을 직접 예측하는 유한 GPU 연구이며 운용 승격은 하지 않습니다."""
import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import signal
import sys
import threading
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('return_foundation_helpers', ROOT/'7차/shadow_research.py')
foundation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(foundation)
DEFAULTS = {'anchor': '2026-10-31T08:00:00Z', 'folds': 2, 'train_days': 90,
            'tune_days': 2, 'policy_days': 2, 'validation_days': 2, 'horizon': 12,
            'rounds': 150, 'early_stopping_rounds': 20, 'depth': 3,
            'learning_rate': 0.05, 'max_train_rows': 100000, 'max_raw_rows': 300000,
            'min_split_rows': 100, 'min_fold_trades': 3, 'min_total_trades': 20,
            'max_drawdown': 0.1, 'max_seconds': 7200, 'min_free_bytes': 20000000000}
OBJECTIVES = ('mean', 'q10', 'q90')


def get_engine():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import engine
    return engine


def get_xgb():
    import xgboost
    return xgboost


def validate_settings(value):
    if not isinstance(value, dict) or set(value)-set(DEFAULTS):
        raise ValueError('지원하지 않는 순수익 연구 설정입니다')
    config = {**DEFAULTS, **value}
    foundation.stamp(config['anchor'])
    ranges = {'folds': (1, 2), 'rounds': (2, 150), 'early_stopping_rounds': (2, 20),
              'depth': (1, 6), 'max_train_rows': (100, 100000), 'max_raw_rows': (1000, 300000),
              'min_split_rows': (100, 100000), 'min_fold_trades': (3, 1000),
              'min_total_trades': (20, 10000), 'max_seconds': (1, 7200),
              'min_free_bytes': (20000000000, 1000000000000)}
    for key, (low, high) in ranges.items():
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError('연구 상한 또는 표본 하한 위반: '+key)
    for key in ('train_days', 'tune_days', 'policy_days', 'validation_days', 'horizon'):
        if type(config[key]) is not int or config[key] != DEFAULTS[key]:
            raise ValueError('동결한 연구 기간·지평 위반: '+key)
    for key, low, high in (('learning_rate', .001, .05), ('max_drawdown', .001, .1)):
        if isinstance(config[key], bool) or not math.isfinite(config[key]) or not low <= config[key] <= high:
            raise ValueError('연구 수치 설정 위반: '+key)
    return config


class Guard(foundation.Guard):
    def __init__(self, root, seconds):
        self.root = Path(root)
        super().__init__(foundation.safe_path(root, '10차/work/STOP'), seconds)


class Status:
    def __init__(self, path, run, config):
        self.path = Path(path)
        self.value = {'schema_version': 1, 'run': run, 'state': 'running', 'stage': 'prepare',
                      'fit_calls': 0, 'completed_fits': 0, 'max_fits': config['folds']*15,
                      'fold_index': 0, 'fold_total': config['folds'], 'market': None,
                      'objective': 'net_forward_return', 'objective_name': None,
                      'evaluation_performed': False, 'gpu_verified': False, 'device': None, 'rows': None,
                      'promotion': False, 'candidate_ready': False,
                      'reason': None, 'comparisons': [], 'started_utc': foundation.now_iso()}

    def update(self, **fields):
        value = {**self.value, **fields, 'updated_utc': foundation.now_iso()}
        if (value['promotion'] is not False or value['candidate_ready'] is not False
                or value['objective'] != 'net_forward_return'
                or not 0 <= value['completed_fits'] <= value['fit_calls'] <= value['max_fits'] <= 30):
            raise ValueError('유한 fit·승격 금지 상태 계약 위반')
        foundation.atomic_json(self.path, value)
        self.value = value


def fold_plans(engine, config, now):
    anchor = engine.utc(config['anchor'])
    if anchor > engine.utc(now) or anchor.timestamp() % 300:
        raise ValueError('anchor는 이미 완료된 5분 경계여야 합니다')
    plans = []
    for index in range(config['folds']):
        end = anchor-engine.pd.Timedelta(days=2*(config['folds']-index-1))
        plan = {}
        for name, days in (('validation', 2), ('rule', 2), ('tune', 2), ('train', 90)):
            start = end-engine.pd.Timedelta(days=days)
            plan[name] = (start, end)
            end = start
        plans.append({name: plan[name] for name in ('train', 'tune', 'rule', 'validation')})
    return plans


def fit_return_model(engine, lib, train, tune, columns, objective, config, status, guard):
    guard.check()
    if objective not in OBJECTIVES or status.value['fit_calls'] >= status.value['max_fits']:
        raise ValueError('지원 objective 또는 최대 fit 수 위반')
    if not lib.build_info().get('USE_CUDA', False):
        raise RuntimeError('CUDA 빌드가 아닙니다. CPU fit 금지')
    np = engine.np
    y = train.net_forward_return.to_numpy(dtype=np.float32)
    yt = tune.net_forward_return.to_numpy(dtype=np.float32)
    if not np.isfinite(y).all() or not np.isfinite(yt).all():
        raise ValueError('순수익 목표가 유한하지 않습니다')
    params = {'objective': 'reg:squarederror' if objective == 'mean' else 'reg:quantileerror',
              'tree_method': 'hist', 'device': 'cuda:0', 'max_depth': config['depth'],
              'eta': config['learning_rate'], 'max_bin': 128, 'seed': 42, 'nthread': 4,
              'subsample': .9, 'colsample_bytree': .9}
    if objective == 'mean':
        params['eval_metric'] = 'rmse'
    else:
        params['quantile_alpha'] = .1 if objective == 'q10' else .9
    dtrain = lib.DMatrix(train[columns].to_numpy(dtype=np.float32), label=y,
                         weight=engine.recency_weights(train.index, 30), feature_names=columns)
    dtune = lib.DMatrix(tune[columns].to_numpy(dtype=np.float32), label=yt, feature_names=columns)
    preflight = lib.Booster(params, cache=[dtrain, dtune])
    engine.assert_cuda_config(preflight.save_config())
    del preflight
    updates = []

    class CudaOnly(lib.callback.TrainingCallback):
        def before_training(self, model):
            guard.check()
            engine.assert_cuda_config(model.save_config())
            return model

        def before_iteration(self, model, epoch, evals_log):
            guard.check()
            engine.assert_cuda_config(model.save_config())
            return False

        def after_iteration(self, model, epoch, evals_log):
            guard.check()
            device = engine.assert_cuda_config(model.save_config())
            updates.append(epoch+1)
            losses = next(iter(evals_log['tune'].values()))
            loss = float(losses[-1])
            if not math.isfinite(loss):
                raise ValueError('튜닝 손실이 유한하지 않습니다')
            status.update(round=epoch+1, loss=loss, gpu_verified=True, device=device, rows=len(train))
            return False

    history = {}
    try:
        model = lib.train(params, dtrain, num_boost_round=config['rounds'], evals=[(dtune, 'tune')],
            early_stopping_rounds=min(config['early_stopping_rounds'], config['rounds']),
            callbacks=[CudaOnly()], evals_result=history, verbose_eval=False)
    finally:
        # 실제 호출 뒤의 관측값은 표시 저장 실패에도 실패 증거에서 사라지지 않습니다.
        status.value.update(fit_calls=status.value['fit_calls']+1, observed_updates=len(updates))
        status.update()
    device = engine.assert_cuda_config(model.save_config())
    if not updates or not 1 <= model.num_boosted_rounds() <= config['rounds']:
        raise ValueError('실제 boosting 업데이트 증거가 없습니다')
    if not math.isfinite(float(model.best_score)):
        raise ValueError('튜닝 점수가 유한하지 않습니다')
    status.update(completed_fits=status.value['completed_fits']+1)
    return model, {'objective_name': objective, 'parameters': params, 'actual_device': device,
        'best_iteration': int(model.best_iteration), 'best_score': float(model.best_score),
        'boosted_rounds': int(model.num_boosted_rounds()), 'train_rows': len(train), 'tune_rows': len(tune),
        'fit_calls': 1, 'selection_scope': 'tune_only', 'xgboost_version': lib.__version__}


def prediction(engine, lib, model, frame, columns, best_iteration):
    values = model.predict(lib.DMatrix(frame[columns].to_numpy(dtype=engine.np.float32),
                                     feature_names=columns), iteration_range=(0, best_iteration+1))
    values = engine.np.asarray(values).reshape(-1)
    if len(values) != len(frame) or not engine.np.isfinite(values).all():
        raise ValueError('순수익 예측 형태 또는 유한성 위반')
    return values


def weight_hash(model):
    config = json.loads(model.save_config())
    return foundation.json_hash({'trees': model.get_dump(dump_format='json'),
        'base_score': config['learner']['learner_model_param']['base_score']})


def save_verified_model(engine, lib, model, evidence, tune, columns, path, guard):
    guard.check()
    before = weight_hash(model)
    if not model.get_dump(dump_format='json'):
        raise ValueError('완료 모델에 학습 트리가 없습니다')
    probe = tune.iloc[:32]
    expected = prediction(engine, lib, model, probe, columns, evidence['best_iteration'])
    model.save_model(path)
    digest = foundation.file_hash(path, guard)
    reloaded = lib.Booster()
    reloaded.load_model(path)
    reloaded.set_param('device', 'cuda:0')
    engine.assert_cuda_config(reloaded.save_config())
    actual = prediction(engine, lib, reloaded, probe, columns, evidence['best_iteration'])
    proof = {'model_sha256': digest, 'weight_sha256': before,
             'reload_weight_match': before == weight_hash(reloaded),
             'reload_prediction_match': bool(engine.np.allclose(expected, actual, rtol=1e-6, atol=1e-7)),
             'reload_file_match': digest == foundation.file_hash(path, guard),
             'probe_scope': 'tune_only', 'predictions_before': expected.tolist(),
             'predictions_after': actual.tolist()}
    proof['reload_verified'] = all(proof[k] for k in ('reload_weight_match', 'reload_prediction_match', 'reload_file_match'))
    foundation.atomic_json(path.with_suffix('.proof.json'), proof)
    if not proof['reload_verified']:
        raise ValueError('가중치·예측·파일의 저장 재로드 검증 실패')
    return proof


def return_signals(values, policy, factor_ratio):
    import numpy as np
    if policy['family'] == 'cash':
        return np.ones(len(values), dtype=int)
    if not math.isfinite(factor_ratio) or not 0 < factor_ratio <= 1:
        raise ValueError('비용 스트레스 비율 위반')
    mean = (1+values['mean'].to_numpy())*factor_ratio-1
    spread = (values['q90']-values['q10']).to_numpy()*factor_ratio
    finite = np.isfinite(values[['mean', 'q10', 'q90']].to_numpy()).all(axis=1)
    score = mean-policy['penalty']*spread
    return np.where(finite & (spread >= 0) & (mean > 0) & (score > policy['minimum']), 2, 1)


def select_policies(config, evaluate):
    cash = {'family': 'cash', 'minimum': 0.0, 'penalty': 0.0}
    selected, trials = {}, []
    for family in ('plain', 'uncertainty'):
        best, best_score = cash, 0.0
        for penalty in ([0.0] if family == 'plain' else [.25, .5, 1.0]):
            for minimum in (0.0, .0005, .001):
                policy = {'family': family, 'minimum': minimum, 'penalty': penalty}
                result = evaluate(policy)
                score = result['net_return']-.5*result['MDD']
                eligible = (result['trades'] >= config['min_fold_trades'] and result['net_return'] > 0
                            and 0 <= result['MDD'] <= config['max_drawdown'] and math.isfinite(score))
                trials.append({'policy': policy, 'trading': result, 'eligible': eligible})
                if eligible and score > best_score:
                    best, best_score = policy, score
        selected[family] = {'policy': best, 'policy_score': best_score}
    winner = max(selected, key=lambda key: selected[key]['policy_score'])
    return {**selected, 'selected': selected[winner]['policy'], 'selection_scope': 'rule_only'}, trials


def review_eligibility(comparisons, config):
    reasons = []
    if config['folds'] != 2:
        reasons.append('two_folds_required')
    chosen = [v for v in comparisons if v['candidate'] == 'selected']
    if len(chosen) != config['folds']*2:
        reasons.append('missing_cost_or_fold_result')
    for row in chosen:
        if not math.isfinite(row['net_return']) or row['net_return'] <= 0:
            reasons.append('non_positive_fold_return')
        if not math.isfinite(row['MDD']) or not 0 <= row['MDD'] <= config['max_drawdown']:
            reasons.append('max_drawdown_exceeded')
        if row['trades'] < config['min_fold_trades']:
            reasons.append('insufficient_fold_trades')
    if sum(v['trades'] for v in chosen if v['cost_multiplier'] == 1) < config['min_total_trades']:
        reasons.append('insufficient_total_closed_trades')
    return {'eligible': not reasons, 'rejection_reasons': sorted(set(reasons)),
            'min_total_closed_trades': config['min_total_trades'], 'max_drawdown': config['max_drawdown'],
            'scope': 'research_review_only_not_promotion_or_statistical_guarantee'}


def incumbent_scope(engine, manifest, validation_start):
    periods = manifest.get('plan', manifest.get('periods', {}))
    try:
        ends = [engine.utc(periods[name][1]) for name in ('train', 'tune', 'calibration', 'rule', 'validation') if name in periods]
        if 'train' not in periods or not ends or max(ends) >= engine.utc(validation_start):
            return 'retrospective_diagnostic'
    except (ValueError, TypeError, IndexError):
        return 'retrospective_diagnostic'
    return 'out_of_sample_comparison'


def readonly_store(engine, path, forbidden, guard):
    class ReadOnlyCandleStore(engine.CandleStore):
        def __init__(self):
            self.path = Path(path)

        def read(self, market, tf, start=None, end=None, limit=120000):
            guard.check()
            conditions, args = ['market=?', 'tf=?'], [market, str(tf)]
            if start is not None:
                conditions.append('close_time>=?')
                args.append(engine.iso(start))
            if end is not None:
                conditions.append('close_time<=?')
                args.append(engine.iso(end))
            with foundation.readonly_database(self.path, guard) as db:
                frame = engine.pd.read_sql_query('SELECT open_time,close_time,open,high,low,close,volume FROM candles WHERE '+
                    ' AND '.join(conditions)+' ORDER BY close_time DESC LIMIT ?', db, params=[*args, int(limit)+1])
            if len(frame) > limit:
                raise ValueError('원시 봉 읽기 상한 초과; 자동 축소하지 않습니다')
            if frame.empty:
                return engine.pd.DataFrame()
            frame['open_time'] = engine.pd.to_datetime(frame.open_time, utc=True)
            closing = engine.pd.to_datetime(frame.pop('close_time'), utc=True)
            if not closing.eq(frame.open_time+engine.pd.Timedelta(minutes=engine.minutes(tf))).all():
                raise ValueError('원시 봉 완료 시각 위반')
            for left, right in forbidden:
                frame = frame.loc[~((frame.open_time < right) & (closing >= left))]
                closing = closing.loc[frame.index]
            return frame.set_index('open_time').sort_index()
    return ReadOnlyCandleStore()


def freeze_incumbent(engine, base, directory, guard):
    source, manifest = engine.load_current(base)
    foundation.assert_no_links(source)
    target = directory/'incumbent'
    target.mkdir()
    paths = {'manifest.json'} | {name for package in manifest['packages'].values() for name in package['models']}
    hashes = {}
    for name in sorted(paths):
        guard.check()
        src = foundation.safe_path(source, name)
        before = foundation.file_hash(src, guard)
        shutil.copyfile(src, target/name)
        if foundation.file_hash(target/name, guard) != before or foundation.file_hash(src, guard) != before:
            raise ValueError('기존 적용 모델 동결 복사 중 변경 또는 해시 불일치')
        hashes[name] = before
    pointer = foundation.read_json(Path(base['models_dir'])/'current.json')
    if pointer.get('run') != manifest['run'] or pointer.get('manifest_hash') != engine.stable_hash(manifest):
        raise ValueError('기존 모델 동결 중 current가 바뀌었습니다')
    return {'directory': target, 'manifest': manifest, 'files': hashes, 'run': manifest['run']}


def foundation_exposures(root, guard):
    work = foundation.safe_path(root, '7차/work')
    # 성공과 실패 회차 모두 기존 엄격 결합 검사를 먼저 통과해야 합니다.
    foundation.read_exposure_end(work, guard)
    periods, proofs = [], []
    for path in sorted((work/'exposures').glob('*.json')):
        value = foundation.read_json(path)['reservation']
        periods.append((foundation.stamp(value['validation_start']), foundation.stamp(value['exposure_end'])))
        proofs.append({'run': value.get('run'), 'sha256': foundation.file_hash(path, guard)})
    return periods, proofs


def prepare_fold(engine, base, incumbent, plan, config, audit, run, guard):
    previous, exposure_proof = foundation_exposures(guard.root, guard)
    consumed = [*audit.consumed_periods(), *previous]
    candidate = copy.deepcopy(base)
    candidate['external']['enabled'] = False
    candidate['external']['wait_for_dataset'] = False
    candidate['research'].update(active=False, continuous=False)
    candidate['training'].update(train_days=90, tune_days=2, rule_days=2, validation_days=2,
        horizon=config['horizon'], gap_bars=config['horizon']+1, min_split_rows=config['min_split_rows'],
        max_rows_per_market=config['max_train_rows'], max_raw_rows=config['max_raw_rows'])
    candidate['paper']['max_drawdown'] = config['max_drawdown']
    forbidden = engine.protected_periods(candidate, consumed)
    for name in ('rule', 'validation'):
        if any(plan[name][0] < right and plan[name][1] > left for left, right in forbidden):
            raise ValueError('정책·후속 기간이 보호 또는 소비 시험과 겹칩니다')
    store = readonly_store(engine, Path(base['data_dir'])/'candles.sqlite3', forbidden, guard)
    all_parts, raw_frames, incumbent_frames, columns = {}, {}, {}, None
    incumbent_config = copy.deepcopy(incumbent['manifest']['config'])
    incumbent_config['training'].update(max_rows_per_market=config['max_train_rows'], max_raw_rows=config['max_raw_rows'])
    for market in engine.MARKETS:
        guard.check()
        frame = engine.load_market_features(store, market, candidate, plan['train'][0], plan['validation'][1])
        labelled = engine.label_frame(frame, config['horizon'], candidate['training']['label_threshold'],
                                      base['paper']['fee'], base['paper']['slippage'])
        cols = engine.feature_columns(labelled)
        if columns is not None and cols != columns:
            raise ValueError('시장별 피처 스키마 불일치')
        columns = cols
        for tf in engine.TIMEFRAMES:
            available = labelled['available_'+str(tf)]
            if (available.notna() & (available > labelled.index)).any():
                raise ValueError('완료되지 않은 미래 시간봉 피처')
        execution = labelled.dropna(subset=cols).copy()
        for left, right in forbidden:
            execution = execution[(execution.index < left) | (execution.open_time >= right)]
        labelled = execution.dropna(subset=['label', 'label_end', 'net_forward_return'])
        parts = engine.research_splits(labelled, plan, candidate, consumed)
        if len(parts['train']) > config['max_train_rows']:
            raise ValueError('시장 학습 행 상한 초과')
        for name, part in parts.items():
            for left, right in forbidden:
                part = part[(part.label_end < left) | (part.open_time >= right)]
            if len(part) < config['min_split_rows']:
                raise ValueError('전체 봉 보호 제외 후 표본 부족: '+market+'/'+name)
            parts[name] = part
        # 지평 purge는 진입 판단에만 적용하고 청산용 마지막 완료 봉은 유지합니다.
        raw_frames[market], all_parts[market] = execution, parts
        # 외부 피처를 쓰던 기존 모델은 동결된 그 스키마로 별도 재구성합니다.
        incumbent_frames[market] = engine.load_market_features(store, market, incumbent_config,
            plan['rule'][0], plan['validation'][1])
        for name, part in parts.items():
            if name in ('rule', 'validation'):
                observed = execution[(execution.index >= plan[name][0]) & (execution.index < plan[name][1])]
                if len(observed):
                    audit.expose(run, observed.open_time.min(), observed.index.max(), 'return_research_'+name+'_execution')
            edges = [(None, a) for a, _ in forbidden[:1]] if forbidden else []
            edges += [(forbidden[i][1], forbidden[i+1][0]) for i in range(len(forbidden)-1)]
            if forbidden:
                edges.append((forbidden[-1][1], None))
            for left, right in edges or [(None, None)]:
                selected = part if left is None else part[part.index >= left]
                selected = selected if right is None else selected[selected.label_end < right]
                if len(selected):
                    audit.expose(run, selected.index.min(), selected.label_end.max(), 'return_research_'+name)
    return {'parts': all_parts, 'frames': raw_frames, 'incumbent_frames': incumbent_frames,
            'columns': columns, 'paper': candidate['paper'], 'forbidden': forbidden,
            'foundation_exposures': exposure_proof, 'external_data_used_for_training': False}


def frame_hash(engine, frame):
    digest = hashlib.sha256()
    digest.update('|'.join(frame.columns).encode())
    digest.update(engine.pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def evaluation_frames(engine, prepared, plan, name):
    frames = {m: frame[(frame.index >= plan[name][0]) & (frame.index < plan[name][1])].copy()
              for m, frame in prepared['frames'].items()}
    common = None
    for frame in frames.values():
        common = frame.index if common is None else common.intersection(frame.index)
    if common is None or len(common) < 100:
        raise ValueError('동일 평가 완료 봉이 부족합니다')
    aligned = {}
    for market, frame in frames.items():
        execution = frame.loc[common].copy()
        execution['research_entry_allowed'] = common.isin(prepared['parts'][market][name].index)
        aligned[market] = execution
    return aligned


def restrict_entries(engine, frame, probabilities, rule):
    allowed = frame.get('research_entry_allowed')
    if allowed is None or allowed.dtype != bool or allowed.isna().any():
        raise ValueError('공통 진입 가능 마스크가 없거나 유효하지 않습니다')
    values = engine.np.asarray(probabilities).copy()
    if values.shape != (len(frame), 3):
        raise ValueError('공통 진입 마스크와 확률의 행 수가 다릅니다')
    # purge 꼬리에서는 매수만 막고 보유 포지션의 매도·평가는 그대로 허용합니다.
    blocked = ~allowed.to_numpy() & (engine.signals_from_proba(values, rule) == 2)
    values[blocked] = [0, 1, 0]
    return values


def evaluate(engine, frames, predictions, policy, paper, horizon, multiplier):
    stressed = copy.deepcopy(paper)
    stressed.update(fee=paper['fee']*multiplier, slippage=paper['slippage']*multiplier)
    ratio = engine.roundtrip_cost_factor(stressed)/engine.roundtrip_cost_factor(paper)
    probabilities, rules = {}, {}
    for market, frame in frames.items():
        data = predictions[market].reindex(frame.index)
        signal = return_signals(data, policy, ratio)
        probs = engine.np.zeros((len(frame), 3))
        probs[:, 1] = 1
        probs[signal == 2] = [0, 0, 1]
        rules[market] = {'confidence': .9, 'edge': .9, 'horizon_minutes': horizon*5}
        probabilities[market] = restrict_entries(engine, frame, probs, rules[market])
    result = engine.backtest_portfolio(frames, probabilities, rules, stressed)
    return {**result, 'historical_quote_extra_cost': None,
            'expected_return_target': 'next_open_to_horizon_close_after_base_fee_and_slippage'}


def evaluate_baseline(engine, name, frames, incumbent_frames, incumbent, paper, horizon, multiplier):
    stressed = copy.deepcopy(paper)
    stressed.update(fee=paper['fee']*multiplier, slippage=paper['slippage']*multiplier)
    probabilities, rules = {}, {}
    for market, frame in frames.items():
        if name == 'incumbent':
            aligned = incumbent_frames[market].reindex(frame.index)
            cols = incumbent['manifest']['feature_columns']
            if aligned[cols].isna().any().any() or not aligned[['open', 'close']].equals(frame[['open', 'close']]):
                raise ValueError('기존 모델과 후보의 동일 평가 행·가격·피처를 확보하지 못했습니다')
            probabilities[market] = engine.predict_package(incumbent['directory'], incumbent['manifest'], market, aligned)
            rules[market] = {**incumbent['manifest']['packages'][market]['rule'],
                'horizon_minutes': incumbent['manifest']['config']['training']['horizon']*5}
        else:
            probs = engine.np.zeros((len(frame), 3))
            probs[:, 1] = 1
            if name == 'trend':
                buy = (frame['5_ma_ratio'] > 0) & (frame['5_return_12'] > 0)
                probs[buy] = [0, 0, 1]
            probabilities[market] = probs
            rules[market] = {'confidence': .9, 'edge': .9, 'horizon_minutes': horizon*5}
        probabilities[market] = restrict_entries(engine, frame, probabilities[market], rules[market])
    return engine.backtest_portfolio(frames, probabilities, rules, stressed)


def run_study(root, settings, run, allow_training=False):
    if not allow_training:
        raise PermissionError('기본 학습 중지를 존중합니다. 유한 실험에 --allow-training을 명시해야 합니다')
    if sys.platform != 'linux':
        raise RuntimeError('실제 DB와 GPU 연구는 Linux 컨테이너 전용입니다')
    root, config = Path(root).resolve(), validate_settings(settings)
    if not isinstance(run, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run):
        raise ValueError('유효한 실험 run 식별자가 필요합니다')
    engine = get_engine()
    work = foundation.safe_path(root, '10차/work')
    foundation.assert_no_links(work)
    guard = Guard(root, config['max_seconds'])
    with foundation.exclusive_lock(work/'runner.lock'):
        guard.check()
        directory = foundation.safe_path(work, 'runs/'+run)
        if directory.exists():
            raise RuntimeError('실험 예약이 존재합니다. 자동 재훈련·중간 resume는 하지 않습니다')
        directory.mkdir(parents=True)
        status = Status(work/'status.json', run, config)
        old_signals = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                old_signals[sig] = signal.signal(sig, lambda *_: guard.event.set())
        try:
            status.update()
            if shutil.disk_usage(work).free < config['min_free_bytes']:
                raise OSError('디스크 여유 부족; 자동 삭제 금지')
            base = engine.load_config(root/'engine-config.json')
            if tuple(base['timeframes']) != tuple(engine.TIMEFRAMES) or tuple(base['markets']) != tuple(engine.MARKETS):
                raise ValueError('9개 시간봉·5개 KRW 시장이 필요합니다')
            plans = fold_plans(engine, config, engine.utc(None))
            request = {'run': run, 'settings': config, 'source_sha256': foundation.file_hash(Path(__file__), guard),
                'engine_sha256': foundation.file_hash(Path(engine.__file__), guard),
                'periods': [{k: [engine.iso(a), engine.iso(b)] for k, (a,b) in plan.items()} for plan in plans],
                'allow_training': True, 'training_paused_marker_present': engine._training_pause_requested(),
                'promotion': False, 'independent_final_test': False, 'external_data_used_for_training': False,
                'candidate_features': 'upbit_completed_ohlcv_9_timeframes_only',
                'incumbent_features': 'frozen_manifest_schema_without_fallback'}
            foundation.atomic_json(directory/'request.json', request)
            incumbent = freeze_incumbent(engine, base, directory, guard)
            audit = engine.ExperimentAudit(Path(base['data_dir'])/'experiment_audit.sqlite3')
            lib = get_xgb()
            folds, comparisons, model_files = [], [], {}
            for fold_index, plan in enumerate(plans, 1):
                guard.check()
                status.update(fold_index=fold_index, stage='prepare', market=None, objective_name=None)
                fold_dir = directory/f'fold-{fold_index}'
                fold_dir.mkdir()
                prepared = prepare_fold(engine, base, incumbent, plan, config, audit, run, guard)
                parts, columns = prepared['parts'], prepared['columns']
                foundation.atomic_json(fold_dir/'data.json', {'columns': columns,
                    'parts': {m: {name: {'rows': len(frame), 'sha256': frame_hash(engine, frame)}
                                   for name, frame in split.items()} for m, split in parts.items()},
                    'protected': [[engine.iso(a), engine.iso(b)] for a,b in prepared['forbidden']],
                    'audit_kind': 'return_research_exposure_not_final_test',
                    'foundation_exposures': prepared['foundation_exposures'],
                    'external_data_used_for_training': False})
                models, gpu = {}, {}
                for market in engine.MARKETS:
                    models[market], gpu[market] = {}, {}
                    for objective in OBJECTIVES:
                        guard.check()
                        if shutil.disk_usage(work).free < config['min_free_bytes']:
                            raise OSError('학습 전 디스크 여유 부족')
                        status.update(stage='fit', market=market, objective_name=objective,
                                      gpu_verified=False, device=None, rows=None, round=0, loss=None)
                        model, evidence = fit_return_model(engine, lib, parts[market]['train'], parts[market]['tune'],
                            columns, objective, config, status, guard)
                        path = fold_dir/(market+'-'+objective+'.ubj')
                        proof = save_verified_model(engine, lib, model, evidence, parts[market]['tune'], columns, path, guard)
                        model_files[path.relative_to(directory).as_posix()] = proof['model_sha256']
                        models[market][objective], gpu[market][objective] = model, {**evidence, 'save_reload': proof}
                status.update(stage='policy', market=None, objective_name=None)
                rule_frames = evaluation_frames(engine, prepared, plan, 'rule')
                val_frames = evaluation_frames(engine, prepared, plan, 'validation')
                def predict_part(name):
                    return {market: engine.pd.DataFrame({objective: prediction(engine, lib, models[market][objective],
                        parts[market][name], columns, gpu[market][objective]['best_iteration']) for objective in OBJECTIVES},
                        index=parts[market][name].index) for market in engine.MARKETS}
                rule_predictions = predict_part('rule')
                selected, trials = select_policies(config, lambda p: evaluate(engine, rule_frames, rule_predictions,
                    p, prepared['paper'], config['horizon'], 1))
                foundation.atomic_json(fold_dir/'policy.json', {'selected': selected, 'trials': trials})
                status.update(stage='validation')
                guard.check()
                val_predictions = predict_part('validation')
                foundation.atomic_json(fold_dir/'predictions.json', {m: {'index': [engine.iso(i) for i in frame.index],
                    'predictions': frame.to_dict(orient='list')} for m,frame in val_predictions.items()})
                fold_rows = []
                for multiplier in (1, 2):
                    for name in ('plain', 'uncertainty', 'selected', 'cash', 'trend', 'incumbent'):
                        guard.check()
                        if name in ('plain', 'uncertainty', 'selected'):
                            policy = selected['selected'] if name == 'selected' else selected[name]['policy']
                            result = evaluate(engine, val_frames, val_predictions, policy, prepared['paper'], config['horizon'], multiplier)
                        else:
                            result = evaluate_baseline(engine, name, val_frames, prepared['incumbent_frames'], incumbent,
                                                       prepared['paper'], config['horizon'], multiplier)
                        row = {'fold': fold_index, 'candidate': name, 'cost_multiplier': multiplier,
                               'net_return': float(result['net_return']), 'MDD': float(result['MDD']),
                               'trades': int(result['trades']), 'scope': (incumbent_scope(engine, incumbent['manifest'],
                                    plan['validation'][0]) if name == 'incumbent' else 'research_validation')}
                        fold_rows.append({'summary': row, 'details': result})
                        comparisons.append(row)
                report = {'fold': fold_index, 'gpu': gpu, 'policy': selected, 'comparisons': fold_rows,
                    'validation_used_for_policy_selection': False, 'independent_final_test': False,
                    'common_entry_mask_sha256': {m: frame_hash(engine, frame[['research_entry_allowed']])
                                                  for m, frame in val_frames.items()},
                    'common_entry_rows': {m: int(frame.research_entry_allowed.sum()) for m,frame in val_frames.items()},
                    'same_validation_rows_sha256': {m: foundation.json_hash([engine.iso(t) for t in frame.index])
                                                  for m, frame in val_frames.items()}}
                foundation.atomic_json(fold_dir/'report.json', report)
                folds.append(report)
                status.update(comparisons=comparisons)
            eligibility = review_eligibility(comparisons, config)
            if status.value['fit_calls'] != config['folds']*15 or status.value['completed_fits'] != config['folds']*15:
                raise ValueError('완료한 fit 수가 동결 계획과 다릅니다')
            manifest = {'run': run, 'complete': True, 'kind': 'return_research_not_deployed',
                'request_sha256': foundation.file_hash(directory/'request.json', guard),
                'models': model_files, 'fit_calls': status.value['fit_calls'],
                'completed_fits': status.value['completed_fits'], 'review': eligibility,
                'promotion': False, 'candidate_ready': False, 'comparisons': comparisons,
                'external_data_used_for_training': False,
                'incumbent': {'run': incumbent['run'], 'files': incumbent['files']},
                'original_account_modified': False, 'independent_final_test': False,
                'historical_quote_extra_cost': None,
                'observed_validation_best': {str(i): max((v for v in comparisons if v['fold']==i and v['cost_multiplier']==1),
                      key=lambda v:(v['net_return'], v['candidate']=='cash'))['candidate'] for i in range(1, config['folds']+1)}}
            manifest['artifacts'] = {path.relative_to(directory).as_posix(): foundation.file_hash(path, guard)
                                     for path in sorted(directory.rglob('*')) if path.is_file()}
            foundation.atomic_json(directory/'manifest.json', manifest)
            status.update(state='complete', stage='complete', reason=None if eligibility['eligible'] else 'research_review_rejected',
                manifest_sha256=foundation.file_hash(directory/'manifest.json', guard),
                research_review_eligible=eligibility['eligible'], evaluation_performed=True)
            return status.value
        except BaseException as exc:
            foundation.atomic_json(directory/'failure.json', {'utc': foundation.now_iso(), 'type': type(exc).__name__,
                'error': str(exc), 'traceback': traceback.format_exc(), 'argv': sys.argv, 'cwd': str(Path.cwd()),
                'run': run, 'stage': status.value['stage'], 'fit_calls': status.value['fit_calls'],
                'observed_updates': status.value.get('observed_updates', 0),
                'completed_fits': status.value['completed_fits'], **foundation.routing_counters(root)})
            status.update(state='stopped' if isinstance(exc, (KeyboardInterrupt, foundation.StopRequested)) else 'error',
                          stage='failed', reason='실험 중단; 비공개 실패 원문 보존')
            raise
        finally:
            for sig, previous in old_signals.items():
                signal.signal(sig, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', required=True, action='store_true', help='유한한 단일 실험만 수행')
    parser.add_argument('--run', required=True, help='중복 재실행하지 않을 실험 식별자')
    parser.add_argument('--allow-training', action='store_true', help='기본 pause를 삭제하지 않고 이 실험의 GPU 적합만 명시 허용')
    parser.add_argument('--config', default='10차/return-research-config.json')
    args = parser.parse_args(argv)
    try:
        config = validate_settings(foundation.read_json(foundation.safe_path(ROOT, args.config)))
        value = run_study(ROOT, config, args.run, args.allow_training)
        print(json.dumps({k: value[k] for k in ('run', 'state', 'fit_calls', 'completed_fits', 'promotion')}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print('순수익 연구 실행 실패: '+str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
