"""고정 train/tune으로 50-step fit을 10회 이어가는 미검증 연구 실행기입니다."""
import argparse
import datetime as dt
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
_spec = importlib.util.spec_from_file_location('foundation_seven_for_repetition', ROOT/'7차/shadow_research.py')
foundation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(foundation)
FROZEN = {'repetitions': 10, 'steps': 50, 'context_length': 192, 'batch_size': 8,
          'learning_rate': 1e-6, 'prediction_length': 12}
DEFAULTS = {**FROZEN, 'max_seconds': 18000, 'min_free_bytes': 20_000_000_000,
            'max_rows_per_market': 50000, 'seed': 42}
METRICS = ('validation_mae', 'baseline_mae', 'direction_accuracy', 'net_return', 'trades')
ORIGIN_SOURCE_SHA256 = 'ea361c6da394fc2ff128615d583e8079eca0eeb43bc23e3c37613ddde1425fc9'


def validate_settings(value):
    if not isinstance(value, dict) or set(value)-set(DEFAULTS):
        raise ValueError('8차 설정의 형식 또는 키가 유효하지 않습니다')
    config = {**DEFAULTS, **value}
    for key, expected in FROZEN.items():
        actual = config[key]
        if isinstance(actual, bool) or not isinstance(actual, type(expected)) or actual != expected:
            raise ValueError(f'승인한 반복 학습 설정은 고정입니다: {key}')
    for key in ('max_seconds', 'min_free_bytes', 'max_rows_per_market', 'seed'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'양의 정수 설정이 필요합니다: {key}')
    if config['max_seconds'] > 18000 or config['min_free_bytes'] < 20_000_000_000:
        raise ValueError('5시간 상한 또는 20GB 여유 기준을 약화할 수 없습니다')
    return config


class RepeatGuard(foundation.Guard):
    def __init__(self, root, seconds):
        super().__init__(foundation.safe_path(root, '8차/work/STOP'), seconds)
        self.stop_seven = foundation.safe_path(root, '7차/work/STOP')

    def check(self):
        if self.stop_seven.exists():
            self.event.set()
        super().check()


class RepeatStatus:
    def __init__(self, path, config, run):
        self.path = path
        self.value = {'run': run, 'model_id': foundation.MODEL_ID, 'state': 'running', 'stage': 'extract',
                      'rep_index': 1, 'total_repetitions': 10, 'completed_repetitions': 0,
                      'step': 0, 'total_steps': 500, 'loss': None, 'device': None, 'rows': None,
                      'context_length': config['context_length'], 'batch_size': config['batch_size'],
                      'learning_rate': config['learning_rate'], 'checkpoint_verified': False,
                      'evaluation_performed': False, 'candidate_ready': False, 'promote': False,
                      'metrics': None, 'started_utc': foundation.now_iso(), 'finished_utc': None,
                      'has_error': False, **dict.fromkeys(METRICS)}

    def update(self, **fields):
        value = {**self.value, **fields, 'updated_utc': foundation.now_iso()}
        if (value['evaluation_performed'] is not False or value['candidate_ready'] is not False
                or value['promote'] is not False or value['metrics'] is not None
                or any(value[key] is not None for key in METRICS)
                or type(value['completed_repetitions']) is not int
                or not 0 <= value['completed_repetitions'] <= 10
                or type(value['rep_index']) is not int or not 1 <= value['rep_index'] <= 10
                or type(value['step']) is not int or not 0 <= value['step'] <= 500
                or value['stage'] not in ('extract', 'finetune', 'verify', 'complete')):
            raise ValueError('train-only 상태 계약을 위반했습니다')
        foundation.atomic_json(self.path, value)
        self.value = value


class StepProxy:
    def __init__(self, status, index):
        self.status, self.index = status, index
        self.value = {'step': 0, 'loss': None}

    def update(self, **fields):
        value = {**self.value, **fields}
        if (type(value['step']) is not int or not self.value['step'] <= value['step'] <= 50
                or (value['loss'] is not None and not math.isfinite(float(value['loss'])))):
            raise ValueError('회차 내 step 또는 손실이 유효하지 않습니다')
        self.status.update(step=(self.index-1)*50+value['step'], loss=value['loss'],
                           actual_fits_consumed=max(self.status.value.get('actual_fits_consumed', 0),
                                                    self.index if value['step'] else self.index-1))
        self.value = value


def prepare_data(root, parent, config, guard):
    plan = parent['manifest']['plan']
    periods = {}
    for name in ('train', 'tune', 'validation'):
        a, b = map(foundation.stamp, plan[name])
        if a >= b or b > dt.datetime.now(foundation.UTC):
            raise ValueError('부모의 고정 기간이 역행하거나 미래에 있습니다')
        periods[name] = (a, b)
    if periods['train'][1] > periods['tune'][0] or periods['tune'][1] > periods['validation'][0]:
        raise ValueError('부모 train/tune/validation 기간이 겹칩니다')
    work_seven = foundation.safe_path(root, '7차/work')
    protected = list(foundation.protected_periods(root, guard))
    # 성공·실패 평가 예약 모두 검증한 뒤 원래 검증 구간까지 입력에서 제외합니다.
    foundation.read_exposure_end(work_seven, guard)
    for path in sorted((work_seven/'exposures').glob('*.json')):
        receipt = foundation.read_json(path)['reservation']
        protected.append((receipt['validation_start'], receipt['exposure_end']))
    ancestor = parent['manifest']
    while True:
        protected.append(ancestor['plan']['validation'])
        protected.extend(ancestor.get('data', {}).get('protected_periods', []))
        if ancestor.get('parent_run') is None:
            break
        ancestor = foundation.read_json(foundation.safe_path(work_seven,
            'runs/'+ancestor['parent_run']+'/manifest.json'))
    protected = sorted({(foundation.stamp(a).isoformat(), foundation.stamp(b).isoformat())
                        for a, b in protected})
    if any(foundation.stamp(a) >= foundation.stamp(b) for a, b in protected):
        raise ValueError('금지 기간의 순서가 유효하지 않습니다')
    arrays = {'train': [], 'tune': []}
    records, contexts, counts, evidence = {}, [], {}, {}
    minimum = config['context_length']+config['prediction_length']
    for name in ('train', 'tune'):
        guard.check()
        a, b = plan[name]
        rows, evidence[name] = foundation.extract_rows(root, a, b, config['max_rows_per_market'], guard)
        records[name] = {}
        for market in foundation.MARKETS:
            segments = [s for s in foundation.split_segments(rows[market], a, b, protected) if len(s) >= minimum]
            if not segments:
                raise foundation.NotReady(f'{market}: 고정 {name}의 연속 시계열이 부족합니다')
            records[name][market] = [[(t.isoformat(), v) for t, v in segment] for segment in segments]
            arrays[name].extend([[v for _, v in segment] for segment in segments])
            counts.setdefault(market, {})[name+'_rows'] = sum(map(len, segments))
            if name == 'train':
                contexts.append([v for _, v in segments[0][:config['context_length']]])
    data = {'source': 'upbit', 'timeframe_minutes': 5, 'external_data_used_for_training': False,
            'plan': {name: plan[name] for name in ('train', 'tune')}, 'protected_periods': protected,
            'used_by_market': counts, 'extraction': evidence,
            'inputs_sha256': foundation.json_hash(records), 'probe_scope': 'train_context_only',
            'probe_contexts_sha256': foundation.json_hash(contexts)}
    return {'training': arrays['train'], 'tuning': arrays['tune'], 'contexts': contexts,
            'records': records, 'data': data, 'rows': sum(c['train_rows'] for c in counts.values())}


def check_predictions(values, count):
    if len(values) != count or any(len(row) != 12 for row in values):
        raise ValueError('학습 context probe의 예측 형태가 다릅니다')
    if any(isinstance(v, bool) or not math.isfinite(float(v)) for row in values for v in row):
        raise ValueError('학습 context probe가 유한하지 않습니다')


def unwrap_model_for_inference(model):
    from accelerate.utils import extract_model_from_parallel
    return extract_model_from_parallel(model, keep_fp32_wrapper=False)


def normalize_inference(runtime, pipeline):
    before = runtime.weights_hash(pipeline)
    wrapped = hasattr(pipeline.model, '_original_forward')
    model = unwrap_model_for_inference(pipeline.model)
    if model is not pipeline.model:
        raise ValueError('승인한 단일 GPU 모델 이외의 병렬 래퍼는 지원하지 않습니다')
    model.eval()
    runtime.check_device(pipeline)
    if (str(model.dtype) != 'torch.float32' or hasattr(model, '_original_forward')
            or hasattr(model.forward, '__wrapped__')):
        raise ValueError('FP32 추론 정규화 또는 AMP 래퍼 제거를 확인하지 못했습니다')
    after = runtime.weights_hash(pipeline)
    if before != after:
        raise ValueError('추론 정규화가 파라미터 해시를 변경했습니다')
    backends = getattr(runtime.torch, 'backends', None)
    mode = {'dtype': str(model.dtype), 'training': bool(model.training),
            'attention': getattr(model.config, '_attn_implementation', None),
            'matmul_fp32_precision': (backends.cuda.matmul.fp32_precision if backends else None),
            'cudnn_fp32_precision': (backends.cudnn.fp32_precision if backends else None)}
    return {'parameter_sha256': after, 'mode': mode, 'original_had_amp_wrapper': wrapped,
            'normalization': 'extract_model_from_parallel_keep_fp32_wrapper_false'}


def capture_probe(runtime, pipeline, checkpoint, contexts, path, guard, binding=None):
    normalized = normalize_inference(runtime, pipeline)
    files = foundation.checkpoint_files(checkpoint, guard)
    predictions = runtime.predict(pipeline, contexts)
    check_predictions(predictions, len(contexts))
    value = {**normalized, 'scope': 'train_context_only', 'context_sha256': foundation.json_hash(contexts),
             'predictions': predictions, 'checkpoint_files': files, 'binding': binding, 'utc': foundation.now_iso()}
    foundation.atomic_json(path, value)
    return value


def compare_probes(runtime, first, second, path):
    result = {'parameter_hash_match': first['parameter_sha256'] == second['parameter_sha256'],
              'inference_mode_match': first['mode'] == second['mode'],
              'context_hash_match': first['context_sha256'] == second['context_sha256'],
              'checkpoint_files_match': first['checkpoint_files'] == second['checkpoint_files'],
              'predictions_match': bool(runtime.np.allclose(first['predictions'], second['predictions'],
                                        rtol=1e-5, atol=1e-5, equal_nan=False)), 'rtol': 1e-5, 'atol': 1e-5}
    foundation.atomic_json(path, result)
    for field in ('parameter_hash_match', 'inference_mode_match', 'context_hash_match',
                  'checkpoint_files_match', 'predictions_match'):
        if not result[field]:
            raise ValueError('저장 검증 불일치: '+field)
    return result


def original_diagnostic_paths(root, batch):
    work, rep = Path(root)/'8차/work', batch/'rep-01'
    return [work/'status.json', work/'request.json', batch/'inputs.json', rep/'attempt.json',
            Path(root)/'7차/work/knowledge.json', rep/'checkpoint/config.json',
            rep/'checkpoint/model.safetensors', rep/'checkpoint-50/config.json',
            rep/'checkpoint-50/model.safetensors']


def validate_recovery_origin(root, run, config, guard):
    work = foundation.safe_path(root, '8차/work')
    batch = foundation.safe_path(work, 'runs/'+run)
    if (batch/'recovery-first').exists():
        raise RuntimeError('첫 회차 복구 예약이 존재하여 재실행을 거부합니다')
    foundation.assert_no_links(batch)
    request = foundation.read_json(work/'request.json')
    freeze = foundation.read_json(work/'evidence/source-freeze-repair70.json')
    if (request.get('run') != run or request.get('settings') != config
            or request.get('source_sha256') != ORIGIN_SOURCE_SHA256
            or freeze.get('sha256', {}).get('8차/train_only_repeats.py') != ORIGIN_SOURCE_SHA256
            or request.get('helper_sha256') != foundation.file_hash(Path(foundation.__file__), guard)
            or request.get('model_id') != foundation.MODEL_ID or request.get('revision') != foundation.REVISION
            or foundation.file_hash(work/'request.json', guard) != foundation.file_hash(batch/'request.json', guard)
            or (work/'knowledge.json').exists() or (batch/'rep-01/manifest.json').exists()
            or any(batch.glob('rep-0[2-9]')) or (batch/'rep-10').exists()):
        raise ValueError('수리70 실패 batch 기원 또는 복구 범위가 다릅니다')
    diagnostic = foundation.read_json(work/'evidence/repair71-readonly-reload-diagnostic.log')
    expected = diagnostic.get('before_files')
    paths = original_diagnostic_paths(root, batch)
    observed = {p.relative_to(root).as_posix(): foundation.file_hash(p, guard) for p in paths}
    if (observed != expected or observed != diagnostic.get('after_files')
            or diagnostic.get('fit_calls') != 0 or diagnostic.get('evaluation_performed') is not False
            or diagnostic.get('all_original_files_unchanged') is not True
            or diagnostic.get('probe_scope') != 'saved_train_context_only'):
        raise ValueError('무학습 진단의 9개 원본 해시 결합이 다릅니다')
    status = foundation.read_json(work/'status.json')
    completion = foundation.read_json(batch/'completion.json')
    trainer = foundation.read_json(batch/'rep-01/checkpoint-50/trainer_state.json')
    attempt = foundation.read_json(batch/'rep-01/attempt.json')
    if (status.get('run') != run or status.get('state') != 'error' or status.get('stage') != 'verify'
            or status.get('step') != 50 or status.get('completed_repetitions') != 0
            or status.get('device') != 'cuda:0' or not math.isfinite(float(status['loss']))
            or type(trainer.get('global_step')) is not int or trainer['global_step'] != 50
            or completion.get('complete') is not False or completion.get('completed_repetitions') != 0
            or completion.get('round_manifests') != [] or attempt.get('rep_index') != 1
            or attempt.get('run') != run or attempt.get('request_sha256') != foundation.file_hash(batch/'request.json', guard)):
        raise ValueError('원래 첫 fit의 50 step·실패·완료0 증거가 다릅니다')
    parent = foundation.read_parent(foundation.safe_path(root, '7차/work'), guard)
    if (parent is None or parent['run'] != request['parent_run']
            or parent['manifest_sha256'] != request['parent_manifest_sha256']
            or attempt.get('previous_manifest_sha256') != parent['manifest_sha256']):
        raise ValueError('원래 7차 부모 계보가 다릅니다')
    checkpoint = batch/'rep-01/checkpoint'
    if foundation.checkpoint_files(checkpoint, guard) != {
            k: v for k, v in foundation.checkpoint_files(batch/'rep-01/checkpoint-50', guard).items()
            if k in ('config.json', 'model.safetensors')}:
        raise ValueError('최종 checkpoint와 실제 step50 가중치가 다릅니다')
    records = foundation.read_json(batch/'inputs.json')
    if foundation.json_hash(records) != request['data']['inputs_sha256'] or set(records) != {'train', 'tune'}:
        raise ValueError('고정 원래 입력의 해시 또는 자료 구분이 다릅니다')
    protected = [*foundation.protected_periods(root, guard), *request['data']['protected_periods'],
                 parent['manifest']['plan']['validation']]
    foundation.read_exposure_end(foundation.safe_path(root, '7차/work'), guard)
    for path in (root/'7차/work/exposures').glob('*.json'):
        receipt = foundation.read_json(path)['reservation']
        protected.append((receipt['validation_start'], receipt['exposure_end']))
    arrays, contexts, counts = {'train': [], 'tune': []}, [], {}
    for name in ('train', 'tune'):
        if request['data']['plan'][name] != parent['manifest']['plan'][name] or set(records[name]) != set(foundation.MARKETS):
            raise ValueError('원래 고정 기간 또는 시장 구성이 다릅니다')
        a, b = request['data']['plan'][name]
        for market in foundation.MARKETS:
            previous = None
            for segment in records[name][market]:
                bars = [(foundation.stamp(t), v) for t, v in segment]
                parts = foundation.split_segments(bars, a, b, protected)
                if (len(parts) != 1 or parts[0] != bars or len(bars) < 204
                        or (previous is not None and bars[0][0] <= previous)):
                    raise ValueError('복구 입력이 보호·기간·공백 기준을 위반했습니다')
                previous = bars[-1][0]
                arrays[name].append([v for _, v in bars])
            if previous is None:
                raise ValueError('복구 입력 시장의 시계열이 비었습니다')
            count = sum(map(len, records[name][market]))
            if count != request['data']['used_by_market'][market][name+'_rows']:
                raise ValueError('복구 입력 행 수가 원래 계획과 다릅니다')
            counts.setdefault(market, {})[name+'_rows'] = count
            if name == 'train':
                contexts.append([v for _, v in records[name][market][0][:192]])
    if foundation.json_hash(contexts) != request['data']['probe_contexts_sha256']:
        raise ValueError('원래 train context 해시가 다릅니다')
    failures = []
    for path in (work/'evidence').glob('failure-*.json'):
        value = foundation.read_json(path)
        if value.get('run') == run:
            if (value.get('stage') != 'verify' or value.get('settings') != config
                    or value.get('error') != '저장 전후 train context 예측 또는 파일 해시가 다릅니다'):
                raise ValueError('원래 실패 증거가 승인한 첫 회차와 다릅니다')
            failures.append(path)
    if len(failures) != 1:
        raise ValueError('원래 첫 실패 증거를 유일하게 확인하지 못했습니다')
    for path in [batch/'request.json', batch/'completion.json', batch/'rep-01/checkpoint-50/trainer_state.json', *failures]:
        observed[path.relative_to(root).as_posix()] = foundation.file_hash(path, guard)
    prepared = {'training': arrays['train'], 'tuning': arrays['tune'], 'contexts': contexts,
                'records': records, 'data': request['data'], 'rows': sum(v['train_rows'] for v in counts.values())}
    return {'batch': batch, 'request': request, 'status': status, 'parent': parent, 'prepared': prepared,
            'original_files': observed, 'diagnostic': diagnostic, 'checkpoint': checkpoint,
            'diagnostic_sha256': foundation.file_hash(work/'evidence/repair71-readonly-reload-diagnostic.log', guard)}


def check_disk(work, config, guard):
    guard.check()
    if shutil.disk_usage(work).free < config['min_free_bytes']:
        raise OSError('디스크 여유가 부족합니다. 자동 삭제는 하지 않습니다')


def write_aggregate(directory, status, manifests, fields=None):
    foundation.atomic_json(directory/'completion.json', {**status.value, **(fields or {}),
        'complete': len(manifests) == 10, 'validated': False, 'kind': 'unvalidated_train_only',
        'round_manifests': manifests, 'continuity': 'completed_weights_only_no_optimizer_resume'})


def run_batch(root, settings, run, recover_first=False):
    root = Path(root).resolve()
    config = validate_settings(settings)
    if not isinstance(run, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run):
        raise ValueError('batch는 영문·숫자·밑줄·하이픈 1~80자여야 합니다')
    work = foundation.safe_path(root, '8차/work')
    foundation.assert_no_links(work)
    work.mkdir(parents=True, exist_ok=True)
    guard = RepeatGuard(root, config['max_seconds'])
    with foundation.exclusive_lock(foundation.safe_path(root, '7차/work/runner.lock')), \
            foundation.exclusive_lock(work/'runner.lock'):
        guard.check()
        # 이름 변경도 같은 요청의 중복 fit을 허용하지 않습니다. 실패 원본은 보존합니다.
        if not recover_first and ((work/'request.json').exists() or any((work/'runs').glob('*'))):
            raise RuntimeError('8차 batch 예약이 존재하여 재실행을 거부합니다. 자동 재학습·초기화는 없습니다')
        origin = validate_recovery_origin(root, run, config, guard) if recover_first else None
        directory = origin['batch']/'recovery-first' if origin else foundation.safe_path(work, 'runs/'+run)
        if origin:
            check_disk(work, config, guard)
            directory.mkdir(exist_ok=False)
            foundation.atomic_json(directory/'original-status.json', origin['status'])
            status_key = (work/'status.json').relative_to(root).as_posix()
            if foundation.file_hash(directory/'original-status.json', guard) != origin['original_files'][status_key]:
                raise ValueError('원래 상태의 바이트 보존을 확인하지 못했습니다')
            foundation.atomic_json(directory/'origin.json', {'original_files': origin['original_files'],
                'diagnostic_sha256': origin['diagnostic_sha256'],
                'source_sha256': foundation.file_hash(Path(__file__), guard),
                'label': 'recovered_after_reload_check', 'original_verification_failed': True,
                'original_before_after_probe_verified': False, 'consumed_first_fit': 1, 'maximum_new_fits': 9})
        status = RepeatStatus(work/'status.json', config, run)
        status.value['actual_fits_consumed'] = 1 if origin else 0
        if origin:
            status.value.update(step=50, recovery={'recovered_first': True,
                'label': 'recovered_after_reload_check', 'original_verification_failed': True,
                'original_before_after_probe_verified': False, 'consumed_first_fit': 1, 'maximum_new_fits': 9})
        old_signals, manifests = {}, []
        pending_manifest = None
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                old_signals[sig] = signal.signal(sig, lambda *_: guard.event.set())
        try:
            status.update()
            check_disk(work, config, guard)
            parent = origin['parent'] if origin else foundation.read_parent(foundation.safe_path(root, '7차/work'), guard)
            if parent is None:
                raise ValueError('검증된 7차 완료 부모가 없습니다. 원본 모델 초기화는 금지합니다')
            readiness = foundation.read_json(foundation.safe_path(root, '3차/work/status.json'))
            initial = foundation.initial_completion(root, readiness, guard)
            prepared = origin['prepared'] if origin else prepare_data(root, parent, config, guard)
            guard.check()
            if not origin:
                directory.mkdir(parents=True, exist_ok=False)
            request = origin['request'] if origin else {'schema_version': 1, 'run': run, 'kind': 'unvalidated_train_only',
                       'model_id': foundation.MODEL_ID, 'revision': foundation.REVISION, 'settings': config,
                       'parent_run': parent['run'], 'parent_manifest_sha256': parent['manifest_sha256'],
                       'data': prepared['data'], 'initial_completion': initial,
                       'source_sha256': foundation.file_hash(Path(__file__)),
                       'helper_sha256': foundation.file_hash(Path(foundation.__file__)),
                       'created_utc': foundation.now_iso(), 'restart_policy': 'refuse_after_reservation'}
            if not origin:
                foundation.atomic_json(work/'request.json', request)
            foundation.atomic_json(directory/'request.json', request)
            foundation.atomic_json(directory/'inputs.json', prepared['records'])
            request_hash = foundation.file_hash(directory/'request.json', guard)
            inputs_hash = foundation.file_hash(directory/'inputs.json', guard)
            status.update(rows=prepared['rows'], parent_run=parent['run'])
            write_aggregate(directory, status, manifests)
            runtime_config = foundation.validate_settings({k: config[k] for k in foundation.DEFAULTS if k in config})
            runtime = foundation.ChronosRuntime(root, runtime_config, guard)
            if runtime.evidence.get('device') != 'cuda:0':
                raise RuntimeError('실제 CUDA 전용 런타임이 아닙니다')
            source, previous_hash = parent['checkpoint'], parent['manifest_sha256']
            previous_weights = parent['manifest']['execution']['after_weights_sha256']
            training = [runtime.np.asarray(x, dtype=runtime.np.float32) for x in prepared['training']]
            tuning = [runtime.np.asarray(x, dtype=runtime.np.float32) for x in prepared['tuning']]
            if origin:
                round_dir = directory/'rep-01'
                round_dir.mkdir()
                status.update(stage='verify', device='cuda:0')
                binding = {'diagnostic_sha256': origin['diagnostic_sha256'],
                           'original_request_sha256': request_hash, 'original_source_sha256': ORIGIN_SOURCE_SHA256}
                first = runtime.load(origin['checkpoint'])
                proof = capture_probe(runtime, first, origin['checkpoint'], prepared['contexts'],
                                      round_dir/'fresh-load-1.json', guard, binding)
                del first
                runtime.torch.cuda.empty_cache()
                second = runtime.load(origin['checkpoint'])
                repeated = capture_probe(runtime, second, origin['checkpoint'], prepared['contexts'],
                                         round_dir/'fresh-load-2.json', guard, binding)
                del second
                runtime.torch.cuda.empty_cache()
                compare_probes(runtime, proof, repeated, round_dir/'comparison.json')
                after = proof['parameter_sha256']
                if after == previous_weights or after != origin['diagnostic']['weights_sha256']:
                    raise ValueError('원래 seed 변화 또는 무학습 진단 가중치 결합이 다릅니다')
                recovered = {'schema_version': 1, 'run': run, 'rep_index': 1,
                    'model_id': foundation.MODEL_ID, 'revision': foundation.REVISION, 'settings': config,
                    'kind': 'unvalidated_train_only', 'complete': True, 'validated': False,
                    'checkpoint_verified': True, 'evaluation_performed': False, 'candidate_ready': False,
                    'promote': False, 'metrics': None, 'verification_method': 'recovered_after_reload_check',
                    'original_verification_failed': True, 'original_before_after_probe_verified': False,
                    'previous_manifest_sha256': previous_hash, 'seed_run': parent['run'],
                    'seed_manifest_sha256': parent['manifest_sha256'], 'request_sha256': request_hash,
                    'inputs_file_sha256': inputs_hash, 'checkpoint': str(origin['checkpoint'].relative_to(work)),
                    'checkpoint_files': proof['checkpoint_files'], 'original_files': origin['original_files'],
                    'diagnostic_sha256': origin['diagnostic_sha256'],
                    'new_fit_calls': 0, 'actual_fits_consumed': 1,
                    'execution': {**runtime.evidence, 'before_weights_sha256': previous_weights,
                        'after_weights_sha256': after, 'weight_changed': True, 'completed_steps': 50,
                        'loss': origin['status']['loss']}, 'finished_utc': foundation.now_iso()}
                pending_manifest = (round_dir/'manifest.json', recovered)
                foundation.atomic_json(round_dir/'manifest.json', recovered)
                digest = foundation.file_hash(round_dir/'manifest.json', guard)
                manifests.append({'rep_index': 1, 'manifest_sha256': digest})
                pending_manifest = None
                source, previous_hash, previous_weights = origin['checkpoint'], digest, after
                status.update(completed_repetitions=1, checkpoint_verified=True, loss=origin['status']['loss'])
                write_aggregate(directory, status, manifests)
                foundation.atomic_json(work/'knowledge.json', {'run': run, 'rep_index': 1,
                    'kind': 'unvalidated_train_only', 'validated': False, 'promote': False,
                    'manifest_sha256': digest, 'manifest': str((round_dir/'manifest.json').relative_to(work)),
                    'checkpoint': str(source.relative_to(work)), 'request_sha256': request_hash,
                    'verification_method': 'recovered_after_reload_check'})
            for index in range(2 if origin else 1, 11):
                check_disk(work, config, guard)
                round_dir = directory/f'rep-{index:02d}'
                round_dir.mkdir(exist_ok=False)
                foundation.atomic_json(round_dir/'attempt.json', {'run': run, 'rep_index': index,
                    'previous_manifest_sha256': previous_hash, 'request_sha256': request_hash,
                    'prepared_utc': foundation.now_iso(), 'state': 'prepared',
                    'record_type': 'fit_preparation'})
                status.update(rep_index=index, state='training', stage='finetune', device='cuda:0',
                              step=(index-1)*50, loss=None, checkpoint_verified=False)
                pipeline = runtime.load(source)
                before = runtime.weights_hash(pipeline)
                if before != previous_weights:
                    raise ValueError('직전 완료 가중치를 실제로 이어받지 못했습니다')
                proxy = StepProxy(status, index)
                trained = runtime.train(pipeline, training, tuning, round_dir, proxy)
                runtime.check_device(trained)
                after = runtime.weights_hash(trained)
                if before == after or proxy.value['step'] != 50 or proxy.value['loss'] is None:
                    raise ValueError('50-step fit·유한 손실·실제 가중치 변화 증거가 부족합니다')
                status.update(state='running', stage='verify')
                checkpoint = round_dir/'checkpoint'
                probe = capture_probe(runtime, trained, checkpoint, prepared['contexts'],
                                      round_dir/'probe-before-reload.json', guard)
                if probe['parameter_sha256'] != after:
                    raise ValueError('저장 전 정규화된 가중치 해시가 다릅니다')
                del trained, pipeline
                runtime.torch.cuda.empty_cache()
                reloaded = runtime.load(checkpoint)
                repeated = capture_probe(runtime, reloaded, checkpoint, prepared['contexts'],
                                         round_dir/'probe-after-reload.json', guard)
                compare_probes(runtime, probe, repeated, round_dir/'comparison.json')
                files = probe['checkpoint_files']
                del reloaded
                runtime.torch.cuda.empty_cache()
                check_disk(work, config, guard)
                manifest = {'schema_version': 1, 'run': run, 'rep_index': index,
                    'kind': 'unvalidated_train_only', 'complete': True, 'validated': False,
                    'checkpoint_verified': True, 'evaluation_performed': False, 'candidate_ready': False,
                    'promote': False, 'metrics': None, 'model_id': foundation.MODEL_ID,
                    'revision': foundation.REVISION, 'settings': config,
                    'previous_manifest_sha256': previous_hash, 'seed_run': parent['run'],
                    'seed_manifest_sha256': parent['manifest_sha256'], 'request_sha256': request_hash,
                    'inputs_file_sha256': inputs_hash, 'data_sha256': foundation.json_hash(prepared['data']),
                    'checkpoint': str(checkpoint.relative_to(work)),
                    'checkpoint_files': files, 'probe_scope': 'train_context_only',
                    'probe_sha256': foundation.file_hash(round_dir/'probe-before-reload.json', guard),
                    'verification_method': 'normalized_fp32_before_after_reload',
                    'continuity': 'completed_weights_only_no_optimizer_resume',
                    'execution': {**runtime.evidence, 'before_weights_sha256': before,
                                  'after_weights_sha256': after, 'weight_changed': True,
                                  'completed_steps': 50, 'loss': proxy.value['loss']},
                    'finished_utc': foundation.now_iso()}
                pending_manifest = (round_dir/'manifest.json', manifest)
                foundation.atomic_json(round_dir/'manifest.json', manifest)
                digest = foundation.file_hash(round_dir/'manifest.json', guard)
                manifests.append({'rep_index': index, 'manifest_sha256': digest})
                pending_manifest = None
                source, previous_hash, previous_weights = checkpoint, digest, after
                status.update(completed_repetitions=index, checkpoint_verified=True)
                write_aggregate(directory, status, manifests)
                foundation.atomic_json(work/'knowledge.json', {'run': run, 'rep_index': index,
                    'kind': 'unvalidated_train_only', 'validated': False, 'promote': False,
                    'manifest_sha256': digest, 'checkpoint': str(checkpoint.relative_to(work)),
                    'manifest': str((round_dir/'manifest.json').relative_to(work)),
                    'request_sha256': request_hash})
            status.update(state='complete', stage='complete', finished_utc=foundation.now_iso())
            write_aggregate(directory, status, manifests)
            return status.value
        except Exception as exc:
            original_traceback = traceback.format_exc()
            confirmation_error = None
            if pending_manifest is not None:
                path, expected = pending_manifest
                try:
                    # 원자 교체 뒤 fsync 오류도 원문은 실패로 남기되 실제 완료 수는 되돌리지 않습니다.
                    if path.is_file() and foundation.read_json(path) == expected:
                        manifests.append({'rep_index': expected['rep_index'],
                                          'manifest_sha256': foundation.file_hash(path)})
                except Exception:
                    confirmation_error = traceback.format_exc()
            failure = {'utc': foundation.now_iso(), 'type': type(exc).__name__, 'error': str(exc),
                       'traceback': original_traceback, 'argv': sys.argv, 'cwd': str(Path.cwd()),
                       'run': run, 'stage': status.value['stage'], 'settings': config,
                       'commit_confirmation_error': confirmation_error,
                       'completed_repetitions': len(manifests), **foundation.routing_counters(root)}
            evidence = work/'evidence'/('failure-'+uuid.uuid4().hex+'.json')
            foundation.atomic_json(evidence, failure)
            state = ('stopped' if isinstance(exc, foundation.StopRequested) else
                     'waiting' if isinstance(exc, foundation.NotReady) else 'error')
            failure_fields = {'state': state, 'has_error': state == 'error',
                              'finished_utc': foundation.now_iso(), 'updated_utc': foundation.now_iso(),
                              'completed_repetitions': len(manifests), 'candidate_ready': False,
                              'checkpoint_verified': bool(manifests) and len(manifests) == status.value['rep_index']}
            try:
                status.update(**failure_fields)
            finally:
                if directory.exists():
                    write_aggregate(directory, status, manifests, failure_fields)
            raise
        finally:
            for sig, previous in old_signals.items():
                signal.signal(sig, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true', required=True, help='고정 10회 fit을 한 batch로 실행')
    parser.add_argument('--batch', required=True, help='8차의 고유 batch 식별자; 예약 이후 재실행 거부')
    parser.add_argument('--config', default='8차/train-only-config.json', help='프로젝트 내부 고정 설정 JSON')
    parser.add_argument('--recover-first', action='store_true', help='승인된 동일 실패 batch의 첫 fit을 재학습 없이 복구하고 최대 9회 추가')
    args = parser.parse_args(argv)
    try:
        settings = foundation.read_json(foundation.safe_path(ROOT, args.config))
        result = run_batch(ROOT, settings, args.batch, recover_first=args.recover_first)
        print(json.dumps({key: result[key] for key in ('run', 'state', 'completed_repetitions',
              'step', 'checkpoint_verified', 'evaluation_performed', 'promote')}, ensure_ascii=False))
        return 0
    except foundation.StopRequested:
        print('중단 요청을 보존했습니다. batch 재실행은 거부됩니다.', file=sys.stderr)
        return 130
    except foundation.AlreadyRunning:
        print('7차 또는 8차 실행 중이므로 중복 실행을 거부합니다.', file=sys.stderr)
        return 2
    except foundation.NotReady:
        print('고정 학습 자료 준비 대기입니다. 보호 기준은 변경하지 않습니다.', file=sys.stderr)
        return 3
    except Exception as exc:
        print(f'반복 학습 실행 실패: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
