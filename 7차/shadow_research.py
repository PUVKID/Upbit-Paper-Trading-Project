"""고정 Chronos-2의 완료 가중치를 이어 학습하는 독립 연구 전용 실행기."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import signal
import sqlite3
import sys
import threading
import time
import traceback
import uuid


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'amazon/chronos-2'
REVISION = '29ec3766d36d6f73f0696f85560a422f50e8498c'
MODEL_SHA256 = 'ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42'
MARKETS = ('KRW-BTC', 'KRW-SOL', 'KRW-ETH', 'KRW-XRP', 'KRW-DOGE')
UTC = dt.timezone.utc
BAR = dt.timedelta(minutes=5)
PROTECTED = [('2026-10-19T00:00:00Z', '2026-10-20T00:00:00Z')]
DEFAULTS = {'model_id': MODEL_ID, 'revision': REVISION, 'steps': 50,
            'context_length': 256, 'batch_size': 8, 'prediction_length': 12,
            'learning_rate': 1e-6, 'train_days': 30, 'tune_days': 2,
            'validation_days': 2, 'evaluation_windows_per_market': 4,
            'max_rows_per_market': 50000, 'min_free_bytes': 20_000_000_000,
            'max_seconds': 1800, 'seed': 42}
PACKAGES = {'chronos-forecasting': '2.3.2', 'transformers': '4.57.6',
            'accelerate': '1.12.0', 'huggingface-hub': '0.36.0',
            'tokenizers': '0.22.2', 'einops': '0.8.2', 'regex': '2026.9.29'}


class NotReady(RuntimeError):
    """새로운 미사용 평가 자료가 아직 충분하지 않습니다."""


class StopRequested(RuntimeError):
    """사용자의 중단 신호를 보존합니다."""


class AlreadyRunning(RuntimeError):
    """기존 실행의 상태 파일을 덮어쓰지 않습니다."""


def stamp(value):
    value = dt.datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
    if not isinstance(value, dt.datetime) or value.utcoffset() != dt.timedelta(0):
        raise ValueError('시각은 시간대가 명시된 UTC여야 합니다')
    return value.astimezone(UTC)


def now_iso():
    return dt.datetime.now(UTC).isoformat()


def validate_settings(value):
    if not isinstance(value, dict) or set(value)-set(DEFAULTS):
        raise ValueError('지원하지 않는 연구 설정입니다')
    result = {**DEFAULTS, **value}
    if result['model_id'] != MODEL_ID or result['revision'] != REVISION:
        raise ValueError('공식 고정 모델과 revision만 허용합니다')
    limits = {'steps': (1, 100000), 'context_length': (16, 8192), 'batch_size': (1, 256),
              'prediction_length': (1, 1024), 'train_days': (1, 3650), 'tune_days': (1, 365),
              'validation_days': (1, 365), 'evaluation_windows_per_market': (1, 1000),
              'max_rows_per_market': (100, 2_000_000), 'min_free_bytes': (5_000_000_000, 10**15),
              'max_seconds': (30, 86400), 'seed': (0, 2**31-1)}
    for key, (low, high) in limits.items():
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError(f'{key}: {low}~{high}의 정수가 필요합니다')
    lr = result['learning_rate']
    if type(lr) not in (float, int) or not math.isfinite(lr) or not 0 < lr <= 0.001:
        raise ValueError('학습률은 유한한 양수이며 0.001 이하여야 합니다')
    return result


def assert_no_links(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f'연결 경로를 허용하지 않습니다: {path}')
    if path.is_dir():
        for folder, directories, files in os.walk(path, followlinks=False):
            for name in directories + files:
                if (Path(folder)/name).is_symlink():
                    raise ValueError('연구 자료 안의 연결 파일을 허용하지 않습니다')


def safe_path(root, relative):
    root = Path(root).resolve()
    p = Path(relative)
    if (p.is_absolute() or PureWindowsPath(str(relative)).drive or '\\' in str(relative)
            or '..' in p.parts or not p.parts):
        raise ValueError('프로젝트 내부 상대 경로만 허용합니다')
    target = root
    for part in p.parts:
        target /= part
        if target.is_symlink():
            raise ValueError('경로 구성 요소에 연결이 있습니다')
    if not target.resolve().is_relative_to(root):
        raise ValueError('경로가 프로젝트 밖으로 나갑니다')
    return target


def read_json(path):
    def invalid(value):
        raise ValueError(f'비유한 JSON 값: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def atomic_json(path, value):
    payload = (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                          indent=2)+'\n').encode('utf-8')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('JSON 대상이 연결 파일입니다')
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if sys.platform == 'linux':
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def file_hash(path, guard=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024*1024):
            if guard:
                guard.check()
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def initial_completion(root, readiness, guard):
    size = readiness.get('ingested_archive_bytes')
    verified = readiness.get('verified_archive_bytes')
    if (readiness.get('schema_version') != 1 or type(size) is not int or size < 50_000_000_000
            or type(verified) is not int or verified < size):
        raise NotReady('검증·적재된 초기 50GB 원본 근거가 부족합니다')
    control = read_json(safe_path(root, '4차/work/research_control.json'))
    batches = [b for b in control['batches'] if b.get('id') == 'bootstrap']
    if len(batches) != 1:
        raise NotReady('유일한 초기 연구 완료 기록이 필요합니다')
    batch = batches[0]
    attempt = batch.get('attempts')
    if (batch.get('kind') != 'bootstrap' or batch.get('status') != 'completed'
            or batch.get('baseline_bytes') != 0 or batch.get('target_bytes') != 50_000_000_000
            or type(attempt) is not int or attempt < 1 or not batch.get('completed_at')):
        raise NotReady('초기 50GB 연구가 정상 완료되지 않았습니다')
    record = read_json(safe_path(root, f'4차/work/research_control_attempts/bootstrap-{attempt}.json'))
    run = record.get('run')
    if not isinstance(run, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', run):
        raise ValueError('초기 연구 실행 ID가 잘못됐습니다')
    binding = {k:record[k] for k in ('schema_version','batch_id','claim','attempt','target_bytes','run')}
    report = record['report']
    directory = safe_path(root, 'work/models/versions/'+run)
    manifest = read_json(directory/'manifest.json')
    report_file = read_json(safe_path(root, 'work/reports/experiments/'+run+'.json'))
    initial = report.get('external_resolution', {})
    if (record.get('phase') != 'complete' or run != batch.get('run') or record['attempt'] != attempt
            or record.get('batch_id') != 'bootstrap' or record.get('target_bytes') != 50_000_000_000
            or record.get('models_dir') != 'work/models' or record.get('reports_dir') != 'work/reports'
            or report.get('run') != run or report.get('paper_only') is not True
            or report.get('research', {}).get('status') != 'complete'
            or report.get('research_batch') != binding or manifest.get('research_batch') != binding
            or manifest.get('run') != run or manifest.get('paper_only') is not True
            or manifest.get('schema_version') != 2 or manifest.get('execution_policy') != 'horizon_or_risk_v2'
            or json_hash(manifest) != record.get('manifest_sha256')
            or json_hash(report) != record.get('report_sha256') or json_hash(report_file) != record.get('report_sha256')
            or initial.get('training_ready') is not True or type(initial.get('ingested_archive_bytes')) is not int
            or initial['ingested_archive_bytes'] < 50_000_000_000
            or initial.get('target_archive_bytes') != 50_000_000_000
            or set(manifest.get('packages', {})) != set(MARKETS)):
        raise ValueError('초기 완료 보고서·manifest·요청 결합이 일치하지 않습니다')
    hashes = {}
    for package in manifest['packages'].values():
        for name in package['models']:
            if not isinstance(name, str) or Path(name).name != name or '\\' in name or not name.endswith('.ubj') or name in hashes:
                raise ValueError('초기 연구 모델 경로·중복이 잘못됐습니다')
            hashes[name] = file_hash(safe_path(directory, name), guard)
    if hashes != record.get('model_sha256') or len(hashes) != report.get('training', {}).get('total_candidates'):
        raise ValueError('초기 완료 모델의 실제 해시가 다릅니다')
    return {'bootstrap_run':run, 'bootstrap_manifest_sha256':record['manifest_sha256'],
            'reported_ingested_bytes':size, 'current_increment_training_ready':readiness.get('training_ready'),
            'current_increment_target_bytes':readiness.get('target_archive_bytes'),
            'full_archive_reverified':False}


def reserve_exposure(work, run, plan, data, parent):
    directory = safe_path(work, 'runs/'+run)
    path = safe_path(work, 'exposures/'+run+'.json')
    if path.exists():
        raise ValueError('기존 평가 예약을 덮어쓸 수 없습니다')
    start, end = map(stamp, plan['validation'])
    if start >= end or stamp(data['exposure_end']) != end:
        raise ValueError('평가 예약의 기간과 자료 기간이 다릅니다')
    value = {'schema_version':1, 'run':run, 'model_id':MODEL_ID, 'revision':REVISION,
             'parent_run':parent['run'] if parent else None,
             'parent_manifest_sha256':parent['manifest_sha256'] if parent else None,
             'validation_start':start.isoformat(), 'exposure_end':end.isoformat(),
             'data_sha256':file_hash(directory/'data.json'),
             'extracted_sha256':data['extracted_sha256'], 'reserved_utc':now_iso(),
             'scope':'재로드 검증용 context와 후속 예측 전에 전체 평가 기간을 보수적으로 소비'}
    digest = json_hash(value)
    atomic_json(path, {'reservation':value, 'sha256':digest})
    return digest


def read_exposure_end(work, guard=None):
    work = Path(work)
    found, end = {}, None
    for path in sorted((work/'exposures').glob('*.json')):
        assert_no_links(path)
        record = read_json(path)
        value = record['reservation']
        run = value.get('run')
        if (record.get('sha256') != json_hash(value) or value.get('schema_version') != 1
                or value.get('model_id') != MODEL_ID or value.get('revision') != REVISION
                or not isinstance(run,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',run)
                or path.name != run+'.json' or run in found):
            raise ValueError('평가 예약의 결합 또는 해시가 다릅니다')
        directory = safe_path(work,'runs/'+run)
        data = read_json(directory/'data.json')
        endpoint = stamp(value['exposure_end'])
        if (stamp(value['validation_start']) >= endpoint or stamp(data['exposure_end']) != endpoint
                or file_hash(directory/'data.json',guard) != value['data_sha256']
                or data.get('extracted_sha256') != value['extracted_sha256']):
            raise ValueError('실패 회차를 포함한 평가 예약과 실제 자료 결합이 다릅니다')
        found[run] = record['sha256']
        end = max(end,endpoint) if end else endpoint
    for pattern in ('*/failure.json','*/manifest.json'):
        for path in (work/'runs').glob(pattern):
            value = read_json(path)
            digest = value.get('exposure_reservation_sha256')
            if digest is not None and found.get(path.parent.name) != digest:
                raise ValueError('완료 또는 실패 회차의 평가 예약이 누락·변경됐습니다')
    return end.isoformat() if end else None


def routing_counters(root):
    records = []
    for path in (Path(root)/'.codex/routing-state').glob('*.json'):
        if re.fullmatch(r'[0-9a-f]{24}\.json', path.name):
            try:
                value = read_json(path)
                if all(type(value.get(k)) is int and value[k] >= 0 for k in ('repair_used','input_fix_used')):
                    records.append((path, value))
            except (OSError, ValueError):
                continue
    if len(records) != 1:
        return {'repair_used':None, 'input_fix_used':None, 'counter_evidence':'상태 원본을 유일하게 확인하지 못함'}
    path, value = records[0]
    return {k:value[k] for k in ('repair_used','input_fix_used')} | {
        'counter_evidence':str(path.relative_to(root)), 'counter_observed_utc':now_iso()}


class Guard:
    def __init__(self, stop, seconds):
        self.stop = Path(stop)
        self.deadline = time.monotonic()+seconds
        self.event = threading.Event()

    def check(self):
        if self.event.is_set() or self.stop.exists():
            raise StopRequested('연구 중단이 요청됐습니다')
        if time.monotonic() >= self.deadline:
            raise TimeoutError('연구 실행의 유한 시간 상한을 초과했습니다')


@contextlib.contextmanager
def exclusive_lock(path):
    if sys.platform != 'linux':
        raise RuntimeError('실행과 운영 DB 접근은 Linux 컨테이너 전용입니다')
    import fcntl
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a+b') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AlreadyRunning('기존 shadow 연구가 실행 중입니다') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class Status:
    def __init__(self, path, config, run):
        self.path = Path(path)
        self.value = {'state': 'running', 'model_id': MODEL_ID, 'stage': 'preflight',
                      'step': 0, 'total_steps': config['steps'], 'loss': None,
                      'started_utc': now_iso(), 'finished_utc': None, 'run': run,
                      'parent_run': None, 'device': None, 'rows': None,
                      'context_length': config['context_length'], 'horizon_minutes': 5*config['prediction_length'],
                      'batch_size': config['batch_size'], 'learning_rate': config['learning_rate'],
                      'validation_mae': None, 'baseline_mae': None, 'direction_accuracy': None,
                      'net_return': None, 'trades': None, 'candidate_ready': False,
                      'checkpoint_verified': False, 'has_error': False, 'error': None,
                      'has_status_warning': False}

    def update(self, **fields):
        value = {**self.value, **fields, 'updated_utc': now_iso()}
        if value['candidate_ready'] and not (value['checkpoint_verified'] and value['state'] == 'complete'):
            raise ValueError('검증된 완료 연구 후보만 준비 상태가 될 수 있습니다')
        atomic_json(self.path, value)
        self.value = value


def make_plan(config, now, ancestor_end=None):
    now = stamp(now)
    end = now.replace(minute=now.minute//5*5, second=0, microsecond=0)
    validation_start = end-dt.timedelta(days=config['validation_days'])
    if ancestor_end:
        validation_start = max(validation_start, stamp(ancestor_end))
    if end <= validation_start:
        raise NotReady('부모 모델에 노출되지 않은 미래 평가 구간이 아직 없습니다')
    tune_start = validation_start-dt.timedelta(days=config['tune_days'])
    return {'train': [(tune_start-dt.timedelta(days=config['train_days'])).isoformat(), tune_start.isoformat()],
            'tune': [tune_start.isoformat(), validation_start.isoformat()],
            'validation': [validation_start.isoformat(), end.isoformat()]}


def split_segments(rows, start, end, protected):
    start, end = stamp(start), stamp(end)
    forbidden = [(stamp(a), stamp(b)) for a, b in protected]
    segments, current, previous = [], [], None
    for moment, value in rows:
        moment = stamp(moment)
        if previous is not None and moment <= previous:
            raise ValueError('중복 또는 역행하는 시계열입니다')
        gap = previous is not None and moment-previous != BAR
        previous = moment
        # 닫힌 시각뿐 아니라 해당 봉이 차지하는 기간도 보호 구간과 겹치지 않습니다.
        allowed = start <= moment < end and not any(moment >= a and moment-BAR < b for a, b in forbidden)
        if gap or not allowed:
            if current:
                segments.append(current)
            current = []
        if allowed:
            if not math.isfinite(value) or value <= 0:
                raise ValueError('시계열 종가는 유한한 양수여야 합니다')
            current.append((moment, float(value)))
    if current:
        segments.append(current)
    return segments


def evaluation_windows(segments, context, horizon, limit):
    windows = []
    for segment in segments:
        for i in range(context, len(segment)-horizon+1, horizon):
            windows.append(([v for _, v in segment[i-context:i]],
                            [v for _, v in segment[i:i+horizon]]))
    if not windows:
        raise NotReady('공백과 보호 기간을 제외한 연속 평가 창이 부족합니다')
    if len(windows) > limit:
        indexes = [i*(len(windows)-1)//max(1, limit-1) for i in range(limit)]
        windows = [windows[i] for i in indexes]
    return windows


@contextlib.contextmanager
def readonly_database(path, guard):
    if sys.platform != 'linux':
        raise RuntimeError('운영 SQLite는 Linux 안에서만 읽습니다')
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    db = sqlite3.connect(Path(path).as_uri()+'?mode=ro', uri=True, timeout=3)
    deadline = time.monotonic()+25
    db.set_progress_handler(lambda: int(time.monotonic()>deadline or guard.event.is_set()
                                       or guard.stop.exists()), 10000)
    try:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        yield db
    finally:
        db.close()


def protected_periods(root, guard):
    path = safe_path(root, 'work/data/experiment_audit.sqlite3')
    if not path.is_file():
        raise NotReady('기존 보호 시험 감사 DB가 필요합니다')
    with readonly_database(path, guard) as db:
        values = db.execute('SELECT start,end FROM final_tests').fetchall()
    for a, b in values:
        if stamp(a) >= stamp(b):
            raise ValueError('보호 시험 기간이 유효하지 않습니다')
    return [*PROTECTED, *values]


def extract_rows(root, start, end, limit, guard):
    start, end = stamp(start), stamp(end)
    if end > dt.datetime.now(UTC):
        raise ValueError('미래 완료 시각까지 자료를 읽을 수 없습니다')
    result, digest = {}, hashlib.sha256()
    path = safe_path(root, 'work/data/candles.sqlite3')
    with readonly_database(path, guard) as db:
        for market in MARKETS:
            guard.check()
            rows = db.execute('SELECT open_time,close_time,open,high,low,close,volume FROM candles '
                              'WHERE market=? AND tf=? AND close_time>=? AND close_time<? '
                              'ORDER BY close_time LIMIT ?',
                              (market, '5', start.isoformat(), end.isoformat(), limit+1)).fetchall()
            if len(rows) > limit:
                raise ValueError('읽기 행 상한을 초과했습니다. 자료를 몰래 축소하지 않습니다')
            output, last = [], None
            for opening, closing, o, h, low, c, volume in rows:
                guard.check()
                opened, closed = stamp(opening), stamp(closing)
                numbers = (o, h, low, c, volume)
                if (any(type(v) not in (int, float) or not math.isfinite(v) for v in numbers)
                        or min(o, h, low, c) <= 0 or volume < 0 or not low <= min(o, c) <= max(o, c) <= h
                        or closed-opened != BAR or closed.timestamp()%300 != 0
                        or not start <= closed < end or (last is not None and closed <= last)):
                    raise ValueError(f'{market}: 불완전·중복·비정상 OHLCV 봉입니다')
                last = closed
                output.append((closed, float(c)))
                digest.update(json.dumps([market, opening, closing, *numbers], allow_nan=False).encode('utf-8')+b'\n')
            result[market] = output
    return result, {'source': 'upbit', 'timeframe_minutes': 5, 'currency': 'KRW',
                    'db': 'work/data/candles.sqlite3', 'rows': sum(map(len, result.values())),
                    'rows_by_market': {k:len(v) for k, v in result.items()},
                    'extracted_sha256': digest.hexdigest(), 'range': [start.isoformat(), end.isoformat()],
                    'external_data_used_for_training': False}


def forecast_metrics(predictions, targets, origins):
    if not predictions or len(predictions) != len(targets) or len(targets) != len(origins):
        raise ValueError('예측·정답·기준 가격의 표본 수가 다릅니다')
    error = baseline = count = directions = 0
    for predicted, actual, origin in zip(predictions, targets, origins):
        if (not predicted or len(predicted) != len(actual) or not math.isfinite(origin) or origin <= 0
                or any(not math.isfinite(v) for v in [*predicted, *actual])):
            raise ValueError('예측과 평가 값이 유효하지 않습니다')
        error += sum(abs(a-p)/origin for p, a in zip(predicted, actual))
        baseline += sum(abs(a-origin)/origin for a in actual)
        count += len(actual)
        directions += ((predicted[-1]>origin)-(predicted[-1]<origin) ==
                       (actual[-1]>origin)-(actual[-1]<origin))
    return {'validation_mae': error/count, 'baseline_mae': baseline/count,
            'direction_accuracy': directions/len(targets), 'windows': len(targets),
            'mae_unit': 'absolute_error_divided_by_context_last_close',
            'net_return': None, 'trades': None, 'promote': False}


def checkpoint_files(directory, guard=None):
    directory = Path(directory)
    assert_no_links(directory)
    files = {}
    for path in sorted(directory.rglob('*')):
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = {'bytes': path.stat().st_size,
                                                            'sha256': file_hash(path, guard)}
    if 'config.json' not in files or 'model.safetensors' not in files:
        raise ValueError('완료 checkpoint의 설정 또는 safetensors가 없습니다')
    return files


def validate_pretrained(directory, guard=None):
    directory = Path(directory)
    assert_no_links(directory)
    if directory.name != REVISION:
        raise ValueError('고정 revision 경로가 아닙니다')
    if not all((directory/name).is_file() for name in ('config.json','model.safetensors','download-evidence.json')):
        raise ValueError('부모 다운로드가 완료된 실제 파일과 증거가 필요합니다')
    records = read_json(directory/'download-evidence.json')
    if records.get('model_id') != MODEL_ID or records.get('revision') != REVISION:
        raise ValueError('부모 다운로드의 고정 모델 증거가 다릅니다')
    files = {r['name']:r for r in records['files']}
    for name in ('config.json', 'model.safetensors'):
        path = directory/name
        record = files[name]
        if path.stat().st_size != record['bytes'] or file_hash(path, guard) != record['sha256']:
            raise ValueError('사전학습 모델 파일의 크기 또는 해시가 다릅니다')
    if files['model.safetensors']['sha256'] != MODEL_SHA256:
        raise ValueError('공식 LFS 모델 해시가 다릅니다')
    return records


def make_manifest(run, parent, plan, settings, data, metrics, checkpoint, execution):
    return {'schema_version': 1, 'run': run, 'complete': True, 'checkpoint_verified': True,
            'model_id': MODEL_ID, 'revision': REVISION, 'paper_only': True, 'promote': False,
            'continuity': 'completed_weights_only_no_optimizer_resume',
            'parent_run': parent['run'] if parent else None,
            'parent_manifest_sha256': parent['manifest_sha256'] if parent else None,
            'exposure_end': data['exposure_end'], 'plan': plan, 'settings': settings,
            'data': data, 'metrics': metrics, 'execution': execution,
            'checkpoint_files': checkpoint_files(checkpoint), 'finished_utc': now_iso()}


def read_completion(directory, manifest, expected=None):
    digest = manifest.get('completion_sha256')
    if not isinstance(digest,str) or not re.fullmatch(r'[0-9a-f]{64}',digest) or (expected is not None and expected != digest):
        raise ValueError('완료 스냅샷의 해시 결합이 유효하지 않습니다')
    path = Path(directory)/'completion.json'
    assert_no_links(path)
    if file_hash(path) != digest:
        raise ValueError('완료 스냅샷 파일 해시가 다릅니다')
    value = read_json(path)
    if (value.get('run') != manifest['run'] or value.get('parent_run') != manifest.get('parent_run')
            or value.get('model_id') != MODEL_ID or value.get('state') != 'complete'
            or value.get('stage') != 'complete' or value.get('candidate_ready') is not True
            or value.get('checkpoint_verified') is not True or value.get('has_error') is not False
            or value.get('error') is not None or value.get('has_status_warning') is not False
            or value.get('step') != manifest['settings']['steps']
            or value.get('total_steps') != manifest['settings']['steps']
            or value.get('device') != manifest['execution']['device']
            or any(value.get(k) != manifest['metrics'][k] for k in
                   ('validation_mae','baseline_mae','direction_accuracy','net_return','trades'))):
        raise ValueError('완료 스냅샷의 상태·학습·평가 의미가 manifest와 다릅니다')
    return value


def committed_completion(work, pointer):
    path = Path(work)/'knowledge.json'
    try:
        current = read_json(path)
    except FileNotFoundError:
        return None
    if any(current.get(k) != pointer[k] for k in ('run','manifest_sha256','completion_sha256')):
        return None
    directory = safe_path(work,'runs/'+pointer['run'])
    if file_hash(directory/'manifest.json') != pointer['manifest_sha256']:
        raise ValueError('커밋된 완료 manifest를 확인할 수 없습니다')
    manifest = read_json(directory/'manifest.json')
    return read_completion(directory,manifest,pointer['completion_sha256'])


def record_finalization_error(root, directory, exc, committed=True):
    record = {'utc':now_iso(),'type':type(exc).__name__,'error':str(exc),
              'traceback':''.join(traceback.format_exception(type(exc),exc,exc.__traceback__)),
              'committed':committed,'phase':'display_finalization' if committed else 'commit_confirmation', 'argv':sys.argv,
              'cwd':str(Path.cwd()), **routing_counters(root)}
    try:
        atomic_json(Path(directory)/'finalization-errors'/(uuid.uuid4().hex+'.json'),record)
    except Exception as logging_error:
        # 저장 장치 전체가 실패하면 비공개 실행 stderr에 두 오류 원문을 함께 보존합니다.
        print(json.dumps({**record,'logging_error':str(logging_error)},ensure_ascii=False),file=sys.stderr)


def finalize_committed(root, work, run, manifest, completion, initial_error=None):
    directory = safe_path(work,'runs/'+run)
    if initial_error is not None:
        record_finalization_error(root,directory,initial_error)
    for _ in range(2):
        try:
            # 봉인된 시각을 포함한 payload를 그대로 저장하여 raw SHA 결합을 유지합니다.
            atomic_json(Path(work)/'status.json',completion)
            atomic_json(directory/'status.json',completion)
            return dict(completion)
        except Exception as exc:
            record_finalization_error(root,directory,exc)
    print('연구 완료 커밋은 검증됐으나 표시 저장이 대기 중입니다. 재시작 시 최종화만 재시도합니다.',file=sys.stderr)
    warning = {**completion,'has_status_warning':True}
    try:
        atomic_json(Path(work)/'status.json',warning)
    except Exception as exc:
        record_finalization_error(root,directory,exc)
    return warning


def resume_finalization(root, work, guard):
    path = Path(work)/'knowledge.json'
    if not path.exists():
        return None
    pointer = read_json(path)
    if 'completion_sha256' not in pointer:
        return None
    try:
        cached = read_json(Path(work)/'status.json')
    except (OSError,ValueError):
        cached = None
    if (isinstance(cached,dict) and isinstance(cached.get('run'),str) and cached['run']
            and cached['run'] != pointer['run']):
        return None
    directory = safe_path(work,'runs/'+pointer['run'])
    caches = (Path(work)/'status.json',directory/'status.json')
    try:
        synchronized = all(file_hash(p) == pointer['completion_sha256'] for p in caches)
    except OSError:
        synchronized = False
    if synchronized:
        return None
    # 재시작 검사는 한 번만 모델 파일까지 대조하고 모델 로드·평가·재학습은 하지 않습니다.
    parent = read_parent(work,guard)
    return finalize_committed(root,work,parent['run'],parent['manifest'],parent['completion'])


def read_parent(work, guard=None):
    work = Path(work)
    pointer = work/'knowledge.json'
    if not pointer.exists():
        if any((work/'runs').glob('*/manifest.json')):
            raise ValueError('완료 이력이 있는데 연구 포인터가 없습니다. 원본 초기화를 거부합니다')
        return None
    assert_no_links(pointer)
    item = read_json(pointer)
    latest, seen, exposure = None, set(), None
    for _ in range(10000):
        run = item.get('run')
        if not isinstance(run, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', run) or run in seen:
            raise ValueError('연구 계보의 ID 또는 순환이 유효하지 않습니다')
        seen.add(run)
        directory = safe_path(work, 'runs/'+run)
        path = directory/'manifest.json'
        assert_no_links(directory)
        if file_hash(path, guard) != item.get('manifest_sha256'):
            raise ValueError('부모 manifest 해시가 다릅니다')
        value = read_json(path)
        if (value.get('run') != run or value.get('schema_version') != 1
                or value.get('complete') is not True or value.get('checkpoint_verified') is not True
                or value.get('model_id') != MODEL_ID or value.get('revision') != REVISION
                or value.get('paper_only') is not True or value.get('promote') is not False
                or value.get('execution', {}).get('weight_changed') is not True
                or not value.get('execution', {}).get('device', '').startswith('cuda:')
                or checkpoint_files(directory/'checkpoint', guard) != value.get('checkpoint_files')):
            raise ValueError('완료 부모 checkpoint의 내용 또는 CUDA·추가학습 증거가 유효하지 않습니다')
        completion = None
        if 'completion_sha256' in value or (latest is None and 'completion_sha256' in item):
            if latest is None and item.get('completion_sha256') != value.get('completion_sha256'):
                raise ValueError('연구 포인터와 완료 스냅샷 해시가 다릅니다')
            completion = read_completion(directory,value)
        endpoint = stamp(value['exposure_end'])
        exposure = max(exposure, endpoint) if exposure else endpoint
        if latest is None:
            latest = {'run': run, 'manifest_sha256': item['manifest_sha256'],
                      'checkpoint': directory/'checkpoint', 'manifest': value, 'completion':completion}
        if value.get('parent_run') is None:
            latest['exposure_end'] = exposure.isoformat()
            return latest
        item = {'run':value['parent_run'], 'manifest_sha256':value['parent_manifest_sha256']}
    raise ValueError('계보 검사 상한을 초과했습니다')


class ChronosRuntime:
    def __init__(self, root, config, guard):
        from importlib import metadata
        packages = safe_path(root, '7차/work/python-packages')
        assert_no_links(packages)
        if not packages.is_dir() or any((packages/name).exists() for name in ('torch', 'numpy', 'pandas')):
            raise ValueError('격리 패키지 경로가 없거나 공유 핵심 패키지를 가리고 있습니다')
        sys.path.insert(0, str(packages))
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        import torch
        import numpy as np
        from chronos import Chronos2Pipeline
        from transformers import TrainerCallback
        if not Path(torch.__file__).resolve().is_relative_to('/opt/venvs/torch'):
            raise ValueError('지정된 시스템 torch가 아닙니다')
        versions = {name:metadata.version(name) for name in PACKAGES}
        if versions != PACKAGES:
            raise ValueError(f'검사한 고정 의존성과 다릅니다: {versions}')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA가 없어 연구를 거부합니다. CPU 대체는 없습니다')
        guard.check()
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        self.torch, self.np, self.pipeline_type = torch, np, Chronos2Pipeline
        self.callback_type = TrainerCallback
        self.config, self.guard = config, guard
        self.evidence = {'packages': versions, 'torch':torch.__version__, 'torch_file':torch.__file__,
                         'cuda_runtime':torch.version.cuda, 'gpu_name':torch.cuda.get_device_name(0),
                         'device':'cuda:0', 'weight_changed':False}

    def load(self, path):
        self.guard.check()
        model = self.pipeline_type.from_pretrained(str(path), device_map='cuda:0',
                                                   local_files_only=True, trust_remote_code=False)
        self.check_device(model)
        return model

    def check_device(self, pipeline):
        self.check_model_device(pipeline.model)

    def check_model_device(self, model):
        if model is None:
            raise RuntimeError('실제 학습 모델 장치를 확인할 수 없습니다')
        devices = {str(p.device) for p in model.parameters()}
        if devices != {'cuda:0'}:
            raise RuntimeError(f'실제 모델 장치가 CUDA 0 전용이 아닙니다: {devices}')
        self.guard.check()

    def weights_hash(self, pipeline):
        digest = hashlib.sha256()
        for name, param in sorted(pipeline.model.named_parameters()):
            self.guard.check()
            tensor = param.detach().cpu().contiguous()
            if not self.torch.isfinite(tensor).all().item():
                raise ValueError('모델 가중치에 비유한 값이 있습니다')
            digest.update(name.encode('utf-8'))
            digest.update(tensor.view(self.torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def train(self, pipeline, training, tuning, directory, status):
        runtime, config = self, self.config

        class Progress(self.callback_type):
            def on_train_begin(self, args, state, control, **kwargs):
                # 공식 fit이 복제한 실제 Trainer 모델을 첫 optimizer step 전에 검사합니다.
                runtime.check_model_device(kwargs.get('model'))

            def on_step_end(self, args, state, control, **kwargs):
                runtime.check_model_device(kwargs.get('model'))
                status.update(step=int(state.global_step))

            def on_log(self, args, state, control, logs=None, **kwargs):
                runtime.guard.check()
                loss = (logs or {}).get('loss', (logs or {}).get('train_loss'))
                if loss is not None:
                    if not math.isfinite(float(loss)):
                        raise ValueError('학습 손실이 유한하지 않습니다')
                    status.update(step=int(state.global_step), loss=float(loss))

        return pipeline.fit(inputs=training, validation_inputs=tuning,
                            prediction_length=config['prediction_length'], finetune_mode='full',
                            context_length=config['context_length'], batch_size=config['batch_size'],
                            learning_rate=config['learning_rate'], num_steps=config['steps'],
                            output_dir=str(directory), finetuned_ckpt_name='checkpoint', callbacks=[Progress()],
                            disable_data_parallel=True, save_total_limit=None, save_only_model=True,
                            logging_steps=1, eval_steps=config['steps'], save_steps=config['steps'],
                            report_to='none', seed=config['seed'], data_seed=config['seed'])

    def predict(self, pipeline, contexts):
        self.check_device(pipeline)
        pipeline.model.eval()
        values = pipeline.predict([self.np.asarray(x, dtype=self.np.float32) for x in contexts],
                                  prediction_length=self.config['prediction_length'],
                                  context_length=self.config['context_length'],
                                  batch_size=self.config['batch_size'], cross_learning=False)
        median = min(range(len(pipeline.quantiles)), key=lambda i:abs(pipeline.quantiles[i]-0.5))
        if abs(pipeline.quantiles[median]-0.5) > 1e-8:
            raise ValueError('모델에 중앙값 분위수가 없습니다')
        return [v[0, median, :self.config['prediction_length']].detach().cpu().tolist() for v in values]


def run_once(root, config):
    root = Path(root).resolve()
    config = validate_settings(config)
    work = safe_path(root, '7차/work')
    assert_no_links(work)
    guard = Guard(work/'STOP', config['max_seconds'])
    with exclusive_lock(work/'runner.lock'):
        guard.check()
        recovered = resume_finalization(root,work,guard)
        if recovered is not None:
            return recovered
        run = dt.datetime.now(UTC).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
        directory = safe_path(work, 'runs/'+run)
        directory.mkdir(parents=True, exist_ok=False)
        status = Status(work/'status.json', config, run)
        exposure_receipt = None
        pointer = None
        old_signals = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                old_signals[sig] = signal.signal(sig, lambda *_:guard.event.set())
        try:
            status.update()
            guard.check()
            if shutil.disk_usage(work).free < config['min_free_bytes']:
                raise OSError('디스크 여유가 부족합니다. 자료 삭제나 자동 정리는 하지 않습니다')
            parent = read_parent(work, guard)
            status.update(parent_run=parent['run'] if parent else None)
            seen = read_exposure_end(work,guard)
            if parent:
                seen = max(stamp(seen),stamp(parent['exposure_end'])).isoformat() if seen else parent['exposure_end']
            plan = make_plan(config, dt.datetime.now(UTC), seen)
            protected = protected_periods(root, guard)
            readiness = read_json(safe_path(root, '3차/work/status.json'))
            initial = initial_completion(root, readiness, guard)
            status.update(stage='extract')
            rows, data = extract_rows(root, plan['train'][0], plan['validation'][1],
                                      config['max_rows_per_market'], guard)
            training, tuning, windows = [], [], []
            counts = {}
            for market in MARKETS:
                parts = {name:split_segments(rows[market], a, b, protected) for name, (a,b) in plan.items()}
                minimum = config['context_length']+config['prediction_length']
                market_train = [s for s in parts['train'] if len(s) >= minimum]
                market_tune = [s for s in parts['tune'] if len(s) >= minimum]
                if not market_train or not market_tune:
                    raise NotReady(f'{market}: 공백 없는 학습·튜닝 시계열이 부족합니다')
                training.extend([[v for _,v in s] for s in market_train])
                tuning.extend([[v for _,v in s] for s in market_tune])
                market_windows = evaluation_windows(parts['validation'], config['context_length'],
                                                    config['prediction_length'], config['evaluation_windows_per_market'])
                windows.extend(market_windows)
                counts[market] = {'train_rows':sum(map(len,market_train)),
                                  'tune_rows':sum(map(len,market_tune)), 'validation_windows':len(market_windows)}
            data.update(used_by_market=counts, protected_periods=protected,
                        exposure_end=plan['validation'][1],
                        archive_readiness=initial)
            atomic_json(directory/'data.json', data)
            status.update(rows=sum(v['train_rows'] for v in counts.values()), stage='load')
            source = parent['checkpoint'] if parent else safe_path(work, 'pretrained/'+REVISION)
            if parent is None:
                validate_pretrained(source, guard)
            runtime = ChronosRuntime(root, config, guard)
            pipeline = runtime.load(source)
            before = runtime.weights_hash(pipeline)
            status.update(state='training', stage='finetune', device='cuda:0')
            trained = runtime.train(pipeline, [runtime.np.asarray(x,dtype=runtime.np.float32) for x in training],
                                    [runtime.np.asarray(x,dtype=runtime.np.float32) for x in tuning], directory, status)
            runtime.check_device(trained)
            after = runtime.weights_hash(trained)
            if before == after or status.value['step'] != config['steps'] or status.value['loss'] is None:
                raise ValueError('실제 가중치 변화·완료 step·손실 근거가 없습니다')
            runtime.evidence.update(weight_changed=True, before_weights_sha256=before,
                                    after_weights_sha256=after, completed_steps=status.value['step'])
            contexts = [c for c,_ in windows]
            # 실패하더라도 재로드 probe와 후속 평가 자료를 새 미사용 구간으로 재선택하지 않습니다.
            guard.check()
            exposure_receipt = reserve_exposure(work,run,plan,data,parent)
            status.update(state='running', stage='verify')
            probe = runtime.predict(trained, contexts[:1])
            del trained, pipeline
            runtime.torch.cuda.empty_cache()
            reloaded = runtime.load(directory/'checkpoint')
            if runtime.weights_hash(reloaded) != after:
                raise ValueError('저장 후 재로드한 가중치 해시가 다릅니다')
            repeated = runtime.predict(reloaded, contexts[:1])
            if not runtime.np.allclose(probe, repeated, rtol=1e-5, atol=1e-5, equal_nan=False):
                raise ValueError('저장 전후 예측이 일치하지 않습니다')
            status.update(stage='validation')
            predictions = runtime.predict(reloaded, contexts)
            metrics = forecast_metrics(predictions, [t for _,t in windows], [c[-1] for c in contexts])
            atomic_json(directory/'evaluation.json', {'metrics':metrics, 'predictions':predictions,
                                                       'targets':[t for _,t in windows], 'origins':[c[-1] for c in contexts]})
            guard.check()
            if shutil.disk_usage(work).free < config['min_free_bytes']:
                raise OSError('완료 저장 전 디스크 여유가 부족합니다')
            manifest = make_manifest(run, parent, plan, config, data, metrics, directory/'checkpoint', runtime.evidence)
            manifest['exposure_reservation_sha256'] = exposure_receipt
            completion = {**status.value, 'state':'complete', 'stage':'complete',
                          'finished_utc':now_iso(), 'updated_utc':now_iso(), 'candidate_ready':True,
                          'checkpoint_verified':True, 'has_status_warning':False,
                          **{k:metrics[k] for k in ('validation_mae','baseline_mae','direction_accuracy')}}
            atomic_json(directory/'completion.json',completion)
            manifest['completion_sha256'] = file_hash(directory/'completion.json',guard)
            atomic_json(directory/'manifest.json', manifest)
            pointer = {'run':run, 'manifest_sha256':file_hash(directory/'manifest.json', guard),
                       'completion_sha256':manifest['completion_sha256']}
            # 운영 모델 포인터와 무관한 연구 계보만 원자적으로 바꿉니다.
            atomic_json(work/'knowledge.json', pointer)
            confirmed = committed_completion(work,pointer)
            if confirmed is None:
                raise ValueError('완료 포인터 커밋을 확인할 수 없습니다')
            return finalize_committed(root,work,run,manifest,confirmed)
        except Exception as exc:
            if pointer is not None:
                try:
                    confirmed = committed_completion(work,pointer)
                except Exception as proof_error:
                    record_finalization_error(root,directory,exc,committed=None)
                    raise RuntimeError('완료 커밋 확인 불가; 가중치 재학습 없이 원문 진단이 필요합니다') from proof_error
                if confirmed is not None:
                    return finalize_committed(root,work,run,manifest,confirmed,initial_error=exc)
            state = 'stopped' if isinstance(exc, StopRequested) else 'waiting' if isinstance(exc, NotReady) else 'error'
            atomic_json(directory/'failure.json', {'utc':now_iso(), 'type':type(exc).__name__,
                        'error':str(exc), 'traceback':traceback.format_exc(), 'state':state,
                        'config':config, 'cwd':str(Path.cwd()), 'argv':sys.argv,
                        'stage':status.value['stage'], 'exposure_reservation_sha256':exposure_receipt,
                        **routing_counters(root)})
            status.update(state=state, finished_utc=now_iso(), candidate_ready=False,
                          checkpoint_verified=False, has_error=state=='error',
                          error='연구 실행 실패; 비공개 기록 참조' if state=='error' else None)
            atomic_json(directory/'status.json', status.value)
            raise
        finally:
            for sig, previous in old_signals.items():
                signal.signal(sig, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='7차/foundation-config.json', help='프로젝트 내부 연구 설정 JSON')
    parser.add_argument('--once', action='store_true', required=True, help='유한한 연구 한 회차만 실행')
    args = parser.parse_args(argv)
    try:
        config = validate_settings(read_json(safe_path(ROOT, args.config)))
        result = run_once(ROOT, config)
        print(json.dumps({k:result[k] for k in ('run','parent_run','state','checkpoint_verified','has_status_warning')}, ensure_ascii=False))
        return 0
    except NotReady:
        print('새로운 미사용 자료 준비 대기입니다.', file=sys.stderr)
        return 3
    except StopRequested:
        print('사용자 중단 요청을 보존했습니다.', file=sys.stderr)
        return 130
    except AlreadyRunning:
        print('기존 연구 실행을 유지합니다. 중복 실행은 거부합니다.', file=sys.stderr)
        return 2
    except Exception:
        print('연구 실행 실패: 비공개 회차 기록을 확인하십시오.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
