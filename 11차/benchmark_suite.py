"""사전에 고정한 열 기간의 CUDA 예측 비교만 실행하고 자동 학습 재개 없이 종료합니다."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('suite_benchmark', ROOT/'11차/pretrained_benchmark.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
f = b.f
ANCHORS = tuple(f'{date}T08:00:00Z' for date in (
    '2024-11-15','2024-11-22','2024-11-29','2024-12-06','2024-12-13',
    '2024-12-20','2024-12-27','2025-01-03','2025-01-10','2025-01-17'))


def settings(value):
    if value != {'schema_version':1,'experiments':10,'max_seconds':21600,'anchors':list(ANCHORS)}:
        raise ValueError('사전에 고정한10회·기간·총6시간 계약과 다릅니다')
    return value


def verified_run(root,run,expected):
    directory = f.safe_path(root,'11차/work/runs/'+run)
    manifest = f.read_json(directory/'manifest.json')
    request = f.read_json(directory/'request.json')
    report = f.read_json(directory/'report.json')
    required = {'request.json','data.json','predictions.json','policy.json','report.json',
        'reload-probe-before.json','reload-proof.json','incumbent/manifest.json'}
    inventory = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()
        and p != directory/'manifest.json'}
    artifacts = manifest.get('artifacts')
    if (not isinstance(artifacts,dict) or not required.issubset(artifacts)
            or set(artifacts) != inventory):
        raise ValueError('필수 산출물과 전체 inventory가 일치하지 않습니다')
    if (manifest.get('run') != run or manifest.get('complete') is not True
            or manifest.get('evaluation_performed') is not True
            or manifest.get('checkpoint_verified') is not True
            or manifest.get('model_id') != b.MODEL_ID or manifest.get('revision') != b.REVISION
            or manifest.get('fit_calls') != 0 or manifest.get('promotion') is not False
            or manifest.get('candidate_ready') is not False or request.get('settings') != expected):
        raise ValueError('완료 실험의 고정 설정·평가·미적용 증거가 다릅니다')
    for name,digest in artifacts.items():
        if f.file_hash(f.safe_path(directory,name)) != digest:
            raise ValueError('완료 실험 산출물 해시 불일치: '+name)
    reload = f.read_json(directory/'reload-proof.json')
    model = request.get('model',{})
    if (request.get('source_sha256') != f.file_hash(Path(root)/'11차/pretrained_benchmark.py')
            or model.get('model_id') != b.MODEL_ID or model.get('revision') != b.REVISION
            or model.get('files',{}).get('model.safetensors') != b.WEIGHT_SHA256
            or model.get('license') != 'apache-2.0'
            or reload.get('reload_prediction_match') is not True
            or reload.get('weight_file_unchanged') is not True
            or report.get('fit_calls') != 0 or report.get('promotion') is not False
            or report.get('parameter_count') != 231289280):
        raise ValueError('고정 소스·가중치·재로드 증거가 다릅니다')
    counts = manifest['counters']
    if (counts.get('gpu_verified') is not True or report.get('device') != 'cuda:0'
            or counts['windows_completed'] != counts['attempted_windows']
            or not 2 < counts['windows_completed'] <= counts['padded_windows'] <= 6000
            or counts['inference_calls'] < 3):
        raise ValueError('실제 CUDA 실행 또는 예측 예산 검증 실패')
    comparisons = report.get('comparisons',[])
    pairs = {(v['summary']['candidate'],v['summary']['cost_multiplier']) for v in comparisons}
    if pairs != {(name,cost) for name in ('foundation','cash','trend','incumbent') for cost in (1,2)} or len(comparisons) != 8:
        raise ValueError('같은 비용의 네 기준 비교가 모두 필요합니다')
    metrics = report.get('forecast_metrics',{})
    if (not math.isfinite(report['seconds']) or report['seconds'] <= 0
            or not 0 <= metrics.get('direction_accuracy',-1) <= 1 or metrics.get('samples',0) <= 0
            or any(not math.isfinite(metrics.get(k,float('nan'))) or metrics[k] < 0
                   for k in ('relative_mae','naive_relative_mae'))):
        raise ValueError('유효한 예측 오차 집계가 필요합니다')
    return {'run':run,'anchor':expected['anchor'],'manifest_sha256':f.file_hash(directory/'manifest.json'),
        'gpu_verified':True,'fit_calls':0,'windows':counts['windows_completed'],
        'padded_windows':counts['padded_windows'],'inference_calls':counts['inference_calls'],
        'seconds':report['seconds'],'policy':report['policy'],'forecast_metrics':report['forecast_metrics'],
        'comparisons':[v['summary'] for v in report['comparisons']]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',required=True)
    args = parser.parse_args(argv)
    if sys.platform != 'linux':
        raise RuntimeError('실제 실행은 Linux 컨테이너 전용입니다')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,50}',args.suite):
        raise ValueError('안전한 명시적 suite 식별자가 필요합니다')
    config = settings(f.read_json(ROOT/'11차/suite-config.json'))
    base = b.settings(f.read_json(ROOT/'11차/benchmark-config.json'))
    work = f.safe_path(ROOT,'11차/work')
    directory = f.safe_path(work,'suites/'+args.suite)
    guard = b.Guard(ROOT,config['max_seconds'])
    active = None
    old = {}
    for sig in (signal.SIGINT,signal.SIGTERM):
        old[sig] = signal.signal(sig,lambda *_:guard.event.set())
    try:
        with f.exclusive_lock(work/'suite.lock'):
            guard.check()
            request = {'suite':args.suite,'config':config,'base':base,
                'runner_sha256':f.file_hash(ROOT/'11차/pretrained_benchmark.py'),
                'suite_sha256':f.file_hash(Path(__file__)),'fit_calls':0,'promotion':False}
            if directory.exists():
                raise ValueError('기존 suite 예약은 재개·자동 재시도할 수 없습니다')
            directory.mkdir(parents=True)
            request['started_utc'] = f.now_iso()
            f.atomic_json(directory/'request.json',request)
            started = time.monotonic()
            completed = []
            status = {'schema_version':1,'suite':args.suite,'state':'running','total':10,
                'completed':0,'fit_calls':0,'promotion':False,'started_utc':f.now_iso()}
            def save(**fields):
                status.update(fields,updated_utc=f.now_iso())
                f.atomic_json(directory/'status.json',status)
                f.atomic_json(work/'suite-status.json',status)
            save()
            try:
                for index,anchor in enumerate(config['anchors'],1):
                    guard.check()
                    if f.file_hash(ROOT/'11차/pretrained_benchmark.py') != request['runner_sha256']:
                        raise ValueError('실행 중 runner 소스 변경 감지')
                    run = f'{args.suite}-{index:02d}'
                    expected = b.settings({**base,'anchor':anchor})
                    run_dir = f.safe_path(work,'runs/'+run)
                    save(current=index,current_run=run,anchor=anchor)
                    if run_dir.exists():
                        raise ValueError('기존 회차 예약을 자동 재실행하지 않습니다')
                    else:
                        config_path = directory/f'config-{index:02d}.json'
                        f.atomic_json(config_path,expected)
                        # 자식의 초기 import 실패도 재시도할 수 없도록 부모가 먼저 영구 예약합니다.
                        f.atomic_json(directory/f'launch-{index:02d}.json',{'run':run,'started_utc':f.now_iso()})
                        command = [sys.executable,'-B','-X','utf8','11차/pretrained_benchmark.py',
                            '--once','--run',run,'--config',config_path.relative_to(ROOT).as_posix()]
                        with (directory/f'run-{index:02d}.log').open('wb') as log:
                            active = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                start_new_session=True)
                            limit = time.monotonic()+min(base['max_seconds'],
                                config['max_seconds']-(time.monotonic()-started))
                            while active.poll() is None:
                                guard.check()
                                if time.monotonic() >= limit:
                                    raise TimeoutError('회차 또는 전체 실행 예산 초과')
                                time.sleep(1)
                            code = active.returncode
                            active = None
                            if code != 0:
                                raise RuntimeError(f'{run} 종료 코드{code}; 원문 보존, 자동 재시도 금지')
                        result = verified_run(ROOT,run,expected)
                    completed.append(result)
                    f.atomic_json(directory/'results.json',completed)
                    save(completed=len(completed))
                    print(json.dumps({'completed':len(completed),'total':10,'run':run,
                        'windows':result['windows'],'seconds':result['seconds']},ensure_ascii=False),flush=True)
                guard.check()
                aggregate = {'schema_version':1,'model_id':b.MODEL_ID,'revision':b.REVISION,
                    'study':'zero_shot_retrospective','experiments_completed':10,'fit_calls':0,
                    'gpu_verified':True,'promotion':False,'candidate_ready':False,
                    'scope':'retrospective_not_independent_final_test',
                    'pretraining_overlap_unknown':True,'historical_quote_extra_cost':None,
                    'after_suite':'paused_until_explicit_user_request','results':completed,
                    'windows':sum(v['windows'] for v in completed),
                    'seconds':time.monotonic()-started,'finished_utc':f.now_iso()}
                f.atomic_json(directory/'aggregate.json',aggregate)
                f.atomic_json(ROOT/'11차/실험결과.json',aggregate)
                save(state='complete',finished_utc=f.now_iso(),automatic_repetition=False)
                return 0
            except BaseException as exc:
                f.atomic_json(directory/'failure.json',{'utc':f.now_iso(),'error':str(exc),
                    'type':type(exc).__name__,'argv':sys.argv,'cwd':str(ROOT),
                    'completed':len(completed),**f.routing_counters(ROOT)})
                save(state='stopped' if guard.event.is_set() or isinstance(exc,(f.StopRequested,KeyboardInterrupt)) else 'error')
                raise
    finally:
        if active is not None and active.poll() is None:
            os.killpg(active.pid,signal.SIGTERM)
            try:
                active.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid,signal.SIGKILL)
                active.wait(timeout=10)
        for sig,previous in old.items():
            signal.signal(sig,previous)


if __name__ == '__main__':
    raise SystemExit(main())
