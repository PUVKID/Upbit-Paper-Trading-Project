"""학습·매매를 유지한 채 소유한 웹 작업자만 새 UI로 갱신합니다."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import docker_control as control

OUT = ROOT / 'evidence/디자인/deploy.json'
report = {'started_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'commands': []}


def run(args, stdin=None):
    argv = [control.docker_path(), *args]
    result = subprocess.run(argv, input=stdin, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=60, check=False)
    text = result.stdout.decode('utf-8', errors='strict')
    report['commands'].append({'argv': argv, 'exit_code': result.returncode,
                               'output': text[:4096], 'output_bytes': len(result.stdout)})
    if result.returncode:
        raise RuntimeError(f'웹 갱신 명령 실패: {text[:1000]}')
    return text


def financial_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
            ('engine.py', 'engine-config.json', 'service.json', 'supervisor.py', 'healthcheck.py')}


try:
    before = control.owned_container()
    if not before or not before['State']['Running']:
        raise RuntimeError('소유한 실행 컨테이너가 필요합니다')
    report['before'] = {'container': before['Id'], 'started_at': before['State']['StartedAt'],
                        'financial_hashes': financial_hashes()}
    run(['cp', str(ROOT / 'dashboard.py'), control.NAME + ':/app/dashboard.py'])
    # 서버 재시작 동안에 정적 파일을 교체하되 다른 작업자의 프로세스는 건드리지 않습니다.
    script = '''import json,os,signal
from pathlib import Path
targets=[]
for path in Path('/proc').iterdir():
    if not path.name.isdigit(): continue
    try: args=(path/'cmdline').read_bytes().split(b'\\0')
    except FileNotFoundError: continue
    if b'dashboard.py' in args and b'8765' in args:
        targets.append(int(path.name))
if len(targets)>1: raise RuntimeError('웹 작업자가 여러 개이므로 중지하지 않습니다')
print(json.dumps({'dashboard_pids':targets}),flush=True)
for pid in targets: os.kill(pid,signal.SIGTERM)
'''
    run(['exec', '-i', control.NAME, 'python', '-'], script.encode('utf-8'))
    run(['cp', str(ROOT / 'web') + '/.', control.NAME + ':/app/web/'])
    deadline = time.monotonic() + 120
    while True:
        try:
            with urllib.request.urlopen('http://127.0.0.1:8765/api/status', timeout=5) as response:
                status = json.load(response)
            with urllib.request.urlopen('http://127.0.0.1:8765/scene.js', timeout=5) as response:
                scene_bytes = response.read()
            if hashlib.sha256(scene_bytes).digest() != hashlib.sha256((ROOT / 'web/scene.js').read_bytes()).digest():
                raise RuntimeError('실행 장면 파일과 동결 소스가 다릅니다')
            if status['policy']['initial_capital'] != 10_000_000 or len(status.get('quotes', [])) != 4:
                raise RuntimeError('실제 모의계좌·공개 시세 연결을 확인하지 못했습니다')
            break
        except (OSError, ValueError, KeyError, RuntimeError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)
    after = control.owned_container()
    report['after'] = {'container': after['Id'], 'started_at': after['State']['StartedAt'],
                       'financial_hashes': financial_hashes(),
                       'training': status['services']['trainer']['state'],
                       'training_cycles': status['services']['trainer']['successful_cycles'],
                       'equity': status['account']['equity'], 'quotes': len(status['quotes'])}
    if (report['before']['container'] != report['after']['container'] or
        report['before']['started_at'] != report['after']['started_at'] or
        report['before']['financial_hashes'] != report['after']['financial_hashes']):
        raise RuntimeError('컨테이너 또는 금융 파일의 연속성 경계가 바뀌었습니다')
    report['passed'] = True
except Exception as error:
    report['passed'] = False
    report['error'] = repr(error)
    raise
finally:
    report['finished_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'passed': report.get('passed'), 'report': str(OUT)}, ensure_ascii=False))
