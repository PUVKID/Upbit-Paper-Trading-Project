"""기존2026의 공용 패키지를 바꾸지 않고 프로젝트 학습 환경을 준비합니다."""
import datetime as dt
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DOCKER = 'C:/Program Files/Docker/Docker/resources/bin/docker.exe'
linux = '/workspace/2026/' + ROOT.name
report = {'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'commands': [],
          'host_sqlite': False, 'shared_environment_modified': False}

def run(args, timeout=60):
    argv = [DOCKER, *args]
    result = subprocess.run(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            shell=False, timeout=timeout, check=False)
    raw = result.stdout.decode('utf-8', errors='strict')
    report['commands'].append({'argv': argv, 'exit_code': result.returncode,
                               'output': raw[-4096:], 'output_bytes': len(result.stdout)})
    if result.returncode:
        raise RuntimeError(raw[-2000:])
    return raw

try:
    container = json.loads(run(['inspect', '2026']))[0]
    if (not container['State']['Running'] or
            container['Config']['Labels'].get('io.yunseok.managed') != 'docker2026-v1' or
            not any(m['Destination'] == '/workspace/2026' and m['RW'] and
                    m['Source'].replace('\\','/').rstrip('/').lower() == str(ROOT.parent).replace('\\','/').lower()
                    for m in container['Mounts'])):
        raise RuntimeError('기존2026의 소유 표식·프로젝트 마운트를 확인하지 못했습니다')
    report['container_id'] = container['Id']
    run(['exec', '2026', 'python', '-c',
         'import venv; venv.EnvBuilder(system_site_packages=True,with_pip=True).create(' + repr(linux + '/.venv') + ')'], 120)
    run(['exec', '--workdir', linux, '2026', linux + '/.venv/bin/python', '-m', 'pip', 'install', '-r', 'requirements.txt'], 900)
    report['packages'] = json.loads(run(['exec', '2026', linux + '/.venv/bin/python', '-c',
         'import numpy,pandas,xgboost,json; print(json.dumps({"numpy":numpy.__version__,"pandas":pandas.__version__,"xgboost":xgboost.__version__}))']))
    report['passed'] = True
finally:
    report['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    (ROOT / 'evidence/prepare-2026.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
