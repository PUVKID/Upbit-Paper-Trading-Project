"""소유 컨테이너 안에서 도지코인의 초기 공개 캔들만 수집합니다."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import docker_control as control

report = {'started_kst': dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(),
          'purpose': '사용자 요청 도지코인 초기 데이터 축적', 'host_sqlite': False,
          'paper_orders': False, 'other_workers_stopped': False}
script = '''import engine,json,time
config=engine.load_config('engine-config.json')
# 초기 수집만 도지코인으로 제한하며 모델·매매 검증을 우회하지 않습니다.
config['markets']=['KRW-DOGE']
print(json.dumps({'stage':'start','markets':config['markets'],'timeframes':config['timeframes']},ensure_ascii=False),flush=True)
rows=engine.collect_cycle(config,backfill=True,pages=100)
selected=[r for r in rows if r['market']=='KRW-DOGE']
print(json.dumps({'stage':'complete','datasets':selected,'rows':sum(r['rows'] for r in selected)},ensure_ascii=False),flush=True)
'''
try:
    container = control.owned_container()
    if not container or not container['State']['Running']:
        raise RuntimeError('소유한 실행 컨테이너가 필요합니다')
    report['container_id'] = container['Id']
    argv = [control.docker_path(), 'exec', '-i', control.NAME, 'python', '-']
    report['argv'], report['cwd'] = argv, str(ROOT)
    with (ROOT / 'evidence/doge-bootstrap.log').open('wb') as stream:
        result = subprocess.run(argv, input=script.encode('utf-8'), stdout=stream,
                                stderr=subprocess.STDOUT, cwd=ROOT, timeout=1200,
                                shell=False, check=False)
    report['exit_code'] = result.returncode
    report['passed'] = result.returncode == 0
    if result.returncode:
        raise RuntimeError('도지코인 초기 수집 실패. 원문: evidence/doge-bootstrap.log')
finally:
    report['finished_kst'] = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat()
    (ROOT / 'evidence/doge-bootstrap.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
