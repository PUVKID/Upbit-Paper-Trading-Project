"""실제 운용 파일과 시세·현물 계좌의 불변 조건을 주기적으로 검사합니다."""

from __future__ import annotations

import ast
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time

import engine

ROOT = Path(__file__).resolve().parent
INTEGRITY_TIMEOUT = 1800
INTEGRITY_MAX_AGE = 21600
INTEGRITY_RETRY = 900


def database_identity(database):
    stat = database.stat()
    return {'path': str(database.resolve()), 'device': stat.st_dev, 'inode': stat.st_ino}


def latest_rows(db, markets, timeframes):
    rows = []
    for market in markets:
        for timeframe in timeframes:
            rows.extend(db.execute(
                'SELECT market,tf,NULL,close_time FROM candles INDEXED BY candle_close '
                'WHERE market=? AND tf=? ORDER BY close_time DESC LIMIT 1',
                (market, str(timeframe))).fetchall())
    return rows


def full_integrity(database, deadline):
    def remaining(*_):
        if time.monotonic() >= deadline:
            raise TimeoutError('전체 무결성 검사 시간 제한 초과')

    remaining()
    identity = database_identity(database)
    observed = engine.iso(None)
    # WAL을 포함한 일관성 백업만 사용하고 느린 공유 디스크의 무작위 읽기를 피합니다.
    with tempfile.TemporaryDirectory(prefix='upbit-integrity-', dir='/tmp' if sys.platform == 'linux' else None) as directory:
        with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True, timeout=1)) as source, \
             closing(sqlite3.connect(str(Path(directory) / 'snapshot.sqlite3'))) as snapshot:
            # 수집 쓰기가 이어져도 백업이 처음부터 재시작되지 않도록 읽기 시점을 고정합니다.
            source.execute('BEGIN')
            source.execute('SELECT name FROM sqlite_schema LIMIT 1').fetchone()
            source.backup(snapshot, pages=1024, progress=remaining, sleep=0.05)
            snapshot.set_progress_handler(lambda: int(time.monotonic() >= deadline), 10000)
            remaining()
            checks = snapshot.execute('PRAGMA integrity_check').fetchall()
            if checks != [('ok',)]:
                raise sqlite3.DatabaseError('전체 무결성 검사 실패: ' + str(checks))
            rows = snapshot.execute('SELECT market,tf,COUNT(*) FROM candles GROUP BY market,tf').fetchall()
            remaining()
    if database_identity(database) != identity:
        raise RuntimeError('검사 중 저장소 파일이 교체되었습니다')
    return {'state': 'ok', 'check': 'PRAGMA integrity_check', 'database': identity,
            'checked_utc': observed, 'completed_utc': engine.iso(None),
            'counts': {f'{market}/{tf}': count for market, tf, count in rows}}


def scan_command(database, timeout):
    return [sys.executable, '-B', '-X', 'utf8', str(Path(__file__).resolve()),
            '--integrity-scan', str(database), str(timeout)]


def integrity_worker(database, service, timeout=INTEGRITY_TIMEOUT, lock_fd=None):
    process = None
    started = engine.iso(None)
    try:
        process = subprocess.Popen(scan_command(database, timeout), stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, encoding='utf-8',
                                   pass_fds=() if lock_fd is None else (lock_fd,))
        output, error = process.communicate(timeout=timeout)
        if process.returncode:
            raise RuntimeError(f'전체 검사 종료 {process.returncode}: {error[-4000:]}')
        result = json.loads(output)
        if result['state'] != 'ok' or result['database'] != database_identity(database):
            raise ValueError('전체 검사 결과와 저장소가 일치하지 않습니다')
        engine.atomic_json(service / 'integrity_result.json', result)
        progress = {'state': 'completed', 'started_utc': started, 'finished_utc': engine.iso(None)}
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
        if process is not None and process.poll() is None:
            process.kill()
            try:
                process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                # 커널 I/O 대기라면 상속된 잠금이 남아 중복 검사를 계속 차단합니다.
                pass
        progress = {'state': 'failed', 'started_utc': started, 'finished_utc': engine.iso(None),
                    'error': f'{type(error).__name__}: {error}'}
    engine.atomic_json(service / 'integrity_progress.json', progress)


def start_integrity(database, service):
    import fcntl

    service.mkdir(parents=True, exist_ok=True)
    with (service / 'integrity.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        previous_path = service / 'integrity_progress.json'
        previous = engine.read_json(previous_path) if previous_path.is_file() else {}
        last_failure = previous if previous.get('state') == 'failed' else previous.get('last_failure')
        engine.atomic_json(previous_path,
                           {'state': 'running', 'started_utc': engine.iso(None),
                            'timeout_seconds': INTEGRITY_TIMEOUT, 'last_failure': last_failure})
        try:
            subprocess.Popen([sys.executable, '-B', '-X', 'utf8', str(Path(__file__).resolve()),
                              '--integrity-worker', str(database), str(service), str(lock.fileno())],
                             pass_fds=(lock.fileno(),), start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as error:
            engine.atomic_json(service / 'integrity_progress.json',
                               {'state': 'failed', 'finished_utc': engine.iso(None),
                                'error': f'{type(error).__name__}: {error}'})
            raise
    return True


def integrity_status(database, service, launch=True, now=None):
    def read(name):
        path = service / name
        return engine.read_json(path) if path.is_file() else None

    result, progress = read('integrity_result.json'), read('integrity_progress.json')
    now = engine.utc() if now is None else now
    if result and result.get('database') != database_identity(database):
        result = None
    fresh = (result is not None and result.get('state') == 'ok'
             and 0 <= (now - engine.utc(result['checked_utc'])).total_seconds() <= INTEGRITY_MAX_AGE)
    state = 'ok' if fresh else 'pending'
    if progress and progress.get('state') == 'failed':
        state = 'failed'
    if progress and progress.get('state') == 'running':
        state = 'failed' if progress.get('last_failure') else 'pending'
        if (now - engine.utc(progress['started_utc'])).total_seconds() > INTEGRITY_TIMEOUT + 30:
            state = 'failed'
            progress = {**progress, 'error': '전체 검사 완료 기록이 제한 시간 안에 도착하지 않았습니다'}
    retry_due = not progress or (now - engine.utc(progress.get('finished_utc', progress.get('started_utc')))).total_seconds() >= INTEGRITY_RETRY
    if launch and (not fresh or state == 'failed') and retry_due:
        start_integrity(database, service)
        progress = read('integrity_progress.json')
    return {'state': state, 'last_result': result, 'progress': progress,
            'max_age_seconds': INTEGRITY_MAX_AGE,
            'age_seconds': (now - engine.utc(result['checked_utc'])).total_seconds() if result else None}


def check_account(state: dict, start, now, initial_cash=10_000_000) -> list[str]:
    errors = []
    if not isinstance(state, dict):
        return ["계좌 자료가 객체 형식이 아닙니다"]
    if state.get("initial_cash") != initial_cash or state.get("paper_only") is not True:
        errors.append("모의 원금 또는 모의 전용 표시가 설정과 다릅니다")
    try:
        if not math.isfinite(float(state["cash"])) or float(state["cash"]) < 0:
            errors.append("현금 잔고가 음수입니다")
        for market, position in state.get("positions", {}).items():
            numbers = [float(position["qty"]), float(position["cost"])]
            if market not in engine.MARKETS or any(not math.isfinite(value) or value < 0 for value in numbers):
                errors.append("현물 보유 목록 또는 잔고가 유효하지 않습니다")
        for trade in state.get("trades", []):
            if engine.utc(trade["timestamp"]) < engine.utc(start):
                errors.append("예약 시작 전 모의체결이 발견되었습니다")
                break
            if engine.utc(trade["timestamp"]) > engine.utc(now) + engine.pd.Timedelta(seconds=5):
                errors.append("미래 시각으로 기록된 모의체결이 있습니다")
                break
    except (AttributeError, KeyError, ValueError, TypeError):
        errors.append("계좌 또는 체결 자료 형식이 유효하지 않습니다")
    return errors


def inspect(root=ROOT) -> dict:
    now = engine.utc()
    errors, pending, counts = [], [], {}
    integrity = None
    rows = []
    config = engine.load_config(root / "engine-config.json")
    for relative in ("engine.py", "supervisor.py", "dashboard.py", "healthcheck.py"):
        ast.parse((root / relative).read_text(encoding="utf-8"), filename=relative)
    database = Path(config["data_dir"]) / "candles.sqlite3"
    if database.is_file():
        integrity = integrity_status(database, root / 'work/service', launch=root.resolve() == ROOT, now=now)
        if integrity['state'] == 'failed':
            errors.append('캔들 저장소 전체 무결성 검사 실패 또는 시간 제한 초과')
        elif integrity['state'] != 'ok':
            pending.append('전체 무결성 검사 미완료 또는 완료 근거가 오래됨')
        if integrity['last_result']:
            counts = integrity['last_result'].get('counts', {})
        with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)) as db:
            deadline = time.monotonic() + 15
            db.set_progress_handler(lambda: int(time.monotonic() >= deadline), 10000)
            rows = latest_rows(db, engine.MARKETS, config['timeframes'])
        # 검사 도중 새로 완료된 봉은 실제 조회 시각을 기준으로 판정합니다.
        now = engine.utc()
        for market, timeframe, count, latest in rows:
            if engine.utc(latest) > now:
                errors.append(f"미완성 미래 봉이 저장되었습니다: {market}/{timeframe}")
            if str(timeframe) == str(config["base_timeframe"]) and (now - engine.utc(latest)).total_seconds() > 900:
                errors.append(f"기본봉이 15분 이상 갱신되지 않았습니다: {market}")
        if len(rows) < len(engine.MARKETS) * len(config["timeframes"]):
            pending.append("일부 시간봉 초기 수집 중")
    else:
        pending.append("초기 시세 수집 대기")
    pointer = Path(config["models_dir"]) / "current.json"
    if pointer.is_file():
        _, manifest = engine.load_current(config)
        if set(manifest["packages"]) != set(engine.MARKETS) or not all(package["gpu"] for package in manifest["packages"].values()):
            errors.append("다섯 코인의 완성 모델 근거가 모두 필요합니다")
        elif not all(evidence.get("actual_device", "").startswith("cuda")
                   for package in manifest["packages"].values() for evidence in package["gpu"]):
            errors.append("운용 모델의 실제 GPU 학습 근거가 없습니다")
    else:
        pending.append("첫 완성 모델 대기")
    account = root / "work/paper/state.json"
    if account.is_file():
        errors.extend(check_account(engine.read_json(account), config["paper"]["start_time"], now,
                                    config["paper"]["initial_cash"]))
    elif now >= engine.utc(config["paper"]["start_time"]) + engine.pd.Timedelta(minutes=5):
        errors.append("예약 시각 이후 실시간 모의계좌가 생성되지 않았습니다")
    else:
        pending.append("예약 시각 전 모의계좌 대기")
    return {"state": "검사 실패" if errors else "초기 준비 중" if pending else "검사 통과",
            "checked_utc": engine.iso(now), "errors": errors, "pending": pending,
            "checked_datasets": len(rows),
            "total_rows": sum(counts.values()) if integrity and integrity['last_result'] else None,
            "counts_source": 'integrity_snapshot' if integrity and integrity['last_result'] else None,
            "counts_observed_utc": integrity['last_result']['checked_utc']
            if integrity and integrity['last_result'] else None,
            "integrity": integrity, "errors_count": len(errors)}


def main() -> int:
    try:
        result = inspect()
    except (AttributeError, OSError, ValueError, KeyError, TypeError, SyntaxError, sqlite3.Error) as error:
        result = {"state": "검사 실패", "checked_utc": engine.iso(None),
                  "errors": [f"{type(error).__name__}: {error}"], "errors_count": 1}
    engine.atomic_json(ROOT / "work/service/health_result.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 1 if result["errors_count"] else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--integrity-worker':
        integrity_worker(Path(sys.argv[2]), Path(sys.argv[3]), lock_fd=int(sys.argv[4]))
        raise SystemExit(0)
    if len(sys.argv) > 1 and sys.argv[1] == '--integrity-scan':
        print(json.dumps(full_integrity(Path(sys.argv[2]), time.monotonic() + float(sys.argv[3]))))
        raise SystemExit(0)
    raise SystemExit(main())
