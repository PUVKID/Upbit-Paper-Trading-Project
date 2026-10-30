"""수집·학습·현물 모의거래를 독립 실행하고 상태를 보존합니다."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
STOP = threading.Event()


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def gpu_snapshot() -> dict:
    result = {"checked_utc": utcnow().isoformat(), "state": "모니터링 미지원"}
    library = None
    for path in ("libnvidia-ml.so.1", "/usr/lib/wsl/lib/libnvidia-ml.so.1"):
        try:
            library = ctypes.CDLL(path)
            break
        except OSError:
            continue
    if library is None or library.nvmlInit_v2() != 0:
        return result
    try:
        device = ctypes.c_void_p()
        if library.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(device)) != 0:
            return result
        name = ctypes.create_string_buffer(128)
        if library.nvmlDeviceGetName(device, name, len(name)) == 0:
            result["gpu_name"] = name.value.decode("utf-8")

        class Memory(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

        class Utilization(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

        memory = Memory()
        utilization = Utilization()
        if library.nvmlDeviceGetMemoryInfo(device, ctypes.byref(memory)) == 0:
            result.update(gpu_memory_total_mb=round(memory.total / 1048576), gpu_memory_used_mb=round(memory.used / 1048576))
        if library.nvmlDeviceGetUtilizationRates(device, ctypes.byref(utilization)) == 0:
            result["gpu_utilization_pct"] = utilization.gpu
        result["state"] = "실측"
        return result
    finally:
        library.nvmlShutdown()


def logger_for(name: str, work: Path) -> logging.Logger:
    logger = logging.getLogger(f"업비트.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = RotatingFileHandler(work / "service" / f"{name}.log", maxBytes=8_000_000, backupCount=4, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def worker(name: str, settings: dict, work: Path, start_at: dt.datetime) -> None:
    logger = logger_for(name, work)
    status_path = work / "service" / f"{name}.json"
    failures = 0
    iteration = 0
    successful_cycles = 0
    previous_finished = None
    observation = {}
    while not STOP.is_set():
        if name == "paper" and utcnow() < start_at:
            atomic_json(status_path, {"state": "예정시각 대기", "start_utc": start_at.isoformat(), "checked_utc": utcnow().isoformat()})
            STOP.wait(min(30, max(0.1, (start_at - utcnow()).total_seconds())))
            continue
        iteration += 1
        command = [sys.executable, "-u", settings.get("script", "engine.py"), *settings["args"]]
        started = utcnow()
        observation = {"previous_finished_utc": previous_finished.isoformat() if previous_finished else None,
                       "success_gap_seconds": (started - previous_finished).total_seconds()
                       if previous_finished is not None and failures == 0 else None,
                       "successful_cycles": successful_cycles, "next_iteration": iteration + 1,
                       "interval_seconds": settings["interval_seconds"]}
        atomic_json(status_path, {"state": "실행 중", "started_utc": started.isoformat(), "iteration": iteration,
                                  "consecutive_failures": failures, "argv": command, **observation})
        logger.info("실행 시작 %s", json.dumps(command, ensure_ascii=False))
        process = None
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="strict", bufsize=1, start_new_session=os.name != "nt")
            limit = settings.get("timeout_seconds", 7200)
            deadline = time.monotonic() + limit if limit is not None else float("inf")
            # 출력 배출은 별도 스레드가 맡아 중단과 시간 제한 감시를 막지 않습니다.
            def drain() -> None:
                try:
                    for line in process.stdout:
                        logger.info("%s", line.rstrip("\n"))
                except (OSError, UnicodeError) as error:
                    logger.error("자식 출력 수집 오류: %s", error)

            reader = threading.Thread(target=drain, daemon=True)
            reader.start()
            while process.poll() is None:
                if STOP.wait(0.5) or time.monotonic() >= deadline:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    if not STOP.is_set():
                        raise TimeoutError("작업 시간 제한 초과")
            reader.join(timeout=5)
            code = process.returncode
            if code:
                raise RuntimeError(f"자식 프로세스 종료 코드 {code}")
            failures = 0
            finished = utcnow()
            successful_cycles += 1
            observation.update(successful_cycles=successful_cycles,
                               duration_seconds=(finished - started).total_seconds())
            atomic_json(status_path, {"state": "다음 회차 연결" if settings["interval_seconds"] == 0 else "완료 후 다음 회차 대기",
                                      "started_utc": started.isoformat(), "finished_utc": finished.isoformat(),
                                      "iteration": iteration, "exit_code": code, **observation})
            previous_finished = finished
            logger.info("회차 완료 %s", iteration)
            delay = settings["interval_seconds"]
        except (OSError, RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
            failures += 1
            atomic_json(status_path, {"state": "오류 후 재시도 대기", "started_utc": started.isoformat(),
                                      "failed_utc": utcnow().isoformat(), "iteration": iteration,
                                      "consecutive_failures": failures, "error": str(error), **observation})
            logger.exception("회차 실패, 원인 기록 후 재시도")
            delay = min(900, 30 * 2 ** min(failures - 1, 5))
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()
        STOP.wait(delay)
    atomic_json(status_path, {"state": "중지됨", "stopped_utc": utcnow().isoformat(), "iteration": iteration,
                              **observation})


def validate_settings(config: dict, engine_config: dict) -> dt.datetime:
    paper = engine_config["paper"]
    if (config.get("initial_capital") != 10_000_000 or paper.get("initial_cash") != config.get("initial_capital")
            or config.get("market_mode") != "spot-paper" or config.get("notifications") is not False):
        raise ValueError("원금 1,000만원·현물 모의거래·알림 없음 설정을 확인하십시오.")
    start_at = dt.datetime.fromisoformat(config["paper_start"])
    engine_start = dt.datetime.fromisoformat(paper["start_time"])
    if start_at.tzinfo is None or engine_start.tzinfo is None or start_at != engine_start:
        raise ValueError("서비스와 엔진의 시작 시각·시간대가 일치해야 합니다.")
    if config["trainer"]["interval_seconds"] != 0:
        raise ValueError("성공한 학습 회차의 고정 대기는 0초여야 합니다.")
    return start_at


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "service.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    engine_config = json.loads((ROOT / "engine-config.json").read_text(encoding="utf-8"))
    start_at = validate_settings(config, engine_config)
    work = ROOT / "work"
    (work / "service").mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    threads = [threading.Thread(target=worker, args=(name, config[name], work, start_at), name=name)
               for name in ("collector", "backfill", "trainer", "paper", "dashboard", "health")]
    for thread in threads:
        thread.start()
    while not STOP.wait(5):
        atomic_json(work / "service" / "gpu.json", gpu_snapshot())
        if any(not thread.is_alive() for thread in threads):
            STOP.set()
            raise RuntimeError("작업 감시 스레드가 종료되어 서비스 재시작이 필요합니다.")
    for thread in threads:
        thread.join(timeout=35)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
