"""실제 운용 파일과 시세·현물 계좌의 불변 조건을 주기적으로 검사합니다."""

from __future__ import annotations

import ast
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3

import engine

ROOT = Path(__file__).resolve().parent


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
    config = engine.load_config(root / "engine-config.json")
    for relative in ("engine.py", "supervisor.py", "dashboard.py", "healthcheck.py"):
        ast.parse((root / relative).read_text(encoding="utf-8"), filename=relative)
    database = Path(config["data_dir"]) / "candles.sqlite3"
    if database.is_file():
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=20)) as db:
            integrity = db.execute("PRAGMA quick_check").fetchone()[0]
            if integrity != "ok":
                errors.append("캔들 저장소 무결성 검사 실패")
            rows = db.execute("SELECT market,tf,COUNT(*),MAX(close_time) FROM candles GROUP BY market,tf").fetchall()
        # 검사 도중 새로 완료된 봉은 실제 조회 시각을 기준으로 판정합니다.
        now = engine.utc()
        for market, timeframe, count, latest in rows:
            counts[f"{market}/{timeframe}"] = count
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
            errors.append("네 코인의 완성 모델 근거가 모두 필요합니다")
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
            "checked_datasets": len(counts), "total_rows": sum(counts.values()), "errors_count": len(errors)}


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
    raise SystemExit(main())
