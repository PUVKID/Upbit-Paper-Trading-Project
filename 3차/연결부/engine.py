"""공개 시세만 사용하는 GPU 학습 및 현물 모의투자 엔진."""
from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import copy
import hashlib
from importlib import import_module
import itertools
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUNTIME_WORK = (ROOT / "work").resolve()
MARKETS = ("KRW-BTC", "KRW-SOL", "KRW-ETH", "KRW-XRP", "KRW-DOGE")
TIMEFRAMES = (1, 3, 5, 10, 15, 30, 60, 240, "day")
PAPER_ONLY = True
TRAINING_ETA = 0.05
EXECUTION_POLICY = "horizon_or_risk_v2"
PROTECTED_HOLDOUT = ("2026-10-19T00:00:00Z", "2026-10-20T00:00:00Z")
DEFAULT_CONFIG = {
    "data_dir": "work/data", "models_dir": "work/models",
    "reports_dir": "work/reports", "markets": list(MARKETS),
    "timeframes": list(TIMEFRAMES), "base_timeframe": 5,
    "external": {"enabled": False, "db_path": "3차/work/data/external.sqlite3",
                 "status_path": "3차/work/status.json"},
    "collection": {"recent_pages": 10, "backfill_pages_per_cycle": 5,
                   "poll_seconds": 30, "request_interval": 0.16, "retries": 4},
    "training": {"horizon": 12, "label_threshold": 0.006,
                 "max_rows_per_market": 120000, "max_raw_rows": 600000,
                 "train_days": 180, "tune_days": 30, "calibration_days": 30,
                 "holdout_days": 30, "gap_bars": 13, "min_split_rows": 100,
                 "half_life_days": 30, "candidate_depths": [3, 5], "rounds": 200,
                  "retrain_seconds": 0, "holdout_start": None, "holdout_end": None,
                  "rule_days": 2, "validation_days": 2, "min_closed_trades": 3,
                  "class_weight_mode": "recency", "class_weight_cap": 3.0,
                  "horizon_candidates": [12, 24], "threshold_candidates": [0.006, 0.003],
                  "class_weight_candidates": ["recency", "bounded_inverse_sqrt"]},
    "paper": {"initial_cash": 10000000, "fee": 0.0005, "slippage": 0.0005,
              "max_allocation": 0.125, "max_total_allocation": 0.5,
              "daily_loss_limit": 0.015, "stop_loss": 0.02, "take_profit": 0.04,
              "min_order": 5000, "poll_seconds": 30,
              "start_time": None, "max_quote_age_seconds": 10},
}


def utc(value=None):
    value = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def iso(value):
    return utc(value).isoformat()


def minutes(tf):
    return 1440 if str(tf) == "day" else int(tf)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(path=None):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        incoming = read_json(path)
        for key, value in incoming.items():
            if key not in config:
                raise ValueError(f"지원하지 않는 설정: {key}")
            if isinstance(config[key], dict):
                if key == "external" and not isinstance(value, dict):
                    raise ValueError("external 설정은 객체여야 합니다")
                unknown = set(value) - set(config[key])
                if unknown:
                    raise ValueError(f"지원하지 않는 설정: {key}.{sorted(unknown)}")
                config[key].update(value)
            else:
                config[key] = value
    if config["paper"]["start_time"] is None:
        config["paper"]["start_time"] = iso(None)
    if sorted(config["markets"]) != sorted(MARKETS):
        raise ValueError("모의 대상은 BTC/SOL/ETH/XRP/DOGE 다섯 종목으로 고정됩니다")
    if len(set(map(str, config["timeframes"]))) != len(config["timeframes"]):
        raise ValueError("중복 시간봉입니다")
    if any(str(tf) not in set(map(str, TIMEFRAMES)) for tf in config["timeframes"]):
        raise ValueError("지원하지 않는 시간봉입니다")
    if str(config["base_timeframe"]) not in set(map(str, config["timeframes"])):
        raise ValueError("기본 시간봉이 수집 목록에 없습니다")
    for key in ("data_dir", "models_dir", "reports_dir"):
        target = (ROOT / config[key]).resolve()
        if not target.is_relative_to((ROOT / "work").resolve()):
            raise ValueError("유동 산출물은 프로젝트 work 내부에만 저장합니다")
        config[key] = str(target)
    tr, paper, collection = config["training"], config["paper"], config["collection"]
    for key in ("horizon", "max_rows_per_market", "max_raw_rows", "train_days", "tune_days",
                "calibration_days", "rule_days", "validation_days", "min_closed_trades",
                "holdout_days", "gap_bars", "min_split_rows", "rounds"):
        if isinstance(tr[key], bool) or not isinstance(tr[key], int) or tr[key] <= 0:
            raise ValueError(f"training.{key}는 양의 정수여야 합니다")
    if tr["gap_bars"] < tr["horizon"] + 1:
        raise ValueError("분할 간격은 다음봉 체결을 포함한 horizon+1 이상이어야 합니다")
    if not tr["candidate_depths"] or any(not isinstance(d, int) or not 1 <= d <= 10 for d in tr["candidate_depths"]):
        raise ValueError("후보 깊이는 1~10 범위입니다")
    for key in ("fee", "slippage", "max_allocation", "max_total_allocation", "daily_loss_limit", "stop_loss", "take_profit"):
        if not math.isfinite(paper[key]) or not 0 <= paper[key] < 1:
            raise ValueError(f"paper.{key} 범위 오류")
    if not 0 < paper["initial_cash"] or paper["min_order"] < 5000:
        raise ValueError("초기 자본과 최소 주문금액을 확인하세요")
    if paper["max_allocation"] > paper["max_total_allocation"]:
        raise ValueError("코인별 배분이 전체 한도를 초과합니다")
    if (not math.isfinite(tr["half_life_days"]) or tr["half_life_days"] <= 0 or
            isinstance(tr["retrain_seconds"], bool) or not isinstance(tr["retrain_seconds"], (int, float)) or
            not math.isfinite(tr["retrain_seconds"]) or tr["retrain_seconds"] < 0):
        raise ValueError("최근 가중치/재학습 주기가 유효하지 않습니다")
    if (isinstance(tr["class_weight_cap"], bool) or not math.isfinite(tr["class_weight_cap"]) or
            not 1 <= tr["class_weight_cap"] <= 3):
        raise ValueError("클래스 가중치 상한은 1~3입니다")
    if (tr["class_weight_mode"] not in {"recency", "bounded_inverse_sqrt"} or
            not tr["class_weight_candidates"] or any(m not in {"recency", "bounded_inverse_sqrt"}
                                                     for m in tr["class_weight_candidates"])):
        raise ValueError("지원하지 않는 클래스 가중치 후보입니다")
    if (not tr["horizon_candidates"] or any(isinstance(h, bool) or not isinstance(h, int) or not 1 <= h <= 288
                                           for h in tr["horizon_candidates"])):
        raise ValueError("지평 후보는 1~288봉 정수입니다")
    if (not tr["threshold_candidates"] or any(isinstance(t, bool) or not math.isfinite(t) or not 0 < t < 1
                                              for t in [tr["label_threshold"], *tr["threshold_candidates"]])):
        raise ValueError("비용 차감 후 라벨 임계값이 유효하지 않습니다")
    if collection["request_interval"] < 0.11 or not 1 <= collection["retries"] <= 8:
        raise ValueError("공유 요청 간격은 0.11초 이상, 재시도는 1~8회입니다")
    if min(collection["recent_pages"], collection["backfill_pages_per_cycle"], collection["poll_seconds"], paper["poll_seconds"]) <= 0:
        raise ValueError("수집 페이지와 대기 시간은 양수여야 합니다")
    utc(paper["start_time"])
    config["external"] = import_module("3차.external_features").validate_config(config["external"], ROOT)
    return config


@contextlib.contextmanager
def connect(path):
    path = Path(path)
    # 운영 DB를 Windows와 도커의 서로 다른 잠금 체계로 동시에 열지 않습니다.
    if sys.platform == "win32" and path.resolve().is_relative_to(RUNTIME_WORK):
        raise RuntimeError("운영 SQLite는 도커 내부에서만 접근하십시오")
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=30000")
        # 내부 컨텍스트는 커밋/롤백, 외부 finally는 핸들 종료를 담당합니다.
        with db:
            yield db
    finally:
        db.close()


class CandleStore:
    def __init__(self, path):
        self.path = Path(path)
        with connect(path) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS candles (
                    market TEXT NOT NULL, tf TEXT NOT NULL, open_time TEXT NOT NULL,
                    close_time TEXT NOT NULL, open REAL, high REAL, low REAL,
                    close REAL, volume REAL, PRIMARY KEY(market,tf,open_time));
                CREATE INDEX IF NOT EXISTS candle_close ON candles(market,tf,close_time);
                CREATE TABLE IF NOT EXISTS cursors (
                    market TEXT, tf TEXT, cursor TEXT, done INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(market,tf));
                CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY, next_at REAL);
            """)

    def rate_limit(self, interval):
        with connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT next_at FROM budget WHERE id=1").fetchone()
            scheduled = max(time.time(), row[0] if row else 0)
            db.execute("INSERT OR REPLACE INTO budget VALUES(1,?)", (scheduled + interval,))
        time.sleep(max(0, scheduled - time.time()))

    def cursor(self, market, tf):
        with connect(self.path) as db:
            row = db.execute("SELECT cursor,done FROM cursors WHERE market=? AND tf=?", (market, str(tf))).fetchone()
            if row:
                return row[0], bool(row[1])
            row = db.execute("SELECT MIN(open_time) FROM candles WHERE market=? AND tf=?", (market, str(tf))).fetchone()
            return row[0], False

    def save_page(self, market, tf, frame, cursor=None, done=False, backfill=False):
        with connect(self.path) as db:
            rows = [(market, str(tf), iso(i), iso(i + pd.Timedelta(minutes=minutes(tf))),
                     *map(float, row)) for i, row in frame[["open", "high", "low", "close", "volume"]].iterrows()]
            # 완료 봉은 첫 저장 값을 보존하여 뒤늦은 조회가 과거 피처를 바꾸지 않습니다.
            db.executemany("INSERT OR IGNORE INTO candles VALUES(?,?,?,?,?,?,?,?,?)", rows)
            if backfill:
                db.execute("INSERT OR REPLACE INTO cursors VALUES(?,?,?,?)", (market, str(tf), cursor, int(done)))

    def read(self, market, tf, start=None, end=None, limit=120000):
        conditions, args = ["market=?", "tf=?"], [market, str(tf)]
        if start is not None:
            conditions.append("close_time>=?")
            args.append(iso(start))
        if end is not None:
            conditions.append("close_time<=?")
            args.append(iso(end))
        args.append(int(limit))
        with connect(self.path) as db:
            # 고정 시간봉은 종료 시각 순서도 같으므로 범위 인덱스를 그대로 활용합니다.
            frame = pd.read_sql_query("SELECT open_time,close_time,open,high,low,close,volume FROM candles WHERE " +
                                      " AND ".join(conditions) + " ORDER BY close_time DESC LIMIT ?", db, params=args)
        if frame.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                                index=pd.DatetimeIndex([], tz="UTC", name="open_time"))
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
        close_times = pd.to_datetime(frame.pop("close_time"), utc=True)
        if not close_times.eq(frame["open_time"] + pd.Timedelta(minutes=minutes(tf))).all():
            raise ValueError(f"{market}/{tf}: 캔들 종료 시각이 시간봉 간격과 다릅니다")
        return frame.set_index("open_time").sort_index()

    def summary(self):
        with connect(self.path) as db:
            rows = db.execute("SELECT market,tf,COUNT(*),MIN(open_time),MAX(close_time) FROM candles GROUP BY market,tf").fetchall()
            cursors = {(r[0], r[1]): bool(r[2]) for r in db.execute("SELECT market,tf,done FROM cursors")}
        return [{"market": m, "timeframe": tf, "rows": n, "start": a, "end": b,
                 "backfill_exhausted": cursors.get((m, tf), False)} for m, tf, n, a, b in rows]


def public_get(store, endpoint, params, collection):
    if endpoint != "orderbook" and not endpoint.startswith("candles/"):
        raise ValueError("공개 캔들/호가 읽기만 허용됩니다")
    url = "https://api.upbit.com/v1/" + endpoint + "?" + urllib.parse.urlencode(params)
    for attempt in range(collection["retries"]):
        store.rate_limit(collection["request_interval"])
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "paper-research/5"}, method="GET")
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read(2_000_001).decode("utf-8"))
            if not isinstance(payload, list):
                raise ValueError("공개 시세 응답이 배열이 아닙니다")
            return payload
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt + 1 == collection["retries"]:
                raise
            retry = exc.headers.get("Retry-After", "0")
            delay = max(2 ** attempt, float(retry) if retry.replace(".", "", 1).isdigit() else 0)
        except (urllib.error.URLError, TimeoutError):
            if attempt + 1 == collection["retries"]:
                raise
            delay = 2 ** attempt
        time.sleep(min(30, delay))
    raise RuntimeError("유한 재시도가 종료되었습니다")


def candles_from_api(payload, tf, now=None):
    rows = []
    for row in payload:
        if not isinstance(row["candle_date_time_utc"], str):
            raise ValueError("캔들 시각이 유효하지 않습니다")
        stamp = utc(row["candle_date_time_utc"])
        if pd.isna(stamp):
            raise ValueError("캔들 시각이 유효하지 않습니다")
        if stamp + pd.Timedelta(minutes=minutes(tf)) <= utc(now):
            rows.append((stamp, row["opening_price"], row["high_price"], row["low_price"],
                         row["trade_price"], row["candle_acc_trade_volume"]))
    frame = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                            index=pd.DatetimeIndex([], tz="UTC", name="open_time"))
    frame = frame.set_index("open_time").sort_index()
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("캔들 값이 유효하지 않습니다")
    if ((frame.high < frame[["open", "close", "low"]].max(axis=1)) |
        (frame.low > frame[["open", "close"]].min(axis=1))).any():
        raise ValueError("OHLC 범위가 유효하지 않습니다")
    return frame[~frame.index.duplicated()]


def collection_frame(payload, tf, market, params, data_dir):
    try:
        return candles_from_api(payload, tf)
    except ValueError as error:
        if str(error) not in {"캔들 값이 유효하지 않습니다", "OHLC 범위가 유효하지 않습니다"}:
            raise
    valid = []
    for row in payload:
        try:
            candles_from_api([row], tf)
        except ValueError as error:
            if str(error) not in {"캔들 값이 유효하지 않습니다", "OHLC 범위가 유효하지 않습니다"}:
                raise
            # 원본을 고치거나 버리지 않고 격리 저장이 성공한 뒤에만 커서를 진행합니다.
            identity = json.dumps([market, str(tf), row], sort_keys=True, allow_nan=False)
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            target = Path(data_dir) / "quarantine" / (digest + ".json")
            if not target.exists():
                atomic_json(target, {"market": market, "timeframe": str(tf), "raw": row,
                                     "reason": str(error), "request": params, "recorded_at": iso(None)})
        else:
            valid.append(row)
    return candles_from_api(valid, tf)


def collect_cycle(config, backfill=False, pages=None):
    count = pages or config["collection"]["backfill_pages_per_cycle" if backfill else "recent_pages"]
    if count <= 0:
        raise ValueError("페이지 수는 양수여야 합니다")
    store = CandleStore(Path(config["data_dir"]) / "candles.sqlite3")
    for market in config["markets"]:
        for tf in config["timeframes"]:
            cursor, done = store.cursor(market, tf) if backfill else (None, False)
            if done:
                continue
            for _ in range(count):
                params = {"market": market, "count": 200}
                if cursor:
                    params["to"] = cursor
                endpoint = "candles/days" if str(tf) == "day" else f"candles/minutes/{tf}"
                payload = public_get(store, endpoint, params, config["collection"])
                frame = collection_frame(payload, tf, market, params, config["data_dir"])
                oldest = min((utc(r["candle_date_time_utc"]) for r in payload), default=None)
                if cursor and oldest is not None and oldest >= utc(cursor):
                    raise RuntimeError(f"페이지 커서가 과거로 진행하지 않습니다: {market}/{tf}")
                next_cursor = iso(oldest) if oldest is not None else cursor
                exhausted = not payload
                store.save_page(market, tf, frame, next_cursor, exhausted, backfill)
                cursor = next_cursor
                if exhausted:
                    break
    datasets = store.summary()
    quarantined = sum(1 for _ in (Path(config["data_dir"]) / "quarantine").glob("*.json"))
    atomic_json(Path(config["data_dir"]) / "summary.json", {
        "updated_at": iso(None), "data": {"total_rows": sum(d["rows"] for d in datasets),
        "backfill": backfill, "quarantined_count": quarantined,
        "status": f"수집 진행 중; 오류 원본 {quarantined}개 격리; 무거래 누락은 보간하지 않음"}, "datasets": datasets})
    return datasets


def single_features(raw, tf):
    close, volume = raw.close, raw.volume
    result = pd.DataFrame(index=raw.index + pd.Timedelta(minutes=minutes(tf)))
    result.index.name = "close_time"
    prefix = str(tf) + "_"
    values = {"range": (raw.high - raw.low) / close, "body": (close - raw.open) / raw.open,
              "volume_ratio": volume / volume.rolling(24).mean().replace(0, np.nan),
              "volatility": close.pct_change(fill_method=None).rolling(24).std(),
              "ma_ratio": close / close.rolling(24).mean() - 1}
    for n in (1, 3, 6, 12, 24):
        values[f"return_{n}"] = close.pct_change(n, fill_method=None)
    for name, series in values.items():
        result[prefix + name] = series.to_numpy(dtype=np.float32)
    return result


def build_features(frames, base_timeframe=5):
    lookup = {str(k): v for k, v in frames.items()}
    base = lookup[str(base_timeframe)].sort_index()
    result = base.copy()
    result["open_time"] = base.index
    result.index = base.index + pd.Timedelta(minutes=minutes(base_timeframe))
    result.index.name = "close_time"
    for key, raw in lookup.items():
        feature = single_features(raw.sort_index(), key)
        available = f"available_{key}"
        feature[available] = feature.index
        result = pd.merge_asof(result.sort_index(), feature.sort_index(), left_index=True,
                               right_index=True, direction="backward", allow_exact_matches=True,
                               tolerance=pd.Timedelta(minutes=minutes(key) * 2))
    result["hour_sin"] = np.sin(2 * np.pi * result.index.hour / 24).astype(np.float32)
    result["hour_cos"] = np.cos(2 * np.pi * result.index.hour / 24).astype(np.float32)
    return result.replace([np.inf, -np.inf], np.nan)


def feature_columns(frame):
    excluded = {"open", "high", "low", "close", "volume", "open_time", "label", "label_end", "net_forward_return"}
    return [c for c in frame if c not in excluded and not c.startswith("available_")]


def label_frame(frame, horizon=12, threshold=0.006, fee=0.0005, slippage=0.0005):
    if horizon <= 0:
        raise ValueError("예측 지평은 양수여야 합니다")
    out = frame.copy()
    # 신호 확정 뒤 다음 봉 시가 진입, horizon번째 봉 종가 청산을 가정합니다.
    if out.empty:
        return out.assign(label=np.nan, label_end=pd.NaT, net_forward_return=np.nan)
    lengths = out.index - pd.DatetimeIndex(out.open_time)
    bar_length = lengths[0]
    if bar_length <= pd.Timedelta(0) or not (lengths == bar_length).all():
        raise ValueError("라벨 봉 길이가 일관되지 않습니다")
    next_close = out.index + bar_length
    target_close = out.index + horizon * bar_length
    # 행 이동은 누락 봉이 있을 때 실제 예측 시간을 늘리므로 정확한 시각으로 조회합니다.
    entry = pd.Series(out.open.reindex(next_close).to_numpy(), index=out.index) * (1 + slippage) * (1 + fee)
    exit_value = pd.Series(out.close.reindex(target_close).to_numpy(), index=out.index) * (1 - slippage) * (1 - fee)
    ret = exit_value / entry - 1
    labels = pd.Series(np.nan, index=out.index)
    valid = ret.notna() & (pd.DatetimeIndex(out.open_time.reindex(next_close)) == out.index)
    labels.loc[valid] = 1
    labels.loc[valid & (ret > threshold)] = 2
    labels.loc[valid & (ret < -threshold)] = 0
    out["label"] = labels
    out["label_end"] = pd.Series(target_close, index=out.index)
    out["net_forward_return"] = ret
    return out


def chronological_split(frame, boundaries, gap_bars, base_timeframe=5):
    gap = pd.Timedelta(minutes=gap_bars * minutes(base_timeframe))
    result = {}
    for name, (start, end) in boundaries.items():
        start, end = utc(start), utc(end)
        selected = frame[(frame.index >= start) & (frame.index < end - gap)]
        selected = selected[selected.label.notna() & (selected.label_end < end)]
        result[name] = selected
    return result


def recency_weights(index, half_life_days):
    ages = (index.max() - index).total_seconds().to_numpy() / 86400
    weights = np.exp2(-np.minimum(ages / half_life_days, 50))
    return (weights / weights.mean()).astype(np.float32)


def training_weights(train, settings):
    labels = train.label.to_numpy(dtype=int)
    counts = np.bincount(labels, minlength=3)
    factors = np.ones(3)
    if settings["class_weight_mode"] == "bounded_inverse_sqrt":
        factors = np.clip(np.sqrt(counts.max() / np.maximum(counts, 1)), 1, settings["class_weight_cap"])
    weights = recency_weights(train.index, settings["half_life_days"]) * factors[labels]
    weights = (weights / weights.mean()).astype(np.float32)
    return weights, {"scope": "train_only", "mode": settings["class_weight_mode"],
                     "class_counts": counts.tolist(), "class_factors": factors.tolist(),
                     "cap": settings["class_weight_cap"], "half_life_days": settings["half_life_days"]}


def next_research_candidate(config):
    path = Path(config["reports_dir"]) / "research_state.json"
    state = read_json(path) if path.exists() else {"next_sequence": 0}
    sequence = state["next_sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("연구 회차 커서가 손상되었습니다; 초기화하지 않습니다")
    tr = config["training"]
    grid = list(itertools.product(tr["horizon_candidates"], tr["threshold_candidates"], tr["class_weight_candidates"]))
    horizon, threshold, mode = grid[sequence % len(grid)]
    candidate = copy.deepcopy(config)
    candidate["training"].update(horizon=horizon, label_threshold=threshold, class_weight_mode=mode,
                                 gap_bars=max(tr["gap_bars"], horizon + 1))
    metadata = {"sequence": sequence, "candidate_id": f"h{horizon}-t{threshold:g}-{mode}",
                "candidate_index": sequence % len(grid) + 1, "candidate_total": len(grid),
                "horizon": horizon, "horizon_minutes": horizon * minutes(config["base_timeframe"]),
                "label_threshold": threshold, "class_weight_mode": mode,
                "scope": "순환 연구; 반복된 과거 표본은 독립 시험이 아님"}
    # 실패 회차도 같은 입력의 무의미한 반복으로 바꾸지 않도록 시도 전에 커서를 보존합니다.
    atomic_json(path, {"schema_version": 1, "next_sequence": sequence + 1,
                       "last_attempt": metadata, "updated_at": iso(None)})
    return candidate, metadata


def normalize_probs(probs):
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p < 0).any() or (p.sum(axis=1) <= 0).any():
        raise ValueError("세 클래스 확률 배열이 유효하지 않습니다")
    p = np.clip(p, 1e-12, 1)
    return p / p.sum(axis=1, keepdims=True)


def logloss(labels, probs):
    p = normalize_probs(probs)
    return float(-np.log(p[np.arange(len(p)), np.asarray(labels, dtype=int)]).mean())


def ensemble_weights(probabilities, labels, steps=200):
    p = np.stack([normalize_probs(v) for v in probabilities])
    truth = p[:, np.arange(p.shape[1]), np.asarray(labels, dtype=int)]
    weights = np.ones(len(p)) / len(p)
    for _ in range(steps):
        mix = np.maximum(weights @ truth, 1e-12)
        gradient = -(truth / mix).mean(axis=1)
        weights *= np.exp(-0.05 * np.clip(gradient - gradient.mean(), -20, 20))
        weights = np.maximum(weights, 1e-9)
        weights /= weights.sum()
    return weights


def temperature_probs(probs, temperature):
    logits = np.log(normalize_probs(probs)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    return normalize_probs(np.exp(logits))


def signals_from_proba(probs, rule):
    p = normalize_probs(probs)
    signals = np.ones(len(p), dtype=int)
    signals[(p[:, 2] >= rule["confidence"]) & (p[:, 2] - p[:, 0] >= rule["edge"])] = 2
    signals[(p[:, 0] >= rule["confidence"]) & (p[:, 0] - p[:, 2] >= rule["edge"])] = 0
    return signals


def probability_diagnostics(probs, rule):
    p = normalize_probs(probs)
    signals = signals_from_proba(p, rule)
    no_trade = rule["confidence"] > 1 or rule["edge"] > 1
    return {"rows": len(p), "argmax_counts": np.bincount(p.argmax(axis=1), minlength=3).tolist(),
            "signal_counts": np.bincount(signals, minlength=3).tolist(),
            "probability_quantiles": {str(c): dict(zip(("min", "p25", "p50", "p75", "p95", "max"),
                np.quantile(p[:, c], [0, .25, .5, .75, .95, 1]).tolist())) for c in range(3)},
            "filter_reasons": {"no_trade_policy": len(p) if no_trade else 0,
                "buy_confidence_below": int((p[:, 2] < rule["confidence"]).sum()),
                "buy_edge_below": int((p[:, 2] - p[:, 0] < rule["edge"]).sum()),
                "neutral_signal": int((signals == 1).sum())}}


def trading_eligibility(result, min_closed_trades):
    reasons = []
    objective = result["net_return"] - 0.5 * result["MDD"]
    if not math.isfinite(objective) or result["net_return"] <= 0:
        reasons.append("non_positive_net_return")
    if not math.isfinite(objective) or objective <= 0:
        reasons.append("non_positive_risk_adjusted_return")
    if result["trades"] < min_closed_trades:
        reasons.append("insufficient_closed_trades")
    return {"eligible": not reasons, "rejection_reasons": reasons, "objective": objective,
            "min_closed_trades": min_closed_trades,
            "scope": "운용 보류를 위한 증거 하한; 최소 거래 강제나 통계적 수익 보장이 아님"}


def deployment_guard(manifest):
    if manifest.get("schema_version") != 2 or manifest.get("execution_policy") != EXECUTION_POLICY:
        return "legacy_schema_requires_validation"
    if manifest.get("deployment", {}).get("eligible") is not True:
        return "unvalidated_or_unprofitable_model"
    return None


def classification_metrics(labels, probs, majority_class=1):
    y = np.asarray(labels, dtype=int)
    pred = normalize_probs(probs).argmax(axis=1)
    if not len(y) or len(y) != len(pred):
        raise ValueError("전체 표본 평가를 위한 행이 없습니다")
    per_class, recalls, f1s = {}, [], []
    for cls in range(3):
        tp = int(((y == cls) & (pred == cls)).sum())
        support, predicted = int((y == cls).sum()), int((pred == cls).sum())
        precision, recall = tp / max(1, predicted), tp / max(1, support)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class[str(cls)] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
        if support:
            recalls.append(recall)
        f1s.append(f1)
    accuracy = float((y == pred).mean())
    return {"accuracy": accuracy, "macro_f1": float(np.mean(f1s)),
            "balanced_accuracy": float(np.mean(recalls)), "majority_baseline": float((y == majority_class).mean()),
            "majority_class_from_train": int(majority_class), "per_class": per_class, "support": len(y),
            "class_distribution": np.bincount(y, minlength=3).tolist(), "target": 0.95,
            "target_met": accuracy >= 0.95, "target_status": "95% 목표 달성" if accuracy >= 0.95 else "95% 목표 미달",
            "logloss": logloss(y, probs)}


def sweep_book(units, side, amount, fee=0.0005, slippage=0.0005):
    """매수 amount는 수수료 포함 원화, 매도 amount는 현물 수량입니다."""
    if side not in ("buy", "sell") or not math.isfinite(amount) or amount < 0:
        raise ValueError("현물 모의체결 입력 오류")
    remaining, qty, gross, cash = amount, 0.0, 0.0, 0.0
    price_key, size_key = ("ask_price", "ask_size") if side == "buy" else ("bid_price", "bid_size")
    levels = sorted(units, key=lambda u: u[price_key], reverse=side == "sell")
    for unit in levels:
        price, size = float(unit[price_key]), float(unit[size_key])
        if not math.isfinite(price + size) or price <= 0 or size < 0:
            raise ValueError("호가 가격/잔량 오류")
        adjusted = price * (1 + slippage if side == "buy" else 1 - slippage)
        if side == "buy":
            fill = min(size, remaining / (adjusted * (1 + fee)))
            flow = fill * adjusted * (1 + fee)
        else:
            fill = min(size, remaining)
            flow = fill * adjusted * (1 - fee)
        qty += fill
        gross += fill * adjusted
        cash += flow
        remaining = max(0.0, remaining - (flow if side == "buy" else fill))
        if remaining <= 1e-10:
            break
    return {"qty": qty, "cash": cash, "vwap": gross / qty if qty else None,
            "fee_paid": abs(cash - gross), "unfilled": remaining, "partial": remaining > 1e-8}


def volatility_stop_fraction(volatility, stop_cap):
    volatility = float(volatility)
    if not math.isfinite(volatility) or volatility < 0:
        raise ValueError("현재 변동성이 유효하지 않습니다")
    return min(stop_cap, max(min(0.005, stop_cap), 2 * volatility))


def roundtrip_cost_factor(settings):
    return ((1 - settings["slippage"]) * (1 - settings["fee"]) /
            ((1 + settings["slippage"]) * (1 + settings["fee"])))


class Portfolio:
    def __init__(self, settings=None, state=None):
        self.settings = copy.deepcopy(DEFAULT_CONFIG["paper"] if settings is None else settings)
        initial = self.settings["initial_cash"]
        if state is not None and (not isinstance(state, dict) or
                                  state.get("initial_cash") != initial or state.get("paper_only") is not True):
            raise ValueError("복원 계좌의 초기 원금 또는 모의 전용 표시가 설정과 다릅니다")
        self.state = copy.deepcopy(state) if state is not None else {
            "cash": float(initial), "initial_cash": float(initial), "positions": {},
            "day": None, "daily_start_equity": float(initial), "new_buys_blocked": False,
            "last_bar": None, "peak_equity": float(initial), "MDD": 0.0, "trades": [],
            "completed_trades": 0, "fees_paid": 0.0, "equity_history": [], "paper_only": True,
        }
        if self.state["cash"] < 0 or any(v["qty"] < 0 for v in self.state["positions"].values()):
            raise ValueError("현물 계좌에 음수 잔액/공매도가 있습니다")
        # 기존 장부와 모델 정책을 유지하는 신규진입 리스크 보강입니다.
        self.state.setdefault("stop_reentry_until", {})
        self.state["risk_overlay"] = "stop_horizon_quote_cost_v1"

    def equity(self, prices):
        total = self.state["cash"]
        for market, position in self.state["positions"].items():
            if market not in prices or not math.isfinite(prices[market]) or prices[market] <= 0:
                raise ValueError(f"보유자산 평가 가격이 없습니다: {market}")
            total += position["qty"] * prices[market] * (1 - self.settings["slippage"]) * (1 - self.settings["fee"])
        return float(total)

    def process(self, bar, prices, signals, books=None, mark_prices=None, stop_fractions=None, allow_buys=True,
                entry_expiries=None, expiry_prices=None, execution_at=None):
        stamp = iso(bar)
        execution_at = utc(bar if execution_at is None else execution_at)
        if self.state["last_bar"] is not None and utc(stamp) <= utc(self.state["last_bar"]):
            return []
        if any(m not in MARKETS for m in prices) or any(v not in (0, 1, 2) for v in signals.values()):
            raise ValueError("지원하지 않는 현물 시장/신호입니다")
        before = self.equity(prices)
        day = utc(bar).date().isoformat()
        if self.state["day"] != day:
            self.state.update(day=day, daily_start_equity=before, new_buys_blocked=False)
        if before <= self.state["daily_start_equity"] * (1 - self.settings["daily_loss_limit"]):
            self.state["new_buys_blocked"] = True
        events, sold = [], set()
        decisions = {}
        self.state["decision_diagnostics"] = decisions

        def execute(market, side, amount, at_price=None):
            if books is None:
                price = prices[market] if at_price is None else at_price
                units = [{"ask_price": price, "bid_price": price,
                          "ask_size": 1e30, "bid_size": 1e30}]
            else:
                units = books[market]
            return sweep_book(units, side, amount, self.settings["fee"], self.settings["slippage"])

        def sell(market, position, reason, at_price=None, at_time=None):
            fill = execute(market, "sell", position["qty"], at_price)
            if fill["qty"] <= 0:
                return
            if reason == "stop_loss":
                until = position.get("expires_at")
                if until is None:
                    horizon = self.state.get("risk_horizon_minutes", {}).get(
                        market, DEFAULT_CONFIG["training"]["horizon"] * minutes(DEFAULT_CONFIG["base_timeframe"]))
                    # 만료 미기록 legacy는 손절부터 전체 지평을 보수적으로 보류합니다.
                    if isinstance(horizon, (int, float)) and not isinstance(horizon, bool) and math.isfinite(horizon) and horizon > 0:
                        until = iso(utc(execution_at if at_time is None else at_time) + pd.Timedelta(horizon, unit="min"))
                self.state["stop_reentry_until"][market] = until
            old_qty = position["qty"]
            basis = position["cost"] * fill["qty"] / old_qty
            position["qty"] = max(0.0, old_qty - fill["qty"])
            position["cost"] -= basis
            self.state["cash"] += fill["cash"]
            if position["qty"] <= 1e-12:
                del self.state["positions"][market]
                self.state["completed_trades"] += 1
            sold.add(market)
            events.append({"market": market, "side": "sell", "reason": reason,
                           "realized_pnl": fill["cash"] - basis,
                           "timestamp": iso(execution_at if at_time is None else at_time), **fill})

        # 손실 한도는 매수에만 적용합니다. 청산과 손절은 계속 허용합니다.
        for market, position in list(self.state["positions"].items()):
            stop = min(self.settings["stop_loss"], (stop_fractions or {}).get(market, self.settings["stop_loss"]))
            reason = "signal"
            change = prices[market] / position["entry_price"] - 1
            if change <= -stop:
                reason = "stop_loss"
            elif change >= self.settings["take_profit"]:
                reason = "take_profit"
            elif expiry_prices is None and position.get("expires_at") and execution_at >= utc(position["expires_at"]):
                reason = "time_exit"
            elif signals.get(market, 1) != 0:
                continue
            sell(market, position, reason)
        for market in sorted(prices):
            decision = {"signal": signals.get(market, 1), "reason": "ready"}
            decisions[market] = decision
            if not allow_buys:
                decision["reason"] = "buys_disabled"
                continue
            if self.state["new_buys_blocked"]:
                decision["reason"] = "daily_loss_limit"
                continue
            if signals.get(market, 1) != 2:
                decision["reason"] = "no_buy_signal"
                continue
            if market in self.state["positions"]:
                decision["reason"] = "existing_position"
                continue
            if market in sold:
                decision["reason"] = "sold_this_cycle"
                continue
            cooldowns = self.state["stop_reentry_until"]
            if market in cooldowns:
                try:
                    if cooldowns[market] is None:
                        raise ValueError("재진입 보류 만료시각이 없습니다")
                    until = utc(cooldowns[market])
                    if pd.isna(until):
                        raise ValueError("재진입 보류 만료시각이 유효하지 않습니다")
                except (ValueError, TypeError, OverflowError):
                    decision.update(reason="invalid_reentry_expiry", blocked_until=cooldowns[market])
                    continue
                if execution_at < until:
                    decision.update(reason="stop_reentry_cooldown", blocked_until=iso(until))
                    continue
                del cooldowns[market]
            exposure = sum(p["qty"] * prices[m] for m, p in self.state["positions"].items())
            initial = self.state["initial_cash"]
            budget = min(self.state["cash"], initial * self.settings["max_allocation"],
                         max(0.0, initial * self.settings["max_total_allocation"] - exposure))
            if budget < self.settings["min_order"]:
                decision.update(reason="below_min_order", budget=budget)
                continue
            expiry = (entry_expiries or {}).get(market)
            if expiry is not None and utc(expiry) <= execution_at:
                decision["reason"] = "expired_signal"
                continue
            fill = execute(market, "buy", budget)
            if fill["qty"] <= 0:
                decision["reason"] = "no_ask_liquidity"
                continue
            net_budget = self.state.get("entry_cost_limits", {}).get(market)
            if books is not None and net_budget is not None:
                if not math.isfinite(net_budget) or net_budget <= 0:
                    decision["reason"] = "invalid_entry_cost_limit"
                    continue
                exit_fill = execute(market, "sell", fill["qty"])
                if exit_fill["partial"]:
                    decision["reason"] = "insufficient_exit_liquidity"
                    continue
                base_factor = self.state.get("entry_cost_baselines", {}).get(market, roundtrip_cost_factor(self.settings))
                if not math.isfinite(base_factor) or base_factor <= 0:
                    decision["reason"] = "invalid_entry_cost_baseline"
                    continue
                retention = exit_fill["cash"] / fill["cash"]
                # 라벨은 기본 수수료·슬리피지 차감 후입니다. 추가 호가비용만 비율로 반영합니다.
                remaining_budget = (1 + net_budget) * retention / base_factor - 1
                decision.update(roundtrip_cost=1 - retention, incremental_quote_drag=1 - retention / base_factor,
                                net_label_budget=net_budget, net_label_budget_after_book_cost=remaining_budget,
                                cost_budget_scope="순수익 라벨 기준의 위험 예산 대리값; 기대이익 아님")
                if remaining_budget <= 0:
                    decision["reason"] = "execution_cost_exceeds_label_budget"
                    continue
            decision.update(reason="buy_filled", partial=fill["partial"])
            self.state["cash"] = max(0.0, self.state["cash"] - fill["cash"])
            self.state["positions"][market] = {"qty": fill["qty"], "cost": fill["cash"], "entry_price": fill["vwap"]}
            self.state["positions"][market]["entry_time"] = iso(execution_at)
            if expiry is not None:
                self.state["positions"][market]["expires_at"] = iso(expiry)
            events.append({"market": market, "side": "buy", "reason": "signal", "timestamp": iso(execution_at), **fill})
        # 시가 매수에 미래 종가 청산 대금을 쓰지 않도록 지평 종가 청산은 매수 처리 뒤에 합니다.
        if expiry_prices is not None:
            for market, position in list(self.state["positions"].items()):
                if position.get("expires_at") and utc(bar) >= utc(position["expires_at"]):
                    sell(market, position, "time_exit", expiry_prices[market], bar)
        final = self.equity(prices if mark_prices is None else mark_prices)
        self.state["peak_equity"] = max(self.state["peak_equity"], final)
        self.state["MDD"] = max(self.state["MDD"], 1 - final / self.state["peak_equity"])
        if final <= self.state["daily_start_equity"] * (1 - self.settings["daily_loss_limit"]):
            self.state["new_buys_blocked"] = True
        for i, event in enumerate(events):
            event.update(event_id=f"{stamp}/{event['market']}/{i}", paper_only=True)
            self.state["fees_paid"] += event["fee_paid"]
        self.state["trades"] = (self.state["trades"] + events)[-2000:]
        self.state["last_bar"] = stamp
        self.state["equity"] = final
        self.state["net_return"] = final / self.state["initial_cash"] - 1
        self.state["equity_history"] = (self.state["equity_history"] + [{"timestamp": stamp, "equity": final}])[-5000:]
        self.state["assets"] = [{"market": m, **p, "price": (mark_prices or prices)[m],
                                  "value": p["qty"] * (mark_prices or prices)[m]} for m, p in self.state["positions"].items()]
        return events


def backtest_portfolio(frames, probabilities, rules, settings):
    common = None
    for frame in frames.values():
        common = frame.index if common is None else common.intersection(frame.index)
    if common is None or len(common) < 2:
        raise ValueError("공통계좌 백테스트에 공통 완료 봉이 부족합니다")
    portfolio = Portfolio(settings)
    portfolio.state["risk_horizon_minutes"] = {m: rules[m].get("horizon_minutes",
        DEFAULT_CONFIG["training"]["horizon"] * minutes(DEFAULT_CONFIG["base_timeframe"])) for m in frames}
    portfolio.state["entry_cost_limits"] = {m: rules[m]["net_label_threshold"] for m in frames if "net_label_threshold" in rules[m]}
    portfolio.state["entry_cost_baselines"] = {m: rules[m].get("label_cost_factor", roundtrip_cost_factor(settings)) for m in frames}
    volatility_columns = {}
    for market, frame in frames.items():
        bar_minutes = int((utc(frame.index[0]) - utc(frame.open_time.iloc[0])).total_seconds() / 60)
        column = ("day" if bar_minutes == 1440 else str(bar_minutes)) + "_volatility"
        if column not in frame and any(c.endswith("_volatility") or c.startswith("available_") for c in frame.columns):
            raise ValueError(f"{market}: 기본 시간봉 변동성 피처가 없습니다")
        volatility_columns[market] = column if column in frame else None
    signals = {m: pd.Series(signals_from_proba(probabilities[m], rules[m]), index=frames[m].index) for m in frames}
    pending, pending_at = {}, None
    pending_stops = {m: settings["stop_loss"] for m in frames}
    volatility_ready = {m: False for m in frames}
    volatility_fallback_rows = {m: 0 for m in frames}
    for stamp in common:
        rows = {m: frames[m].loc[stamp] for m in frames}
        opening = {m: float(r.open) for m, r in rows.items()}
        closing = {m: float(r.close) for m, r in rows.items()}
        executable = {m: pending.get(m, 1) if pending_at is not None and utc(rows[m].open_time) == pending_at else 1 for m in frames}
        expiries = {m: iso(pending_at + pd.Timedelta(minutes=rules[m]["horizon_minutes"]))
                    for m in frames if pending_at is not None and rules[m].get("horizon_minutes")}
        execution_at = utc(next(iter(rows.values())).open_time)
        portfolio.process(stamp, opening, executable, mark_prices=closing, entry_expiries=expiries,
                          expiry_prices=closing, execution_at=execution_at, stop_fractions=pending_stops)
        # 초기 미준비 fixture만 고정 손절로 표시합니다. 운영 학습은 전체 피처 dropna 후 들어옵니다.
        for market, column in volatility_columns.items():
            if column is None or pd.isna(rows[market][column]) and not volatility_ready[market]:
                pending_stops[market] = settings["stop_loss"]
                volatility_fallback_rows[market] += 1
            else:
                pending_stops[market] = volatility_stop_fraction(rows[market][column], settings["stop_loss"])
                volatility_ready[market] = True
        pending, pending_at = {m: int(signals[m].loc[stamp]) for m in frames}, stamp
    state = portfolio.state
    exit_reasons = {reason: sum(e["side"] == "sell" and e["reason"] == reason for e in state["trades"])
                    for reason in ("signal", "stop_loss", "take_profit", "time_exit")}
    return {"net_return": state["net_return"], "MDD": state["MDD"], "trades": state["completed_trades"],
            "executions": len(state["trades"]), "fees_paid": state["fees_paid"], "equity": state["equity"],
            "realized_pnl": sum(e.get("realized_pnl", 0) for e in state["trades"]), "exit_reasons": exit_reasons,
            "common_rows": len(common), "start": iso(common[0]), "end": iso(common[-1]),
            "risk_overlay": state["risk_overlay"], "historical_quote_costs": "과거 호가 없음; 스프레드 성과 미검증",
            "volatility_fallback_rows": volatility_fallback_rows,
            "valuation": "기말 현물의 매도 수수료·슬리피지 차감 청산가치", "fill": "신호 이후 다음 봉 시가"}


def tune_rule(frame, probabilities, settings, market, min_closed_trades=3, horizon_minutes=None):
    best, best_score, best_result = None, -float("inf"), None
    trials = []
    # 무거래 후보를 포함하여 비용 차감 후 손실을 유발하는 진입을 억제합니다.
    for confidence, edge in [(1.01, 1.01)] + [(c, e) for c in (0.4, 0.5, 0.6, 0.7) for e in (0.05, 0.15, 0.25)]:
        rule = {"confidence": confidence, "edge": edge}
        if horizon_minutes is not None:
            rule["horizon_minutes"] = horizon_minutes
        result = backtest_portfolio({market: frame}, {market: probabilities}, {market: rule}, settings)
        score = result["net_return"] - result["MDD"] * 0.5
        eligibility = trading_eligibility(result, min_closed_trades)
        trials.append({"rule": rule, "trading": result, **eligibility,
                       "diagnostics": probability_diagnostics(probabilities, rule)})
        if best is None or eligibility["eligible"] and score > best_score:
            best, best_score, best_result = rule, score, result
    eligible = trading_eligibility(best_result, min_closed_trades)
    return {**best, "validation_objective": best_score, "validation_trading": best_result,
            **eligible, "trials": trials, "selection_scope": "rule_only"}


def assert_cuda_config(config):
    data = json.loads(config) if isinstance(config, str) else config
    device = data.get("learner", {}).get("generic_param", {}).get("device", "")
    if not device.startswith("cuda"):
        raise RuntimeError(f"GPU 실제 부스터 설정이 아닙니다: device={device!r}; CPU 대체 금지")
    return device


_gpu_matrices = ContextVar("gpu_matrices", default=None)


@contextlib.contextmanager
def gpu_matrix_scope():
    # 같은 시장의 읽기 전용 후보 묶음 안에서만 행렬을 보존합니다.
    token = _gpu_matrices.set({})
    try:
        yield
    finally:
        _gpu_matrices.get().clear()
        _gpu_matrices.reset(token)


def fit_gpu(train, tune, columns, depth, config, on_progress=None):
    started = time.monotonic()
    import xgboost as xgb
    if not xgb.build_info().get("USE_CUDA", False):
        raise RuntimeError("설치한 XGBoost가 CUDA를 지원하지 않습니다")
    params = {"objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
              "tree_method": "hist", "device": "cuda:0", "max_depth": depth,
              "eta": TRAINING_ETA, "subsample": 0.9, "colsample_bytree": 0.9,
              "max_bin": 128, "seed": 42, "nthread": 4}
    tr = config["training"]
    matrix_started = time.monotonic()
    cache = _gpu_matrices.get()
    signature = (tuple(columns), tr["half_life_days"], tr["class_weight_mode"], tr["class_weight_cap"])
    reused = (cache is not None and cache.get("train") is train and cache.get("tune") is tune
              and cache.get("signature") == signature)
    if reused:
        dtrain, dtune, weighting_evidence = cache["matrices"]
    else:
        weights, weighting_evidence = training_weights(train, tr)
        dtrain = xgb.DMatrix(train[columns].to_numpy(dtype=np.float32), label=train.label.to_numpy(dtype=int),
                             weight=weights, feature_names=columns)
        dtune = xgb.DMatrix(tune[columns].to_numpy(dtype=np.float32), label=tune.label.to_numpy(dtype=int), feature_names=columns)
        if cache is not None:
            cache.clear()
            cache.update(train=train, tune=tune, signature=signature, matrices=(dtrain, dtune, weighting_evidence))
    matrix_seconds = time.monotonic() - matrix_started
    preflight_started = time.monotonic()
    # save_config는 실제 CUDA 컨텍스트를 구성하므로 학습 0회 상태에서 CPU 전환을 검출합니다.
    preflight = xgb.Booster(params, cache=[dtrain, dtune])
    assert_cuda_config(preflight.save_config())
    del preflight
    preflight_seconds = time.monotonic() - preflight_started

    class CudaOnly(xgb.callback.TrainingCallback):
        def before_training(self, model):
            # 별도 사전 검사 뒤 생성된 실제 학습 부스터도 첫 업데이트 전에 확인합니다.
            assert_cuda_config(model.save_config())
            return model

        def before_iteration(self, model, epoch, evals_log):
            assert_cuda_config(model.save_config())
            return False

    callbacks, history = [CudaOnly()], {}
    if on_progress is not None:
        class MeasuredProgress(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                rounds = epoch + 1
                if rounds == 1 or rounds % 10 == 0:
                    on_progress(rounds, float(evals_log["tune"]["mlogloss"][-1]))
                return False
        callbacks.append(MeasuredProgress())
    fit_started = time.monotonic()
    booster = xgb.train(params, dtrain, num_boost_round=tr["rounds"], evals=[(dtune, "tune")],
                        early_stopping_rounds=min(30, max(2, tr["rounds"] // 5)), verbose_eval=False,
                        callbacks=callbacks, evals_result=history)
    fit_seconds = time.monotonic() - fit_started
    device = assert_cuda_config(booster.save_config())
    boosted_rounds = int(booster.num_boosted_rounds())
    last_loss = float(history["tune"]["mlogloss"][-1])
    if on_progress is not None:
        on_progress(boosted_rounds, last_loss)
    return booster, {"actual_device": device, "best_iteration": int(booster.best_iteration),
                     "requested_device": "cuda:0", "depth": depth, "xgboost_version": xgb.__version__,
                     "parameters": params.copy(),
                     "eta": params["eta"],
                     "duration_seconds": time.monotonic() - started, "boosted_rounds": boosted_rounds,
                     "validation_logloss": float(booster.best_score), "last_validation_logloss": last_loss,
                     "validation_scope": "tune", "train_rows": len(train), "tune_rows": len(tune),
                      "features": len(columns), "weighting": weighting_evidence,
                      "matrix_reused": reused,
                      "timings": {"matrix_prepare_seconds": matrix_seconds,
                                  "cuda_preflight_seconds": preflight_seconds, "fit_seconds": fit_seconds}}


def predict_booster(booster, frame, columns, best_iteration):
    import xgboost as xgb
    matrix = xgb.DMatrix(frame[columns].to_numpy(dtype=np.float32), feature_names=columns)
    return normalize_probs(booster.predict(matrix, iteration_range=(0, best_iteration + 1)))


def check_gpu():
    index = pd.date_range("2020-01-01", periods=120, freq="h", tz="UTC")
    frame = pd.DataFrame({"x": np.arange(120) % 11, "label": np.arange(120) % 3}, index=index)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["training"]["rounds"] = 4
    _, evidence = fit_gpu(frame.iloc[:90], frame.iloc[90:], ["x"], 2, cfg)
    return {"status": "GPU 실제 학습 확인", **evidence, "financial_performance_evidence": False}


class ExperimentAudit:
    def __init__(self, path):
        self.path = Path(path)
        with connect(path) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS exposures (run TEXT, start TEXT, end TEXT, kind TEXT);
                CREATE TABLE IF NOT EXISTS final_tests (id TEXT PRIMARY KEY, start TEXT, end TEXT,
                    status TEXT, manifest_hash TEXT, created_at TEXT);
            """)

    def expose(self, run, start, end, kind):
        with connect(self.path) as db:
            db.execute("INSERT INTO exposures VALUES(?,?,?,?)", (run, iso(start), iso(end), kind))

    def reserve_final(self, run, start, end, manifest_hash):
        start, end = iso(start), iso(end)
        with connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM final_tests WHERE start<? AND end>?", (end, start)).fetchone():
                raise RuntimeError("이 시험 기간은 이미 소비 또는 평가 시도되었습니다. 반복 튜닝 금지")
            if db.execute("SELECT 1 FROM exposures WHERE start<? AND end>?", (end, start)).fetchone():
                raise RuntimeError("시험 기간이 이전 학습/검증에 노출되어 미사용 시험으로 평가할 수 없습니다")
            db.execute("INSERT INTO final_tests VALUES(?,?,?,?,?,?)", (run, start, end, "reserved", manifest_hash, iso(None)))

    def assert_final_available(self, start, end):
        with connect(self.path) as db:
            if db.execute("SELECT 1 FROM final_tests WHERE start<? AND end>?", (iso(end), iso(start))).fetchone():
                raise RuntimeError("이 시험 기간은 이미 소비 또는 평가 시도되었습니다. 반복 튜닝 금지")
            if db.execute("SELECT 1 FROM exposures WHERE start<? AND end>?", (iso(end), iso(start))).fetchone():
                raise RuntimeError("시험 기간이 이전 학습/검증에 노출되어 미사용 시험으로 평가할 수 없습니다")

    def consumed_periods(self):
        with connect(self.path) as db:
            return db.execute("SELECT start,end FROM final_tests").fetchall()

    def finish(self, run):
        with connect(self.path) as db:
            db.execute("UPDATE final_tests SET status='complete' WHERE id=?", (run,))


def progress(**fields):
    atomic_json(ROOT / "work/service/training_progress.json", {"updated_at": iso(None), **fields})


def protected_periods(config, consumed=()):
    tr = config["training"]
    values = [PROTECTED_HOLDOUT, (tr["holdout_start"], tr["holdout_end"]), *consumed]
    merged = []
    for start, end in sorted((utc(a), utc(b)) for a, b in values if a and b):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def experiment_plan(config, live=False, now=None, consumed=()):
    tr = config["training"]
    if not tr["holdout_start"] or not tr["holdout_end"]:
        raise ValueError("training.holdout_start/holdout_end를 고정 UTC 시각으로 명시해야 합니다")
    hs, he = utc(tr["holdout_start"]), utc(tr["holdout_end"])
    if he <= hs:
        raise ValueError("잠금 시험 종료는 시작 이후여야 합니다")
    if abs((he - hs).total_seconds() / 86400 - tr["holdout_days"]) > 1e-6:
        raise ValueError("holdout_days와 명시 시험 기간이 일치하지 않습니다")
    forbidden = protected_periods(config, consumed)
    end = utc(now).floor(f"{minutes(config['base_timeframe'])}min") if live else hs
    result = {}
    for name in ("validation", "rule", "calibration", "tune", "train"):
        for left, right in reversed(forbidden):
            if left < end <= right:
                end = left
        start = end - pd.Timedelta(days=tr[name + "_days"])
        # 보호 기간을 제거한 뒤에도 요청한 실제 연구 기간을 확보합니다.
        for left, right in reversed(forbidden):
            if start < right and end > left:
                start -= min(end, right) - left
        result[name] = (start, end)
        end = start
    return {name: result[name] for name in ("train", "tune", "calibration", "rule", "validation")} | {"holdout": (hs, he)}


def research_splits(labelled, plan, config, consumed=()):
    tr = config["training"]
    result = chronological_split(labelled, {k: v for k, v in plan.items() if k != "holdout"},
                                 tr["gap_bars"], config["base_timeframe"])
    for name, part in result.items():
        for left, right in protected_periods(config, consumed):
            part = part[(part.label_end < left) | (part.index >= right)]
        if len(part) < tr["min_split_rows"]:
            raise ValueError(f"{name}: 보호 기간/지평 제외 후 {len(part)}행, 최소 {tr['min_split_rows']}행 필요")
        result[name] = part
    return result


def load_market_features(store, market, config, start, end):
    frames = {}
    tr = config["training"]
    for tf in config["timeframes"]:
        warm_start = utc(start) - pd.Timedelta(minutes=minutes(tf) * 26)
        raw = store.read(market, tf, warm_start, end, tr["max_raw_rows"])
        if raw.empty:
            raise ValueError(f"수집된 {market}/{tf} 봉이 없습니다")
        frames[str(tf)] = raw
    result = build_features(frames, config["base_timeframe"]).tail(tr["max_rows_per_market"])
    if "external" in config:
        external = import_module("3차.external_features")
        settings = external.validate_config(config["external"], ROOT)
        if type(settings["enabled"]) is not bool:
            raise ValueError("외부 특징 사용 여부는 회차 시작에 bool로 동결되어야 합니다")
        if settings["enabled"]:
            result = external.load_and_join(result, market, settings, ROOT, utc(None))
    return result


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def promotion_decision(candidate, incumbent):
    for metrics in (candidate, incumbent):
        if not all(math.isfinite(metrics[key]) for key in ("logloss", "net_return", "MDD")):
            raise ValueError("모델 승격 비교 지표가 유한하지 않습니다")
    candidate_objective = candidate["net_return"] - 0.5 * candidate["MDD"]
    incumbent_objective = incumbent["net_return"] - 0.5 * incumbent["MDD"]
    loss_ok = candidate["logloss"] <= incumbent["logloss"] + 1e-12
    trading_ok = candidate_objective >= incumbent_objective - 1e-12
    improved = (candidate["logloss"] < incumbent["logloss"] - 1e-12 or
                candidate_objective > incumbent_objective + 1e-12)
    promote = loss_ok and trading_ok and improved
    return {"promote": promote, "candidate": candidate, "incumbent": incumbent,
            "candidate_objective": candidate_objective, "incumbent_objective": incumbent_objective,
            "logloss_not_worse": loss_ok, "trading_not_worse": trading_ok,
            "reason": "동일 보정 표본에서 확률/거래 지표 악화 없이 개선" if promote else
                      "확률/거래 지표 악화 또는 유의한 개선 없음; 현재 모델 유지"}


def assess_promotion(config, frames, probabilities, rules):
    total = sum(len(v) for v in frames.values())
    candidate = {"logloss": sum(len(frames[m]) * logloss(frames[m].label, probabilities[m]) for m in frames) / total}
    candidate_portfolio = backtest_portfolio(frames, probabilities, rules, config["paper"])
    candidate.update({k: candidate_portfolio[k] for k in ("net_return", "MDD")})
    eligibility = trading_eligibility(candidate_portfolio, config["training"]["min_closed_trades"])
    periods = {m: {"start": iso(v.index.min()), "end": iso(v.index.max()), "rows": len(v)} for m, v in frames.items()}
    pointer = Path(config["models_dir"]) / "current.json"
    if not pointer.exists():
        return {"promote": eligibility["eligible"], "reason": "후속 평가 통과 최초 후보" if eligibility["eligible"] else
                "후속 평가 수익/완료 거래 근거 부족; 운용 보류", "candidate": candidate,
                "eligibility": eligibility, "incumbent_run": None, "comparison_periods": periods}
    try:
        try:
            current_dir, current = load_current(config)
        except MarketUniverseTransitionError as exc:
            if any(set(values) != set(MARKETS) for values in (frames, probabilities, rules)):
                raise ValueError("종목 전환 후보는 다섯 시장 모두의 후속 평가가 필요합니다") from exc
            current_dir, current = None, {"run": exc.run}
            guard = "legacy_four_market_universe_requires_revalidation"
        else:
            guard = deployment_guard(current)
        if guard is not None:
            # 운용할 수 없는 이전 모델의 확률 손실로 새 후보를 기각하지 않습니다.
            return {"promote": eligibility["eligible"], "candidate": candidate,
                    "eligibility": eligibility, "comparison_baseline": "cash",
                    "incumbent": {"net_return": 0.0, "MDD": 0.0, "logloss": None},
                    "candidate_objective": eligibility["objective"], "incumbent_objective": 0.0,
                    "probability_comparable": False, "excluded_incumbent_run": current.get("run"),
                    "excluded_incumbent_reason": guard, "incumbent_run": current.get("run"),
                    "comparison_periods": periods,
                    "reason": "운용 불가 이전 모델 제외; 현금 기준선 통과" if eligibility["eligible"] else
                              "후속 평가 수익/완료 거래 근거 부족; 운용 보류"}
        current_config = current["config"]
        same_labels = (str(current_config["base_timeframe"]) == str(config["base_timeframe"]) and
                       all(current_config["training"][k] == config["training"][k] for k in ("horizon", "label_threshold")))
        current_probs = {m: predict_package(current_dir, current, m, frame) for m, frame in frames.items()}
        incumbent = {"logloss": sum(len(frames[m]) * logloss(frames[m].label, current_probs[m]) for m in frames) / total
                     if same_labels else None}
        current_rules = {m: {**current["packages"][m]["rule"],
                             "net_label_threshold": current_config["training"]["label_threshold"],
                             "label_cost_factor": roundtrip_cost_factor(current_config["paper"])} for m in frames}
        current_portfolio = backtest_portfolio(frames, current_probs, current_rules, config["paper"])
        incumbent.update({k: current_portfolio[k] for k in ("net_return", "MDD")})
        if same_labels:
            result = promotion_decision(candidate, incumbent)
        else:
            candidate_objective = candidate["net_return"] - 0.5 * candidate["MDD"]
            incumbent_objective = incumbent["net_return"] - 0.5 * incumbent["MDD"]
            result = {"promote": candidate_objective > incumbent_objective + 1e-12,
                      "candidate": candidate, "incumbent": incumbent,
                      "candidate_objective": candidate_objective, "incumbent_objective": incumbent_objective,
                      "logloss_not_worse": None, "reason": "라벨 의미가 달라 확률 오차는 비교하지 않고 동일 후속 구간 거래 비교"}
        result["eligibility"] = eligibility
        result["probability_comparable"] = same_labels
        if not eligibility["eligible"]:
            result.update(promote=False, reason="후속 평가 수익/완료 거래 근거 부족; 운용 보류")
        result.update(incumbent_run=current["run"], comparison_periods=periods,
                      scope="정책 선정 이후 validation 구간; 잠금 시험 미사용; 반복 연구로 독립 일반화 보장은 아님")
        return result
    except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError, ImportError) as exc:
        return {"promote": False, "candidate": candidate, "reason": "현재 모델 비교 실패; 포인터 유지",
                "eligibility": eligibility,
                "comparison_error": {"type": type(exc).__name__, "message": str(exc)}, "comparison_periods": periods}


@contextlib.contextmanager
def market_feature_loader(store, config, start, end):
    # 완료 결과도 다음 시장 하나만 보존하며 DB 연결은 각 read 내부에서 닫습니다.
    markets = iter(config["markets"])
    pending = None
    expected = None

    def prepare(market):
        started = time.monotonic()
        frame = load_market_features(store, market, config, start, end)
        return frame, time.monotonic() - started

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="upbit-features") as executor:
        def schedule():
            nonlocal pending, expected
            expected = next(markets, None)
            pending = executor.submit(prepare, expected) if expected is not None else None

        def read(market):
            nonlocal pending
            if pending is None or market != expected:
                raise ValueError("특징 선행 준비의 시장 순서가 다릅니다")
            current, pending = pending, None
            waited = time.monotonic()
            frame, seconds = current.result()
            wait_seconds = time.monotonic() - waited
            schedule()
            return frame, {"feature_seconds": seconds, "feature_wait_seconds": wait_seconds}

        schedule()
        try:
            yield read
        finally:
            if pending is not None and not pending.cancel():
                active_error = sys.exc_info()[1]
                try:
                    pending.result()
                except Exception:
                    if active_error is None:
                        raise
                    # 현재 시장 오류를 보존하면서 동시에 발생한 선행 오류 원문도 남깁니다.
                    active_error.add_note("선행 특징 준비 실패:\n" + traceback.format_exc())


def train_cycle(config, final=False, live=False):
    started = time.monotonic()
    started_at = iso(None)
    if final and live:
        raise ValueError("실시간 학습은 잠금 시험을 평가할 수 없습니다")
    config = copy.deepcopy(config)
    external_resolution = None
    if "external" in config:
        config["external"], external_resolution = import_module("3차.external_features").resolve_for_training(
            config["external"], ROOT, utc(started_at))
    audit = ExperimentAudit(Path(config["data_dir"]) / "experiment_audit.sqlite3")
    consumed = audit.consumed_periods()
    research = {"scope": "명시 설정 단일 연구", "candidate_id": "explicit", "sequence": None}
    if live:
        config, research = next_research_candidate(config)
    plan = experiment_plan(config, live, consumed=consumed)
    if final and utc(None) < plan["holdout"][1]:
        raise ValueError("잠금 시험 기간이 끝나지 않았습니다")
    if final:
        audit.assert_final_available(*plan["holdout"])
    run = utc(None).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]

    def update_progress(**fields):
        progress(run=run, started_at=started_at, elapsed_seconds=time.monotonic() - started, **fields)

    directory = Path(config["models_dir"]) / "versions" / run
    directory.mkdir(parents=True, exist_ok=False)
    store = CandleStore(Path(config["data_dir"]) / "candles.sqlite3")
    tr = config["training"]
    columns, packages, validation_frames, validation_probs, rules = None, {}, {}, {}, {}
    total = len(config["markets"]) * len(tr["candidate_depths"])
    completed = 0
    report_plan = {k: [iso(a), iso(b)] for k, (a, b) in plan.items()}
    report = {"run": run, "updated_at": iso(None), "paper_only": True,
              "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "config": config, "periods": report_plan, "external_resolution": external_resolution,
              "locked_test": {"status": "미평가", "used_for_selection": False},
              "research": {"schema_version": 2, **research, "status": "running",
                           "protected_periods": [[iso(a), iso(b)] for a, b in protected_periods(config, consumed)],
                           "stages": {"train": "GPU 적합", "tune": "조기 종료·앙상블",
                                      "calibration": "확률 보정만", "rule": "정책 선정만", "validation": "후속 승격 평가만"},
                           "fold_id": stable_hash(report_plan),
                           "periods": report_plan},
              "markets": {}, "training": {"status": "학습 중", "gpu_required": True, "started_at": started_at}, "rounds": []}
    with market_feature_loader(store, config, plan["train"][0], plan["validation"][1]) as read_market:
        for market in config["markets"]:
            update_progress(status="running", stage="features", market=market, candidate=None,
                     completed_candidates=completed, total_candidates=total, error=None)
            raw, stage_times = read_market(market)
            stage_started = time.monotonic()
            labelled = label_frame(raw, tr["horizon"], tr["label_threshold"], config["paper"]["fee"], config["paper"]["slippage"])
            cols = feature_columns(labelled)
            if columns is not None and cols != columns:
                raise ValueError("코인별 피처 스키마가 다릅니다")
            columns = cols
            dropped_features = int(labelled[cols].isna().any(axis=1).sum())
            labelled = labelled.dropna(subset=cols + ["label", "label_end"])
            split = research_splits(labelled, plan, config, consumed)
            forbidden = protected_periods(config, consumed)
            for name in split:
                for left, right in zip([None] + [b for a, b in forbidden], [a for a, b in forbidden] + [None]):
                    segment = split[name]
                    segment = segment[segment.index >= left] if left is not None else segment[segment.label_end < right]
                    if right is not None:
                        segment = segment[segment.label_end < right]
                    if len(segment):
                        audit.expose(run, segment.index.min(), segment.label_end.max(), name)
            if set(split["train"].label.astype(int)) != {0, 1, 2}:
                raise ValueError(f"{market}: 학습 구간에 세 클래스가 모두 필요합니다")
            models, gpu_evidence, tune_probs, candidate_names = [], [], [], []
            stage_times["label_split_audit_seconds"] = time.monotonic() - stage_started
            fit_stage_started = time.monotonic()
            with gpu_matrix_scope():
                for idx, depth in enumerate(tr["candidate_depths"]):
                    def fit_progress(rounds, loss):
                        update_progress(status="running", stage="fit", market=market, candidate=idx + 1,
                                        completed_candidates=completed, total_candidates=total, error=None,
                                        rows=len(split["train"]), tune_rows=len(split["tune"]), features=len(cols),
                                        current_round=rounds, target_rounds=tr["rounds"], validation_logloss=loss,
                                        learning_rate=TRAINING_ETA)
                    fit_progress(0, None)
                    model, evidence = fit_gpu(split["train"], split["tune"], cols, depth, config, on_progress=fit_progress)
                    filename = market + f"-{idx}.ubj"
                    model.save_model(directory / filename)
                    models.append(model)
                    gpu_evidence.append(evidence)
                    report["rounds"].append({"market": market, "candidate": idx + 1, **evidence})
                    candidate_names.append(filename)
                    tune_probs.append(predict_booster(model, split["tune"], cols, evidence["best_iteration"]))
                    completed += 1
            stage_times["fit_and_tune_predict_seconds"] = time.monotonic() - fit_stage_started
            evaluation_started = time.monotonic()
            weights = ensemble_weights(tune_probs, split["tune"].label)
            calibration = split["calibration"]
            update_progress(status="running", stage="calibration", market=market, candidate=None,
                            completed_candidates=completed, total_candidates=total, error=None,
                            rows=len(calibration), features=len(cols))
            mixed = sum(w * predict_booster(m, calibration, cols, e["best_iteration"]) for w, m, e in zip(weights, models, gpu_evidence))
            temperature = min((0.75, 1.0, 1.25, 1.5, 2.0), key=lambda t: logloss(calibration.label, temperature_probs(mixed, t)))
            calibrated = temperature_probs(mixed, temperature)
            policy_frame, validation = split["rule"], split["validation"]

            def probabilities(frame):
                return temperature_probs(sum(w * predict_booster(m, frame, cols, e["best_iteration"])
                                             for w, m, e in zip(weights, models, gpu_evidence)), temperature)

            policy_probs, evaluated = probabilities(policy_frame), probabilities(validation)
            rule = tune_rule(policy_frame, policy_probs, config["paper"], market, tr["min_closed_trades"],
                             tr["horizon"] * minutes(config["base_timeframe"]))
            rule.update(net_label_threshold=tr["label_threshold"], label_cost_factor=roundtrip_cost_factor(config["paper"]))
            trials = rule.pop("trials")
            majority = int(split["train"].label.astype(int).value_counts().idxmax())
            metrics = classification_metrics(validation.label, evaluated, majority)
            metrics.update(start=iso(validation.index.min()), end=iso(validation.index.max()))
            diagnostics = {**probability_diagnostics(evaluated, rule), "scope": "validation_after_policy_selection",
                           "rule_trials": trials, "weighting": training_weights(split["train"], tr)[1],
                           "calibration": classification_metrics(calibration.label, calibrated, majority),
                           "policy_selection": probability_diagnostics(policy_probs, rule),
                           "feature_rows_dropped": dropped_features, "policy_eligible": rule["eligible"]}
            stage_times["calibration_policy_validation_seconds"] = time.monotonic() - evaluation_started
            report["markets"][market] = {"validation": metrics, "splits": {
                k: {"rows": len(v), "start": iso(v.index.min()), "end": iso(v.index.max()),
                    "class_distribution": np.bincount(v.label.astype(int), minlength=3).tolist()} for k, v in split.items()},
                "gpu": gpu_evidence, "rule": rule, "weights": weights.tolist(), "temperature": temperature,
                "diagnostics": diagnostics, "timings": stage_times}
            packages[market] = {"models": candidate_names, "gpu": gpu_evidence, "weights": weights.tolist(),
                                "temperature": temperature, "rule": rule, "majority_class": majority}
            validation_frames[market], validation_probs[market], rules[market] = validation, evaluated, rule
            del raw, labelled, split, models
    report["validation"] = {"scope": "정책 선정 이후 시간순 후속 평가; 반복 연구이며 잠금 시험 아님", "portfolio":
                             backtest_portfolio(validation_frames, validation_probs, rules, config["paper"])}
    report["promotion"] = assess_promotion(config, validation_frames, validation_probs, rules)
    report["research"].update(status="complete", deployment=report["promotion"]["eligibility"],
                              promoted=report["promotion"]["promote"])
    manifest = {"schema_version": 2, "execution_policy": EXECUTION_POLICY,
                "deployment": report["promotion"]["eligibility"],
                "run": run, "feature_columns": columns, "packages": packages, "config": config,
                "source_hash": report["source_hash"], "paper_only": True,
                "plan": report["periods"], "created_at": iso(None), "external_resolution": external_resolution}
    atomic_json(directory / "manifest.json", manifest)
    manifest_hash = stable_hash(manifest)
    if final:
        # 이 지점 이후에만 잠금 시험 봉을 읽고 예측합니다. 중단된 시도도 소비 이력을 보존합니다.
        audit.reserve_final(run, *plan["holdout"], manifest_hash)
        test_frames, test_probs, test_metrics = {}, {}, {}
        for market in config["markets"]:
            update_progress(status="running", stage="locked_test", market=market, candidate=None,
                     completed_candidates=completed, total_candidates=total, error=None)
            frame = load_market_features(store, market, config, plan["holdout"][0], plan["holdout"][1])
            frame = label_frame(frame, tr["horizon"], tr["label_threshold"], config["paper"]["fee"], config["paper"]["slippage"])
            test = chronological_split(frame, {"test": plan["holdout"]}, tr["gap_bars"], config["base_timeframe"])["test"]
            test = test.dropna(subset=columns + ["label", "label_end"])
            if len(test) < tr["min_split_rows"]:
                raise ValueError(f"{market}: 잠금 시험 표본 부족; 이번 평가 시도는 소비됨")
            probs = predict_package(directory, manifest, market, test)
            test_frames[market], test_probs[market] = test, probs
            metric = classification_metrics(test.label, probs, packages[market]["majority_class"])
            metric.update(start=iso(test.index.min()), end=iso(test.index.max()))
            test_metrics[market] = metric
        all_y = np.concatenate([v.label.to_numpy(dtype=int) for v in test_frames.values()])
        all_p = np.concatenate(list(test_probs.values()))
        aggregate = classification_metrics(all_y, all_p)
        aggregate["majority_baseline"] = float(sum(len(test_frames[m]) * test_metrics[m]["majority_baseline"] for m in test_frames) / len(all_y))
        report["locked_test"] = {"status": "최종 1회 평가 완료", "used_for_selection": False,
                                  "metrics": aggregate, "markets": test_metrics, "manifest_hash": manifest_hash,
                                  "portfolio": backtest_portfolio(test_frames, test_probs, rules, config["paper"])}
        audit.finish(run)
    report["training"].update(status="완료", completed_candidates=completed, total_candidates=total,
                              completed_at=iso(None), duration_seconds=time.monotonic() - started,
                              rows=sum(r["splits"]["train"]["rows"] for r in report["markets"].values()),
                              features=len(columns))
    report["class_distribution"] = {m: r["validation"]["class_distribution"] for m, r in report["markets"].items()}
    atomic_json(Path(config["reports_dir"]) / "experiments" / (run + ".json"), report)
    atomic_json(Path(config["reports_dir"]) / "training.json", report)
    # 다섯 코인의 모델과 보고서가 모두 완성된 뒤 포인터 하나만 바꿉니다. 이전 버전은 삭제하지 않습니다.
    if report["promotion"]["promote"]:
        atomic_json(Path(config["models_dir"]) / "current.json", {"run": run, "manifest_hash": manifest_hash})
    update_progress(status="complete", stage="complete", market=None, candidate=None,
                    completed_candidates=completed, total_candidates=total, error=None,
                    rows=report["training"]["rows"], features=len(columns),
                    promoted=report["promotion"]["promote"])
    return report


def predict_package(directory, manifest, market, frame):
    import xgboost as xgb
    package = manifest["packages"][market]
    mixed = np.zeros((len(frame), 3))
    for filename, weight, evidence in zip(package["models"], package["weights"], package["gpu"]):
        model = xgb.Booster()
        model.load_model(Path(directory) / filename)
        mixed += weight * predict_booster(model, frame, manifest["feature_columns"], evidence["best_iteration"])
    return temperature_probs(mixed, package["temperature"])


class MarketUniverseTransitionError(ValueError):
    """검사를 통과한 기존 네 시장 모델도 다섯 시장의 운용에는 사용할 수 없습니다."""

    def __init__(self, run):
        self.run = run
        super().__init__("완성 모델의 기존 네 코인 구성은 DOGE 포함 다섯 코인 운용에 맞지 않습니다")


def load_current(config):
    pointer = read_json(Path(config["models_dir"]) / "current.json")
    if Path(pointer["run"]).name != pointer["run"]:
        raise ValueError("모델 포인터 경로 오류")
    directory = Path(config["models_dir"]) / "versions" / pointer["run"]
    manifest = read_json(directory / "manifest.json")
    if stable_hash(manifest) != pointer["manifest_hash"] or not manifest["paper_only"]:
        raise ValueError("완성 모델 manifest가 일치하지 않습니다")
    package_markets = set(manifest.get("packages", {}))
    legacy_markets = {"KRW-BTC", "KRW-SOL", "KRW-ETH", "KRW-XRP"}
    model_config = manifest.get("config")
    configured_markets = model_config.get("markets") if isinstance(model_config, dict) else None
    known_transition = (package_markets == legacy_markets and isinstance(configured_markets, list)
                        and len(configured_markets) == len(legacy_markets)
                        and set(configured_markets) == legacy_markets
                        and len(config.get("markets", ())) == len(MARKETS)
                        and set(config.get("markets", ())) == set(MARKETS)
                        and manifest["paper_only"] is True)
    if manifest.get("run") != pointer["run"] or (package_markets != set(MARKETS) and not known_transition):
        raise ValueError("완성 모델의 실행 ID/다섯 코인 구성이 일치하지 않습니다")
    columns = manifest.get("feature_columns")
    if not isinstance(columns, list) or not columns or len(set(columns)) != len(columns):
        raise ValueError("완성 모델의 피처 목록이 유효하지 않습니다")
    external_columns = [c for c in columns if c.startswith("ext_")]
    if isinstance(model_config, dict) and "external" in model_config:
        external = import_module("3차.external_features")
        settings = external.validate_config(model_config["external"], ROOT)
        if type(settings["enabled"]) is not bool:
            raise ValueError("완성 모델의 외부 설정이 bool로 동결되지 않았습니다")
        if external_columns != (external.FEATURE_COLUMNS if settings["enabled"] else []):
            raise ValueError("완성 모델의 외부 특징 스키마가 설정과 다릅니다")
    elif external_columns:
        raise ValueError("외부 특징 모델에 동결된 외부 설정이 없습니다")
    for market, package in manifest["packages"].items():
        if market not in MARKETS:
            raise ValueError("지원하지 않는 모델 시장")
        if not package["models"] or not len(package["models"]) == len(package["weights"]) == len(package["gpu"]):
            raise ValueError("완성 모델의 후보/가중치/적합 증거 수가 다릅니다")
        if any(not math.isfinite(w) or w < 0 for w in package["weights"]) or not math.isclose(sum(package["weights"]), 1, abs_tol=1e-6):
            raise ValueError("완성 모델의 확률 가중치가 유효하지 않습니다")
        for filename in package["models"]:
            if Path(filename).name != filename or not (directory / filename).is_file():
                raise ValueError("완성 모델 파일이 없습니다")
    # 손상 검사를 모두 마친 정확한 기존 네 시장만 승격 비교에서 현금 기준선과 구분합니다.
    if known_transition:
        raise MarketUniverseTransitionError(manifest["run"])
    return directory, manifest


def paper_signals(config, store, portfolio, now):
    expected_errors = (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError, ImportError, sqlite3.Error)
    signals = {m: 1 for m in config["markets"]}
    stops = dict(portfolio.state.get("stop_fractions", {}))
    signal_times, errors = {}, {}
    diagnostics = {}
    portfolio.state["inference_diagnostics"] = diagnostics
    portfolio.state["entry_expiries"] = {}
    portfolio.state["entry_cost_limits"] = {}
    portfolio.state["entry_cost_baselines"] = {}
    portfolio.state["risk_horizon_minutes"] = {m: minutes(config["base_timeframe"]) * config["training"]["horizon"]
                                               for m in config["markets"]}
    portfolio.state["deployment_guard"] = None
    try:
        directory, manifest = load_current(config)
    except expected_errors as exc:
        errors["model"] = {"type": type(exc).__name__, "message": str(exc), "stage": "load_current"}
        portfolio.state["deployment_guard"] = "model_load_error"
        diagnostics.update({m: {"reason": "model_load_error", "checked_at": iso(now)} for m in config["markets"]})
        return signals, stops, signal_times, errors, None
    guard = deployment_guard(manifest)
    portfolio.state["deployment_guard"] = guard
    if guard:
        diagnostics.update({m: {"reason": guard, "model_run": manifest["run"], "checked_at": iso(now)} for m in config["markets"]})
        return signals, stops, signal_times, errors, manifest["run"]
    for market in config["markets"]:
        try:
            model_config = manifest["config"]
            portfolio.state["risk_horizon_minutes"][market] = minutes(model_config["base_timeframe"]) * model_config["training"]["horizon"]
            portfolio.state["entry_cost_limits"][market] = model_config["training"]["label_threshold"]
            portfolio.state["entry_cost_baselines"][market] = roundtrip_cost_factor(model_config["paper"])
            start = now - pd.Timedelta(minutes=max(minutes(tf) for tf in model_config["timeframes"]) * 28 + 2880)
            frame = load_market_features(store, market, model_config, start, now)
            latest_raw = iso(frame.index[-1]) if not frame.empty else None
            missing = [c for c in manifest["feature_columns"] if not frame.empty and pd.isna(frame[c].iloc[-1])]
            if model_config.get("external", {}).get("enabled") is True and not frame.empty:
                available = frame.get("available_external")
                external_time = available.iloc[-1] if available is not None else pd.NaT
                age = now - utc(external_time) if not pd.isna(external_time) else None
                if (any(c.startswith("ext_") for c in missing) or age is None
                        or age < pd.Timedelta(0) or age > pd.Timedelta(minutes=5)):
                    diagnostics[market] = {"checked_at": iso(now), "model_run": manifest["run"],
                                           "latest_raw_bar": latest_raw, "missing_latest_features": missing,
                                           "reason": "external_features_unavailable"}
                    continue
            frame = frame.dropna(subset=manifest["feature_columns"])
            diagnostics[market] = {"checked_at": iso(now), "model_run": manifest["run"],
                                   "latest_raw_bar": latest_raw, "missing_latest_features": missing,
                                   "base_timeframe": model_config["base_timeframe"]}
            if frame.empty:
                diagnostics[market]["reason"] = "no_complete_features"
                continue
            age = (now - frame.index[-1]).total_seconds()
            diagnostics[market].update(signal_time=iso(frame.index[-1]), feature_age_seconds=age)
            if age < 0 or age > minutes(model_config["base_timeframe"]) * 120:
                diagnostics[market]["reason"] = "stale_features" if age >= 0 else "future_features"
                continue
            signal_time = iso(frame.index[-1])
            if portfolio.state.get("signal_times", {}).get(market) == signal_time:
                diagnostics[market]["reason"] = "duplicate_signal_bar"
                continue
            probs = predict_package(directory, manifest, market, frame.iloc[-1:])
            signal = int(signals_from_proba(probs, manifest["packages"][market]["rule"])[0])
            vol = float(frame[str(model_config["base_timeframe"]) + "_volatility"].iloc[-1])
            stop = volatility_stop_fraction(vol, config["paper"]["stop_loss"])
            signals[market] = signal
            rule = manifest["packages"][market]["rule"]
            diagnostics[market].update(probabilities=probs[0].tolist(), argmax=int(probs[0].argmax()),
                                       signal=signal, confidence=rule["confidence"], edge=rule["edge"],
                                       reason="no_trade_policy" if rule["confidence"] > 1 else
                                              "model_neutral_or_threshold" if signal == 1 else "signal_ready")
            signal_times[market] = signal_time
            expiry = utc(signal_time) + pd.Timedelta(minutes=minutes(model_config["base_timeframe"]) * model_config["training"]["horizon"])
            portfolio.state["entry_expiries"][market] = iso(expiry)
            stops[market] = stop
        except expected_errors as exc:
            signals[market] = 1
            errors[market] = {"type": type(exc).__name__, "message": str(exc), "stage": "inference"}
            diagnostics[market] = {**diagnostics.get(market, {}), "reason": "inference_error", "error": errors[market]}
    return signals, stops, signal_times, errors, manifest["run"]


def paper_cycle(config, now=None):
    now = utc(now)
    paper_dir = ROOT / "work/paper"
    settings = config["paper"]
    if now < utc(settings["start_time"]):
        result = {"status": "예정 시작 대기", "start_time": iso(settings["start_time"]), "updated_at": iso(now), "paper_only": True}
        atomic_json(paper_dir / "status.json", result)
        return result
    store = CandleStore(Path(config["data_dir"]) / "candles.sqlite3")
    with connect(paper_dir / "ledger.sqlite3") as db:
        db.executescript("CREATE TABLE IF NOT EXISTS checkpoint(id INTEGER PRIMARY KEY,payload TEXT);"
                         "CREATE TABLE IF NOT EXISTS trades(event_id TEXT PRIMARY KEY,payload TEXT);"
                         "CREATE TABLE IF NOT EXISTS equity_history(timestamp TEXT PRIMARY KEY,equity REAL,cash REAL,net_return REAL,drawdown REAL,payload TEXT);"
                         "CREATE TABLE IF NOT EXISTS inference_errors(timestamp TEXT,market TEXT,payload TEXT,PRIMARY KEY(timestamp,market));")
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT payload FROM checkpoint WHERE id=1").fetchone()
        portfolio = Portfolio(settings, json.loads(row[0]) if row else None)
        signals, stops, signal_times, errors, model_run = paper_signals(config, store, portfolio, now)
        # 전처리/추론이 끝난 뒤 새 호가를 읽어 오래된 가격으로 소급 체결하지 않습니다.
        payload = public_get(store, "orderbook", {"markets": ",".join(config["markets"])}, config["collection"])
        observed_at = utc(None)
        quotes = {q["market"]: q for q in payload}
        if set(quotes) != set(config["markets"]):
            raise ValueError("다섯 코인의 공개 호가가 모두 필요합니다")
        prices, books, bids = {}, {}, {}
        for market, quote in quotes.items():
            quote_time = pd.Timestamp(quote["timestamp"], unit="ms", tz="UTC")
            age = (observed_at - quote_time).total_seconds()
            if age > settings["max_quote_age_seconds"] or age < -5:
                raise ValueError(f"호가 시각이 유효하지 않습니다: {market}, age={age}")
            books[market] = quote["orderbook_units"]
            if not books[market]:
                raise ValueError(f"호가 잔량이 없습니다: {market}")
            prices[market] = (float(books[market][0]["ask_price"]) + float(books[market][0]["bid_price"])) / 2
            bids[market] = float(books[market][0]["bid_price"])
        # 호가 관측 시점의 시장가 모의체결입니다. 과거 다음 봉 시가로 소급 체결하지 않습니다.
        events = portfolio.process(observed_at, prices, signals, books=books, mark_prices=bids,
                                   stop_fractions=stops, allow_buys=not errors and not portfolio.state.get("deployment_guard"),
                                   entry_expiries=portfolio.state.get("entry_expiries"))
        for market, diagnostic in portfolio.state.get("inference_diagnostics", {}).items():
            diagnostic["entry_filter"] = portfolio.state.get("decision_diagnostics", {}).get(market)
        portfolio.state.setdefault("signal_times", {}).update(signal_times)
        model_error = "; ".join(f"{m}: {e['type']}: {e['message']}" for m, e in errors.items()) or None
        portfolio.state.update(updated_at=iso(observed_at), status="실시간 공개 호가 모의투자", model_error=model_error,
                               inference_errors=errors, stop_fractions=stops,
                                entry_guard="추론 오류로 신규 진입 차단" if errors else portfolio.state.get("deployment_guard"),
                               model_run=model_run, prices=prices,
                               start_time=iso(settings["start_time"]), execution_mode="현재 공개 호가 잔량 소진·부분체결, 추가 슬리피지/양방향 수수료")
        for event in events:
            db.execute("INSERT OR IGNORE INTO trades VALUES(?,?)", (event["event_id"], json.dumps(event, ensure_ascii=False, allow_nan=False)))
        for market, error in errors.items():
            db.execute("INSERT OR IGNORE INTO inference_errors VALUES(?,?,?)", (iso(observed_at), market, json.dumps(error, ensure_ascii=False)))
        state = portfolio.state
        valuation = {"positions": state["positions"], "prices": bids, "initial_cash": state["initial_cash"],
                     "model_run": model_run, "paper_only": True}
        db.execute("INSERT OR IGNORE INTO equity_history VALUES(?,?,?,?,?,?)",
                   (iso(observed_at), state["equity"], state["cash"], state["net_return"], state["MDD"],
                    json.dumps(valuation, ensure_ascii=False, allow_nan=False)))
        db.execute("INSERT OR REPLACE INTO checkpoint VALUES(1,?)", (json.dumps(portfolio.state, ensure_ascii=False, allow_nan=False),))
    atomic_json(paper_dir / "state.json", portfolio.state)
    return portfolio.state


@contextlib.contextmanager
def process_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def build_parser():
    parser = argparse.ArgumentParser(description="공개 시세·GPU 학습·현물 모의 전용")
    parser.add_argument("--config", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    collector = sub.add_parser("collector")
    collector.add_argument("--backfill", action="store_true")
    collector.add_argument("--once", action="store_true")
    collector.add_argument("--pages", type=int)
    trainer = sub.add_parser("trainer")
    trainer.add_argument("--once", action="store_true")
    paper = sub.add_parser("paper")
    paper.add_argument("--once", action="store_true")
    paper.add_argument("--paper-start")
    experiment = sub.add_parser("experiment")
    experiment.add_argument("--final", action="store_true")
    sub.add_parser("check-gpu")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "check-gpu":
        print(json.dumps(check_gpu(), ensure_ascii=False))
        return 0
    if args.command == "experiment":
        with process_lock(ROOT / "work/service/trainer.lock"):
            result = train_cycle(config, final=args.final)
        print(json.dumps({"run": result["run"], "locked_test": result["locked_test"]}, ensure_ascii=False))
        return 0
    if args.command == "paper" and args.paper_start:
        config["paper"]["start_time"] = iso(args.paper_start)
    lock_name = "backfill" if args.command == "collector" and args.backfill else args.command
    with process_lock(ROOT / "work/service" / (lock_name + ".lock")):
        while True:
            if args.command == "collector":
                result = collect_cycle(config, args.backfill, args.pages)
                interval = config["collection"]["poll_seconds"]
                summary = {"datasets": len(result), "rows": sum(r["rows"] for r in result)}
            elif args.command == "trainer":
                result = train_cycle(config, live=True)
                interval, summary = config["training"]["retrain_seconds"], {"run": result["run"]}
            else:
                result = paper_cycle(config)
                interval = config["paper"]["poll_seconds"]
                summary = {k: result.get(k) for k in ("status", "equity", "net_return", "MDD", "model_error")}
            print(json.dumps({"at": iso(None), "command": args.command, **summary}, ensure_ascii=False), flush=True)
            if args.once:
                break
            if interval > 0:
                time.sleep(interval)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        import traceback
        details = {"at": iso(None), "argv": sys.argv, "cwd": str(Path.cwd()),
                   "error": repr(error), "traceback": traceback.format_exc(), "exit_code": 1}
        atomic_json(ROOT / "work/service/engine_error.json", details)
        if build_parser().parse_args().command in {"trainer", "experiment"}:
            progress(status="failed", stage="error", error=details["error"])
        print(details["traceback"], file=sys.stderr)
        raise SystemExit(1)
