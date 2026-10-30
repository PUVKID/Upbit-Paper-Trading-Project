"""외부 완료 분봉을 읽기 전용으로 검사하고 단위 없는 특징만 결합합니다."""
from __future__ import annotations

import contextlib
import json
from pathlib import Path, PureWindowsPath
import sqlite3
import sys
import time

import numpy as np
import pandas as pd


MARKET_SYMBOLS = {"KRW-BTC": "BTCUSDT", "KRW-SOL": "SOLUSDT", "KRW-ETH": "ETHUSDT",
                  "KRW-XRP": "XRPUSDT", "KRW-DOGE": "DOGEUSDT"}
DB_SCHEMA = "binance_spot_v1"
BAR_TYPES = {"symbol": "TEXT", "open_time": "INTEGER", "open": "REAL", "high": "REAL",
             "low": "REAL", "close": "REAL", "volume": "REAL", "quote_volume": "REAL",
             "trade_count": "INTEGER", "taker_quote_volume": "REAL", "source": "TEXT"}
FEATURE_COLUMNS = ["ext_return_1", "ext_return_5", "ext_return_15", "ext_return_24",
                   "ext_range", "ext_body", "ext_volume_ratio", "ext_quote_volume_ratio",
                   "ext_trade_count_ratio", "ext_volatility", "ext_taker_buy_ratio"]
READ_CHUNK_ROWS = 50000
MIN_DATASET_BYTES = 50000000000
MIN_OPEN_MS = 1483228800000
MAX_AGE = pd.Timedelta(minutes=5)
# 결측을 무제한 이월하지 않으며 최신 추론에서는 별도로 현재 시각 대비 5분도 검사합니다.
JOIN_TOLERANCE = pd.Timedelta(minutes=2)


class DatasetNotReady(RuntimeError):
    """신규 학습은 대기하되 기존 모델 운용을 바꾸지 않는 준비 상태입니다."""

    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__(evidence["reason"])


def _utc(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("외부 자료 시각은 시간대가 있는 UTC 시각이어야 합니다")
    return stamp.tz_convert("UTC")


def validate_config(settings, root):
    if not isinstance(settings, dict) or set(settings) - {"enabled", "db_path", "status_path", "wait_for_dataset", "target_bytes"}:
        raise ValueError("지원하지 않는 외부 자료 설정입니다")
    result = {"enabled": False, "db_path": "3차/work/data/external.sqlite3",
              "status_path": "3차/work/status.json", "wait_for_dataset": False,
              "target_bytes": MIN_DATASET_BYTES, **settings}
    mode = result["enabled"]
    if type(mode) is not bool and not (isinstance(mode, str) and mode == "auto"):
        raise ValueError("external.enabled는 false/true/auto만 허용합니다")
    if type(result["wait_for_dataset"]) is not bool or (result["wait_for_dataset"] and mode is False):
        raise ValueError("외부 자료 대기는 bool이며 외부 사용 설정과 함께 지정해야 합니다")
    if type(result["target_bytes"]) is not int or result["target_bytes"] < MIN_DATASET_BYTES:
        raise ValueError("외부 자료 목표는 최소 50,000,000,000바이트 정수여야 합니다")
    root = Path(root).resolve()
    boundary = (root / "3차" / "work").resolve()
    if not boundary.is_relative_to(root):
        raise ValueError("외부 자료 경계가 프로젝트를 벗어났습니다")
    for key in ("db_path", "status_path"):
        value = result[key]
        if not isinstance(value, str) or not value.strip() or "\x00" in value or value.startswith("file:"):
            raise ValueError(f"external.{key} 경로가 유효하지 않습니다")
        if sys.platform != "win32" and (PureWindowsPath(value).drive or "\\" in value):
            raise ValueError("다른 운영체제의 절대 경로를 외부 자료 경로로 사용하지 않습니다")
        target = (root / value).resolve()
        if not target.is_relative_to(boundary) or target == boundary:
            raise ValueError("외부 자료 경로는 프로젝트 3차/work 내부여야 합니다")
        # manifest는 호스트/컨테이너에 공통인 프로젝트 상대 경로만 보존합니다.
        result[key] = target.relative_to(root).as_posix()
    if result["db_path"] == result["status_path"]:
        raise ValueError("외부 DB와 상태 파일 경로는 달라야 합니다")
    return result


def _check_schema(db):
    kind = db.execute("SELECT type FROM sqlite_master WHERE name='bars'").fetchone()
    info = db.execute("PRAGMA table_info(bars)").fetchall()
    if (kind != ("table",) or [(r[1], r[2].upper()) for r in info] != list(BAR_TYPES.items())
            or {r[1]: r[5] for r in info if r[5]} != {"symbol": 1, "open_time": 2}):
        raise ValueError("외부 bars 스키마/기본키가 계약과 다릅니다")
    if db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone() != (DB_SCHEMA,):
        raise ValueError("외부 DB metadata 스키마 버전이 다릅니다")


def _query_deadline(db):
    deadline = time.monotonic() + 25
    db.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)


@contextlib.contextmanager
def _database(settings, root):
    # 테스트 DB를 포함하여 외부 bars는 Linux 밖에서 절대로 열지 않습니다.
    if sys.platform != "linux":
        raise RuntimeError("외부 bars SQLite는 Linux 도커 내부에서만 접근하십시오")
    path = Path(root) / settings["db_path"]
    if not path.is_file():
        raise FileNotFoundError("외부 bars DB가 없습니다")
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)
    _query_deadline(db)
    try:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        _check_schema(db)
        yield db
    finally:
        db.close()


def _nonnegative_int(value):
    return type(value) is int and value >= 0


def _minute(value, now_ms):
    return (_nonnegative_int(value) and MIN_OPEN_MS <= value <= now_ms - 60000
            and value % 60000 == 0)


def _status_ready(payload, now, target):
    if (not isinstance(payload, dict) or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != 1 or payload.get("db_schema") != DB_SCHEMA
            or payload.get("training_ready") is not True):
        raise ValueError("외부 상태 스키마/준비 표시가 유효하지 않습니다")
    updated = _utc(payload["updated_at"])
    if pd.Timestamp(payload["updated_at"]).utcoffset().total_seconds() != 0 or not pd.Timedelta(0) <= now - updated <= MAX_AGE:
        raise ValueError("외부 상태 갱신 시각이 오래되었거나 미래입니다")
    if any(not _nonnegative_int(payload.get(k)) for k in
           ("verified_archive_bytes", "target_archive_bytes", "ingested_archive_bytes", "unique_minute_rows")):
        raise ValueError("외부 상태의 용량/행 수가 유효하지 않습니다")
    if (payload["target_archive_bytes"] < target or payload["ingested_archive_bytes"] < target
            or payload["verified_archive_bytes"] < payload["ingested_archive_bytes"]):
        raise ValueError("검증·적재된 고유 외부 아카이브가 목표 용량에 미달합니다")
    if not isinstance(payload.get("state"), str) or not payload["state"] or payload.get("error") is not None:
        raise ValueError("외부 상태가 오류 또는 미확인 상태입니다")
    markets = payload.get("markets")
    if not isinstance(markets, dict) or set(markets) != set(MARKET_SYMBOLS.values()):
        raise ValueError("외부 상태의 다섯 종목이 일치하지 않습니다")
    now_ms = int(now.timestamp() * 1000)
    for values in markets.values():
        if (not isinstance(values, dict) or not _nonnegative_int(values.get("rows")) or values["rows"] < 20000
                or not all(_minute(values.get(k), now_ms) for k in ("first_open_ms", "last_open_ms"))
                or values["last_open_ms"] - values["first_open_ms"] < 30 * 86400000
                or now_ms - values["last_open_ms"] - 60000 > 300000):
            raise ValueError("외부 상태의 범위/완료 분봉 수/최신성이 부족합니다")
    if sum(v["rows"] for v in markets.values()) != payload["unique_minute_rows"]:
        raise ValueError("외부 상태의 고유 행 수 합계가 다릅니다")


def resolve_for_training(settings, root, now):
    resolved = validate_config(settings, root)
    now = _utc(now)
    evidence = {"requested": resolved["enabled"], "checked_at": now.isoformat()}
    if resolved["enabled"] != "auto" and not resolved["wait_for_dataset"]:
        return resolved, {**evidence, "training_ready": None, "reason": "명시 설정; 준비 검사 미실행"}
    evidence.update(verified_archive_bytes=0, ingested_archive_bytes=0,
                    target_archive_bytes=resolved["target_bytes"])
    try:
        payload = json.loads((Path(root) / resolved["status_path"]).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("verified_archive_bytes", "ingested_archive_bytes"):
                if _nonnegative_int(payload.get(key)):
                    evidence[key] = payload[key]
            evidence["reported_target_archive_bytes"] = payload.get("target_archive_bytes")
        _status_ready(payload, now, resolved["target_bytes"])
        now_ms = int(now.timestamp() * 1000)
        with _database(resolved, root) as db:
            metadata = dict(db.execute("SELECT key,value FROM metadata WHERE key IN ('target_archive_bytes','ingested_archive_bytes','verified_archive_bytes')"))
            for key in ("target_archive_bytes", "ingested_archive_bytes", "verified_archive_bytes"):
                value = metadata.get(key)
                if not isinstance(value, str) or not value.isascii() or not value.isdecimal() or int(value) < resolved["target_bytes"]:
                    raise ValueError(f"외부 DB metadata.{key}의 50GB 목표/적재 근거가 부족합니다")
            if (int(metadata["verified_archive_bytes"]) < int(metadata["ingested_archive_bytes"])
                    or int(metadata["ingested_archive_bytes"]) < payload["ingested_archive_bytes"]):
                raise ValueError("외부 DB와 상태의 검증/적재 용량 근거가 일치하지 않습니다")
            actual = {}
            for symbol, reported in payload["markets"].items():
                count, first, last = db.execute("SELECT COUNT(*),MIN(open_time),MAX(open_time) FROM bars WHERE symbol=?", (symbol,)).fetchone()
                recent = db.execute("SELECT COUNT(*) FROM bars WHERE symbol=? AND open_time>=? AND open_time<=?",
                                    (symbol, now_ms - 14 * 86400000, now_ms - 60000)).fetchone()[0]
                if (count < max(20000, reported["rows"]) or not _minute(first, now_ms) or not _minute(last, now_ms)
                        or last - first < 30 * 86400000 or now_ms - last - 60000 > 300000
                        or first > reported["first_open_ms"] or last < reported["last_open_ms"]
                        or recent < 14 * 1440 * .98):
                    raise ValueError(f"{symbol}: 실제 DB의 범위/봉 수/최신성/최근 14일 98% 커버리지 부족")
                latest = pd.read_sql_query("SELECT * FROM bars WHERE symbol=? ORDER BY open_time DESC LIMIT 25", db,
                                           params=(symbol,)).iloc[::-1]
                ratio_features(latest, now)
                actual[symbol] = {"rows": count, "first_open_ms": first, "last_open_ms": last,
                                  "recent_14d_rows": recent}
        resolved["enabled"] = True
        return resolved, {**evidence, "training_ready": True, "reason": "상태 및 실제 DB 검사 통과", "markets": actual}
    except (OSError, ValueError, TypeError, KeyError, OverflowError, sqlite3.Error, RuntimeError) as exc:
        raise DatasetNotReady({**evidence, "training_ready": False, "reason": str(exc),
                               "error_type": type(exc).__name__}) from exc


def ratio_features(raw, now):
    if raw.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="close_time"), dtype=np.float32)
    if set(raw.columns) != set(BAR_TYPES):
        raise ValueError("외부 분봉 열이 저장 계약과 다릅니다")
    times = raw.open_time.to_numpy()
    numeric = list(BAR_TYPES)[2:10]
    values = raw[numeric].to_numpy(dtype=np.float64)
    now_ms = int(_utc(now).timestamp() * 1000)
    if (not np.issubdtype(times.dtype, np.integer) or np.any(times < MIN_OPEN_MS)
            or np.any(times > now_ms - 60000) or np.any(times % 60000) or np.any(np.diff(times) <= 0)
            or not np.isfinite(values).all() or np.any(values[:, :4] <= 0)
            or np.any(values[:, 4:] < 0) or np.any(raw.high < raw[["open", "close", "low"]].max(axis=1))
            or np.any(raw.low > raw[["open", "close", "high"]].min(axis=1))
            or np.any(raw.trade_count % 1) or np.any(raw.taker_quote_volume > raw.quote_volume)
            or not raw.symbol.isin(MARKET_SYMBOLS.values()).all()
            or raw.symbol.nunique() != 1
            or not raw.source.map(lambda value: isinstance(value, str) and bool(value)).all()):
        raise ValueError("외부 분봉의 UTC 밀리초/완료 시각/단위/가격/수량 범위 오류")
    index = pd.to_datetime(times + 60000, unit="ms", utc=True)
    index.name = "close_time"
    data = raw[numeric].copy()
    data.index = index
    out = pd.DataFrame(index=index)
    for n in (1, 5, 15, 24):
        previous = data.close.reindex(index - pd.Timedelta(minutes=n)).to_numpy()
        out[f"ext_return_{n}"] = data.close.to_numpy() / previous - 1
    out["ext_range"] = (data.high - data.low) / data.close
    out["ext_body"] = (data.close - data.open) / data.open
    contiguous = pd.Series(times, index=index).diff(23).eq(23 * 60000)
    for source, name in (("volume", "volume"), ("quote_volume", "quote_volume"), ("trade_count", "trade_count")):
        average = data[source].rolling(24).mean().replace(0, np.nan)
        out[f"ext_{name}_ratio"] = (data[source] / average).where(contiguous)
    out["ext_volatility"] = out.ext_return_1.rolling(24).std()
    out["ext_taker_buy_ratio"] = data.taker_quote_volume / data.quote_volume.replace(0, np.nan)
    return out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).astype(np.float32)


def join_features(base, features):
    left = base.sort_index()
    if not isinstance(left.index, pd.DatetimeIndex) or left.index.tz is None or not left.index.is_unique:
        raise ValueError("외부 특징 결합 대상은 고유한 UTC 시간 인덱스여야 합니다")
    right = features.copy()
    right.index = right.index.tz_convert(left.index.tz).as_unit(left.index.unit)
    right["available_external"] = right.index
    return pd.merge_asof(left, right.sort_index(), left_index=True, right_index=True, direction="backward",
                         allow_exact_matches=True, tolerance=JOIN_TOLERANCE)


def load_and_join(base, market, settings, root, now):
    settings = validate_config(settings, root)
    if settings["enabled"] is not True:
        raise ValueError("외부 특징 사용 여부는 회차/manifest의 bool로 동결되어야 합니다")
    if market not in MARKET_SYMBOLS:
        raise ValueError("지원하지 않는 외부 시장 대응입니다")
    now = _utc(now)
    with _database(settings, root) as db:
        symbol = MARKET_SYMBOLS[market]
        if db.execute("SELECT 1 FROM bars WHERE symbol=? LIMIT 1", (symbol,)).fetchone() is None:
            raise ValueError(f"{symbol}: 외부 분봉 자료가 비어 있습니다")
        if base.empty:
            return join_features(base, ratio_features(pd.DataFrame(), now))
        left = base.sort_index()
        start = int((_utc(left.index.min()) - pd.Timedelta(minutes=26)).timestamp() * 1000)
        end = int(min(_utc(left.index.max()), now).timestamp() * 1000) - 60000
        cursor = db.execute("SELECT * FROM bars WHERE symbol=? AND open_time>=? AND open_time<=? ORDER BY open_time",
                            (symbol, start, end))
        joined = pd.DataFrame(np.nan, index=left.index, columns=FEATURE_COLUMNS, dtype=np.float32)
        joined["available_external"] = pd.Series(pd.NaT, index=left.index, dtype=left.index.dtype)
        carry = pd.DataFrame()
        previous_end = None
        features = ratio_features(pd.DataFrame(), now)
        while True:
            # CPU 특징 계산 시간은 다음 SQL 호출의 제한 시간을 소진하지 않습니다.
            _query_deadline(db)
            rows = cursor.fetchmany(READ_CHUNK_ROWS)
            if not rows:
                break
            chunk = pd.DataFrame.from_records(rows, columns=list(BAR_TYPES))
            if not carry.empty:
                chunk = pd.concat([carry, chunk], ignore_index=True)
            features = ratio_features(chunk, now)
            selected = left.index <= features.index[-1]
            if previous_end is not None:
                selected &= left.index > previous_end
            targets = left.index[selected]
            if len(targets):
                part = join_features(pd.DataFrame(index=targets), features)
                joined.loc[targets, :] = part
            previous_end = features.index[-1]
            carry = chunk.tail(24)
        targets = left.index if previous_end is None else left.index[left.index > previous_end]
        if len(targets):
            joined.loc[targets, :] = join_features(pd.DataFrame(index=targets), features)
    return left.join(joined)
