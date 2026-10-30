"""외부 자료의 시점·단위·준비 판정과 기존 매매 경계를 검사합니다."""
import copy
import importlib
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import engine


NOW = pd.Timestamp("2026-10-30T00:00:00Z")
SYMBOLS = ("BTCUSDT", "SOLUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT")
SCHEMA = ("CREATE TABLE bars(symbol TEXT,open_time INTEGER,open REAL,high REAL,low REAL,"
          "close REAL,volume REAL,quote_volume REAL,trade_count INTEGER,taker_quote_volume REAL,"
          "source TEXT,PRIMARY KEY(symbol,open_time))")


def external():
    return importlib.import_module("3차.external_features")


def bars(count=60, start="2026-10-29T22:00:00Z"):
    index = pd.date_range(start, periods=count, freq="min")
    close = np.arange(count, dtype=float) + 100
    return pd.DataFrame({"symbol": "BTCUSDT", "open_time": index.as_unit("ns").asi8 // 1000000,
                         "open": close, "high": close + 1, "low": close - 1, "close": close,
                         "volume": 10., "quote_volume": 1000., "trade_count": 10,
                         "taker_quote_volume": 250., "source": "통제된 자료"})


def settings(enabled=True):
    return {"enabled": enabled, "db_path": "3차/work/data/external.sqlite3",
            "status_path": "3차/work/status.json"}


def status():
    first = int((NOW - pd.Timedelta(days=31)).timestamp() * 1000)
    last = int((NOW - pd.Timedelta(minutes=1)).timestamp() * 1000)
    return {"schema_version": 1, "db_schema": "binance_spot_v1", "training_ready": True, "updated_at": NOW.isoformat(),
            "verified_archive_bytes": 100, "target_archive_bytes": 30000000000,
            "unique_minute_rows": 100005,
            "markets": {s: {"rows": 20001, "first_open_ms": first, "last_open_ms": last}
                        for s in SYMBOLS}, "state": "ready", "error": None}


def make_db(root, ready=False):
    path = root / settings()["db_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute(SCHEMA)
        db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)")
        db.execute("INSERT INTO metadata VALUES('schema','binance_spot_v1')")
        if ready:
            times = [int((NOW - pd.Timedelta(days=31)).timestamp() * 1000)]
            times.extend(int(NOW.timestamp() * 1000) - n * 60000 for n in range(20000, 0, -1))
            for symbol in SYMBOLS:
                db.executemany("INSERT INTO bars VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                               ((symbol, t, 100., 101., 99., 100., 10., 1000., 10, 250., "fixture")
                                for t in times))
    return path


class ExternalConfigTests(unittest.TestCase):
    def test_default_and_old_manifest_keep_original_feature_path(self):
        config = engine.load_config()
        self.assertIn("external", config)
        self.assertIs(config["external"]["enabled"], False)
        config.pop("external")
        raw = bars().rename(columns={"quote_volume": "unused"})
        raw.index = pd.to_datetime(raw.open_time, unit="ms", utc=True)
        config.update(timeframes=[1], base_timeframe=1)
        with patch.object(engine.CandleStore, "read", return_value=raw[["open", "high", "low", "close", "volume"]]):
            frame = engine.load_market_features(engine.CandleStore.__new__(engine.CandleStore), "KRW-BTC", config,
                                                NOW - pd.Timedelta(days=1), NOW)
        self.assertFalse(any(c.startswith("ext_") for c in frame))

    def test_invalid_modes_unknown_fields_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in (1, 0, None, "true", "false", [], {}):
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    external().validate_config(settings(mode), root)
            for key in ("db_path", "status_path"):
                for path in ("../escape", "work/data/old.sqlite3", "3차/work/../../escape", "", "file:evil?mode=ro"):
                    candidate = settings()
                    candidate[key] = path
                    with self.subTest(key=key, path=path), self.assertRaises(ValueError):
                        external().validate_config(candidate, root)
            with self.assertRaises(ValueError):
                external().validate_config({**settings(), "unexpected": 1}, root)

    def test_auto_missing_status_freezes_false_without_touching_database(self):
        with tempfile.TemporaryDirectory() as directory:
            frozen, evidence = external().resolve_for_training(settings("auto"), Path(directory), NOW)
            self.assertIs(frozen["enabled"], False)
            self.assertFalse(evidence["training_ready"])

    def test_windows_rejects_bars_before_sqlite_open_even_when_status_says_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine.atomic_json(root / settings()["status_path"], status())
            with patch.object(external().sys, "platform", "win32"), patch.object(external().sqlite3, "connect") as connect:
                frozen, evidence = external().resolve_for_training(settings("auto"), root, NOW)
                self.assertIs(frozen["enabled"], False)
                with self.assertRaisesRegex(RuntimeError, "Linux"):
                    external().load_and_join(pd.DataFrame(index=pd.DatetimeIndex([NOW])), "KRW-BTC", settings(), root, NOW)
            connect.assert_not_called()


class ExternalRatioTests(unittest.TestCase):
    def test_closed_minute_backward_join_never_uses_future_or_price_levels(self):
        raw = bars()
        base = pd.DataFrame({"close": [100000., 100001., 100002.]},
                            index=pd.DatetimeIndex(["2026-10-29T22:30:30Z", "2026-10-29T23:02:00Z", "2026-10-29T23:03:00Z"]))
        result = external().join_features(base, external().ratio_features(raw, NOW))
        self.assertAlmostEqual(float(result.ext_return_1.iloc[0]), 129 / 128 - 1, places=7)
        self.assertEqual(result.available_external.iloc[0], pd.Timestamp("2026-10-29T22:30:00Z"))
        self.assertTrue(result.filter(like="ext_").iloc[-1].isna().all())
        self.assertEqual(result.close.tolist(), [100000., 100001., 100002.])
        scaled = raw.copy()
        scaled[["open", "high", "low", "close", "quote_volume", "taker_quote_volume"]] *= 1000
        pd.testing.assert_frame_equal(external().ratio_features(raw, NOW), external().ratio_features(scaled, NOW),
                                      check_exact=False, rtol=1e-6, atol=1e-7)
        self.assertFalse(any(c in result for c in ("ext_close", "ext_label", "ext_open")))

    def test_gaps_and_zero_activity_are_nan_not_zero_or_filled(self):
        raw = bars().drop(index=30)
        features = external().ratio_features(raw, NOW)
        self.assertTrue(pd.isna(features.loc[pd.Timestamp("2026-10-29T22:32:00Z"), "ext_return_1"]))
        self.assertTrue(pd.isna(features.loc[pd.Timestamp("2026-10-29T22:40:00Z"), "ext_volatility"]))
        quiet = bars()
        quiet[["volume", "quote_volume", "trade_count", "taker_quote_volume"]] = 0
        self.assertTrue(external().ratio_features(quiet, NOW).ext_taker_buy_ratio.isna().all())

    def test_wrong_units_duplicates_future_and_invalid_ranges_rejected(self):
        for column, value in (("open_time", 1793311200), ("open_time", 1793311200000000),
                              ("open_time", 1793311200001), ("high", 1), ("close", float("inf")),
                              ("volume", -1), ("trade_count", 1.5), ("taker_quote_volume", 1001),
                              ("open_time", int(NOW.timestamp() * 1000))):
            raw = bars()
            if column == "trade_count":
                raw[column] = raw[column].astype(float)
            raw.loc[5, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                external().ratio_features(raw, NOW)
        with self.assertRaises(ValueError):
            external().ratio_features(pd.concat([bars(), bars().iloc[:1]]), NOW)


@unittest.skipUnless(sys.platform == "linux", "외부 bars SQLite는 Linux 전용입니다")
class ExternalDatabaseTests(unittest.TestCase):
    def test_real_database_readiness_and_bool_freeze_not_json_assertion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_db(root, ready=True)
            engine.atomic_json(root / settings()["status_path"], status())
            frozen, evidence = external().resolve_for_training(settings("auto"), root, NOW)
            self.assertIs(frozen["enabled"], True)
            self.assertTrue(evidence["training_ready"])
            engine.atomic_json(root / settings()["status_path"], {**status(), "training_ready": False})
            still_frozen, _ = external().resolve_for_training(frozen, root, NOW)
            self.assertIs(still_frozen["enabled"], True)
            changed, _ = external().resolve_for_training(settings("auto"), root, NOW)
            self.assertIs(changed["enabled"], False)

    def test_fake_ready_empty_bad_schema_and_stale_status_do_not_enable_auto(self):
        for corruption in ("empty", "schema", "missing", "stale", "string_bool", "future", "count"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = make_db(root, ready=corruption not in ("empty", "schema", "missing"))
                payload = status()
                if corruption == "schema":
                    with sqlite3.connect(path) as db:
                        db.execute("ALTER TABLE bars RENAME TO old_bars")
                        db.execute("CREATE TABLE bars(symbol TEXT,open_time INTEGER)")
                elif corruption == "missing":
                    path.unlink()
                elif corruption == "stale":
                    payload["updated_at"] = (NOW - pd.Timedelta(minutes=6)).isoformat()
                elif corruption == "string_bool":
                    payload["training_ready"] = "true"
                elif corruption == "future":
                    payload["markets"]["BTCUSDT"]["last_open_ms"] = int(NOW.timestamp() * 1000)
                elif corruption == "count":
                    payload["unique_minute_rows"] = True
                engine.atomic_json(root / settings()["status_path"], payload)
                frozen, evidence = external().resolve_for_training(settings("auto"), root, NOW)
                self.assertIs(frozen["enabled"], False)
                self.assertFalse(evidence["training_ready"])

    def test_true_missing_empty_and_bad_schema_fail_instead_of_base_fallback(self):
        for corruption in ("empty", "schema", "missing"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = make_db(root)
                if corruption == "schema":
                    with sqlite3.connect(path) as db:
                        db.execute("ALTER TABLE bars RENAME TO old_bars")
                elif corruption == "missing":
                    path.unlink()
                base = pd.DataFrame(index=pd.DatetimeIndex([NOW]))
                with self.assertRaises((ValueError, OSError, sqlite3.Error)):
                    external().load_and_join(base, "KRW-BTC", settings(), root, NOW)

    def test_streaming_chunk_boundary_matches_single_frame_and_engine_wiring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = make_db(root)
            raw = bars(90)
            with sqlite3.connect(path) as db:
                db.executemany("INSERT INTO bars VALUES(?,?,?,?,?,?,?,?,?,?,?)", raw.itertuples(index=False, name=None))
            upbit = raw[["open", "high", "low", "close", "volume"]].copy()
            upbit.index = pd.to_datetime(raw.open_time.to_numpy(), unit="ms", utc=True)
            config = copy.deepcopy(engine.DEFAULT_CONFIG)
            config.update(timeframes=[1], base_timeframe=1, external=settings())
            base = engine.build_features({1: upbit}, 1)
            expected = external().join_features(base, external().ratio_features(raw, NOW))
            with patch.object(external(), "READ_CHUNK_ROWS", 19), patch.object(engine, "ROOT", root), \
                 patch.object(engine, "utc", return_value=NOW), patch.object(engine.CandleStore, "read", return_value=upbit):
                actual = engine.load_market_features(engine.CandleStore.__new__(engine.CandleStore), "KRW-BTC", config, NOW, NOW)
            pd.testing.assert_frame_equal(actual, expected)
            self.assertGreater(len(actual.dropna(subset=engine.feature_columns(actual))), 0)


class ExternalPaperTests(unittest.TestCase):
    def test_missing_latest_external_does_not_fall_back_to_recent_buy_but_risk_exit_continues(self):
        config = copy.deepcopy(engine.DEFAULT_CONFIG)
        config["external"] = settings()
        config["paper"]["start_time"] = "2026-10-01T00:00:00Z"
        manifest = {"run": "fixture", "schema_version": 2, "execution_policy": engine.EXECUTION_POLICY,
                    "deployment": {"eligible": True}, "config": config,
                    "feature_columns": ["ext_return_1"],
                    "packages": {m: {"rule": {"confidence": .5, "edge": .1}} for m in engine.MARKETS}}
        frame = pd.DataFrame({"ext_return_1": [.01, np.nan], "5_volatility": [.001, .001],
                              "available_external": [NOW - pd.Timedelta(minutes=5), pd.NaT]},
                             index=pd.DatetimeIndex([NOW - pd.Timedelta(minutes=5), NOW]))
        portfolio = engine.Portfolio(config["paper"])
        portfolio.process(NOW - pd.Timedelta(minutes=10), {"KRW-BTC": 100.}, {"KRW-BTC": 2})
        self.assertIn("KRW-BTC", portfolio.state["positions"])
        with patch.object(engine, "load_current", return_value=(engine.ROOT, manifest)), \
             patch.object(engine, "load_market_features", return_value=frame), \
             patch.object(engine, "predict_package", return_value=np.array([[.01, .01, .98]])):
            signals, stops, times, errors, run = engine.paper_signals(config, object(), portfolio, NOW)
        self.assertEqual(signals["KRW-BTC"], 1)
        self.assertEqual(portfolio.state["inference_diagnostics"]["KRW-BTC"]["reason"], "external_features_unavailable")
        events = portfolio.process(NOW, {"KRW-BTC": 90.}, signals, stop_fractions=stops, allow_buys=not errors)
        self.assertNotIn("KRW-BTC", portfolio.state["positions"])
        self.assertTrue(any(e["reason"] == "stop_loss" for e in events))


if __name__ == "__main__":
    unittest.main()
