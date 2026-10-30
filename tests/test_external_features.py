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


def settings(enabled=True, wait=False):
    return {"enabled": enabled, "db_path": "3차/work/data/external.sqlite3",
            "status_path": "3차/work/status.json", "wait_for_dataset": wait, "target_bytes": 50000000000}


def status():
    first = int((NOW - pd.Timedelta(days=31)).timestamp() * 1000)
    last = int((NOW - pd.Timedelta(minutes=1)).timestamp() * 1000)
    return {"schema_version": 1, "db_schema": "binance_spot_v1", "training_ready": True, "updated_at": NOW.isoformat(),
            "verified_archive_bytes": 50000000000, "target_archive_bytes": 50000000000,
            "ingested_archive_bytes": 50000000000,
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
        db.executemany("INSERT INTO metadata VALUES(?,?)", (("target_archive_bytes", "50000000000"),
                                                          ("ingested_archive_bytes", "50000000000"),
                                                          ("verified_archive_bytes", "50000000000")))
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

    def test_auto_missing_status_blocks_training_without_base_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(external().DatasetNotReady) as error:
                external().resolve_for_training(settings("auto", True), Path(directory), NOW)
            self.assertFalse(error.exception.evidence["training_ready"])

    def test_windows_rejects_bars_before_sqlite_open_even_when_status_says_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine.atomic_json(root / settings()["status_path"], status())
            with patch.object(external().sys, "platform", "win32"), patch.object(external().sqlite3, "connect") as connect:
                with self.assertRaises(external().DatasetNotReady):
                    external().resolve_for_training(settings("auto"), root, NOW)
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
            with self.assertRaises(external().DatasetNotReady):
                external().resolve_for_training(settings("auto"), root, NOW)

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
                with self.assertRaises(external().DatasetNotReady) as error:
                    external().resolve_for_training(settings("auto"), root, NOW)
                self.assertFalse(error.exception.evidence["training_ready"])

    def test_old_30gb_status_or_insufficient_ingested_metadata_never_starts_new_training(self):
        for corruption in ("old_target", "small_ingested", "metadata", "coverage", "db_version"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = make_db(root, ready=True)
                payload = status()
                if corruption == "old_target":
                    payload["target_archive_bytes"] = 30000000000
                elif corruption == "small_ingested":
                    payload["ingested_archive_bytes"] = 18000000
                else:
                    with sqlite3.connect(path) as db:
                        if corruption == "metadata":
                            db.execute("UPDATE metadata SET value='18000000' WHERE key='ingested_archive_bytes'")
                        elif corruption == "db_version":
                            db.execute("UPDATE metadata SET value='old' WHERE key='schema'")
                        else:
                            db.execute("DELETE FROM bars WHERE symbol='BTCUSDT' AND open_time>? AND open_time<?",
                                       (int((NOW - pd.Timedelta(days=5)).timestamp() * 1000),
                                        int((NOW - pd.Timedelta(days=3)).timestamp() * 1000)))
                engine.atomic_json(root / settings()["status_path"], payload)
                with self.assertRaises(external().DatasetNotReady):
                    external().resolve_for_training(settings("auto", True), root, NOW)

    def test_bounded_poll_waits_until_ready_and_remains_stop_responsive(self):
        import threading
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = make_db(root, ready=True)
            payload = status()
            payload["ingested_archive_bytes"] = 18000000
            engine.atomic_json(root / settings()["status_path"], payload)
            config = copy.deepcopy(engine.DEFAULT_CONFIG)
            config["external"] = settings("auto", True)
            stop = threading.Event()
            waits = []

            def poll(seconds):
                waits.append(seconds)
                engine.atomic_json(root / settings()["status_path"], status())
                return False

            with patch.object(engine, "ROOT", root), patch.object(engine, "utc", return_value=NOW), \
                 patch.object(engine, "progress") as progress, patch.object(stop, "wait", side_effect=poll):
                engine.wait_for_training_data(config, stop)
            self.assertEqual(waits, [12])
            self.assertEqual(progress.call_args.kwargs["status"], "waiting_data")
            self.assertEqual(progress.call_args.kwargs["stage"], "download")
            self.assertEqual(progress.call_args.kwargs["ingested_archive_bytes"], 18000000)
            self.assertEqual(progress.call_args.kwargs["target_archive_bytes"], 50000000000)
            stop.set()
            with self.assertRaises(KeyboardInterrupt):
                engine.wait_for_training_data(config, stop)


class ExternalTrainingGateTests(unittest.TestCase):
    def test_default_exposes_explicit_dataset_wait_policy(self):
        self.assertIn("wait_for_dataset", engine.load_config()["external"])
        self.assertIn("target_bytes", engine.load_config()["external"])

    def test_direct_train_cycle_blocks_before_audit_or_new_model_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = copy.deepcopy(engine.DEFAULT_CONFIG)
            config["external"] = settings("auto", True)
            with patch.object(engine, "ROOT", root), patch.object(engine, "ExperimentAudit") as audit:
                with self.assertRaises(external().DatasetNotReady):
                    engine.train_cycle(config, live=True)
            audit.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_invalid_dataset_gate_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(external().validate_config(settings("auto", True), Path(directory))["target_bytes"], 50000000000)
            for target in (30000000000, 0, True, 50000000000.0, "50000000000"):
                with self.subTest(target=target), self.assertRaises(ValueError):
                    external().validate_config({**settings("auto", True), "target_bytes": target}, Path(directory))
            with self.assertRaises(ValueError):
                external().validate_config(settings(False, True), Path(directory))

    def test_main_trainer_waits_without_reporting_a_successful_base_cycle(self):
        import contextlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = copy.deepcopy(engine.DEFAULT_CONFIG)
            config["external"] = settings("auto", True)
            with patch.object(engine, "ROOT", root), patch.object(engine, "load_config", return_value=config), \
                 patch.object(engine, "process_lock", return_value=contextlib.nullcontext()), \
                 patch.object(engine, "ExperimentAudit") as audit, patch.object(engine, "progress") as progress, \
                 patch("engine.threading.Event.wait", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    engine.main(["trainer", "--once"])
            audit.assert_not_called()
            self.assertEqual(progress.call_args.kwargs["status"], "waiting_data")
            self.assertEqual(list(root.iterdir()), [])

    def test_manifest_schema_compatibility_requires_frozen_bool_and_exact_external_columns(self):
        from test_market_universe import isolated_config, save_manifest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = isolated_config(root)
            model_dir, manifest = save_manifest(config)

            def save():
                engine.atomic_json(model_dir / "manifest.json", manifest)
                engine.atomic_json(Path(config["models_dir"]) / "current.json",
                                   {"run": "controlled", "manifest_hash": engine.stable_hash(manifest)})

            with patch.object(engine, "ROOT", root):
                manifest["config"].pop("external")
                save()
                self.assertEqual(engine.load_current(config)[1]["feature_columns"], ["x"])
                manifest["config"]["external"] = settings("auto", True)
                manifest["feature_columns"] = list(external().FEATURE_COLUMNS)
                save()
                with self.assertRaisesRegex(ValueError, "동결"):
                    engine.load_current(config)
                manifest["config"]["external"]["enabled"] = True
                save()
                self.assertIs(engine.load_current(config)[1]["config"]["external"]["enabled"], True)
                manifest["feature_columns"].pop()
                save()
                with self.assertRaisesRegex(ValueError, "스키마"):
                    engine.load_current(config)

    def test_training_manifest_freezes_auto_once_and_keeps_promotion_guard(self):
        from test_engine import candles
        from test_market_universe import isolated_config
        from test_gpu_throughput import fake_xgb
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = isolated_config(root)
            config["external"] = settings("auto", True)
            raw = candles(10000, start="2026-09-20")
            price = 100000 * (1 + .025 * np.sin(np.arange(len(raw)) / 10))
            raw["open"] = raw["close"] = price
            raw["high"], raw["low"] = price + 100, price - 100
            features = engine.build_features({5: raw})
            for column in external().FEATURE_COLUMNS:
                features[column] = np.float32(.01)
            fake, matrices, fits = fake_xgb()
            resolved = settings(True, True)
            actual_utc = engine.utc

            def clock(value=None):
                return actual_utc("2026-10-24T00:00:00Z" if value is None else value)

            def policy(frame, probabilities, paper, market, minimum, horizon):
                return {"confidence": 1.01, "edge": 1.01, "eligible": False,
                        "horizon_minutes": horizon, "trials": []}

            # 준비 판정과 GPU만 대역이며 bool 동결·시간 분할·manifest·승격 보류는 실제 경로입니다.
            with patch.dict("sys.modules", {"xgboost": fake}), patch.object(engine, "ROOT", root), \
                 patch.object(engine, "utc", side_effect=clock), patch.object(engine, "CandleStore"), \
                 patch.object(external(), "resolve_for_training", return_value=(resolved, {"training_ready": True})) as resolve, \
                 patch.object(engine, "load_market_features", return_value=features), \
                 patch.object(engine, "predict_booster", side_effect=lambda model, frame, columns, best: np.tile([.05, .9, .05], (len(frame), 1))), \
                 patch.object(engine, "tune_rule", side_effect=policy):
                result = engine.train_cycle(config, live=True)
            manifest = engine.read_json(Path(config["models_dir"]) / "versions" / result["run"] / "manifest.json")
            self.assertEqual(resolve.call_count, 1)
            self.assertIs(manifest["config"]["external"]["enabled"], True)
            self.assertEqual(config["external"]["enabled"], "auto")
            self.assertFalse(result["promotion"]["promote"])
            self.assertFalse((Path(config["models_dir"]) / "current.json").exists())
            self.assertEqual(len(matrices), 10)
            self.assertTrue(set(external().FEATURE_COLUMNS).issubset(manifest["feature_columns"]))

    @unittest.skipUnless(sys.platform == "linux", "외부 bars SQLite는 Linux 전용입니다")
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

    @unittest.skipUnless(sys.platform == "linux", "외부 bars SQLite는 Linux 전용입니다")
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
    def test_base_feature_drop_cannot_reuse_stale_external_from_an_older_row(self):
        config = copy.deepcopy(engine.DEFAULT_CONFIG)
        config["external"] = settings()
        manifest = {"run": "fixture", "schema_version": 2, "execution_policy": engine.EXECUTION_POLICY,
                    "deployment": {"eligible": True}, "config": config,
                    "feature_columns": ["ext_return_1", "internal"],
                    "packages": {m: {"rule": {"confidence": .5, "edge": .1}} for m in engine.MARKETS}}
        frame = pd.DataFrame({"ext_return_1": [.01, .02], "internal": [1., np.nan],
                              "5_volatility": [.001, .001],
                              "available_external": [NOW - pd.Timedelta(minutes=7), NOW]},
                             index=pd.DatetimeIndex([NOW - pd.Timedelta(minutes=5), NOW]))
        portfolio = engine.Portfolio(config["paper"])
        with patch.object(engine, "load_current", return_value=(engine.ROOT, manifest)), \
             patch.object(engine, "load_market_features", return_value=frame), \
             patch.object(engine, "predict_package", return_value=np.array([[.01, .01, .98]])):
            signals, *_ = engine.paper_signals(config, object(), portfolio, NOW)
        self.assertEqual(signals["KRW-BTC"], 1)
        self.assertEqual(portfolio.state["inference_diagnostics"]["KRW-BTC"]["reason"], "external_features_unavailable")

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
