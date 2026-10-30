"""네트워크·GPU 없이 실행하는 시점/계좌/감사 회귀검사."""
import copy
import contextlib
import io
import json
from pathlib import Path
import runpy
import shutil
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import engine


def candles(n=240, tf=5, start="2026-01-01", price=100000):
    index = pd.date_range(start, periods=n, freq=f"{tf}min", tz="UTC")
    close = price + np.arange(n, dtype=float) * 10 + np.sin(np.arange(n)) * 100
    return pd.DataFrame({"open": close - 10, "high": close + 200, "low": close - 200,
                         # 공개 API의 소수 수량과 SQLite REAL 계약을 그대로 사용합니다.
                         "close": close, "volume": (np.arange(n) % 20 + 1).astype(float)}, index=index)


def settings():
    result = copy.deepcopy(engine.DEFAULT_CONFIG["paper"])
    result.update(initial_cash=1000000, max_allocation=0.125, max_total_allocation=0.5,
                  daily_loss_limit=0.015, stop_loss=0.02)
    return result


class TemporalTests(unittest.TestCase):
    def test_completed_candle_filter_utc(self):
        payload = [{"candle_date_time_utc": "2026-01-01T00:00:00", "opening_price": 100,
                    "high_price": 101, "low_price": 99, "trade_price": 100, "candle_acc_trade_volume": 2}]
        self.assertTrue(engine.candles_from_api(payload, 60, "2026-01-01T00:59:59Z").empty)
        result = engine.candles_from_api(payload, 60, "2026-01-01T01:00:00Z")
        self.assertEqual(len(result), 1)
        self.assertEqual(str(result.index.tz), "UTC")

    def test_higher_timeframe_not_available_before_close(self):
        base, higher = candles(800), candles(70, 60)
        features = engine.build_features({5: base, 60: higher})
        before = pd.Timestamp("2026-01-02T00:55:00Z")
        after = pd.Timestamp("2026-01-02T01:00:00Z")
        self.assertEqual(features.loc[before, "available_60"], pd.Timestamp("2026-01-02T00:00:00Z"))
        self.assertEqual(features.loc[after, "available_60"], after)
        self.assertTrue((features.available_60.dropna() <= features.available_60.dropna().index).all())

    def test_future_mutation_cannot_change_past_features(self):
        base, higher = candles(900), candles(80, 60)
        cutoff = pd.Timestamp("2026-01-02T13:00:00Z")
        first = engine.build_features({5: base, 60: higher})
        mutated = {}
        for tf, frame in ((5, base), (60, higher)):
            frame = frame.copy()
            future = frame.index + pd.Timedelta(minutes=tf) > cutoff
            frame.loc[future, ["open", "high", "low", "close", "volume"]] *= 17
            mutated[tf] = frame
        second = engine.build_features(mutated)
        pd.testing.assert_frame_equal(first.loc[:cutoff], second.loc[:cutoff])

    def test_label_next_open_costs_and_missing_tail(self):
        frame = engine.build_features({5: candles(60)})
        frame["open"] = 100
        frame["close"] = 100
        labelled = engine.label_frame(frame, horizon=3, threshold=0.001, fee=0.001, slippage=0.001)
        self.assertTrue(labelled.label.tail(3).isna().all())
        self.assertTrue((labelled.label.iloc[:-3] == 0).all())
        expected = 100 * 0.999 * 0.999 / (100 * 1.001 * 1.001) - 1
        self.assertAlmostEqual(labelled.net_forward_return.iloc[0], expected)
        self.assertEqual(labelled.label_end.iloc[0], frame.index[3])

    def test_split_horizon_gap_no_shared_future(self):
        frame = engine.label_frame(engine.build_features({5: candles(600)}), horizon=12)
        plan = {"train": (frame.index[0], frame.index[200]),
                "tune": (frame.index[200], frame.index[350]),
                "calibration": (frame.index[350], frame.index[480]),
                "holdout": (frame.index[480], frame.index[-1])}
        split = engine.chronological_split(frame, plan, 13)
        for name, part in split.items():
            self.assertTrue((part.label_end < plan[name][1]).all())
            self.assertLess(part.index.max(), plan[name][1] - pd.Timedelta(minutes=65))
        self.assertLess(split["train"].label_end.max(), split["tune"].index.min())

    def test_recency_uses_elapsed_time(self):
        idx = pd.to_datetime(["2026-01-01", "2026-01-31"], utc=True)
        w = engine.recency_weights(idx, 30)
        self.assertAlmostEqual(w[1] / w[0], 2)

    def test_features_have_no_labels_or_available_time(self):
        frame = engine.label_frame(engine.build_features({5: candles()}))
        columns = engine.feature_columns(frame)
        self.assertNotIn("label", columns)
        self.assertNotIn("label_end", columns)
        self.assertFalse(any(c.startswith("available_") for c in columns))


class CacheAndAuditTests(unittest.TestCase):
    def test_connection_closes_after_commit_and_exception_rolls_back(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "closed.sqlite3"
            with engine.connect(path) as db:
                db.execute("CREATE TABLE records(value INTEGER)")
                db.execute("INSERT INTO records VALUES(1)")
            with self.assertRaises(sqlite3.ProgrammingError):
                db.execute("SELECT 1")
            with self.assertRaisesRegex(ValueError, "통제된"):
                with engine.connect(path) as failed:
                    failed.execute("INSERT INTO records VALUES(2)")
                    raise ValueError("통제된 트랜잭션 실패")
            with self.assertRaises(sqlite3.ProgrammingError):
                failed.execute("SELECT 1")
            with engine.connect(path) as reopened:
                self.assertEqual(reopened.execute("SELECT value FROM records").fetchall(), [(1,)])
    def test_cache_page_atomic_resume_and_immutable_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            store = engine.CandleStore(Path(temp) / "cache.sqlite3")
            frame = candles(4)
            store.save_page("KRW-BTC", 5, frame, engine.iso(frame.index[0]), backfill=True)
            changed = frame * 2
            store.save_page("KRW-BTC", 5, changed)
            pd.testing.assert_frame_equal(store.read("KRW-BTC", 5), frame.rename_axis("open_time"), check_freq=False)
            cursor, done = store.cursor("KRW-BTC", 5)
            self.assertEqual(cursor, engine.iso(frame.index[0]))
            self.assertFalse(done)

    def test_final_holdout_once_and_overlap_block(self):
        with tempfile.TemporaryDirectory() as temp:
            audit = engine.ExperimentAudit(Path(temp) / "audit.sqlite3")
            audit.reserve_final("first", "2026-01-01", "2026-02-01", "frozen")
            audit.finish("first")
            with self.assertRaisesRegex(RuntimeError, "이미 소비"):
                audit.reserve_final("second", "2026-01-15", "2026-02-15", "other")
            audit.expose("train", "2026-03-01", "2026-04-01", "tune")
            with self.assertRaisesRegex(RuntimeError, "노출"):
                audit.reserve_final("third", "2026-03-20", "2026-04-20", "other")

    def test_http_retry_is_finite(self):
        with tempfile.TemporaryDirectory() as temp:
            store = engine.CandleStore(Path(temp) / "cache.sqlite3")
            collection = copy.deepcopy(engine.DEFAULT_CONFIG["collection"])
            collection["retries"] = 2
            with patch("engine.urllib.request.urlopen", side_effect=engine.urllib.error.URLError("통제된 연결 실패")) as request, patch.object(store, "rate_limit"), patch("engine.time.sleep"):
                with self.assertRaises(engine.urllib.error.URLError):
                    engine.public_get(store, "candles/minutes/5", {}, collection)
                self.assertEqual(request.call_count, 2)

    def test_private_or_order_endpoint_is_not_supported(self):
        with self.assertRaises(ValueError):
            engine.public_get(None, "orders", {}, engine.DEFAULT_CONFIG["collection"])

    def test_atomic_json_preserves_previous_on_serialization_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            engine.atomic_json(path, {"old": 1})
            with self.assertRaises(ValueError):
                engine.atomic_json(path, {"bad": float("nan")})
            self.assertEqual(engine.read_json(path), {"old": 1})
            self.assertEqual(len(list(Path(temp).glob("*.tmp"))), 0)


class PortfolioTests(unittest.TestCase):
    def test_four_coins_share_cash_and_equity(self):
        portfolio = engine.Portfolio(settings())
        prices = {m: 100000.0 for m in engine.MARKETS}
        portfolio.process("2026-01-01T00:00:00Z", prices, {m: 2 for m in prices})
        self.assertEqual(len(portfolio.state["positions"]), 4)
        self.assertAlmostEqual(portfolio.state["cash"], 500000)
        self.assertLess(portfolio.equity(prices), 1000000)
        self.assertGreater(portfolio.equity(prices), 990000)
        self.assertGreater(portfolio.state["fees_paid"], 0)

    def test_both_sides_costs_and_no_short(self):
        portfolio = engine.Portfolio(settings())
        portfolio.process("2026-01-01T00:00:00Z", {"KRW-BTC": 100000}, {"KRW-BTC": 2})
        portfolio.process("2026-01-01T00:05:00Z", {"KRW-BTC": 100000}, {"KRW-BTC": 0})
        self.assertEqual(portfolio.state["positions"], {})
        self.assertLess(portfolio.state["cash"], 1000000)
        self.assertEqual(portfolio.state["completed_trades"], 1)
        self.assertTrue(all(e["fee_paid"] > 0 for e in portfolio.state["trades"]))
        portfolio.process("2026-01-01T00:10:00Z", {"KRW-BTC": 100000}, {"KRW-BTC": 0})
        self.assertEqual(portfolio.state["positions"], {})

    def test_daily_block_does_not_block_stop_or_sell(self):
        portfolio = engine.Portfolio(settings())
        prices = {"KRW-BTC": 100000, "KRW-ETH": 100000}
        portfolio.process("2026-01-01T00:00:00Z", prices, {m: 2 for m in prices})
        events = portfolio.process("2026-01-01T00:05:00Z", {"KRW-BTC": 50000, "KRW-ETH": 100000},
                                   {"KRW-BTC": 1, "KRW-ETH": 0})
        self.assertTrue(portfolio.state["new_buys_blocked"])
        self.assertEqual(len(events), 2)
        self.assertEqual(portfolio.state["positions"], {})
        self.assertEqual(events[0]["reason"], "stop_loss")

    def test_state_restore_duplicate_bar_no_second_execution(self):
        portfolio = engine.Portfolio(settings())
        portfolio.process("2026-01-01", {"KRW-BTC": 100000}, {"KRW-BTC": 2})
        restored = engine.Portfolio(settings(), json.loads(json.dumps(portfolio.state)))
        old = copy.deepcopy(restored.state)
        self.assertEqual(restored.process("2026-01-01", {"KRW-BTC": 110000}, {"KRW-BTC": 0}), [])
        self.assertEqual(restored.state, old)

    def test_orderbook_partial_fills_and_depth_vwap(self):
        units = [{"ask_price": 100, "ask_size": 2, "bid_price": 99, "bid_size": 1},
                 {"ask_price": 110, "ask_size": 1, "bid_price": 98, "bid_size": 2}]
        buy = engine.sweep_book(units, "buy", 1000, 0, 0)
        self.assertEqual(buy["qty"], 3)
        self.assertAlmostEqual(buy["vwap"], 310 / 3)
        self.assertTrue(buy["partial"])
        sell = engine.sweep_book(units, "sell", 4, 0, 0)
        self.assertEqual(sell["qty"], 3)
        self.assertEqual(sell["cash"], 295)
        self.assertEqual(sell["unfilled"], 1)

    def test_portfolio_sell_partial_retains_unsold_position(self):
        portfolio = engine.Portfolio(settings())
        prices = {"KRW-BTC": 100000}
        portfolio.process("2026-01-01", prices, {"KRW-BTC": 2})
        old_qty = portfolio.state["positions"]["KRW-BTC"]["qty"]
        units = [{"ask_price": 100000, "ask_size": 0, "bid_price": 100000, "bid_size": old_qty / 2}]
        portfolio.process("2026-01-01T00:05:00Z", prices, {"KRW-BTC": 0}, {"KRW-BTC": units})
        self.assertAlmostEqual(portfolio.state["positions"]["KRW-BTC"]["qty"], old_qty / 2)

    def test_backtest_fills_after_signal_not_at_same_close(self):
        raw = candles(3)
        frame = engine.build_features({5: raw})
        p = np.array([[0.01, 0.01, 0.98], [0.98, 0.01, 0.01], [0.01, 0.98, 0.01]])
        result = engine.backtest_portfolio({"KRW-BTC": frame}, {"KRW-BTC": p},
                                           {"KRW-BTC": {"confidence": 0.5, "edge": 0.1}}, settings())
        self.assertEqual(result["trades"], 1)
        self.assertEqual(result["executions"], 2)

    def test_paper_start_guard_does_not_touch_network(self):
        config = engine.load_config()
        with tempfile.TemporaryDirectory() as temp, patch.object(engine, "ROOT", Path(temp)), patch("engine.public_get") as network:
            result = engine.paper_cycle(config, "2026-10-21T08:59:59Z")
            self.assertEqual(result["status"], "예정 시작 대기")
            network.assert_not_called()


class EvaluationTests(unittest.TestCase):
    def test_incumbent_comparison_uses_exact_current_calibration_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            config = engine.load_config()
            config["models_dir"] = temp
            pointer = Path(temp) / "current.json"
            engine.atomic_json(pointer, {"run": "old", "manifest_hash": "kept"})
            frame = engine.build_features({5: candles(3)})
            frame["label"] = [0, 1, 2]
            frames = {m: frame.copy() for m in engine.MARKETS}
            good = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
            bad = {m: np.full((3, 3), 1 / 3) for m in engine.MARKETS}
            rules = {m: {"confidence": 1.01, "edge": 1.01} for m in engine.MARKETS}
            manifest = {"schema_version": 2, "execution_policy": "horizon_or_risk_v2", "deployment": {"eligible": True},
                        "config": config, "run": "old", "packages": {m: {"rule": rules[m]} for m in engine.MARKETS}}
            with patch("engine.load_current", return_value=(Path(temp), manifest)), patch("engine.predict_package", return_value=good) as predict:
                result = engine.assess_promotion(config, frames, bad, rules)
            self.assertFalse(result["promote"])
            self.assertFalse(result["logloss_not_worse"])
            for call in predict.call_args_list:
                self.assertIs(call.args[3], frames[call.args[2]])
            self.assertEqual(engine.read_json(pointer), {"run": "old", "manifest_hash": "kept"})
            self.assertEqual(result["comparison_periods"]["KRW-BTC"]["rows"], 3)

    def test_corrupt_incumbent_is_retained_with_comparison_error(self):
        with tempfile.TemporaryDirectory() as temp:
            config = engine.load_config()
            config["models_dir"] = temp
            engine.atomic_json(Path(temp) / "current.json", {"run": "corrupt"})
            frame = engine.build_features({5: candles(3)})
            frame["label"] = [0, 1, 2]
            probs = np.full((3, 3), 1 / 3)
            with patch("engine.load_current", side_effect=ValueError("통제된 manifest 손상")):
                result = engine.assess_promotion(config, {"KRW-BTC": frame}, {"KRW-BTC": probs},
                                                 {"KRW-BTC": {"confidence": 1.01, "edge": 1.01}})
            self.assertFalse(result["promote"])
            self.assertEqual(result["comparison_error"]["type"], "ValueError")
    def test_promotion_rejects_worse_probability_or_trading_objective(self):
        current = {"logloss": 0.5, "net_return": 0.02, "MDD": 0.01}
        for candidate in ({"logloss": 0.6, "net_return": 0.03, "MDD": 0.01},
                          {"logloss": 0.4, "net_return": -0.01, "MDD": 0.01}):
            with self.subTest(candidate=candidate):
                self.assertFalse(engine.promotion_decision(candidate, current)["promote"])
        better = {"logloss": 0.4, "net_return": 0.03, "MDD": 0.005}
        self.assertTrue(engine.promotion_decision(better, current)["promote"])
        self.assertFalse(engine.promotion_decision(current, current)["promote"])

    def test_all_three_classes_and_target_shortfall_reported(self):
        result = engine.classification_metrics([0, 1, 2], [[0.1, 0.8, 0.1]] * 3, majority_class=1)
        self.assertAlmostEqual(result["accuracy"], 1 / 3)
        self.assertEqual(result["support"], 3)
        self.assertEqual(result["class_distribution"], [1, 1, 1])
        self.assertEqual(result["target_status"], "95% 목표 미달")

    def test_smooth_logloss_weights_favor_better_candidate(self):
        good = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        bad = np.full((3, 3), 1 / 3)
        w = engine.ensemble_weights([good, bad], [0, 1, 2])
        self.assertGreater(w[0], w[1])
        self.assertAlmostEqual(w.sum(), 1)
        self.assertTrue((w > 0).all())

    def test_cpu_config_is_not_gpu_evidence(self):
        with self.assertRaisesRegex(RuntimeError, "CPU 대체 금지"):
            engine.assert_cuda_config({"learner": {"generic_param": {"device": "cpu"}}})
        self.assertEqual(engine.assert_cuda_config({"learner": {"generic_param": {"device": "cuda:0"}}}), "cuda:0")

    def test_training_mandates_explicit_locked_period(self):
        with self.assertRaisesRegex(ValueError, "고정 UTC"):
            engine.experiment_plan(engine.load_config())

    def test_cli_interface(self):
        parser = engine.build_parser()
        args = parser.parse_args(["--config", "config.json", "collector", "--backfill", "--once", "--pages", "5"])
        self.assertTrue(args.backfill)
        self.assertEqual(args.pages, 5)
        self.assertEqual(parser.parse_args(["paper", "--paper-start", "2026-10-21T09:00:00Z"]).paper_start, "2026-10-21T09:00:00Z")


class GpuBoundaryTests(unittest.TestCase):
    def fake_xgb(self, device):
        booster = Mock()
        booster.save_config.return_value = json.dumps({"learner": {"generic_param": {"device": device}}})
        return SimpleNamespace(build_info=lambda: {"USE_CUDA": True}, DMatrix=Mock(),
                               Booster=Mock(return_value=booster), train=Mock(),
                               callback=SimpleNamespace(TrainingCallback=object))

    def test_cpu_preflight_blocks_train_without_progress(self):
        fake = self.fake_xgb("cpu")
        with patch.dict("sys.modules", {"xgboost": fake}):
            with self.assertRaisesRegex(RuntimeError, "CPU 대체 금지"):
                engine.check_gpu()
        fake.train.assert_not_called()

    def test_configuration_failure_blocks_train(self):
        fake = self.fake_xgb("cuda:0")
        fake.Booster.return_value.save_config.side_effect = RuntimeError("통제된 CUDA 초기화 실패")
        with patch.dict("sys.modules", {"xgboost": fake}):
            with self.assertRaisesRegex(RuntimeError, "초기화 실패"):
                engine.check_gpu()
        fake.train.assert_not_called()

    def test_actual_training_booster_rechecked_before_update(self):
        fake = self.fake_xgb("cuda:0")
        cpu = self.fake_xgb("cpu").Booster.return_value
        update = Mock()

        def train(*args, **kwargs):
            for callback in kwargs["callbacks"]:
                callback.before_training(cpu)
            update()

        fake.train.side_effect = train
        with patch.dict("sys.modules", {"xgboost": fake}):
            with self.assertRaisesRegex(RuntimeError, "CPU 대체 금지"):
                engine.check_gpu()
        update.assert_not_called()


class PaperFailurePathTests(unittest.TestCase):
    def environment(self, root):
        now = engine.utc(None)
        config = engine.load_config()
        config["paper"].update(settings())
        config["paper"]["start_time"] = engine.iso(now - pd.Timedelta(days=1))
        for key, folder in (("data_dir", "data"), ("models_dir", "models"), ("reports_dir", "reports")):
            config[key] = str(root / "work" / folder)
        portfolio = engine.Portfolio(config["paper"])
        portfolio.process(now - pd.Timedelta(minutes=1), {"KRW-BTC": 100000}, {"KRW-BTC": 2})
        with engine.connect(root / "work/paper/ledger.sqlite3") as db:
            db.execute("CREATE TABLE checkpoint(id INTEGER PRIMARY KEY,payload TEXT)")
            db.execute("INSERT INTO checkpoint VALUES(1,?)", (json.dumps(portfolio.state),))
        quotes = []
        for market in engine.MARKETS:
            price = 50000 if market == "KRW-BTC" else 100000
            quotes.append({"market": market, "timestamp": int(now.timestamp() * 1000), "orderbook_units": [
                {"ask_price": price + 1, "ask_size": 100, "bid_price": price - 1, "bid_size": 100}]})
        manifest = {"schema_version": 2, "execution_policy": "horizon_or_risk_v2", "deployment": {"eligible": True},
                    "run": "통제된-모델", "config": config, "feature_columns": ["x"],
                    "packages": {m: {"rule": {"confidence": 0.5, "edge": 0.1}} for m in engine.MARKETS}}
        frame = pd.DataFrame({"x": [1.0], "5_volatility": [0.005]}, index=pd.DatetimeIndex([now.floor("5min")]))
        return config, now, quotes, manifest, frame

    def test_inference_file_and_value_errors_do_not_disable_stop(self):
        for failure in (FileNotFoundError("통제된 모델 파일 없음"), ValueError("통제된 잘못된 확률")):
            with self.subTest(error=type(failure).__name__), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                with patch.object(engine, "ROOT", root):
                    config, now, quotes, manifest, frame = self.environment(root)
                    with patch("engine.public_get", return_value=quotes), patch("engine.load_current", return_value=(root, manifest)), \
                         patch("engine.load_market_features", return_value=frame), patch("engine.predict_package", side_effect=failure):
                        result = engine.paper_cycle(config, now)
                    self.assertNotIn("KRW-BTC", result["positions"])
                    self.assertTrue(result["model_error"])
                    self.assertIn("KRW-BTC", result["inference_errors"])
                    self.assertEqual(result["trades"][-1]["reason"], "stop_loss")
                    self.assertEqual(result["entry_guard"], "추론 오류로 신규 진입 차단")

    def test_missing_current_pointer_still_liquidates_and_appends_equity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(engine, "ROOT", root):
                config, now, quotes, _, _ = self.environment(root)
                with patch("engine.public_get", return_value=quotes), patch("engine.load_current", side_effect=FileNotFoundError("포인터 없음")):
                    first = engine.paper_cycle(config, now)
                    second = engine.paper_cycle(config, now + pd.Timedelta(seconds=1))
                self.assertNotIn("KRW-BTC", first["positions"])
                self.assertEqual(first["cash"], second["cash"])
                with engine.connect(root / "work/paper/ledger.sqlite3") as db:
                    history = db.execute("SELECT timestamp,equity FROM equity_history ORDER BY timestamp").fetchall()
                self.assertEqual(len(history), 2)
                self.assertGreater(history[-1][1], 0)

    def test_one_coin_inference_failure_keeps_other_coin_sell_and_blocks_buys(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(engine, "ROOT", root):
                config, now, quotes, manifest, frame = self.environment(root)
                with engine.connect(root / "work/paper/ledger.sqlite3") as db:
                    state = json.loads(db.execute("SELECT payload FROM checkpoint WHERE id=1").fetchone()[0])
                    portfolio = engine.Portfolio(config["paper"], state)
                    portfolio.process(now - pd.Timedelta(seconds=30), {"KRW-BTC": 100000, "KRW-ETH": 100000}, {"KRW-ETH": 2})
                    db.execute("UPDATE checkpoint SET payload=? WHERE id=1", (json.dumps(portfolio.state),))

                def predict(directory, model_manifest, market, rows):
                    if market == "KRW-BTC":
                        raise ValueError("통제된 BTC 추론 실패")
                    return np.array([[0.99, 0.005, 0.005] if market == "KRW-ETH" else [0.005, 0.005, 0.99]])

                with patch("engine.public_get", return_value=quotes), patch("engine.load_current", return_value=(root, manifest)), \
                     patch("engine.load_market_features", return_value=frame), patch("engine.predict_package", side_effect=predict):
                    result = engine.paper_cycle(config, now)
                self.assertEqual(result["positions"], {})
                events = result["trades"][-2:]
                self.assertEqual({(e["market"], e["reason"]) for e in events}, {("KRW-BTC", "stop_loss"), ("KRW-ETH", "signal")})
                with engine.connect(root / "work/paper/ledger.sqlite3") as db:
                    errors = db.execute("SELECT market,payload FROM inference_errors").fetchall()
                self.assertEqual(len(errors), 1)
                self.assertEqual(errors[0][0], "KRW-BTC")


class CliStatusTests(unittest.TestCase):
    def test_collector_error_does_not_overwrite_training_progress(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            program = root / "engine.py"
            shutil.copyfile(Path(engine.__file__), program)
            status = root / "work/service/training_progress.json"
            original = {"run": "학습중인-실행", "status": "running", "stage": "fit", "current_round": 10}
            engine.atomic_json(status, original)
            with patch("sys.argv", [str(program), "collector", "--once", "--pages", "-1"]), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit_context:
                runpy.run_path(str(program), run_name="__main__")
            self.assertEqual(exit_context.exception.code, 1)
            self.assertEqual(engine.read_json(status), original)
            failure = engine.read_json(root / "work/service/engine_error.json")
            self.assertIn("페이지 수는 양수", failure["error"])


if __name__ == "__main__":
    unittest.main()
