"""DOGE를 포함한 다섯 시장과 기존 현금·모델 안전 경계를 검사합니다."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import engine
from test_engine import candles


MARKETS = ("KRW-BTC", "KRW-SOL", "KRW-ETH", "KRW-XRP", "KRW-DOGE")


def isolated_config(root):
    config = copy.deepcopy(engine.DEFAULT_CONFIG)
    config["markets"] = list(MARKETS)
    config["timeframes"] = [5]
    config["paper"]["start_time"] = "2026-10-26T00:00:00Z"
    config["training"].update(candidate_depths=[3], min_split_rows=10, train_days=14,
                              tune_days=2, calibration_days=2, rule_days=2, validation_days=2,
                              holdout_start="2026-10-19T00:00:00Z",
                              holdout_end="2026-10-20T00:00:00Z", holdout_days=1)
    for key, folder in (("data_dir", "data"), ("models_dir", "models"), ("reports_dir", "reports")):
        config[key] = str(root / "work" / folder)
    return config


def save_manifest(config, markets=MARKETS):
    directory = Path(config["models_dir"]) / "versions" / "controlled"
    directory.mkdir(parents=True)
    packages = {}
    for market in markets:
        filename = market + ".ubj"
        # 로더의 파일 존재 경계만 검사하는 대역이며 실제 GPU 모델이 아닙니다.
        (directory / filename).write_bytes(b"controlled-model-fixture")
        packages[market] = {"models": [filename], "weights": [1.0],
                            "gpu": [{"actual_device": "cuda:0", "best_iteration": 0}],
                            "temperature": 1.0,
                            "rule": {"confidence": .5, "edge": .1, "horizon_minutes": 60}}
    manifest = {"schema_version": 2, "execution_policy": engine.EXECUTION_POLICY,
                "deployment": {"eligible": True}, "paper_only": True, "run": "controlled",
                "config": copy.deepcopy(config), "feature_columns": ["x"], "packages": packages}
    manifest["config"]["markets"] = list(markets)
    engine.atomic_json(directory / "manifest.json", manifest)
    engine.atomic_json(Path(config["models_dir"]) / "current.json",
                       {"run": "controlled", "manifest_hash": engine.stable_hash(manifest)})
    return directory, manifest


class MarketConfigTests(unittest.TestCase):
    def test_default_and_shipped_config_include_doge_as_fifth_market(self):
        for path in (None, Path(engine.__file__).parent / "engine-config.json"):
            with self.subTest(path=path):
                config = engine.load_config(path)
                self.assertEqual(config["markets"], list(MARKETS))
                self.assertEqual(config["paper"]["initial_cash"], 10_000_000)
                self.assertEqual(config["paper"]["max_allocation"], .125)
                self.assertEqual(config["paper"]["max_total_allocation"], .5)
                self.assertEqual(config["training"]["retrain_seconds"], 0)

    def test_explicit_five_markets_are_accepted_in_either_order(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            for markets in (list(MARKETS), list(reversed(MARKETS))):
                with self.subTest(markets=markets):
                    engine.atomic_json(path, {"markets": markets})
                    self.assertEqual(engine.load_config(path)["markets"], markets)

    def test_missing_duplicate_and_unsupported_markets_are_rejected(self):
        invalid = [list(MARKETS[:-1]), list(MARKETS[1:]), [],
                   [*MARKETS, "KRW-DOGE"], [*MARKETS[:-1], "KRW-BTC"],
                   [*MARKETS, "KRW-ADA"], [*MARKETS[:-1], "KRW-ADA"]]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            for markets in invalid:
                with self.subTest(markets=markets):
                    engine.atomic_json(path, {"markets": markets})
                    with self.assertRaises(ValueError):
                        engine.load_config(path)


class MarketPortfolioTests(unittest.TestCase):
    def test_doge_alone_buys_sells_pays_both_fees_and_cannot_short(self):
        portfolio = engine.Portfolio()
        prices = {"KRW-DOGE": 300.0}
        bought = portfolio.process("2026-01-01T00:00:00Z", prices, {"KRW-DOGE": 2})
        self.assertEqual([(e["market"], e["side"]) for e in bought], [("KRW-DOGE", "buy")])
        self.assertAlmostEqual(portfolio.state["cash"], 8_750_000)
        self.assertAlmostEqual(portfolio.state["positions"]["KRW-DOGE"]["cost"], 1_250_000)
        sold = portfolio.process("2026-01-01T00:05:00Z", prices, {"KRW-DOGE": 0})
        self.assertEqual([(e["market"], e["side"]) for e in sold], [("KRW-DOGE", "sell")])
        self.assertEqual(portfolio.state["positions"], {})
        self.assertEqual(portfolio.state["completed_trades"], 1)
        self.assertLess(portfolio.state["cash"], 10_000_000)
        self.assertTrue(all(e["fee_paid"] > 0 and e["paper_only"] is True for e in bought + sold))
        self.assertEqual(portfolio.process("2026-01-01T00:10:00Z", prices, {"KRW-DOGE": 0}), [])

    def test_five_markets_share_one_cash_balance_and_fifty_percent_cap(self):
        portfolio = engine.Portfolio()
        prices = {m: 300.0 if m == "KRW-DOGE" else 100_000.0 for m in MARKETS}
        events = portfolio.process("2026-01-01T00:00:00Z", prices, {m: 2 for m in MARKETS})
        positions = portfolio.state["positions"]
        self.assertIn("KRW-DOGE", positions)
        self.assertEqual(len(positions), 4)
        self.assertAlmostEqual(portfolio.state["cash"], 5_000_000)
        self.assertAlmostEqual(sum(e["cash"] for e in events) + portfolio.state["cash"], 10_000_000)
        exposure = sum(p["qty"] * prices[m] for m, p in positions.items())
        self.assertLessEqual(exposure, 5_000_000)
        self.assertTrue(all(p["cost"] <= 1_250_000.000001 for p in positions.values()))
        self.assertLess(portfolio.equity(prices), 10_000_000)
        self.assertEqual(portfolio.state["decision_diagnostics"]["KRW-XRP"]["reason"], "below_min_order")

    def test_doge_loss_still_blocks_new_buys_but_allows_stop_loss(self):
        portfolio = engine.Portfolio()
        portfolio.process("2026-01-01T00:00:00Z", {"KRW-DOGE": 300}, {"KRW-DOGE": 2})
        events = portfolio.process("2026-01-01T00:05:00Z", {"KRW-DOGE": 150, "KRW-BTC": 100_000},
                                   {"KRW-DOGE": 1, "KRW-BTC": 2})
        self.assertTrue(portfolio.state["new_buys_blocked"])
        self.assertEqual(portfolio.state["positions"], {})
        self.assertEqual([(e["market"], e["reason"]) for e in events], [("KRW-DOGE", "stop_loss")])


class MarketCollectionAndTrainingTests(unittest.TestCase):
    def test_collector_stores_all_five_markets_and_nine_timeframes(self):
        with tempfile.TemporaryDirectory() as temp:
            config = isolated_config(Path(temp))
            config["timeframes"] = list(engine.TIMEFRAMES)
            payload = [{"candle_date_time_utc": "2026-01-01T00:00:00", "opening_price": 300,
                        "high_price": 302, "low_price": 299, "trade_price": 301,
                        "candle_acc_trade_volume": 10.5}]
            with patch("engine.public_get", return_value=payload) as request:
                datasets = engine.collect_cycle(config, pages=1)
            self.assertEqual(len(datasets), 45)
            self.assertEqual({d["market"] for d in datasets}, set(MARKETS))
            self.assertEqual({d["timeframe"] for d in datasets if d["market"] == "KRW-DOGE"},
                             {"1", "3", "5", "10", "15", "30", "60", "240", "day"})
            self.assertEqual(sum(d["rows"] for d in datasets), 45)
            self.assertEqual({call.args[2]["market"] for call in request.call_args_list}, set(MARKETS))
            store = engine.CandleStore(Path(config["data_dir"]) / "candles.sqlite3")
            self.assertEqual(store.read("KRW-DOGE", 5).close.tolist(), [301.0])

    def run_training(self, root, missing_doge=False):
        config = isolated_config(root)
        raw = candles(10_000, start="2026-09-20")
        price = 100_000 * (1 + .025 * np.sin(np.arange(len(raw)) / 10))
        raw["open"] = raw["close"] = price
        raw["high"], raw["low"] = price + 100, price - 100
        features = engine.build_features({5: raw})
        actual_load = engine.load_market_features

        def load(store, market, cfg, start, end):
            if missing_doge and market == "KRW-DOGE":
                return actual_load(store, market, cfg, start, end)
            return features

        def predict(model, frame, columns, best_iteration):
            return np.tile([.05, .9, .05], (len(frame), 1))

        def policy(frame, probabilities, settings, market, minimum, horizon):
            return {"confidence": 1.01, "edge": 1.01, "eligible": False,
                    "horizon_minutes": horizon, "trials": []}

        # 기간 분할·격리 SQLite 감사·보고서·포인터 처리는 실제 코드로 검증합니다.
        with patch.object(engine, "ROOT", root), patch("engine.load_market_features", side_effect=load), \
             patch("engine.fit_gpu", return_value=(Mock(), {"best_iteration": 0, "actual_device": "cuda:0"})), \
             patch("engine.predict_booster", side_effect=predict), patch("engine.tune_rule", side_effect=policy):
            return engine.train_cycle(config), config

    def test_trainer_completes_five_packages_without_promoting_no_trade_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            report, config = self.run_training(Path(temp))
            self.assertEqual(set(report["markets"]), set(MARKETS))
            self.assertEqual({r["market"] for r in report["rounds"]}, set(MARKETS))
            self.assertEqual(report["training"]["completed_candidates"], 5)
            self.assertEqual(report["training"]["total_candidates"], 5)
            manifest = engine.read_json(Path(config["models_dir"]) / "versions" / report["run"] / "manifest.json")
            self.assertEqual(set(manifest["packages"]), set(MARKETS))
            self.assertEqual(report["locked_test"]["status"], "미평가")
            self.assertFalse(report["promotion"]["promote"])
            self.assertFalse((Path(config["models_dir"]) / "current.json").exists())

    def test_empty_doge_history_fails_whole_cycle_without_publishing_partial_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = isolated_config(root)
            pointer = Path(config["models_dir"]) / "current.json"
            engine.atomic_json(pointer, {"run": "preserved", "manifest_hash": "preserved"})
            original = pointer.read_bytes()
            with self.assertRaisesRegex(ValueError, "수집된 KRW-DOGE/5 봉이 없습니다"):
                self.run_training(root, missing_doge=True)
            self.assertEqual(pointer.read_bytes(), original)
            self.assertEqual(list((Path(config["models_dir"]) / "versions").glob("*/manifest.json")), [])


class MarketTransitionPromotionTests(unittest.TestCase):
    def compare(self, config, trade=True, exit_prices=(101_000, 101_000, 101_000), markets=MARKETS):
        frame = engine.build_features({5: candles(7)})
        frame["open"] = frame["close"] = [100_000, 100_000, exit_prices[0], 100_000,
                                          exit_prices[1], 100_000, exit_prices[2]]
        frame["label"] = 1
        frames = {m: frame.copy() for m in markets}
        probabilities = {m: np.tile([.01, .98, .01], (7, 1)) for m in markets}
        if trade and "KRW-DOGE" in probabilities:
            probabilities["KRW-DOGE"] = np.array([[.01, .01, .98], [.98, .01, .01]] * 3 + [[.01, .98, .01]])
        rules = {m: {"confidence": .5, "edge": .1} for m in markets}
        with patch("engine.predict_package") as prediction:
            result = engine.assess_promotion(config, frames, probabilities, rules)
        prediction.assert_not_called()
        return result

    def test_intact_old_four_market_loader_raises_specific_transition_error(self):
        with tempfile.TemporaryDirectory() as temp:
            config = isolated_config(Path(temp))
            save_manifest(config, MARKETS[:-1])
            with self.assertRaises(ValueError) as raised:
                engine.load_current(config)
            self.assertEqual(type(raised.exception).__name__, "MarketUniverseTransitionError")
            self.assertEqual(raised.exception.run, "controlled")

    def test_intact_old_four_market_allows_profitable_five_market_candidate_against_cash(self):
        with tempfile.TemporaryDirectory() as temp:
            config = isolated_config(Path(temp))
            save_manifest(config, MARKETS[:-1])
            pointer = Path(config["models_dir"]) / "current.json"
            original = pointer.read_bytes()
            result = self.compare(config)
            self.assertEqual(pointer.read_bytes(), original)
        self.assertTrue(result["eligibility"]["eligible"])
        self.assertGreater(result["candidate"]["net_return"], 0)
        self.assertGreater(result["eligibility"]["objective"], 0)
        self.assertTrue(result["promote"])
        self.assertEqual(result["comparison_baseline"], "cash")
        self.assertEqual(result["incumbent"], {"net_return": 0.0, "MDD": 0.0, "logloss": None})
        self.assertEqual(result["excluded_incumbent_run"], "controlled")
        self.assertEqual(result["excluded_incumbent_reason"], "legacy_four_market_universe_requires_revalidation")
        self.assertEqual(set(result["comparison_periods"]), set(MARKETS))
        self.assertFalse(result["probability_comparable"])

    def test_transition_cash_baseline_still_rejects_no_trade_and_losing_candidates(self):
        for trade, exits, reason in ((False, (101_000, 101_000, 101_000), "insufficient_closed_trades"),
                                     (True, (99_000, 99_000, 99_000), "non_positive_net_return")):
            with self.subTest(trade=trade), tempfile.TemporaryDirectory() as temp:
                config = isolated_config(Path(temp))
                save_manifest(config, MARKETS[:-1])
                result = self.compare(config, trade, exits)
                self.assertFalse(result["promote"])
                self.assertEqual(result["comparison_baseline"], "cash")
                self.assertIn(reason, result["eligibility"]["rejection_reasons"])

    def test_transition_does_not_relax_minimum_closed_trades(self):
        with tempfile.TemporaryDirectory() as temp:
            config = isolated_config(Path(temp))
            save_manifest(config, MARKETS[:-1])
            config["training"]["min_closed_trades"] = 4
            result = self.compare(config)
        self.assertGreater(result["candidate"]["net_return"], 0)
        self.assertFalse(result["promote"])
        self.assertEqual(result["comparison_baseline"], "cash")
        self.assertEqual(result["eligibility"]["min_closed_trades"], 4)
        self.assertIn("insufficient_closed_trades", result["eligibility"]["rejection_reasons"])

    def test_transition_rejects_positive_return_with_non_positive_risk_adjusted_return(self):
        with tempfile.TemporaryDirectory() as temp:
            config = isolated_config(Path(temp))
            save_manifest(config, MARKETS[:-1])
            result = self.compare(config, exit_prices=(98_000, 101_400, 101_400))
        self.assertGreater(result["candidate"]["net_return"], 0)
        self.assertLessEqual(result["eligibility"]["objective"], 0)
        self.assertFalse(result["promote"])
        self.assertEqual(result["comparison_baseline"], "cash")
        self.assertIn("non_positive_risk_adjusted_return", result["eligibility"]["rejection_reasons"])

    def test_damaged_old_four_market_never_uses_transition_cash_baseline(self):
        cases = ("hash", "missing_file", "wrong_run", "invalid_weights", "candidate_count",
                 "invalid_columns", "paper_flag")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                config = isolated_config(Path(temp))
                directory, manifest = save_manifest(config, MARKETS[:-1])
                if case == "hash":
                    manifest["created_at"] = "hash-mismatch"
                elif case == "missing_file":
                    (directory / "KRW-XRP.ubj").unlink()
                elif case == "wrong_run":
                    manifest["run"] = "different-run"
                elif case == "invalid_weights":
                    manifest["packages"]["KRW-XRP"]["weights"] = [.5]
                elif case == "candidate_count":
                    manifest["packages"]["KRW-XRP"]["gpu"] = []
                elif case == "invalid_columns":
                    manifest["feature_columns"] = []
                else:
                    manifest["paper_only"] = False
                engine.atomic_json(directory / "manifest.json", manifest)
                if case != "hash":
                    engine.atomic_json(Path(config["models_dir"]) / "current.json",
                                       {"run": "controlled", "manifest_hash": engine.stable_hash(manifest)})
                result = self.compare(config)
                self.assertFalse(result["promote"])
                self.assertTrue(result["eligibility"]["eligible"])
                self.assertIn("comparison_error", result)
                self.assertNotIn("comparison_baseline", result)

    def test_unknown_subset_or_inconsistent_old_config_is_not_a_known_transition(self):
        for markets, configured in ((MARKETS[1:], MARKETS[1:]), (MARKETS[:-1], MARKETS),
                                    (MARKETS[:-1], (*MARKETS[:-1], "KRW-BTC")),
                                    (MARKETS[:-1], ("KRW-BTC", "KRW-SOL", "KRW-ETH", "KRW-ADA"))):
            with self.subTest(markets=markets, configured=configured), tempfile.TemporaryDirectory() as temp:
                config = isolated_config(Path(temp))
                directory, manifest = save_manifest(config, markets)
                manifest["config"]["markets"] = list(configured)
                engine.atomic_json(directory / "manifest.json", manifest)
                engine.atomic_json(Path(config["models_dir"]) / "current.json",
                                   {"run": "controlled", "manifest_hash": engine.stable_hash(manifest)})
                result = self.compare(config)
                self.assertFalse(result["promote"])
                self.assertIn("comparison_error", result)
                self.assertNotIn("comparison_baseline", result)

    def test_model_read_oserror_still_blocks_promotion(self):
        with tempfile.TemporaryDirectory() as temp:
            config = isolated_config(Path(temp))
            save_manifest(config, MARKETS[:-1])
            with patch("engine.read_json", side_effect=OSError("통제된 모델 읽기 실패")):
                result = self.compare(config)
        self.assertFalse(result["promote"])
        self.assertEqual(result["comparison_error"]["type"], "OSError")
        self.assertNotIn("comparison_baseline", result)

    def test_partial_candidate_is_not_allowed_to_replace_known_old_four_market(self):
        with tempfile.TemporaryDirectory() as temp:
            config = isolated_config(Path(temp))
            save_manifest(config, MARKETS[:-1])
            result = self.compare(config, markets=("KRW-DOGE",))
        self.assertTrue(result["eligibility"]["eligible"])
        self.assertFalse(result["promote"])
        self.assertIn("comparison_error", result)
        self.assertNotIn("comparison_baseline", result)


class MarketModelAndPaperTests(unittest.TestCase):
    def test_complete_five_market_manifest_is_loadable(self):
        with tempfile.TemporaryDirectory() as temp:
            config = isolated_config(Path(temp))
            directory, manifest = save_manifest(config)
            self.assertEqual(engine.load_current(config), (directory, manifest))

    def test_old_four_market_and_other_incomplete_packages_are_rejected(self):
        for markets in (MARKETS[:-1], MARKETS[1:], (*MARKETS, "KRW-ADA")):
            with self.subTest(markets=markets), tempfile.TemporaryDirectory() as temp:
                config = isolated_config(Path(temp))
                save_manifest(config, markets)
                with self.assertRaisesRegex(ValueError, "구성"):
                    engine.load_current(config)

    def test_missing_doge_model_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            config = isolated_config(Path(temp))
            directory, _ = save_manifest(config)
            (directory / "KRW-DOGE.ubj").unlink()
            with self.assertRaisesRegex(ValueError, "모델 파일"):
                engine.load_current(config)

    def test_four_market_model_keeps_five_market_signals_neutral_and_blocks_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            config = isolated_config(Path(temp))
            save_manifest(config, MARKETS[:-1])
            portfolio = engine.Portfolio(config["paper"])
            with patch("engine.predict_package") as prediction:
                signals, _, times, errors, run = engine.paper_signals(config, None, portfolio,
                                                                     engine.utc("2026-10-26T01:00:00Z"))
            self.assertEqual(signals, dict.fromkeys(MARKETS, 1))
            self.assertEqual(times, {})
            self.assertIsNone(run)
            self.assertEqual(errors["model"]["stage"], "load_current")
            self.assertEqual(portfolio.state["deployment_guard"], "model_load_error")
            self.assertEqual(portfolio.state["cash"], 10_000_000)
            prediction.assert_not_called()

    def test_public_paper_quotes_include_doge_and_execute_its_buy_and_sell(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = isolated_config(root)
            save_manifest(config)
            now = engine.utc("2026-10-26T01:00:00Z")
            actual_utc = engine.utc
            doge_signal = 2

            def clock(value=None):
                return actual_utc(now if value is None else value)

            def quotes(store, endpoint, params, collection):
                self.assertEqual(endpoint, "orderbook")
                self.assertEqual(params, {"markets": "KRW-BTC,KRW-SOL,KRW-ETH,KRW-XRP,KRW-DOGE"})
                return [{"market": m, "timestamp": int(now.timestamp() * 1000), "orderbook_units": [
                    {"ask_price": 301, "bid_price": 299, "ask_size": 100_000, "bid_size": 100_000}]} for m in MARKETS]

            def features(store, market, cfg, start, end):
                return pd.DataFrame({"x": [1.0], "5_volatility": [.005]},
                                    index=pd.DatetimeIndex([end.floor("5min")]))

            def predict(directory, manifest, market, frame):
                p = [.01, .98, .01]
                if market == "KRW-DOGE":
                    p = [.01, .01, .98] if doge_signal == 2 else [.98, .01, .01]
                return np.array([p])

            with patch.object(engine, "ROOT", root), patch("engine.utc", side_effect=clock), \
                 patch("engine.public_get", side_effect=quotes), patch("engine.load_market_features", side_effect=features), \
                 patch("engine.predict_package", side_effect=predict):
                bought = engine.paper_cycle(config, now)
                self.assertEqual(set(bought["positions"]), {"KRW-DOGE"})
                self.assertAlmostEqual(bought["cash"], 8_750_000)
                self.assertEqual(set(bought["prices"]), set(MARKETS))
                now += pd.Timedelta(minutes=5)
                doge_signal = 0
                sold = engine.paper_cycle(config, now)
            self.assertEqual(sold["positions"], {})
            self.assertEqual(sold["completed_trades"], 1)
            self.assertIsNone(sold["model_error"])
            self.assertTrue(sold["paper_only"])
            self.assertLess(sold["cash"], 10_000_000)
            self.assertEqual([(e["market"], e["side"]) for e in sold["trades"]],
                             [("KRW-DOGE", "buy"), ("KRW-DOGE", "sell")])

    def test_missing_doge_quote_is_rejected_instead_of_trading_four_markets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = isolated_config(root)
            quotes = [{"market": m} for m in MARKETS[:-1]]
            with patch.object(engine, "ROOT", root), patch("engine.public_get", return_value=quotes):
                with self.assertRaisesRegex(ValueError, "다섯 코인"):
                    engine.paper_cycle(config, "2026-10-26T01:00:00Z")


if __name__ == "__main__":
    unittest.main()
