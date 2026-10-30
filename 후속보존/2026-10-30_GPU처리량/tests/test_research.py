"""중립 진단·연속 연구·시간순 독립 구간·지평 청산 회귀검사."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import engine
from test_engine import candles, settings


def research_config():
    config = engine.load_config()
    config["training"].update(holdout_start="2026-10-19T00:00:00Z",
                              holdout_end="2026-10-20T00:00:00Z", holdout_days=1,
                              tune_days=2, calibration_days=2)
    return config


class ResearchTests(unittest.TestCase):
    def test_two_days_means_two_usable_days_when_start_falls_inside_holdout(self):
        config = research_config()
        plan = engine.experiment_plan(config, live=True, now="2026-10-21T05:00:00Z")
        left, right = plan["validation"]
        hs, he = plan["holdout"]
        removed = max(pd.Timedelta(0), min(right, he) - max(left, hs))
        self.assertEqual(right - left - removed, pd.Timedelta(days=2))

    def test_host_controlled_train_cycle_keeps_five_stages_separate_and_holds_pointer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = research_config()
            config["training"].update(candidate_depths=[3], min_split_rows=10)
            for key, folder in (("data_dir", "data"), ("models_dir", "models"), ("reports_dir", "reports")):
                config[key] = str(root / "work" / folder)
            raw = candles(10000, start="2026-09-20")
            price = 100000 * (1 + .025 * np.sin(np.arange(len(raw)) / 10))
            raw["open"] = raw["close"] = price
            raw["high"], raw["low"] = price + 100, price - 100
            features = engine.build_features({5: raw})
            actual_utc = engine.utc

            def fixture_clock(value=None):
                return actual_utc("2026-10-24T00:00:00Z" if value is None else value)

            fitted = []
            policies = []

            def fit(train, tune, columns, depth, cfg, **kwargs):
                fitted.append((train, tune))
                return Mock(), {"best_iteration": 1, "actual_device": "cuda:0"}

            def predict(model, frame, columns, best_iteration):
                return np.tile([.05, .9, .05], (len(frame), 1))

            def policy(frame, probabilities, settings, market, minimum, horizon):
                policies.append(frame)
                return {"confidence": 1.01, "edge": 1.01, "eligible": False,
                        "horizon_minutes": horizon, "trials": [{"controlled_test": True}]}

            # 적합/예측만 통제하며 기간 분할·감사·진단·승격·원자적 보고서는 실제 코드로 실행합니다.
            with patch.object(engine, "ROOT", root), patch("engine.CandleStore"), \
                 patch("engine.load_market_features", return_value=features), patch("engine.fit_gpu", side_effect=fit), \
                 patch("engine.predict_booster", side_effect=predict), patch("engine.tune_rule", side_effect=policy), \
                 patch("engine.utc", side_effect=fixture_clock):
                report = engine.train_cycle(config, live=True)
            self.assertEqual(len(fitted), len(engine.MARKETS))
            self.assertEqual(len(policies), len(engine.MARKETS))
            self.assertFalse(report["promotion"]["promote"])
            self.assertFalse((Path(config["models_dir"]) / "current.json").exists())
            self.assertEqual(report["research"]["schema_version"], 2)
            self.assertEqual(report["research"]["candidate_index"], 1)
            for market, fit_pair, policy_frame in zip(engine.MARKETS, fitted, policies):
                splits = report["markets"][market]["splits"]
                self.assertLess(fit_pair[0].label_end.max(), fit_pair[1].index.min())
                self.assertLess(fit_pair[1].label_end.max(), engine.utc(splits["calibration"]["start"]))
                self.assertLess(engine.utc(splits["calibration"]["end"]), policy_frame.index.min())
                self.assertLess(policy_frame.label_end.max(), engine.utc(splits["validation"]["start"]))
                self.assertEqual(report["markets"][market]["validation"]["support"], splits["validation"]["rows"])
                self.assertIn("rule_trials", report["markets"][market]["diagnostics"])

    def test_continuous_config_allows_zero_but_rejects_negative_and_boolean(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            engine.atomic_json(path, {"training": {"retrain_seconds": 0}})
            self.assertEqual(engine.load_config(path)["training"]["retrain_seconds"], 0)
            for value in (-1, True):
                engine.atomic_json(path, {"training": {"retrain_seconds": value}})
                with self.assertRaises(ValueError):
                    engine.load_config(path)

    def test_research_cursor_survives_restart_and_changes_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            config = research_config()
            config["reports_dir"] = temp
            self.assertTrue(hasattr(engine, "next_research_candidate"), "지속 후보 커서가 없습니다")
            first, first_meta = engine.next_research_candidate(config)
            second, second_meta = engine.next_research_candidate(config)
            self.assertEqual(second_meta["sequence"], first_meta["sequence"] + 1)
            self.assertNotEqual(first_meta["candidate_id"], second_meta["candidate_id"])
            self.assertNotEqual(first["training"], second["training"])
            self.assertEqual(config["training"]["label_threshold"], 0.006)

    def test_bounded_class_weights_only_depend_on_training_labels(self):
        self.assertTrue(hasattr(engine, "training_weights"), "학습 가중치가 없습니다")
        index = pd.date_range("2026-01-01", periods=102, freq="5min", tz="UTC")
        frame = pd.DataFrame({"label": [0] + [1] * 100 + [2]}, index=index)
        config = research_config()
        config["training"]["class_weight_mode"] = "bounded_inverse_sqrt"
        weights, evidence = engine.training_weights(frame, config["training"])
        self.assertEqual(evidence["class_counts"], [1, 100, 1])
        self.assertEqual(evidence["class_factors"], [3.0, 1.0, 3.0])
        recency = engine.recency_weights(index, config["training"]["half_life_days"])
        factors = weights / recency
        self.assertAlmostEqual(factors[0] / factors[1], 3, places=5)
        self.assertAlmostEqual(float(weights.mean()), 1, places=5)

    def test_five_ordered_segments_exclude_locked_labels_and_extend_windows(self):
        config = research_config()
        plan = engine.experiment_plan(config, live=True, now="2026-10-21T05:00:00Z")
        names = ["train", "tune", "calibration", "rule", "validation"]
        self.assertEqual(list(plan), names + ["holdout"])
        for left, right in zip(names, names[1:]):
            self.assertLessEqual(plan[left][1], plan[right][0])
        frame = engine.label_frame(engine.build_features({5: candles(8000, start="2026-09-25")}), 12)
        self.assertTrue(hasattr(engine, "research_splits"), "보호 구간 분할이 없습니다")
        split = engine.research_splits(frame, plan, config)
        hs, he = plan["holdout"]
        for name, part in split.items():
            self.assertGreaterEqual(len(part), config["training"]["min_split_rows"])
            self.assertTrue(((part.label_end < hs) | (part.index >= he)).all())
            self.assertTrue((part.label_end < plan[name][1]).all())

    def test_consumed_final_fails_before_fit_or_model_directory_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            config = research_config()
            for key in ("data_dir", "models_dir", "reports_dir"):
                config[key] = str(Path(temp) / key)
            audit = engine.ExperimentAudit(Path(config["data_dir"]) / "experiment_audit.sqlite3")
            audit.reserve_final("already-consumed", *engine.experiment_plan(config)["holdout"], "original")
            with patch("engine.fit_gpu") as fit:
                with self.assertRaisesRegex(RuntimeError, "이미 소비"):
                    engine.train_cycle(config, final=True)
            fit.assert_not_called()
            self.assertFalse((Path(config["models_dir"]) / "versions").exists())

    def test_probability_diagnostics_distinguish_argmax_and_blocked_signals(self):
        self.assertTrue(hasattr(engine, "probability_diagnostics"), "확률 진단이 없습니다")
        p = np.array([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        result = engine.probability_diagnostics(p, {"confidence": 1.01, "edge": 1.01})
        self.assertEqual(result["argmax_counts"], [0, 1, 1])
        self.assertEqual(result["signal_counts"], [0, 2, 0])
        self.assertEqual(result["filter_reasons"]["no_trade_policy"], 2)
        self.assertEqual(len(result["probability_quantiles"]), 3)

    def test_rule_trials_preserved_without_forcing_trades(self):
        frame = engine.build_features({5: candles(30)})
        p = np.tile([0.01, 0.98, 0.01], (len(frame), 1))
        result = engine.tune_rule(frame, p, settings(), "KRW-BTC")
        self.assertIn("trials", result, "정책별 탈락 근거가 없습니다")
        self.assertEqual(len(result["trials"]), 13)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["confidence"], 1.01)
        self.assertTrue(all(v["trading"]["executions"] == 0 for v in result["trials"]))
        self.assertTrue(all(v["rejection_reasons"] for v in result["trials"]))

    def test_promotion_requires_actual_positive_closed_trade_evidence(self):
        self.assertTrue(hasattr(engine, "trading_eligibility"), "운용 수익 게이트가 없습니다")
        for trade in ({"net_return": 0, "MDD": 0, "trades": 0},
                      {"net_return": 0.1, "MDD": 0, "trades": 0},
                      {"net_return": -0.1, "MDD": 0.1, "trades": 10}):
            self.assertFalse(engine.trading_eligibility(trade, 3)["eligible"])
        self.assertTrue(engine.trading_eligibility({"net_return": 0.01, "MDD": 0.001, "trades": 3}, 3)["eligible"])

    def test_no_current_model_does_not_promote_unprofitable_first_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            config = research_config()
            config["models_dir"] = temp
            frame = engine.build_features({5: candles(3)})
            frame["label"] = [0, 1, 2]
            result = engine.assess_promotion(config, {"KRW-BTC": frame},
                                             {"KRW-BTC": np.full((3, 3), 1 / 3)},
                                             {"KRW-BTC": {"confidence": 1.01, "edge": 1.01}})
            self.assertFalse(result["promote"])
            self.assertFalse(result["eligibility"]["eligible"])

    def test_manifest_legacy_guard_does_not_reinterpret_old_policy(self):
        self.assertTrue(hasattr(engine, "deployment_guard"), "구버전 안전 보류가 없습니다")
        self.assertIsNotNone(engine.deployment_guard({"run": "old"}))
        manifest = {"schema_version": 2, "deployment": {"eligible": True},
                    "execution_policy": "horizon_or_risk_v2"}
        self.assertIsNone(engine.deployment_guard(manifest))


class HorizonTests(unittest.TestCase):
    def test_missing_next_open_or_exact_horizon_cannot_stretch_label_time(self):
        frame = engine.build_features({5: candles(8)})
        for removed in (1, 2):
            with self.subTest(removed=removed):
                gapped = frame.drop(frame.index[removed])
                labelled = engine.label_frame(gapped, horizon=2)
                self.assertTrue(pd.isna(labelled.label.iloc[0]))

    def test_budget_and_external_buy_blocks_are_not_reported_as_model_neutral(self):
        portfolio = engine.Portfolio(settings())
        portfolio.process("2026-01-01", {"KRW-BTC": 100000}, {"KRW-BTC": 2}, allow_buys=False)
        self.assertIn("decision_diagnostics", portfolio.state)
        self.assertEqual(portfolio.state["decision_diagnostics"]["KRW-BTC"]["reason"], "buys_disabled")
        portfolio = engine.Portfolio(settings())
        portfolio.state["cash"] = 1000
        portfolio.process("2026-01-01T00:05:00Z", {"KRW-BTC": 100000}, {"KRW-BTC": 2})
        self.assertEqual(portfolio.state["decision_diagnostics"]["KRW-BTC"]["reason"], "below_min_order")

    def test_live_expiry_persists_and_sells_with_buys_blocked(self):
        portfolio = engine.Portfolio(settings())
        self.assertIn("entry_expiries", __import__("inspect").signature(portfolio.process).parameters,
                      "지평 청산이 없습니다")
        expiry = "2026-01-01T01:00:00Z"
        portfolio.process("2026-01-01T00:00:00Z", {"KRW-BTC": 100000}, {"KRW-BTC": 2},
                          entry_expiries={"KRW-BTC": expiry})
        restored = engine.Portfolio(settings(), json.loads(json.dumps(portfolio.state)))
        self.assertEqual(restored.state["positions"]["KRW-BTC"]["expires_at"], engine.iso(expiry))
        events = restored.process(expiry, {"KRW-BTC": 100500}, {}, allow_buys=False)
        self.assertEqual(events[0]["reason"], "time_exit")
        self.assertEqual(restored.state["positions"], {})

    def test_backtest_horizon_uses_expiry_bar_close_not_earlier_open(self):
        frame = engine.build_features({5: candles(5)})
        frame[["open", "close"]] = 100000
        frame.iloc[2, frame.columns.get_loc("close")] = 100600
        p = np.tile([0.01, 0.98, 0.01], (5, 1))
        p[0] = [0.01, 0.01, 0.98]
        result = engine.backtest_portfolio({"KRW-BTC": frame}, {"KRW-BTC": p},
                                           {"KRW-BTC": {"confidence": 0.5, "edge": 0.1, "horizon_minutes": 10}}, settings())
        self.assertEqual(result["trades"], 1)
        self.assertGreater(result["net_return"], 0)
        self.assertEqual(result["exit_reasons"]["time_exit"], 1)


class PromotionIncumbentTests(unittest.TestCase):
    def compare(self, root, validated, profitable=True, min_closed_trades=3):
        config = research_config()
        config["training"]["min_closed_trades"] = min_closed_trades
        config["models_dir"] = str(root)
        engine.atomic_json(root / "current.json", {"run": "previous", "manifest_hash": "controlled"})
        frame = engine.build_features({5: candles(7)})
        frame["open"] = frame["close"] = [100000, 100000, 101000, 100000, 101000, 100000, 101000]
        frame["label"] = 1
        probabilities = np.array([[.2, .2, .6], [.6, .2, .2]] * 3 + [[.1, .8, .1]])
        if not profitable:
            probabilities[:] = [.01, .98, .01]
        manifest = {"run": "previous", "config": config,
                    "packages": {"KRW-BTC": {"rule": {"confidence": 1.01, "edge": 1.01}}}}
        if validated:
            manifest.update(schema_version=2, execution_policy=engine.EXECUTION_POLICY,
                            deployment={"eligible": validated is True})
        with patch("engine.load_current", return_value=(root, manifest)), \
             patch("engine.predict_package", return_value=np.tile([.005, .99, .005], (7, 1))) as prediction:
            result = engine.assess_promotion(config, {"KRW-BTC": frame}, {"KRW-BTC": probabilities},
                                             {"KRW-BTC": {"confidence": .5, "edge": .15}})
        return result, prediction

    def test_legacy_lower_logloss_cannot_veto_profitable_candidate_against_cash(self):
        with tempfile.TemporaryDirectory() as temp:
            result, prediction = self.compare(Path(temp), validated=False)
        self.assertTrue(result["eligibility"]["eligible"])
        self.assertGreater(result["candidate"]["net_return"], 0)
        self.assertTrue(result["promote"])
        self.assertEqual(result["comparison_baseline"], "cash")
        self.assertEqual(result["excluded_incumbent_run"], "previous")
        prediction.assert_not_called()

    def test_legacy_cash_comparison_still_rejects_no_trade_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            result, prediction = self.compare(Path(temp), validated=False, profitable=False)
        self.assertFalse(result["promote"])
        self.assertEqual(result["comparison_baseline"], "cash")
        prediction.assert_not_called()

    def test_unvalidated_v2_uses_cash_without_incumbent_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            result, prediction = self.compare(Path(temp), validated="unvalidated")
        self.assertTrue(result["promote"])
        self.assertEqual(result["excluded_incumbent_reason"], "unvalidated_or_unprofitable_model")
        prediction.assert_not_called()

    def test_cash_baseline_does_not_relax_minimum_closed_trades(self):
        with tempfile.TemporaryDirectory() as temp:
            result, prediction = self.compare(Path(temp), validated=False, min_closed_trades=4)
        self.assertGreater(result["candidate"]["net_return"], 0)
        self.assertFalse(result["promote"])
        self.assertIn("insufficient_closed_trades", result["eligibility"]["rejection_reasons"])
        prediction.assert_not_called()

    def test_validated_current_keeps_strict_probability_and_trading_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            result, prediction = self.compare(Path(temp), validated=True)
        self.assertTrue(result["eligibility"]["eligible"])
        self.assertFalse(result["promote"])
        self.assertFalse(result["logloss_not_worse"])
        prediction.assert_called_once()


class InferenceDiagnosticTests(unittest.TestCase):
    def test_duplicate_and_stale_frames_have_separate_preserved_reasons(self):
        now = engine.utc("2026-10-21T10:00:00Z")
        config = research_config()
        portfolio = engine.Portfolio(config["paper"])
        stamp = now.floor("5min")
        portfolio.state["signal_times"] = {"KRW-BTC": engine.iso(stamp)}
        manifest = {"schema_version": 2, "execution_policy": "horizon_or_risk_v2",
                    "deployment": {"eligible": True}, "run": "new", "config": config,
                    "feature_columns": ["x"], "packages": {m: {"rule": {"confidence": 0.5, "edge": 0.1}} for m in engine.MARKETS}}
        frame = pd.DataFrame({"x": [1], "5_volatility": [0.01]}, index=pd.DatetimeIndex([stamp]))
        with patch("engine.load_current", return_value=(Path("."), manifest)), \
             patch("engine.load_market_features", return_value=frame), \
             patch("engine.predict_package", return_value=np.array([[0.01, 0.98, 0.01]])):
            engine.paper_signals(config, None, portfolio, now)
        self.assertIn("inference_diagnostics", portfolio.state)
        self.assertEqual(portfolio.state["inference_diagnostics"]["KRW-BTC"]["reason"], "duplicate_signal_bar")
        old = frame.copy()
        old.index = old.index - pd.Timedelta(minutes=20)
        with patch("engine.load_current", return_value=(Path("."), manifest)), \
             patch("engine.load_market_features", return_value=old), patch("engine.predict_package") as predict:
            engine.paper_signals(config, None, portfolio, now)
        predict.assert_not_called()
        self.assertEqual(portfolio.state["inference_diagnostics"]["KRW-BTC"]["reason"], "stale_features")


if __name__ == "__main__":
    unittest.main()
