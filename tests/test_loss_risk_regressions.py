"""손절 재진입·시점별 손절·호가 추가비용의 통제 회귀검사."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import engine


MARKET = "KRW-DOGE"
START = pd.Timestamp("2026-10-28T00:00:00Z")


def settings():
    return copy.deepcopy(engine.DEFAULT_CONFIG["paper"])


def book(ask, bid, ask_size=1e9, bid_size=1e9):
    return [{"ask_price": ask, "bid_price": bid,
             "ask_size": ask_size, "bid_size": bid_size}]


def stopped_portfolio(expiry="2026-10-28T02:00:00+00:00", start=START):
    portfolio = engine.Portfolio(settings())
    portfolio.state["risk_horizon_minutes"] = {MARKET: 120}
    portfolio.process(start, {MARKET: 100}, {MARKET: 2})
    if expiry is not None:
        portfolio.state["positions"][MARKET]["expires_at"] = expiry
    events = portfolio.process(start + pd.Timedelta(minutes=1), {MARKET: 97}, {MARKET: 1})
    return portfolio, events


def backtest_frame(volatility=True):
    index = pd.date_range(START, periods=5, freq="5min")
    frame = pd.DataFrame({"open": [100, 100, 99, 99, 99],
                          "close": [100, 100, 99, 99, 99],
                          "open_time": index - pd.Timedelta(minutes=5)}, index=index)
    if volatility:
        frame["5_volatility"] = 0.001
    return frame


class ReentryRiskTests(unittest.TestCase):
    def test_stop_blocks_new_signal_until_original_prediction_expiry(self):
        portfolio, events = stopped_portfolio()
        self.assertEqual(events[0]["reason"], "stop_loss")
        before = portfolio.state["cash"]
        events = portfolio.process(START + pd.Timedelta(minutes=2), {MARKET: 97}, {MARKET: 2})
        self.assertEqual(events, [])
        self.assertEqual(portfolio.state["cash"], before)
        self.assertEqual(portfolio.state["decision_diagnostics"][MARKET]["reason"], "stop_reentry_cooldown")

    def test_restart_preserves_cooldown_cash_and_initial_capital(self):
        portfolio, _ = stopped_portfolio()
        saved = json.loads(json.dumps(portfolio.state))
        restored = engine.Portfolio(settings(), saved)
        events = restored.process(START + pd.Timedelta(minutes=3), {MARKET: 97}, {MARKET: 2})
        self.assertEqual(events, [])
        self.assertEqual(restored.state["cash"], saved["cash"])
        self.assertEqual(restored.state["initial_cash"], 10000000)

    def test_cooldown_releases_at_expiry_and_duplicate_cycle_cannot_buy_twice(self):
        portfolio, _ = stopped_portfolio()
        stamp = START + pd.Timedelta(hours=2)
        events = portfolio.process(stamp, {MARKET: 97}, {MARKET: 2})
        self.assertEqual([e["side"] for e in events], ["buy"])
        saved = copy.deepcopy(portfolio.state)
        self.assertEqual(portfolio.process(stamp, {MARKET: 97}, {MARKET: 2}), [])
        self.assertEqual(portfolio.state, saved)

    def test_day_boundary_does_not_reset_stop_cooldown(self):
        start = pd.Timestamp("2026-10-27T23:58:00Z")
        portfolio, _ = stopped_portfolio("2026-10-28T01:58:00Z", start)
        events = portfolio.process(START, {MARKET: 97}, {MARKET: 2})
        self.assertEqual(events, [])
        self.assertEqual(portfolio.state["day"], "2026-10-28")

    def test_legacy_position_without_expiry_uses_configured_horizon(self):
        portfolio, _ = stopped_portfolio(None)
        events = portfolio.process(START + pd.Timedelta(minutes=120), {MARKET: 97}, {MARKET: 2})
        self.assertEqual(events, [])
        events = portfolio.process(START + pd.Timedelta(minutes=121), {MARKET: 97}, {MARKET: 2})
        self.assertEqual([e["side"] for e in events], ["buy"])

    def test_invalid_expiry_blocks_reentry_but_does_not_block_stop_sale(self):
        portfolio, events = stopped_portfolio("잘못된-시각")
        self.assertEqual([e["reason"] for e in events], ["stop_loss"])
        self.assertNotIn(MARKET, portfolio.state["positions"])
        events = portfolio.process(START + pd.Timedelta(minutes=2), {MARKET: 97}, {MARKET: 2})
        self.assertEqual(events, [])
        self.assertEqual(portfolio.state["decision_diagnostics"][MARKET]["reason"], "invalid_reentry_expiry")

    def test_partial_stop_liquidates_again_despite_cooldown_and_buys_disabled(self):
        portfolio = engine.Portfolio(settings())
        portfolio.process(START, {MARKET: 100}, {MARKET: 2},
                          entry_expiries={MARKET: "2026-10-28T02:00:00Z"})
        first = portfolio.process(START + pd.Timedelta(minutes=1), {MARKET: 97}, {MARKET: 2},
                                  books={MARKET: book(98, 97, bid_size=1)}, allow_buys=False)
        self.assertEqual([e["side"] for e in first], ["sell"])
        second = portfolio.process(START + pd.Timedelta(minutes=2), {MARKET: 97}, {MARKET: 2},
                                   books={MARKET: book(98, 97)}, allow_buys=False)
        self.assertEqual([e["side"] for e in second], ["sell"])
        self.assertNotIn(MARKET, portfolio.state["positions"])

    def test_daily_loss_guard_still_blocks_after_cooldown_expires(self):
        portfolio, _ = stopped_portfolio()
        portfolio.state["new_buys_blocked"] = True
        daily_start = portfolio.state["daily_start_equity"]
        events = portfolio.process(START + pd.Timedelta(hours=3), {MARKET: 97}, {MARKET: 2})
        self.assertEqual(events, [])
        self.assertEqual(portfolio.state["decision_diagnostics"][MARKET]["reason"], "daily_loss_limit")
        self.assertEqual(portfolio.state["daily_start_equity"], daily_start)


class QuoteCostRiskTests(unittest.TestCase):
    def process_book(self, units, threshold=0.006, paper_settings=None):
        portfolio = engine.Portfolio(paper_settings or settings())
        portfolio.state["entry_cost_limits"] = {MARKET: threshold}
        events = portfolio.process(START, {MARKET: 100}, {MARKET: 2}, books={MARKET: units})
        return portfolio, events

    def test_actual_doge_spread_exceeding_net_label_budget_blocks_entry(self):
        portfolio, events = self.process_book(book(107, 106))
        self.assertEqual(events, [])
        diagnostic = portfolio.state["decision_diagnostics"][MARKET]
        self.assertEqual(diagnostic["reason"], "execution_cost_exceeds_label_budget")
        self.assertAlmostEqual(diagnostic["roundtrip_cost"], 0.01132512298031818)
        self.assertAlmostEqual(diagnostic["incremental_quote_drag"], 1 / 107)

    def test_base_fee_slippage_already_in_net_labels_are_not_charged_twice(self):
        portfolio, events = self.process_book(book(100, 99.45))
        self.assertEqual([e["side"] for e in events], ["buy"])
        diagnostic = portfolio.state["decision_diagnostics"][MARKET]
        self.assertGreater(diagnostic["roundtrip_cost"], 0.006)
        self.assertAlmostEqual(diagnostic["net_label_budget_after_book_cost"], 0.000467, places=8)

    def test_zero_spread_remains_eligible_even_when_base_cost_exceeds_threshold(self):
        paper = settings()
        paper.update(fee=0.01, slippage=0.01)
        portfolio, events = self.process_book(book(100, 100), threshold=0.001, paper_settings=paper)
        self.assertEqual([e["side"] for e in events], ["buy"])
        self.assertAlmostEqual(portfolio.state["decision_diagnostics"][MARKET]["net_label_budget_after_book_cost"], 0.001)

    def test_depth_cost_not_only_best_bid_spread_is_checked(self):
        units = book(100, 100, bid_size=1) + book(101, 98)
        portfolio, events = self.process_book(units)
        self.assertEqual(events, [])
        self.assertEqual(portfolio.state["decision_diagnostics"][MARKET]["reason"], "execution_cost_exceeds_label_budget")

    def test_insufficient_exit_depth_blocks_entry(self):
        portfolio, events = self.process_book(book(100, 100, bid_size=1))
        self.assertEqual(events, [])
        self.assertEqual(portfolio.state["decision_diagnostics"][MARKET]["reason"], "insufficient_exit_liquidity")

    def test_cost_guard_does_not_block_existing_position_sell(self):
        portfolio = engine.Portfolio(settings())
        portfolio.process(START, {MARKET: 100}, {MARKET: 2})
        portfolio.state["entry_cost_limits"] = {MARKET: 0.001}
        events = portfolio.process(START + pd.Timedelta(minutes=1), {MARKET: 100}, {MARKET: 0},
                                   books={MARKET: book(107, 100)})
        self.assertEqual([e["side"] for e in events], ["sell"])

    def test_label_threshold_is_net_of_both_directions_fee_and_slippage(self):
        frame = backtest_frame(False)
        frame["open"] = 100.0
        frame["close"] = 100.75
        labelled = engine.label_frame(frame, horizon=1, threshold=0.006)
        self.assertAlmostEqual(labelled.net_forward_return.iloc[0], 0.00548701399301188)
        self.assertEqual(labelled.label.iloc[0], 1)


class BacktestRiskParityTests(unittest.TestCase):
    def run_backtest(self, frame):
        probs = np.tile([0.01, 0.98, 0.01], (len(frame), 1))
        probs[0] = [0.01, 0.01, 0.98]
        return engine.backtest_portfolio({MARKET: frame}, {MARKET: probs},
                                          {MARKET: {"confidence": 0.5, "edge": 0.1, "horizon_minutes": 120}}, settings())

    def test_backtest_uses_live_volatility_stop_instead_of_fixed_two_percent(self):
        result = self.run_backtest(backtest_frame())
        self.assertEqual(result["exit_reasons"]["stop_loss"], 1)

    def test_next_candle_volatility_cannot_widen_stop_before_open(self):
        frame = backtest_frame()
        frame.iloc[2, frame.columns.get_loc("5_volatility")] = 0.2
        result = self.run_backtest(frame)
        self.assertEqual(result["exit_reasons"]["stop_loss"], 1)

    def test_minimal_fixture_without_volatility_explicitly_uses_settings_stop(self):
        result = self.run_backtest(backtest_frame(False))
        self.assertEqual(result["exit_reasons"]["stop_loss"], 0)

    def test_feature_frame_missing_base_volatility_is_not_silently_downgraded(self):
        frame = backtest_frame().rename(columns={"5_volatility": "60_volatility"})
        with self.assertRaisesRegex(ValueError, "변동성"):
            self.run_backtest(frame)

    def test_negative_feature_volatility_is_rejected(self):
        frame = backtest_frame()
        frame.iloc[0, frame.columns.get_loc("5_volatility")] = -0.1
        with self.assertRaisesRegex(ValueError, "변동성"):
            self.run_backtest(frame)

    def test_initial_warmup_missing_volatility_has_explicit_settings_fallback(self):
        frame = backtest_frame().iloc[:3].copy()
        frame.iloc[:2, frame.columns.get_loc("5_volatility")] = np.nan
        result = self.run_backtest(frame)
        self.assertEqual(result["volatility_fallback_rows"][MARKET], 2)
        self.assertEqual(result["exit_reasons"]["stop_loss"], 0)

    def test_missing_volatility_after_warmup_is_rejected(self):
        frame = backtest_frame()
        frame.iloc[2, frame.columns.get_loc("5_volatility")] = np.nan
        with self.assertRaisesRegex(ValueError, "변동성"):
            self.run_backtest(frame)

    def test_incumbent_and_candidate_comparison_both_apply_shared_risk_overlay(self):
        frame = backtest_frame()
        frame["label"] = 1
        probs = np.tile([0.01, 0.98, 0.01], (len(frame), 1))
        probs[0] = [0.01, 0.01, 0.98]
        rule = {"confidence": 0.5, "edge": 0.1, "horizon_minutes": 120}
        with tempfile.TemporaryDirectory() as temp:
            config = engine.load_config()
            config["models_dir"] = temp
            engine.atomic_json(Path(temp) / "current.json", {"controlled_fixture": True})
            manifest = {"schema_version": 2, "execution_policy": "horizon_or_risk_v2",
                        "deployment": {"eligible": True}, "run": "통제된-기존모델", "config": config,
                        "packages": {MARKET: {"rule": rule}}}
            actual_backtest = engine.backtest_portfolio
            observed = []

            def evaluate(*args):
                result = actual_backtest(*args)
                observed.append(result)
                return result

            with patch.object(engine, "load_current", return_value=(Path(temp), manifest)), \
                 patch.object(engine, "predict_package", return_value=probs), \
                 patch.object(engine, "backtest_portfolio", side_effect=evaluate):
                result = engine.assess_promotion(config, {MARKET: frame}, {MARKET: probs}, {MARKET: rule})
        self.assertNotIn("comparison_error", result)
        self.assertEqual(len(observed), 2)
        self.assertEqual([r["exit_reasons"]["stop_loss"] for r in observed], [1, 1])
        self.assertEqual(observed[0]["risk_overlay"], observed[1]["risk_overlay"])
        self.assertEqual(result["candidate"]["net_return"], result["incumbent"]["net_return"])


class LiveRiskWiringTests(unittest.TestCase):
    def test_model_net_budget_is_used_and_duplicate_signal_stays_consumed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = engine.load_config()
            config["paper"]["start_time"] = "2026-10-27T00:00:00Z"
            for key, folder in (("data_dir", "data"), ("models_dir", "models"), ("reports_dir", "reports")):
                config[key] = str(root / "work" / folder)
            # 운영 설정 0.6%와 달리 실제 선택 모델의 0.3%를 사용해야 합니다.
            model_config = copy.deepcopy(config)
            model_config["training"].update(label_threshold=0.003, horizon=24)
            manifest = {"run": "통제된-기존모델", "schema_version": 2,
                        "execution_policy": "horizon_or_risk_v2", "deployment": {"eligible": True},
                        "config": model_config, "feature_columns": ["x"],
                        "packages": {m: {"rule": {"confidence": 0.5, "edge": 0.1}} for m in engine.MARKETS}}
            frame = pd.DataFrame({"x": [1.0], "5_volatility": [0.001]}, index=pd.DatetimeIndex([START]))
            quote_state = {"bid": 99.6}

            def quotes(*args):
                return [{"market": m, "timestamp": int(START.timestamp() * 1000),
                         "orderbook_units": book(100, quote_state["bid"])} for m in engine.MARKETS]

            def predict(directory, package, market, rows):
                return np.array([[0.01, 0.01, 0.98] if market == MARKET else [0.01, 0.98, 0.01]])

            real_utc = engine.utc
            with patch.object(engine, "ROOT", root), patch.object(engine, "public_get", side_effect=quotes), \
                 patch.object(engine, "load_current", return_value=(root, manifest)), \
                 patch.object(engine, "load_market_features", return_value=frame), \
                 patch.object(engine, "predict_package", side_effect=predict), \
                 patch.object(engine, "utc", side_effect=lambda value=None: START if value is None else real_utc(value)):
                first = engine.paper_cycle(config, START)
                self.assertNotIn(MARKET, first["positions"])
                self.assertEqual(first["decision_diagnostics"][MARKET]["reason"], "execution_cost_exceeds_label_budget")
                self.assertEqual(first["risk_horizon_minutes"][MARKET], 120)
                self.assertIsNone(first["deployment_guard"])
                quote_state["bid"] = 100
                with patch.object(engine, "utc", side_effect=lambda value=None: START + pd.Timedelta(seconds=1) if value is None else real_utc(value)):
                    second = engine.paper_cycle(config, START + pd.Timedelta(seconds=1))
                self.assertNotIn(MARKET, second["positions"])
                self.assertEqual(second["inference_diagnostics"][MARKET]["reason"], "duplicate_signal_bar")


if __name__ == "__main__":
    unittest.main()
