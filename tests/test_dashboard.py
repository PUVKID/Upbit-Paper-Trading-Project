"""대시보드의 읽기 전용 경계와 실제 스냅숏 투영을 검사합니다."""

import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

import dashboard


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def test_missing_is_pending_not_fixture(self):
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertIsNone(status["account"]["equity"])
        self.assertIsNone(status["validation"])
        self.assertIsNone(status["locked_test"])
        self.assertEqual(status["equity"], [])
        self.assertEqual(status["assets"], [])
        self.assertIsNone(status["policy"]["initial_capital"])
        self.assertFalse(status["policy"]["real_orders"])
        self.assertFalse(status["policy"]["notifications"])

    def test_snapshot_preserves_zero_and_real_values(self):
        self.write("work/paper/state.json", {"equity": 25_100_000, "realized_pnl": 0,
                   "assets": [{"symbol": "BTC", "quantity": .01, "price": 100}],
                   "equity_history": [{"timestamp": "2026-10-21T09:01:00Z", "equity": 25_100_000}]})
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertEqual(status["account"]["equity"], 25_100_000)
        self.assertEqual(status["account"]["realized_pnl"], 0)
        self.assertEqual(status["assets"][0]["quantity"], .01)
        self.assertEqual(len(status["equity"]), 1)

    def test_private_fields_and_argv_are_not_exposed(self):
        self.write("work/service/trainer.json", {"state": "실행 중", "argv": ["private"],
                   "error": "private", "api_key": "private", "environment": {"key": "private"}})
        self.write("work/reports/dashboard.json", {"api_key": "private", "account": {"equity": 1, "secret": "private"},
                   "trades": [{"symbol": "BTC", "secret": "private"}]})
        status = json.dumps(dashboard.SnapshotAdapter(self.root).status())
        self.assertNotIn("private", status)
        self.assertNotIn("argv", status)
        self.assertNotIn("api_key", status)

    def test_gpu_device_measurement_only(self):
        self.write("work/service/gpu.json", {"state": "실측", "gpu_name": "측정 GPU", "gpu_memory_used_mb": 100,
                   "gpu_utilization_pct": 0, "credential": "secret"})
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertEqual(status["gpu"]["gpu_name"], "측정 GPU")
        self.assertEqual(status["gpu"]["gpu_utilization_pct"], 0)
        self.assertNotIn("credential", status["gpu"])

    def test_validation_and_locked_test_are_separate(self):
        self.write("work/reports/training.json", {"validation": {"accuracy": .7}, "locked_test": {"accuracy": .6},
                   "class_distribution": {"buy": 4, "hold": 6}, "rounds": [{"round": 1, "accuracy": .7}]})
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertEqual(status["validation"]["accuracy"], .7)
        self.assertEqual(status["locked_test"]["accuracy"], .6)
        self.assertEqual(status["class_distribution"]["buy"], 4)

    def test_invalid_and_wrong_shape_are_pending(self):
        path = self.write("work/paper/state.json", {})
        path.write_text("{", encoding="utf-8")
        self.write("work/reports/dashboard.json", {"account": None, "training": [1], "data": "bad"})
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertIsNone(status["account"]["equity"])
        self.assertEqual(status["sources"]["work/paper/state.json"], "unavailable")

    def test_nonfinite_and_nested_values_are_not_forwarded(self):
        self.write("work/paper/state.json", {"equity": float("nan"), "cash": {"secret": "hidden"}})
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertIsNone(status["account"]["equity"])
        self.assertNotIn("cash", status["account"])
        json.dumps(status, allow_nan=False)

    def test_oversized_source_is_rejected(self):
        path = self.write("work/data/summary.json", {})
        path.write_bytes(b" " * (dashboard.MAX_JSON_BYTES + 1))
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertEqual(status["sources"]["work/data/summary.json"], "too_large")

    def test_rows_are_bounded(self):
        self.write("work/reports/dashboard.json", {"trades": [{"price": i} for i in range(1100)]})
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertEqual(len(status["trades"]), 1000)
        self.assertEqual(status["trades"][0]["price"], 100)

    def test_engine_nested_collection_schema(self):
        self.write("work/data/summary.json", {"updated_at": "2026-10-21T01:00:00Z",
                   "data": {"total_rows": 200, "backfill": True, "status": "수집 중"},
                   "datasets": [{"market": "KRW-BTC", "timeframe": "1", "rows": 200,
                                 "start": "2026-01-01T00:00:00Z", "end": "2026-10-21T00:00:00Z",
                                 "backfill_exhausted": False}]})
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertEqual(status["data"]["total_rows"], 200)
        self.assertEqual(status["data"]["state"], "수집 중")
        self.assertEqual(status["datasets"][0]["symbol"], "KRW-BTC")
        self.assertEqual(status["datasets"][0]["period_start"], "2026-01-01T00:00:00Z")

    def test_local_3d_assets_use_explicit_static_allowlist(self):
        self.assertEqual(dashboard.STATIC.get('/vendor.bundle.js'),
                         ('vendor.bundle.js', 'text/javascript; charset=utf-8'))
        self.assertEqual(dashboard.STATIC.get('/scene.js'),
                         ('scene.js', 'text/javascript; charset=utf-8'))
        for coin in ('btc', 'sol', 'eth', 'xrp'):
            self.assertEqual(dashboard.STATIC.get(f'/assets/{coin}.png'),
                             (f'assets/{coin}.png', 'image/png'))
        self.assertNotIn('/node_modules/three/package.json', dashboard.STATIC)

    def test_public_quotes_do_not_fabricate_positions(self):
        self.write('work/paper/state.json', {'initial_cash': 10_000_000,
            'cash': 10_000_000, 'positions': {}, 'assets': [],
            'updated_at': '2026-10-26T08:00:00Z',
            'prices': {'KRW-BTC': 123456.5, 'KRW-SOL': -1,
                       'KRW-ETH': 'invalid', 'KRW-UNKNOWN': 100}})
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertEqual(status['assets'], [])
        self.assertEqual(status['quotes'], [{'symbol': 'BTC', 'price': 123456.5,
            'quote_utc': '2026-10-26T08:00:00Z', 'source': '공개 호가 중간값'}])

    def test_doge_is_supported_without_fabricating_training_or_holdings(self):
        self.write('work/paper/state.json', {'initial_cash': 10_000_000,
            'cash': 10_000_000, 'positions': {}, 'assets': [],
            'updated_at': '2026-10-26T09:00:00Z', 'prices': {'KRW-DOGE': 188.75}})
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertEqual(status['policy']['symbols'], ['BTC', 'SOL', 'ETH', 'XRP', 'DOGE'])
        self.assertEqual(status['quotes'], [{'symbol': 'DOGE', 'price': 188.75,
            'quote_utc': '2026-10-26T09:00:00Z', 'source': '공개 호가 중간값'}])
        self.assertEqual(status['assets'], [])
        self.assertEqual(status['markets'], {})
        self.assertEqual(dashboard.STATIC['/assets/doge.png'], ('assets/doge.png', 'image/png'))

    def test_text_assets_utf8_and_font_bundled(self):
        web = dashboard.ROOT / "web"
        for name in ("index.html", "app.js", "style.css", "fonts/OFL.txt"):
            content = (web / name).read_bytes().decode("utf-8")
            self.assertNotIn("\ufffd", content)
        self.assertGreater((web / "fonts/NanumGothic-Bold.ttf").stat().st_size, 100_000)
        self.assertIn("장치 전체", (web / "app.js").read_text(encoding="utf-8"))

    def test_actual_market_metrics_progress_and_no_config_leak(self):
        self.write("work/reports/training.json", {
            "run": "run-final", "updated_at": "2026-10-21T01:00:00Z",
            "config": {"secret": "credential", "base_timeframe": 5},
            "periods": {"train": ["1926-01-01", "2026-01-01"]},
            "markets": {"KRW-BTC": {"validation": {"accuracy": .75, "macro_f1": .6,
                "balanced_accuracy": .65, "majority_baseline": .5, "support": 40,
                "class_distribution": [10, 20, 10], "start": "2026-10-17", "end": "2026-10-18"},
                "splits": {"train": {"rows": 100, "start": "2020-01-01", "end": "2026-10-15",
                                      "class_distribution": [30, 40, 30]}},
                "gpu": [{"actual_device": "cuda:0", "secret": "credential"}]}},
            "class_distribution": {"KRW-BTC": [10, 20, 10]},
            "locked_test": {"status": "최종 1회 평가 완료", "used_for_selection": False,
                "metrics": {"accuracy": .7, "support": 20},
                "markets": {"KRW-BTC": {"accuracy": .7, "support": 20}}}})
        self.write("work/service/training_progress.json", {"status": "running", "stage": "fit",
                   "market": "KRW-SOL", "candidate": 2, "completed_candidates": 4, "total_candidates": 12,
                   "error": "credential"})
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertEqual(status["markets"]["KRW-BTC"]["validation"]["baseline_accuracy"], .5)
        self.assertEqual(status["markets"]["KRW-BTC"]["train"]["period_start"], "2020-01-01")
        self.assertEqual(status["training"]["model_id"], "run-final")
        self.assertAlmostEqual(status["training"]["progress"], 1 / 3)
        self.assertEqual(status["training"]["market"], "KRW-SOL")
        self.assertEqual(status["locked_test"]["samples"], 20)
        self.assertEqual(status["markets"]["KRW-BTC"]["locked_test"]["accuracy"], .7)
        self.assertEqual(status["class_distribution"], {"down": 10, "flat": 20, "up": 10})
        self.assertNotIn("credential", json.dumps(status))
        self.assertNotIn("1926-01-01", json.dumps(status))

    def test_actual_account_aliases_and_lifetime_realized(self):
        self.write("engine-config.json", {"paper": {"fee": .01, "slippage": .02}, "secret": "credential"})
        self.write("work/paper/state.json", {"initial_cash": 1000, "cash": 500, "equity": 1082.12,
                   "net_return": .08212, "MDD": .03, "updated_at": "2026-10-21T09:00:00Z",
                   "positions": {"KRW-BTC": {"qty": 2, "cost": 550, "entry_price": 275}},
                   "prices": {"KRW-BTC": 300},
                   "assets": [{"market": "KRW-BTC", "qty": 2, "cost": 550, "entry_price": 275, "price": 300, "value": 600}],
                   "trades": [{"market": "KRW-BTC", "side": "buy", "qty": 2, "vwap": 275,
                               "fee_paid": 5, "reason": "signal"}]})
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertEqual(status["account"]["initial_capital"], 1000)
        self.assertAlmostEqual(status["account"]["unrealized_pnl"], 32.12)
        self.assertAlmostEqual(status["account"]["realized_pnl"], 50)
        self.assertAlmostEqual(status["account"]["return_pct"], 8.212)
        self.assertEqual(status["assets"][0]["quantity"], 2)
        self.assertEqual(status["trades"][0]["price"], 275)
        self.assertEqual(status["trades"][0]["reason"], "모델 신호")

    def test_experiments_last_twenty_allowlist_and_backfill(self):
        for i in range(25):
            self.write(f"work/reports/experiments/20261021T{i:06d}.json", {
                "run": f"실험-{i}", "updated_at": "2026-10-21T00:00:00Z", "config": {"key": "credential"},
                "markets": {"KRW-BTC": {"validation": {"accuracy": .6, "support": 10}}},
                "training": {"status": "완료"}})
        self.write("work/service/backfill.json", {"state": "실행 중", "argv": ["credential"]})
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertEqual(len(status["rounds"]), 20)
        self.assertEqual(status["rounds"][0]["run"], "실험-5")
        self.assertEqual(status["rounds"][-1]["run"], "실험-24")
        self.assertEqual(status["services"]["backfill"]["state"], "실행 중")
        self.assertNotIn("credential", json.dumps(status))

    def test_no_position_zero_pnl_and_unknown_exit_cost_pending(self):
        self.write("work/paper/state.json", {"initial_cash": 1000, "cash": 1020,
                   "equity": 1020, "positions": {}, "trades": []})
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertEqual(status["account"]["realized_pnl"], 20)
        self.assertEqual(status["account"]["unrealized_pnl"], 0)
        self.write("work/paper/state.json", {"initial_cash": 1000, "equity": 1100,
                   "positions": {"KRW-BTC": {"qty": 1, "cost": 100}}, "prices": {"KRW-BTC": 200}})
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertIsNone(status["account"]["unrealized_pnl"])
        self.assertIsNone(status["account"]["realized_pnl"])

    def test_macro_metrics_not_averaged_and_unknown_markets_rejected(self):
        self.write("work/reports/training.json", {"markets": {
            "KRW-BTC": {"validation": {"accuracy": .5, "majority_baseline": .4, "support": 10, "macro_f1": .6}},
            "KRW-SOL": {"validation": {"accuracy": .9, "majority_baseline": .6, "support": 30, "macro_f1": .8}},
            "PRIVATE": {"validation": {"accuracy": 1, "secret": "credential"}}},
            "class_distribution": {"KRW-BTC": [2, 5, 3], "KRW-SOL": [10, 15, 5]}})
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertAlmostEqual(status["validation"]["accuracy"], .8)
        self.assertAlmostEqual(status["validation"]["baseline_accuracy"], .55)
        self.assertIsNone(status["validation"]["macro_f1"])
        self.assertEqual(status["class_distribution"], {"down": 12, "flat": 20, "up": 8})
        self.assertNotIn("PRIVATE", status["markets"])

    def test_candidate_and_current_model_are_distinct_and_promotion_is_projected(self):
        self.write("work/reports/training.json", {"run": "새로운-후보", "promotion": {
            "promote": False, "incumbent_run": "이전-운용", "reason": "확률/거래 지표 악화 또는 유의한 개선 없음; 현재 모델 유지",
            "candidate": {"logloss": 1.1, "net_return": .01, "MDD": .03, "secret": "credential"},
            "incumbent": {"logloss": .9, "net_return": .02, "MDD": .02},
            "comparison_error": {"message": "credential"}}})
        self.write("work/models/current.json", {"run": "이전-운용", "manifest_hash": "credential"})
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertEqual(status["training"]["model_id"], "새로운-후보")
        self.assertEqual(status["current_model"]["run"], "이전-운용")
        self.assertFalse(status["promotion"]["promote"])
        self.assertEqual(status["promotion"]["candidate"]["logloss"], 1.1)
        self.assertEqual(status["promotion"]["reason"], "지표 악화 또는 개선 부족 · 현재 모델 유지")
        self.assertNotIn("credential", json.dumps(status))

    def test_bid_valuation_not_mid_and_inference_error_never_exposed(self):
        self.write("engine-config.json", {"paper": {"fee": .01, "slippage": .02}})
        self.write("work/paper/state.json", {"initial_cash": 1000, "equity": 1082.12,
                   "positions": {"KRW-BTC": {"qty": 2, "cost": 550}}, "prices": {"KRW-BTC": 320},
                   "assets": [{"market": "KRW-BTC", "qty": 2, "cost": 550, "price": 300, "value": 600}],
                   "entry_guard": "추론 오류로 신규 진입 차단", "inference_errors": {"KRW-BTC": {"message": "credential"}}})
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertAlmostEqual(status["account"]["unrealized_pnl"], 32.12)
        self.assertAlmostEqual(status["account"]["realized_pnl"], 50)
        self.assertEqual(status["account"]["entry_guard"], "추론 오류로 신규 진입 차단")
        self.assertNotIn("credential", json.dumps(status))

    def test_health_result_is_separate_from_worker_and_private_messages_are_omitted(self):
        self.write("work/service/health.json", {"state": "실행 중", "argv": ["credential"]})
        self.write("work/service/health_result.json", {"state": "검사실패", "checked_utc": "2026-10-21T00:00:00Z",
                   "checked_datasets": 36, "total_rows": 1000, "errors_count": 1,
                   "errors": ["credential"], "pending": ["credential", "초기 대기"]})
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertEqual(status["health"]["state"], "검사실패")
        self.assertEqual(status["health"]["errors_count"], 1)
        self.assertEqual(status["health"]["pending_count"], 2)
        self.assertEqual(status["services"]["health"]["state"], "실행 중")
        self.assertNotIn("credential", json.dumps(status))
        self.assertNotIn("errors", status["health"])
        self.write("work/service/health_result.json", {"state": "검사 통과", "errors_count": 0, "pending": []})
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertEqual(status["health"]["state"], "검사통과")
        self.assertEqual(status["health"]["pending_count"], 0)

    def test_training_completion_is_not_report_creation_time(self):
        report = {"run": "시간-검사", "updated_at": "2026-10-21T06:15:30+09:00",
                  "training": {"status": "완료", "started_at": "2026-10-21T06:15:30+09:00",
                               "completed_at": "2026-10-21T06:15:53+09:00", "duration_seconds": 23,
                               "rows": 100, "features": 42},
                  "markets": {"KRW-BTC": {"gpu": [{"boosted_rounds": 30, "duration_seconds": 2.5}]}}}
        self.write("work/reports/training.json", report)
        self.write("work/reports/experiments/20261021T061530.json", report)
        self.write("work/service/training_progress.json", {"run": "시간-검사", "status": "running",
                   "started_at": "2026-10-21T06:15:30+09:00", "elapsed_seconds": 12,
                   "rows": 90, "features": 42, "current_round": 10, "target_rounds": 30, "validation_logloss": .8})
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertEqual(status["training"]["started_utc"], "2026-10-21T06:15:30+09:00")
        self.assertEqual(status["training"]["finished_utc"], "2026-10-21T06:15:53+09:00")
        self.assertEqual(status["training"]["duration_seconds"], 23)
        self.assertEqual(status["training"]["elapsed_seconds"], 12)
        self.assertEqual(status["training"]["current_round"], 10)
        self.assertEqual(status["training"]["validation_logloss"], .8)
        self.assertEqual(status["rounds"][0]["finished_utc"], "2026-10-21T06:15:53+09:00")
        self.assertEqual(status["markets"]["KRW-BTC"]["gpu"][0]["boosted_rounds"], 30)
        del report["training"]["completed_at"]
        self.write("work/reports/training.json", report)
        self.write("work/reports/experiments/20261021T061530.json", report)
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertIsNone(status["training"]["finished_utc"])
        self.assertIsNone(status["rounds"][0]["finished_utc"])

    def test_research_and_probe_and_decisions_are_separate_and_bounded(self):
        self.write("work/reports/training.json", {
            "research": {"experiment_index": 3, "unknown": "credential"},
            "markets": {"KRW-BTC": {"diagnostics": {"prediction_distribution": [0, 10, 0],
                "probability_quantiles": {"up": {"p50": .2}}, "secret": "credential"},
                "research": {"candidates": [{"candidate": 1, "trades": 0, "fees_paid": 0,
                    "rejection_reasons": ["no_trades"], "rule": {"confidence": .8}, "secret": "credential"}]}}}})
        self.write("work/reports/execution_probe.json", {"checked_at": "2026-10-21T00:00:00Z",
            "market": "KRW-BTC", "source": "upbit", "scope": "isolated-paper", "passed": True,
            "buy": {"qty": 1, "vwap": 100, "fee_paid": .1, "secret": "credential"},
            "sell": {"qty": 1, "vwap": 99}, "round_trip_net_pnl": -1.1, "fees": .1,
            "quote_timestamps": {"buy": "2026-10-21T00:00:00Z"}, "error": {"message": "credential"},
            "trades": [{"price": 999}], "equity": 999})
        self.write("work/paper/state.json", {"equity": 25000000, "trades": [],
            "decision_diagnostics": {"KRW-BTC": {"probabilities": [.1, .8, .1], "signal": 0,
                "expected_return": .01, "guard": "no_signal", "quote_age": 2, "secret": "credential"}}})
        self.write("work/service/trainer.json", {"previous_finished_utc": "2026-10-21T00:00:00Z",
            "success_gap_seconds": .01, "next_iteration": 2, "successful_cycles": 1, "duration_seconds": 3})
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertEqual(status["markets"]["KRW-BTC"]["diagnostics"]["prediction_distribution"], [0, 10, 0])
        self.assertEqual(status["markets"]["KRW-BTC"]["research"]["candidates"][0]["trades"], 0)
        self.assertEqual(status["research"]["experiment_index"], 3)
        self.assertEqual(status["execution_probe"]["buy"]["vwap"], 100)
        self.assertEqual(status["execution_probe"]["round_trip_net_pnl"], -1.1)
        self.assertEqual(status["decision_diagnostics"]["KRW-BTC"]["quote_age"], 2)
        self.assertEqual(status["services"]["trainer"]["success_gap_seconds"], .01)
        self.assertEqual(status["account"]["equity"], 25000000)
        self.assertEqual(status["trades"], [])
        self.assertNotIn("credential", json.dumps(status))
        self.assertNotIn("unknown", status["research"])
        self.assertNotIn("error", status["execution_probe"])

    def test_diagnostic_projection_has_depth_count_and_string_limits(self):
        source = {"candidates": [{"reason": "x" * 2000, "candidate": i,
                   "rule": {"rule": {"rule": {"rule": {"rule": {"confidence": .5}}}}}}
                  for i in range(500)]}
        self.write("work/reports/training.json", {"research": source})
        status = dashboard.SnapshotAdapter(self.root).status()
        self.assertLessEqual(len(status["research"]["candidates"]), 100)
        self.assertLessEqual(len(status["research"]["candidates"][0]["reason"]), 240)
        self.assertLess(len(json.dumps(status["research"])), 40000)

    def test_observed_engine_diagnostics_and_rule_trials(self):
        diagnostics = {"argmax_counts": [0, 100, 0], "signal_counts": [0, 100, 0],
            "probability_quantiles": {"2": {"p50": .02}},
            "filter_reasons": {"no_trade_policy": 100, "buy_confidence_below": 100,
                               "buy_edge_below": 98, "neutral_signal": 100}}
        self.write("work/reports/training.json", {"research": {
            "sequence": 5, "horizon_minutes": 15, "class_weight_mode": "bounded_inverse_sqrt"},
            "markets": {"KRW-BTC": {"diagnostics": diagnostics, "research": {"trials": [
                {"rule": {"confidence": 1.01, "edge": 1.01, "horizon_minutes": 15},
                 "trading": {"trades": 0, "fees_paid": 0, "executions": 0},
                 "eligible": False, "objective": 0, "min_closed_trades": 3,
                 "rejection_reasons": ["insufficient_closed_trades"], "diagnostics": diagnostics}]}}}})
        status = dashboard.SnapshotAdapter(self.root).status()
        actual = status["markets"]["KRW-BTC"]
        self.assertEqual(actual["diagnostics"]["argmax_counts"], [0, 100, 0])
        self.assertEqual(actual["diagnostics"]["filter_reasons"]["neutral_signal"], 100)
        self.assertEqual(actual["research"]["trials"][0]["diagnostics"]["probability_quantiles"]["2"]["p50"], .02)
        self.assertEqual(status["research"]["class_weight_mode"], "bounded_inverse_sqrt")

    def test_final_engine_schema_rule_trials_promotion_and_inference(self):
        self.write("work/reports/training.json", {"research": {"schema_version": 2,
            "deployment": {"eligible": False, "rejection_reasons": ["insufficient_closed_trades"]}},
            "promotion": {"promote": False, "reason": "후속 평가 수익/완료 거래 근거 부족; 운용 보류",
                          "eligibility": {"eligible": False, "min_closed_trades": 3}},
            "markets": {"KRW-BTC": {"rule": {"confidence": 1.01}, "temperature": 1.25, "weights": [.4, .6],
                "splits": {"rule": {"rows": 10}, "validation": {"rows": 20}},
                "diagnostics": {"feature_rows_dropped": 42, "policy_eligible": False,
                    "rule_trials": [{"trading": {"fees_paid": 0, "trades": 0}, "eligible": False}],
                    "weighting": {"mode": "bounded_inverse_sqrt", "class_counts": [1, 100, 1]}}}}})
        self.write("work/paper/state.json", {"entry_guard": "unvalidated_or_unprofitable_model",
            "inference_diagnostics": {"KRW-BTC": {"checked_at": "2026-10-21T00:00:00Z",
                "reason": "unvalidated_or_unprofitable_model", "signal": 1,
                "error": {"message": "credential"}, "model_run": "actual-model"}}})
        status = dashboard.SnapshotAdapter(self.root).status()
        btc = status["markets"]["KRW-BTC"]
        self.assertEqual(btc["diagnostics"]["rule_trials"][0]["trading"]["trades"], 0)
        self.assertFalse(btc["diagnostics"]["policy_eligible"])
        self.assertEqual(btc["rule"]["confidence"], 1.01)
        self.assertEqual(btc["splits"]["rule"]["rows"], 10)
        self.assertEqual(btc["splits"]["validation"]["rows"], 20)
        self.assertFalse(status["promotion"]["eligibility"]["eligible"])
        self.assertIn("운용 보류", status["promotion"]["reason"])
        self.assertIn("수익 기준 미달", status["account"]["entry_guard"])
        self.assertEqual(status["inference_diagnostics"]["KRW-BTC"]["model_run"], "actual-model")
        self.assertNotIn("credential", json.dumps(status))

    def test_http_get_only_no_arbitrary_paths(self):
        web = self.root / "web"
        web.mkdir()
        (web / "index.html").write_text("대시보드", encoding="utf-8")
        (self.root / "secret.txt").write_text("credential", encoding="utf-8")
        self.write("work/reports/training.json", {"markets": {"KRW-XRP": {
            "validation": {"accuracy": .6, "support": 5, "api_key": "credential"},
            "splits": {"train": {"rows": 30, "start": "2025-01-01", "end": "2026-01-01", "secret": "credential"}}}},
            "config": {"secret": "credential"}})
        server = dashboard.create_server("127.0.0.1", 0, self.root)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            def request(method, path):
                connection = http.client.HTTPConnection(*server.server_address, timeout=2)
                connection.request(method, path)
                response = connection.getresponse()
                result = response.status, response.read(), dict(response.getheaders())
                connection.close()
                return result
            code, body, headers = request("GET", "/api/status")
            self.assertEqual(code, 200)
            self.assertIn("schema_version", json.loads(body))
            self.assertEqual(json.loads(body)["markets"]["KRW-XRP"]["validation"]["samples"], 5)
            self.assertNotIn(b"credential", body)
            self.assertEqual(headers["Cache-Control"], "no-store")
            self.assertIn("default-src 'self'", headers["Content-Security-Policy"])
            self.assertEqual(request("GET", "/")[1].decode("utf-8"), "대시보드")
            for path in ("/../secret.txt", "/%2e%2e/secret.txt", "/secret.txt", "/work/paper/state.json", "/web/", "/.env"):
                self.assertEqual(request("GET", path)[0], 404, path)
            self.assertEqual(request("GET", "/api/status?path=secret.txt")[0], 400)
            for method in ("POST", "PUT", "PATCH", "DELETE", "OPTIONS"):
                self.assertEqual(request(method, "/api/status")[0], 405)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
