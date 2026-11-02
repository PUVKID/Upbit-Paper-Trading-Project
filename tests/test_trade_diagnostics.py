"""실제 장부를 쓰지 않고 비례 원가와 읽기 전용 경계를 검사합니다."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "10차/trade_diagnostics.py"


def load_module():
    spec = importlib.util.spec_from_file_location("trade_diagnostics", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def event(index, side, qty, price, fee, pnl=None, reason="signal"):
    result = {"event_id": str(index), "market": "KRW-DOGE", "side": side,
              "timestamp": f"2026-11-01T00:0{index}:00+00:00", "reason": reason,
              "qty": qty, "vwap": price, "fee_paid": fee,
              "cash": qty * price + (fee if side == "buy" else -fee)}
    if pnl is not None:
        result["realized_pnl"] = pnl
    return result


def fixture():
    return {"initial_cash": 1000, "cash": 949.48, "paper_only": True,
            "fees_paid": 2.52, "completed_trades": 1,
            "positions": {"KRW-DOGE": {"qty": 5, "cost": 50.5}},
            "trades": [event(0, "buy", 10, 10, 1),
                       event(1, "sell", 4, 12, .48, 7.12),
                       event(2, "sell", 6, 9, .54, -7.14, "stop_loss"),
                       event(3, "buy", 5, 10, .5)]}


class TradeDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_partial_sale_reentry_open_cost_and_fee_allocation(self):
        state = fixture()
        original = copy.deepcopy(state)
        result = self.module.diagnose(state)
        self.assertEqual(result["status"], "complete")
        totals = result["totals"]
        self.assertAlmostEqual(totals["net_pnl"], -.02)
        self.assertAlmostEqual(totals["fees"], 2.02)
        self.assertAlmostEqual(totals["fill_price_pnl"], 2)
        self.assertAlmostEqual(result["open_positions"]["KRW-DOGE"]["cost"], 50.5)
        self.assertAlmostEqual(result["open_positions"]["KRW-DOGE"]["entry_fees"], .5)
        self.assertAlmostEqual(result["by_reason"]["signal"]["fees"], .88)
        self.assertEqual(result["holding_seconds"]["mean"], 90)
        self.assertEqual(result["stop_reentry_seconds"]["mean"], 60)
        self.assertEqual(result["concentration"]["largest_market_exit_share"], 1)
        self.assertEqual(result["concentration"]["signal_exit_share"], .5)
        self.assertEqual(state, original)

    def test_average_cost_not_fifo_with_multiple_entries(self):
        state = {"initial_cash": 1000, "cash": 886.6, "paper_only": True,
                 "fees_paid": 3.4, "completed_trades": 0,
                 "positions": {"KRW-DOGE": {"qty": 10, "cost": 151.5}},
                 "trades": [event(0, "buy", 10, 10, 1), event(1, "buy", 10, 20, 2),
                            event(2, "sell", 10, 19, .4, 38.1)]}
        result = self.module.diagnose(state)
        self.assertEqual(result["status"], "complete")
        self.assertAlmostEqual(result["totals"]["net_pnl"], 38.1)
        self.assertAlmostEqual(result["totals"]["fees"], 1.9)

    def test_recorded_pnl_mismatch_is_incomplete_not_corrected(self):
        state = fixture()
        state["trades"][1]["realized_pnl"] += 1
        result = self.module.diagnose(state)
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("realized_pnl_mismatch", result["issues"])
        self.assertFalse(result["reconciled"])

    def test_missing_buy_never_invented(self):
        state = fixture()
        state["trades"] = state["trades"][1:]
        result = self.module.diagnose(state)
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("missing_entry_or_oversell", result["issues"])
        self.assertIsNone(result["totals"])

    def test_state_reconciliation_mismatches(self):
        for field in ("cash", "fees_paid", "completed_trades", "initial_cash", "positions"):
            with self.subTest(field=field):
                state = fixture()
                if field == "positions":
                    state[field]["KRW-DOGE"]["cost"] += 1
                else:
                    state[field] += 1
                self.assertEqual(self.module.diagnose(state)["status"], "incomplete")

    def test_missing_reconciliation_evidence_is_incomplete(self):
        for key in ("fees_paid", "completed_trades"):
            state = fixture()
            del state[key]
            self.assertEqual(self.module.diagnose(state)["status"], "incomplete")

    def test_invalid_fills_chronology_ids_and_nonfinite_rejected(self):
        mutations = [(0, "qty", -1), (0, "vwap", 0), (0, "fee_paid", -1),
                     (0, "cash", 99), (0, "qty", True), (0, "qty", float("nan")),
                     (1, "event_id", "0"), (1, "timestamp", "2020-01-01T00:00:00Z"),
                     (1, "timestamp", "2026-11-01T00:01:00"), (0, "side", "short"),
                     (0, "market", "private-secret"), (0, "vwap", float("inf"))]
        for index, key, value in mutations:
            with self.subTest(key=key, value=value):
                state = fixture()
                state["trades"][index][key] = value
                self.assertEqual(self.module.diagnose(state)["status"], "rejected")

    def test_cash_overspending_rejected(self):
        state = fixture()
        state["initial_cash"] = 10
        self.assertEqual(self.module.diagnose(state)["status"], "rejected")

    def test_metadata_and_cost_unavailable_not_guessed_or_copied(self):
        state = fixture()
        state["private"] = "SECRET"
        state["model_run"] = "CURRENT-SECRET"
        state["trades"][0]["secret"] = "SECRET"
        result = self.module.diagnose(state)
        self.assertEqual(result["metadata_missing"], {"model_run": 4, "probabilities": 4, "quote": 4})
        self.assertEqual(result["cost_separation"]["slippage"], "unavailable")
        self.assertEqual(result["cost_separation"]["spread_vs_market_move"], "unavailable")
        self.assertNotIn("SECRET", json.dumps(result))

    def test_explicit_metadata_only_counts_no_raw_values(self):
        state = fixture()
        state["trades"][0].update(model_run="private-run", probabilities=[.1, .2, .7],
                                  quote={"ask_price": 10, "bid_price": 9, "timestamp": "2026-11-01T00:00:00Z"})
        result = self.module.diagnose(state)
        self.assertEqual(result["metadata_missing"], {"model_run": 3, "probabilities": 3, "quote": 4})
        self.assertNotIn("private-run", json.dumps(result))

    def test_history_at_engine_retention_limit_is_incomplete(self):
        state = {"initial_cash": 10000, "cash": 10000, "paper_only": True,
                 "fees_paid": 0, "completed_trades": 1000, "positions": {}, "trades": []}
        for i in range(1000):
            buy = event(0, "buy", 1, 1, 0)
            sell = event(1, "sell", 1, 1, 0, 0)
            for j, e in enumerate((buy, sell)):
                e.update(event_id=str(i * 2 + j), timestamp="2026-11-01T00:00:00Z")
                state["trades"].append(e)
        result = self.module.diagnose(state)
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("possibly_truncated_history", result["issues"])

    def test_empty_account_reconciles(self):
        state = {"initial_cash": 1000, "cash": 1000, "positions": {}, "trades": [],
                 "paper_only": True, "fees_paid": 0, "completed_trades": 0}
        self.assertEqual(self.module.diagnose(state)["status"], "complete")

    def test_cli_reads_fixture_writes_only_allowed_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "state.json"
            source.write_text(json.dumps(fixture()), encoding="utf-8")
            before = source.read_bytes()
            with patch.object(self.module, "ROOT", root):
                self.assertEqual(self.module.main(["--state", "state.json", "--output", "10차/work/result.json"]), 0)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(json.loads((root / "10차/work/result.json").read_text(encoding="utf-8"))["status"], "complete")

    def test_output_escape_input_overwrite_and_links_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for value in ("../escape.json", "work/out.json", "10차/work/../out.json"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    self.module.safe_path(value, root / "10차/work", root)
            target = root / "10차/work/out.json"
            target.parent.mkdir(parents=True)
            original = Path.lstat
            def lstat(path, *args, **kwargs):
                if path == target.parent:
                    return SimpleNamespace(st_mode=stat.S_IFLNK | 0o777)
                return original(path, *args, **kwargs)
            with patch.object(Path, "lstat", lstat), self.assertRaises(ValueError):
                self.module.safe_path(target, root / "10차/work", root)

    def test_json_duplicate_keys_and_nonfinite_refused(self):
        for raw in ('{"cash":1,"cash":2}', '{"cash":NaN}'):
            with self.assertRaises(ValueError):
                self.module.parse_state(raw)

    def test_new_nested_contexts_and_entry_model_attribution(self):
        state = fixture()
        for e in state["trades"]:
            e.update(model_run="decision-not-entry", decision_context={"probabilities": [.1, .2, .7]},
                     execution_context={"scope": "public_orderbook_at_execution", "best_ask": 10,
                                        "best_bid": 9, "fee_rate": .001, "slippage_rate": .001,
                                        "quote_utc": e["timestamp"], "observed_utc": e["timestamp"],
                                        "quote_age_seconds": 0, "risk_price_source": "observed_bid"})
        state["trades"][1].update(entry_model_run="entry-A", entry_context={"model_run": "entry-A"})
        state["trades"][0]["entry_filter"] = {"roundtrip_cost": .03, "private": "SECRET"}
        result = self.module.diagnose(state)
        key = hashlib.sha256(b"entry-A").hexdigest()
        self.assertEqual(result["by_entry_model"][key]["exits"], 1)
        self.assertEqual(result["unattributed_exits"], 1)
        self.assertEqual(result["metadata_missing"], {"model_run": 0, "probabilities": 0, "quote": 0})
        self.assertEqual(result["context_missing"]["entry_context"], 3)
        self.assertEqual(result["entry_filter_metrics"]["roundtrip_cost"]["mean"], .03)
        self.assertNotIn("SECRET", json.dumps(result))
        self.assertNotIn("entry-A", json.dumps(result))

    def test_legacy_not_attributed_to_current_or_decision_model(self):
        state = fixture()
        state["model_run"] = "current-model"
        for e in state["trades"]:
            e["model_run"] = "decision-model"
        result = self.module.diagnose(state)
        self.assertEqual(result["by_entry_model"], {})
        self.assertEqual(result["unattributed_exits"], 2)

    def test_stable_read_retries_changed_snapshot_and_reports_asof(self):
        first = json.dumps(fixture()).encode()
        second = first + b"\n"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "state.json"
            source.write_bytes(second)
            with patch.object(self.module, "_read_snapshot", side_effect=[first, second, second, second]) as reads:
                raw, asof = self.module.stable_read(source, root)
            self.assertEqual(raw, second)
            self.assertEqual(reads.call_count, 4)
            self.assertIsInstance(self.module.timestamp(asof), float)

    def test_unstable_read_has_finite_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(self.module, "_read_snapshot", side_effect=[b"a", b"b"] * 3) as reads:
                with self.assertRaises(ValueError):
                    self.module.stable_read(root / "state.json", root)
            self.assertEqual(reads.call_count, 6)

    def test_cli_records_read_asof_and_input_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "state.json"
            raw = json.dumps(fixture()).encode()
            source.write_bytes(raw)
            with patch.object(self.module, "ROOT", root):
                self.module.main(["--state", "state.json", "--output", "10차/work/result.json"])
            result = json.loads((root / "10차/work/result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["input_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(result["read_consistency"], "two_identical_reads")
            self.assertIsInstance(self.module.timestamp(result["asof_utc"]), float)

    def test_quote_completeness_requires_times_and_synthetic_is_not_live(self):
        state = fixture()
        state["trades"][0]["execution_context"] = {
            "scope": "public_orderbook_at_execution", "best_ask": 10, "best_bid": 9,
            "fee_rate": .001, "slippage_rate": .001, "risk_price_source": "observed_bid",
            "quote_utc": "2026-11-01T00:00:00Z", "observed_utc": "2026-11-01T00:00:01Z",
            "quote_age_seconds": 1}
        state["trades"][1]["execution_context"] = {
            "scope": "public_orderbook_at_execution", "best_ask": 10, "best_bid": 9}
        state["trades"][2]["execution_context"] = {
            "scope": "synthetic_candle", "best_ask": 10, "best_bid": 9}
        result = self.module.diagnose(state)
        self.assertEqual(result["quote_metadata"], {"complete": 1, "partial": 1, "unknown": 1, "synthetic": 1})
        self.assertEqual(result["metadata_missing"]["quote"], 3)
        state["trades"][0]["execution_context"]["quote_age_seconds"] = 0
        result = self.module.diagnose(state)
        self.assertEqual(result["quote_metadata"]["complete"], 0)
        self.assertEqual(result["quote_metadata"]["partial"], 2)

    def test_invalid_legacy_entry_context_does_not_discard_sell_or_attribute_model(self):
        state = fixture()
        state["trades"][1].update(entry_model_run="not-trusted", entry_context_invalid=True, entry_context=None)
        state["trades"][2].update(entry_model_run="not-trusted", entry_context=["legacy"])
        state["trades"][1]["decision_context"] = {"record_status": "partial_invalid", "probabilities": [float("nan"), 0, 1]}
        result = self.module.diagnose(state)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["totals"]["exits"], 2)
        self.assertEqual(result["by_entry_model"], {})
        self.assertEqual(result["entry_context_invalid_events"], 2)
        self.assertEqual(result["context_record_status"]["decision_context"]["partial_invalid"], 1)

    def test_legacy_quote_without_observed_time_is_partial(self):
        state = fixture()
        state["trades"][0]["quote"] = {"ask_price": 10, "bid_price": 9, "timestamp": "2026-11-01T00:00:00Z"}
        result = self.module.diagnose(state)
        self.assertEqual(result["quote_metadata"]["partial"], 1)
        self.assertEqual(result["quote_metadata"]["complete"], 0)


if __name__ == "__main__":
    unittest.main()
