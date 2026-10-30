"""3차 원금·기존 장부 경계와 실제 대시보드 필드를 검사합니다."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import dashboard
import engine
import healthcheck
import supervisor


class CapitalContractTests(unittest.TestCase):
    def test_default_and_shipped_capital_and_schedule_agree(self):
        root = Path(engine.__file__).parent
        config = engine.load_config(root / "engine-config.json")
        service = json.loads((root / "service.json").read_text(encoding="utf-8"))
        self.assertEqual(engine.DEFAULT_CONFIG["paper"]["initial_cash"], 10_000_000)
        self.assertEqual(config["paper"]["initial_cash"], 10_000_000)
        self.assertEqual(service["initial_capital"], 10_000_000)
        self.assertEqual(engine.utc(service["paper_start"]), engine.utc(config["paper"]["start_time"]))
        supervisor.validate_settings(service, config)

    def test_restored_old_capital_and_false_paper_flag_are_rejected_without_mutation(self):
        settings = copy.deepcopy(engine.DEFAULT_CONFIG["paper"])
        for field, value in (("initial_cash", 25_000_000), ("paper_only", False), ("paper_only", 1)):
            state = engine.Portfolio(settings).state
            state[field] = value
            before = copy.deepcopy(state)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                engine.Portfolio(settings, state)
            self.assertEqual(state, before)

    def test_ten_million_account_passes_and_old_capital_fails_healthcheck(self):
        state = engine.Portfolio().state
        self.assertEqual(healthcheck.check_account(state, "2026-10-26T00:00:00Z", engine.utc()), [])
        state["initial_cash"] = 25_000_000
        self.assertTrue(healthcheck.check_account(state, "2026-10-26T00:00:00Z", engine.utc()))

    def test_service_rejects_capital_schedule_and_notification_mismatch(self):
        root = Path(engine.__file__).parent
        config = engine.load_config(root / "engine-config.json")
        service = json.loads((root / "service.json").read_text(encoding="utf-8"))
        for key, value in (("initial_capital", 25_000_000), ("paper_start", "2026-10-21T09:00:00Z"),
                           ("notifications", True), ("market_mode", "live")):
            changed = {**service, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                supervisor.validate_settings(changed, config)

    def test_dashboard_reads_policy_and_recorded_optimizer_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "engine-config.json").write_text(json.dumps({"paper": {
                "initial_cash": 10_000_000, "start_time": "2026-10-26T06:00:00Z"}}), encoding="utf-8")
            (root / "work/service").mkdir(parents=True)
            (root / "work/service/training_progress.json").write_text(json.dumps({
                "status": "running", "eta": .05, "confidence": .99}), encoding="utf-8")
            status = dashboard.SnapshotAdapter(root).status()
            self.assertEqual(status["policy"]["initial_capital"], 10_000_000)
            self.assertEqual(status["policy"]["paper_start"], "2026-10-26T06:00:00Z")
            self.assertEqual(status["training"]["learning_rate"], .05)
            self.assertNotIn("confidence", status["training"])
            (root / "work/service/training_progress.json").write_text('{}', encoding="utf-8")
            self.assertIsNone(dashboard.SnapshotAdapter(root).status()["training"]["learning_rate"])
