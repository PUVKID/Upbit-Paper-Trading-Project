"""명시 요청 연구의 내구성·실패 경계와 엔진 공통 진입점을 검사합니다."""
import importlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import engine


class ResearchControlTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((engine.ROOT / "4차/research_control.py").is_file(), "연구 제어 모듈이 없습니다")
        self.module = importlib.import_module("4차.research_control")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = {"active": True, "state_path": "4차/work/research_control.json"}
        self.control = self.module.ResearchControl(self.root, self.settings)
        self.fit_calls = 0
        self.last_report = None
        self.promote = False

    def active_config(self):
        config = engine.load_config()
        config["research"] = self.settings
        config["external"].update(enabled="auto", wait_for_dataset=True)
        config["training"]["candidate_depths"] = [3]
        for key, folder in (("data_dir", "data"), ("models_dir", "models"), ("reports_dir", "reports")):
            config[key] = str(self.root / "work" / folder)
        return config

    def completed_cycle(self, config, **kwargs):
        return ResearchRecoveryTests.cycle(self, config, **kwargs)

    def report(self, target=50_000_000_000, run="controlled-run"):
        return {"run": run, "config": {"training": {"candidate_depths": [3]}},
                "training": {"status": "완료", "completed_candidates": 5, "total_candidates": 5},
                "markets": {m: {"gpu": [{"actual_device": "cuda:0"}]} for m in engine.MARKETS},
                "rounds": [{"market": m, "candidate": 1, "actual_device": "cuda:0"} for m in engine.MARKETS],
                "external_resolution": {"training_ready": True, "target_archive_bytes": target,
                                        "ingested_archive_bytes": target + 3,
                                        "verified_archive_bytes": target + 3},
                "promotion": {"promote": False}}

    def finish_initial(self):
        with patch.object(engine, "ROOT", self.root), patch.object(engine, "wait_for_training_data"), \
             patch.object(engine, "_train_cycle", side_effect=self.completed_cycle):
            engine.train_cycle(self.active_config(), live=True)

    def test_bootstrap_target_and_restart_are_exact(self):
        self.assertEqual(self.control.collection_target(), 50_000_000_000)
        again = self.module.ResearchControl(self.root, self.settings)
        self.assertEqual(again.pending_batch(), self.control.pending_batch())
        self.assertEqual(len(again.snapshot()["batches"]), 1)

    def test_initial_request_rejected_before_reading_baseline(self):
        reader = Mock(return_value=51_000_000_000)
        with self.assertRaisesRegex(ValueError, "초기"):
            self.control.request_increment(reader)
        reader.assert_not_called()

    def test_completed_unpromoted_batch_stays_idle_across_restart(self):
        self.finish_initial()
        # 예전 completed 이력은 새 시도 파일이 없어도 다시 적합하지 않습니다.
        self.control.attempt_path(self.control.snapshot()["batches"][0]).unlink()
        again = self.module.ResearchControl(self.root, self.settings)
        self.assertIsNone(again.collection_target())
        self.assertIsNone(again.pending_batch())
        self.assertFalse(again.snapshot()["batches"][0]["promoted"])

    def test_fresh_actual_baseline_excludes_last_archive_overshoot_and_is_frozen(self):
        self.finish_initial()
        reader = Mock(return_value=55_123_456_789)
        batch = self.control.request_increment(reader)
        self.assertEqual(batch["baseline_bytes"], 55_123_456_789)
        self.assertEqual(batch["target_bytes"], 57_123_456_789)
        reader.assert_called_once_with()
        self.assertEqual(self.control.collection_target(), 57_123_456_789)
        with self.assertRaisesRegex(ValueError, "활성"):
            self.control.request_increment(reader)
        reader.assert_called_once_with()

    def test_invalid_or_failed_baseline_does_not_append_request(self):
        self.finish_initial()
        before = self.control.snapshot()
        for invalid in (True, -1, 49_999_999_999, 50_000_000_000.0):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.control.request_increment(lambda: invalid)
        with self.assertRaises(OSError):
            self.control.request_increment(Mock(side_effect=OSError("통제된 읽기 실패")))
        self.assertEqual(before, self.control.snapshot())

    def test_failure_and_interrupted_claim_retry_same_batch_not_completed(self):
        batch = self.control.pending_batch()
        first = self.control.claim_training(batch["id"])
        self.control.fail_training(batch["id"], first["claim"], "통제된 학습 실패")
        self.assertEqual(self.control.pending_batch()["id"], batch["id"])
        self.assertEqual(self.control.pending_batch()["last_error"], "통제된 학습 실패")
        second = self.control.claim_training(batch["id"])
        again = self.module.ResearchControl(self.root, self.settings)
        with again.training_lock():
            with self.assertRaises(ValueError):
                again.claim_training(batch["id"])
            again.fail_training(batch["id"], second["claim"], "통제된 미완료 시도 종료")
            third = again.claim_training(batch["id"])
        self.assertEqual(third["attempts"], 3)
        self.assertNotEqual(third["claim"], second["claim"])
        with self.assertRaises(ValueError):
            self.control.complete_training(batch["id"], second["claim"], self.report())
        self.assertIsNotNone(self.control.collection_target())

    def test_bad_result_cannot_complete_batch(self):
        batch = self.control.pending_batch()
        claim = self.control.claim_training(batch["id"])
        run = "20261102T000000-01234567"
        record = self.control.start_attempt(claim, run, "work/reports", "work/models", None, None)
        invalids = []
        for key, value in (("training_ready", False), ("ingested_archive_bytes", 49_999_999_999)):
            report = self.report()
            report["external_resolution"][key] = value
            invalids.append(report)
        report = self.report()
        report["markets"].pop("KRW-DOGE")
        invalids.append(report)
        report = self.report()
        report["training"]["completed_candidates"] = 4
        invalids.append(report)
        report = self.report()
        report["rounds"][0]["actual_device"] = "cpu"
        invalids.append(report)
        for report in invalids:
            report["run"] = run
            report["research_batch"] = self.module.attempt_binding(record)
            with self.subTest(report=report), self.assertRaises(ValueError):
                self.control.store_result(batch["id"], claim["claim"], report, "0" * 64, {"model.ubj": "0" * 64})
        self.assertEqual(self.control.pending_batch()["status"], "training")

    def test_new_claim_cannot_complete_without_durable_bound_result(self):
        batch = self.control.pending_batch()
        claim = self.control.claim_training(batch["id"])
        with self.assertRaises(ValueError):
            self.control.complete_training(batch["id"], claim["claim"], self.report())

    def test_corrupt_schema_and_duplicate_json_keys_fail_closed_without_rewrite(self):
        self.control.collection_target()
        path = self.root / self.settings["state_path"]
        for raw in ('{"schema_version":99,"batches":[]}', '{"schema_version":1,"schema_version":1,"batches":[]}', '{broken'):
            with self.subTest(raw=raw):
                path.write_text(raw, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.control.collection_target()
                self.assertEqual(path.read_text(encoding="utf-8"), raw)

    def test_config_default_disabled_and_paths_confined(self):
        self.assertFalse(self.module.validate_config({}, self.root)["active"])
        for invalid in ({"active": 1}, {"unknown": True}, {"state_path": "../escape.json"},
                        {"state_path": "work/paper/account.json"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.module.validate_config(invalid, self.root)
        disabled = self.module.ResearchControl(self.root / "disabled", {})
        self.assertIsNone(disabled.collection_target())
        self.assertFalse((self.root / "disabled/4차/work").exists())

    def test_training_lock_excludes_another_owner(self):
        another = self.module.ResearchControl(self.root, self.settings)
        with self.control.training_lock():
            with self.assertRaises((OSError, RuntimeError)):
                with another.training_lock():
                    self.fail("동시 학습 잠금이 허용됐습니다")

    def test_atomic_write_failure_preserves_previous_state(self):
        self.finish_initial()
        before = self.control.snapshot()
        with patch.object(self.module.os, "replace", side_effect=OSError("통제된 저장 실패")):
            with self.assertRaises(OSError):
                self.control.request_increment(lambda: 52_000_000_000)
        self.assertEqual(self.control.snapshot(), before)

    def test_two_concurrent_requests_accept_exactly_one_fresh_baseline(self):
        self.finish_initial()
        barrier = threading.Barrier(2)
        results = []
        reader = Mock(return_value=55_000_000_000)

        def request():
            barrier.wait(timeout=3)
            try:
                self.control.request_increment(reader)
                results.append("accepted")
            except ValueError:
                results.append("rejected")

        threads = [threading.Thread(target=request) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertCountEqual(results, ["accepted", "rejected"])
        reader.assert_called_once_with()

    def test_request_cli_baseline_uses_current_database_not_previous_status_bytes(self):
        self.finish_initial()
        config = engine.load_config()
        config["research"] = self.settings
        config["external"].update(enabled="auto", wait_for_dataset=True)
        external = importlib.import_module("3차.external_features")
        db = Mock()
        db.execute.return_value = [("ingested_archive_bytes", "57123456789"),
                                   ("verified_archive_bytes", "57123456789")]
        from contextlib import nullcontext
        with patch.object(engine, "ROOT", self.root), \
             patch.object(external, "resolve_for_training", return_value=(config["external"], {"ingested_archive_bytes": 50_000_000_000})), \
             patch.object(external, "_database", return_value=nullcontext(db)):
            batch = engine.request_research_increment(config)
        self.assertEqual(batch["baseline_bytes"], 57_123_456_789)
        self.assertEqual(batch["target_bytes"], 59_123_456_789)

    def test_once_trainer_waits_twelve_seconds_before_fit_and_writes_owner_heartbeat(self):
        config = self.active_config()
        external = importlib.import_module("3차.external_features")
        stop = Mock()
        stop.is_set.return_value = False
        stop.wait.return_value = False
        train = Mock(side_effect=self.completed_cycle)
        evidence = {"reason": "통제된 자료 미달", "checked_at": engine.iso(None),
                    "target_archive_bytes": 50_000_000_000, "ingested_archive_bytes": 7_000_000_000}

        def readiness(*args):
            train.assert_not_called()
            if not stop.wait.called:
                raise external.DatasetNotReady(evidence)
            return config["external"], {"training_ready": True}

        with patch.object(engine, "ROOT", self.root), patch.object(engine, "load_config", return_value=config), \
             patch.object(engine, "process_lock"), patch.object(engine, "_train_cycle", train), \
             patch.object(engine, "atomic_json", wraps=engine.atomic_json) as writes, \
             patch.object(engine.threading, "Event", return_value=stop), \
             patch.object(external, "resolve_for_training", side_effect=readiness), patch("builtins.print"):
            self.assertEqual(engine.main(["trainer", "--once"]), 0)
        train.assert_called_once()
        stop.wait.assert_called_once_with(12)
        heartbeat = next(call.args[1] for call in writes.call_args_list
                         if Path(call.args[0]).name == "training_progress.json"
                         and call.args[1].get("status") == "waiting_data")
        self.assertTrue(heartbeat["research_active"])
        self.assertEqual(heartbeat["pid"], engine.os.getpid())
        self.assertEqual(heartbeat["status"], "waiting_data")
        self.assertIsNone(heartbeat["completed_candidates"])
        self.assertIsNone(engine._research_progress.get())

    def test_common_train_entry_waits_then_runs_all_candidates_once_and_goes_idle(self):
        config = self.active_config()
        with patch.object(engine, "ROOT", self.root), patch.object(engine, "_train_cycle", side_effect=self.completed_cycle) as train, \
             patch.object(engine, "wait_for_training_data") as wait:
            report = engine.train_cycle(config, live=True)
        self.assertEqual(report["run"], self.control.snapshot()["batches"][0]["run"])
        train.assert_called_once()
        wait.assert_called_once()
        self.assertEqual(train.call_args.args[0]["external"]["target_bytes"], 50_000_000_000)
        self.assertIsNone(self.control.collection_target())

    def test_train_failure_keeps_batch_and_claim_error(self):
        config = self.active_config()
        with patch.object(engine, "ROOT", self.root), patch.object(engine, "wait_for_training_data"), \
             patch.object(engine, "_train_cycle", side_effect=RuntimeError("통제된 적합 실패")):
            with self.assertRaisesRegex(RuntimeError, "적합 실패"):
                engine.train_cycle(config, live=True)
        self.assertEqual(self.control.pending_batch()["status"], "collecting")
        self.assertIn("적합 실패", self.control.pending_batch()["last_error"])

    def test_idle_has_heartbeat_no_fit_no_fake_candidates_and_no_busy_loop(self):
        self.finish_initial()
        config = engine.load_config()
        config["research"] = self.settings
        stop = Mock()
        stop.is_set.return_value = False
        stop.wait.return_value = True
        records = []
        with patch.object(engine, "ROOT", self.root), patch.object(engine, "_train_cycle") as train, \
             patch.object(engine, "progress", side_effect=lambda **fields: records.append(fields)):
            with self.assertRaises(KeyboardInterrupt):
                engine.wait_for_research_batch(self.control, stop)
        train.assert_not_called()
        stop.wait.assert_called_once_with(12)
        self.assertEqual(records[0]["status"], "waiting_request")
        self.assertEqual(records[0]["stage"], "idle")
        for key in ("run", "progress", "completed_candidates", "total_candidates", "candidate"):
            self.assertIsNone(records[0][key])

    def test_increment_target_is_used_for_training_gate(self):
        self.finish_initial()
        batch = self.control.request_increment(lambda: 56_000_000_000)
        config = self.active_config()
        with patch.object(engine, "ROOT", self.root), patch.object(engine, "wait_for_training_data") as wait, \
             patch.object(engine, "_train_cycle", side_effect=self.completed_cycle) as train:
            engine.train_cycle(config, live=True)
        self.assertEqual(wait.call_args.args[0]["external"]["target_bytes"], 58_000_000_000)
        self.assertEqual(train.call_args.args[0]["external"]["target_bytes"], 58_000_000_000)
        self.assertIsNone(self.control.collection_target())

    def test_request_cli_dispatch_and_experiment_share_public_train_entry(self):
        self.assertEqual(engine.build_parser().parse_args(["research-request"]).command, "research-request")
        config = engine.load_config()
        config["research"] = self.settings
        with patch.object(engine, "ROOT", self.root), patch.object(engine, "load_config", return_value=config), patch.object(engine, "request_research_increment", return_value={"id": "x"}) as request, \
             patch.object(engine, "process_lock"), patch.object(engine, "train_cycle", return_value={"run": "x", "locked_test": {}}) as train, \
             patch("builtins.print"):
            self.assertEqual(engine.main(["research-request"]), 0)
            self.assertEqual(engine.main(["experiment"]), 0)
        request.assert_called_once_with(config)
        train.assert_called_once_with(config, final=False)

    def test_active_config_cannot_disable_external_gate(self):
        path = self.root / "config.json"
        path.write_text(json.dumps({"research": self.settings}), encoding="utf-8")
        with self.assertRaises(ValueError):
            engine.load_config(path)

    def test_paper_dispatch_never_enters_research_gate(self):
        config = engine.load_config()
        config["research"] = self.settings
        with patch.object(engine, "load_config", return_value=config), patch.object(engine, "process_lock"), \
             patch.object(engine, "paper_cycle", return_value={"status": "통제"}) as paper, \
             patch.object(engine, "train_cycle") as train, patch("builtins.print"):
            self.assertEqual(engine.main(["paper", "--once"]), 0)
        paper.assert_called_once_with(config)
        train.assert_not_called()


class ResearchRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.module = importlib.import_module("4차.research_control")
        self.settings = {"active": True, "state_path": "4차/work/research_control.json"}
        self.control = self.module.ResearchControl(self.root, self.settings)
        self.config = engine.load_config()
        self.config["research"] = self.settings
        self.config["external"].update(enabled="auto", wait_for_dataset=True)
        self.config["training"]["candidate_depths"] = [3]
        for key, folder in (("data_dir", "data"), ("models_dir", "models"), ("reports_dir", "reports")):
            self.config[key] = str(self.root / "work" / folder)
        self.fit_calls = 0
        self.last_report = None

    def cycle(self, config, **kwargs):
        self.fit_calls += 1
        batch = self.control.pending_batch()
        attempt = getattr(engine, "_research_attempt", None)
        record = attempt.get() if attempt is not None else None
        run = record["run"] if record else "20261102T000000-01234567"
        binding = {"schema_version": 1, "batch_id": batch["id"], "claim": batch["claim"],
                   "attempt": batch["attempts"], "target_bytes": batch["target_bytes"], "run": run}
        directory = Path(config["models_dir"]) / "versions" / run
        directory.mkdir(parents=True, exist_ok=True)
        packages, hashes = {}, {}
        for market in engine.MARKETS:
            name = market + "-0.ubj"
            (directory / name).write_bytes(("통제된 모델 " + market).encode("utf-8"))
            hashes[name] = engine.hashlib.sha256((directory / name).read_bytes()).hexdigest()
            packages[market] = {"models": [name], "gpu": [{"actual_device": "cuda:0"}]}
        manifest = {"run": run, "schema_version": 2, "execution_policy": engine.EXECUTION_POLICY,
                    "paper_only": True, "config": config, "packages": packages,
                    "source_hash": "controlled-source", "research_batch": binding}
        engine.atomic_json(directory / "manifest.json", manifest)
        report = {"run": run, "config": config, "source_hash": "controlled-source",
                  "research_batch": binding,
                  "training": {"status": "완료", "completed_candidates": 5, "total_candidates": 5,
                               "started_at": engine.iso(None), "duration_seconds": 1, "rows": 100, "features": 1},
                  "markets": packages,
                  "rounds": [{"market": m, "candidate": 1, "actual_device": "cuda:0"} for m in engine.MARKETS],
                  "external_resolution": {"training_ready": True, "target_archive_bytes": batch["target_bytes"],
                                          "ingested_archive_bytes": batch["target_bytes"],
                                          "verified_archive_bytes": batch["target_bytes"]},
                  "promotion": {"promote": getattr(self, "promote", True)}, "locked_test": {}}
        self.last_report = report
        if record is not None:
            record["result_started"] = True
            self.control.store_result(batch["id"], batch["claim"], report, engine.stable_hash(manifest), hashes)
        engine.atomic_json(Path(config["reports_dir"]) / "experiments" / (run + ".json"), report)
        engine.atomic_json(Path(config["reports_dir"]) / "training.json", report)
        if report["promotion"]["promote"]:
            engine.atomic_json(Path(config["models_dir"]) / "current.json",
                               {"run": run, "manifest_hash": engine.stable_hash(manifest)})
        engine.progress(status="complete", stage="complete", run=run)
        return report

    def run_cycle(self):
        def pending(control):
            batch = control.pending_batch()
            if batch is None:
                raise RuntimeError("검사 중 예상하지 않은 요청 대기")
            return batch

        with patch.object(engine, "ROOT", self.root), patch.object(engine, "wait_for_training_data"), \
             patch.object(engine, "wait_for_research_batch", side_effect=pending), \
             patch.object(engine, "_train_cycle", side_effect=self.cycle):
            return engine.train_cycle(self.config, live=True)

    def fail_completion(self, error=None):
        with patch.object(self.module.ResearchControl, "complete_training", side_effect=error or OSError("통제된 완료 상태 저장 실패")):
            with self.assertRaises(BaseException):
                self.run_cycle()

    def test_recovery_after_completion_state_write_failure_never_fits_again(self):
        self.fail_completion()
        self.run_cycle()
        self.assertEqual(self.fit_calls, 1)
        self.assertIsNone(self.control.collection_target())

    def test_recovery_preserves_claim_on_completion_crash(self):
        self.fail_completion(SystemExit("통제된 완료 직전 중단"))
        batch = self.control.pending_batch()
        self.assertEqual(batch["status"], "training")
        self.assertIsNotNone(batch["claim"])
        self.run_cycle()
        self.assertEqual(self.fit_calls, 1)

    def test_recovery_finishes_pointer_report_or_progress_failure_without_fit(self):
        original_write = engine.atomic_json
        for name in ("training.json", "current.json", "training_progress.json"):
            with self.subTest(name=name):
                # 각 장애는 서로 다른 임시 프로젝트에서 같은 초기 batch에 주입합니다.
                with tempfile.TemporaryDirectory() as temp:
                    self.root = Path(temp)
                    self.control = self.module.ResearchControl(self.root, self.settings)
                    for key, folder in (("data_dir", "data"), ("models_dir", "models"), ("reports_dir", "reports")):
                        self.config[key] = str(self.root / "work" / folder)
                    self.fit_calls = 0

                    def write(path, payload):
                        if Path(path).name == name:
                            raise OSError("통제된 완료 후 파일 저장 실패")
                        return original_write(path, payload)

                    with patch.object(engine, "atomic_json", side_effect=write):
                        with self.assertRaises(OSError):
                            self.run_cycle()
                    self.run_cycle()
                    self.assertEqual(self.fit_calls, 1)
                    pointer = json.loads((Path(self.config["models_dir"]) / "current.json").read_text(encoding="utf-8"))
                    self.assertEqual(pointer["run"], self.last_report["run"])
                    self.assertIsNone(self.control.pending_batch())

    def test_recovery_rejects_changed_incomplete_or_foreign_report_without_fit(self):
        self.fail_completion()
        path = Path(self.config["reports_dir"]) / "experiments" / (self.last_report["run"] + ".json")
        original = path.read_text(encoding="utf-8")
        for field in ("status", "claim", "batch", "content"):
            with self.subTest(field=field):
                report = json.loads(original)
                if field == "status":
                    report["training"]["status"] = "학습 중"
                elif field == "claim":
                    report["research_batch"]["claim"] = "0" * 32
                elif field == "batch":
                    report["research_batch"]["batch_id"] = "다른 요청"
                else:
                    report["training"]["rows"] = 101
                path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaises((ValueError, RuntimeError)):
                    self.run_cycle()
                self.assertEqual(self.fit_calls, 1)
                self.assertIsNotNone(self.control.pending_batch())
                path.write_text(original, encoding="utf-8")

    def test_recovery_without_known_attempt_fails_closed(self):
        batch = self.control.pending_batch()
        self.control.claim_training(batch["id"])
        with self.assertRaises((ValueError, RuntimeError)):
            self.run_cycle()
        self.assertEqual(self.fit_calls, 0)

    def test_collecting_state_recovers_bound_result_without_new_claim_or_fit(self):
        self.fail_completion()
        state = self.control.snapshot()
        state["batches"][-1].update(status="collecting", claim=None)
        (self.root / self.settings["state_path"]).write_text(json.dumps(state), encoding="utf-8")
        self.run_cycle()
        self.assertEqual(self.fit_calls, 1)
        self.assertEqual(self.control.snapshot()["batches"][-1]["attempts"], 1)

    def test_durable_result_restores_missing_reports_and_pointer(self):
        self.fail_completion()
        reports, models = Path(self.config["reports_dir"]), Path(self.config["models_dir"])
        for path in (reports / "experiments" / (self.last_report["run"] + ".json"),
                     reports / "training.json", models / "current.json"):
            path.unlink()
        self.run_cycle()
        self.assertEqual(self.fit_calls, 1)
        self.assertEqual(json.loads((reports / "training.json").read_text(encoding="utf-8")), self.last_report)

    def test_changed_models_manifest_or_attempt_record_are_rejected_without_fit(self):
        self.fail_completion()
        run = self.last_report["run"]
        directory = Path(self.config["models_dir"]) / "versions" / run
        paths = [directory / "KRW-BTC-0.ubj", directory / "manifest.json",
                 self.control.attempt_path(self.control.pending_batch())]
        for path in paths:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                if path.suffix == ".ubj":
                    path.write_bytes(b"changed-model")
                else:
                    data = json.loads(original)
                    if path.name == "manifest.json":
                        data["source_hash"] = "changed-source"
                    else:
                        data["report"]["training"]["rows"] += 1
                    path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises((ValueError, RuntimeError)):
                    self.run_cycle()
                self.assertEqual(self.fit_calls, 1)
                path.write_bytes(original)

    def test_result_write_failure_preserves_claim_and_blocks_ambiguous_replay(self):
        with patch.object(self.module.ResearchControl, "store_result", side_effect=OSError("통제된 완료 결과 저장 실패")):
            with self.assertRaises(OSError):
                self.run_cycle()
        batch = self.control.pending_batch()
        self.assertEqual(batch["status"], "training")
        self.assertIsNotNone(batch["claim"])
        with self.assertRaisesRegex(ValueError, "미확인"):
            self.run_cycle()
        self.assertEqual(self.fit_calls, 1)

    def test_model_failure_is_recorded_separately_and_same_batch_can_retry(self):
        with patch.object(engine, "ROOT", self.root), patch.object(engine, "wait_for_training_data"), \
             patch.object(engine, "_train_cycle", side_effect=RuntimeError("통제된 모델 실패")):
            with self.assertRaises(RuntimeError):
                engine.train_cycle(self.config, live=True)
        batch = self.control.pending_batch()
        self.assertEqual(batch["status"], "collecting")
        self.assertEqual(self.control.read_attempt(batch)["phase"], "model_failed")
        self.run_cycle()
        self.assertEqual(self.fit_calls, 1)
        self.assertEqual(self.control.snapshot()["batches"][-1]["attempts"], 2)

    def test_summary_output_failure_after_completion_cannot_replay_fit(self):
        with patch.object(engine, "ROOT", self.root), patch.object(engine, "load_config", return_value=self.config), \
             patch.object(engine, "process_lock"), patch.object(engine, "wait_for_training_data"), \
             patch.object(engine, "_train_cycle", side_effect=self.cycle), \
             patch("builtins.print", side_effect=OSError("통제된 요약 출력 실패")):
            with self.assertRaises(OSError):
                engine.main(["trainer", "--once"])
        self.assertIsNone(self.control.pending_batch())
        with self.assertRaisesRegex(RuntimeError, "요청 대기"):
            self.run_cycle()
        self.assertEqual(self.fit_calls, 1)

    def test_actual_train_cycle_storage_branch_recovers_after_pointer_write_failure(self):
        self.config["training"].update(holdout_start=engine.PROTECTED_HOLDOUT[0],
                                       holdout_end=engine.PROTECTED_HOLDOUT[1], holdout_days=1)
        from contextlib import nullcontext
        import numpy as np
        import pandas as pd

        index = pd.date_range("2026-09-01", periods=3, freq="5min", tz="UTC")
        frame = pd.DataFrame({"x": [1., 2., 3.], "label": [0, 1, 2],
                              "label_end": index + pd.Timedelta(minutes=5)}, index=index)
        split = {key: frame for key in ("train", "tune", "calibration", "rule", "validation")}
        evidence = {"training_ready": True, "target_archive_bytes": 50_000_000_000,
                    "verified_archive_bytes": 50_000_000_000, "ingested_archive_bytes": 50_000_000_000}
        external = importlib.import_module("3차.external_features")
        audit = Mock()
        audit.consumed_periods.return_value = []
        model = Mock()
        model.save_model.side_effect = lambda path: Path(path).write_bytes(b"controlled-gpu-model")
        original_write = engine.atomic_json

        def fault(path, payload):
            if Path(path).name == "current.json":
                raise OSError("통제된 실제 저장 분기 포인터 실패")
            return original_write(path, payload)

        with patch.object(engine, "ROOT", self.root), patch.object(engine, "wait_for_training_data"), \
             patch.object(external, "resolve_for_training", return_value=(self.config["external"], evidence)), \
             patch.object(engine, "ExperimentAudit", return_value=audit), patch.object(engine, "CandleStore"), \
             patch.object(engine, "market_feature_loader", return_value=nullcontext(lambda market: (frame, {}))), \
             patch.object(engine, "label_frame", return_value=frame), patch.object(engine, "feature_columns", return_value=["x"]), \
             patch.object(engine, "research_splits", return_value=split), \
             patch.object(engine, "fit_gpu", return_value=(model, {"best_iteration": 0, "actual_device": "cuda:0"})) as fit, \
             patch.object(engine, "predict_booster", return_value=np.tile([.1, .8, .1], (3, 1))), \
             patch.object(engine, "tune_rule", side_effect=lambda *a, **k: {"confidence": 1.01, "edge": 1.01, "eligible": False, "trials": []}), \
             patch.object(engine, "backtest_portfolio", return_value={"net_return": 0}), \
             patch.object(engine, "assess_promotion", return_value={"promote": True, "eligibility": {"eligible": True}}):
            with patch.object(engine, "atomic_json", side_effect=fault):
                with self.assertRaises(OSError):
                    engine.train_cycle(self.config, live=True)
            self.assertEqual(fit.call_count, 5)
            self.assertEqual(self.control.pending_batch()["status"], "training")
            report = engine.train_cycle(self.config, live=True)
            self.assertEqual(fit.call_count, 5)
        self.assertEqual(report["training"]["completed_candidates"], 5)
        self.assertIsNone(self.control.pending_batch())


if __name__ == "__main__":
    unittest.main()
