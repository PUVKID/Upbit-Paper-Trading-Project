"""추가 학습 상태의 안전한 공개 투영과 이전 연구 분리를 검사합니다."""
import json
from pathlib import Path
import tempfile
import unittest

from dashboard import SnapshotAdapter


class TrainingRepetitionDashboardTests(unittest.TestCase):
    def fixture(self, **updates):
        return {"state":"training", "stage":"finetune", "model_id":"amazon/chronos-2", "run":"repeat-run",
                "rep_index":1, "total_repetitions":10, "completed_repetitions":0,
                "step":7, "total_steps":500, "loss":0.5, "device":"cuda:0", "rows":1500,
                "context_length":256, "batch_size":8, "lr":0.000001, "checkpoint_verified":False,
                "updated_utc":"2026-11-02T15:00:00Z", "evaluation_performed":False,
                "candidate_ready":False, "promote":False,
                "validation_mae":None, "baseline_mae":None, "direction_accuracy":None,
                "net_return":None, "trades":None, **updates}

    def status(self, root, value):
        target = root / "8차/work/status.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value), encoding="utf-8")
        return SnapshotAdapter(root).status()

    def test_missing_repetition_preserves_existing_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            view = SnapshotAdapter(Path(folder)).status()
            self.assertEqual(view["training_repetition"], {})
            self.assertEqual(view["foundation_research"], {})

    def test_only_safe_fields_exposed_and_original_error_is_private(self):
        with tempfile.TemporaryDirectory() as folder:
            value = self.fixture(error="/private/token-secret", checkpoint_path="/private/weights", argv=["secret"])
            view = self.status(Path(folder), value)["training_repetition"]
            self.assertEqual(view["step"], 7)
            self.assertEqual(view["rep_index"], 1)
            self.assertEqual(view["lr"], 0.000001)
            self.assertTrue(view["has_error"])
            for key in ("error", "checkpoint_path", "argv"):
                self.assertNotIn(key, view)
            self.assertNotIn("private", json.dumps(view))

    def test_train_only_metrics_cannot_leak_stale_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            value = self.fixture(validation_mae=.01, baseline_mae=.02, direction_accuracy=.99, net_return=5, trades=99)
            view = self.status(Path(folder), value)["training_repetition"]
            for key in ("validation_mae", "baseline_mae", "direction_accuracy", "net_return", "trades"):
                self.assertIsNone(view[key])
            for key in ("evaluation_performed", "candidate_ready", "promote"):
                self.assertIs(view[key], False)

    def test_completed_training_is_not_evaluation_or_promotion(self):
        with tempfile.TemporaryDirectory() as folder:
            view = self.status(Path(folder), self.fixture(state="complete", stage="complete", rep_index=10,
                completed_repetitions=10, step=500, checkpoint_verified=True))["training_repetition"]
            self.assertEqual(view["state"], "complete")
            self.assertEqual(view["completed_repetitions"], 10)
            self.assertTrue(view["checkpoint_verified"])
            self.assertIs(view["candidate_ready"], False)

    def test_bad_core_types_and_ranges_are_not_an_active_record(self):
        variants = [{"step":True}, {"step":"50"}, {"step":501}, {"step":-1}, {"rep_index":0},
                    {"total_repetitions":9}, {"total_steps":50}, {"completed_repetitions":11},
                    {"run":"/private/path"}, {"model_id":"/private/model"}, {"state":{"private":"secret"}},
                    {"evaluation_performed":True}, {"candidate_ready":True}, {"promote":True}]
        with tempfile.TemporaryDirectory() as folder:
            for update in variants:
                with self.subTest(update=update):
                    self.assertEqual(self.status(Path(folder), self.fixture(**update))["training_repetition"], {})

    def test_bad_optional_values_are_not_passed_as_raw_data(self):
        with tempfile.TemporaryDirectory() as folder:
            value = self.fixture(loss=True, lr="/secret", rows="1500", device="/private/cuda",
                                 updated_utc="/private/time", context_length=False, batch_size=[])
            view = self.status(Path(folder), value)["training_repetition"]
            self.assertEqual(view["step"], 7)
            for key in ("loss", "lr", "rows", "device", "updated_utc", "context_length", "batch_size"):
                self.assertIsNone(view.get(key))

    def test_non_object_records_are_not_an_api_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            for value in (["secret"], "secret", 8, None):
                with self.subTest(value=value):
                    self.assertEqual(self.status(Path(folder), value)["training_repetition"], {})

    def test_foundation_record_is_preserved_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root / "7차/work/status.json"
            old.parent.mkdir(parents=True)
            old.write_text(json.dumps({"state":"waiting", "model_id":"amazon/chronos-2", "run":"foundation-run"}), encoding="utf-8")
            view = self.status(root, self.fixture())
            self.assertEqual(view["foundation_research"]["run"], "foundation-run")
            self.assertEqual(view["training_repetition"]["run"], "repeat-run")

    def test_actual_runner_learning_rate_and_stage_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            for stage in ("extract", "finetune", "verify", "complete"):
                with self.subTest(stage=stage):
                    source = self.fixture(stage=stage, learning_rate=0.000001, context_length=192)
                    del source["lr"]
                    view = self.status(Path(folder), source)["training_repetition"]
                    self.assertEqual(view["lr"], 0.000001)
                    self.assertEqual(view["context_length"], 192)
                    self.assertEqual(view["stage"], stage)


if __name__ == "__main__":
    unittest.main()
