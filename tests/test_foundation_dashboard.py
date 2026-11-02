"""사전학습 연구의 읽기 전용 상태와 비공개 경계를 검사합니다."""

import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from dashboard import SnapshotAdapter


class FoundationDashboardTests(unittest.TestCase):
    def completion_fixture(self, root):
        run = "committed-run"
        work = root / "7차/work"
        folder = work / "runs" / run
        folder.mkdir(parents=True)
        def write(path, value):
            raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
            path.write_bytes(raw)
            return hashlib.sha256(raw).hexdigest()
        completed = {"run":run,"state":"complete","stage":"complete","model_id":"amazon/chronos-2",
                     "candidate_ready":True,"checkpoint_verified":True,"step":50,"total_steps":50,"loss":0.5}
        completion_hash = write(folder / "completion.json", completed)
        manifest_hash = write(folder / "manifest.json", {"run":run,"complete":True,"model_id":"amazon/chronos-2",
             "checkpoint_verified":True,"completion_sha256":completion_hash})
        write(work / "knowledge.json", {"run":run,"manifest_sha256":manifest_hash,"completion_sha256":completion_hash})
        return work, folder, completed

    def test_missing_research_does_not_invent_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            status = SnapshotAdapter(Path(folder)).status()
            self.assertEqual(status["foundation_research"], {})

    def test_only_explicit_public_fields_are_projected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "7차/work/status.json"
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps({"state": "training", "model_id": "amazon/chronos-2",
                "stage": "finetune", "step": 4, "total_steps": 100, "loss": 0.123,
                "updated_utc": "2026-11-02T08:00:00Z", "candidate_ready": False,
                "error": "비공개 경로/인증값", "token": "secret", "checkpoint_path": "/private"}), encoding="utf-8")
            view = SnapshotAdapter(root).status()["foundation_research"]
            self.assertEqual(view["step"], 4)
            self.assertFalse(view["candidate_ready"])
            self.assertTrue(view["has_error"])
            for key in ("error", "token", "checkpoint_path"):
                self.assertNotIn(key, view)

    def test_non_object_research_is_not_an_api_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "7차/work/status.json"
            target.parent.mkdir(parents=True)
            for value in (["secret"], "secret", 7):
                with self.subTest(value=value):
                    target.write_text(json.dumps(value), encoding="utf-8")
                    self.assertEqual(SnapshotAdapter(root).status()["foundation_research"], {})

    def test_committed_completion_recovers_stale_cache_with_warning(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            work, _, completed = self.completion_fixture(root)
            (work / "status.json").write_text(json.dumps({**completed,"state":"error",
                  "candidate_ready":False,"checkpoint_verified":False}), encoding="utf-8")
            view = SnapshotAdapter(root).status()["foundation_research"]
            self.assertEqual(view["state"], "complete")
            self.assertTrue(view["checkpoint_verified"])
            self.assertTrue(view["has_status_warning"])
            self.assertFalse(view["has_error"])

    def test_committed_record_cannot_replace_new_active_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            work, _, _ = self.completion_fixture(root)
            (work / "status.json").write_text(json.dumps({"run":"new-run","state":"training","step":3}), encoding="utf-8")
            view = SnapshotAdapter(root).status()["foundation_research"]
            self.assertEqual(view["run"], "new-run")
            self.assertEqual(view["state"], "training")

    def test_tampered_completion_is_not_shown_as_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            work, run, completed = self.completion_fixture(root)
            (work / "status.json").write_text(json.dumps(completed), encoding="utf-8")
            (run / "completion.json").write_text(json.dumps({**completed,"loss":0}), encoding="utf-8")
            view = SnapshotAdapter(root).status()["foundation_research"]
            self.assertEqual(view["state"], "error")
            self.assertFalse(view["checkpoint_verified"])
            self.assertTrue(view["has_error"])


if __name__ == "__main__":
    unittest.main()
