"""대기 하트비트는 제외하고 실제 학습 시간 제한은 유지하는지 검사합니다."""
import datetime as dt
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import supervisor


class ResearchTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(supervisor, "TrainingBudget"), "대기와 학습을 구분하는 시간 예산이 없습니다")
        self.started = dt.datetime(2026, 11, 2, tzinfo=dt.timezone.utc)

    def heartbeat(self, now, status="waiting_request", stage="idle", pid=17):
        return {"updated_at": now.isoformat(), "pid": pid, "research_active": True,
                "status": status, "stage": stage, "run": None}

    def test_twelve_second_wait_heartbeats_survive_more_than_six_hours(self):
        budget = supervisor.TrainingBudget(21600, 0, self.started, 17)
        for second in range(12, 21600 + 120, 12):
            now = self.started + dt.timedelta(seconds=second)
            self.assertFalse(budget.expired(second, now, self.heartbeat(now)))
        self.assertEqual(budget.active_seconds, 0)

    def test_fit_and_evaluation_progress_never_reset_active_budget(self):
        budget = supervisor.TrainingBudget(21600, 0, self.started, 17)
        self.assertFalse(budget.expired(10000, self.started + dt.timedelta(seconds=10000),
                                      self.heartbeat(self.started + dt.timedelta(seconds=10000), "running", "fit")))
        self.assertTrue(budget.expired(21600, self.started + dt.timedelta(seconds=21600),
                                     self.heartbeat(self.started + dt.timedelta(seconds=21600), "running", "calibration")))

    def test_wait_exclusion_does_not_reset_accumulated_fit_time(self):
        budget = supervisor.TrainingBudget(20, 0, self.started, 17)
        self.assertFalse(budget.expired(8, self.started + dt.timedelta(seconds=8), {}))
        now = self.started + dt.timedelta(seconds=100)
        self.assertFalse(budget.expired(100, now, self.heartbeat(now, "waiting_data", "download")))
        self.assertTrue(budget.expired(112, self.started + dt.timedelta(seconds=112), {}))

    def test_stale_future_foreign_pid_and_wrong_stage_do_not_exempt_timeout(self):
        now = self.started + dt.timedelta(seconds=100)
        invalids = [self.heartbeat(self.started), self.heartbeat(now + dt.timedelta(seconds=1)),
                    self.heartbeat(now, pid=18), self.heartbeat(now, stage="fit"),
                    {**self.heartbeat(now), "research_active": False}, {}, None]
        for record in invalids:
            with self.subTest(record=record):
                budget = supervisor.TrainingBudget(20, 0, self.started, 17)
                self.assertTrue(budget.expired(100, now, record))

    def test_wait_before_child_launch_cannot_exempt_timeout(self):
        budget = supervisor.TrainingBudget(20, 0, self.started, 17)
        self.assertTrue(budget.expired(20, self.started + dt.timedelta(seconds=20),
                                     self.heartbeat(self.started - dt.timedelta(seconds=1))))

    def test_only_trainer_reads_heartbeat_and_missing_file_is_not_wait(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            self.assertEqual(supervisor.training_heartbeat("trainer", work), {})
            with patch.object(supervisor.Path, "read_text", return_value='{"status":"waiting_request"}') as read:
                self.assertEqual(supervisor.training_heartbeat("paper", work), {})
                read.assert_not_called()
                self.assertEqual(supervisor.training_heartbeat("trainer", work)["status"], "waiting_request")


if __name__ == "__main__":
    unittest.main()
