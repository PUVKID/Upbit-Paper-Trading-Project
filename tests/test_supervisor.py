"""예약시각 전에는 실시간 모의거래를 시작하지 않는지 검사합니다."""

import datetime as dt
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

PATH = Path(__file__).resolve().parents[1] / "supervisor.py"
SPEC = importlib.util.spec_from_file_location("supervisor", PATH)
SUPERVISOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUPERVISOR)


class ScheduleTests(unittest.TestCase):
    def test_trainer_success_has_no_fixed_delay(self):
        config = json.loads((PATH.parent / "service.json").read_text(encoding="utf-8"))
        self.assertEqual(config["trainer"]["interval_seconds"], 0)
        self.assertIn("--once", config["trainer"]["args"])
        self.assertEqual(config["collector"]["interval_seconds"], 15)
        self.assertEqual(config["paper"]["interval_seconds"], 15)

    def test_zero_delay_records_actual_success_gap_and_next_cycle(self):
        stop = Mock()
        stop.is_set.side_effect = [False, False, True]
        processes = [Mock(stdout=io.StringIO(""), returncode=0) for _ in range(2)]
        for process in processes:
            process.poll.return_value = 0
        records = []
        times = [dt.datetime(2026, 10, 21, 9, 0, second, tzinfo=dt.timezone.utc)
                 for second in (0, 3, 4, 8, 9)]
        with patch.object(SUPERVISOR, "STOP", stop), \
             patch.object(SUPERVISOR, "logger_for", return_value=Mock()), \
             patch.object(SUPERVISOR, "utcnow", side_effect=times), \
             patch.object(SUPERVISOR, "atomic_json", side_effect=lambda path, value: records.append(value.copy())), \
             patch.object(SUPERVISOR.subprocess, "Popen", side_effect=processes):
            SUPERVISOR.worker("trainer", {"args": ["trainer", "--once"], "interval_seconds": 0},
                              Path("."), times[0])
        starts = [item for item in records if item["state"] == "실행 중"]
        self.assertEqual(len(starts), 2)
        self.assertEqual(starts[1]["previous_finished_utc"], times[1].isoformat())
        self.assertEqual(starts[1]["success_gap_seconds"], 1)
        self.assertEqual(starts[1]["successful_cycles"], 1)
        self.assertEqual(starts[1]["iteration"], 2)
        finished = [item for item in records if item.get("exit_code") == 0]
        self.assertEqual(finished[0]["next_iteration"], 2)
        self.assertEqual(finished[0]["duration_seconds"], 3)
        self.assertEqual(finished[1]["successful_cycles"], 2)
        self.assertEqual([call.args[0] for call in stop.wait.call_args_list], [0, 0])

    def test_error_backoff_is_not_zero_success_delay(self):
        stop = Mock()
        stop.is_set.side_effect = [False, True]
        records = []
        with patch.object(SUPERVISOR, "STOP", stop), \
             patch.object(SUPERVISOR, "logger_for", return_value=Mock()), \
             patch.object(SUPERVISOR, "atomic_json", side_effect=lambda path, value: records.append(value)), \
             patch.object(SUPERVISOR.subprocess, "Popen", side_effect=OSError("검사 오류")):
            SUPERVISOR.worker("trainer", {"args": ["trainer"], "interval_seconds": 0}, Path("."),
                              dt.datetime(2026, 10, 21, 9, tzinfo=dt.timezone.utc))
        stop.wait.assert_called_once_with(30)
        self.assertEqual(records[1]["state"], "오류 후 재시도 대기")

    def test_before_start_never_spawns_paper(self):
        stop = Mock()
        stop.is_set.side_effect = [False, True]
        records = []
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            with patch.object(SUPERVISOR, "STOP", stop), patch.object(SUPERVISOR, "logger_for", return_value=Mock()), \
                 patch.object(SUPERVISOR, "atomic_json", side_effect=lambda path, value: records.append(value)), \
                 patch.object(SUPERVISOR, "utcnow", return_value=dt.datetime(2026, 10, 21, 8, 59, tzinfo=dt.timezone.utc)), \
                 patch.object(SUPERVISOR.subprocess, "Popen") as popen:
                SUPERVISOR.worker("paper", {"args": ["paper"], "interval_seconds": 10}, work,
                                  dt.datetime(2026, 10, 21, 9, tzinfo=dt.timezone.utc))
                popen.assert_not_called()
                self.assertEqual(records[0]["state"], "예정시각 대기")

    def test_two_real_child_processes_connect_and_stop_with_status_file(self):
        stop = threading.Event()
        records = []
        write = SUPERVISOR.atomic_json

        def record(path, value):
            write(path, value)
            records.append(value.copy())
            if value.get("exit_code") == 0 and value["iteration"] == 2:
                stop.set()

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "service").mkdir()
            with patch.object(SUPERVISOR, "STOP", stop), \
                 patch.object(SUPERVISOR, "logger_for", return_value=Mock()), \
                 patch.object(SUPERVISOR, "atomic_json", side_effect=record):
                SUPERVISOR.worker("trainer", {"script": "-c", "args": ["print('cycle')"],
                    "interval_seconds": 0, "timeout_seconds": 10}, work, SUPERVISOR.utcnow())
            final = json.loads((work / "service/trainer.json").read_text(encoding="utf-8"))
        starts = [item for item in records if item["state"] == "실행 중"]
        self.assertEqual(len(starts), 2)
        gap = (dt.datetime.fromisoformat(starts[1]["started_utc"]) -
               dt.datetime.fromisoformat(starts[1]["previous_finished_utc"])).total_seconds()
        self.assertEqual(starts[1]["success_gap_seconds"], gap)
        self.assertGreaterEqual(gap, 0)
        self.assertEqual(final["state"], "중지됨")
        self.assertEqual(final["successful_cycles"], 2)

    def test_timeout_terminates_and_keeps_error_backoff(self):
        stop = Mock()
        stop.is_set.side_effect = [False, False, True]
        stop.wait.return_value = False
        process = Mock(stdout=io.StringIO(""), returncode=0)
        process.poll.side_effect = [None, 0]
        records = []
        with patch.object(SUPERVISOR, "STOP", stop), \
             patch.object(SUPERVISOR, "logger_for", return_value=Mock()), \
             patch.object(SUPERVISOR, "atomic_json", side_effect=lambda path, value: records.append(value)), \
             patch.object(SUPERVISOR.time, "monotonic", side_effect=[0, 2]), \
             patch.object(SUPERVISOR.subprocess, "Popen", return_value=process):
            SUPERVISOR.worker("trainer", {"args": ["trainer"], "interval_seconds": 0,
                "timeout_seconds": 1}, Path("."), SUPERVISOR.utcnow())
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=20)
        self.assertTrue(process.stdout.closed)
        self.assertEqual(records[1]["error"], "작업 시간 제한 초과")
        self.assertEqual(stop.wait.call_args_list[-1].args[0], 30)

    def test_stop_terminates_child_and_closes_output(self):
        stop = Mock()
        stop.is_set.side_effect = [False, True, True]
        stop.wait.return_value = True
        process = Mock(stdout=io.StringIO(""), returncode=-15)
        process.poll.side_effect = [None, -15]
        with patch.object(SUPERVISOR, "STOP", stop), \
             patch.object(SUPERVISOR, "logger_for", return_value=Mock()), \
             patch.object(SUPERVISOR, "atomic_json"), \
             patch.object(SUPERVISOR.subprocess, "Popen", return_value=process):
            SUPERVISOR.worker("trainer", {"args": ["trainer"], "interval_seconds": 0},
                              Path("."), SUPERVISOR.utcnow())
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=20)
        self.assertTrue(process.stdout.closed)


if __name__ == "__main__":
    unittest.main()
