"""운영 학습 없이 사용자 중지 표식과 공통 진입 경계를 검사합니다."""
import contextlib
import copy
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import engine


class TrainingPauseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.marker = self.root / "work/service/TRAINING_PAUSED"
        self.marker.parent.mkdir(parents=True)

    def test_absent_marker_returns_without_heartbeat_or_sleep(self):
        stop = Mock()
        with patch.object(engine, "ROOT", self.root), patch.object(engine, "progress") as progress:
            engine.wait_for_training_pause(stop)
        progress.assert_not_called()
        stop.wait.assert_not_called()
        self.assertFalse(self.marker.exists())

    def test_waits_in_twelve_second_heartbeats_until_explicit_removal(self):
        self.marker.write_text("사용자 중지", encoding="utf-8")
        stop = Mock()
        def tick(seconds):
            self.assertEqual(seconds, 12)
            self.assertTrue(self.marker.is_file())
            if stop.wait.call_count == 3:
                self.marker.unlink()
            return False
        stop.wait.side_effect = tick
        with patch.object(engine, "ROOT", self.root), patch.object(engine, "progress") as progress, \
                patch.object(engine.os, "getpid", return_value=12345):
            engine.wait_for_training_pause(stop)
        self.assertEqual(stop.wait.call_count, 3)
        self.assertEqual(progress.call_count, 3)
        for call in progress.call_args_list:
            fields = call.kwargs
            self.assertEqual(fields["status"], "waiting_request")
            self.assertEqual(fields["stage"], "idle")
            self.assertEqual(fields["reason"], "사용자 요청으로 학습 중지")
            self.assertIs(fields["research_active"], True)
            self.assertIs(fields["training_paused"], True)
            self.assertEqual(fields["pid"], 12345)

    def test_interrupted_wait_never_deletes_marker(self):
        self.marker.touch()
        stop = Mock()
        stop.wait.return_value = True
        with patch.object(engine, "ROOT", self.root), patch.object(engine, "progress"):
            with self.assertRaises(KeyboardInterrupt):
                engine.wait_for_training_pause(stop)
        self.assertTrue(self.marker.is_file())

    def test_marker_directory_fails_closed(self):
        self.marker.mkdir()
        with patch.object(engine, "ROOT", self.root), patch.object(engine, "progress") as progress:
            with self.assertRaises(SystemExit):
                engine.wait_for_training_pause(Mock())
        progress.assert_not_called()
        self.assertTrue(self.marker.is_dir())

    def test_marker_and_parent_symlinks_fail_closed_without_following(self):
        original = Path.lstat
        for link in (self.root, self.root / "work", self.marker.parent, self.marker):
            with self.subTest(link=link):
                def lstat(path, *args, **kwargs):
                    return SimpleNamespace(st_mode=stat.S_IFLNK | 0o777) if path == link else original(path, *args, **kwargs)
                with patch.object(engine, "ROOT", self.root), patch.object(Path, "lstat", lstat), \
                        patch.object(engine, "progress") as progress:
                    with self.assertRaises(SystemExit):
                        engine.wait_for_training_pause(Mock())
                progress.assert_not_called()

    def test_windows_reparse_point_fails_closed(self):
        original = Path.lstat
        def lstat(path, *args, **kwargs):
            return SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_file_attributes=0x400) if path == self.marker.parent else original(path, *args, **kwargs)
        with patch.object(engine, "ROOT", self.root), patch.object(Path, "lstat", lstat), patch.object(engine, "progress") as progress:
            with self.assertRaises(SystemExit):
                engine.wait_for_training_pause(Mock())
        progress.assert_not_called()

    def test_permission_error_is_not_treated_as_marker_absence(self):
        original = Path.lstat
        def lstat(path, *args, **kwargs):
            if path == self.marker:
                raise PermissionError("통제된 접근 거부")
            return original(path, *args, **kwargs)
        with patch.object(engine, "ROOT", self.root), patch.object(Path, "lstat", lstat), patch.object(engine, "progress") as progress:
            with self.assertRaises(SystemExit):
                engine.wait_for_training_pause(Mock())
        progress.assert_not_called()

    def test_main_trainer_and_experiment_block_before_config_gpu_or_database(self):
        self.marker.touch()
        for command in ("trainer", "experiment"):
            with self.subTest(command=command), patch.object(engine, "ROOT", self.root), \
                    patch.object(engine, "progress"), patch.object(engine.threading, "Event") as event, \
                    patch.object(engine, "load_config") as config, patch.object(engine, "train_cycle", return_value={"run":"mock", "locked_test":{}}) as train, \
                    patch.object(engine, "check_gpu") as gpu, patch.object(engine, "CandleStore") as store, \
                    patch.object(engine, "connect") as connect, patch.object(engine, "process_lock") as lock:
                event.return_value.wait.return_value = True
                with self.assertRaises(KeyboardInterrupt):
                    engine.main([command] + (["--once"] if command == "trainer" else []))
                config.assert_not_called()
                train.assert_not_called()
                gpu.assert_not_called()
                store.assert_not_called()
                connect.assert_not_called()
                lock.assert_not_called()
        self.assertTrue(self.marker.exists())

    def test_main_trainer_rechecks_between_cycles(self):
        config = copy.deepcopy(engine.DEFAULT_CONFIG)
        calls = 0
        def first_fit(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise AssertionError("중지 표식 이후 두 번째 학습 진입")
            self.marker.touch()
            return {"run":"mock-only"}
        with patch.object(engine, "ROOT", self.root), patch.object(engine, "load_config", return_value=config), \
                patch.object(engine, "process_lock", return_value=contextlib.nullcontext()), patch.object(engine, "progress"), \
                patch.object(engine.threading, "Event") as event, patch.object(engine, "train_cycle", side_effect=first_fit) as train, \
                patch("builtins.print"):
            event.return_value.wait.return_value = True
            with self.assertRaises(KeyboardInterrupt):
                engine.main(["trainer"])
            self.assertEqual(train.call_count, 1)

    def test_experiment_rechecks_after_lock_before_training(self):
        @contextlib.contextmanager
        def lock(*args):
            self.marker.touch()
            yield
        with patch.object(engine, "ROOT", self.root), patch.object(engine, "load_config", return_value=copy.deepcopy(engine.DEFAULT_CONFIG)), \
                patch.object(engine, "process_lock", side_effect=lock), patch.object(engine, "progress"), \
                patch.object(engine.threading, "Event") as event, patch.object(engine, "train_cycle") as train:
            event.return_value.wait.return_value = True
            with self.assertRaises(KeyboardInterrupt):
                engine.main(["experiment", "--final"])
            train.assert_not_called()

    def test_absent_marker_preserves_trainer_and_experiment_calls(self):
        for argv, kwargs in ((["trainer", "--once"], {"live":True}), (["experiment", "--final"], {"final":True})):
            with self.subTest(argv=argv), patch.object(engine, "ROOT", self.root), \
                    patch.object(engine, "load_config", return_value=copy.deepcopy(engine.DEFAULT_CONFIG)), \
                    patch.object(engine, "process_lock", return_value=contextlib.nullcontext()), \
                    patch.object(engine, "progress") as progress, patch.object(engine.threading, "Event") as event, \
                    patch.object(engine, "train_cycle", return_value={"run":"mock", "locked_test":{}}) as train, patch("builtins.print"):
                self.assertEqual(engine.main(argv), 0)
                self.assertEqual(train.call_args.kwargs, kwargs)
                train.assert_called_once()
                progress.assert_not_called()
                event.return_value.wait.assert_not_called()

    def test_collector_and_paper_ignore_training_pause_marker(self):
        self.marker.touch()
        for command in ("collector", "paper"):
            with self.subTest(command=command), patch.object(engine, "ROOT", self.root), \
                    patch.object(engine, "load_config", return_value=copy.deepcopy(engine.DEFAULT_CONFIG)), \
                    patch.object(engine, "process_lock", return_value=contextlib.nullcontext()), patch.object(engine, "progress") as progress, \
                    patch.object(engine.threading, "Event") as event, patch.object(engine, "collect_cycle", return_value=[]) as collect, \
                    patch.object(engine, "paper_cycle", return_value={"status":"mock-only"}) as paper, patch.object(engine, "train_cycle") as train, \
                    patch("builtins.print"):
                self.assertEqual(engine.main([command, "--once"]), 0)
                (collect if command == "collector" else paper).assert_called_once()
                (paper if command == "collector" else collect).assert_not_called()
                train.assert_not_called()
                progress.assert_not_called()
                event.assert_not_called()


if __name__ == "__main__":
    unittest.main()
