"""연속 연구의 중복 요청·수동 모드·선행 수집 경계를 검사합니다."""
import importlib
import json
from concurrent.futures import ThreadPoolExecutor
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import test_research_control as fixtures


control_module = importlib.import_module('4차.research_control')
external = importlib.import_module('3차.external_data')


class ContinuousResearchTests(unittest.TestCase):
    setUp = fixtures.ResearchControlTests.setUp
    active_config = fixtures.ResearchControlTests.active_config
    completed_cycle = fixtures.ResearchControlTests.completed_cycle
    finish_initial = fixtures.ResearchControlTests.finish_initial

    def test_engine_loader_accepts_continuous_config_without_affecting_paper(self):
        path = self.root / 'config.json'
        path.write_text(json.dumps({'research': {**self.settings, 'continuous': True},
                                    'external': {'enabled': 'auto', 'wait_for_dataset': True}}), encoding='utf-8')
        config = fixtures.engine.load_config(path)
        self.assertIs(config['research']['continuous'], True)
        self.assertEqual(config['paper']['initial_cash'], 10_000_000)

    def test_continuous_defaults_off_and_rejects_non_bool(self):
        settings = control_module.validate_config(self.settings, self.root)
        self.assertIs(settings.get('continuous'), False)
        with self.assertRaises(ValueError):
            control_module.validate_config({**self.settings, 'continuous': 'true'}, self.root)

    def test_completed_batch_auto_queues_once_from_actual_ingested(self):
        self.finish_initial()
        control = control_module.ResearchControl(self.root, {**self.settings, 'continuous': True})
        reader = Mock(return_value=50_236_409_654)
        batch = control.ensure_continuous_batch(reader)
        self.assertEqual(batch['baseline_bytes'], 50_236_409_654)
        self.assertEqual(batch['target_bytes'], 52_236_409_654)
        self.assertEqual(control.ensure_continuous_batch(reader), batch)
        reader.assert_called_once_with()
        self.assertEqual(len(control.snapshot()['batches']), 2)

    def test_manual_mode_never_reads_or_creates_increment(self):
        self.finish_initial()
        reader = Mock(side_effect=AssertionError('수동 모드에서 적재량을 읽었습니다'))
        self.assertIsNone(self.control.ensure_continuous_batch(reader))
        self.assertEqual(len(self.control.snapshot()['batches']), 1)

    def test_training_reuses_batch_without_reading_baseline(self):
        control = control_module.ResearchControl(self.root, {**self.settings, 'continuous': True})
        first = control.pending_batch()
        training = control.claim_training(first['id'])
        reader = Mock(side_effect=AssertionError('학습 중 새 batch를 생성했습니다'))
        self.assertEqual(control.ensure_continuous_batch(reader), training)
        reader.assert_not_called()

    def test_bad_baseline_preserves_completed_state(self):
        self.finish_initial()
        control = control_module.ResearchControl(self.root, {**self.settings, 'continuous': True})
        before = control.snapshot()
        with self.assertRaises(ValueError):
            control.ensure_continuous_batch(lambda: True)
        self.assertEqual(before, control.snapshot())

    def test_concurrent_requests_and_restart_keep_one_batch(self):
        self.finish_initial()
        settings = {**self.settings, 'continuous': True}
        control = control_module.ResearchControl(self.root, settings)
        reader = Mock(return_value=51_000_000_000)
        with ThreadPoolExecutor(max_workers=3) as pool:
            batches = list(pool.map(lambda _: control.ensure_continuous_batch(reader), range(3)))
        self.assertEqual(len({batch['id'] for batch in batches}), 1)
        reader.assert_called_once_with()
        restarted = control_module.ResearchControl(self.root, settings)
        self.assertEqual(restarted.ensure_continuous_batch(reader), batches[0])
        self.assertEqual(len(restarted.snapshot()['batches']), 2)

    def test_stop_during_baseline_read_never_persists_next_request(self):
        self.finish_initial()
        control = control_module.ResearchControl(self.root, {**self.settings, 'continuous': True})
        stop = threading.Event()

        def actual():
            stop.set()
            return 51_000_000_000

        result = control.ensure_continuous_batch(actual, stop=stop)
        self.assertIsNone(result)
        self.assertEqual(len(control.snapshot()['batches']), 1)

    def test_archive_reverification_does_not_hold_research_state_lock(self):
        self.finish_initial()
        control = control_module.ResearchControl(self.root, {**self.settings, 'continuous': True})
        with ThreadPoolExecutor(max_workers=1) as pool:
            def actual():
                # 실제 원본 검증이 느려도 trainer의 상태 조회는 바로 완료돼야 합니다.
                state = pool.submit(control.snapshot).result(timeout=.5)
                self.assertEqual(state['batches'][-1]['status'], 'completed')
                return 51_000_000_000

            batch = control.ensure_continuous_batch(actual)
        self.assertEqual(batch['target_bytes'], 53_000_000_000)


class ContinuousCollectorTests(unittest.TestCase):
    def test_ready_bootstrap_does_not_bypass_prefetch_cap(self):
        collector = external.Collector.__new__(external.Collector)
        collector.research_control = Mock()
        collector.collection_target = Mock(return_value=52_000_000_000)
        collector.status = Mock(return_value={'ingested_archive_bytes': 52_000_000_000,
                                              'training_ready': True})
        collector.archive = Mock()
        stop = threading.Event()
        with patch.object(external, 'STOP', stop), patch.object(external.threading, 'Thread') as thread, \
             patch.object(external, 'bootstrap_plan', return_value=[object()]), \
             patch.object(external, 'bulk_plan', return_value=[]), \
             patch.object(stop, 'wait', side_effect=lambda _: stop.set()):
            thread.return_value.is_alive.return_value = False
            collector.run()
        collector.archive.assert_not_called()

    def test_prefetch_target_is_bounded_and_does_not_change_training_target(self):
        with tempfile.TemporaryDirectory() as temp:
            control = control_module.ResearchControl(Path(temp), {'active': True, 'continuous': True})
            batch = control.pending_batch()
            control.claim_training(batch['id'])
            collector = external.Collector.__new__(external.Collector)
            collector.lock = threading.RLock()
            collector.research_control = control
            collector.target = 50_000_000_000
            collector.status = Mock(side_effect=AssertionError('기존 학습에서 상태 재검증 불필요'))
            with patch.object(external, 'STOP', threading.Event()):
                self.assertEqual(collector.collection_target(), 52_000_000_000)
                self.assertEqual(collector.collection_target(), 52_000_000_000)
                self.assertEqual(collector.target, 50_000_000_000)
            self.assertEqual(control.pending_batch()['target_bytes'], 50_000_000_000)

    def test_stop_does_not_create_new_batch(self):
        collector = external.Collector.__new__(external.Collector)
        collector.lock = threading.RLock()
        collector.research_control = Mock()
        collector.research_control.settings = {'continuous': True}
        stop = threading.Event()
        stop.set()
        with patch.object(external, 'STOP', stop):
            self.assertIsNone(collector.collection_target())
        collector.research_control.ensure_continuous_batch.assert_not_called()
