"""8차의 반복 fit·계보·재실행 거부와 평가 미노출을 검사합니다."""
import contextlib
import datetime as dt
import importlib.util
import io
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1]/'8차/train_only_repeats.py'
UTC = dt.timezone.utc


class TrainOnlyRepeatsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = None
        if SOURCE.is_file():
            spec = importlib.util.spec_from_file_location('train_only_eight_test', SOURCE)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def setUp(self):
        self.assertIsNotNone(self.module, '8차 train-only runner가 아직 없습니다')
        self.m = self.module
        self.f = self.m.foundation
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = self.m.validate_settings({})
        self.plan = {'train': ['2026-10-01T00:00:00Z', '2026-10-02T14:00:00Z'],
                     'tune': ['2026-10-02T14:00:00Z', '2026-10-03T08:00:00Z'],
                     'validation': ['2026-10-03T08:00:00Z', '2026-10-04T08:00:00Z']}
        self.start = dt.datetime(2026, 10, 1, tzinfo=UTC)
        self.rows = {market: [(self.start+dt.timedelta(minutes=5*i), 100.0+i)
                             for i in range(673)] for market in self.f.MARKETS}
        seed = self.root/'7차/work/runs/seed/checkpoint'
        seed.mkdir(parents=True)
        self.f.atomic_json(seed/'config.json', {'fake': True})
        (seed/'model.safetensors').write_bytes(b'0')
        manifest = {'schema_version': 1, 'run': 'seed', 'complete': True,
                    'checkpoint_verified': True, 'model_id': self.f.MODEL_ID,
                    'revision': self.f.REVISION, 'paper_only': True, 'promote': False,
                    'execution': {'weight_changed': True, 'device': 'cuda:0',
                                  'after_weights_sha256': self.weight(0)},
                    'checkpoint_files': self.f.checkpoint_files(seed),
                    'exposure_end': self.plan['validation'][1], 'plan': self.plan,
                    'data': {'protected_periods': []}, 'parent_run': None}
        self.f.atomic_json(seed.parent/'manifest.json', manifest)
        self.f.atomic_json(self.root/'7차/work/knowledge.json',
                           {'run': 'seed', 'manifest_sha256': self.f.file_hash(seed.parent/'manifest.json')})
        self.f.atomic_json(self.root/'3차/work/status.json', {'schema_version': 1})
        self.initial = {'bootstrap_run': 'verified-initial-50GB', 'reported_ingested_bytes': 52_250_140_144}
        self.train_calls, self.loads, self.probes = [], [], []
        self.fail_fit = None
        self.no_change = False
        self.bad_reload = False
        self.cpu = False
        self.fail_load = False
        self.history = []

    def pipeline(self, number, wrapped=False):
        forward = lambda *args, **kwargs: None
        model = SimpleNamespace(dtype='torch.float32', training=False,
                                config=SimpleNamespace(_attn_implementation='sdpa'), forward=forward)
        model.eval = lambda: setattr(model, 'training', False)
        if wrapped:
            model._original_forward = forward
            model.forward = lambda *args, **kwargs: None
            model.forward.__wrapped__ = forward
        return SimpleNamespace(number=number, model=model, inner_model=model)

    @staticmethod
    def unwrap(model):
        if hasattr(model, '_original_forward'):
            model.forward = model._original_forward
            del model._original_forward
        return model

    @staticmethod
    def weight(number):
        import hashlib
        return hashlib.sha256(('actual-weight-'+str(number)).encode()).hexdigest()

    def extract(self, root, start, end, limit, guard):
        self.assertNotEqual([start, end], self.plan['validation'])
        a, b = self.f.stamp(start), self.f.stamp(end)
        rows = {market: [(t, v) for t, v in values if a <= t < b]
                for market, values in self.rows.items()}
        return rows, {'source': 'upbit', 'extracted_sha256': self.f.json_hash(
            {m: [(t.isoformat(), v) for t, v in values] for m, values in rows.items()}),
                      'external_data_used_for_training': False}

    def runtime(self, root, config, guard):
        case = self

        class Runtime:
            np = SimpleNamespace(asarray=lambda x, dtype: list(x), float32=float,
                                 allclose=lambda a, b, **kw: a == b)
            torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
            evidence = {'device': 'cuda:0', 'packages': {'test_runtime': 'fake'}}

            def check_device(self, pipeline):
                guard.check()
                if case.cpu:
                    raise RuntimeError('실제 학습 모델이 CPU입니다')

            def load(self, path):
                if case.fail_load:
                    raise RuntimeError('주입된 모델 로드 실패')
                number = int((Path(path)/'model.safetensors').read_bytes())
                case.loads.append(number)
                if case.bad_reload and number:
                    number += 1
                result = case.pipeline(number)
                self.check_device(result)
                return result

            def weights_hash(self, pipeline):
                return case.weight(pipeline.number)

            def train(self, pipeline, training, tuning, directory, status):
                self.check_device(pipeline)
                case.train_calls.append(pipeline.number)
                for values in training+tuning:
                    case.assertTrue(all(100 <= v < 772 for v in values))
                if case.fail_fit == len(case.train_calls):
                    raise RuntimeError('주입된 fit 중단')
                for step in (1, 17, 50):
                    guard.check()
                    status.update(step=step, loss=1.0/step)
                number = pipeline.number+(0 if case.no_change else 1)
                directory = Path(directory)/'checkpoint'
                directory.mkdir(parents=True)
                case.f.atomic_json(directory/'config.json', {'fake': True})
                (directory/'model.safetensors').write_bytes(str(number).encode())
                return case.pipeline(number, wrapped=True)

            def predict(self, pipeline, contexts):
                if hasattr(pipeline.model, '_original_forward'):
                    raise ValueError('학습 반환 모델의 BF16 래퍼가 남았습니다')
                case.probes.extend(contexts)
                case.assertTrue(all(len(c) == 192 and max(c) < 556 for c in contexts))
                return [[float(pipeline.number)]*12 for _ in contexts]

        return Runtime()

    @contextlib.contextmanager
    def dependencies(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(self.f, 'initial_completion', return_value=self.initial))
            stack.enter_context(patch.object(self.f, 'protected_periods', return_value=[]))
            stack.enter_context(patch.object(self.f, 'extract_rows', side_effect=self.extract))
            stack.enter_context(patch.object(self.f, 'ChronosRuntime', side_effect=self.runtime))
            stack.enter_context(patch.object(self.m, 'unwrap_model_for_inference', side_effect=self.unwrap, create=True))
            if sys.platform != 'linux':
                stack.enter_context(patch.object(self.f, 'exclusive_lock', side_effect=lambda p: contextlib.nullcontext()))
            stack.enter_context(patch.object(self.m.shutil, 'disk_usage', return_value=SimpleNamespace(free=100_000_000_000)))
            original = self.f.atomic_json

            def recording(path, value):
                if Path(path) == self.root/'8차/work/status.json':
                    self.history.append(dict(value))
                return original(path, value)

            stack.enter_context(patch.object(self.f, 'atomic_json', side_effect=recording))
            yield

    def test_frozen_config_rejects_changed_training_budget(self):
        for update in ({'repetitions': 11}, {'repetitions': 9}, {'steps': 500}, {'batch_size': 16},
                       {'context_length': 256}, {'learning_rate': 1e-5}, {'steps': True},
                       {'prediction_length': 24}, {'unexpected': 1}, {'max_seconds': 0}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.m.validate_settings(update)

    def test_ten_separate_fits_use_previous_weights_and_train_only_probes(self):
        seed_before = {str(p.relative_to(self.root)): p.read_bytes()
                       for p in (self.root/'7차/work').rglob('*') if p.is_file()}
        with self.dependencies():
            result = self.m.run_batch(self.root, self.settings, 'ten-fixed')
        self.assertEqual(self.train_calls, list(range(10)))
        self.assertEqual(result['completed_repetitions'], 10)
        self.assertEqual(result['rep_index'], 10)
        self.assertEqual(result['step'], 500)
        self.assertEqual(result['total_steps'], 500)
        self.assertTrue(result['checkpoint_verified'])
        self.assertEqual(result['state'], 'complete')
        self.assertFalse(result['evaluation_performed'])
        self.assertFalse(result['candidate_ready'])
        self.assertFalse(result['promote'])
        self.assertIsNone(result['metrics'])
        self.assertTrue(all(not s['candidate_ready'] and s['metrics'] is None for s in self.history))
        self.assertEqual(sorted({s['completed_repetitions'] for s in self.history}), list(range(11)))
        self.assertEqual({str(p.relative_to(self.root)): p.read_bytes()
                          for p in (self.root/'7차/work').rglob('*')
                          if p.is_file() and p.name != 'runner.lock'}, seed_before)
        aggregate = self.f.read_json(self.root/'8차/work/runs/ten-fixed/completion.json')
        self.assertEqual(aggregate['completed_repetitions'], 10)
        previous = self.f.read_json(self.root/'7차/work/knowledge.json')['manifest_sha256']
        for index in range(1, 11):
            path = self.root/f'8차/work/runs/ten-fixed/rep-{index:02d}/manifest.json'
            manifest = self.f.read_json(path)
            self.assertEqual(manifest['previous_manifest_sha256'], previous)
            self.assertEqual(manifest['execution']['before_weights_sha256'], self.weight(index-1))
            self.assertEqual(manifest['execution']['after_weights_sha256'], self.weight(index))
            self.assertEqual(manifest['execution']['completed_steps'], 50)
            self.assertFalse(manifest['validated'])
            previous = self.f.file_hash(path)
        pointer = self.f.read_json(self.root/'8차/work/knowledge.json')
        self.assertFalse(pointer['validated'])
        self.assertEqual(pointer['manifest_sha256'], previous)

    def test_failure_keeps_committed_count_and_blocks_same_or_new_batch(self):
        self.fail_fit = 3
        with self.dependencies(), self.assertRaisesRegex(RuntimeError, '주입된 fit'):
            self.m.run_batch(self.root, self.settings, 'failed')
        state = self.f.read_json(self.root/'8차/work/status.json')
        self.assertEqual(state['completed_repetitions'], 2)
        self.assertEqual(state['step'], 100)
        self.assertFalse(state['candidate_ready'])
        self.assertEqual(len(list((self.root/'8차/work/runs/failed').glob('rep-*/manifest.json'))), 2)
        self.fail_fit = None
        with self.dependencies():
            for batch in ('failed', 'renamed'):
                with self.subTest(batch=batch), self.assertRaisesRegex(RuntimeError, '재실행'):
                    self.m.run_batch(self.root, self.settings, batch)
        self.assertEqual(self.train_calls, [0, 1, 2])

    def test_complete_batch_is_not_fitted_again(self):
        with self.dependencies():
            self.m.run_batch(self.root, self.settings, 'complete')
            with self.assertRaisesRegex(RuntimeError, '재실행'):
                self.m.run_batch(self.root, self.settings, 'complete')
        self.assertEqual(len(self.train_calls), 10)

    def test_manifest_write_failure_preserves_readable_commit_count_without_refit(self):
        original = self.f.atomic_json

        def injected(path, value):
            original(path, value)
            if Path(path).name == 'manifest.json' and Path(path).parent.name == 'rep-01':
                raise OSError('주입된 manifest 교체 후 디렉터리 동기화 실패')

        with self.dependencies(), patch.object(self.f, 'atomic_json', side_effect=injected):
            with self.assertRaisesRegex(OSError, '디렉터리 동기화'):
                self.m.run_batch(self.root, self.settings, 'commit-error')
        state = self.f.read_json(self.root/'8차/work/status.json')
        self.assertEqual(state['completed_repetitions'], 1)
        self.assertEqual(state['step'], 50)
        self.assertTrue(state['checkpoint_verified'])
        self.assertFalse(state['candidate_ready'])
        self.assertEqual(state['state'], 'error')
        with self.dependencies(), self.assertRaisesRegex(RuntimeError, '재실행'):
            self.m.run_batch(self.root, self.settings, 'retry-error')
        self.assertEqual(self.train_calls, [0])

    def test_manifest_failure_before_replace_does_not_count_an_uncommitted_fit(self):
        original = self.f.atomic_json

        def injected(path, value):
            if Path(path).name == 'manifest.json' and Path(path).parent.name == 'rep-01':
                raise OSError('주입된 manifest 교체 전 실패')
            return original(path, value)

        with self.dependencies(), patch.object(self.f, 'atomic_json', side_effect=injected):
            with self.assertRaisesRegex(OSError, '교체 전 실패'):
                self.m.run_batch(self.root, self.settings, 'precommit-error')
        state = self.f.read_json(self.root/'8차/work/status.json')
        self.assertEqual(state['completed_repetitions'], 0)
        self.assertFalse(state['checkpoint_verified'])
        self.assertEqual(self.train_calls, [0])

    def test_model_load_failure_has_zero_fits_and_only_a_preparation_record(self):
        self.fail_load = True
        with self.dependencies(), self.assertRaisesRegex(RuntimeError, '모델 로드 실패'):
            self.m.run_batch(self.root, self.settings, 'load-error')
        attempt = self.f.read_json(self.root/'8차/work/runs/load-error/rep-01/attempt.json')
        self.assertNotIn('fit_calls', attempt)
        self.assertEqual(attempt['state'], 'prepared')
        self.assertEqual(self.train_calls, [])
        state = self.f.read_json(self.root/'8차/work/status.json')
        self.assertEqual(state['completed_repetitions'], 0)
        self.assertEqual(state['step'], 0)
        self.assertFalse(state['checkpoint_verified'])

    def test_status_cache_failure_after_round_commit_keeps_count_but_not_success(self):
        original = self.f.atomic_json
        injected_once = []

        def injected(path, value):
            if (Path(path) == self.root/'8차/work/status.json'
                    and value['completed_repetitions'] == 1 and not injected_once):
                injected_once.append(True)
                raise OSError('주입된 완료 회차 상태 캐시 실패')
            return original(path, value)

        with self.dependencies(), patch.object(self.f, 'atomic_json', side_effect=injected):
            with self.assertRaisesRegex(OSError, '상태 캐시 실패'):
                self.m.run_batch(self.root, self.settings, 'cache-error')
        state = self.f.read_json(self.root/'8차/work/status.json')
        self.assertEqual(state['state'], 'error')
        self.assertEqual(state['completed_repetitions'], 1)
        self.assertTrue(state['checkpoint_verified'])
        self.assertFalse(state['candidate_ready'])
        completion = self.f.read_json(self.root/'8차/work/runs/cache-error/completion.json')
        self.assertEqual(completion['completed_repetitions'], 1)
        self.assertFalse(completion['complete'])
        self.assertEqual(len(completion['round_manifests']), 1)
        self.assertEqual(self.train_calls, [0])

    def test_persistent_status_failure_preserves_error_aggregate_and_original_failure(self):
        original = self.f.atomic_json

        def injected(path, value):
            if (Path(path) == self.root/'8차/work/status.json'
                    and value['completed_repetitions'] == 1):
                raise OSError('주입된 지속 상태 캐시 실패')
            return original(path, value)

        with self.dependencies(), patch.object(self.f, 'atomic_json', side_effect=injected):
            with self.assertRaisesRegex(OSError, '지속 상태 캐시 실패'):
                self.m.run_batch(self.root, self.settings, 'persistent-cache-error')
        completion = self.f.read_json(self.root/'8차/work/runs/persistent-cache-error/completion.json')
        self.assertEqual(completion['completed_repetitions'], 1)
        self.assertEqual(completion['state'], 'error')
        self.assertFalse(completion['complete'])
        self.assertFalse(completion['candidate_ready'])
        failures = list((self.root/'8차/work/evidence').glob('failure-*.json'))
        self.assertEqual(len(failures), 1)
        self.assertEqual(self.f.read_json(failures[0])['error'], '주입된 지속 상태 캐시 실패')
        self.assertEqual(self.train_calls, [0])

    def test_tampered_seed_is_not_replaced_by_pretrained(self):
        (self.root/'7차/work/runs/seed/checkpoint/model.safetensors').write_bytes(b'9')
        with self.dependencies(), self.assertRaises(ValueError):
            self.m.run_batch(self.root, self.settings, 'bad-seed')
        self.assertEqual(self.train_calls, [])
        self.assertEqual(self.loads, [])

    def test_weight_change_and_reload_hash_are_required(self):
        for name in ('no_change', 'bad_reload', 'cpu'):
            with self.subTest(name=name):
                folder = self.root/'8차'
                if folder.exists():
                    # 테스트 임시 원본을 보존한 채 다른 fixture 루트를 사용합니다.
                    self.tearDown()
                    self.setUp()
                setattr(self, name, True)
                with self.dependencies(), self.assertRaises((ValueError, RuntimeError)):
                    self.m.run_batch(self.root, self.settings, 'bad-model')
                state = self.f.read_json(self.root/'8차/work/status.json')
                self.assertEqual(state['completed_repetitions'], 0)
                self.assertFalse(state['checkpoint_verified'])

    def test_stop_seven_eight_deadline_and_disk_fail_before_fit(self):
        for relative in ('7차/work/STOP', '8차/work/STOP'):
            path = self.root/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            guard = self.m.RepeatGuard(self.root, 10)
            with self.assertRaises(self.f.StopRequested):
                guard.check()
            self.assertTrue(path.is_file())
            path.rename(path.with_suffix('.preserved'))
        with self.assertRaises(TimeoutError):
            self.m.RepeatGuard(self.root, 0).check()
        with self.dependencies(), patch.object(self.m.shutil, 'disk_usage', return_value=SimpleNamespace(free=0)):
            with self.assertRaises(OSError):
                self.m.run_batch(self.root, self.settings, 'no-space')
        self.assertEqual(self.train_calls, [])

    @unittest.skipUnless(sys.platform == 'linux', '실제 Linux flock 경계입니다')
    def test_both_locks_block_before_fit(self):
        for relative in ('7차/work/runner.lock', '8차/work/runner.lock'):
            with self.dependencies(), self.f.exclusive_lock(self.root/relative):
                with self.assertRaises(self.f.AlreadyRunning):
                    self.m.run_batch(self.root, self.settings, 'locked')
        self.assertEqual(self.train_calls, [])

    def test_plan_must_be_ordered_and_validation_not_train_or_tune(self):
        parent = self.f.read_parent(self.root/'7차/work')
        parent['manifest']['plan']['tune'] = self.plan['validation']
        with self.assertRaises(ValueError):
            self.m.prepare_data(self.root, parent, self.settings, self.m.RepeatGuard(self.root, 10))

    @unittest.skipUnless(sys.platform == 'linux', 'SQLite는 Linux 읽기 전용으로만 검사합니다')
    def test_real_extraction_excludes_protected_consumed_and_missing_bars(self):
        path = self.root/'work/data/candles.sqlite3'
        path.parent.mkdir(parents=True)
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE candles(market TEXT,tf TEXT,open_time TEXT,close_time TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL)')
            for market, rows in self.rows.items():
                for moment, value in rows:
                    if moment == self.start+dt.timedelta(minutes=5*226):
                        continue
                    db.execute('INSERT INTO candles VALUES(?,?,?,?,?,?,?,?,?)',
                               (market, '5', (moment-dt.timedelta(minutes=5)).isoformat(), moment.isoformat(),
                                value, value+1, value-1, value, 1.0))
        before = path.read_bytes()
        parent = self.f.read_parent(self.root/'7차/work')
        protected = [('2026-10-01T18:55:00Z', '2026-10-01T19:05:00Z')]
        # 실패 회차 예약도 읽기 검증 후 같은 금지 기간으로 취급합니다.
        reservation_work = self.root/'7차/work'
        data = {'exposure_end': '2026-10-01T19:15:00Z', 'extracted_sha256': 'fixture'}
        self.f.atomic_json(reservation_work/'runs/failed-evaluation/data.json', data)
        self.f.reserve_exposure(reservation_work, 'failed-evaluation',
                                {'validation': ['2026-10-01T19:05:00Z', '2026-10-01T19:15:00Z']}, data, None)
        with patch.object(self.f, 'protected_periods', return_value=protected):
            prepared = self.m.prepare_data(self.root, parent, self.settings, self.m.RepeatGuard(self.root, 20))
        self.assertEqual(path.read_bytes(), before)
        for values in prepared['training']:
            self.assertTrue(all(value < 326 or value > 331 for value in values))
            self.assertTrue(all(b-a == 1 for a, b in zip(values, values[1:])))
        self.assertTrue(all(max(c) < 556 for c in prepared['contexts']))
        self.assertFalse(prepared['data']['external_data_used_for_training'])

    def test_help_has_no_machine_learning_import(self):
        output = io.StringIO()
        with patch.dict(sys.modules, {'torch': None, 'chronos': None, 'transformers': None}), \
                contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            self.m.main(['--help'])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn('--once', output.getvalue())
        self.assertIn('--batch', output.getvalue())
        self.assertIn('--config', output.getvalue())
        self.assertIn('--recover-first', output.getvalue())

    def test_normalization_removes_wrapper_without_changing_parameters(self):
        self.assertTrue(callable(getattr(self.m, 'normalize_inference', None)), '정밀도 정규화가 없습니다')
        runtime = self.runtime(self.root, self.settings, self.m.RepeatGuard(self.root, 20))
        pipeline = self.pipeline(1, wrapped=True)
        before = runtime.weights_hash(pipeline)
        with self.dependencies():
            proof = self.m.normalize_inference(runtime, pipeline)
        self.assertFalse(hasattr(pipeline.model, '_original_forward'))
        self.assertEqual(proof['parameter_sha256'], before)
        self.assertEqual(runtime.weights_hash(pipeline), before)
        self.assertEqual(proof['mode']['dtype'], 'torch.float32')

    def recovery_fixture(self):
        batch = self.root/'8차/work/runs/ten-fixed'
        with self.dependencies():
            parent = self.f.read_parent(self.root/'7차/work')
            prepared = self.m.prepare_data(self.root, parent, self.settings, self.m.RepeatGuard(self.root, 20))
        request = {'run': 'ten-fixed', 'settings': self.settings, 'model_id': self.f.MODEL_ID,
                   'revision': self.f.REVISION, 'source_sha256': self.m.ORIGIN_SOURCE_SHA256,
                   'helper_sha256': self.f.file_hash(Path(self.f.__file__)), 'data': prepared['data'],
                   'parent_run': parent['run'], 'parent_manifest_sha256': parent['manifest_sha256']}
        self.f.atomic_json(self.root/'8차/work/request.json', request)
        self.f.atomic_json(batch/'request.json', request)
        self.f.atomic_json(batch/'inputs.json', prepared['records'])
        self.f.atomic_json(self.root/'8차/work/evidence/source-freeze-repair70.json',
                           {'sha256': {'8차/train_only_repeats.py': self.m.ORIGIN_SOURCE_SHA256}})
        for name in ('checkpoint', 'checkpoint-50'):
            checkpoint = batch/'rep-01'/name
            checkpoint.mkdir(parents=True)
            self.f.atomic_json(checkpoint/'config.json', {'fake': True})
            (checkpoint/'model.safetensors').write_bytes(b'1')
        self.f.atomic_json(batch/'rep-01/checkpoint-50/trainer_state.json', {'global_step': 50})
        self.f.atomic_json(batch/'rep-01/attempt.json', {'run': 'ten-fixed', 'state': 'prepared',
            'rep_index': 1, 'request_sha256': self.f.file_hash(batch/'request.json'),
            'previous_manifest_sha256': parent['manifest_sha256']})
        self.f.atomic_json(self.root/'8차/work/status.json', {'run': 'ten-fixed', 'state': 'error',
            'stage': 'verify', 'step': 50, 'loss': 1.0, 'completed_repetitions': 0, 'device': 'cuda:0'})
        self.f.atomic_json(batch/'completion.json', {'run': 'ten-fixed', 'complete': False,
            'completed_repetitions': 0, 'round_manifests': []})
        failure = self.root/'8차/work/evidence/failure-original.json'
        self.f.atomic_json(failure, {'run': 'ten-fixed', 'stage': 'verify', 'type': 'ValueError',
            'error': '저장 전후 train context 예측 또는 파일 해시가 다릅니다', 'settings': self.settings})
        paths = self.m.original_diagnostic_paths(self.root, batch)
        hashes = {p.relative_to(self.root).as_posix(): self.f.file_hash(p) for p in paths}
        self.f.atomic_json(self.root/'8차/work/evidence/repair71-readonly-reload-diagnostic.log',
            {'fit_calls': 0, 'evaluation_performed': False, 'probe_scope': 'saved_train_context_only',
             'before_files': hashes, 'after_files': hashes, 'all_original_files_unchanged': True,
             'parent_verified': True, 'input_contract_verified': True, 'weights_sha256': self.weight(1)})
        preserved = paths+[batch/'completion.json', failure]
        return batch, {p: p.read_bytes() for p in preserved}

    def test_explicit_recovery_uses_existing_first_fit_then_only_nine_new_fits(self):
        self.assertTrue(callable(getattr(self.m, 'validate_recovery_origin', None)), '명시적 복구가 없습니다')
        batch, originals = self.recovery_fixture()
        with self.dependencies():
            result = self.m.run_batch(self.root, self.settings, 'ten-fixed', recover_first=True)
        self.assertEqual(self.train_calls, list(range(1, 10)))
        self.assertEqual(result['completed_repetitions'], 10)
        self.assertEqual(result['actual_fits_consumed'], 10)
        self.assertFalse(result['evaluation_performed'])
        self.assertFalse(result['promote'])
        for path, content in originals.items():
            if path == self.root/'8차/work/status.json':
                self.assertEqual((batch/'recovery-first/original-status.json').read_bytes(), content)
            else:
                self.assertEqual(path.read_bytes(), content)
        recovered = self.f.read_json(batch/'recovery-first/rep-01/manifest.json')
        self.assertEqual(recovered['verification_method'], 'recovered_after_reload_check')
        self.assertFalse(recovered['original_before_after_probe_verified'])
        self.assertTrue(recovered['original_verification_failed'])
        diagnostic_hash = self.f.file_hash(self.root/'8차/work/evidence/repair71-readonly-reload-diagnostic.log')
        self.assertEqual(recovered['diagnostic_sha256'], diagnostic_hash)
        for name in ('fresh-load-1.json', 'fresh-load-2.json'):
            proof = self.f.read_json(batch/'recovery-first/rep-01'/name)
            self.assertEqual(proof['binding']['diagnostic_sha256'], diagnostic_hash)
        aggregate = self.f.read_json(batch/'recovery-first/completion.json')
        self.assertTrue(aggregate['recovery']['recovered_first'])
        self.assertEqual(len(aggregate['round_manifests']), 10)
        self.assertTrue((batch/'recovery-first/rep-02/probe-before-reload.json').is_file())
        self.assertTrue((batch/'recovery-first/rep-02/probe-after-reload.json').is_file())
        with self.dependencies():
            for recover in (False, True):
                with self.assertRaises(RuntimeError):
                    self.m.run_batch(self.root, self.settings, 'ten-fixed', recover_first=recover)
        self.assertEqual(len(self.train_calls), 9)

    def test_recovery_rejects_changed_original_without_new_fit(self):
        self.assertTrue(callable(getattr(self.m, 'validate_recovery_origin', None)), '명시적 복구가 없습니다')
        batch, _ = self.recovery_fixture()
        self.f.atomic_json(batch/'inputs.json', {'changed': True})
        with self.dependencies(), self.assertRaises(ValueError):
            self.m.run_batch(self.root, self.settings, 'ten-fixed', recover_first=True)
        self.assertEqual(self.train_calls, [])
        self.assertEqual(self.loads, [])

    def test_recovery_failure_never_retrains_first_or_retries_second(self):
        self.assertTrue(callable(getattr(self.m, 'validate_recovery_origin', None)), '명시적 복구가 없습니다')
        batch, _ = self.recovery_fixture()
        self.fail_fit = 1
        with self.dependencies(), self.assertRaisesRegex(RuntimeError, '주입된 fit'):
            self.m.run_batch(self.root, self.settings, 'ten-fixed', recover_first=True)
        state = self.f.read_json(self.root/'8차/work/status.json')
        self.assertEqual(state['completed_repetitions'], 1)
        self.assertEqual(state['actual_fits_consumed'], 1)
        self.assertEqual(self.train_calls, [1])
        with self.dependencies(), self.assertRaises(RuntimeError):
            self.m.run_batch(self.root, self.settings, 'ten-fixed', recover_first=True)

    def test_normalization_rejects_any_parameter_change(self):
        runtime = self.runtime(self.root, self.settings, self.m.RepeatGuard(self.root, 20))
        pipeline = self.pipeline(1, wrapped=True)

        def bad_unwrap(model):
            pipeline.number += 1
            return self.unwrap(model)

        with self.dependencies(), patch.object(self.m, 'unwrap_model_for_inference', side_effect=bad_unwrap):
            with self.assertRaisesRegex(ValueError, '파라미터 해시'):
                self.m.normalize_inference(runtime, pipeline)

    def test_probe_errors_are_separate_and_saved_before_failure(self):
        runtime = self.runtime(self.root, self.settings, self.m.RepeatGuard(self.root, 20))
        first = {'parameter_sha256': 'a', 'mode': {'dtype': 'torch.float32'},
                 'context_sha256': 'context', 'checkpoint_files': {'x': 'a'}, 'predictions': [[1.0]*12]}
        for key, value, field in (('parameter_sha256', 'b', 'parameter_hash_match'),
                                 ('checkpoint_files', {'x': 'b'}, 'checkpoint_files_match'),
                                 ('predictions', [[2.0]*12], 'predictions_match')):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                self.m.compare_probes(runtime, first, {**first, key: value}, self.root/(field+'.json'))
            result = self.f.read_json(self.root/(field+'.json'))
            self.assertFalse(result[field])
            self.assertEqual((result['rtol'], result['atol']), (1e-5, 1e-5))

    def test_actual_training_status_is_accepted_by_project_api(self):
        from dashboard import training_repetition
        self.fail_fit = 1
        with self.dependencies(), self.assertRaisesRegex(RuntimeError, '주입된 fit'):
            self.m.run_batch(self.root, self.settings, 'api-stage-contract')
        training = [value for value in self.history if value['state'] == 'training']
        self.assertEqual(len(training), 1)
        view = training_repetition(training[0])
        self.assertNotEqual(view, {}, '실제 학습 상태가 API에서 사라졌습니다')
        self.assertEqual(view['stage'], 'finetune')
        self.assertEqual(view['rep_index'], 1)
        self.assertEqual(view['total_steps'], 500)


if __name__ == '__main__':
    unittest.main()
