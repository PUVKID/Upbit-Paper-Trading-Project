"""사전학습 연구의 데이터·계보·저장·중단 경계를 검사합니다."""
import contextlib
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / '7차/shadow_research.py'
UTC = dt.timezone.utc


class FoundationResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = None
        if SOURCE.is_file():
            spec = importlib.util.spec_from_file_location('foundation_runner_test', SOURCE)
            cls.module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = cls.module
            spec.loader.exec_module(cls.module)

    def setUp(self):
        self.assertIsNotNone(self.module, '지정 shadow runner가 아직 구현되지 않았습니다')
        self.m = self.module
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def bars(self, count=30, start='2026-10-01T00:05:00+00:00'):
        start = dt.datetime.fromisoformat(start)
        return [(start + dt.timedelta(minutes=5*i), 100.0+i) for i in range(count)]

    def test_config_rejects_unknown_bool_and_unpinned_revision(self):
        for patch in ({'steps': True}, {'steps': 0}, {'unknown': 1},
                      {'revision': 'main'}, {'batch_size': 0}, {'learning_rate': float('nan')}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                self.m.validate_settings(patch)
        cfg = self.m.validate_settings({})
        self.assertEqual(cfg['context_length'], 256)
        self.assertLessEqual(cfg['batch_size'], 16)
        self.assertTrue(30 <= cfg['steps'] <= 100)

    def test_plan_only_uses_new_validation_after_parent_exposure(self):
        cfg = self.m.validate_settings({})
        plan = self.m.make_plan(cfg, dt.datetime(2026, 11, 2, 12, tzinfo=UTC),
                                '2026-11-02T10:00:00+00:00')
        self.assertEqual(plan['validation'][0], '2026-11-02T10:00:00+00:00')
        self.assertEqual(plan['validation'][1], '2026-11-02T12:00:00+00:00')
        self.assertLessEqual(plan['tune'][1], plan['validation'][0])
        with self.assertRaises(self.m.NotReady):
            self.m.make_plan(cfg, dt.datetime(2026, 11, 2, 12, tzinfo=UTC),
                             '2026-11-02T12:00:00+00:00')

    def test_segments_do_not_bridge_missing_bar_or_protected_period(self):
        rows = self.bars(12)
        rows.pop(4)
        segments = self.m.split_segments(rows, rows[0][0], rows[-1][0]+dt.timedelta(minutes=5),
                                         [('2026-10-01T00:40:00Z', '2026-10-01T00:55:00Z')])
        self.assertEqual([len(s) for s in segments], [4, 2, 1])
        for segment in segments:
            for a, b in zip(segment, segment[1:]):
                self.assertEqual((b[0]-a[0]).total_seconds(), 300)

    def test_windows_use_only_past_context_and_future_targets(self):
        windows = self.m.evaluation_windows([self.bars(12)], 4, 2, 3)
        self.assertEqual(len(windows), 3)
        for context, target in windows:
            self.assertEqual(len(context), 4)
            self.assertEqual(len(target), 2)
            self.assertEqual(target[0], context[-1]+1)

    def test_metrics_compare_same_targets_and_never_invent_trades(self):
        metrics = self.m.forecast_metrics([[102.0, 104.0]], [[101.0, 103.0]], [100.0])
        self.assertAlmostEqual(metrics['validation_mae'], 0.01)
        self.assertAlmostEqual(metrics['baseline_mae'], 0.02)
        self.assertEqual(metrics['direction_accuracy'], 1.0)
        self.assertIsNone(metrics['net_return'])
        self.assertIsNone(metrics['trades'])
        self.assertFalse(metrics['promote'])
        with self.assertRaises(ValueError):
            self.m.forecast_metrics([[float('nan')]], [[1.0]], [1.0])

    def test_atomic_json_preserves_previous_complete_file_on_bad_value(self):
        path = self.root/'state.json'
        self.m.atomic_json(path, {'step': 3})
        with self.assertRaises(ValueError):
            self.m.atomic_json(path, {'step': float('nan')})
        self.assertEqual(json.loads(path.read_text()), {'step': 3})

    def test_status_only_marks_candidate_after_verified_completion(self):
        status = self.m.Status(self.root/'status.json', self.m.validate_settings({}), 'test-run')
        status.update(stage='finetune', state='training', step=2, loss=0.2)
        value = json.loads(status.path.read_text())
        self.assertFalse(value['candidate_ready'])
        self.assertFalse(value['checkpoint_verified'])
        self.assertEqual(value['step'], 2)
        self.assertIsNone(value['net_return'])
        with self.assertRaises(ValueError):
            status.update(candidate_ready=True)

    @unittest.skipUnless(sys.platform == 'linux', 'Linux flock 경계입니다')
    def test_duplicate_lock_rejects_without_deleting_lock(self):
        path = self.root/'runner.lock'
        with self.m.exclusive_lock(path):
            with self.assertRaises(self.m.AlreadyRunning):
                with self.m.exclusive_lock(path):
                    self.fail('중복 연구가 허용됐습니다')
        self.assertTrue(path.is_file())
        with self.m.exclusive_lock(path):
            pass

    def test_stop_and_deadline_are_explicit(self):
        stop = self.root/'STOP'
        guard = self.m.Guard(stop, 10)
        guard.check()
        stop.touch()
        with self.assertRaises(self.m.StopRequested):
            guard.check()
        self.assertTrue(stop.exists())
        with self.assertRaises(TimeoutError):
            self.m.Guard(self.root/'absent', 0).check()

    @unittest.skipUnless(sys.platform == 'linux', '새 링크를 만들지 않고 기존 OS 링크만 검사합니다')
    def test_path_guard_rejects_escape_and_existing_symlink(self):
        with self.assertRaises(ValueError):
            self.m.safe_path(self.root, '../escape')
        with self.assertRaises(ValueError):
            self.m.assert_no_links(Path('/proc/self'))

    @unittest.skipUnless(sys.platform == 'linux', '운영 자료와 동일한 Linux 읽기 전용 경계입니다')
    def test_sqlite_extraction_is_read_only_and_rejects_incomplete_or_bad_ohlc(self):
        dbpath = self.root/'work/data/candles.sqlite3'
        dbpath.parent.mkdir(parents=True)
        with sqlite3.connect(dbpath) as db:
            db.execute('CREATE TABLE candles(market TEXT,tf TEXT,open_time TEXT,close_time TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL)')
            for market in self.m.MARKETS:
                for stamp, close in self.bars(6):
                    db.execute('INSERT INTO candles VALUES(?,?,?,?,?,?,?,?,?)',
                               (market,'5',(stamp-dt.timedelta(minutes=5)).isoformat(),stamp.isoformat(),
                                close,close+1,close-1,close,1.0))
        before = hashlib.sha256(dbpath.read_bytes()).hexdigest()
        guard = self.m.Guard(self.root/'STOP', 20)
        rows, evidence = self.m.extract_rows(self.root, '2026-10-01T00:00:00Z',
                                            '2026-10-01T01:00:00Z', 50, guard)
        self.assertEqual([len(rows[m]) for m in self.m.MARKETS], [6]*5)
        self.assertEqual(evidence['rows'], 30)
        self.assertEqual(hashlib.sha256(dbpath.read_bytes()).hexdigest(), before)
        with sqlite3.connect(dbpath) as db:
            db.execute("UPDATE candles SET high=1 WHERE market='KRW-BTC'")
        with self.assertRaises(ValueError):
            self.m.extract_rows(self.root, '2026-10-01T00:00:00Z', '2026-10-01T01:00:00Z', 50, guard)

    def test_checkpoint_tamper_and_missing_head_never_fall_back(self):
        work = self.root/'7차/work'
        run = work/'runs/run-one'
        checkpoint = run/'checkpoint'
        checkpoint.mkdir(parents=True)
        (checkpoint/'config.json').write_text('{}')
        (checkpoint/'model.safetensors').write_bytes(b'original-weights')
        manifest = self.m.make_manifest('run-one', None, {}, self.m.validate_settings({}),
                                        {'exposure_end': '2026-10-02T00:00:00Z'},
                                        {'validation_mae': 0.1}, checkpoint,
                                        {'device':'cuda:0','weight_changed':True})
        self.m.atomic_json(run/'manifest.json', manifest)
        with self.assertRaises(ValueError):
            self.m.read_parent(work)
        self.m.atomic_json(work/'knowledge.json', {'run':'run-one',
                            'manifest_sha256':self.m.file_hash(run/'manifest.json')})
        parent = self.m.read_parent(work)
        self.assertEqual(parent['run'], 'run-one')
        (checkpoint/'model.safetensors').write_bytes(b'changed-weights')
        with self.assertRaises(ValueError):
            self.m.read_parent(work)
        self.assertTrue((work/'knowledge.json').exists())

    def test_incomplete_manifest_cannot_become_parent(self):
        work = self.root/'7차/work'
        run = work/'runs/run-one'
        run.mkdir(parents=True)
        self.m.atomic_json(run/'manifest.json', {'run':'run-one','complete':False})
        self.m.atomic_json(work/'knowledge.json', {'run':'run-one',
                            'manifest_sha256':self.m.file_hash(run/'manifest.json')})
        with self.assertRaises(ValueError):
            self.m.read_parent(work)

    def test_pretrained_requires_pinned_regular_files_and_official_hash(self):
        p = self.root/self.m.REVISION
        p.mkdir()
        (p/'config.json').write_text('{}')
        (p/'model.safetensors').write_bytes(b'not-the-official-model')
        with self.assertRaises(ValueError):
            self.m.validate_pretrained(p)

    def test_help_needs_no_ml_imports(self):
        with subprocess.Popen([sys.executable,'-B','-X','utf8',str(SOURCE),'--help'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8') as process:
            stdout, stderr = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 0, stderr)
        self.assertIn('--once', stdout)
        self.assertIn('--config', stdout)

    def bootstrap_fixture(self):
        control = self.root/'4차/work/research_control.json'
        run = 'bootstrap-run'
        batch = {'id':'bootstrap','kind':'bootstrap','baseline_bytes':0,'target_bytes':50000000000,
                 'status':'completed','attempts':1,'run':run,'completed_at':'2026-11-02T01:00:00Z'}
        binding = {'schema_version':1,'batch_id':'bootstrap','claim':'original-claim','attempt':1,
                   'target_bytes':50000000000,'run':run}
        resolution = {'training_ready':True,'ingested_archive_bytes':50236409654,'target_archive_bytes':50000000000}
        report = {'run':run,'paper_only':True,'research':{'status':'complete'},'research_batch':binding,
                  'external_resolution':resolution,'training':{'total_candidates':5}}
        directory = self.root/'work/models/versions'/run
        directory.mkdir(parents=True)
        packages, hashes = {}, {}
        for market in self.m.MARKETS:
            name = market+'.ubj'
            (directory/name).write_bytes(b'fixture-model')
            packages[market] = {'models':[name]}
            hashes[name] = self.m.file_hash(directory/name)
        manifest = {'schema_version':2,'execution_policy':'horizon_or_risk_v2','paper_only':True,
                    'run':run,'research_batch':binding,'packages':packages}
        record = {**binding,'phase':'complete','report':report,'report_sha256':self.m.json_hash(report),
                  'manifest_sha256':self.m.json_hash(manifest),'model_sha256':hashes,
                  'models_dir':'work/models','reports_dir':'work/reports'}
        self.m.atomic_json(directory/'manifest.json',manifest)
        self.m.atomic_json(control,{'schema_version':1,'batches':[batch]})
        self.m.atomic_json(self.root/'4차/work/research_control_attempts/bootstrap-1.json',record)
        self.m.atomic_json(self.root/'work/reports/experiments'/f'{run}.json',report)
        return {'schema_version':1,'training_ready':False,'ingested_archive_bytes':52250140144,
                'verified_archive_bytes':53175730977,'target_archive_bytes':54250140144}

    def test_completed_bootstrap_is_not_blocked_by_next_increment_collection(self):
        readiness = self.bootstrap_fixture()
        result = self.m.initial_completion(self.root, readiness, self.m.Guard(self.root/'STOP',20))
        self.assertEqual(result['bootstrap_run'],'bootstrap-run')
        self.assertFalse(result['current_increment_training_ready'])
        self.assertEqual(result['reported_ingested_bytes'],52250140144)

    def test_bootstrap_gate_rejects_underfilled_or_tampered_completion(self):
        readiness = self.bootstrap_fixture()
        for wrong in (True, 49999999999):
            with self.subTest(wrong=wrong), self.assertRaises((ValueError,self.m.NotReady)):
                self.m.initial_completion(self.root,{**readiness,'ingested_archive_bytes':wrong},
                                          self.m.Guard(self.root/'STOP',20))
        path = self.root/'work/models/versions/bootstrap-run/KRW-BTC.ubj'
        path.write_bytes(b'tampered')
        with self.assertRaises(ValueError):
            self.m.initial_completion(self.root,readiness,self.m.Guard(self.root/'STOP',20))

    def test_actual_fit_clone_is_checked_before_any_cpu_step(self):
        runtime = object.__new__(self.m.ChronosRuntime)
        runtime.config = self.m.validate_settings({})
        runtime.guard = self.m.Guard(self.root/'STOP',20)
        runtime.callback_type = object
        steps = []

        class Model:
            def __init__(self, device):
                self.device = device

            def parameters(self):
                return [SimpleNamespace(device=self.device)]

        class Pipeline:
            model = Model('cuda:0')

            def fit(self, **kwargs):
                kwargs['callbacks'][0].on_train_begin(None,None,None,model=Model('cpu'))
                steps.append('학습 step')
                return self

        with self.assertRaises(RuntimeError):
            runtime.train(Pipeline(), [], [], self.root, None)
        self.assertEqual(steps, [])

    def test_failed_evaluation_reservation_prevents_reuse(self):
        work = self.root/'7차/work'
        directory = work/'runs/failed-run'
        directory.mkdir(parents=True)
        plan = {'validation':['2026-10-01T00:00:00Z','2026-10-02T00:00:00Z']}
        data = {'exposure_end':'2026-10-02T00:00:00Z','extracted_sha256':'a'*64}
        self.m.atomic_json(directory/'data.json',data)
        receipt = self.m.reserve_exposure(work,'failed-run',plan,data,None)
        # 평가 뒤 저장이 실패해 완료 manifest와 knowledge 포인터가 없는 상황입니다.
        self.m.atomic_json(directory/'failure.json',{'exposure_reservation_sha256':receipt})
        end = self.m.read_exposure_end(work)
        self.assertEqual(end,'2026-10-02T00:00:00+00:00')
        with self.assertRaises(self.m.NotReady):
            self.m.make_plan(self.m.validate_settings({}),dt.datetime(2026,10,2,tzinfo=UTC),end)
        self.assertFalse((work/'knowledge.json').exists())

    def test_corrupt_or_missing_failed_reservation_is_not_ignored(self):
        work = self.root/'7차/work'
        directory = work/'runs/failed-run'
        directory.mkdir(parents=True)
        plan = {'validation':['2026-10-01T00:00:00Z','2026-10-02T00:00:00Z']}
        data = {'exposure_end':'2026-10-02T00:00:00Z','extracted_sha256':'a'*64}
        self.m.atomic_json(directory/'data.json',data)
        digest = self.m.reserve_exposure(work,'failed-run',plan,data,None)
        self.m.atomic_json(directory/'failure.json',{'exposure_reservation_sha256':digest})
        self.m.atomic_json(directory/'data.json',{**data,'extracted_sha256':'b'*64})
        with self.assertRaises(ValueError):
            self.m.read_exposure_end(work)
        self.m.atomic_json(directory/'data.json',data)
        # 원본을 삭제하지 않고 예약을 다른 이름으로 옮겨 누락을 주입합니다.
        reservation = work/'exposures/failed-run.json'
        reservation.rename(reservation.with_suffix('.preserved'))
        with self.assertRaises(ValueError):
            self.m.read_exposure_end(work)

    @contextlib.contextmanager
    def commit_fixture(self):
        m, root = self.m, self.root
        cfg = m.validate_settings({'steps':1,'context_length':16,'prediction_length':2})
        clock = dt.datetime.now(UTC)
        active = {'plan':m.make_plan(cfg,clock-dt.timedelta(days=5))}
        calls = {'train':0,'predict':0,'runtime':0}
        m.atomic_json(root/'3차/work/status.json',{})
        m.atomic_json(root/'7차/foundation-config.json',cfg)

        def extract(*args):
            rows = {}
            for market in m.MARKETS:
                rows[market] = [(m.stamp(a)+m.BAR*i,100.0+i)
                                for a,b in active['plan'].values() for i in range(30)]
            return rows, {'extracted_sha256':'a'*64}

        class Runtime:
            def __init__(self,*args):
                calls['runtime'] += 1
                self.np = SimpleNamespace(asarray=lambda x,dtype:x,float32=float,
                                          allclose=lambda a,b,**kw:a==b)
                self.torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None))
                self.evidence = {'device':'cuda:0'}

            def load(self,path):
                path = Path(path)/'model.safetensors'
                generation = int(path.read_bytes().decode().split(':')[-1]) if path.exists() else 0
                return SimpleNamespace(generation=generation)

            def weights_hash(self,pipeline):
                return hashlib.sha256(str(pipeline.generation).encode()).hexdigest()

            def train(self,pipeline,training,tuning,directory,status):
                calls['train'] += 1
                checkpoint = directory/'checkpoint'
                checkpoint.mkdir()
                (checkpoint/'config.json').write_text('{}',encoding='utf-8')
                generation = pipeline.generation+1
                (checkpoint/'model.safetensors').write_bytes(('controlled-weights:'+str(generation)).encode())
                status.update(step=1,loss=1.0)
                return SimpleNamespace(generation=generation)

            def check_device(self,pipeline):
                pass

            def predict(self,pipeline,contexts):
                calls['predict'] += 1
                return [[float(c[-1])]*2 for c in contexts]

        with contextlib.ExitStack() as stack:
            replacements = {'exclusive_lock':lambda p:contextlib.nullcontext(),
                            'protected_periods':lambda *a:[], 'initial_completion':lambda *a:{},
                            'extract_rows':extract,'validate_pretrained':lambda *a:None,
                            'ChronosRuntime':Runtime,'make_plan':lambda *a:active['plan']}
            for key,value in replacements.items():
                stack.enter_context(patch.object(m,key,value))
            yield cfg, calls, active, clock

    def test_single_postcommit_status_failure_recovers_without_refit(self):
        with self.commit_fixture() as (cfg,calls,active,clock):
            original, errors = self.m.atomic_json, []

            def write(path,value):
                if Path(path)==self.root/'7차/work/status.json' and value.get('state')=='complete' and not errors:
                    self.assertTrue((self.root/'7차/work/knowledge.json').exists())
                    errors.append('단발 표시 저장 실패')
                    raise OSError(errors[-1])
                return original(path,value)

            with patch.object(self.m,'atomic_json',write):
                result = self.m.run_once(self.root,cfg)
            parent = self.m.read_parent(self.root/'7차/work')
            status = self.m.read_json(self.root/'7차/work/status.json')
            self.assertEqual(status['state'],'complete')
            self.assertTrue(status['checkpoint_verified'])
            self.assertEqual(status['run'],parent['run'])
            self.assertFalse(result['has_status_warning'])
            self.assertEqual(calls['train'],1)
            self.assertEqual(calls['predict'],3)
            logs = list((self.root/'7차/work/runs'/result['run']/'finalization-errors').glob('*.json'))
            self.assertEqual(len(logs),1)
            self.assertIn('단발 표시 저장 실패',self.m.read_json(logs[0])['error'])

    def test_persistent_postcommit_failure_returns_committed_complete_and_cli_zero(self):
        with self.commit_fixture() as (cfg,calls,active,clock):
            original = self.m.atomic_json

            def write(path,value):
                if Path(path)==self.root/'7차/work/status.json' and value.get('state')=='complete':
                    raise OSError('지속 표시 저장 실패')
                return original(path,value)

            with patch.object(self.m,'atomic_json',write), contextlib.redirect_stderr(__import__('io').StringIO()):
                result = self.m.run_once(self.root,cfg)
                with patch.object(self.m,'ROOT',self.root), contextlib.redirect_stdout(__import__('io').StringIO()):
                    self.assertEqual(self.m.main(['--once']),0)
            parent = self.m.read_parent(self.root/'7차/work')
            self.assertTrue(result['has_status_warning'])
            self.assertEqual(result['state'],'complete')
            self.assertTrue(result['candidate_ready'])
            self.assertTrue(result['checkpoint_verified'])
            self.assertEqual(parent['completion']['run'],result['run'])
            self.assertNotEqual(self.m.read_json(self.root/'7차/work/status.json')['state'],'error')
            self.assertEqual(calls['train'],1)
            self.assertEqual(calls['predict'],3)
            self.assertEqual(len(list((self.root/'7차/work/runs').iterdir())),1)

    def test_restart_only_finalizes_pending_completion_and_never_predicts_again(self):
        with self.commit_fixture() as (cfg,calls,active,clock):
            original = self.m.atomic_json

            def write(path,value):
                if Path(path)==self.root/'7차/work/status.json' and value.get('state')=='complete':
                    raise OSError('재시작 전 표시 실패')
                return original(path,value)

            with patch.object(self.m,'atomic_json',write), contextlib.redirect_stderr(__import__('io').StringIO()):
                first = self.m.run_once(self.root,cfg)
            with patch.object(self.m,'ChronosRuntime',side_effect=AssertionError('복구 중 모델을 다시 로드했습니다')):
                second = self.m.run_once(self.root,cfg)
            self.assertEqual(first['run'],second['run'])
            self.assertFalse(second['has_status_warning'])
            self.assertEqual(self.m.read_json(self.root/'7차/work/status.json')['state'],'complete')
            self.assertEqual(calls['predict'],3)
            self.assertEqual(calls['train'],1)

    def test_precommit_failure_keeps_previous_parent_and_fails(self):
        with self.commit_fixture() as (cfg,calls,active,clock):
            first = self.m.run_once(self.root,cfg)
            work = self.root/'7차/work'
            before = (work/'knowledge.json').read_bytes()
            # 새 미래 구간을 명시하며 이전 평가 기간을 재사용하지 않습니다.
            end = clock.replace(minute=clock.minute//5*5,second=0,microsecond=0)
            active['plan'] = {'train':[(end-dt.timedelta(days=34)).isoformat(),(end-dt.timedelta(days=4)).isoformat()],
                              'tune':[(end-dt.timedelta(days=4)).isoformat(),(end-dt.timedelta(days=2)).isoformat()],
                              'validation':[(end-dt.timedelta(days=2)).isoformat(),end.isoformat()]}
            original = self.m.atomic_json

            def write(path,value):
                if Path(path)==work/'knowledge.json':
                    raise OSError('포인터 커밋 이전 실패')
                return original(path,value)

            with patch.object(self.m,'atomic_json',write), self.assertRaises(OSError):
                self.m.run_once(self.root,cfg)
            self.assertEqual((work/'knowledge.json').read_bytes(),before)
            self.assertEqual(self.m.read_parent(work)['run'],first['run'])
            self.assertEqual(self.m.read_json(work/'status.json')['state'],'error')

    def test_pointer_write_error_after_replace_is_recognized_as_committed(self):
        with self.commit_fixture() as (cfg,calls,active,clock):
            original, injected = self.m.atomic_json, []

            def write(path,value):
                original(path,value)
                if Path(path)==self.root/'7차/work/knowledge.json' and not injected:
                    injected.append(True)
                    raise OSError('포인터 교체 뒤 디렉터리 동기화 실패')

            with patch.object(self.m,'atomic_json',write):
                result = self.m.run_once(self.root,cfg)
            self.assertEqual(result['state'],'complete')
            self.assertTrue(result['checkpoint_verified'])
            self.assertEqual(self.m.read_parent(self.root/'7차/work')['run'],result['run'])
            self.assertEqual(calls['predict'],3)

    def test_new_completion_payload_tamper_is_rejected_without_legacy_migration(self):
        with self.commit_fixture() as (cfg,calls,active,clock):
            result = self.m.run_once(self.root,cfg)
            work = self.root/'7차/work'
            pointer = self.m.read_json(work/'knowledge.json')
            manifest = self.m.read_json(work/'runs'/result['run']/'manifest.json')
            self.assertEqual(pointer['completion_sha256'],manifest['completion_sha256'])
            path = work/'runs'/result['run']/'completion.json'
            snapshot = self.m.read_json(path)
            self.m.atomic_json(path,{**snapshot,'checkpoint_verified':False})
            with self.assertRaises(ValueError):
                self.m.read_parent(work)

    def test_persistent_run_cache_failure_marks_public_complete_with_warning(self):
        with self.commit_fixture() as (cfg,calls,active,clock):
            original = self.m.atomic_json

            def write(path,value):
                if Path(path).name=='status.json' and Path(path).parent.parent.name=='runs':
                    raise OSError('회차 표시 캐시 지속 실패')
                return original(path,value)

            with patch.object(self.m,'atomic_json',write), contextlib.redirect_stderr(__import__('io').StringIO()):
                result = self.m.run_once(self.root,cfg)
            cached = self.m.read_json(self.root/'7차/work/status.json')
            self.assertTrue(result['has_status_warning'])
            self.assertTrue(cached['has_status_warning'])
            self.assertEqual(cached['state'],'complete')
            self.assertTrue(cached['checkpoint_verified'])

    def test_unreadable_commit_proof_is_abnormal_and_not_logged_as_confirmed(self):
        with self.commit_fixture() as (cfg,calls,active,clock):
            with patch.object(self.m,'read_completion',side_effect=OSError('완료 증거 판독 실패')):
                with self.assertRaises(RuntimeError):
                    self.m.run_once(self.root,cfg)
            run = next((self.root/'7차/work/runs').iterdir())
            logs = list((run/'finalization-errors').glob('*.json'))
            self.assertEqual(len(logs),1)
            self.assertIsNone(self.m.read_json(logs[0])['committed'])

    def test_normal_finalize_allows_next_run_instead_of_recovering_forever(self):
        with self.commit_fixture() as (cfg,calls,active,clock):
            result = self.m.run_once(self.root,cfg)
            work = self.root/'7차/work'
            pointer = self.m.read_json(work/'knowledge.json')
            for path in (work/'status.json',work/'runs'/result['run']/'status.json'):
                self.assertEqual(self.m.file_hash(path),pointer['completion_sha256'])
            self.assertIsNone(self.m.resume_finalization(self.root,work,self.m.Guard(work/'STOP',20)))
            self.assertEqual(calls['train'],1)
            self.assertEqual(calls['predict'],3)

    def test_finalization_does_not_overwrite_other_run_waiting_or_error(self):
        with self.commit_fixture() as (cfg,calls,active,clock):
            result = self.m.run_once(self.root,cfg)
            work = self.root/'7차/work'
            for state in ('waiting','error','training'):
                other = {**result,'run':'different-newer-run','state':state,
                         'candidate_ready':False,'checkpoint_verified':False}
                self.m.atomic_json(work/'status.json',other)
                before = (work/'status.json').read_bytes()
                self.assertIsNone(self.m.resume_finalization(self.root,work,self.m.Guard(work/'STOP',20)))
                self.assertEqual((work/'status.json').read_bytes(),before)
            self.assertEqual(calls['train'],1)
            self.assertEqual(calls['predict'],3)

    def test_missing_or_corrupt_cache_recovers_only_from_verified_completion(self):
        with self.commit_fixture() as (cfg,calls,active,clock):
            result = self.m.run_once(self.root,cfg)
            work = self.root/'7차/work'
            path = work/'status.json'
            path.rename(work/'status-preserved.json')
            restored = self.m.resume_finalization(self.root,work,self.m.Guard(work/'STOP',20))
            self.assertEqual(restored['run'],result['run'])
            path.write_text('{invalid-json',encoding='utf-8')
            restored = self.m.resume_finalization(self.root,work,self.m.Guard(work/'STOP',20))
            self.assertEqual(restored['state'],'complete')
            self.assertEqual(calls['train'],1)
            self.assertEqual(calls['predict'],3)


if __name__ == '__main__':
    unittest.main()
