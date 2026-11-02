"""직접 순수익 연구의 순수 mock·분할·정책·CUDA 경계를 검사합니다."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import engine
import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parents[1]/'10차/return_research.py'


class FakeBooster:
    def __init__(self, params=None, cache=None):
        self.params, self.rounds, self.best_iteration, self.best_score = dict(params or {}), 0, 0, 0.1
        self.value = 0.001

    def save_config(self):
        return json.dumps({'learner': {'generic_param': {'device': self.params.get('device', 'cpu')},
                         'learner_model_param': {'base_score': str(self.value)}}})

    def set_param(self, key, value=None):
        self.params.update(key if isinstance(key, dict) else {key: value})

    def get_dump(self, **kwargs):
        return [str(self.value)]*self.rounds

    def num_boosted_rounds(self):
        return self.rounds

    def predict(self, matrix, **kwargs):
        return np.full(len(matrix.values), self.value, dtype=np.float32)

    def save_model(self, path):
        Path(path).write_text(json.dumps({'params': self.params, 'rounds': self.rounds, 'value': self.value}), encoding='utf-8')

    def load_model(self, path):
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        self.params, self.rounds, self.value = value['params'], value['rounds'], value['value']


class FakeXGB:
    __version__ = 'mock-no-fit'
    callback = SimpleNamespace(TrainingCallback=object)

    def __init__(self):
        self.calls, self.updates = [], 0
        self.preflight_cpu = self.training_cpu = self.iteration_cpu = False

    @staticmethod
    def build_info():
        return {'USE_CUDA': True}

    @staticmethod
    def DMatrix(values, **kwargs):
        return SimpleNamespace(values=np.asarray(values), kwargs=kwargs)

    def Booster(self, params=None, cache=None):
        model = FakeBooster(params, cache)
        if self.preflight_cpu and params is not None:
            model.params['device'] = 'cpu'
        return model

    def train(self, params, dtrain, **kwargs):
        self.calls.append(copy.deepcopy(params))
        model = FakeBooster(params)
        model.value = float(np.mean(dtrain.kwargs['label']))
        if params.get('quantile_alpha') == 0.1:
            model.value -= 0.001
        elif params.get('quantile_alpha') == 0.9:
            model.value += 0.001
        if self.training_cpu:
            model.params['device'] = 'cpu'
        callbacks = kwargs['callbacks']
        for callback in callbacks:
            if hasattr(callback, 'before_training'):
                callback.before_training(model)
        for epoch in range(2):
            if self.iteration_cpu:
                model.params['device'] = 'cpu'
            for callback in callbacks:
                if hasattr(callback, 'before_iteration'):
                    callback.before_iteration(model, epoch, {})
            self.updates += 1
            model.rounds = epoch+1
            for callback in callbacks:
                if hasattr(callback, 'after_iteration'):
                    callback.after_iteration(model, epoch, {'tune': {'mock_loss': [0.1]}})
        kwargs['evals_result']['tune'] = {'mock_loss': [0.1, 0.1]}
        model.best_iteration = 1
        return model


class ReturnResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = None
        if SOURCE.is_file():
            spec = importlib.util.spec_from_file_location('return_research_test', SOURCE)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def setUp(self):
        self.assertIsNotNone(self.module, '10차 순수익 연구 실행기가 아직 없습니다')
        self.m = self.module
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.m.validate_settings({})
        self.status = self.m.Status(self.root/'status.json', 'mock-run', self.config)
        self.guard = self.m.Guard(self.root, 100)
        self.lib = FakeXGB()

    def frame(self, count=12, start='2026-01-01T00:05:00Z'):
        index = pd.date_range(start, periods=count, freq='5min')
        return pd.DataFrame({'x': np.arange(count, dtype=float), 'open': 100.0,
            'close': 101.0, 'high': 102.0, 'low': 99.0, 'volume': 1.0,
            'open_time': index-pd.Timedelta(minutes=5), '5_volatility': 0.01,
            'net_forward_return': 0.002, 'label': 2,
            'label_end': index+pd.Timedelta(minutes=60)}, index=index)

    def test_budget_and_utc_config_are_bounded(self):
        for changed in ({'folds': 3}, {'rounds': 151}, {'max_train_rows': 100001},
                        {'train_days': 91}, {'rounds': True}, {'anchor': '2026-11-03'},
                        {'max_drawdown': 0.11}, {'unknown': 1}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.m.validate_settings(changed)
        self.assertEqual(self.config['folds']*15, 30)

    def test_rolling_plan_has_90_2_2_2_days_and_no_future_anchor(self):
        plans = self.m.fold_plans(engine, self.config, engine.utc('2026-11-04T00:00:00Z'))
        self.assertEqual(len(plans), 2)
        for plan in plans:
            for key, days in (('train', 90), ('tune', 2), ('rule', 2), ('validation', 2)):
                self.assertEqual((plan[key][1]-plan[key][0]).total_seconds(), days*86400)
            self.assertEqual(plan['train'][1], plan['tune'][0])
            self.assertEqual(plan['tune'][1], plan['rule'][0])
            self.assertEqual(plan['rule'][1], plan['validation'][0])
        with self.assertRaises(ValueError):
            self.m.fold_plans(engine, self.config, engine.utc('2020-01-01T00:00:00Z'))

    def test_frozen_anchor_precedes_foundation_evaluation_without_weakening_guard(self):
        config = self.m.validate_settings(json.loads(SOURCE.with_name('return-research-config.json').read_text(encoding='utf-8')))
        self.assertEqual(config['anchor'], '2026-10-31T08:00:00Z')
        plans = self.m.fold_plans(engine, config, engine.utc('2026-11-02T18:13:06Z'))
        forbidden = [(engine.utc('2026-10-19T00:00:00Z'),engine.utc('2026-10-21T00:00:00Z')),
                     (engine.utc('2026-10-31T09:00:00Z'),engine.utc('2026-11-02T09:00:00Z'))]
        for plan in plans:
            for key in ('rule','validation'):
                self.assertFalse(any(plan[key][0] < b and plan[key][1] > a for a,b in forbidden))

    def test_target_is_exact_engine_next_open_net_return_not_label_threshold(self):
        frame = self.frame(20)
        frame['open'] = 100.0
        frame['close'] = 103.0
        labelled = engine.label_frame(frame, 2, 0.5, 0.001, 0.002)
        wanted = 103*(1-.002)*(1-.001)/(100*(1+.002)*(1+.001))-1
        self.assertAlmostEqual(labelled.net_forward_return.iloc[0], wanted)
        self.assertEqual(labelled.label.iloc[0], 1)
        model, proof = self.m.fit_return_model(engine, self.lib, labelled.dropna(), labelled.dropna(),
            ['x'], 'mean', self.config, self.status, self.guard)
        self.assertAlmostEqual(model.value, wanted)
        self.assertEqual(proof['parameters']['objective'], 'reg:squarederror')
        self.assertEqual(self.status.value['fit_calls'], 1)

    def test_cuda_preflight_cpu_refuses_before_train_call(self):
        self.lib.preflight_cpu = True
        with self.assertRaises(RuntimeError):
            self.m.fit_return_model(engine, self.lib, self.frame(), self.frame(), ['x'],
                                    'mean', self.config, self.status, self.guard)
        self.assertEqual(self.lib.calls, [])
        self.assertEqual(self.lib.updates, 0)
        self.assertEqual(self.status.value['fit_calls'], 0)

    def test_real_training_cpu_is_blocked_before_first_update(self):
        for field in ('training_cpu', 'iteration_cpu'):
            with self.subTest(field=field):
                lib = FakeXGB()
                setattr(lib, field, True)
                with self.assertRaises(RuntimeError):
                    self.m.fit_return_model(engine, lib, self.frame(), self.frame(), ['x'],
                                            'q10', self.config, self.status, self.guard)
                self.assertEqual(len(lib.calls), 1)
                self.assertEqual(lib.updates, 0)

    def test_three_objectives_and_reload_proof_are_explicit(self):
        for name, objective, quantile in (('mean', 'reg:squarederror', None),
                ('q10', 'reg:quantileerror', .1), ('q90', 'reg:quantileerror', .9)):
            model, evidence = self.m.fit_return_model(engine, self.lib, self.frame(), self.frame(),
                ['x'], name, self.config, self.status, self.guard)
            self.assertEqual(evidence['parameters']['objective'], objective)
            self.assertEqual(evidence['parameters'].get('quantile_alpha'), quantile)
            proof = self.m.save_verified_model(engine, self.lib, model, evidence,
                self.frame(), ['x'], self.root/(name+'.ubj'), self.guard)
            self.assertTrue(proof['reload_verified'])
        self.assertEqual(len(self.lib.calls), 3)
        self.assertEqual(self.status.value['completed_fits'], 3)

    def test_uncertainty_and_stress_never_double_subtract_base_cost(self):
        values = pd.DataFrame({'mean': [.003, -.001, .003, .003],
                              'q10': [.001, -.003, .004, -.01], 'q90': [.005, .001, .002, .01]})
        plain = self.m.return_signals(values, {'family': 'plain', 'minimum': 0, 'penalty': 0}, 1)
        margin = self.m.return_signals(values, {'family': 'uncertainty', 'minimum': 0, 'penalty': .5}, 1)
        self.assertEqual(plain.tolist(), [2, 1, 1, 2])
        self.assertEqual(margin.tolist(), [2, 1, 1, 1])
        self.assertEqual(self.m.return_signals(values.iloc[:1], {'family':'plain','minimum':.0029,'penalty':0},1).tolist(), [2])
        self.assertEqual(self.m.return_signals(values.iloc[:1], {'family':'plain','minimum':.0029,'penalty':0},.999).tolist(), [1])

    def test_policy_selection_only_receives_policy_rows_and_can_choose_cash(self):
        def score(policy):
            return {'net_return': 0 if policy['family'] == 'cash' else -.01,
                    'MDD': 0.0, 'trades': 0 if policy['family'] == 'cash' else 5}
        selected, trials = self.m.select_policies(self.config, score)
        self.assertEqual(selected['selected']['family'], 'cash')
        self.assertTrue(trials)
        self.assertEqual(selected['selection_scope'], 'rule_only')

    def test_review_gate_requires_two_folds_twenty_trades_and_positive_stress(self):
        rows = [{'fold':fold,'candidate':'selected','cost_multiplier':cost,'net_return':.01,
                 'MDD':.04,'trades':10,'scope':'research_validation'} for fold in (1,2) for cost in (1,2)]
        self.assertTrue(self.m.review_eligibility(rows, self.config)['eligible'])
        for changed in ({'MDD':.1001}, {'net_return':0}, {'trades':0}):
            bad = copy.deepcopy(rows)
            bad[0].update(changed)
            self.assertFalse(self.m.review_eligibility(bad, self.config)['eligible'])

    def test_incumbent_exposure_is_retrospective_unless_before_validation(self):
        metadata = {'periods': {'train':['2026-01-01T00:00:00Z','2026-02-01T00:00:00Z'],
                    'validation':['2026-02-05T00:00:00Z','2026-02-07T00:00:00Z']}}
        self.assertEqual(self.m.incumbent_scope(engine, metadata, '2026-02-06T00:00:00Z'), 'retrospective_diagnostic')
        self.assertEqual(self.m.incumbent_scope(engine, metadata, '2026-02-08T00:00:00Z'), 'out_of_sample_comparison')
        self.assertEqual(self.m.incumbent_scope(engine, {}, '2026-02-08T00:00:00Z'), 'retrospective_diagnostic')

    def test_explicit_training_permission_and_stop_are_required(self):
        with self.assertRaises(PermissionError):
            self.m.run_study(self.root, self.config, 'not-approved', allow_training=False)
        self.assertFalse((self.root/'10차/work/runs').exists())
        stop = self.root/'10차/work/STOP'
        stop.parent.mkdir(parents=True)
        stop.touch()
        with self.assertRaises(self.m.foundation.StopRequested):
            self.guard.check()
        self.assertTrue(stop.exists())

    def test_help_does_not_import_ml_or_start_loop(self):
        stream = io.StringIO()
        with patch.object(self.m, 'get_engine', side_effect=AssertionError('ML import 금지')), contextlib.redirect_stdout(stream), self.assertRaises(SystemExit) as raised:
            self.m.main(['--help'])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn('--allow-training', stream.getvalue())
        self.assertIn('--once', stream.getvalue())

    def test_status_matches_project_api_and_only_reports_observed_gpu(self):
        import dashboard
        self.status.update()
        value = dashboard.return_research(self.status.value)
        self.assertEqual(value['schema_version'], 1)
        self.assertEqual(value['objective'], 'net_forward_return')
        self.assertFalse(value['evaluation_performed'])
        self.assertIsNone(value['device'])
        self.m.fit_return_model(engine, self.lib, self.frame(), self.frame(), ['x'],
                                'mean', self.config, self.status, self.guard)
        value = dashboard.return_research(self.status.value)
        self.assertEqual(value['device'], 'cuda:0')
        self.assertEqual(value['rows'], 12)
        self.assertEqual(self.status.value['objective'], 'net_forward_return')
        self.assertTrue(self.status.value['gpu_verified'])

    def test_prepare_preserves_execution_tail_and_excludes_all_protected_labels(self):
        config = {**self.config, 'anchor': '2026-03-01T00:00:00Z'}
        plan = self.m.fold_plans(engine, config, engine.utc('2026-03-02T00:00:00Z'))[0]
        index = pd.date_range(plan['train'][0], plan['validation'][1], freq='5min')
        frame = self.frame(len(index), str(index[0]))
        frame = frame.drop(columns=['label', 'label_end', 'net_forward_return'])
        for tf in engine.TIMEFRAMES:
            frame['available_'+str(tf)] = frame.index
        base = copy.deepcopy(engine.DEFAULT_CONFIG)
        base['data_dir'] = str(self.root/'data')
        protected = (plan['tune'][0]+pd.Timedelta(hours=4), plan['tune'][0]+pd.Timedelta(hours=5))
        audit = SimpleNamespace(consumed_periods=lambda:[protected], expose=unittest.mock.Mock())
        incumbent = {'manifest': {'config':base}}
        with patch.object(self.m, 'readonly_store'), patch.object(self.m, 'foundation_exposures', return_value=([], [])), patch.object(engine, 'load_market_features', return_value=frame):
            prepared = self.m.prepare_fold(engine, base, incumbent, plan, config, audit, 'mock-run', self.guard)
        frames = self.m.evaluation_frames(engine, prepared, plan, 'validation')
        self.assertEqual(next(iter(frames.values())).index[-1], plan['validation'][1]-pd.Timedelta(minutes=5))
        for market, execution in frames.items():
            self.assertEqual(execution['research_entry_allowed'].tolist(),
                             execution.index.isin(prepared['parts'][market]['validation'].index).tolist())
            self.assertFalse(execution['research_entry_allowed'].iloc[-13:].any())
        for split in prepared['parts'].values():
            for name, part in split.items():
                self.assertTrue((part.label_end < plan[name][1]).all())
                self.assertTrue((part.index < plan[name][1]-pd.Timedelta(minutes=65)).all())
                self.assertFalse(((part.open_time < protected[1]) & (part.label_end >= protected[0])).any())
        self.assertTrue(audit.expose.called)
        self.assertFalse(prepared['external_data_used_for_training'])
        with patch.object(self.m, 'foundation_exposures', return_value=([plan['validation']], [])), self.assertRaisesRegex(ValueError, '보호'):
            self.m.prepare_fold(engine, base, incumbent, plan, config, audit, 'blocked', self.guard)

    def test_foundation_failed_and_complete_exposures_are_strictly_combined(self):
        work = self.root/'7차/work'
        directory = work/'exposures'
        directory.mkdir(parents=True)
        for run in ('failed', 'complete'):
            self.m.foundation.atomic_json(directory/(run+'.json'), {'reservation': {
                'validation_start':'2026-02-01T00:00:00Z', 'exposure_end':'2026-02-02T00:00:00Z'}})
        with patch.object(self.m.foundation, 'read_exposure_end', return_value='2026-02-02T00:00:00Z') as strict:
            periods, proof = self.m.foundation_exposures(self.root, self.guard)
        strict.assert_called_once_with(work, self.guard)
        self.assertEqual(len(periods), 2)
        self.assertEqual(len(proof), 2)
        with patch.object(self.m.foundation, 'read_exposure_end', side_effect=ValueError('예약 변조')):
            with self.assertRaisesRegex(ValueError, '예약 변조'):
                self.m.foundation_exposures(self.root, self.guard)

    def test_synthetic_portfolio_executes_next_open_and_stress_is_separate(self):
        frame = self.frame(100)
        frame['research_entry_allowed'] = True
        frame['open'] = 100.0
        frame['close'] = 100.0
        values = pd.DataFrame({'mean': [.02], 'q10':[.01], 'q90':[.03]}, index=frame.index[:1])
        paper = copy.deepcopy(engine.DEFAULT_CONFIG['paper'])
        calls = []
        process = engine.Portfolio.process
        def observe(portfolio, timestamp, prices, signals, **kwargs):
            self.assertIsNone(kwargs.get('risk_prices'))
            calls.append((timestamp, prices, dict(signals), kwargs['execution_at']))
            return process(portfolio, timestamp, prices, signals, **kwargs)
        with patch.object(engine.Portfolio, 'process', observe):
            one = self.m.evaluate(engine, {'KRW-BTC':frame}, {'KRW-BTC':values},
                {'family':'plain','penalty':0,'minimum':0}, paper, 12, 1)
            two = self.m.evaluate(engine, {'KRW-BTC':frame}, {'KRW-BTC':values},
                {'family':'plain','penalty':0,'minimum':0}, paper, 12, 2)
        self.assertEqual(calls[0][2]['KRW-BTC'], 1)
        self.assertEqual(calls[1][2]['KRW-BTC'], 2)
        self.assertEqual(calls[1][3], frame.index[0])
        self.assertEqual(one['trades'], 1)
        self.assertEqual(one['exit_reasons']['time_exit'], 1)
        self.assertLess(two['net_return'], one['net_return'])
        self.assertIsNone(one['historical_quote_extra_cost'])

    def test_all_comparators_share_entry_mask_but_keep_tail_sell_and_valuation(self):
        frame = self.frame(120)
        frame['research_entry_allowed'] = np.arange(120) < 100
        frame['5_ma_ratio'] = frame['5_return_12'] = .01
        predictions = pd.DataFrame({'mean':.02,'q10':.01,'q90':.03}, index=frame.index)
        incumbent = {'directory':self.root, 'manifest':{'feature_columns':['x'],
            'config':{'training':{'horizon':12}}, 'packages':{'KRW-BTC':{'rule':{'confidence':.9,'edge':.9}}}}}
        original_probs = np.tile([0.,0.,1.], (120,1))
        original_probs[-1] = [1.,0.,0.]
        captured = []
        def capture(frames, probabilities, rules, settings):
            self.assertEqual(len(frames['KRW-BTC']), 120)
            signal = engine.signals_from_proba(probabilities['KRW-BTC'], rules['KRW-BTC'])
            self.assertTrue((signal[:100] == 2).all())
            self.assertFalse((signal[100:] == 2).any())
            captured.append(signal)
            return {'net_return':0.,'MDD':0.,'trades':0}
        with patch.object(engine, 'backtest_portfolio', side_effect=capture), patch.object(engine, 'predict_package', return_value=original_probs):
            for cost in (1,2):
                self.m.evaluate(engine, {'KRW-BTC':frame}, {'KRW-BTC':predictions},
                    {'family':'plain','penalty':0,'minimum':0}, engine.DEFAULT_CONFIG['paper'],12,cost)
                for name in ('trend','incumbent'):
                    self.m.evaluate_baseline(engine, name, {'KRW-BTC':frame}, {'KRW-BTC':frame}, incumbent,
                                            engine.DEFAULT_CONFIG['paper'],12,cost)
        self.assertEqual(len(captured), 6)
        self.assertEqual(captured[2][-1], 0)
        self.assertEqual(captured[5][-1], 0)
        self.assertTrue((original_probs[:-1,2] == 1).all())
        with self.assertRaisesRegex(ValueError, '진입'):
            self.m.evaluate(engine, {'KRW-BTC':frame.drop(columns='research_entry_allowed')},
                {'KRW-BTC':predictions},{'family':'plain','penalty':0,'minimum':0},engine.DEFAULT_CONFIG['paper'],12,1)

    def test_mock_full_run_is_thirty_fits_reserved_and_never_promoted(self):
        config = {**self.config, 'anchor':'2026-03-01T00:00:00Z'}
        base = copy.deepcopy(engine.DEFAULT_CONFIG)
        base.update(data_dir=str(self.root/'data'), models_dir=str(self.root/'models'))
        marker = self.root/'work/service/TRAINING_PAUSED'
        marker.parent.mkdir(parents=True)
        marker.write_bytes(b'pause-preserved')
        prepared = {'parts': {m:{k:self.frame(120) for k in ('train','tune','rule','validation')}
                    for m in engine.MARKETS}, 'frames':{}, 'incumbent_frames':{}, 'columns':['x'],
                    'forbidden':[], 'paper':base['paper'], 'foundation_exposures':[],
                    'external_data_used_for_training':False}
        val_frames = {m:self.frame(120) for m in engine.MARKETS}
        for frame in val_frames.values():
            frame['research_entry_allowed'] = True
        result = {'net_return':0.0,'MDD':0.0,'trades':0}
        incumbent = {'run':'old','files':{},'manifest':{'config':base},'directory':self.root}
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(self.m, 'sys', SimpleNamespace(platform='linux', argv=sys.argv)))
            stack.enter_context(patch.object(self.m.foundation, 'exclusive_lock', lambda *_:contextlib.nullcontext()))
            stack.enter_context(patch.object(self.m, 'get_engine', return_value=engine))
            stack.enter_context(patch.object(self.m, 'get_xgb', return_value=self.lib))
            stack.enter_context(patch.object(engine, 'load_config', return_value=base))
            stack.enter_context(patch.object(engine, '_training_pause_requested', return_value=True))
            stack.enter_context(patch.object(engine, 'ExperimentAudit'))
            stack.enter_context(patch.object(self.m, 'freeze_incumbent', return_value=incumbent))
            stack.enter_context(patch.object(self.m, 'prepare_fold', return_value=prepared))
            stack.enter_context(patch.object(self.m, 'evaluation_frames', return_value=val_frames))
            stack.enter_context(patch.object(self.m, 'evaluate', return_value=result))
            stack.enter_context(patch.object(self.m, 'evaluate_baseline', return_value=result))
            stack.enter_context(patch.object(self.m.shutil, 'disk_usage', return_value=SimpleNamespace(free=10**12)))
            value = self.m.run_study(self.root, config, 'bounded-mock', True)
            with self.assertRaisesRegex(RuntimeError, '예약'):
                self.m.run_study(self.root, config, 'bounded-mock', True)
        self.assertEqual((value['fit_calls'], value['completed_fits']), (30,30))
        self.assertEqual(len(self.lib.calls), 30)
        self.assertTrue(value['evaluation_performed'])
        self.assertFalse(value['candidate_ready'])
        self.assertFalse(value['promotion'])
        self.assertFalse(value['research_review_eligible'])
        self.assertEqual(len(value['comparisons']), 24)
        directory = self.root/'10차/work/runs/bounded-mock'
        manifest = self.m.foundation.read_json(directory/'manifest.json')
        self.assertEqual(set(manifest['observed_validation_best'].values()), {'cash'})
        self.assertFalse(manifest['external_data_used_for_training'])
        self.assertEqual(marker.read_bytes(), b'pause-preserved')
        self.assertFalse((self.root/'models').exists())
        self.assertEqual(len(manifest['models']), 30)
        for relative, digest in manifest['artifacts'].items():
            self.assertEqual(self.m.foundation.file_hash(directory/relative), digest)

    def test_run_failure_records_actual_fit_and_never_retrains_same_reservation(self):
        config = {**self.config, 'anchor':'2026-03-01T00:00:00Z'}
        base = copy.deepcopy(engine.DEFAULT_CONFIG)
        base.update(data_dir=str(self.root/'data'), models_dir=str(self.root/'models'))
        prepared = {'parts':{m:{k:self.frame() for k in ('train','tune','rule','validation')}
                        for m in engine.MARKETS}, 'columns':['x'], 'forbidden':[],
                        'foundation_exposures':[]}
        directory = self.root/'10차/work/runs/failed-mock'
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(self.m, 'sys', SimpleNamespace(platform='linux', argv=sys.argv)))
            stack.enter_context(patch.object(self.m.foundation, 'exclusive_lock', lambda *_:contextlib.nullcontext()))
            stack.enter_context(patch.object(self.m, 'get_engine', return_value=engine))
            stack.enter_context(patch.object(self.m, 'get_xgb', return_value=self.lib))
            stack.enter_context(patch.object(engine, 'load_config', return_value=base))
            stack.enter_context(patch.object(engine, '_training_pause_requested', return_value=True))
            stack.enter_context(patch.object(engine, 'ExperimentAudit'))
            stack.enter_context(patch.object(self.m, 'freeze_incumbent', return_value={}))
            stack.enter_context(patch.object(self.m, 'prepare_fold', return_value=prepared))
            stack.enter_context(patch.object(self.m, 'save_verified_model', side_effect=ValueError('통제 재로드 불일치')))
            stack.enter_context(patch.object(self.m.shutil, 'disk_usage', return_value=SimpleNamespace(free=10**12)))
            with self.assertRaisesRegex(ValueError, '통제 재로드 불일치'):
                self.m.run_study(self.root, config, 'failed-mock', True)
            failure = self.m.foundation.read_json(directory/'failure.json')
            cache = self.m.foundation.read_json(self.root/'10차/work/status.json')
            before = (directory/'failure.json').read_bytes()
            with self.assertRaisesRegex(RuntimeError, '예약'):
                self.m.run_study(self.root, config, 'failed-mock', True)
        self.assertEqual(len(self.lib.calls), 1)
        self.assertEqual(failure['fit_calls'], 1)
        self.assertEqual(failure['completed_fits'], 1)
        self.assertEqual(cache['state'], 'error')
        self.assertFalse(cache['evaluation_performed'])
        self.assertFalse((directory/'manifest.json').exists())
        self.assertEqual((directory/'failure.json').read_bytes(), before)

    def test_fit_status_write_failure_preserves_actual_calls_once_and_persistently(self):
        for persistent in (False, True):
            with self.subTest(persistent=persistent), tempfile.TemporaryDirectory() as tmp:
                root, lib = Path(tmp), FakeXGB()
                base = copy.deepcopy(engine.DEFAULT_CONFIG)
                base.update(data_dir=str(root/'data'), models_dir=str(root/'models'))
                prepared = {'parts':{m:{k:self.frame() for k in ('train','tune','rule','validation')}
                            for m in engine.MARKETS}, 'columns':['x'], 'forbidden':[], 'foundation_exposures':[]}
                original_atomic = self.m.foundation.atomic_json
                errors = []
                def fail_status(path, value):
                    if Path(path).name == 'status.json' and value['fit_calls'] == 1 and (persistent or not errors):
                        errors.append('통제 fit 이후 상태 저장 실패')
                        raise OSError(errors[-1])
                    return original_atomic(path, value)
                train = lib.train
                def observe_call(*args, **kwargs):
                    cache = self.m.foundation.read_json(root/'10차/work/status.json')
                    self.assertEqual((cache['fit_calls'],cache['completed_fits']), (0,0))
                    return train(*args, **kwargs)
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(self.m, 'sys', SimpleNamespace(platform='linux', argv=sys.argv)))
                    stack.enter_context(patch.object(self.m.foundation, 'exclusive_lock', lambda *_:contextlib.nullcontext()))
                    stack.enter_context(patch.object(self.m, 'get_engine', return_value=engine))
                    stack.enter_context(patch.object(self.m, 'get_xgb', return_value=lib))
                    stack.enter_context(patch.object(engine, 'load_config', return_value=base))
                    stack.enter_context(patch.object(engine, '_training_pause_requested', return_value=True))
                    stack.enter_context(patch.object(engine, 'ExperimentAudit'))
                    stack.enter_context(patch.object(self.m, 'freeze_incumbent', return_value={}))
                    stack.enter_context(patch.object(self.m, 'prepare_fold', return_value=prepared))
                    stack.enter_context(patch.object(self.m.shutil, 'disk_usage', return_value=SimpleNamespace(free=10**12)))
                    stack.enter_context(patch.object(self.m.foundation, 'atomic_json', side_effect=fail_status))
                    stack.enter_context(patch.object(lib, 'train', side_effect=observe_call))
                    with self.assertRaisesRegex(OSError, '통제 fit 이후 상태 저장 실패'):
                        self.m.run_study(root, self.config, 'fit-io', True)
                    with self.assertRaisesRegex(RuntimeError, '예약'):
                        self.m.run_study(root, self.config, 'fit-io', True)
                failure = self.m.foundation.read_json(root/'10차/work/runs/fit-io/failure.json')
                self.assertEqual(len(lib.calls), 1)
                self.assertEqual(failure['fit_calls'], 1)
                self.assertEqual(failure['completed_fits'], 0)
                self.assertEqual(lib.updates, 2)
                self.assertEqual(failure['error'], '통제 fit 이후 상태 저장 실패')
                cache = self.m.foundation.read_json(root/'10차/work/status.json')
                if not persistent:
                    self.assertEqual(cache['state'], 'error')
                    self.assertEqual(cache['fit_calls'], 1)


if __name__ == '__main__':
    unittest.main()
