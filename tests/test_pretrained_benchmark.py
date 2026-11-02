"""11차의 미래 입력·CUDA·고정 모델·공통 기회를 순수 대역으로 검사합니다."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import engine
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'11차/pretrained_benchmark.py'


class FakeTensor:
    def __init__(self,device='cuda'):
        self.device = SimpleNamespace(type=device,index=0 if device=='cuda' else None)
        self.dtype = 'mock-float32'
    def numel(self):
        return 231289280


class FakeNative:
    input_cpu = False
    omit_hooks = False
    instances = []
    def __init__(self,torch_compile):
        if torch_compile:
            raise AssertionError('torch.compile 금지')
        self.__class__.instances.append(self)
        self.model = SimpleNamespace(device=SimpleNamespace(type='cuda',index=0),device_count=1,
            training=False,parameters=lambda:iter([FakeTensor()]),register_forward_pre_hook=self.pre,
            register_forward_hook=self.post)
    def pre(self,hook,**kwargs):
        self.before = hook
        return SimpleNamespace(remove=lambda:None)
    def post(self,hook,**kwargs):
        self.after = hook
        return SimpleNamespace(remove=lambda:None)
    def load_checkpoint(self,path,torch_compile):
        self.path = path
        if torch_compile:
            raise AssertionError('torch.compile 금지')
    def compile(self,config):
        self.config = config
    def forecast(self,horizon,inputs):
        if not self.omit_hooks:
            self.before(self.model,(FakeTensor('cpu' if self.input_cpu else 'cuda'),),{})
            self.after(self.model,(),(FakeTensor(),))
        q = np.zeros((len(inputs),horizon,10))
        q[:,:,0] = 103
        q[:,:,1:] = np.arange(96,105)
        return q[:,:,5],q


FAKE_TORCH = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True),float32='mock-float32',
    is_tensor=lambda value:isinstance(value,FakeTensor),inference_mode=contextlib.nullcontext,
    autocast=lambda *args,**kwargs:contextlib.nullcontext())
FAKE_TIMESFM = SimpleNamespace(TimesFM_2p5_200M_torch=FakeNative,ForecastConfig=SimpleNamespace)


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = None
        if SOURCE.exists():
            spec = importlib.util.spec_from_file_location('benchmark_test', SOURCE)
            cls.m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.m)

    def setUp(self):
        self.assertIsNotNone(self.m, '11차 실행기 미구현')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.m.settings({})

    def raw(self, n=1100):
        index = pd.date_range('2026-10-26T00:00:00Z', periods=n, freq='5min')
        return pd.DataFrame({'close':100+np.arange(n)*.01,'open':100.,'high':120.,'low':99.,
                             'volume':1.,'open_time':index-pd.Timedelta(minutes=5)}, index=index)

    def test_pins_bounds_and_future_anchor_never_relaxed(self):
        for value in ({'model_id':'google/timesfm-3.0-pytorch'},{'revision':'main'},
                      {'context_length':256},{'horizon':13},{'max_forecast_windows':6001},
                      {'batch_size':17},{'torch_compile':True},{'unknown':1}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.m.settings(value)
        plan = self.m.make_plan(engine, self.config, engine.utc('2026-11-02T00:00:00Z'))
        self.assertEqual((plan['validation'][1]-plan['rule'][0]).total_seconds(),4*86400)
        with self.assertRaises(ValueError):
            self.m.make_plan(engine,self.config,engine.utc('2020-01-01T00:00:00Z'))

    def test_windows_use_only_past_512_contiguous_prices(self):
        raw = self.raw()
        times = raw.index[520:525]
        windows = self.m.contexts(engine,raw,times,512)
        self.assertEqual(set(windows),set(times))
        self.assertEqual(windows[times[0]].shape,(512,))
        self.assertEqual(windows[times[0]][-1],np.float32(raw.close.loc[times[0]]))
        changed = raw.copy()
        changed.loc[changed.index > times[-1],'close'] = 99999
        for time, values in self.m.contexts(engine,changed,times,512).items():
            np.testing.assert_array_equal(values,windows[time])
        gap = raw.drop(raw.index[500])
        self.assertEqual(self.m.contexts(engine,gap,times,512),{})
        bad = raw.copy()
        bad.loc[bad.index[500],'close'] = np.nan
        self.assertEqual(self.m.contexts(engine,bad,times,512),{})

    def test_quantile_layout_and_median_are_not_confused_with_mean(self):
        quantiles = np.ones((2,12,10))*100
        quantiles[:,:,0] = 103
        quantiles[:,:,1:] = np.arange(96,105)
        point = quantiles[:,:,5].copy()
        values = self.m.checked_forecast(np,point,quantiles,2,12)
        np.testing.assert_array_equal(values['mean'],[103,103])
        np.testing.assert_array_equal(values['q10'],[96,96])
        np.testing.assert_array_equal(values['q90'],[104,104])
        for invalid in (quantiles[:,:,:9],np.full((2,12,10),np.nan)):
            with self.assertRaises(ValueError):
                self.m.checked_forecast(np,point,invalid,2,12)
        quantiles[:,:,8] = 200
        with self.assertRaises(ValueError):
            self.m.checked_forecast(np,point,quantiles,2,12)

    def test_return_denominator_is_observed_close_not_future_open(self):
        values = {'mean':np.array([103.]),'q10':np.array([101.]),'q90':np.array([105.])}
        result = self.m.net_predictions(engine,values,[100.],engine.DEFAULT_CONFIG['paper'])
        factor = engine.roundtrip_cost_factor(engine.DEFAULT_CONFIG['paper'])
        self.assertAlmostEqual(result['mean'][0],1.03*factor-1)

    def test_cuda_parameters_and_actual_tensor_io_are_both_required(self):
        param = SimpleNamespace(device=SimpleNamespace(type='cpu',index=None),numel=lambda:231289280)
        model = SimpleNamespace(parameters=lambda:iter([param]),training=False,device=SimpleNamespace(type='cuda',index=0),device_count=1)
        with self.assertRaises(RuntimeError):
            self.m.assert_cuda_model(model)
        param.device = SimpleNamespace(type='cuda',index=0)
        self.assertEqual(self.m.assert_cuda_model(model),231289280)
        tensor = SimpleNamespace(device=SimpleNamespace(type='cpu',index=None))
        torch = SimpleNamespace(is_tensor=lambda value:value is tensor)
        with self.assertRaises(RuntimeError):
            self.m.assert_tensor_cuda(torch,(tensor,))

    def test_snapshot_rejects_wrong_weights_and_non_apache(self):
        path = self.root/'snapshot'
        path.mkdir()
        (path/'model.safetensors').write_bytes(b'not-official')
        (path/'config.json').write_text('{}',encoding='utf-8')
        (path/'README.md').write_text('---\nlicense: apache-2.0\n---\n',encoding='utf-8')
        guard = self.m.Guard(self.root,60)
        with self.assertRaisesRegex(ValueError,'해시'):
            self.m.verify_snapshot(path,guard)
        original = self.m.f.file_hash
        def digest(p,guard=None):
            return self.m.WEIGHT_SHA256 if Path(p).name=='model.safetensors' else original(p,guard)
        (path/'README.md').write_text('---\nlicense: non-commercial\n---\n',encoding='utf-8')
        with patch.object(self.m.f,'file_hash',side_effect=digest),self.assertRaisesRegex(ValueError,'Apache'):
            self.m.verify_snapshot(path,guard)

    def test_common_mask_is_same_for_all_five_markets_and_keeps_tail(self):
        frame = self.raw(600)
        common = frame.index[520:525]
        frames = self.m.execution_frames(engine,{m:frame for m in engine.MARKETS},
             (frame.index[500],frame.index[-1]+pd.Timedelta(minutes=5)),common)
        self.assertEqual({int(v.research_entry_allowed.sum()) for v in frames.values()},{5})
        self.assertEqual({len(v) for v in frames.values()},{100})
        self.assertFalse(next(iter(frames.values())).research_entry_allowed.iloc[-1])

    def test_policy_selection_never_receives_validation(self):
        seen = []
        def evaluate(policy):
            seen.append(policy)
            return {'net_return':-.01,'MDD':.02,'trades':0}
        policy,trials = self.m.choose_policy(evaluate)
        self.assertEqual(policy['family'],'cash')
        self.assertEqual(len(trials),6)
        self.assertEqual(len(seen),6)

    def test_status_has_no_training_or_promotion_and_budget_is_strict(self):
        status = self.m.Status(self.root/'status.json','mock',self.config)
        status.update(total_windows=10,windows_completed=2)
        self.assertEqual(status.value['study'],'pretrained_forecast')
        self.assertEqual(status.value['fit_calls'],0)
        self.assertFalse(status.value['evaluation_performed'])
        for fields in ({'promotion':True},{'fit_calls':1},{'total_windows':6001},{'windows_completed':11}):
            with self.assertRaises(ValueError):
                status.update(**fields)

    def test_stop_and_duplicate_reservation_do_not_touch_other_tracks(self):
        guard = self.m.Guard(self.root,60)
        stop = self.root/'11차/work/STOP'
        stop.parent.mkdir(parents=True)
        stop.touch()
        with self.assertRaises(self.m.f.StopRequested):
            guard.check()
        self.assertTrue(stop.exists())
        with patch.object(self.m,'platform_name',return_value='win32'), self.assertRaises(RuntimeError):
            self.m.run_once(self.root,self.config,'blocked')
        self.assertFalse((self.root/'11차/work/runs').exists())

    def test_help_has_no_ml_or_network_import(self):
        out = io.StringIO()
        with patch.object(self.m,'runtime_modules',side_effect=AssertionError('실행 금지')), contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as raised:
            self.m.main(['--help'])
        self.assertEqual(raised.exception.code,0)
        self.assertIn('--once',out.getvalue())
        self.assertIn('--run',out.getvalue())

    def test_native_adapter_checks_cuda_forward_and_counts_actual_calls_on_failure(self):
        for cpu,omit in ((False,False),(True,False),(False,True)):
            counters = {'attempted_windows':0,'windows_completed':0,'padded_windows':0,'inference_calls':0,'gpu_verified':False}
            with patch.object(FakeNative,'input_cpu',cpu),patch.object(FakeNative,'omit_hooks',omit):
                runtime = self.m.Runtime(FAKE_TORCH,FAKE_TIMESFM,np,self.root,self.config,counters,self.m.Guard(self.root,60))
                if cpu or omit:
                    with self.assertRaises(RuntimeError):
                        runtime.forecast([np.ones(512)])
                    self.assertEqual(counters['windows_completed'],0)
                    self.assertFalse(counters['gpu_verified'])
                else:
                    values = runtime.forecast([np.ones(512)])
                    self.assertEqual(values['mean'][0],103)
                    self.assertEqual(counters['windows_completed'],1)
                runtime.close()
                self.assertEqual(counters['inference_calls'],1)
                self.assertEqual(counters['attempted_windows'],1)
                self.assertEqual(counters['padded_windows'],16)

    def mock_run(self,fail_reload=False,fail_status=False):
        base = json.loads(json.dumps(engine.DEFAULT_CONFIG))
        base.update(data_dir=str(self.root/'data'),models_dir=str(self.root/'models'))
        plan = self.m.make_plan(engine,self.config,engine.utc('2026-11-02T00:00:00Z'))
        index = pd.date_range(plan['rule'][0]-pd.Timedelta(minutes=512*5),plan['validation'][1],freq='5min')
        raw = pd.DataFrame({'open':100.,'close':100.,'high':101.,'low':99.,'volume':1.,
                            'open_time':index-pd.Timedelta(minutes=5),'x':1.},index=index)
        times = {name:pd.date_range(a,periods=3,freq='5min') for name,(a,b) in plan.items()}
        tasks = [{'market':m,'stamp':t,'split':name,'context':np.ones(512)*100}
                 for name,idx in times.items() for t in idx for m in engine.MARKETS]
        prepared = {'plan':plan,'raw':{m:raw for m in engine.MARKETS},'frames':{m:raw for m in engine.MARKETS},
                    'incumbent_frames':{m:raw for m in engine.MARKETS},'tasks':tasks,'split_times':times,
                    'foundation_exposures':[],'forbidden':[],'rows':5*len(raw)}
        incumbent = {'run':'old','files':{},'directory':self.root,'manifest':{'config':base,'feature_columns':['x']}}
        original_atomic = self.m.f.atomic_json
        def atomic(path,value):
            if fail_status and Path(path).name=='status.json' and value['windows_completed'] > 0:
                raise OSError('통제 표시 저장 실패')
            return original_atomic(path,value)
        original_forecast = self.m.Runtime.forecast
        calls = []
        def forecast(rt,inputs):
            output = original_forecast(rt,inputs)
            calls.append(len(inputs))
            if fail_reload and len(calls)==4:
                output['mean'] = output['mean']+1
            return output
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(self.m,'platform_name',return_value='linux'))
            stack.enter_context(patch.object(self.m.f,'exclusive_lock',lambda *_:contextlib.nullcontext()))
            stack.enter_context(patch.object(self.m,'runtime_modules',return_value=(FAKE_TORCH,FAKE_TIMESFM,{'timesfm':'2.0.2'},{})))
            stack.enter_context(patch.object(self.m.r,'get_engine',return_value=engine))
            stack.enter_context(patch.object(engine,'load_config',return_value=base))
            stack.enter_context(patch.object(engine,'ExperimentAudit'))
            stack.enter_context(patch.object(self.m.r,'freeze_incumbent',return_value=incumbent))
            stack.enter_context(patch.object(self.m,'prepare_data',return_value=prepared))
            stack.enter_context(patch.object(self.m,'verify_snapshot',return_value={'files':{'model.safetensors':'mocksha'}}))
            stack.enter_context(patch.object(self.m.shutil,'disk_usage',return_value=SimpleNamespace(free=10**12)))
            stack.enter_context(patch.object(self.m.r,'evaluate',return_value={'net_return':0.,'MDD':0.,'trades':0}))
            stack.enter_context(patch.object(self.m.r,'evaluate_baseline',return_value={'net_return':0.,'MDD':0.,'trades':0}))
            original_hash = self.m.f.file_hash
            stack.enter_context(patch.object(self.m.f,'file_hash',side_effect=lambda p,g=None:'mocksha' if Path(p).name=='model.safetensors' else original_hash(p,g)))
            stack.enter_context(patch.object(self.m.f,'atomic_json',side_effect=atomic))
            stack.enter_context(patch.object(self.m.Runtime,'forecast',forecast))
            try:
                value = self.m.run_once(self.root,self.config,'controlled')
            finally:
                with self.assertRaisesRegex(RuntimeError,'예약'):
                    self.m.run_once(self.root,self.config,'controlled')
        return value,calls

    def test_mock_once_reloads_same_model_no_fit_and_hashes_artifacts(self):
        value,calls = self.mock_run()
        self.assertEqual(value['state'],'complete')
        self.assertEqual(value['fit_calls'],0)
        self.assertFalse(value['promotion'])
        self.assertEqual(value['windows_completed'],value['total_windows'])
        self.assertEqual(value['windows_completed'],32)
        self.assertEqual(len(calls),4)
        self.assertTrue(value['checkpoint_verified'])
        directory = self.root/'11차/work/runs/controlled'
        manifest = self.m.f.read_json(directory/'manifest.json')
        for p,h in manifest['artifacts'].items():
            self.assertEqual(self.m.f.file_hash(directory/p),h)
        self.assertFalse((self.root/'models').exists())

    def test_reload_mismatch_keeps_original_probe_and_fails_without_retry(self):
        with self.assertRaisesRegex(ValueError,'재로드'):
            self.mock_run(fail_reload=True)
        directory = self.root/'11차/work/runs/controlled'
        proof = self.m.f.read_json(directory/'reload-proof.json')
        self.assertFalse(proof['reload_prediction_match'])
        self.assertTrue((directory/'reload-probe-before.json').exists())
        self.assertFalse((directory/'manifest.json').exists())

    def test_persistent_progress_write_error_preserves_actual_inference_counters(self):
        with self.assertRaisesRegex(OSError,'통제 표시'):
            self.mock_run(fail_status=True)
        failure = self.m.f.read_json(self.root/'11차/work/runs/controlled/failure.json')
        self.assertEqual(failure['fit_calls'],0)
        self.assertEqual(failure['counters']['inference_calls'],2)
        self.assertEqual(failure['counters']['attempted_windows'],17)
        self.assertEqual(failure['counters']['windows_completed'],17)

    def test_preparation_reserves_before_reads_and_protection_blocks_before_extract(self):
        plan = self.m.make_plan(engine,self.config,engine.utc('2026-11-02T00:00:00Z'))
        warm = plan['rule'][0]-pd.Timedelta(minutes=512*5+5)
        index = pd.date_range(warm,plan['validation'][1],freq='5min')
        raw = pd.DataFrame({'open':100.,'high':101.,'low':99.,'close':100.,'volume':1.},
                           index=index-pd.Timedelta(minutes=5))
        feature = raw.copy()
        feature['open_time'] = feature.index
        feature.index = index
        feature['x'] = 1.
        for tf in engine.TIMEFRAMES:
            feature['available_'+str(tf)] = feature.index
        base = json.loads(json.dumps(engine.DEFAULT_CONFIG))
        base['data_dir'] = str(self.root/'data')
        incumbent = {'manifest':{'config':base,'feature_columns':['x']}}
        events = []
        audit = SimpleNamespace(consumed_periods=lambda:[],expose=lambda *args:events.append('reserve'))
        def read(*args):
            self.assertEqual(events[0],'reserve')
            events.append('read')
            return raw
        chosen = {time:np.ones(512)*100 for period in plan.values() for time in pd.date_range(period[0],period[0]+pd.Timedelta(minutes=5*99),freq='5min')}
        with patch.object(self.m.r,'foundation_exposures',return_value=([],[])),\
                patch.object(self.m.r,'readonly_store',return_value=SimpleNamespace(read=read)),\
                patch.object(engine,'load_market_features',return_value=feature),\
                patch.object(self.m,'contexts',return_value=chosen):
            prepared = self.m.prepare_data(engine,base,incumbent,self.config,audit,'mock',self.m.Guard(self.root,60))
        self.assertEqual(len(prepared['tasks']),1000)
        self.assertTrue(all(set(t)=={'market','stamp','split','context'} for t in prepared['tasks']))
        self.assertTrue(all(t['context'].shape==(512,) for t in prepared['tasks']))
        with patch.object(self.m.r,'foundation_exposures',return_value=([],[])),\
                patch.object(self.m.r,'readonly_store',return_value=SimpleNamespace(read=read)),\
                patch.object(engine,'load_market_features',return_value=feature),\
                patch.object(self.m,'contexts',side_effect=[chosen,chosen,chosen,chosen,{}]),\
                self.assertRaisesRegex(self.m.f.NotReady,'KRW-DOGE.*512'):
            self.m.prepare_data(engine,base,incumbent,self.config,audit,'no-doge-context',self.m.Guard(self.root,60))
        empty = feature.copy()
        empty['x'] = np.nan
        with patch.object(self.m.r,'foundation_exposures',return_value=([],[])),\
                patch.object(self.m.r,'readonly_store',return_value=SimpleNamespace(read=read)),\
                patch.object(engine,'load_market_features',return_value=empty),\
                self.assertRaisesRegex(self.m.f.NotReady,'KRW-BTC.*피처'):
            self.m.prepare_data(engine,base,incumbent,self.config,audit,'no-features',self.m.Guard(self.root,60))
        events.clear()
        audit.consumed_periods = lambda:[plan['validation']]
        with patch.object(self.m.r,'foundation_exposures',return_value=([],[])),\
                patch.object(self.m.r,'readonly_store',side_effect=AssertionError('보호 전 읽기 금지')),\
                self.assertRaisesRegex(ValueError,'보호'):
            self.m.prepare_data(engine,base,incumbent,self.config,audit,'blocked',self.m.Guard(self.root,60))
        self.assertEqual(events,[])

    def test_budget_includes_padding_and_no_forecast_after_stop(self):
        config = {**self.config,'max_forecast_windows':16}
        counts = {'attempted_windows':0,'windows_completed':0,'padded_windows':0,'inference_calls':0,'gpu_verified':False}
        guard = self.m.Guard(self.root,60)
        rt = self.m.Runtime(FAKE_TORCH,FAKE_TIMESFM,np,self.root,config,counts,guard)
        rt.forecast([np.ones(512)])
        with self.assertRaisesRegex(ValueError,'예산'):
            rt.forecast([np.ones(512)])
        self.assertEqual(counts['inference_calls'],1)
        guard.event.set()
        with self.assertRaises(self.m.f.StopRequested):
            rt.forecast([np.ones(512)])
        rt.close()


if __name__ == '__main__':
    unittest.main()
