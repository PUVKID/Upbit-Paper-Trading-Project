"""고정 TimesFM 2.5의 ZERO-SHOT 후향 비교입니다. 추가학습·자동 승격은 하지 않습니다."""
import argparse
import copy
import gc
import importlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import signal
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('benchmark_return_helpers', ROOT/'10차/return_research.py')
r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r)
f = r.foundation
MODEL_ID = 'google/timesfm-2.5-200m-pytorch'
REVISION = '1d952420fba87f3c6dee4f240de0f1a0fbc790e3'
WEIGHT_SHA256 = '2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8'
DEFAULTS = {'model_id':MODEL_ID, 'revision':REVISION, 'anchor':'2026-10-31T08:00:00Z',
    'context_length':512, 'horizon':12, 'policy_days':2, 'validation_days':2,
    'batch_size':16, 'max_forecast_windows':6000, 'max_raw_rows':100000,
    'max_seconds':7200, 'min_free_bytes':20000000000, 'torch_compile':False,
    'package_version':'2.0.2', 'dependencies_dir':'11차/work/python-packages',
    'pretrained_dir':'11차/work/pretrained/timesfm-2.5-200m'}


def platform_name():
    return sys.platform


def settings(value):
    if not isinstance(value,dict) or set(value)-set(DEFAULTS):
        raise ValueError('지원하지 않는 ZERO-SHOT 설정입니다')
    config = {**DEFAULTS, **value}
    for key in ('model_id','revision','context_length','horizon','policy_days','validation_days',
                'torch_compile','package_version','dependencies_dir','pretrained_dir'):
        if type(config[key]) is not type(DEFAULTS[key]) or config[key] != DEFAULTS[key]:
            raise ValueError('고정 모델·기간·격리 설정 위반: '+key)
    f.stamp(config['anchor'])
    for key,low,high in (('batch_size',1,16),('max_forecast_windows',10,6000),
            ('max_raw_rows',2000,100000),('max_seconds',1,7200),('min_free_bytes',20000000000,10**12)):
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError('유한 연구 예산 위반: '+key)
    return config


class Guard(f.Guard):
    def __init__(self,root,seconds):
        self.root = Path(root)
        super().__init__(f.safe_path(root,'11차/work/STOP'),seconds)


class Status:
    def __init__(self,path,run,config):
        self.path = Path(path)
        self.value = {'schema_version':1,'study':'pretrained_forecast','model_id':MODEL_ID,
            'run':run,'state':'running','stage':'prepare','windows_completed':0,'total_windows':0,
            'fit_calls':0,'gpu_verified':False,'device':None,'rows':None,
            'promotion':False,'candidate_ready':False,'evaluation_performed':False,
            'comparisons':[],'context_length':config['context_length'],'horizon_minutes':60,
            'batch_size':config['batch_size'],'started_utc':f.now_iso()}

    def update(self,**fields):
        value = {**self.value,**fields,'updated_utc':f.now_iso()}
        if (value['fit_calls'] != 0 or value['promotion'] is not False or value['candidate_ready'] is not False
                or not 0 <= value['windows_completed'] <= value['total_windows'] <= 6000):
            raise ValueError('추가학습·승격 금지 또는 예측 창 예산 위반')
        f.atomic_json(self.path,value)
        self.value = value


def runtime_modules(root,config):
    target = f.safe_path(root,config['dependencies_dir'])
    f.assert_no_links(target)
    if not target.is_dir():
        raise FileNotFoundError('부모가 준비한 격리 의존성 폴더가 필요합니다')
    for name in ('torch','numpy','pandas'):
        if (target/name).exists():
            raise ValueError('공유 Torch·배열 환경을 target 패키지로 덮을 수 없습니다')
    sys.path.insert(0,str(target))
    versions = {name:importlib.metadata.version(name) for name in
                ('timesfm','xgboost','torch','numpy','pandas','huggingface-hub','safetensors')}
    if versions['timesfm'] != '2.0.2' or versions['xgboost'] != '3.0.5':
        raise ValueError('timesfm2.0.2와 기존 모델용 xgboost3.0.5가 필요합니다')
    torch = importlib.import_module('torch')
    timesfm = importlib.import_module('timesfm')
    if not Path(timesfm.__file__).resolve().is_relative_to(target.resolve()):
        raise ValueError('TimesFM이 지정 target 밖에서 로드됐습니다')
    sources = {p.relative_to(target).as_posix():f.file_hash(p) for p in sorted((target/'timesfm').rglob('*.py'))}
    return torch,timesfm,versions,sources


def verify_snapshot(path,guard):
    path = Path(path)
    f.assert_no_links(path)
    for item in path.rglob('*'):
        f.assert_no_links(item)
    files = {name:f.file_hash(f.safe_path(path,name),guard) for name in ('config.json','model.safetensors','README.md')}
    if files['model.safetensors'] != WEIGHT_SHA256:
        raise ValueError('공식 safetensors 해시가 다릅니다')
    f.read_json(path/'config.json')
    card = (path/'README.md').read_text(encoding='utf-8')
    header = card.split('---',2)
    if len(header) != 3 or header[0].strip() or not re.search(r'^license:\s*apache-2\.0\s*$',header[1],re.M):
        raise ValueError('공식 Apache-2.0 모델 카드가 필요합니다')
    return {'model_id':MODEL_ID,'revision':REVISION,'license':'apache-2.0','files':files,
            'weight_bytes':(path/'model.safetensors').stat().st_size,'trust_remote_code':False,
            'loader':'native_constructor_and_local_load_checkpoint_no_hub_call'}


def assert_cuda_model(model):
    if model.training or model.device.type != 'cuda' or model.device.index != 0 or model.device_count != 1:
        raise RuntimeError('단일 cuda:0 평가 모델이 필요합니다. CPU 대체 금지')
    count = 0
    for parameter in model.parameters():
        if parameter.device.type != 'cuda' or parameter.device.index != 0:
            raise RuntimeError('실제 모델 parameter가 CUDA가 아닙니다')
        count += parameter.numel()
    if count != 231289280:
        raise RuntimeError('고정 2.5 평균·분위수 모델의 parameter 수가 다릅니다')
    return count


def assert_tensor_cuda(torch,value):
    count = 0
    if torch.is_tensor(value):
        if value.device.type != 'cuda' or value.device.index != 0:
            raise RuntimeError('실제 forward 입출력이 CUDA가 아닙니다')
        return 1
    if isinstance(value,(list,tuple,dict)):
        for child in (value.values() if isinstance(value,dict) else value):
            count += assert_tensor_cuda(torch,child)
    return count


def checked_forecast(np,point,quantiles,batch,horizon):
    point,quantiles = np.asarray(point),np.asarray(quantiles)
    if (point.shape != (batch,horizon) or quantiles.shape != (batch,horizon,10)
            or not np.isfinite(point).all() or not np.isfinite(quantiles).all()
            or (quantiles <= 0).any() or (np.diff(quantiles[:,:,1:],axis=2) < 0).any()
            or not np.allclose(point,quantiles[:,:,5],rtol=1e-5,atol=1e-5)):
        raise ValueError('공식 평균·분위수·중앙값 형태 또는 유한성 위반')
    return {'mean':quantiles[:,-1,0], 'q10':quantiles[:,-1,1], 'q90':quantiles[:,-1,9],
            'full_mean':quantiles[:,:,0], 'point':point}


class Runtime:
    def __init__(self,torch,timesfm,np,path,config,counters,guard):
        self.torch,self.np,self.config,self.counters,self.guard = torch,np,config,counters,guard
        guard.check()
        if not torch.cuda.is_available():
            raise RuntimeError('실제 CUDA가 없습니다. CPU 대체 금지')
        self.pipeline = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
        self.pipeline.load_checkpoint(str(path),torch_compile=False)
        self.parameter_count = assert_cuda_model(self.pipeline.model)
        if any(p.dtype != torch.float32 for p in self.pipeline.model.parameters()):
            raise RuntimeError('공식 F32 가중치만 허용합니다')
        self.pipeline.compile(timesfm.ForecastConfig(max_context=512,max_horizon=12,
            per_core_batch_size=config['batch_size'],normalize_inputs=True,
            use_continuous_quantile_head=True,force_flip_invariance=True,infer_is_positive=True,
            fix_quantile_crossing=True,return_backcast=False))
        self.io = {'before':0,'after':0}
        def before(module,args,kwargs):
            guard.check()
            assert_cuda_model(module)
            if assert_tensor_cuda(torch,(args,kwargs)) == 0:
                raise RuntimeError('CUDA 입력 tensor 관측이 없습니다')
            self.io['before'] += 1
        def after(module,args,output):
            if assert_tensor_cuda(torch,output) == 0:
                raise RuntimeError('CUDA 출력 tensor 관측이 없습니다')
            self.io['after'] += 1
        self.hooks = [self.pipeline.model.register_forward_pre_hook(before,with_kwargs=True),
                      self.pipeline.model.register_forward_hook(after)]

    def forecast(self,inputs):
        self.guard.check()
        if not 0 < len(inputs) <= self.config['batch_size']:
            raise ValueError('예측 batch 상한 위반')
        values = [self.np.asarray(v,dtype=self.np.float32).copy() for v in inputs]
        if any(v.shape != (512,) or not self.np.isfinite(v).all() or (v <= 0).any() for v in values):
            raise ValueError('유효한 과거 연속512봉 배열만 입력할 수 있습니다')
        padded = self.config['batch_size']
        if self.counters['padded_windows']+padded > self.config['max_forecast_windows']:
            raise ValueError('padding을 포함한 예측 예산 초과')
        before = dict(self.io)
        try:
            with self.torch.inference_mode(), self.torch.autocast('cuda',enabled=False):
                point,quantiles = self.pipeline.forecast(horizon=12,inputs=values)
        finally:
            # 실제 호출이 끝난 뒤에만 관측 수를 올려 표시 저장 실패와 분리합니다.
            self.counters['attempted_windows'] += len(inputs)
            self.counters['padded_windows'] += padded
            self.counters['inference_calls'] += 1
        if self.io['before'] <= before['before'] or self.io['after'] <= before['after']:
            raise RuntimeError('실제 CUDA forward 입출력 증거가 없습니다')
        assert_cuda_model(self.pipeline.model)
        result = checked_forecast(self.np,point,quantiles,len(inputs),12)
        self.counters['windows_completed'] += len(inputs)
        self.counters['gpu_verified'] = True
        return result

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.pipeline = None


def make_plan(engine,config,now):
    end = engine.utc(config['anchor'])
    if end > engine.utc(now) or end.timestamp()%300:
        raise ValueError('anchor는 과거 완료5분 경계여야 합니다')
    mid = end-engine.pd.Timedelta(days=2)
    return {'rule':(mid-engine.pd.Timedelta(days=2),mid),'validation':(mid,end)}


def contexts(engine,raw,times,length):
    if not raw.index.is_monotonic_increasing or not raw.index.is_unique:
        raise ValueError('원시 시각 역행·중복')
    out = {}
    for stamp in times:
        location = raw.index.get_indexer([stamp])[0]
        if location < length-1:
            continue
        part = raw.iloc[location-length+1:location+1]
        values = part.close.to_numpy(dtype=engine.np.float32)
        if (len(part) == length and ((part.index[1:]-part.index[:-1]) == engine.pd.Timedelta(minutes=5)).all()
                and engine.np.isfinite(values).all() and (values > 0).all()):
            out[stamp] = values.copy()
    return out


def prepare_data(engine,base,incumbent,config,audit,run,guard):
    plan = make_plan(engine,config,engine.utc(None))
    previous,proof = r.foundation_exposures(guard.root,guard)
    forbidden = engine.protected_periods(base,[*audit.consumed_periods(),*previous])
    if any(a < right and b > left for a,b in plan.values() for left,right in forbidden):
        raise ValueError('정책·평가 기간이 보호 또는7차 소비 구간과 겹칩니다')
    warm = plan['rule'][0]-engine.pd.Timedelta(minutes=512*5+5)
    # 자료 읽기 전에 전체 문맥·평가 노출을 예약하여 준비 실패도 보수적으로 남깁니다.
    audit.expose(run,warm,plan['validation'][1],'timesfm_zero_shot_retrospective_context_policy_validation')
    store = r.readonly_store(engine,Path(base['data_dir'])/'candles.sqlite3',forbidden,guard)
    feature_config = copy.deepcopy(base)
    feature_config['external'].update(enabled=False,wait_for_dataset=False)
    feature_config['training'].update(max_rows_per_market=20000,max_raw_rows=config['max_raw_rows'])
    inc_config = copy.deepcopy(incumbent['manifest']['config'])
    inc_config['training'].update(max_rows_per_market=20000,max_raw_rows=config['max_raw_rows'])
    raw,frames,inc_frames,available = {},{},{},{}
    for market in engine.MARKETS:
        guard.check()
        value = store.read(market,5,warm,plan['validation'][1],config['max_raw_rows'])
        if value.empty:
            raise ValueError('실제5분봉이 없습니다: '+market)
        value = value.copy()
        value['open_time'] = value.index
        value.index = value.index+engine.pd.Timedelta(minutes=5)
        raw[market] = value
        frame = engine.load_market_features(store,market,feature_config,warm,plan['validation'][1])
        inc_frame = engine.load_market_features(store,market,inc_config,warm,plan['validation'][1])
        for candidate in (frame,inc_frame):
            for tf in engine.TIMEFRAMES:
                observed = candidate['available_'+str(tf)]
                if (observed.notna() & (observed > candidate.index)).any():
                    raise ValueError('미완료 시간봉 특징 누수')
        frames[market] = frame.dropna(subset=engine.feature_columns(frame))
        inc_frames[market] = inc_frame.dropna(subset=incumbent['manifest']['feature_columns'])
        if frames[market].empty or inc_frames[market].empty:
            raise f.NotReady(f'{market}: 완료 피처 행 부족 '
                f'(후보={len(frames[market])}, 기존={len(inc_frames[market])}); 피처 변경·미래 채움 금지')
        indices = frames[market].index.intersection(inc_frames[market].index)
        available[market] = contexts(engine,value,indices,512)
        if not available[market]:
            groups = engine.pd.Series(value.index).diff().ne(engine.pd.Timedelta(minutes=5)).cumsum()
            longest = int(groups.value_counts().max())
            raise f.NotReady(f'{market}: 연속512봉 문맥0개 '
                f'(원시={len(value)}, 완료피처교집합={len(indices)}, 최장연속={longest}); 기준 완화·보간 금지')
    split_times,tasks = {},[]
    for name,(start,end) in plan.items():
        common = None
        for market in engine.MARKETS:
            index = engine.pd.DatetimeIndex(list(available[market]),tz='UTC')
            index = index[(index >= start) & (index < end-engine.pd.Timedelta(minutes=65))]
            common = index if common is None else common.intersection(index)
        if common is None or len(common) < 100:
            raise ValueError('5시장 공통 연속 문맥 창이 부족합니다: '+name)
        split_times[name] = common.sort_values()
        for stamp in split_times[name]:
            for market in engine.MARKETS:
                tasks.append({'market':market,'stamp':stamp,'split':name,'context':available[market][stamp]})
    return {'plan':plan,'raw':raw,'frames':frames,'incumbent_frames':inc_frames,
            'split_times':split_times,'tasks':tasks,'foundation_exposures':proof,
            'forbidden':forbidden,'rows':sum(len(v) for v in raw.values())}


def execution_frames(engine,frames,period,entry_times):
    start,end = period
    values = {m:v[(v.index >= start)&(v.index < end)].copy() for m,v in frames.items()}
    common = None
    for frame in values.values():
        common = frame.index if common is None else common.intersection(frame.index)
    if common is None or len(common) < 100 or not entry_times.isin(common).all():
        raise ValueError('5시장 동일 평가 행과 진입 기회를 확보하지 못했습니다')
    out = {}
    for market,frame in values.items():
        frame = frame.loc[common].copy()
        frame['research_entry_allowed'] = common.isin(entry_times)
        out[market] = frame
    return out


def net_predictions(engine,values,last_closes,paper):
    denominator = engine.np.asarray(last_closes)
    if not engine.np.isfinite(denominator).all() or (denominator <= 0).any():
        raise ValueError('입력 마지막 완료 종가가 유효하지 않습니다')
    factor = engine.roundtrip_cost_factor(paper)
    return {key:engine.np.asarray(values[key])/denominator*factor-1 for key in ('mean','q10','q90')}


def choose_policy(evaluate):
    best,score = {'family':'cash','penalty':0.,'minimum':0.},0.
    trials = []
    for penalty in (0.,.5):
        for minimum in (0.,.001,.003):
            policy = {'family':'plain' if penalty == 0 else 'uncertainty','penalty':penalty,'minimum':minimum}
            result = evaluate(policy)
            objective = result['net_return']-.5*result['MDD']
            eligible = (result['trades'] >= 3 and result['net_return'] > 0
                        and 0 <= result['MDD'] <= .1 and math.isfinite(objective))
            trials.append({'policy':policy,'trading':result,'eligible':eligible})
            if eligible and objective > score:
                best,score = policy,objective
    return best,trials


def error_metrics(engine,prepared,records):
    errors,naive,directions = [],[],[]
    for record in records:
        if record['split'] != 'validation':
            continue
        raw = prepared['raw'][record['market']]
        targets = record['stamp']+engine.pd.to_timedelta(engine.np.arange(1,13)*5,unit='min')
        actual = raw.close.reindex(targets).to_numpy()
        if not engine.np.isfinite(actual).all():
            raise ValueError('예측 후 오차 계산의 실제 지평 봉이 누락됐습니다')
        last = record['last_close']
        predicted = engine.np.asarray(record['full_mean'])
        errors.extend((engine.np.abs(predicted-actual)/last).tolist())
        naive.extend((engine.np.abs(last-actual)/last).tolist())
        directions.append(int(engine.np.sign(predicted[-1]-last) == engine.np.sign(actual[-1]-last)))
    if not errors:
        raise ValueError('후속 예측 오차 표본이 없습니다')
    return {'relative_mae':float(engine.np.mean(errors)),'naive_relative_mae':float(engine.np.mean(naive)),
            'direction_accuracy':float(engine.np.mean(directions)),'samples':len(directions),
            'scope':'retrospective_not_independent_final_test','price_prediction_not_trading_return':True}


def run_once(root,config,run):
    if platform_name() != 'linux':
        raise RuntimeError('실제 DB·모델 추론은 Linux 컨테이너 전용입니다')
    config,root = settings(config),Path(root).resolve()
    if not isinstance(run,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',run):
        raise ValueError('명시적인 실험 식별자가 필요합니다')
    work = f.safe_path(root,'11차/work')
    f.assert_no_links(work)
    guard = Guard(root,config['max_seconds'])
    counters = {'attempted_windows':0,'windows_completed':0,'padded_windows':0,'inference_calls':0,'gpu_verified':False}
    runtime,old_signals = None,{}
    with f.exclusive_lock(work/'benchmark.lock'):
        guard.check()
        directory = f.safe_path(work,'runs/'+run)
        if directory.exists():
            raise RuntimeError('기존 실험 예약은 재실행·자동 resume할 수 없습니다')
        directory.mkdir(parents=True)
        status = Status(work/'status.json',run,config)
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM,signal.SIGINT):
                old_signals[sig] = signal.signal(sig,lambda *_:guard.event.set())
        started = time.monotonic()
        try:
            status.update()
            if shutil.disk_usage(work).free < config['min_free_bytes']:
                raise OSError('20GB 디스크 여유가 필요합니다. 자동 삭제 금지')
            torch,timesfm,versions,sources = runtime_modules(root,config)
            engine = r.get_engine()
            base = engine.load_config(root/'engine-config.json')
            if tuple(base['markets']) != tuple(engine.MARKETS) or tuple(base['timeframes']) != tuple(engine.TIMEFRAMES):
                raise ValueError('기존5시장·9시간봉 계약 위반')
            path = f.safe_path(root,config['pretrained_dir'])
            snapshot = verify_snapshot(path,guard)
            f.atomic_json(directory/'request.json',{'run':run,'settings':config,'model':snapshot,
                'packages':versions,'package_sources':sources,'source_sha256':f.file_hash(Path(__file__),guard),
                'engine_sha256':f.file_hash(Path(engine.__file__),guard),'fit_calls':0,
                'scope':'zero_shot_retrospective_not_independent_final_test',
                'external_features_for_foundation':False,'promotion':False,
                'return_estimate_basis':'last_observed_close_proxy_for_unknown_next_open',
                'historical_quote_extra_cost':None})
            incumbent = r.freeze_incumbent(engine,base,directory,guard)
            audit = engine.ExperimentAudit(Path(base['data_dir'])/'experiment_audit.sqlite3')
            prepared = prepare_data(engine,base,incumbent,config,audit,run,guard)
            tasks = prepared['tasks']
            padded = math.ceil(len(tasks)/config['batch_size'])*config['batch_size']+2*config['batch_size']
            if len(tasks)+2 > config['max_forecast_windows'] or padded > config['max_forecast_windows']:
                raise ValueError('전체5시장·재로드·padding 포함 예측 예산 초과; 자동 축소하지 않습니다')
            status.update(rows=prepared['rows'],total_windows=len(tasks)+2)
            f.atomic_json(directory/'data.json',{'rows':prepared['rows'],
                'frames_sha256':{m:r.frame_hash(engine,v) for m,v in prepared['raw'].items()},
                'plan':{k:[engine.iso(a),engine.iso(b)] for k,(a,b) in prepared['plan'].items()},
                'entry_times':{k:[engine.iso(i) for i in v] for k,v in prepared['split_times'].items()},
                'foundation_exposures':prepared['foundation_exposures'],'labels_passed_to_model':False,
                'contexts_sha256':f.json_hash([[t['market'],engine.iso(t['stamp']),t['context'].tolist()] for t in tasks])})
            runtime = Runtime(torch,timesfm,engine.np,path,config,counters,guard)
            status.update(stage='predict')
            probe = [tasks[0]['context']]
            first = runtime.forecast(probe)
            f.atomic_json(directory/'reload-probe-before.json',{k:v.tolist() for k,v in first.items()})
            records = []
            for offset in range(0,len(tasks),config['batch_size']):
                guard.check()
                if shutil.disk_usage(work).free < config['min_free_bytes']:
                    raise OSError('예측 중 디스크 여유 부족')
                batch = tasks[offset:offset+config['batch_size']]
                output = runtime.forecast([t['context'] for t in batch])
                for i,task in enumerate(batch):
                    records.append({'market':task['market'],'stamp':task['stamp'],'split':task['split'],
                        'last_close':float(task['context'][-1]), **{k:float(output[k][i]) for k in ('mean','q10','q90')},
                        'full_mean':output['full_mean'][i].tolist()})
                status.update(windows_completed=counters['windows_completed'],gpu_verified=True,device='cuda:0')
            status.update(stage='verify')
            runtime.close()
            runtime = None
            gc.collect()
            guard.check()
            runtime = Runtime(torch,timesfm,engine.np,path,config,counters,guard)
            second = runtime.forecast(probe)
            proof = {'reload_prediction_match':all(engine.np.allclose(first[k],second[k],rtol=1e-5,atol=1e-5) for k in first),
                     'weight_file_unchanged':f.file_hash(path/'model.safetensors',guard) == snapshot['files']['model.safetensors'],
                     'rtol':1e-5,'atol':1e-5,'before_sha256':f.file_hash(directory/'reload-probe-before.json',guard),
                     'after':{k:v.tolist() for k,v in second.items()},'probe_scope':'first_policy_context_only_no_labels'}
            f.atomic_json(directory/'reload-proof.json',proof)
            if not proof['reload_prediction_match'] or not proof['weight_file_unchanged']:
                raise ValueError('실제 재로드 예측 또는 공식 가중치 파일 검증 실패')
            status.update(windows_completed=counters['windows_completed'],checkpoint_verified=True)
            f.atomic_json(directory/'predictions.json', [{**v,'stamp':engine.iso(v['stamp'])} for v in records])
            predictions = {name:{market:engine.pd.DataFrame(net_predictions(engine,
                {k:[v[k] for v in records if v['split']==name and v['market']==market] for k in ('mean','q10','q90')},
                [v['last_close'] for v in records if v['split']==name and v['market']==market],base['paper']),
                index=prepared['split_times'][name]) for market in engine.MARKETS} for name in prepared['plan']}
            frames = {name:execution_frames(engine,prepared['frames'],period,prepared['split_times'][name])
                      for name,period in prepared['plan'].items()}
            status.update(stage='policy')
            policy,trials = choose_policy(lambda p:r.evaluate(engine,frames['rule'],predictions['rule'],p,base['paper'],12,1))
            f.atomic_json(directory/'policy.json',{'policy':policy,'trials':trials,'selection_scope':'rule_only',
                                                   'validation_used_for_selection':False})
            status.update(stage='validation')
            metrics = error_metrics(engine,prepared,records)
            comparisons,details = [],[]
            for cost in (1,2):
                for name in ('foundation','cash','trend','incumbent'):
                    guard.check()
                    if name == 'foundation':
                        result = r.evaluate(engine,frames['validation'],predictions['validation'],policy,base['paper'],12,cost)
                    else:
                        result = r.evaluate_baseline(engine,name,frames['validation'],prepared['incumbent_frames'],
                                                    incumbent,base['paper'],12,cost)
                    summary = {'fold':1,'candidate':name,'cost_multiplier':cost,'net_return':float(result['net_return']),
                        'MDD':float(result['MDD']),'trades':int(result['trades']),
                        'scope':r.incumbent_scope(engine,incumbent['manifest'],prepared['plan']['validation'][0])
                                if name=='incumbent' else 'retrospective_not_independent_final_test'}
                    comparisons.append(summary)
                    details.append({'summary':summary,'trading':result})
            report = {'model':snapshot,'packages':versions,'policy':policy,'comparisons':details,'forecast_metrics':metrics,
                'same_entry_mask_sha256':{m:r.frame_hash(engine,v[['research_entry_allowed']]) for m,v in frames['validation'].items()},
                'counters':counters,'device':'cuda:0','gpu_verified':counters['gpu_verified'],
                'parameter_count':runtime.parameter_count,'compile_requested':False,'fit_calls':0,'promotion':False,
                'historical_quote_extra_cost':None,'scope':'retrospective_not_independent_final_test',
                'seconds':time.monotonic()-started}
            f.atomic_json(directory/'report.json',report)
            manifest = {'run':run,'complete':True,'study':'pretrained_forecast','model_id':MODEL_ID,'revision':REVISION,
                'fit_calls':0,'promotion':False,'candidate_ready':False,'checkpoint_verified':True,'evaluation_performed':True,
                'counters':counters,'incumbent':{'run':incumbent['run'],'files':incumbent['files']},
                'artifacts':{p.relative_to(directory).as_posix():f.file_hash(p,guard) for p in sorted(directory.rglob('*')) if p.is_file()}}
            f.atomic_json(directory/'manifest.json',manifest)
            status.update(state='complete',stage='complete',evaluation_performed=True,comparisons=comparisons,
                          forecast_metrics=metrics,manifest_sha256=f.file_hash(directory/'manifest.json',guard),finished_utc=f.now_iso())
            return status.value
        except BaseException as exc:
            f.atomic_json(directory/'failure.json',{'utc':f.now_iso(),'run':run,'error':str(exc),
                'type':type(exc).__name__,'traceback':traceback.format_exc(),'argv':sys.argv,'cwd':str(Path.cwd()),
                'stage':status.value['stage'],'counters':counters,'fit_calls':0,**f.routing_counters(root)})
            status.update(state='stopped' if isinstance(exc,(KeyboardInterrupt,f.StopRequested)) else 'error',
                          stage='failed',reason='실험 중단; 비공개 원문 보존')
            raise
        finally:
            if runtime is not None:
                runtime.close()
            for sig,previous in old_signals.items():
                signal.signal(sig,previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true',required=True,help='유한 ZERO-SHOT 실험1회')
    parser.add_argument('--run',required=True,help='재실행하지 않을 실험 식별자')
    parser.add_argument('--config',default='11차/benchmark-config.json')
    args = parser.parse_args(argv)
    try:
        value = run_once(ROOT,settings(f.read_json(f.safe_path(ROOT,args.config))),args.run)
        print(json.dumps({k:value[k] for k in ('run','state','windows_completed','total_windows','fit_calls','promotion')},ensure_ascii=False))
        return 0
    except Exception as exc:
        print('사전학습 비교 실행 실패: '+str(exc),file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
