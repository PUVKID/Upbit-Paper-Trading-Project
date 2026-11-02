<div align="center">

# 📈 8차: 고정 자료 반복 학습

검증된 7차 완료 checkpoint에서 출발하여 동일한 고정 train/tune 자료로 `fit()`을 10번 호출합니다. 각 호출은 full fine-tuning 50 step이며 context 192, batch 8, learning rate 0.000001입니다. 500-step 단일 fit, 10 epoch, 독립평가 10회가 아닙니다.

**기술 분류:** Python · 시세 수집 · GPU 연구 · 호가 기반 모의체결

[핵심 기능](#features) · [사용 흐름](#usage) · [자료 구성](#structure) · [준비와 실행](#setup) · [결과 읽기](#results) · [직전 변경](#changes)

</div>

[🏠 프로젝트](../README.md) · [🗂️ 전체 차수](../docs/REVISION_INDEX.md) · [📁 원본 자료](.) · [← 이전: 7차](../7%EC%B0%A8/README.md) · [다음: 9차 →](../9%EC%B0%A8/README.md)

## 🎯 목적과 이 차수의 역할

검증된 7차 완료 checkpoint에서 출발하여 동일한 고정 train/tune 자료로 `fit()`을 10번 호출합니다. 각 호출은 full fine-tuning 50 step이며 context 192, batch 8, learning rate 0.000001입니다. 500-step 단일 fit, 10 epoch, 독립평가 10회가 아닙니다.

<a name="features"></a>

## ✨ 핵심 기능과 보존 범위

| 항목 | 이 차수에서 확인할 내용 |
|---|---|
| 1 | 검증된 7차 완료 checkpoint에서 출발하여 동일한 고정 train/tune 자료로 `fit()`을 10번 호출합니다. 각 호출은 full fine-tuning 50 step이며 context 192, batch 8, learning rate 0.000001입니다. 500-step 단일 fit, 10 epoch, 독립평가 10회가 아닙니다. |
| 2 | 새 독립평가는 수행하지 않습니다. 지표는 null이고 candidate_ready/promote는 false입니다. 실제 성능 향상·수익·방향 정확도는 이 실험으로 확인하지 않습니다. 모델은 미검증 train-only 연구 산출물이며 기존 모의계좌와 실매매에 적용하지 않습니다. |
| 3 | 2026-11-03 RTX 4090의 CUDA에서 5코인 학습 자료 40,998행을 사용해 실제 fit 10회·총 500스텝을 실행했습니다. 첫 fit의 저장 검증 실패를 보존하고 추가 fit 없이 별도 재로드 복구 검사를 거친 뒤 나머지 9회만 추가 학습했습니다. 실제 완료 체크포인트 10개의 파일 해시·가중치 계보·각 50스텝을 대조했습니다. |
| 4 | 학습 손실은 4.521545→3.839211로 감소했으나 고정 튜닝 손실은 7.453165→8.106275로 증가했습니다. 과적합 징후이며 성능 개선으로 판단하지 않습니다. 독립 미래 평가·정확도·수익은 미평가이고 매매 적용도 하지 않았습니다. [회차별 집계와 검증 범위](변경기록.md)를 함께 확인할 수 있습니다. |

## 🔄 입력·처리·출력

| 단계 | 내용 |
|---|---|
| 📥 입력 | 7차 완료 checkpoint와 동일한 고정 train/tune 자료 |
| ⚙️ 처리 | 50스텝 full fine-tuning fit 10회 → 저장/재로드·가중치 계보 확인 |
| 📤 출력 | 10개 완료 checkpoint와 학습/튜닝 손실, 미평가 상태 |

<a name="usage"></a>

## 🖐️ 화면과 사용 흐름

1. 7차 완료 checkpoint·고정 자료의 해시와 학습 계약을 확인합니다.
2. 각 fit 50스텝·문맥192·배치8·학습률0.000001의 10회 계보를 읽습니다.
3. 학습 손실과 고정 튜닝 손실을 나눠 확인합니다.
4. candidate_ready/promote=false와 독립평가 미실시를 확인하고 현재 학습 중지는 9차에서 읽습니다.

<a name="structure"></a>

## 📁 실제 자료 구성

| 경로 | 자료 종류 |
|---|---|
| [README.md](README.md) | MD 파일 |
| [train-only-config.json](train-only-config.json) | JSON 파일 |
| [train_only_repeats.py](train_only_repeats.py) | PY 파일 |
| [변경기록.md](%EB%B3%80%EA%B2%BD%EA%B8%B0%EB%A1%9D.md) | MD 파일 |
| [연구계약.md](%EC%97%B0%EA%B5%AC%EA%B3%84%EC%95%BD.md) | MD 파일 |
| [전체비교.md](%EC%A0%84%EC%B2%B4%EB%B9%84%EA%B5%90.md) | MD 파일 |

## 🧩 주요 모듈과 코드 읽기

| 파일 | 원문에 정의된 역할·구성 |
|---|---|
| [train_only_repeats.py](train_only_repeats.py) | 고정 train/tune으로 50-step fit을 10회 이어가는 미검증 연구 실행기입니다. |

<a name="setup"></a>

## 🚀 설치 환경과 실행·사용법

requirements와 engine-config.json, CUDA·GPU·DB 및 자료 저장 경로를 확인합니다. 실제 거래 계정·운영 DB·모델·개인 로그는 공개 코드와 별도입니다. 외부 수집에는 네트워크·디스크 여유·체크섬과 최신성 조건이 필요합니다.

8차는 이미 수행한 고정 자료 반복 학습의 보관본입니다. 500-step 단일 fit·10 epoch·독립평가10회가 아닙니다. 이후 9차에서 자동 학습을 중지했으며 새 적합을 시작하려면 별도 명시 재개와 검증 계약이 필요합니다.

### 📖 이 차수 원래 안내의 세부 조건

### 실제 결과

2026-11-03 RTX 4090의 CUDA에서 5코인 학습 자료 40,998행을 사용해 실제 fit 10회·총 500스텝을 실행했습니다. 첫 fit의 저장 검증 실패를 보존하고 추가 fit 없이 별도 재로드 복구 검사를 거친 뒤 나머지 9회만 추가 학습했습니다. 실제 완료 체크포인트 10개의 파일 해시·가중치 계보·각 50스텝을 대조했습니다.

학습 손실은 4.521545→3.839211로 감소했으나 고정 튜닝 손실은 7.453165→8.106275로 증가했습니다. 과적합 징후이며 성능 개선으로 판단하지 않습니다. 독립 미래 평가·정확도·수익은 미평가이고 매매 적용도 하지 않았습니다. [회차별 집계와 검증 범위](%EB%B3%80%EA%B2%BD%EA%B8%B0%EB%A1%9D.md)를 함께 확인할 수 있습니다.

### 재실행 정책

최소판은 중간 resume 대신 **예약 후 재실행 거부** 정책입니다. `8차/work/request.json` 또는 `8차/work/runs/*`가 있으면 같은 batch와 다른 이름의 batch 모두 거부합니다. 실패 checkpoint를 자동 삭제하거나 처음 가중치로 돌아가지 않습니다. 실패 전 완료된 회차의 manifest·checkpoint와 미완료 회차를 그대로 보존합니다. 미완료 fit을 자동으로 다시 수행하지 않습니다.

예약 전 준비 실패는 fit을 시작하지 않았으므로 재검사할 수 있습니다. 예약 후 중단된 실험의 별도 복구·새 실행 승인은 부모가 원본을 보존한 상태에서 판단해야 하며, 이 CLI에는 우회·초기화 옵션이 없습니다.



### 산출물

- `work/status.json`: 8차 진행 상태만 기록합니다. API 이름은 `training_repetition`이며 runner가 대시보드를 수정하지는 않습니다.
- `work/runs/<batch>/request.json`, `inputs.json`: 고정 기간·입력 해시·실제 내부 입력 tuple. 원시 입력은 공개하지 않습니다.
- `work/runs/<batch>/rep-01`부터 `rep-10`: 각 fit의 attempt, checkpoint, 완료 manifest. 직전 manifest 해시와 실제 가중치 해시를 연결합니다.
- `work/runs/<batch>/completion.json`: 실제 완료 manifest 수 0~10과 목록. 10개일 때만 complete입니다.
- `work/knowledge.json`: **미검증 train-only 전용** 포인터. 7차 knowledge와 분리됩니다.
- `work/evidence`: 실패 원문과 구현 검사 기록. 공개 대상이 아닙니다.

상세 경계는 [연구계약.md](%EC%97%B0%EA%B5%AC%EA%B3%84%EC%95%BD.md)를 따릅니다. 공개 가능한 범위는 소스·설정·테스트·문서이며 work의 시세·모델·운영 원문은 제외합니다.

### 검증과 실제 결과

2026-11-03 실제 RTX 4090 CUDA에서 fit 10회·각 50스텝·총 500스텝이 완료됐습니다. 사용한 학습 행은 40,998개이며 전체 시계열을 10 epoch 순회했다는 뜻은 아닙니다. 10개 checkpoint 파일의 전체 SHA256·manifest 계보·연속 가중치 해시와 Trainer의 각 global_step=50을 대조했습니다. 7차 연구 포인터는 바뀌지 않았습니다.

| 회차 | 학습 손실 | 고정 튜닝 손실 |
| --- | ---: | ---: |
| 1 | 4.521545 | 7.453165 |
| 2 | 4.430647 | 7.467674 |
| 3 | 4.347526 | 7.518273 |
| 4 | 4.269733 | 7.571614 |
| 5 | 4.193344 | 7.647175 |
| 6 | 4.120640 | 7.735616 |
| 7 | 4.048472 | 7.847617 |
| 8 | 3.978817 | 7.916269 |
| 9 | 3.907470 | 8.032920 |
| 10 | 3.839211 | 8.106275 |

학습 손실 감소와 반대로 튜닝 손실이 증가해 과적합 징후로 해석합니다. 이는 고정 튜닝 자료의 모델 loss이며 독립 미래 평가·방향 정확도·거래 수익이 아닙니다. 성능 개선이나 매매 승격을 주장하지 않습니다.

### 실패·복구·실행 소스

첫 50스텝 학습 뒤 재로드 예측 또는 파일 결합 검사에서 실패했습니다. 원래 저장 전 probe가 없어 최초 실패 원인을 소급 확정할 수 없습니다. 무학습 진단에서 같은 checkpoint의 두 재로드 예측은 완전히 같았고 원본 9개 파일 해시도 불변이었습니다. BF16과 FP32 실행 경로 차이에서는 60개 값 중 35개가 기존 허용오차를 벗어났으며 FP32로 복원하면 다시 일치했습니다.

허용오차를 늘리지 않고 추론 경로를 FP32로 맞추며, 이후 회차는 삭제 전 probe와 분리된 검증 결과를 보존합니다. 첫 fit은 소비한 상태로 원래 실패와 원문을 보존하고 별도의 무학습 재로드 복구 근거로만 승계했습니다. 원래 저장 전후 검사를 PASS로 바꾸지 않았습니다. 추가 fit은 9회로 제한해 실제 총 10회를 지켰습니다.

실제 GPU 실행 소스는 수리72 SHA256 `b73f7ac918b236133ddce654556a07313ae73d6569023a5ea66698dfb32494ca`입니다. 실행이 종료된 뒤 Windows 검사에서 드러난 메타데이터 경로 키의 슬래시 표현을 정규화하는 수리73을 별도로 적용합니다. 실제 GPU72와 최종 소스73의 검증은 구분하며 73으로 추가 학습했다고 주장하지 않습니다. 실행기 단계명 오타와 API 계약도 회귀 검사로 맞췄습니다.

API focused 45개·순수 상태 회귀 12개·브라우저 통제 검사 36항목과 사전 독립 검토를 실제 GPU 실행과 구분합니다. 최종 소스의 전체 검사·독립 검토·공개 원격 확인은 별도 최종 영수증으로 대조합니다. 원고·논문·PDF는 이번 변경에 없습니다.

### ⚙️ 소스에서 제공하는 CLI 옵션

| 진입 소스 | 정의된 옵션 이름 |
|---|---|
| [train_only_repeats.py](train_only_repeats.py) | `--once`, `--batch`, `--config`, `--recover-first` |

각 옵션의 기본값·필수값·부작용은 해당 main과 인자 정의를 함께 확인합니다.


<details>
<summary>🔬 함수 역할·기본값·입력 거부 조건의 세부 참조</summary>

다음 항목은 이 자료의 Python 소스에 명시된 선언입니다. 전체 분기와 호출 순서는 각 행의 코드 위치에서 이어서 읽을 수 있습니다.

### [train_only_repeats.py](train_only_repeats.py)


| 설정·옵션 | 코드의 선언 |
|---|---|
| [`--once`](train_only_repeats.py#L566) | `action='store_true'; required=True; help='고정 10회 fit을 한 batch로 실행'` |
| [`--batch`](train_only_repeats.py#L567) | `required=True; help='8차의 고유 batch 식별자; 예약 이후 재실행 거부'` |
| [`--config`](train_only_repeats.py#L568) | `default='8차/train-only-config.json'; help='프로젝트 내부 고정 설정 JSON'` |
| [`--recover-first`](train_only_repeats.py#L569) | `action='store_true'; help='승인된 동일 실패 batch의 첫 fit을 재학습 없이 복구하고 최대 9회 추가'` |
| [`ROOT`](train_only_repeats.py#L17) | `정의=Path(__file__).resolve().parents[1]` |
| [`FROZEN`](train_only_repeats.py#L21) | `정의={'repetitions': 10, 'steps': 50, 'context_length': 192, 'batch_size': 8,           'learning_rate': 1e-6, 'prediction_length': 12}` |
| [`DEFAULTS`](train_only_repeats.py#L23) | `정의={**FROZEN, 'max_seconds': 18000, 'min_free_bytes': 20_000_000_000,             'max_rows_per_market': 50000, 'seed': 42}` |
| [`METRICS`](train_only_repeats.py#L25) | `정의=('validation_mae', 'baseline_mae', 'direction_accuracy', 'net_return', 'trades')` |
| [`ORIGIN_SOURCE_SHA256`](train_only_repeats.py#L26) | `정의='ea361c6da394fc2ff128615d583e8079eca0eeb43bc23e3c37613ddde1425fc9'` |

| 거부 조건 | 오류 메시지 |
|---|---|
| [`not isinstance(value, dict) or set(value)-set(DEFAULTS)`](train_only_repeats.py#L30) | 8차 설정의 형식 또는 키가 유효하지 않습니다 |
| [`config['max_seconds'] &gt; 18000 or config['min_free_bytes'] &lt; 20_000_000_000`](train_only_repeats.py#L40) | 5시간 상한 또는 20GB 여유 기준을 약화할 수 없습니다 |
| [`periods['train'][1] &gt; periods['tune'][0] or periods['tune'][1] &gt; periods['validation'][0]`](train_only_repeats.py#L107) | 부모 train/tune/validation 기간이 겹칩니다 |
| [`any(foundation.stamp(a) &gt;= foundation.stamp(b) for a, b in protected)`](train_only_repeats.py#L126) | 금지 기간의 순서가 유효하지 않습니다 |
| [`len(values) != count or any(len(row) != 12 for row in values)`](train_only_repeats.py#L155) | 학습 context probe의 예측 형태가 다릅니다 |
| [`any(isinstance(v, bool) or not math.isfinite(float(v)) for row in values for v in row)`](train_only_repeats.py#L157) | 학습 context probe가 유한하지 않습니다 |
| [`model is not pipeline.model`](train_only_repeats.py#L170) | 승인한 단일 GPU 모델 이외의 병렬 래퍼는 지원하지 않습니다 |
| [`str(model.dtype) != 'torch.float32' or hasattr(model, '_original_forward')             or hasattr(model.forward, '__wrapped__')`](train_only_repeats.py#L174) | FP32 추론 정규화 또는 AMP 래퍼 제거를 확인하지 못했습니다 |
| [`before != after`](train_only_repeats.py#L178) | 추론 정규화가 파라미터 해시를 변경했습니다 |
| [`(batch/'recovery-first').exists()`](train_only_repeats.py#L226) | 첫 회차 복구 예약이 존재하여 재실행을 거부합니다 |
| [`parent is None or parent['run'] != request['parent_run']             or parent['manifest_sha256'] != request['parent_manifest_sha256']             or attempt.get('previous_manifest_sha256') != parent['manifest_sha256']`](train_only_repeats.py#L262) | 원래 7차 부모 계보가 다릅니다 |
| [`foundation.checkpoint_files(checkpoint, guard) != {             k: v for k, v in foundation.checkpoint_files(batch/'rep-01/checkpoint-50', guard).items()             if k in ('config.json', 'model.safetensors')}`](train_only_repeats.py#L267) | 최종 checkpoint와 실제 step50 가중치가 다릅니다 |
| [`foundation.json_hash(records) != request['data']['inputs_sha256'] or set(records) != {'train', 'tune'}`](train_only_repeats.py#L272) | 고정 원래 입력의 해시 또는 자료 구분이 다릅니다 |
| [`foundation.json_hash(contexts) != request['data']['probe_contexts_sha256']`](train_only_repeats.py#L303) | 원래 train context 해시가 다릅니다 |
| [`len(failures) != 1`](train_only_repeats.py#L313) | 원래 첫 실패 증거를 유일하게 확인하지 못했습니다 |
| [`shutil.disk_usage(work).free &lt; config['min_free_bytes']`](train_only_repeats.py#L326) | 디스크 여유가 부족합니다. 자동 삭제는 하지 않습니다 |
| [`not isinstance(run, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run)`](train_only_repeats.py#L339) | batch는 영문·숫자·밑줄·하이픈 1~80자여야 합니다 |
| [`type(value['step']) is not int or not self.value['step'] &lt;= value['step'] &lt;= 50                 or (value['loss'] is not None and not math.isfinite(float(value['loss'])))`](train_only_repeats.py#L90) | 회차 내 step 또는 손실이 유효하지 않습니다 |
| [`a &gt;= b or b &gt; dt.datetime.now(foundation.UTC)`](train_only_repeats.py#L104) | 부모의 고정 기간이 역행하거나 미래에 있습니다 |
| [`request['data']['plan'][name] != parent['manifest']['plan'][name] or set(records[name]) != set(foundation.MARKETS)`](train_only_repeats.py#L282) | 원래 고정 기간 또는 시장 구성이 다릅니다 |
| [`not recover_first and ((work/'request.json').exists() or any((work/'runs').glob('*')))`](train_only_repeats.py#L349) | 8차 batch 예약이 존재하여 재실행을 거부합니다. 자동 재학습·초기화는 없습니다 |
| [`previous is None`](train_only_repeats.py#L295) | 복구 입력 시장의 시계열이 비었습니다 |
| [`count != request['data']['used_by_market'][market][name+'_rows']`](train_only_repeats.py#L298) | 복구 입력 행 수가 원래 계획과 다릅니다 |
| [`value.get('stage') != 'verify' or value.get('settings') != config                     or value.get('error') != '저장 전후 train context 예측 또는 파일 해시가 다릅니다'`](train_only_repeats.py#L309) | 원래 실패 증거가 승인한 첫 회차와 다릅니다 |
| [`foundation.file_hash(directory/'original-status.json', guard) != origin['original_files'][status_key]`](train_only_repeats.py#L358) | 원래 상태의 바이트 보존을 확인하지 못했습니다 |
| [`parent is None`](train_only_repeats.py#L380) | 검증된 7차 완료 부모가 없습니다. 원본 모델 초기화는 금지합니다 |
| [`runtime.evidence.get('device') != 'cuda:0'`](train_only_repeats.py#L405) | 실제 CUDA 전용 런타임이 아닙니다 |
| [`len(parts) != 1 or parts[0] != bars or len(bars) &lt; 204                         or (previous is not None and bars[0][0] &lt;= previous)`](train_only_repeats.py#L290) | 복구 입력이 보호·기간·공백 기준을 위반했습니다 |
| [`after == previous_weights or after != origin['diagnostic']['weights_sha256']`](train_only_repeats.py#L429) | 원래 seed 변화 또는 무학습 진단 가중치 결합이 다릅니다 |
| [`before != previous_weights`](train_only_repeats.py#L471) | 직전 완료 가중치를 실제로 이어받지 못했습니다 |
| [`before == after or proxy.value['step'] != 50 or proxy.value['loss'] is None`](train_only_repeats.py#L477) | 50-step fit·유한 손실·실제 가중치 변화 증거가 부족합니다 |
| [`probe['parameter_sha256'] != after`](train_only_repeats.py#L483) | 저장 전 정규화된 가중치 해시가 다릅니다 |

</details>

<a name="results"></a>

## 📊 처리 결과와 출력 읽기

자료 크기·고유 아카이브 크기·DB 크기를 구분하고 준비/학습/승격 상태를 함께 확인합니다. 모의 손익은 실제 주문 성과가 아닙니다. GPU 사용률·처리량 개선도 수익 개선과 별개입니다.

## 🛠️ 제약과 문제 해결

| 상황 | 확인할 내용 |
|---|---|
| 연구가 대기함 | 자료량·최근 분봉 커버리지·최신성·GPU 사전 조건을 확인합니다. 대기 상태를 임의 학습 성공으로 바꾸지 않습니다. |
| 외부 자료 검증 실패 | 공식 아카이브·체크섬·원문 체결 형식·완료봉을 확인합니다. 보간이나 기존 봉 덮어쓰기로 우회하지 않습니다. |
| 차수 폴더만 실행할 수 없음 | 루트 공통 엔진·설정·검사와 차수 연결 파일의 실제 import 경로를 확인합니다. |

<a name="changes"></a>

## 🔍 직전 자료와 전체 비교

7차에서 8차로 이어지는 실제 파일 비교입니다.

[📄 전체 비교·자료 범위](%EC%A0%84%EC%B2%B4%EB%B9%84%EA%B5%90.md)

추가·수정·삭제, 주석·공백·개행과 바이너리 크기·해시는 연결된 원래 비교 자료에서 확인합니다.
