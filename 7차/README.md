<div align="center">

<a name="-사전학습-시계열-연구"></a>
# 사전학습 시계열 연구

공개 사전학습 모델의 가중치를 코인 시계열에 추가 학습하는 별도 연구 트랙입니다. 기존 XGBoost 모델과 5코인 현물 가상계좌는 그대로 유지합니다. 모델 이름이나 일반 시계열 벤치마크 순위만으로 업비트 수익성을 단정하지 않습니다.

**기술 분류:** Python · 시세 수집 · GPU 연구 · 호가 기반 모의체결

[핵심 기능](#features) · [사용 흐름](#usage) · [자료 구성](#structure) · [준비와 실행](#setup) · [결과 읽기](#results) · [직전 변경](#changes)

</div>

[프로젝트](../README.md) · [전체 차수](../docs/REVISION_INDEX.md) · [원본 자료](.) · [← 이전: 6차](../6%EC%B0%A8/README.md) · [다음: 8차 →](../8%EC%B0%A8/README.md)

<a name="-목적과-이-차수의-역할"></a>
## 목적과 이 차수의 역할

공개 사전학습 모델의 가중치를 코인 시계열에 추가 학습하는 별도 연구 트랙입니다. 기존 XGBoost 모델과 5코인 현물 가상계좌는 그대로 유지합니다. 모델 이름이나 일반 시계열 벤치마크 순위만으로 업비트 수익성을 단정하지 않습니다.

<a name="features"></a>

<a name="-핵심-기능과-보존-범위"></a>
## 핵심 기능과 보존 범위

| 항목 | 이 차수에서 확인할 내용 |
|---|---|
| 1 | 공개 사전학습 모델의 가중치를 코인 시계열에 추가 학습하는 별도 연구 트랙입니다. 기존 XGBoost 모델과 5코인 현물 가상계좌는 그대로 유지합니다. 모델 이름이나 일반 시계열 벤치마크 순위만으로 업비트 수익성을 단정하지 않습니다. |
| 2 | 승인된 월요일 PBL 디자인의 밝은 화이트·옅은 민트, 나눔고딕 700, 좌측 탐색과 현재 작업 중심 구성을 적용합니다. 수집·정리·학습·검증·적용 흐름은 실제 Three.js 형상과 현재 상태로 표현하며 정지 이미지를 기능 화면으로 사용하지 않습니다. 다크/라이트와 모션 제어, 기존 거래·계좌·진단 접근을 유지합니다. |
| 3 | RTX 4090의 CUDA에서 5코인 학습 행 40,998개·문맥 192봉·배치 8·50스텝의 추가 학습을 완료했습니다. 가중치 변화와 저장 후 재로딩·예측 일치를 확인했습니다. [실행 계약](연구계약.md)과 [화면 구성](프런트엔드_변경기록_초안.md)을 함께 확인할 수 있습니다. |
| 4 | 후속 평가 20개 창의 상대 MAE는 **0.210743%**, 같은 종가 유지 기준선은 **0.204550%**, 마지막 예측 지평의 방향 일치율은 **20%**였습니다. 기준선보다 오차가 커 성능 개선을 확인하지 못했습니다. 이는 기존 XGBoost의 3클래스 정확도와 다르며 거래 수익도 아닙니다. 매매 평가·적용은 하지 않았습니다. |

<a name="-입력처리출력"></a>
## 입력·처리·출력

| 단계 | 내용 |
|---|---|
| 입력 | 5코인 완료 분봉과 공개 사전학습 시계열 모델 |
| 처리 | 고정 자료 준비 → CUDA 추가 학습 50스텝 → 저장/재로드 확인 → 후속 20개 창 평가 |
| 출력 | 추가 학습 checkpoint·예측 평가 지표·실제 상태 대시보드 |

<a name="usage"></a>

<a name="️-화면과-사용-흐름"></a>
## 화면과 사용 흐름

1. 연구계약과 자료·시점 경계, 기존 XGBoost 모의매매와의 분리를 확인합니다.
2. 학습 행·문맥·배치·스텝과 저장/재로딩 검사를 읽습니다.
3. 20개 후속 평가 창에서 종가 유지 기준선과 오차·방향 지표를 비교합니다.
4. 학습 완료와 매매 적용을 구분하고 현재 중지 상태는 9차를 확인합니다.

<a name="structure"></a>

<a name="-실제-자료-구성"></a>
## 실제 자료 구성

| 경로 | 자료 종류 |
|---|---|
| [README.md](README.md) | MD 파일 |
| [foundation-config.json](foundation-config.json) | JSON 파일 |
| [research-requirements.txt](research-requirements.txt) | TXT 파일 |
| [shadow_research.py](shadow_research.py) | PY 파일 |
| [변경기록.md](%EB%B3%80%EA%B2%BD%EA%B8%B0%EB%A1%9D.md) | MD 파일 |
| [연구계약.md](%EC%97%B0%EA%B5%AC%EA%B3%84%EC%95%BD.md) | MD 파일 |
| [전체비교.md](%EC%A0%84%EC%B2%B4%EB%B9%84%EA%B5%90.md) | MD 파일 |
| [프런트엔드_변경기록_초안.md](%ED%94%84%EB%9F%B0%ED%8A%B8%EC%97%94%EB%93%9C_%EB%B3%80%EA%B2%BD%EA%B8%B0%EB%A1%9D_%EC%B4%88%EC%95%88.md) | MD 파일 |

<a name="-주요-모듈과-코드-읽기"></a>
## 주요 모듈과 코드 읽기

| 파일 | 원문에 정의된 역할·구성 |
|---|---|
| [shadow_research.py](shadow_research.py) | 고정 Chronos-2의 완료 가중치를 이어 학습하는 독립 연구 전용 실행기. |

<a name="setup"></a>

<a name="-설치-환경과-실행사용법"></a>
## 설치 환경과 실행·사용법

requirements와 engine-config.json, CUDA·GPU·DB 및 자료 저장 경로를 확인합니다. 실제 거래 계정·운영 DB·모델·개인 로그는 공개 코드와 별도입니다. 외부 수집에는 네트워크·디스크 여유·체크섬과 최신성 조건이 필요합니다.

7차는 사전학습 시계열 모델의 별도 연구 트랙입니다. 원래 가중치·CUDA 환경·데이터와 실행 계약이 필요합니다. 이후 9차에서 새 학습을 중지했으므로 과거 안내를 현재 자동 재개 명령으로 사용하지 않습니다.

<a name="-이-차수-원래-안내의-세부-조건"></a>
### 이 차수 원래 안내의 세부 조건

### 구조

- 첫 후보는 Apache-2.0의 `amazon/chronos-2`입니다. 공개 모델 revision을 고정하고 실제 CUDA에서 적합 여부를 확인합니다.
- 첫 실행은 5코인 업비트 5분봉의 제한된 기간을 사용합니다. 실제 학습 행·기간·설정·손실값을 기록하며 수집한 외부 50GB 전체를 이 모델에 적합했다는 뜻이 아닙니다.
- 완료된 가중치를 저장·재로드해 검증하고 다음 회차의 부모 모델로 이어 사용합니다. 모델 가중치 연속 학습과 optimizer까지 포함한 중간 훈련 재개는 구분합니다.
- 연구 후보와 운용 모델은 별개입니다. 예측 오차 감소만으로 매매 모델을 교체하지 않습니다. 수익 평가가 없는 값은 미평가로 표시합니다.
- 원시 자료 삭제는 아직 수행하지 않습니다. 체크포인트와 운영 입력·독립 평가·지식 보존 경계를 검증한 뒤 별도 적용해야 합니다.

### 첫 실제 결과

RTX 4090의 CUDA에서 5코인 학습 행 40,998개·문맥 192봉·배치 8·50스텝의 추가 학습을 완료했습니다. 가중치 변화와 저장 후 재로딩·예측 일치를 확인했습니다. [실행 계약](%EC%97%B0%EA%B5%AC%EA%B3%84%EC%95%BD.md)과 [화면 구성](%ED%94%84%EB%9F%B0%ED%8A%B8%EC%97%94%EB%93%9C_%EB%B3%80%EA%B2%BD%EA%B8%B0%EB%A1%9D_%EC%B4%88%EC%95%88.md)을 함께 확인할 수 있습니다.

후속 평가 20개 창의 상대 MAE는 **0.210743%**, 같은 종가 유지 기준선은 **0.204550%**, 마지막 예측 지평의 방향 일치율은 **20%**였습니다. 기준선보다 오차가 커 성능 개선을 확인하지 못했습니다. 이는 기존 XGBoost의 3클래스 정확도와 다르며 거래 수익도 아닙니다. 매매 평가·적용은 하지 않았습니다.

이번 단일 회차 실행기는 완료 가중치를 다음 회차에 이어 쓸 수 있지만 미사용 평가 자료가 충분해야 합니다. 기존 6차 연속 연구와 모의매매는 별도 서비스로 계속됩니다. 새 모델을 검증 없이 자동 매매에 연결하거나 동일 평가 구간을 반복 사용하지 않습니다.

최강·정확도 100%·수익 개선은 확인되지 않았습니다. 모델·시세 데이터·계좌 장부·내부 로그는 공개하지 않습니다. 원시 자료 삭제와 중간 optimizer 재개도 이번 구현의 완료 범위가 아닙니다.

### 완료 저장 경계

독립 제어 주입에서 완료 포인터가 저장된 뒤 표시 상태 저장만 실패하면, 정상 부모 checkpoint와 실패 화면이 모순되는 문제를 재현했습니다. 완료 스냅샷을 먼저 저장해 manifest·포인터와 SHA256으로 결합하고, 후속 상태 쓰기는 유한하게 복구합니다. 지속 실패 시에도 검증 가능한 완료와 저장 경고를 구분합니다. API는 같은 회차의 작은 완료 JSON 결합을 확인하며 전체 모델을 매번 다시 읽지 않습니다. 다른 새 회차의 대기·오류·학습 상태는 과거 완료로 덮어쓰지 않습니다.

불변 완료 payload의 시각을 캐시에서도 유지해 정상 완료 뒤 매번 최종화만 반복하는 문제도 막았습니다. 기존 실제 GPU checkpoint는 수정하지 않았고, 새 저장 경계 코드에서 다시 CUDA로 로드해 같은 가중치임을 확인했습니다. 새 GPU 학습·평가 구간 재사용은 하지 않았습니다. 저장 경계는 임시 대역의 실패 주입 회귀로 별도 검증하며 기존 실제 학습 증거와 구분합니다.

### 실제 학습 관측

- 공식 고정 Chronos-2 가중치 477,930,472바이트의 SHA256을 원본 LFS 값과 대조했습니다.
- 초기 50GB 완료 기록과 다음 2GB 수집 준비 상태의 혼동을 수리했습니다. 현재 추가 수집 대기 때문에 정상 완료된 초기 연구를 미완료로 만들지 않습니다.
- 입력 256봉과 도지코인 평가 구간 최장 연속 250봉의 불일치를 확인했습니다. 문맥을 192봉으로 정한 새 실험 설정을 사용하고 누락 가격을 보간하지 않았습니다.
- 실제 학습: 5코인 업비트 5분봉, 학습 40,998행, 50스텝, 배치 8, 초기 학습률 0.000001, CUDA 0의 RTX 4090. 완료 가중치 변화·저장·재로드·예측 일치 검증을 통과했습니다.
- 상대 MAE 0.210743%, 종가 유지 상대 MAE 0.204550%, 방향 일치율 20개 창에서 20%. 기준선보다 나빠 성능 개선을 확인하지 못했습니다. 수익률·완료 거래는 미평가이며 기존 운용 모델을 교체하지 않았습니다.
- 미사용 평가 예약을 첫 예측 전에 저장하고 실패 회차도 다음 노출 상한에 포함합니다. 실제 Trainer 복제 모델의 첫 optimizer step 전 CUDA 검사와 손상 checkpoint의 원본 초기화 거부를 유지합니다.

추가 독립 검토·공식 영수증·공개 반영·원격 확인은 실제 완료 기록과 구분합니다. 앞선 실패는 후속 통과로 지우지 않습니다. 새 모델은 단일 회차 연구 도구이며 기존 연속 연구·모의매매와 별도입니다.

<a name="️-소스에서-제공하는-cli-옵션"></a>
### 소스에서 제공하는 CLI 옵션

| 진입 소스 | 정의된 옵션 이름 |
|---|---|
| [shadow_research.py](shadow_research.py) | `--config`, `--once` |

각 옵션의 기본값·필수값·부작용은 해당 main과 인자 정의를 함께 확인합니다.


<details>
<summary>함수 역할·기본값·입력 거부 조건의 세부 참조</summary>

다음 항목은 이 자료의 Python 소스에 명시된 선언입니다. 전체 분기와 호출 순서는 각 행의 코드 위치에서 이어서 읽을 수 있습니다.

### [shadow_research.py](shadow_research.py)

| 정의 | 소스에 적힌 역할 |
|---|---|
| [`NotReady`](shadow_research.py#L42) | 새로운 미사용 평가 자료가 아직 충분하지 않습니다. |
| [`StopRequested`](shadow_research.py#L46) | 사용자의 중단 신호를 보존합니다. |
| [`AlreadyRunning`](shadow_research.py#L50) | 기존 실행의 상태 파일을 덮어쓰지 않습니다. |

| 설정·옵션 | 코드의 선언 |
|---|---|
| [`--config`](shadow_research.py#L898) | `default='7차/foundation-config.json'; help='프로젝트 내부 연구 설정 JSON'` |
| [`--once`](shadow_research.py#L899) | `action='store_true'; required=True; help='유한한 연구 한 회차만 실행'` |
| [`ROOT`](shadow_research.py#L23) | `정의=Path(__file__).resolve().parents[1]` |
| [`MODEL_ID`](shadow_research.py#L24) | `정의='amazon/chronos-2'` |
| [`REVISION`](shadow_research.py#L25) | `정의='29ec3766d36d6f73f0696f85560a422f50e8498c'` |
| [`MODEL_SHA256`](shadow_research.py#L26) | `정의='ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42'` |
| [`MARKETS`](shadow_research.py#L27) | `정의=('KRW-BTC', 'KRW-SOL', 'KRW-ETH', 'KRW-XRP', 'KRW-DOGE')` |
| [`UTC`](shadow_research.py#L28) | `정의=dt.timezone.utc` |
| [`BAR`](shadow_research.py#L29) | `정의=dt.timedelta(minutes=5)` |
| [`PROTECTED`](shadow_research.py#L30) | `정의=[('2026-10-19T00:00:00Z', '2026-10-20T00:00:00Z')]` |

| 거부 조건 | 오류 메시지 |
|---|---|
| [`not isinstance(value, dt.datetime) or value.utcoffset() != dt.timedelta(0)`](shadow_research.py#L56) | 시각은 시간대가 명시된 UTC여야 합니다 |
| [`not isinstance(value, dict) or set(value)-set(DEFAULTS)`](shadow_research.py#L66) | 지원하지 않는 연구 설정입니다 |
| [`result['model_id'] != MODEL_ID or result['revision'] != REVISION`](shadow_research.py#L69) | 공식 고정 모델과 revision만 허용합니다 |
| [`type(lr) not in (float, int) or not math.isfinite(lr) or not 0 &lt; lr &lt;= 0.001`](shadow_research.py#L80) | 학습률은 유한한 양수이며 0.001 이하여야 합니다 |
| [`p.is_absolute() or PureWindowsPath(str(relative)).drive or '\\' in str(relative)             or '..' in p.parts or not p.parts`](shadow_research.py#L99) | 프로젝트 내부 상대 경로만 허용합니다 |
| [`not target.resolve().is_relative_to(root)`](shadow_research.py#L107) | 경로가 프로젝트 밖으로 나갑니다 |
| [`path.is_symlink()`](shadow_research.py#L123) | JSON 대상이 연결 파일입니다 |
| [`readiness.get('schema_version') != 1 or type(size) is not int or size &lt; 50_000_000_000             or type(verified) is not int or verified &lt; size`](shadow_research.py#L157) | 검증·적재된 초기 50GB 원본 근거가 부족합니다 |
| [`len(batches) != 1`](shadow_research.py#L162) | 유일한 초기 연구 완료 기록이 필요합니다 |
| [`not isinstance(run, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', run)`](shadow_research.py#L172) | 초기 연구 실행 ID가 잘못됐습니다 |
| [`hashes != record.get('model_sha256') or len(hashes) != report.get('training', {}).get('total_candidates')`](shadow_research.py#L201) | 초기 완료 모델의 실제 해시가 다릅니다 |
| [`path.exists()`](shadow_research.py#L212) | 기존 평가 예약을 덮어쓸 수 없습니다 |
| [`start &gt;= end or stamp(data['exposure_end']) != end`](shadow_research.py#L215) | 평가 예약의 기간과 자료 기간이 다릅니다 |
| [`sys.platform != 'linux'`](shadow_research.py#L292) | 실행과 운영 DB 접근은 Linux 컨테이너 전용입니다 |
| [`end &lt;= validation_start`](shadow_research.py#L335) | 부모 모델에 노출되지 않은 미래 평가 구간이 아직 없습니다 |
| [`not windows`](shadow_research.py#L374) | 공백과 보호 기간을 제외한 연속 평가 창이 부족합니다 |
| [`sys.platform != 'linux'`](shadow_research.py#L384) | 운영 SQLite는 Linux 안에서만 읽습니다 |
| [`not path.is_file()`](shadow_research.py#L402) | 기존 보호 시험 감사 DB가 필요합니다 |
| [`end &gt; dt.datetime.now(UTC)`](shadow_research.py#L414) | 미래 완료 시각까지 자료를 읽을 수 없습니다 |
| [`not predictions or len(predictions) != len(targets) or len(targets) != len(origins)`](shadow_research.py#L449) | 예측·정답·기준 가격의 표본 수가 다릅니다 |
| [`'config.json' not in files or 'model.safetensors' not in files`](shadow_research.py#L475) | 완료 checkpoint의 설정 또는 safetensors가 없습니다 |
| [`directory.name != REVISION`](shadow_research.py#L483) | 고정 revision 경로가 아닙니다 |
| [`not all((directory/name).is_file() for name in ('config.json','model.safetensors','download-evidence.json'))`](shadow_research.py#L485) | 부모 다운로드가 완료된 실제 파일과 증거가 필요합니다 |
| [`records.get('model_id') != MODEL_ID or records.get('revision') != REVISION`](shadow_research.py#L488) | 부모 다운로드의 고정 모델 증거가 다릅니다 |
| [`files['model.safetensors']['sha256'] != MODEL_SHA256`](shadow_research.py#L496) | 공식 LFS 모델 해시가 다릅니다 |
| [`not isinstance(digest,str) or not re.fullmatch(r'[0-9a-f]{64}',digest) or (expected is not None and expected != digest)`](shadow_research.py#L514) | 완료 스냅샷의 해시 결합이 유효하지 않습니다 |
| [`file_hash(path) != digest`](shadow_research.py#L518) | 완료 스냅샷 파일 해시가 다릅니다 |
| [`file_hash(directory/'manifest.json') != pointer['manifest_sha256']`](shadow_research.py#L544) | 커밋된 완료 manifest를 확인할 수 없습니다 |
| [`target.is_symlink()`](shadow_research.py#L105) | 경로 구성 요소에 연결이 있습니다 |
| [`self.event.is_set() or self.stop.exists()`](shadow_research.py#L284) | 연구 중단이 요청됐습니다 |
| [`time.monotonic() &gt;= self.deadline`](shadow_research.py#L286) | 연구 실행의 유한 시간 상한을 초과했습니다 |
| [`value['candidate_ready'] and not (value['checkpoint_verified'] and value['state'] == 'complete')`](shadow_research.py#L323) | 검증된 완료 연구 후보만 준비 상태가 될 수 있습니다 |
| [`previous is not None and moment &lt;= previous`](shadow_research.py#L349) | 중복 또는 역행하는 시계열입니다 |
| [`stamp(a) &gt;= stamp(b)`](shadow_research.py#L407) | 보호 시험 기간이 유효하지 않습니다 |
| [`not predicted or len(predicted) != len(actual) or not math.isfinite(origin) or origin &lt;= 0                 or any(not math.isfinite(v) for v in [*predicted, *actual])`](shadow_research.py#L453) | 예측과 평가 값이 유효하지 않습니다 |
| [`path.stat().st_size != record['bytes'] or file_hash(path, guard) != record['sha256']`](shadow_research.py#L494) | 사전학습 모델 파일의 크기 또는 해시가 다릅니다 |
| [`any((work/'runs').glob('*/manifest.json'))`](shadow_research.py#L614) | 완료 이력이 있는데 연구 포인터가 없습니다. 원본 초기화를 거부합니다 |
| [`not isinstance(run, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', run) or run in seen`](shadow_research.py#L622) | 연구 계보의 ID 또는 순환이 유효하지 않습니다 |
| [`file_hash(path, guard) != item.get('manifest_sha256')`](shadow_research.py#L628) | 부모 manifest 해시가 다릅니다 |
| [`not packages.is_dir() or any((packages/name).exists() for name in ('torch', 'numpy', 'pandas'))`](shadow_research.py#L661) | 격리 패키지 경로가 없거나 공유 핵심 패키지를 가리고 있습니다 |
| [`not Path(torch.__file__).resolve().is_relative_to('/opt/venvs/torch')`](shadow_research.py#L670) | 지정된 시스템 torch가 아닙니다 |
| [`not torch.cuda.is_available()`](shadow_research.py#L675) | CUDA가 없어 연구를 거부합니다. CPU 대체는 없습니다 |
| [`model is None`](shadow_research.py#L698) | 실제 학습 모델 장치를 확인할 수 없습니다 |
| [`abs(pipeline.quantiles[median]-0.5) &gt; 1e-8`](shadow_research.py#L753) | 모델에 중앙값 분위수가 없습니다 |
| [`not isinstance(name, str) or Path(name).name != name or '\\' in name or not name.endswith('.ubj') or name in hashes`](shadow_research.py#L198) | 초기 연구 모델 경로·중복이 잘못됐습니다 |
| [`digest is not None and found.get(path.parent.name) != digest`](shadow_research.py#L255) | 완료 또는 실패 회차의 평가 예약이 누락·변경됐습니다 |
| [`not math.isfinite(value) or value &lt;= 0`](shadow_research.py#L360) | 시계열 종가는 유한한 양수여야 합니다 |
| [`len(rows) &gt; limit`](shadow_research.py#L425) | 읽기 행 상한을 초과했습니다. 자료를 몰래 축소하지 않습니다 |
| [`latest is None and item.get('completion_sha256') != value.get('completion_sha256')`](shadow_research.py#L641) | 연구 포인터와 완료 스냅샷 해시가 다릅니다 |
| [`not self.torch.isfinite(tensor).all().item()`](shadow_research.py#L710) | 모델 가중치에 비유한 값이 있습니다 |
| [`shutil.disk_usage(work).free &lt; config['min_free_bytes']`](shadow_research.py#L782) | 디스크 여유가 부족합니다. 자료 삭제나 자동 정리는 하지 않습니다 |
| [`before == after or status.value['step'] != config['steps'] or status.value['loss'] is None`](shadow_research.py#L828) | 실제 가중치 변화·완료 step·손실 근거가 없습니다 |
| [`runtime.weights_hash(reloaded) != after`](shadow_research.py#L841) | 저장 후 재로드한 가중치 해시가 다릅니다 |
| [`not runtime.np.allclose(probe, repeated, rtol=1e-5, atol=1e-5, equal_nan=False)`](shadow_research.py#L844) | 저장 전후 예측이 일치하지 않습니다 |
| [`shutil.disk_usage(work).free &lt; config['min_free_bytes']`](shadow_research.py#L852) | 완료 저장 전 디스크 여유가 부족합니다 |
| [`confirmed is None`](shadow_research.py#L868) | 완료 포인터 커밋을 확인할 수 없습니다 |
| [`(Path(folder)/name).is_symlink()`](shadow_research.py#L92) | 연구 자료 안의 연결 파일을 허용하지 않습니다 |
| [`not math.isfinite(float(loss))`](shadow_research.py#L732) | 학습 손실이 유한하지 않습니다 |

</details>

<a name="results"></a>

<a name="-처리-결과와-출력-읽기"></a>
## 처리 결과와 출력 읽기

자료 크기·고유 아카이브 크기·DB 크기를 구분하고 준비/학습/승격 상태를 함께 확인합니다. 모의 손익은 실제 주문 성과가 아닙니다. GPU 사용률·처리량 개선도 수익 개선과 별개입니다.

<a name="️-제약과-문제-해결"></a>
## 제약과 문제 해결

| 상황 | 확인할 내용 |
|---|---|
| 연구가 대기함 | 자료량·최근 분봉 커버리지·최신성·GPU 사전 조건을 확인합니다. 대기 상태를 임의 학습 성공으로 바꾸지 않습니다. |
| 외부 자료 검증 실패 | 공식 아카이브·체크섬·원문 체결 형식·완료봉을 확인합니다. 보간이나 기존 봉 덮어쓰기로 우회하지 않습니다. |
| 차수 폴더만 실행할 수 없음 | 루트 공통 엔진·설정·검사와 차수 연결 파일의 실제 import 경로를 확인합니다. |

<a name="changes"></a>

<a name="-직전-자료와-전체-비교"></a>
## 직전 자료와 전체 비교

6차에서 7차로 이어지는 실제 파일 비교입니다.

[전체 비교·자료 범위](%EC%A0%84%EC%B2%B4%EB%B9%84%EA%B5%90.md)

추가·수정·삭제, 주석·공백·개행과 바이너리 크기·해시는 연결된 원래 비교 자료에서 확인합니다.
