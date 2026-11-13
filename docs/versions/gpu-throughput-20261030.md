<div align="center">

<a name="-2026-10-30-gpu-처리량-개선"></a>
# 2026-10-30 GPU 처리량 개선

기존 공개 자료의 차수 표기는 `기본`이며 번호를 임의로 소급 부여하지 않습니다. 이번 추가 기록은 날짜로 구분합니다. 활성 실행 폴더를 복제하지 않고 기존 서비스에 적용하며 이전 코드는 Git 이력과 로컬 소스 전용 압축본으로 보존합니다.

**기술 분류:** Python · 시세 수집 · GPU 연구 · 호가 기반 모의체결

[핵심 기능](#features) · [사용 흐름](#usage) · [자료 구성](#structure) · [준비와 실행](#setup) · [결과 읽기](#results) · [직전 변경](#changes)

</div>

[프로젝트](../../README.md) · [전체 차수](../REVISION_INDEX.md) · [원본 자료](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89)

<a name="-목적과-이-차수의-역할"></a>
## 목적과 이 차수의 역할

기존 공개 자료의 차수 표기는 `기본`이며 번호를 임의로 소급 부여하지 않습니다. 이번 추가 기록은 날짜로 구분합니다. 활성 실행 폴더를 복제하지 않고 기존 서비스에 적용하며 이전 코드는 Git 이력과 로컬 소스 전용 압축본으로 보존합니다.

<a name="features"></a>

<a name="-핵심-기능과-보존-범위"></a>
## 핵심 기능과 보존 범위

| 항목 | 이 차수에서 확인할 내용 |
|---|---|
| 1 | 기존 공개 자료의 차수 표기는 `기본`이며 번호를 임의로 소급 부여하지 않습니다. 이번 추가 기록은 날짜로 구분합니다. 활성 실행 폴더를 복제하지 않고 기존 서비스에 적용하며 이전 코드는 Git 이력과 로컬 소스 전용 압축본으로 보존합니다. |
| 2 | `CandleStore.read`는 완료 시각 범위 인덱스와 같은 순서로 조회해 임시 정렬을 제거합니다. 반환 행의 완료 시각 간격이 시간봉과 다르면 거부합니다. 기존 요청 범위·제한 수·최종 시작 시각 오름차순은 유지합니다. |
| 3 | `gpu_matrix_scope`는 같은 시장·같은 입력 객체·같은 열·같은 가중치 설정의 후보 묶음에서만 두 DMatrix를 재사용합니다. 범위 밖 호출과 다른 시장·가중치는 공유하지 않으며 종료 시 참조를 해제합니다. 해당 범위의 입력은 읽기 전용이며 제자리 변경은 지원하지 않습니다. 기존 CUDA 사전 검사와 매 반복 검사도 그대로 수행합니다. |
| 4 | `market_feature_loader`는 하나의 CPU 작업자로 다음 시장 특징만 선행 준비합니다. 누적 자료를 임의로 축소하지 않으며 학습·검증 범위는 같습니다. 준비 오류는 상위 호출로 전달하고, 본 작업 오류와 동시에 발생한 오류 원문도 예외에 남깁니다. 진행 중 SQL을 강제 종료하는 기능은 없습니다. |

<a name="-입력처리출력"></a>
## 입력·처리·출력

| 단계 | 내용 |
|---|---|
| 입력 | 시세·호가·완료 분봉·외부 아카이브·연구 설정 |
| 처리 | 검증·저장 → 특징 준비·연구 → 정책 평가 → 모의체결·화면 |
| 출력 | 자료 준비 상태·연구 지표·모의계좌 장부·대시보드 |

<a name="usage"></a>

<a name="️-화면과-사용-흐름"></a>
## 화면과 사용 흐름

1. 모의매매와 GPU 연구·외부자료 수집의 독립 트랙을 확인합니다.
2. 설정·시점 경계·PAPER_ONLY와 실제 자료 준비 상태를 읽습니다.
3. 해당 요청의 수집·연구 조건이 충족될 때만 지정된 한 회차를 수행합니다.
4. 대시보드와 저장 지표에서 완료·대기·오류를 구분하고 기존 장부·모델을 보존합니다.

<a name="structure"></a>

<a name="-실제-자료-구성"></a>
## 실제 자료 구성

| 경로 | 자료 종류 |
|---|---|
| [docs/](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/docs) | 하위 자료 폴더 |
| [tests/](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/tests) | 하위 자료 폴더 |
| [tools/](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/tools) | 하위 자료 폴더 |
| [web/](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/web) | 하위 자료 폴더 |
| [README.md](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/README.md) | MD 파일 |
| [dashboard.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py) | PY 파일 |
| [engine-config.json](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine-config.json) | JSON 파일 |
| [engine.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py) | PY 파일 |
| [execution_probe.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py) | PY 파일 |
| [healthcheck.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py) | PY 파일 |
| [package.json](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/package.json) | JSON 파일 |
| [pnpm-lock.yaml](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/pnpm-lock.yaml) | YAML 파일 |
| [pnpm-workspace.yaml](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/pnpm-workspace.yaml) | YAML 파일 |
| [requirements.txt](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/requirements.txt) | TXT 파일 |
| [service.json](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/service.json) | JSON 파일 |
| [supervisor.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py) | PY 파일 |
| [upbit_auto_trader_v5.ipynb](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/upbit_auto_trader_v5.ipynb) | IPYNB 파일 |
| [운영.ps1](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/%EC%9A%B4%EC%98%81.ps1) | PS1 파일 |

<a name="-주요-모듈과-코드-읽기"></a>
## 주요 모듈과 코드 읽기

| 파일 | 원문에 정의된 역할·구성 |
|---|---|
| [engine.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py) | 공개 시세만 사용하는 GPU 학습 및 현물 모의투자 엔진. |
| [dashboard.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py) | 로컬 전용 읽기 전용 현물 모의거래 대시보드입니다. |
| [execution_probe.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py) | 실제 공개 호가의 체결 경로를 전략 계좌와 별도로 검증합니다. |
| [healthcheck.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py) | 실제 운용 파일과 시세·현물 계좌의 불변 조건을 주기적으로 검사합니다. |
| [supervisor.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py) | 수집·학습·현물 모의거래를 독립 실행하고 상태를 보존합니다. |
| [upbit_auto_trader_v5.ipynb](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/upbit_auto_trader_v5.ipynb) | 노트북 16개 셀 |
| [tests/test_artifact_scope.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/tests/test_artifact_scope.py) | Linux 실행 의존성을 프로젝트 직접 구현 소스와 구분하는지 검사합니다. |
| [tests/test_capital_contract.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/tests/test_capital_contract.py) | 3차 원금·기존 장부 경계와 실제 대시보드 필드를 검사합니다. |
| [tests/test_collection_recovery.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/tests/test_collection_recovery.py) | 손상된 원본 행을 보존하면서 과거 수집을 계속하는지 검사합니다. |

<a name="setup"></a>

<a name="-설치-환경과-실행사용법"></a>
## 설치 환경과 실행·사용법

requirements와 engine-config.json, CUDA·GPU·DB 및 자료 저장 경로를 확인합니다. 실제 거래 계정·운영 DB·모델·개인 로그는 공개 코드와 별도입니다. 외부 수집에는 네트워크·디스크 여유·체크섬과 최신성 조건이 필요합니다.

이 차수 폴더만이 독립 실행 환경인지 먼저 확인합니다. 3차 외부자료 모듈과 4·5차 제어는 루트 engine.py·engine-config.json 및 tests와 연결됩니다. 실행 명령과 운영 경로는 해당 차수 README의 계약을 따릅니다.

<a name="-이-보관본의-의존성-목록"></a>
### 이 보관본의 의존성 목록

[requirements.txt](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/requirements.txt)의 실제 항목입니다. 외부 실행 도구와 GPU 드라이버는 별도로 준비합니다.

```text
numpy>=1.26.4,<3
pandas>=2.2.3,<3
xgboost==3.0.5
```

<a name="️-소스에서-제공하는-cli-옵션"></a>
### 소스에서 제공하는 CLI 옵션

| 진입 소스 | 정의된 옵션 이름 |
|---|---|
| [dashboard.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py) | `--host`, `--port` |
| [engine.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py) | `--config`, `--backfill`, `--once`, `--pages`, `--paper-start`, `--final` |
| [execution_probe.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py) | `--config` |
| [supervisor.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py) | `--config` |

각 옵션의 기본값·필수값·부작용은 해당 main과 인자 정의를 함께 확인합니다.


<details>
<summary>함수 역할·기본값·입력 거부 조건의 세부 참조</summary>

다음 항목은 이 자료의 Python 소스에 명시된 선언입니다. 전체 분기와 호출 순서는 각 행의 코드 위치에서 이어서 읽을 수 있습니다.

### [dashboard.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py)

| 정의 | 소스에 적힌 역할 |
|---|---|
| [`diagnostic`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L73) | 허용 키·깊이·항목 수·총 문자수로 로컬 진단을 제한합니다. |
| [`execution_probe`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L105) | 본 계좌와 합산하지 않는 공개호가 경로 검사만 투영합니다. |
| [`project`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L123) | 허용한 필드의 단순 값만 추출합니다. |
| [`SnapshotAdapter`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L289) | 경로를 외부 입력으로 받지 않는 제한된 스냅숏 어댑터입니다. |

| 설정·옵션 | 코드의 선언 |
|---|---|
| [`--host`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L554) | `default="127.0.0.1"` |
| [`--port`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L555) | `type=int; default=8765` |
| [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L15) | `정의=Path(__file__).resolve().parent` |
| [`MAX_JSON_BYTES`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L16) | `정의=4 * 1024 * 1024` |
| [`COINS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L17) | `정의=("BTC", "SOL", "ETH", "XRP", "DOGE")` |
| [`MAX_DIAGNOSTIC_DEPTH`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L67) | `정의=8` |
| [`MAX_DIAGNOSTIC_ITEMS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L68) | `정의=100` |
| [`MAX_DIAGNOSTIC_NODES`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L69) | `정의=1500` |
| [`MAX_DIAGNOSTIC_CHARS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L70) | `정의=24000` |

### [engine.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py)

| 정의 | 소스에 적힌 역할 |
|---|---|
| [`sweep_book`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L582) | 매수 amount는 수수료 포함 원화, 매도 amount는 현물 수량입니다. |
| [`MarketUniverseTransitionError`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1431) | 검사를 통과한 기존 네 시장 모델도 다섯 시장의 운용에는 사용할 수 없습니다. |

| 설정·옵션 | 코드의 선언 |
|---|---|
| [`--config`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1641) | `type=Path; default=None` |
| [`--backfill`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1644) | `action="store_true"` |
| [`--once`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1645) | `action="store_true"` |
| [`--pages`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1646) | `type=int` |
| [`--once`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1648) | `action="store_true"` |
| [`--once`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1650) | `action="store_true"` |
| [`--paper-start`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1651) | `` |
| [`--final`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1653) | `action="store_true"` |
| [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L27) | `정의=Path(__file__).resolve().parent` |
| [`RUNTIME_WORK`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L28) | `정의=(ROOT / "work").resolve()` |
| [`MARKETS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L29) | `정의=("KRW-BTC", "KRW-SOL", "KRW-ETH", "KRW-XRP", "KRW-DOGE")` |
| [`TIMEFRAMES`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L30) | `정의=(1, 3, 5, 10, 15, 30, 60, 240, "day")` |
| [`PAPER_ONLY`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L31) | `정의=True` |
| [`TRAINING_ETA`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L32) | `정의=0.05` |
| [`EXECUTION_POLICY`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L33) | `정의="horizon_or_risk_v2"` |
| [`PROTECTED_HOLDOUT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L34) | `정의=("2026-10-19T00:00:00Z", "2026-10-20T00:00:00Z")` |

| 거부 조건 | 오류 메시지 |
|---|---|
| [`sorted(config["markets"]) != sorted(MARKETS)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L106) | 모의 대상은 BTC/SOL/ETH/XRP/DOGE 다섯 종목으로 고정됩니다 |
| [`len(set(map(str, config["timeframes"]))) != len(config["timeframes"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L108) | 중복 시간봉입니다 |
| [`any(str(tf) not in set(map(str, TIMEFRAMES)) for tf in config["timeframes"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L110) | 지원하지 않는 시간봉입니다 |
| [`str(config["base_timeframe"]) not in set(map(str, config["timeframes"]))`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L112) | 기본 시간봉이 수집 목록에 없습니다 |
| [`tr["gap_bars"] &lt; tr["horizon"] + 1`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L125) | 분할 간격은 다음봉 체결을 포함한 horizon+1 이상이어야 합니다 |
| [`not tr["candidate_depths"] or any(not isinstance(d, int) or not 1 &lt;= d &lt;= 10 for d in tr["candidate_depths"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L127) | 후보 깊이는 1~10 범위입니다 |
| [`not 0 &lt; paper["initial_cash"] or paper["min_order"] &lt; 5000`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L132) | 초기 자본과 최소 주문금액을 확인하세요 |
| [`paper["max_allocation"] &gt; paper["max_total_allocation"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L134) | 코인별 배분이 전체 한도를 초과합니다 |
| [`isinstance(tr["class_weight_cap"], bool) or not math.isfinite(tr["class_weight_cap"]) or             not 1 &lt;= tr["class_weight_cap"] &lt;= 3`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L140) | 클래스 가중치 상한은 1~3입니다 |
| [`not tr["horizon_candidates"] or any(isinstance(h, bool) or not isinstance(h, int) or not 1 &lt;= h &lt;= 288                                            for h in tr["horizon_candidates"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L147) | 지평 후보는 1~288봉 정수입니다 |
| [`not tr["threshold_candidates"] or any(isinstance(t, bool) or not math.isfinite(t) or not 0 &lt; t &lt; 1                                               for t in [tr["label_threshold"], *tr["threshold_candidates"]])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L150) | 비용 차감 후 라벨 임계값이 유효하지 않습니다 |
| [`collection["request_interval"] &lt; 0.11 or not 1 &lt;= collection["retries"] &lt;= 8`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L153) | 공유 요청 간격은 0.11초 이상, 재시도는 1~8회입니다 |
| [`min(collection["recent_pages"], collection["backfill_pages_per_cycle"], collection["poll_seconds"], paper["poll_seconds"]) &lt;= 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L155) | 수집 페이지와 대기 시간은 양수여야 합니다 |
| [`sys.platform == "win32" and path.resolve().is_relative_to(RUNTIME_WORK)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L165) | 운영 SQLite는 도커 내부에서만 접근하십시오 |
| [`endpoint != "orderbook" and not endpoint.startswith("candles/")`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L251) | 공개 캔들/호가 읽기만 허용됩니다 |
| [`not np.isfinite(values).all() or (values[:, :4] &lt;= 0).any() or (values[:, 4] &lt; 0).any()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L293) | 캔들 값이 유효하지 않습니다 |
| [`((frame.high &lt; frame[["open", "close", "low"]].max(axis=1)) &#124;         (frame.low &gt; frame[["open", "close"]].min(axis=1))).any()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L295) | OHLC 범위가 유효하지 않습니다 |
| [`count &lt;= 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L328) | 페이지 수는 양수여야 합니다 |
| [`horizon &lt;= 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L402) | 예측 지평은 양수여야 합니다 |
| [`bar_length &lt;= pd.Timedelta(0) or not (lengths == bar_length).all()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L410) | 라벨 봉 길이가 일관되지 않습니다 |
| [`isinstance(sequence, bool) or not isinstance(sequence, int) or sequence &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L463) | 연구 회차 커서가 손상되었습니다; 초기화하지 않습니다 |
| [`p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p &lt; 0).any() or (p.sum(axis=1) &lt;= 0).any()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L484) | 세 클래스 확률 배열이 유효하지 않습니다 |
| [`not len(y) or len(y) != len(pred)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L561) | 전체 표본 평가를 위한 행이 없습니다 |
| [`side not in ("buy", "sell") or not math.isfinite(amount) or amount &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L584) | 현물 모의체결 입력 오류 |
| [`not math.isfinite(volatility) or volatility &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L612) | 현재 변동성이 유효하지 않습니다 |
| [`common is None or len(common) &lt; 2`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L820) | 공통계좌 백테스트에 공통 완료 봉이 부족합니다 |
| [`not xgb.build_info().get("USE_CUDA", False)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L915) | 설치한 XGBoost가 CUDA를 지원하지 않습니다 |
| [`not tr["holdout_start"] or not tr["holdout_end"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1060) | training.holdout_start/holdout_end를 고정 UTC 시각으로 명시해야 합니다 |
| [`he &lt;= hs`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1063) | 잠금 시험 종료는 시작 이후여야 합니다 |
| [`abs((he - hs).total_seconds() / 86400 - tr["holdout_days"]) &gt; 1e-6`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1065) | holdout_days와 명시 시험 기간이 일치하지 않습니다 |
| [`final and live`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1244) | 실시간 학습은 잠금 시험을 평가할 수 없습니다 |
| [`final and utc(None) &lt; plan["holdout"][1]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1252) | 잠금 시험 기간이 끝나지 않았습니다 |
| [`Path(pointer["run"]).name != pointer["run"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1441) | 모델 포인터 경로 오류 |
| [`stable_hash(manifest) != pointer["manifest_hash"] or not manifest["paper_only"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1445) | 완성 모델 manifest가 일치하지 않습니다 |
| [`manifest.get("run") != pointer["run"] or (package_markets != set(MARKETS) and not known_transition)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1457) | 완성 모델의 실행 ID/다섯 코인 구성이 일치하지 않습니다 |
| [`not isinstance(columns, list) or not columns or len(set(columns)) != len(columns)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1460) | 완성 모델의 피처 목록이 유효하지 않습니다 |
| [`not target.is_relative_to((ROOT / "work").resolve())`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L116) | 유동 산출물은 프로젝트 work 내부에만 저장합니다 |
| [`not isinstance(row["candle_date_time_utc"], str)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L279) | 캔들 시각이 유효하지 않습니다 |
| [`pd.isna(stamp)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L282) | 캔들 시각이 유효하지 않습니다 |
| [`not math.isfinite(price + size) or price &lt;= 0 or size &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L591) | 호가 가격/잔량 오류 |
| [`state is not None and (not isinstance(state, dict) or                                   state.get("initial_cash") != initial or state.get("paper_only") is not True)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L626) | 복원 계좌의 초기 원금 또는 모의 전용 표시가 설정과 다릅니다 |
| [`self.state["cash"] &lt; 0 or any(v["qty"] &lt; 0 for v in self.state["positions"].values())`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L635) | 현물 계좌에 음수 잔액/공매도가 있습니다 |
| [`any(m not in MARKETS for m in prices) or any(v not in (0, 1, 2) for v in signals.values())`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L655) | 지원하지 않는 현물 시장/신호입니다 |
| [`not all(math.isfinite(metrics[key]) for key in ("logloss", "net_return", "MDD"))`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1115) | 모델 승격 비교 지표가 유한하지 않습니다 |
| [`market not in MARKETS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1463) | 지원하지 않는 모델 시장 |
| [`not package["models"] or not len(package["models"]) == len(package["weights"]) == len(package["gpu"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1465) | 완성 모델의 후보/가중치/적합 증거 수가 다릅니다 |
| [`any(not math.isfinite(w) or w &lt; 0 for w in package["weights"]) or not math.isclose(sum(package["weights"]), 1, abs_tol=1e-6)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1467) | 완성 모델의 확률 가중치가 유효하지 않습니다 |
| [`set(quotes) != set(config["markets"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1572) | 다섯 코인의 공개 호가가 모두 필요합니다 |
| [`not isinstance(payload, list)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L260) | 공개 시세 응답이 배열이 아닙니다 |
| [`db.execute("SELECT 1 FROM final_tests WHERE start&lt;? AND end&gt;?", (end, start)).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1020) | 이 시험 기간은 이미 소비 또는 평가 시도되었습니다. 반복 튜닝 금지 |
| [`db.execute("SELECT 1 FROM exposures WHERE start&lt;? AND end&gt;?", (end, start)).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1022) | 시험 기간이 이전 학습/검증에 노출되어 미사용 시험으로 평가할 수 없습니다 |
| [`db.execute("SELECT 1 FROM final_tests WHERE start&lt;? AND end&gt;?", (iso(end), iso(start))).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1028) | 이 시험 기간은 이미 소비 또는 평가 시도되었습니다. 반복 튜닝 금지 |
| [`db.execute("SELECT 1 FROM exposures WHERE start&lt;? AND end&gt;?", (iso(end), iso(start))).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1030) | 시험 기간이 이전 학습/검증에 노출되어 미사용 시험으로 평가할 수 없습니다 |
| [`pending is None or market != expected`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1217) | 특징 선행 준비의 시장 순서가 다릅니다 |
| [`columns is not None and cols != columns`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1288) | 코인별 피처 스키마가 다릅니다 |
| [`Path(filename).name != filename or not (directory / filename).is_file()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1470) | 완성 모델 파일이 없습니다 |
| [`any(set(values) != set(MARKETS) for values in (frames, probabilities, rules))`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1147) | 종목 전환 후보는 다섯 시장 모두의 후속 평가가 필요합니다 |
| [`cooldowns[market] is None`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L737) | 재진입 보류 만료시각이 없습니다 |
| [`pd.isna(until)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L740) | 재진입 보류 만료시각이 유효하지 않습니다 |

### [execution_probe.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py)


| 설정·옵션 | 코드의 선언 |
|---|---|
| [`--config`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L60) | `type=Path; default=engine.ROOT / 'engine-config.json'` |
| [`SCOPE`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L13) | `정의='체결 경로 진단 · 전략 성과 아님'` |

| 거부 조건 | 오류 메시지 |
|---|---|
| [`account.state['positions']`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L48) | 별도 진단 계좌에 미청산 잔량이 남았습니다 |
| [`len(payload) != 1 or payload[0].get('market') != market or not payload[0].get('orderbook_units')`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L28) | 진단 시장의 호가가 없습니다 |
| [`observed and now &lt;= observed[-1]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L35) | 호가 관측 시각이 증가하지 않았습니다 |
| [`not math.isfinite(bid + ask) or not 0 &lt; bid &lt;= ask`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L39) | 호가 가격 또는 스프레드가 유효하지 않습니다 |

### [healthcheck.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py)


| 설정·옵션 | 코드의 선언 |
|---|---|
| [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L18) | `정의=Path(__file__).resolve().parent` |
| [`INTEGRITY_TIMEOUT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L19) | `정의=1800` |
| [`INTEGRITY_MAX_AGE`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L20) | `정의=21600` |
| [`INTEGRITY_RETRY`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L21) | `정의=900` |

| 거부 조건 | 오류 메시지 |
|---|---|
| [`database_identity(database) != identity`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L63) | 검사 중 저장소 파일이 교체되었습니다 |
| [`time.monotonic() &gt;= deadline`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L42) | 전체 무결성 검사 시간 제한 초과 |
| [`result['state'] != 'ok' or result['database'] != database_identity(database)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L86) | 전체 검사 결과와 저장소가 일치하지 않습니다 |

### [supervisor.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py)


| 설정·옵션 | 코드의 선언 |
|---|---|
| [`--config`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L176) | `type=Path; default=ROOT / "service.json"` |
| [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L19) | `정의=Path(__file__).resolve().parent` |
| [`STOP`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L20) | `정의=threading.Event()` |

| 거부 조건 | 오류 메시지 |
|---|---|
| [`config.get("initial_capital") != 10_000_000 or paper.get("initial_cash") != config.get("initial_capital")             or config.get("market_mode") != "spot-paper" or config.get("notifications") is not False`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L162) | 원금 1,000만원·현물 모의거래·알림 없음 설정을 확인하십시오. |
| [`start_at.tzinfo is None or engine_start.tzinfo is None or start_at != engine_start`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L167) | 서비스와 엔진의 시작 시각·시간대가 일치해야 합니다. |
| [`config["trainer"]["interval_seconds"] != 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L169) | 성공한 학습 회차의 고정 대기는 0초여야 합니다. |
| [`not STOP.is_set()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L128) | 작업 시간 제한 초과 |

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

이 자료의 보존 범위와 기준 파일을 연결합니다.

[전체 비교·자료 범위](../revision-audit/GPU%EC%B2%98%EB%A6%AC%EB%9F%89.md)

추가·수정·삭제, 주석·공백·개행과 바이너리 크기·해시는 연결된 원래 비교 자료에서 확인합니다.
