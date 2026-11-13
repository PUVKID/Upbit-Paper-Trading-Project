<div align="center">

<a name="-2026-10-30-gpu-처리량-개선"></a>
# 2026-10-30 GPU 처리량 개선

기존 공개 자료의 차수 표기는 `기본`이며 번호를 임의로 소급 부여하지 않습니다. 이번 추가 기록은 날짜로 구분합니다. 활성 실행 폴더를 복제하지 않고 기존 서비스에 적용하며 이전 코드는 Git 이력과 로컬 소스 전용 압축본으로 보존합니다.

**기술 분류:** Python · 시세 수집 · GPU 연구 · 호가 기반 모의체결

[핵심 기능](#features) · [사용 흐름](#usage) · [자료 구성](#structure) · [준비와 실행](#setup) · [결과 읽기](#results) · [직전 변경](#changes)

</div>
<!-- reading-levels:start -->
[LEVEL 1 - Beginner](#reading-level-1) · [LEVEL 2 - Explorer](#reading-level-2) · [LEVEL 3 - Practitioner](#reading-level-3) · [LEVEL 4 - Advanced](#reading-level-4) · [LEVEL 5 - Expert](#reading-level-5)

프로젝트 전체의 설명은 [LEVEL 1 - Beginner](../%EC%9D%BD%EA%B8%B0%EC%95%88%EB%82%B4/%EB%A0%88%EB%B2%A81.md)부터 [LEVEL 5 - Expert](../%EC%9D%BD%EA%B8%B0%EC%95%88%EB%82%B4/%EB%A0%88%EB%B2%A85.md)까지 같은 범위로 연결됩니다.

<a id="reading-level-1"></a>
## LEVEL 1 - Beginner

### LEVEL 1 · 이 자료가 하는 일과 달라진 점

기존 공개 자료의 차수 표기는 `기본`이며 번호를 임의로 소급 부여하지 않습니다. 이번 추가 기록은 날짜로 구분합니다. 활성 실행 폴더를 복제하지 않고 기존 서비스에 적용하며 이전 코드는 Git 이력과 로컬 소스 전용 압축본으로 보존합니다.

### LEVEL 1 · 실제 동작을 따라 읽기

기존 공개 자료의 차수 표기는 `기본`이며 번호를 임의로 소급 부여하지 않습니다. 이번 추가 기록은 날짜로 구분합니다. 활성 실행 폴더를 복제하지 않고 기존 서비스에 적용하며 이전 코드는 Git 이력과 로컬 소스 전용 압축본으로 보존합니다.

`CandleStore.read`는 완료 시각 범위 인덱스와 같은 순서로 조회해 임시 정렬을 제거합니다. 반환 행의 완료 시각 간격이 시간봉과 다르면 거부합니다. 기존 요청 범위·제한 수·최종 시작 시각 오름차순은 유지합니다.

`gpu_matrix_scope`는 같은 시장·같은 입력 객체·같은 열·같은 가중치 설정의 후보 묶음에서만 두 DMatrix를 재사용합니다. 범위 밖 호출과 다른 시장·가중치는 공유하지 않으며 종료 시 참조를 해제합니다. 해당 범위의 입력은 읽기 전용이며 제자리 변경은 지원하지 않습니다. 기존 CUDA 사전 검사와 매 반복 검사도 그대로 수행합니다.

`market_feature_loader`는 하나의 CPU 작업자로 다음 시장 특징만 선행 준비합니다. 누적 자료를 임의로 축소하지 않으며 학습·검증 범위는 같습니다. 준비 오류는 상위 호출로 전달하고, 본 작업 오류와 동시에 발생한 오류 원문도 예외에 남깁니다. 진행 중 SQL을 강제 종료하는 기능은 없습니다.

CLI는 명령어와 옵션을 입력해 프로그램을 실행하는 방식입니다. 체크섬은 받은 바이트가 형식상 손상되지 않았는지 대조하는 값입니다. 모의 운용은 계산상 계좌와 거래를 다루므로 실제 계좌 체결과 구분합니다.

입력 — 시세·호가·완료 분봉·외부 아카이브·연구 설정

처리 — 검증·저장 → 특징 준비·연구 → 정책 평가 → 모의체결·화면

출력 — 자료 준비 상태·연구 지표·모의계좌 장부·대시보드

### LEVEL 1 · 준비하고 확인할 것

requirements와 engine-config.json, CUDA·GPU·DB 및 자료 저장 경로를 확인합니다. 실제 거래 계정·운영 DB·모델·개인 로그는 공개 코드와 별도입니다. 외부 수집에는 네트워크·디스크 여유·체크섬과 최신성 조건이 필요합니다.

이 차수 폴더만이 독립 실행 환경인지 먼저 확인합니다. 3차 외부자료 모듈과 4·5차 제어는 루트 engine.py·engine-config.json 및 tests와 연결됩니다. 실행 명령과 운영 경로는 해당 차수 README의 계약을 따릅니다.

#### LEVEL 1 · 이 보관본의 의존성 목록

[requirements.txt](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/requirements.txt)의 실제 항목입니다. 외부 실행 도구와 GPU 드라이버는 별도로 준비합니다.

```text
numpy>=1.26.4,<3
pandas>=2.2.3,<3
xgboost==3.0.5
```

#### LEVEL 1 · 소스에서 제공하는 CLI 옵션

**진입 소스:** [dashboard.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py)

**정의된 옵션 이름:** `--host`, `--port`

**진입 소스:** [engine.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py)

**정의된 옵션 이름:** `--config`, `--backfill`, `--once`, `--pages`, `--paper-start`, `--final`

**진입 소스:** [execution_probe.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py)

**정의된 옵션 이름:** `--config`

**진입 소스:** [supervisor.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py)

**정의된 옵션 이름:** `--config`


각 옵션의 기본값·필수값·부작용은 해당 main과 인자 정의를 함께 확인합니다.


<details>
<summary>함수 역할·기본값·입력 거부 조건의 세부 참조</summary>

다음 항목은 이 자료의 Python 소스에 명시된 선언입니다. 전체 분기와 호출 순서는 각 행의 코드 위치에서 이어서 읽을 수 있습니다.

#### LEVEL 1 · [dashboard.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py)

| 내용 |
| --- |
| **정의:** [`diagnostic`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L73)<br>**소스에 적힌 역할:** 허용 키·깊이·항목 수·총 문자수로 로컬 진단을 제한합니다. |
| **정의:** [`execution_probe`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L105)<br>**소스에 적힌 역할:** 본 계좌와 합산하지 않는 공개호가 경로 검사만 투영합니다. |
| **정의:** [`project`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L123)<br>**소스에 적힌 역할:** 허용한 필드의 단순 값만 추출합니다. |
| **정의:** [`SnapshotAdapter`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L289)<br>**소스에 적힌 역할:** 경로를 외부 입력으로 받지 않는 제한된 스냅숏 어댑터입니다. |

| 내용 |
| --- |
| **설정·옵션:** [`--host`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L554)<br>**코드의 선언:** `default="127.0.0.1"` |
| **설정·옵션:** [`--port`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L555)<br>**코드의 선언:** `type=int; default=8765` |
| **설정·옵션:** [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L15)<br>**코드의 선언:** `정의=Path(__file__).resolve().parent` |
| **설정·옵션:** [`MAX_JSON_BYTES`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L16)<br>**코드의 선언:** `정의=4 * 1024 * 1024` |
| **설정·옵션:** [`COINS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L17)<br>**코드의 선언:** `정의=("BTC", "SOL", "ETH", "XRP", "DOGE")` |
| **설정·옵션:** [`MAX_DIAGNOSTIC_DEPTH`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L67)<br>**코드의 선언:** `정의=8` |
| **설정·옵션:** [`MAX_DIAGNOSTIC_ITEMS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L68)<br>**코드의 선언:** `정의=100` |
| **설정·옵션:** [`MAX_DIAGNOSTIC_NODES`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L69)<br>**코드의 선언:** `정의=1500` |
| **설정·옵션:** [`MAX_DIAGNOSTIC_CHARS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L70)<br>**코드의 선언:** `정의=24000` |

#### LEVEL 1 · [engine.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py)

| 내용 |
| --- |
| **정의:** [`sweep_book`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L582)<br>**소스에 적힌 역할:** 매수 amount는 수수료 포함 원화, 매도 amount는 현물 수량입니다. |
| **정의:** [`MarketUniverseTransitionError`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1431)<br>**소스에 적힌 역할:** 검사를 통과한 기존 네 시장 모델도 다섯 시장의 운용에는 사용할 수 없습니다. |

**설정·옵션:** [`--config`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1641)

**코드의 선언:** `type=Path; default=None`

**설정·옵션:** [`--backfill`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1644)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`--once`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1645)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`--pages`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1646)

**코드의 선언:** `type=int`

**설정·옵션:** [`--once`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1648)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`--once`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1650)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`--paper-start`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1651)

**코드의 선언:** ``

**설정·옵션:** [`--final`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1653)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L27)

**코드의 선언:** `정의=Path(__file__).resolve().parent`

**설정·옵션:** [`RUNTIME_WORK`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L28)

**코드의 선언:** `정의=(ROOT / "work").resolve()`

**설정·옵션:** [`MARKETS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L29)

**코드의 선언:** `정의=("KRW-BTC", "KRW-SOL", "KRW-ETH", "KRW-XRP", "KRW-DOGE")`

**설정·옵션:** [`TIMEFRAMES`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L30)

**코드의 선언:** `정의=(1, 3, 5, 10, 15, 30, 60, 240, "day")`

**설정·옵션:** [`PAPER_ONLY`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L31)

**코드의 선언:** `정의=True`

**설정·옵션:** [`TRAINING_ETA`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L32)

**코드의 선언:** `정의=0.05`

**설정·옵션:** [`EXECUTION_POLICY`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L33)

**코드의 선언:** `정의="horizon_or_risk_v2"`

**설정·옵션:** [`PROTECTED_HOLDOUT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L34)

**코드의 선언:** `정의=("2026-10-19T00:00:00Z", "2026-10-20T00:00:00Z")`


**거부 조건:** [`sorted(config["markets"]) != sorted(MARKETS)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L106)

**오류 메시지:** 모의 대상은 BTC/SOL/ETH/XRP/DOGE 다섯 종목으로 고정됩니다

**거부 조건:** [`len(set(map(str, config["timeframes"]))) != len(config["timeframes"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L108)

**오류 메시지:** 중복 시간봉입니다

**거부 조건:** [`any(str(tf) not in set(map(str, TIMEFRAMES)) for tf in config["timeframes"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L110)

**오류 메시지:** 지원하지 않는 시간봉입니다

**거부 조건:** [`str(config["base_timeframe"]) not in set(map(str, config["timeframes"]))`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L112)

**오류 메시지:** 기본 시간봉이 수집 목록에 없습니다

**거부 조건:** [`tr["gap_bars"] &lt; tr["horizon"] + 1`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L125)

**오류 메시지:** 분할 간격은 다음봉 체결을 포함한 horizon+1 이상이어야 합니다

**거부 조건:** [`not tr["candidate_depths"] or any(not isinstance(d, int) or not 1 &lt;= d &lt;= 10 for d in tr["candidate_depths"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L127)

**오류 메시지:** 후보 깊이는 1~10 범위입니다

**거부 조건:** [`not 0 &lt; paper["initial_cash"] or paper["min_order"] &lt; 5000`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L132)

**오류 메시지:** 초기 자본과 최소 주문금액을 확인하세요

**거부 조건:** [`paper["max_allocation"] &gt; paper["max_total_allocation"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L134)

**오류 메시지:** 코인별 배분이 전체 한도를 초과합니다

**거부 조건:** [`isinstance(tr["class_weight_cap"], bool) or not math.isfinite(tr["class_weight_cap"]) or             not 1 &lt;= tr["class_weight_cap"] &lt;= 3`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L140)

**오류 메시지:** 클래스 가중치 상한은 1~3입니다

**거부 조건:** [`not tr["horizon_candidates"] or any(isinstance(h, bool) or not isinstance(h, int) or not 1 &lt;= h &lt;= 288                                            for h in tr["horizon_candidates"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L147)

**오류 메시지:** 지평 후보는 1~288봉 정수입니다

**거부 조건:** [`not tr["threshold_candidates"] or any(isinstance(t, bool) or not math.isfinite(t) or not 0 &lt; t &lt; 1                                               for t in [tr["label_threshold"], *tr["threshold_candidates"]])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L150)

**오류 메시지:** 비용 차감 후 라벨 임계값이 유효하지 않습니다

**거부 조건:** [`collection["request_interval"] &lt; 0.11 or not 1 &lt;= collection["retries"] &lt;= 8`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L153)

**오류 메시지:** 공유 요청 간격은 0.11초 이상, 재시도는 1~8회입니다

**거부 조건:** [`min(collection["recent_pages"], collection["backfill_pages_per_cycle"], collection["poll_seconds"], paper["poll_seconds"]) &lt;= 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L155)

**오류 메시지:** 수집 페이지와 대기 시간은 양수여야 합니다

**거부 조건:** [`sys.platform == "win32" and path.resolve().is_relative_to(RUNTIME_WORK)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L165)

**오류 메시지:** 운영 SQLite는 도커 내부에서만 접근하십시오

**거부 조건:** [`endpoint != "orderbook" and not endpoint.startswith("candles/")`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L251)

**오류 메시지:** 공개 캔들/호가 읽기만 허용됩니다

**거부 조건:** [`not np.isfinite(values).all() or (values[:, :4] &lt;= 0).any() or (values[:, 4] &lt; 0).any()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L293)

**오류 메시지:** 캔들 값이 유효하지 않습니다

**거부 조건:** [`((frame.high &lt; frame[["open", "close", "low"]].max(axis=1)) &#124;         (frame.low &gt; frame[["open", "close"]].min(axis=1))).any()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L295)

**오류 메시지:** OHLC 범위가 유효하지 않습니다

**거부 조건:** [`count &lt;= 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L328)

**오류 메시지:** 페이지 수는 양수여야 합니다

**거부 조건:** [`horizon &lt;= 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L402)

**오류 메시지:** 예측 지평은 양수여야 합니다

**거부 조건:** [`bar_length &lt;= pd.Timedelta(0) or not (lengths == bar_length).all()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L410)

**오류 메시지:** 라벨 봉 길이가 일관되지 않습니다

**거부 조건:** [`isinstance(sequence, bool) or not isinstance(sequence, int) or sequence &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L463)

**오류 메시지:** 연구 회차 커서가 손상되었습니다; 초기화하지 않습니다

**거부 조건:** [`p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p &lt; 0).any() or (p.sum(axis=1) &lt;= 0).any()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L484)

**오류 메시지:** 세 클래스 확률 배열이 유효하지 않습니다

**거부 조건:** [`not len(y) or len(y) != len(pred)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L561)

**오류 메시지:** 전체 표본 평가를 위한 행이 없습니다

**거부 조건:** [`side not in ("buy", "sell") or not math.isfinite(amount) or amount &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L584)

**오류 메시지:** 현물 모의체결 입력 오류

**거부 조건:** [`not math.isfinite(volatility) or volatility &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L612)

**오류 메시지:** 현재 변동성이 유효하지 않습니다

**거부 조건:** [`common is None or len(common) &lt; 2`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L820)

**오류 메시지:** 공통계좌 백테스트에 공통 완료 봉이 부족합니다

**거부 조건:** [`not xgb.build_info().get("USE_CUDA", False)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L915)

**오류 메시지:** 설치한 XGBoost가 CUDA를 지원하지 않습니다

**거부 조건:** [`not tr["holdout_start"] or not tr["holdout_end"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1060)

**오류 메시지:** training.holdout_start/holdout_end를 고정 UTC 시각으로 명시해야 합니다

**거부 조건:** [`he &lt;= hs`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1063)

**오류 메시지:** 잠금 시험 종료는 시작 이후여야 합니다

**거부 조건:** [`abs((he - hs).total_seconds() / 86400 - tr["holdout_days"]) &gt; 1e-6`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1065)

**오류 메시지:** holdout_days와 명시 시험 기간이 일치하지 않습니다

**거부 조건:** [`final and live`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1244)

**오류 메시지:** 실시간 학습은 잠금 시험을 평가할 수 없습니다

**거부 조건:** [`final and utc(None) &lt; plan["holdout"][1]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1252)

**오류 메시지:** 잠금 시험 기간이 끝나지 않았습니다

**거부 조건:** [`Path(pointer["run"]).name != pointer["run"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1441)

**오류 메시지:** 모델 포인터 경로 오류

**거부 조건:** [`stable_hash(manifest) != pointer["manifest_hash"] or not manifest["paper_only"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1445)

**오류 메시지:** 완성 모델 manifest가 일치하지 않습니다

**거부 조건:** [`manifest.get("run") != pointer["run"] or (package_markets != set(MARKETS) and not known_transition)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1457)

**오류 메시지:** 완성 모델의 실행 ID/다섯 코인 구성이 일치하지 않습니다

**거부 조건:** [`not isinstance(columns, list) or not columns or len(set(columns)) != len(columns)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1460)

**오류 메시지:** 완성 모델의 피처 목록이 유효하지 않습니다

**거부 조건:** [`not target.is_relative_to((ROOT / "work").resolve())`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L116)

**오류 메시지:** 유동 산출물은 프로젝트 work 내부에만 저장합니다

**거부 조건:** [`not isinstance(row["candle_date_time_utc"], str)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L279)

**오류 메시지:** 캔들 시각이 유효하지 않습니다

**거부 조건:** [`pd.isna(stamp)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L282)

**오류 메시지:** 캔들 시각이 유효하지 않습니다

**거부 조건:** [`not math.isfinite(price + size) or price &lt;= 0 or size &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L591)

**오류 메시지:** 호가 가격/잔량 오류

**거부 조건:** [`state is not None and (not isinstance(state, dict) or                                   state.get("initial_cash") != initial or state.get("paper_only") is not True)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L626)

**오류 메시지:** 복원 계좌의 초기 원금 또는 모의 전용 표시가 설정과 다릅니다

**거부 조건:** [`self.state["cash"] &lt; 0 or any(v["qty"] &lt; 0 for v in self.state["positions"].values())`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L635)

**오류 메시지:** 현물 계좌에 음수 잔액/공매도가 있습니다

**거부 조건:** [`any(m not in MARKETS for m in prices) or any(v not in (0, 1, 2) for v in signals.values())`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L655)

**오류 메시지:** 지원하지 않는 현물 시장/신호입니다

**거부 조건:** [`not all(math.isfinite(metrics[key]) for key in ("logloss", "net_return", "MDD"))`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1115)

**오류 메시지:** 모델 승격 비교 지표가 유한하지 않습니다

**거부 조건:** [`market not in MARKETS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1463)

**오류 메시지:** 지원하지 않는 모델 시장

**거부 조건:** [`not package["models"] or not len(package["models"]) == len(package["weights"]) == len(package["gpu"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1465)

**오류 메시지:** 완성 모델의 후보/가중치/적합 증거 수가 다릅니다

**거부 조건:** [`any(not math.isfinite(w) or w &lt; 0 for w in package["weights"]) or not math.isclose(sum(package["weights"]), 1, abs_tol=1e-6)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1467)

**오류 메시지:** 완성 모델의 확률 가중치가 유효하지 않습니다

**거부 조건:** [`set(quotes) != set(config["markets"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1572)

**오류 메시지:** 다섯 코인의 공개 호가가 모두 필요합니다

**거부 조건:** [`not isinstance(payload, list)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L260)

**오류 메시지:** 공개 시세 응답이 배열이 아닙니다

**거부 조건:** [`db.execute("SELECT 1 FROM final_tests WHERE start&lt;? AND end&gt;?", (end, start)).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1020)

**오류 메시지:** 이 시험 기간은 이미 소비 또는 평가 시도되었습니다. 반복 튜닝 금지

**거부 조건:** [`db.execute("SELECT 1 FROM exposures WHERE start&lt;? AND end&gt;?", (end, start)).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1022)

**오류 메시지:** 시험 기간이 이전 학습/검증에 노출되어 미사용 시험으로 평가할 수 없습니다

**거부 조건:** [`db.execute("SELECT 1 FROM final_tests WHERE start&lt;? AND end&gt;?", (iso(end), iso(start))).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1028)

**오류 메시지:** 이 시험 기간은 이미 소비 또는 평가 시도되었습니다. 반복 튜닝 금지

**거부 조건:** [`db.execute("SELECT 1 FROM exposures WHERE start&lt;? AND end&gt;?", (iso(end), iso(start))).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1030)

**오류 메시지:** 시험 기간이 이전 학습/검증에 노출되어 미사용 시험으로 평가할 수 없습니다

**거부 조건:** [`pending is None or market != expected`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1217)

**오류 메시지:** 특징 선행 준비의 시장 순서가 다릅니다

**거부 조건:** [`columns is not None and cols != columns`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1288)

**오류 메시지:** 코인별 피처 스키마가 다릅니다

**거부 조건:** [`Path(filename).name != filename or not (directory / filename).is_file()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1470)

**오류 메시지:** 완성 모델 파일이 없습니다

**거부 조건:** [`any(set(values) != set(MARKETS) for values in (frames, probabilities, rules))`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1147)

**오류 메시지:** 종목 전환 후보는 다섯 시장 모두의 후속 평가가 필요합니다

**거부 조건:** [`cooldowns[market] is None`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L737)

**오류 메시지:** 재진입 보류 만료시각이 없습니다

**거부 조건:** [`pd.isna(until)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L740)

**오류 메시지:** 재진입 보류 만료시각이 유효하지 않습니다


#### LEVEL 1 · [execution_probe.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py)


| 내용 |
| --- |
| **설정·옵션:** [`--config`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L60)<br>**코드의 선언:** `type=Path; default=engine.ROOT / 'engine-config.json'` |
| **설정·옵션:** [`SCOPE`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L13)<br>**코드의 선언:** `정의='체결 경로 진단 · 전략 성과 아님'` |

**거부 조건:** [`account.state['positions']`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L48)

**오류 메시지:** 별도 진단 계좌에 미청산 잔량이 남았습니다

**거부 조건:** [`len(payload) != 1 or payload[0].get('market') != market or not payload[0].get('orderbook_units')`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L28)

**오류 메시지:** 진단 시장의 호가가 없습니다

**거부 조건:** [`observed and now &lt;= observed[-1]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L35)

**오류 메시지:** 호가 관측 시각이 증가하지 않았습니다

**거부 조건:** [`not math.isfinite(bid + ask) or not 0 &lt; bid &lt;= ask`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L39)

**오류 메시지:** 호가 가격 또는 스프레드가 유효하지 않습니다


#### LEVEL 1 · [healthcheck.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py)


| 내용 |
| --- |
| **설정·옵션:** [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L18)<br>**코드의 선언:** `정의=Path(__file__).resolve().parent` |
| **설정·옵션:** [`INTEGRITY_TIMEOUT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L19)<br>**코드의 선언:** `정의=1800` |
| **설정·옵션:** [`INTEGRITY_MAX_AGE`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L20)<br>**코드의 선언:** `정의=21600` |
| **설정·옵션:** [`INTEGRITY_RETRY`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L21)<br>**코드의 선언:** `정의=900` |

**거부 조건:** [`database_identity(database) != identity`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L63)

**오류 메시지:** 검사 중 저장소 파일이 교체되었습니다

**거부 조건:** [`time.monotonic() &gt;= deadline`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L42)

**오류 메시지:** 전체 무결성 검사 시간 제한 초과

**거부 조건:** [`result['state'] != 'ok' or result['database'] != database_identity(database)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L86)

**오류 메시지:** 전체 검사 결과와 저장소가 일치하지 않습니다


#### LEVEL 1 · [supervisor.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py)


| 내용 |
| --- |
| **설정·옵션:** [`--config`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L176)<br>**코드의 선언:** `type=Path; default=ROOT / "service.json"` |
| **설정·옵션:** [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L19)<br>**코드의 선언:** `정의=Path(__file__).resolve().parent` |
| **설정·옵션:** [`STOP`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L20)<br>**코드의 선언:** `정의=threading.Event()` |

**거부 조건:** [`config.get("initial_capital") != 10_000_000 or paper.get("initial_cash") != config.get("initial_capital")             or config.get("market_mode") != "spot-paper" or config.get("notifications") is not False`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L162)

**오류 메시지:** 원금 1,000만원·현물 모의거래·알림 없음 설정을 확인하십시오.

**거부 조건:** [`start_at.tzinfo is None or engine_start.tzinfo is None or start_at != engine_start`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L167)

**오류 메시지:** 서비스와 엔진의 시작 시각·시간대가 일치해야 합니다.

**거부 조건:** [`config["trainer"]["interval_seconds"] != 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L169)

**오류 메시지:** 성공한 학습 회차의 고정 대기는 0초여야 합니다.

**거부 조건:** [`not STOP.is_set()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L128)

**오류 메시지:** 작업 시간 제한 초과


</details>

1. 모의매매와 GPU 연구·외부자료 수집의 독립 트랙을 확인합니다.
2. 설정·시점 경계·PAPER_ONLY와 실제 자료 준비 상태를 읽습니다.
3. 해당 요청의 수집·연구 조건이 충족될 때만 지정된 한 회차를 수행합니다.
4. 대시보드와 저장 지표에서 완료·대기·오류를 구분하고 기존 장부·모델을 보존합니다.

### LEVEL 1 · 결과와 주의할 범위

자료 크기·고유 아카이브 크기·DB 크기를 구분하고 준비/학습/승격 상태를 함께 확인합니다. 모의 손익은 실제 주문 성과가 아닙니다. GPU 사용률·처리량 개선도 수익 개선과 별개입니다.

**상황:** 연구가 대기함

**확인할 내용:** 자료량·최근 분봉 커버리지·최신성·GPU 사전 조건을 확인합니다. 대기 상태를 임의 학습 성공으로 바꾸지 않습니다.

**상황:** 외부 자료 검증 실패

**확인할 내용:** 공식 아카이브·체크섬·원문 체결 형식·완료봉을 확인합니다. 보간이나 기존 봉 덮어쓰기로 우회하지 않습니다.

**상황:** 차수 폴더만 실행할 수 없음

**확인할 내용:** 루트 공통 엔진·설정·검사와 차수 연결 파일의 실제 import 경로를 확인합니다.


[자료 폴더](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89) · [직전 비교·자료 범위](../revision-audit/GPU%EC%B2%98%EB%A6%AC%EB%9F%89.md) · [이 차수의 원래 전체 상세](#reading-level-5)

<a id="reading-level-2"></a>
## LEVEL 2 - Explorer

### LEVEL 2 · 목적과 구성의 연결

기존 공개 자료의 차수 표기는 `기본`이며 번호를 임의로 소급 부여하지 않습니다. 이번 추가 기록은 날짜로 구분합니다. 활성 실행 폴더를 복제하지 않고 기존 서비스에 적용하며 이전 코드는 Git 이력과 로컬 소스 전용 압축본으로 보존합니다.

기존 공개 자료의 차수 표기는 `기본`이며 번호를 임의로 소급 부여하지 않습니다. 이번 추가 기록은 날짜로 구분합니다. 활성 실행 폴더를 복제하지 않고 기존 서비스에 적용하며 이전 코드는 Git 이력과 로컬 소스 전용 압축본으로 보존합니다.

`CandleStore.read`는 완료 시각 범위 인덱스와 같은 순서로 조회해 임시 정렬을 제거합니다. 반환 행의 완료 시각 간격이 시간봉과 다르면 거부합니다. 기존 요청 범위·제한 수·최종 시작 시각 오름차순은 유지합니다.

`gpu_matrix_scope`는 같은 시장·같은 입력 객체·같은 열·같은 가중치 설정의 후보 묶음에서만 두 DMatrix를 재사용합니다. 범위 밖 호출과 다른 시장·가중치는 공유하지 않으며 종료 시 참조를 해제합니다. 해당 범위의 입력은 읽기 전용이며 제자리 변경은 지원하지 않습니다. 기존 CUDA 사전 검사와 매 반복 검사도 그대로 수행합니다.

`market_feature_loader`는 하나의 CPU 작업자로 다음 시장 특징만 선행 준비합니다. 누적 자료를 임의로 축소하지 않으며 학습·검증 범위는 같습니다. 준비 오류는 상위 호출로 전달하고, 본 작업 오류와 동시에 발생한 오류 원문도 예외에 남깁니다. 진행 중 SQL을 강제 종료하는 기능은 없습니다.

### LEVEL 2 · 입력에서 출력으로

**단계:** 입력

**내용:** 시세·호가·완료 분봉·외부 아카이브·연구 설정

**단계:** 처리

**내용:** 검증·저장 → 특징 준비·연구 → 정책 평가 → 모의체결·화면

**단계:** 출력

**내용:** 자료 준비 상태·연구 지표·모의계좌 장부·대시보드


입력·처리·출력은 위의 같은 자료 흐름을 다른 단계에서 읽는 기준입니다. 처리 조건과 출력의 의미를 함께 확인합니다.

### LEVEL 2 · 설정과 처리 순서의 관계

1. 모의매매와 GPU 연구·외부자료 수집의 독립 트랙을 확인합니다.
2. 설정·시점 경계·PAPER_ONLY와 실제 자료 준비 상태를 읽습니다.
3. 해당 요청의 수집·연구 조건이 충족될 때만 지정된 한 회차를 수행합니다.
4. 대시보드와 저장 지표에서 완료·대기·오류를 구분하고 기존 장부·모델을 보존합니다.

requirements와 engine-config.json, CUDA·GPU·DB 및 자료 저장 경로를 확인합니다. 실제 거래 계정·운영 DB·모델·개인 로그는 공개 코드와 별도입니다. 외부 수집에는 네트워크·디스크 여유·체크섬과 최신성 조건이 필요합니다.

이 차수 폴더만이 독립 실행 환경인지 먼저 확인합니다. 3차 외부자료 모듈과 4·5차 제어는 루트 engine.py·engine-config.json 및 tests와 연결됩니다. 실행 명령과 운영 경로는 해당 차수 README의 계약을 따릅니다.

#### LEVEL 2 · 이 보관본의 의존성 목록

[requirements.txt](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/requirements.txt)의 실제 항목입니다. 외부 실행 도구와 GPU 드라이버는 별도로 준비합니다.

```text
numpy>=1.26.4,<3
pandas>=2.2.3,<3
xgboost==3.0.5
```

#### LEVEL 2 · 소스에서 제공하는 CLI 옵션

**진입 소스:** [dashboard.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py)

**정의된 옵션 이름:** `--host`, `--port`

**진입 소스:** [engine.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py)

**정의된 옵션 이름:** `--config`, `--backfill`, `--once`, `--pages`, `--paper-start`, `--final`

**진입 소스:** [execution_probe.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py)

**정의된 옵션 이름:** `--config`

**진입 소스:** [supervisor.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py)

**정의된 옵션 이름:** `--config`


각 옵션의 기본값·필수값·부작용은 해당 main과 인자 정의를 함께 확인합니다.


<details>
<summary>함수 역할·기본값·입력 거부 조건의 세부 참조</summary>

다음 항목은 이 자료의 Python 소스에 명시된 선언입니다. 전체 분기와 호출 순서는 각 행의 코드 위치에서 이어서 읽을 수 있습니다.

#### LEVEL 2 · [dashboard.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py)

| 내용 |
| --- |
| **정의:** [`diagnostic`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L73)<br>**소스에 적힌 역할:** 허용 키·깊이·항목 수·총 문자수로 로컬 진단을 제한합니다. |
| **정의:** [`execution_probe`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L105)<br>**소스에 적힌 역할:** 본 계좌와 합산하지 않는 공개호가 경로 검사만 투영합니다. |
| **정의:** [`project`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L123)<br>**소스에 적힌 역할:** 허용한 필드의 단순 값만 추출합니다. |
| **정의:** [`SnapshotAdapter`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L289)<br>**소스에 적힌 역할:** 경로를 외부 입력으로 받지 않는 제한된 스냅숏 어댑터입니다. |

| 내용 |
| --- |
| **설정·옵션:** [`--host`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L554)<br>**코드의 선언:** `default="127.0.0.1"` |
| **설정·옵션:** [`--port`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L555)<br>**코드의 선언:** `type=int; default=8765` |
| **설정·옵션:** [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L15)<br>**코드의 선언:** `정의=Path(__file__).resolve().parent` |
| **설정·옵션:** [`MAX_JSON_BYTES`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L16)<br>**코드의 선언:** `정의=4 * 1024 * 1024` |
| **설정·옵션:** [`COINS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L17)<br>**코드의 선언:** `정의=("BTC", "SOL", "ETH", "XRP", "DOGE")` |
| **설정·옵션:** [`MAX_DIAGNOSTIC_DEPTH`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L67)<br>**코드의 선언:** `정의=8` |
| **설정·옵션:** [`MAX_DIAGNOSTIC_ITEMS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L68)<br>**코드의 선언:** `정의=100` |
| **설정·옵션:** [`MAX_DIAGNOSTIC_NODES`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L69)<br>**코드의 선언:** `정의=1500` |
| **설정·옵션:** [`MAX_DIAGNOSTIC_CHARS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L70)<br>**코드의 선언:** `정의=24000` |

#### LEVEL 2 · [engine.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py)

| 내용 |
| --- |
| **정의:** [`sweep_book`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L582)<br>**소스에 적힌 역할:** 매수 amount는 수수료 포함 원화, 매도 amount는 현물 수량입니다. |
| **정의:** [`MarketUniverseTransitionError`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1431)<br>**소스에 적힌 역할:** 검사를 통과한 기존 네 시장 모델도 다섯 시장의 운용에는 사용할 수 없습니다. |

**설정·옵션:** [`--config`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1641)

**코드의 선언:** `type=Path; default=None`

**설정·옵션:** [`--backfill`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1644)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`--once`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1645)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`--pages`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1646)

**코드의 선언:** `type=int`

**설정·옵션:** [`--once`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1648)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`--once`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1650)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`--paper-start`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1651)

**코드의 선언:** ``

**설정·옵션:** [`--final`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1653)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L27)

**코드의 선언:** `정의=Path(__file__).resolve().parent`

**설정·옵션:** [`RUNTIME_WORK`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L28)

**코드의 선언:** `정의=(ROOT / "work").resolve()`

**설정·옵션:** [`MARKETS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L29)

**코드의 선언:** `정의=("KRW-BTC", "KRW-SOL", "KRW-ETH", "KRW-XRP", "KRW-DOGE")`

**설정·옵션:** [`TIMEFRAMES`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L30)

**코드의 선언:** `정의=(1, 3, 5, 10, 15, 30, 60, 240, "day")`

**설정·옵션:** [`PAPER_ONLY`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L31)

**코드의 선언:** `정의=True`

**설정·옵션:** [`TRAINING_ETA`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L32)

**코드의 선언:** `정의=0.05`

**설정·옵션:** [`EXECUTION_POLICY`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L33)

**코드의 선언:** `정의="horizon_or_risk_v2"`

**설정·옵션:** [`PROTECTED_HOLDOUT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L34)

**코드의 선언:** `정의=("2026-10-19T00:00:00Z", "2026-10-20T00:00:00Z")`


**거부 조건:** [`sorted(config["markets"]) != sorted(MARKETS)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L106)

**오류 메시지:** 모의 대상은 BTC/SOL/ETH/XRP/DOGE 다섯 종목으로 고정됩니다

**거부 조건:** [`len(set(map(str, config["timeframes"]))) != len(config["timeframes"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L108)

**오류 메시지:** 중복 시간봉입니다

**거부 조건:** [`any(str(tf) not in set(map(str, TIMEFRAMES)) for tf in config["timeframes"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L110)

**오류 메시지:** 지원하지 않는 시간봉입니다

**거부 조건:** [`str(config["base_timeframe"]) not in set(map(str, config["timeframes"]))`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L112)

**오류 메시지:** 기본 시간봉이 수집 목록에 없습니다

**거부 조건:** [`tr["gap_bars"] &lt; tr["horizon"] + 1`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L125)

**오류 메시지:** 분할 간격은 다음봉 체결을 포함한 horizon+1 이상이어야 합니다

**거부 조건:** [`not tr["candidate_depths"] or any(not isinstance(d, int) or not 1 &lt;= d &lt;= 10 for d in tr["candidate_depths"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L127)

**오류 메시지:** 후보 깊이는 1~10 범위입니다

**거부 조건:** [`not 0 &lt; paper["initial_cash"] or paper["min_order"] &lt; 5000`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L132)

**오류 메시지:** 초기 자본과 최소 주문금액을 확인하세요

**거부 조건:** [`paper["max_allocation"] &gt; paper["max_total_allocation"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L134)

**오류 메시지:** 코인별 배분이 전체 한도를 초과합니다

**거부 조건:** [`isinstance(tr["class_weight_cap"], bool) or not math.isfinite(tr["class_weight_cap"]) or             not 1 &lt;= tr["class_weight_cap"] &lt;= 3`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L140)

**오류 메시지:** 클래스 가중치 상한은 1~3입니다

**거부 조건:** [`not tr["horizon_candidates"] or any(isinstance(h, bool) or not isinstance(h, int) or not 1 &lt;= h &lt;= 288                                            for h in tr["horizon_candidates"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L147)

**오류 메시지:** 지평 후보는 1~288봉 정수입니다

**거부 조건:** [`not tr["threshold_candidates"] or any(isinstance(t, bool) or not math.isfinite(t) or not 0 &lt; t &lt; 1                                               for t in [tr["label_threshold"], *tr["threshold_candidates"]])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L150)

**오류 메시지:** 비용 차감 후 라벨 임계값이 유효하지 않습니다

**거부 조건:** [`collection["request_interval"] &lt; 0.11 or not 1 &lt;= collection["retries"] &lt;= 8`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L153)

**오류 메시지:** 공유 요청 간격은 0.11초 이상, 재시도는 1~8회입니다

**거부 조건:** [`min(collection["recent_pages"], collection["backfill_pages_per_cycle"], collection["poll_seconds"], paper["poll_seconds"]) &lt;= 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L155)

**오류 메시지:** 수집 페이지와 대기 시간은 양수여야 합니다

**거부 조건:** [`sys.platform == "win32" and path.resolve().is_relative_to(RUNTIME_WORK)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L165)

**오류 메시지:** 운영 SQLite는 도커 내부에서만 접근하십시오

**거부 조건:** [`endpoint != "orderbook" and not endpoint.startswith("candles/")`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L251)

**오류 메시지:** 공개 캔들/호가 읽기만 허용됩니다

**거부 조건:** [`not np.isfinite(values).all() or (values[:, :4] &lt;= 0).any() or (values[:, 4] &lt; 0).any()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L293)

**오류 메시지:** 캔들 값이 유효하지 않습니다

**거부 조건:** [`((frame.high &lt; frame[["open", "close", "low"]].max(axis=1)) &#124;         (frame.low &gt; frame[["open", "close"]].min(axis=1))).any()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L295)

**오류 메시지:** OHLC 범위가 유효하지 않습니다

**거부 조건:** [`count &lt;= 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L328)

**오류 메시지:** 페이지 수는 양수여야 합니다

**거부 조건:** [`horizon &lt;= 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L402)

**오류 메시지:** 예측 지평은 양수여야 합니다

**거부 조건:** [`bar_length &lt;= pd.Timedelta(0) or not (lengths == bar_length).all()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L410)

**오류 메시지:** 라벨 봉 길이가 일관되지 않습니다

**거부 조건:** [`isinstance(sequence, bool) or not isinstance(sequence, int) or sequence &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L463)

**오류 메시지:** 연구 회차 커서가 손상되었습니다; 초기화하지 않습니다

**거부 조건:** [`p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p &lt; 0).any() or (p.sum(axis=1) &lt;= 0).any()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L484)

**오류 메시지:** 세 클래스 확률 배열이 유효하지 않습니다

**거부 조건:** [`not len(y) or len(y) != len(pred)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L561)

**오류 메시지:** 전체 표본 평가를 위한 행이 없습니다

**거부 조건:** [`side not in ("buy", "sell") or not math.isfinite(amount) or amount &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L584)

**오류 메시지:** 현물 모의체결 입력 오류

**거부 조건:** [`not math.isfinite(volatility) or volatility &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L612)

**오류 메시지:** 현재 변동성이 유효하지 않습니다

**거부 조건:** [`common is None or len(common) &lt; 2`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L820)

**오류 메시지:** 공통계좌 백테스트에 공통 완료 봉이 부족합니다

**거부 조건:** [`not xgb.build_info().get("USE_CUDA", False)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L915)

**오류 메시지:** 설치한 XGBoost가 CUDA를 지원하지 않습니다

**거부 조건:** [`not tr["holdout_start"] or not tr["holdout_end"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1060)

**오류 메시지:** training.holdout_start/holdout_end를 고정 UTC 시각으로 명시해야 합니다

**거부 조건:** [`he &lt;= hs`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1063)

**오류 메시지:** 잠금 시험 종료는 시작 이후여야 합니다

**거부 조건:** [`abs((he - hs).total_seconds() / 86400 - tr["holdout_days"]) &gt; 1e-6`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1065)

**오류 메시지:** holdout_days와 명시 시험 기간이 일치하지 않습니다

**거부 조건:** [`final and live`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1244)

**오류 메시지:** 실시간 학습은 잠금 시험을 평가할 수 없습니다

**거부 조건:** [`final and utc(None) &lt; plan["holdout"][1]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1252)

**오류 메시지:** 잠금 시험 기간이 끝나지 않았습니다

**거부 조건:** [`Path(pointer["run"]).name != pointer["run"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1441)

**오류 메시지:** 모델 포인터 경로 오류

**거부 조건:** [`stable_hash(manifest) != pointer["manifest_hash"] or not manifest["paper_only"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1445)

**오류 메시지:** 완성 모델 manifest가 일치하지 않습니다

**거부 조건:** [`manifest.get("run") != pointer["run"] or (package_markets != set(MARKETS) and not known_transition)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1457)

**오류 메시지:** 완성 모델의 실행 ID/다섯 코인 구성이 일치하지 않습니다

**거부 조건:** [`not isinstance(columns, list) or not columns or len(set(columns)) != len(columns)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1460)

**오류 메시지:** 완성 모델의 피처 목록이 유효하지 않습니다

**거부 조건:** [`not target.is_relative_to((ROOT / "work").resolve())`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L116)

**오류 메시지:** 유동 산출물은 프로젝트 work 내부에만 저장합니다

**거부 조건:** [`not isinstance(row["candle_date_time_utc"], str)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L279)

**오류 메시지:** 캔들 시각이 유효하지 않습니다

**거부 조건:** [`pd.isna(stamp)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L282)

**오류 메시지:** 캔들 시각이 유효하지 않습니다

**거부 조건:** [`not math.isfinite(price + size) or price &lt;= 0 or size &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L591)

**오류 메시지:** 호가 가격/잔량 오류

**거부 조건:** [`state is not None and (not isinstance(state, dict) or                                   state.get("initial_cash") != initial or state.get("paper_only") is not True)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L626)

**오류 메시지:** 복원 계좌의 초기 원금 또는 모의 전용 표시가 설정과 다릅니다

**거부 조건:** [`self.state["cash"] &lt; 0 or any(v["qty"] &lt; 0 for v in self.state["positions"].values())`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L635)

**오류 메시지:** 현물 계좌에 음수 잔액/공매도가 있습니다

**거부 조건:** [`any(m not in MARKETS for m in prices) or any(v not in (0, 1, 2) for v in signals.values())`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L655)

**오류 메시지:** 지원하지 않는 현물 시장/신호입니다

**거부 조건:** [`not all(math.isfinite(metrics[key]) for key in ("logloss", "net_return", "MDD"))`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1115)

**오류 메시지:** 모델 승격 비교 지표가 유한하지 않습니다

**거부 조건:** [`market not in MARKETS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1463)

**오류 메시지:** 지원하지 않는 모델 시장

**거부 조건:** [`not package["models"] or not len(package["models"]) == len(package["weights"]) == len(package["gpu"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1465)

**오류 메시지:** 완성 모델의 후보/가중치/적합 증거 수가 다릅니다

**거부 조건:** [`any(not math.isfinite(w) or w &lt; 0 for w in package["weights"]) or not math.isclose(sum(package["weights"]), 1, abs_tol=1e-6)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1467)

**오류 메시지:** 완성 모델의 확률 가중치가 유효하지 않습니다

**거부 조건:** [`set(quotes) != set(config["markets"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1572)

**오류 메시지:** 다섯 코인의 공개 호가가 모두 필요합니다

**거부 조건:** [`not isinstance(payload, list)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L260)

**오류 메시지:** 공개 시세 응답이 배열이 아닙니다

**거부 조건:** [`db.execute("SELECT 1 FROM final_tests WHERE start&lt;? AND end&gt;?", (end, start)).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1020)

**오류 메시지:** 이 시험 기간은 이미 소비 또는 평가 시도되었습니다. 반복 튜닝 금지

**거부 조건:** [`db.execute("SELECT 1 FROM exposures WHERE start&lt;? AND end&gt;?", (end, start)).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1022)

**오류 메시지:** 시험 기간이 이전 학습/검증에 노출되어 미사용 시험으로 평가할 수 없습니다

**거부 조건:** [`db.execute("SELECT 1 FROM final_tests WHERE start&lt;? AND end&gt;?", (iso(end), iso(start))).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1028)

**오류 메시지:** 이 시험 기간은 이미 소비 또는 평가 시도되었습니다. 반복 튜닝 금지

**거부 조건:** [`db.execute("SELECT 1 FROM exposures WHERE start&lt;? AND end&gt;?", (iso(end), iso(start))).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1030)

**오류 메시지:** 시험 기간이 이전 학습/검증에 노출되어 미사용 시험으로 평가할 수 없습니다

**거부 조건:** [`pending is None or market != expected`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1217)

**오류 메시지:** 특징 선행 준비의 시장 순서가 다릅니다

**거부 조건:** [`columns is not None and cols != columns`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1288)

**오류 메시지:** 코인별 피처 스키마가 다릅니다

**거부 조건:** [`Path(filename).name != filename or not (directory / filename).is_file()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1470)

**오류 메시지:** 완성 모델 파일이 없습니다

**거부 조건:** [`any(set(values) != set(MARKETS) for values in (frames, probabilities, rules))`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1147)

**오류 메시지:** 종목 전환 후보는 다섯 시장 모두의 후속 평가가 필요합니다

**거부 조건:** [`cooldowns[market] is None`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L737)

**오류 메시지:** 재진입 보류 만료시각이 없습니다

**거부 조건:** [`pd.isna(until)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L740)

**오류 메시지:** 재진입 보류 만료시각이 유효하지 않습니다


#### LEVEL 2 · [execution_probe.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py)


| 내용 |
| --- |
| **설정·옵션:** [`--config`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L60)<br>**코드의 선언:** `type=Path; default=engine.ROOT / 'engine-config.json'` |
| **설정·옵션:** [`SCOPE`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L13)<br>**코드의 선언:** `정의='체결 경로 진단 · 전략 성과 아님'` |

**거부 조건:** [`account.state['positions']`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L48)

**오류 메시지:** 별도 진단 계좌에 미청산 잔량이 남았습니다

**거부 조건:** [`len(payload) != 1 or payload[0].get('market') != market or not payload[0].get('orderbook_units')`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L28)

**오류 메시지:** 진단 시장의 호가가 없습니다

**거부 조건:** [`observed and now &lt;= observed[-1]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L35)

**오류 메시지:** 호가 관측 시각이 증가하지 않았습니다

**거부 조건:** [`not math.isfinite(bid + ask) or not 0 &lt; bid &lt;= ask`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L39)

**오류 메시지:** 호가 가격 또는 스프레드가 유효하지 않습니다


#### LEVEL 2 · [healthcheck.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py)


| 내용 |
| --- |
| **설정·옵션:** [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L18)<br>**코드의 선언:** `정의=Path(__file__).resolve().parent` |
| **설정·옵션:** [`INTEGRITY_TIMEOUT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L19)<br>**코드의 선언:** `정의=1800` |
| **설정·옵션:** [`INTEGRITY_MAX_AGE`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L20)<br>**코드의 선언:** `정의=21600` |
| **설정·옵션:** [`INTEGRITY_RETRY`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L21)<br>**코드의 선언:** `정의=900` |

**거부 조건:** [`database_identity(database) != identity`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L63)

**오류 메시지:** 검사 중 저장소 파일이 교체되었습니다

**거부 조건:** [`time.monotonic() &gt;= deadline`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L42)

**오류 메시지:** 전체 무결성 검사 시간 제한 초과

**거부 조건:** [`result['state'] != 'ok' or result['database'] != database_identity(database)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L86)

**오류 메시지:** 전체 검사 결과와 저장소가 일치하지 않습니다


#### LEVEL 2 · [supervisor.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py)


| 내용 |
| --- |
| **설정·옵션:** [`--config`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L176)<br>**코드의 선언:** `type=Path; default=ROOT / "service.json"` |
| **설정·옵션:** [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L19)<br>**코드의 선언:** `정의=Path(__file__).resolve().parent` |
| **설정·옵션:** [`STOP`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L20)<br>**코드의 선언:** `정의=threading.Event()` |

**거부 조건:** [`config.get("initial_capital") != 10_000_000 or paper.get("initial_cash") != config.get("initial_capital")             or config.get("market_mode") != "spot-paper" or config.get("notifications") is not False`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L162)

**오류 메시지:** 원금 1,000만원·현물 모의거래·알림 없음 설정을 확인하십시오.

**거부 조건:** [`start_at.tzinfo is None or engine_start.tzinfo is None or start_at != engine_start`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L167)

**오류 메시지:** 서비스와 엔진의 시작 시각·시간대가 일치해야 합니다.

**거부 조건:** [`config["trainer"]["interval_seconds"] != 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L169)

**오류 메시지:** 성공한 학습 회차의 고정 대기는 0초여야 합니다.

**거부 조건:** [`not STOP.is_set()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L128)

**오류 메시지:** 작업 시간 제한 초과


</details>

### LEVEL 2 · 관측과 해석을 구분하기

자료 크기·고유 아카이브 크기·DB 크기를 구분하고 준비/학습/승격 상태를 함께 확인합니다. 모의 손익은 실제 주문 성과가 아닙니다. GPU 사용률·처리량 개선도 수익 개선과 별개입니다.

**상황:** 연구가 대기함

**확인할 내용:** 자료량·최근 분봉 커버리지·최신성·GPU 사전 조건을 확인합니다. 대기 상태를 임의 학습 성공으로 바꾸지 않습니다.

**상황:** 외부 자료 검증 실패

**확인할 내용:** 공식 아카이브·체크섬·원문 체결 형식·완료봉을 확인합니다. 보간이나 기존 봉 덮어쓰기로 우회하지 않습니다.

**상황:** 차수 폴더만 실행할 수 없음

**확인할 내용:** 루트 공통 엔진·설정·검사와 차수 연결 파일의 실제 import 경로를 확인합니다.


[자료 폴더](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89) · [직전 비교·자료 범위](../revision-audit/GPU%EC%B2%98%EB%A6%AC%EB%9F%89.md) · [이 차수의 원래 전체 상세](#reading-level-5)

<a id="reading-level-3"></a>
## LEVEL 3 - Practitioner

### LEVEL 3 · 이 차수를 준비하는 범위

기존 공개 자료의 차수 표기는 `기본`이며 번호를 임의로 소급 부여하지 않습니다. 이번 추가 기록은 날짜로 구분합니다. 활성 실행 폴더를 복제하지 않고 기존 서비스에 적용하며 이전 코드는 Git 이력과 로컬 소스 전용 압축본으로 보존합니다.

requirements와 engine-config.json, CUDA·GPU·DB 및 자료 저장 경로를 확인합니다. 실제 거래 계정·운영 DB·모델·개인 로그는 공개 코드와 별도입니다. 외부 수집에는 네트워크·디스크 여유·체크섬과 최신성 조건이 필요합니다.

이 차수 폴더만이 독립 실행 환경인지 먼저 확인합니다. 3차 외부자료 모듈과 4·5차 제어는 루트 engine.py·engine-config.json 및 tests와 연결됩니다. 실행 명령과 운영 경로는 해당 차수 README의 계약을 따릅니다.

#### LEVEL 3 · 이 보관본의 의존성 목록

[requirements.txt](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/requirements.txt)의 실제 항목입니다. 외부 실행 도구와 GPU 드라이버는 별도로 준비합니다.

```text
numpy>=1.26.4,<3
pandas>=2.2.3,<3
xgboost==3.0.5
```

#### LEVEL 3 · 소스에서 제공하는 CLI 옵션

**진입 소스:** [dashboard.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py)

**정의된 옵션 이름:** `--host`, `--port`

**진입 소스:** [engine.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py)

**정의된 옵션 이름:** `--config`, `--backfill`, `--once`, `--pages`, `--paper-start`, `--final`

**진입 소스:** [execution_probe.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py)

**정의된 옵션 이름:** `--config`

**진입 소스:** [supervisor.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py)

**정의된 옵션 이름:** `--config`


각 옵션의 기본값·필수값·부작용은 해당 main과 인자 정의를 함께 확인합니다.


<details>
<summary>함수 역할·기본값·입력 거부 조건의 세부 참조</summary>

다음 항목은 이 자료의 Python 소스에 명시된 선언입니다. 전체 분기와 호출 순서는 각 행의 코드 위치에서 이어서 읽을 수 있습니다.

#### LEVEL 3 · [dashboard.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py)

| 내용 |
| --- |
| **정의:** [`diagnostic`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L73)<br>**소스에 적힌 역할:** 허용 키·깊이·항목 수·총 문자수로 로컬 진단을 제한합니다. |
| **정의:** [`execution_probe`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L105)<br>**소스에 적힌 역할:** 본 계좌와 합산하지 않는 공개호가 경로 검사만 투영합니다. |
| **정의:** [`project`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L123)<br>**소스에 적힌 역할:** 허용한 필드의 단순 값만 추출합니다. |
| **정의:** [`SnapshotAdapter`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L289)<br>**소스에 적힌 역할:** 경로를 외부 입력으로 받지 않는 제한된 스냅숏 어댑터입니다. |

| 내용 |
| --- |
| **설정·옵션:** [`--host`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L554)<br>**코드의 선언:** `default="127.0.0.1"` |
| **설정·옵션:** [`--port`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L555)<br>**코드의 선언:** `type=int; default=8765` |
| **설정·옵션:** [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L15)<br>**코드의 선언:** `정의=Path(__file__).resolve().parent` |
| **설정·옵션:** [`MAX_JSON_BYTES`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L16)<br>**코드의 선언:** `정의=4 * 1024 * 1024` |
| **설정·옵션:** [`COINS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L17)<br>**코드의 선언:** `정의=("BTC", "SOL", "ETH", "XRP", "DOGE")` |
| **설정·옵션:** [`MAX_DIAGNOSTIC_DEPTH`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L67)<br>**코드의 선언:** `정의=8` |
| **설정·옵션:** [`MAX_DIAGNOSTIC_ITEMS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L68)<br>**코드의 선언:** `정의=100` |
| **설정·옵션:** [`MAX_DIAGNOSTIC_NODES`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L69)<br>**코드의 선언:** `정의=1500` |
| **설정·옵션:** [`MAX_DIAGNOSTIC_CHARS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L70)<br>**코드의 선언:** `정의=24000` |

#### LEVEL 3 · [engine.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py)

| 내용 |
| --- |
| **정의:** [`sweep_book`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L582)<br>**소스에 적힌 역할:** 매수 amount는 수수료 포함 원화, 매도 amount는 현물 수량입니다. |
| **정의:** [`MarketUniverseTransitionError`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1431)<br>**소스에 적힌 역할:** 검사를 통과한 기존 네 시장 모델도 다섯 시장의 운용에는 사용할 수 없습니다. |

**설정·옵션:** [`--config`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1641)

**코드의 선언:** `type=Path; default=None`

**설정·옵션:** [`--backfill`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1644)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`--once`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1645)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`--pages`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1646)

**코드의 선언:** `type=int`

**설정·옵션:** [`--once`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1648)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`--once`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1650)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`--paper-start`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1651)

**코드의 선언:** ``

**설정·옵션:** [`--final`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1653)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L27)

**코드의 선언:** `정의=Path(__file__).resolve().parent`

**설정·옵션:** [`RUNTIME_WORK`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L28)

**코드의 선언:** `정의=(ROOT / "work").resolve()`

**설정·옵션:** [`MARKETS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L29)

**코드의 선언:** `정의=("KRW-BTC", "KRW-SOL", "KRW-ETH", "KRW-XRP", "KRW-DOGE")`

**설정·옵션:** [`TIMEFRAMES`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L30)

**코드의 선언:** `정의=(1, 3, 5, 10, 15, 30, 60, 240, "day")`

**설정·옵션:** [`PAPER_ONLY`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L31)

**코드의 선언:** `정의=True`

**설정·옵션:** [`TRAINING_ETA`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L32)

**코드의 선언:** `정의=0.05`

**설정·옵션:** [`EXECUTION_POLICY`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L33)

**코드의 선언:** `정의="horizon_or_risk_v2"`

**설정·옵션:** [`PROTECTED_HOLDOUT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L34)

**코드의 선언:** `정의=("2026-10-19T00:00:00Z", "2026-10-20T00:00:00Z")`


**거부 조건:** [`sorted(config["markets"]) != sorted(MARKETS)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L106)

**오류 메시지:** 모의 대상은 BTC/SOL/ETH/XRP/DOGE 다섯 종목으로 고정됩니다

**거부 조건:** [`len(set(map(str, config["timeframes"]))) != len(config["timeframes"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L108)

**오류 메시지:** 중복 시간봉입니다

**거부 조건:** [`any(str(tf) not in set(map(str, TIMEFRAMES)) for tf in config["timeframes"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L110)

**오류 메시지:** 지원하지 않는 시간봉입니다

**거부 조건:** [`str(config["base_timeframe"]) not in set(map(str, config["timeframes"]))`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L112)

**오류 메시지:** 기본 시간봉이 수집 목록에 없습니다

**거부 조건:** [`tr["gap_bars"] &lt; tr["horizon"] + 1`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L125)

**오류 메시지:** 분할 간격은 다음봉 체결을 포함한 horizon+1 이상이어야 합니다

**거부 조건:** [`not tr["candidate_depths"] or any(not isinstance(d, int) or not 1 &lt;= d &lt;= 10 for d in tr["candidate_depths"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L127)

**오류 메시지:** 후보 깊이는 1~10 범위입니다

**거부 조건:** [`not 0 &lt; paper["initial_cash"] or paper["min_order"] &lt; 5000`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L132)

**오류 메시지:** 초기 자본과 최소 주문금액을 확인하세요

**거부 조건:** [`paper["max_allocation"] &gt; paper["max_total_allocation"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L134)

**오류 메시지:** 코인별 배분이 전체 한도를 초과합니다

**거부 조건:** [`isinstance(tr["class_weight_cap"], bool) or not math.isfinite(tr["class_weight_cap"]) or             not 1 &lt;= tr["class_weight_cap"] &lt;= 3`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L140)

**오류 메시지:** 클래스 가중치 상한은 1~3입니다

**거부 조건:** [`not tr["horizon_candidates"] or any(isinstance(h, bool) or not isinstance(h, int) or not 1 &lt;= h &lt;= 288                                            for h in tr["horizon_candidates"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L147)

**오류 메시지:** 지평 후보는 1~288봉 정수입니다

**거부 조건:** [`not tr["threshold_candidates"] or any(isinstance(t, bool) or not math.isfinite(t) or not 0 &lt; t &lt; 1                                               for t in [tr["label_threshold"], *tr["threshold_candidates"]])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L150)

**오류 메시지:** 비용 차감 후 라벨 임계값이 유효하지 않습니다

**거부 조건:** [`collection["request_interval"] &lt; 0.11 or not 1 &lt;= collection["retries"] &lt;= 8`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L153)

**오류 메시지:** 공유 요청 간격은 0.11초 이상, 재시도는 1~8회입니다

**거부 조건:** [`min(collection["recent_pages"], collection["backfill_pages_per_cycle"], collection["poll_seconds"], paper["poll_seconds"]) &lt;= 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L155)

**오류 메시지:** 수집 페이지와 대기 시간은 양수여야 합니다

**거부 조건:** [`sys.platform == "win32" and path.resolve().is_relative_to(RUNTIME_WORK)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L165)

**오류 메시지:** 운영 SQLite는 도커 내부에서만 접근하십시오

**거부 조건:** [`endpoint != "orderbook" and not endpoint.startswith("candles/")`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L251)

**오류 메시지:** 공개 캔들/호가 읽기만 허용됩니다

**거부 조건:** [`not np.isfinite(values).all() or (values[:, :4] &lt;= 0).any() or (values[:, 4] &lt; 0).any()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L293)

**오류 메시지:** 캔들 값이 유효하지 않습니다

**거부 조건:** [`((frame.high &lt; frame[["open", "close", "low"]].max(axis=1)) &#124;         (frame.low &gt; frame[["open", "close"]].min(axis=1))).any()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L295)

**오류 메시지:** OHLC 범위가 유효하지 않습니다

**거부 조건:** [`count &lt;= 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L328)

**오류 메시지:** 페이지 수는 양수여야 합니다

**거부 조건:** [`horizon &lt;= 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L402)

**오류 메시지:** 예측 지평은 양수여야 합니다

**거부 조건:** [`bar_length &lt;= pd.Timedelta(0) or not (lengths == bar_length).all()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L410)

**오류 메시지:** 라벨 봉 길이가 일관되지 않습니다

**거부 조건:** [`isinstance(sequence, bool) or not isinstance(sequence, int) or sequence &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L463)

**오류 메시지:** 연구 회차 커서가 손상되었습니다; 초기화하지 않습니다

**거부 조건:** [`p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p &lt; 0).any() or (p.sum(axis=1) &lt;= 0).any()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L484)

**오류 메시지:** 세 클래스 확률 배열이 유효하지 않습니다

**거부 조건:** [`not len(y) or len(y) != len(pred)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L561)

**오류 메시지:** 전체 표본 평가를 위한 행이 없습니다

**거부 조건:** [`side not in ("buy", "sell") or not math.isfinite(amount) or amount &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L584)

**오류 메시지:** 현물 모의체결 입력 오류

**거부 조건:** [`not math.isfinite(volatility) or volatility &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L612)

**오류 메시지:** 현재 변동성이 유효하지 않습니다

**거부 조건:** [`common is None or len(common) &lt; 2`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L820)

**오류 메시지:** 공통계좌 백테스트에 공통 완료 봉이 부족합니다

**거부 조건:** [`not xgb.build_info().get("USE_CUDA", False)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L915)

**오류 메시지:** 설치한 XGBoost가 CUDA를 지원하지 않습니다

**거부 조건:** [`not tr["holdout_start"] or not tr["holdout_end"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1060)

**오류 메시지:** training.holdout_start/holdout_end를 고정 UTC 시각으로 명시해야 합니다

**거부 조건:** [`he &lt;= hs`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1063)

**오류 메시지:** 잠금 시험 종료는 시작 이후여야 합니다

**거부 조건:** [`abs((he - hs).total_seconds() / 86400 - tr["holdout_days"]) &gt; 1e-6`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1065)

**오류 메시지:** holdout_days와 명시 시험 기간이 일치하지 않습니다

**거부 조건:** [`final and live`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1244)

**오류 메시지:** 실시간 학습은 잠금 시험을 평가할 수 없습니다

**거부 조건:** [`final and utc(None) &lt; plan["holdout"][1]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1252)

**오류 메시지:** 잠금 시험 기간이 끝나지 않았습니다

**거부 조건:** [`Path(pointer["run"]).name != pointer["run"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1441)

**오류 메시지:** 모델 포인터 경로 오류

**거부 조건:** [`stable_hash(manifest) != pointer["manifest_hash"] or not manifest["paper_only"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1445)

**오류 메시지:** 완성 모델 manifest가 일치하지 않습니다

**거부 조건:** [`manifest.get("run") != pointer["run"] or (package_markets != set(MARKETS) and not known_transition)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1457)

**오류 메시지:** 완성 모델의 실행 ID/다섯 코인 구성이 일치하지 않습니다

**거부 조건:** [`not isinstance(columns, list) or not columns or len(set(columns)) != len(columns)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1460)

**오류 메시지:** 완성 모델의 피처 목록이 유효하지 않습니다

**거부 조건:** [`not target.is_relative_to((ROOT / "work").resolve())`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L116)

**오류 메시지:** 유동 산출물은 프로젝트 work 내부에만 저장합니다

**거부 조건:** [`not isinstance(row["candle_date_time_utc"], str)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L279)

**오류 메시지:** 캔들 시각이 유효하지 않습니다

**거부 조건:** [`pd.isna(stamp)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L282)

**오류 메시지:** 캔들 시각이 유효하지 않습니다

**거부 조건:** [`not math.isfinite(price + size) or price &lt;= 0 or size &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L591)

**오류 메시지:** 호가 가격/잔량 오류

**거부 조건:** [`state is not None and (not isinstance(state, dict) or                                   state.get("initial_cash") != initial or state.get("paper_only") is not True)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L626)

**오류 메시지:** 복원 계좌의 초기 원금 또는 모의 전용 표시가 설정과 다릅니다

**거부 조건:** [`self.state["cash"] &lt; 0 or any(v["qty"] &lt; 0 for v in self.state["positions"].values())`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L635)

**오류 메시지:** 현물 계좌에 음수 잔액/공매도가 있습니다

**거부 조건:** [`any(m not in MARKETS for m in prices) or any(v not in (0, 1, 2) for v in signals.values())`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L655)

**오류 메시지:** 지원하지 않는 현물 시장/신호입니다

**거부 조건:** [`not all(math.isfinite(metrics[key]) for key in ("logloss", "net_return", "MDD"))`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1115)

**오류 메시지:** 모델 승격 비교 지표가 유한하지 않습니다

**거부 조건:** [`market not in MARKETS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1463)

**오류 메시지:** 지원하지 않는 모델 시장

**거부 조건:** [`not package["models"] or not len(package["models"]) == len(package["weights"]) == len(package["gpu"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1465)

**오류 메시지:** 완성 모델의 후보/가중치/적합 증거 수가 다릅니다

**거부 조건:** [`any(not math.isfinite(w) or w &lt; 0 for w in package["weights"]) or not math.isclose(sum(package["weights"]), 1, abs_tol=1e-6)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1467)

**오류 메시지:** 완성 모델의 확률 가중치가 유효하지 않습니다

**거부 조건:** [`set(quotes) != set(config["markets"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1572)

**오류 메시지:** 다섯 코인의 공개 호가가 모두 필요합니다

**거부 조건:** [`not isinstance(payload, list)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L260)

**오류 메시지:** 공개 시세 응답이 배열이 아닙니다

**거부 조건:** [`db.execute("SELECT 1 FROM final_tests WHERE start&lt;? AND end&gt;?", (end, start)).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1020)

**오류 메시지:** 이 시험 기간은 이미 소비 또는 평가 시도되었습니다. 반복 튜닝 금지

**거부 조건:** [`db.execute("SELECT 1 FROM exposures WHERE start&lt;? AND end&gt;?", (end, start)).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1022)

**오류 메시지:** 시험 기간이 이전 학습/검증에 노출되어 미사용 시험으로 평가할 수 없습니다

**거부 조건:** [`db.execute("SELECT 1 FROM final_tests WHERE start&lt;? AND end&gt;?", (iso(end), iso(start))).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1028)

**오류 메시지:** 이 시험 기간은 이미 소비 또는 평가 시도되었습니다. 반복 튜닝 금지

**거부 조건:** [`db.execute("SELECT 1 FROM exposures WHERE start&lt;? AND end&gt;?", (iso(end), iso(start))).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1030)

**오류 메시지:** 시험 기간이 이전 학습/검증에 노출되어 미사용 시험으로 평가할 수 없습니다

**거부 조건:** [`pending is None or market != expected`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1217)

**오류 메시지:** 특징 선행 준비의 시장 순서가 다릅니다

**거부 조건:** [`columns is not None and cols != columns`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1288)

**오류 메시지:** 코인별 피처 스키마가 다릅니다

**거부 조건:** [`Path(filename).name != filename or not (directory / filename).is_file()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1470)

**오류 메시지:** 완성 모델 파일이 없습니다

**거부 조건:** [`any(set(values) != set(MARKETS) for values in (frames, probabilities, rules))`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1147)

**오류 메시지:** 종목 전환 후보는 다섯 시장 모두의 후속 평가가 필요합니다

**거부 조건:** [`cooldowns[market] is None`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L737)

**오류 메시지:** 재진입 보류 만료시각이 없습니다

**거부 조건:** [`pd.isna(until)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L740)

**오류 메시지:** 재진입 보류 만료시각이 유효하지 않습니다


#### LEVEL 3 · [execution_probe.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py)


| 내용 |
| --- |
| **설정·옵션:** [`--config`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L60)<br>**코드의 선언:** `type=Path; default=engine.ROOT / 'engine-config.json'` |
| **설정·옵션:** [`SCOPE`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L13)<br>**코드의 선언:** `정의='체결 경로 진단 · 전략 성과 아님'` |

**거부 조건:** [`account.state['positions']`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L48)

**오류 메시지:** 별도 진단 계좌에 미청산 잔량이 남았습니다

**거부 조건:** [`len(payload) != 1 or payload[0].get('market') != market or not payload[0].get('orderbook_units')`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L28)

**오류 메시지:** 진단 시장의 호가가 없습니다

**거부 조건:** [`observed and now &lt;= observed[-1]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L35)

**오류 메시지:** 호가 관측 시각이 증가하지 않았습니다

**거부 조건:** [`not math.isfinite(bid + ask) or not 0 &lt; bid &lt;= ask`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L39)

**오류 메시지:** 호가 가격 또는 스프레드가 유효하지 않습니다


#### LEVEL 3 · [healthcheck.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py)


| 내용 |
| --- |
| **설정·옵션:** [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L18)<br>**코드의 선언:** `정의=Path(__file__).resolve().parent` |
| **설정·옵션:** [`INTEGRITY_TIMEOUT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L19)<br>**코드의 선언:** `정의=1800` |
| **설정·옵션:** [`INTEGRITY_MAX_AGE`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L20)<br>**코드의 선언:** `정의=21600` |
| **설정·옵션:** [`INTEGRITY_RETRY`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L21)<br>**코드의 선언:** `정의=900` |

**거부 조건:** [`database_identity(database) != identity`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L63)

**오류 메시지:** 검사 중 저장소 파일이 교체되었습니다

**거부 조건:** [`time.monotonic() &gt;= deadline`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L42)

**오류 메시지:** 전체 무결성 검사 시간 제한 초과

**거부 조건:** [`result['state'] != 'ok' or result['database'] != database_identity(database)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L86)

**오류 메시지:** 전체 검사 결과와 저장소가 일치하지 않습니다


#### LEVEL 3 · [supervisor.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py)


| 내용 |
| --- |
| **설정·옵션:** [`--config`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L176)<br>**코드의 선언:** `type=Path; default=ROOT / "service.json"` |
| **설정·옵션:** [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L19)<br>**코드의 선언:** `정의=Path(__file__).resolve().parent` |
| **설정·옵션:** [`STOP`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L20)<br>**코드의 선언:** `정의=threading.Event()` |

**거부 조건:** [`config.get("initial_capital") != 10_000_000 or paper.get("initial_cash") != config.get("initial_capital")             or config.get("market_mode") != "spot-paper" or config.get("notifications") is not False`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L162)

**오류 메시지:** 원금 1,000만원·현물 모의거래·알림 없음 설정을 확인하십시오.

**거부 조건:** [`start_at.tzinfo is None or engine_start.tzinfo is None or start_at != engine_start`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L167)

**오류 메시지:** 서비스와 엔진의 시작 시각·시간대가 일치해야 합니다.

**거부 조건:** [`config["trainer"]["interval_seconds"] != 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L169)

**오류 메시지:** 성공한 학습 회차의 고정 대기는 0초여야 합니다.

**거부 조건:** [`not STOP.is_set()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L128)

**오류 메시지:** 작업 시간 제한 초과


</details>

### LEVEL 3 · 실제 사용·실험 순서

1. 모의매매와 GPU 연구·외부자료 수집의 독립 트랙을 확인합니다.
2. 설정·시점 경계·PAPER_ONLY와 실제 자료 준비 상태를 읽습니다.
3. 해당 요청의 수집·연구 조건이 충족될 때만 지정된 한 회차를 수행합니다.
4. 대시보드와 저장 지표에서 완료·대기·오류를 구분하고 기존 장부·모델을 보존합니다.

**단계:** 입력

**내용:** 시세·호가·완료 분봉·외부 아카이브·연구 설정

**단계:** 처리

**내용:** 검증·저장 → 특징 준비·연구 → 정책 평가 → 모의체결·화면

**단계:** 출력

**내용:** 자료 준비 상태·연구 지표·모의계좌 장부·대시보드


### LEVEL 3 · 결과와 실패를 확인하기

자료 크기·고유 아카이브 크기·DB 크기를 구분하고 준비/학습/승격 상태를 함께 확인합니다. 모의 손익은 실제 주문 성과가 아닙니다. GPU 사용률·처리량 개선도 수익 개선과 별개입니다.

**상황:** 연구가 대기함

**확인할 내용:** 자료량·최근 분봉 커버리지·최신성·GPU 사전 조건을 확인합니다. 대기 상태를 임의 학습 성공으로 바꾸지 않습니다.

**상황:** 외부 자료 검증 실패

**확인할 내용:** 공식 아카이브·체크섬·원문 체결 형식·완료봉을 확인합니다. 보간이나 기존 봉 덮어쓰기로 우회하지 않습니다.

**상황:** 차수 폴더만 실행할 수 없음

**확인할 내용:** 루트 공통 엔진·설정·검사와 차수 연결 파일의 실제 import 경로를 확인합니다.


### LEVEL 3 · 실행 전 대조할 차수 조건

기존 공개 자료의 차수 표기는 `기본`이며 번호를 임의로 소급 부여하지 않습니다. 이번 추가 기록은 날짜로 구분합니다. 활성 실행 폴더를 복제하지 않고 기존 서비스에 적용하며 이전 코드는 Git 이력과 로컬 소스 전용 압축본으로 보존합니다.

`CandleStore.read`는 완료 시각 범위 인덱스와 같은 순서로 조회해 임시 정렬을 제거합니다. 반환 행의 완료 시각 간격이 시간봉과 다르면 거부합니다. 기존 요청 범위·제한 수·최종 시작 시각 오름차순은 유지합니다.

`gpu_matrix_scope`는 같은 시장·같은 입력 객체·같은 열·같은 가중치 설정의 후보 묶음에서만 두 DMatrix를 재사용합니다. 범위 밖 호출과 다른 시장·가중치는 공유하지 않으며 종료 시 참조를 해제합니다. 해당 범위의 입력은 읽기 전용이며 제자리 변경은 지원하지 않습니다. 기존 CUDA 사전 검사와 매 반복 검사도 그대로 수행합니다.

`market_feature_loader`는 하나의 CPU 작업자로 다음 시장 특징만 선행 준비합니다. 누적 자료를 임의로 축소하지 않으며 학습·검증 범위는 같습니다. 준비 오류는 상위 호출로 전달하고, 본 작업 오류와 동시에 발생한 오류 원문도 예외에 남깁니다. 진행 중 SQL을 강제 종료하는 기능은 없습니다.

[자료 폴더](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89) · [직전 비교·자료 범위](../revision-audit/GPU%EC%B2%98%EB%A6%AC%EB%9F%89.md) · [이 차수의 원래 전체 상세](#reading-level-5)

<a id="reading-level-4"></a>
## LEVEL 4 - Advanced

### LEVEL 4 · 설계 의도와 검토 경계

기존 공개 자료의 차수 표기는 `기본`이며 번호를 임의로 소급 부여하지 않습니다. 이번 추가 기록은 날짜로 구분합니다. 활성 실행 폴더를 복제하지 않고 기존 서비스에 적용하며 이전 코드는 Git 이력과 로컬 소스 전용 압축본으로 보존합니다.

기존 공개 자료의 차수 표기는 `기본`이며 번호를 임의로 소급 부여하지 않습니다. 이번 추가 기록은 날짜로 구분합니다. 활성 실행 폴더를 복제하지 않고 기존 서비스에 적용하며 이전 코드는 Git 이력과 로컬 소스 전용 압축본으로 보존합니다.

`CandleStore.read`는 완료 시각 범위 인덱스와 같은 순서로 조회해 임시 정렬을 제거합니다. 반환 행의 완료 시각 간격이 시간봉과 다르면 거부합니다. 기존 요청 범위·제한 수·최종 시작 시각 오름차순은 유지합니다.

`gpu_matrix_scope`는 같은 시장·같은 입력 객체·같은 열·같은 가중치 설정의 후보 묶음에서만 두 DMatrix를 재사용합니다. 범위 밖 호출과 다른 시장·가중치는 공유하지 않으며 종료 시 참조를 해제합니다. 해당 범위의 입력은 읽기 전용이며 제자리 변경은 지원하지 않습니다. 기존 CUDA 사전 검사와 매 반복 검사도 그대로 수행합니다.

`market_feature_loader`는 하나의 CPU 작업자로 다음 시장 특징만 선행 준비합니다. 누적 자료를 임의로 축소하지 않으며 학습·검증 범위는 같습니다. 준비 오류는 상위 호출로 전달하고, 본 작업 오류와 동시에 발생한 오류 원문도 예외에 남깁니다. 진행 중 SQL을 강제 종료하는 기능은 없습니다.

### LEVEL 4 · 이 차수 구현의 실제 근거

| 내용 |
| --- |
| **파일:** [engine.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py)<br>**원문에 정의된 역할·구성:** 공개 시세만 사용하는 GPU 학습 및 현물 모의투자 엔진. |
| **파일:** [dashboard.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py)<br>**원문에 정의된 역할·구성:** 로컬 전용 읽기 전용 현물 모의거래 대시보드입니다. |
| **파일:** [execution_probe.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py)<br>**원문에 정의된 역할·구성:** 실제 공개 호가의 체결 경로를 전략 계좌와 별도로 검증합니다. |
| **파일:** [healthcheck.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py)<br>**원문에 정의된 역할·구성:** 실제 운용 파일과 시세·현물 계좌의 불변 조건을 주기적으로 검사합니다. |
| **파일:** [supervisor.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py)<br>**원문에 정의된 역할·구성:** 수집·학습·현물 모의거래를 독립 실행하고 상태를 보존합니다. |
| **파일:** [upbit_auto_trader_v5.ipynb](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/upbit_auto_trader_v5.ipynb)<br>**원문에 정의된 역할·구성:** 노트북 16개 셀 |
| **파일:** [tests/test_artifact_scope.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/tests/test_artifact_scope.py)<br>**원문에 정의된 역할·구성:** Linux 실행 의존성을 프로젝트 직접 구현 소스와 구분하는지 검사합니다. |
| **파일:** [tests/test_capital_contract.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/tests/test_capital_contract.py)<br>**원문에 정의된 역할·구성:** 3차 원금·기존 장부 경계와 실제 대시보드 필드를 검사합니다. |
| **파일:** [tests/test_collection_recovery.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/tests/test_collection_recovery.py)<br>**원문에 정의된 역할·구성:** 손상된 원본 행을 보존하면서 과거 수집을 계속하는지 검사합니다. |

**단계:** 입력

**내용:** 시세·호가·완료 분봉·외부 아카이브·연구 설정

**단계:** 처리

**내용:** 검증·저장 → 특징 준비·연구 → 정책 평가 → 모의체결·화면

**단계:** 출력

**내용:** 자료 준비 상태·연구 지표·모의계좌 장부·대시보드


### LEVEL 4 · 조건·변경·결과를 함께 검토하기

requirements와 engine-config.json, CUDA·GPU·DB 및 자료 저장 경로를 확인합니다. 실제 거래 계정·운영 DB·모델·개인 로그는 공개 코드와 별도입니다. 외부 수집에는 네트워크·디스크 여유·체크섬과 최신성 조건이 필요합니다.

이 차수 폴더만이 독립 실행 환경인지 먼저 확인합니다. 3차 외부자료 모듈과 4·5차 제어는 루트 engine.py·engine-config.json 및 tests와 연결됩니다. 실행 명령과 운영 경로는 해당 차수 README의 계약을 따릅니다.

#### LEVEL 4 · 이 보관본의 의존성 목록

[requirements.txt](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/requirements.txt)의 실제 항목입니다. 외부 실행 도구와 GPU 드라이버는 별도로 준비합니다.

```text
numpy>=1.26.4,<3
pandas>=2.2.3,<3
xgboost==3.0.5
```

#### LEVEL 4 · 소스에서 제공하는 CLI 옵션

**진입 소스:** [dashboard.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py)

**정의된 옵션 이름:** `--host`, `--port`

**진입 소스:** [engine.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py)

**정의된 옵션 이름:** `--config`, `--backfill`, `--once`, `--pages`, `--paper-start`, `--final`

**진입 소스:** [execution_probe.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py)

**정의된 옵션 이름:** `--config`

**진입 소스:** [supervisor.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py)

**정의된 옵션 이름:** `--config`


각 옵션의 기본값·필수값·부작용은 해당 main과 인자 정의를 함께 확인합니다.


<details>
<summary>함수 역할·기본값·입력 거부 조건의 세부 참조</summary>

다음 항목은 이 자료의 Python 소스에 명시된 선언입니다. 전체 분기와 호출 순서는 각 행의 코드 위치에서 이어서 읽을 수 있습니다.

#### LEVEL 4 · [dashboard.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py)

| 내용 |
| --- |
| **정의:** [`diagnostic`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L73)<br>**소스에 적힌 역할:** 허용 키·깊이·항목 수·총 문자수로 로컬 진단을 제한합니다. |
| **정의:** [`execution_probe`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L105)<br>**소스에 적힌 역할:** 본 계좌와 합산하지 않는 공개호가 경로 검사만 투영합니다. |
| **정의:** [`project`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L123)<br>**소스에 적힌 역할:** 허용한 필드의 단순 값만 추출합니다. |
| **정의:** [`SnapshotAdapter`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L289)<br>**소스에 적힌 역할:** 경로를 외부 입력으로 받지 않는 제한된 스냅숏 어댑터입니다. |

| 내용 |
| --- |
| **설정·옵션:** [`--host`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L554)<br>**코드의 선언:** `default="127.0.0.1"` |
| **설정·옵션:** [`--port`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L555)<br>**코드의 선언:** `type=int; default=8765` |
| **설정·옵션:** [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L15)<br>**코드의 선언:** `정의=Path(__file__).resolve().parent` |
| **설정·옵션:** [`MAX_JSON_BYTES`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L16)<br>**코드의 선언:** `정의=4 * 1024 * 1024` |
| **설정·옵션:** [`COINS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L17)<br>**코드의 선언:** `정의=("BTC", "SOL", "ETH", "XRP", "DOGE")` |
| **설정·옵션:** [`MAX_DIAGNOSTIC_DEPTH`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L67)<br>**코드의 선언:** `정의=8` |
| **설정·옵션:** [`MAX_DIAGNOSTIC_ITEMS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L68)<br>**코드의 선언:** `정의=100` |
| **설정·옵션:** [`MAX_DIAGNOSTIC_NODES`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L69)<br>**코드의 선언:** `정의=1500` |
| **설정·옵션:** [`MAX_DIAGNOSTIC_CHARS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L70)<br>**코드의 선언:** `정의=24000` |

#### LEVEL 4 · [engine.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py)

| 내용 |
| --- |
| **정의:** [`sweep_book`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L582)<br>**소스에 적힌 역할:** 매수 amount는 수수료 포함 원화, 매도 amount는 현물 수량입니다. |
| **정의:** [`MarketUniverseTransitionError`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1431)<br>**소스에 적힌 역할:** 검사를 통과한 기존 네 시장 모델도 다섯 시장의 운용에는 사용할 수 없습니다. |

**설정·옵션:** [`--config`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1641)

**코드의 선언:** `type=Path; default=None`

**설정·옵션:** [`--backfill`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1644)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`--once`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1645)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`--pages`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1646)

**코드의 선언:** `type=int`

**설정·옵션:** [`--once`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1648)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`--once`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1650)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`--paper-start`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1651)

**코드의 선언:** ``

**설정·옵션:** [`--final`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1653)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L27)

**코드의 선언:** `정의=Path(__file__).resolve().parent`

**설정·옵션:** [`RUNTIME_WORK`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L28)

**코드의 선언:** `정의=(ROOT / "work").resolve()`

**설정·옵션:** [`MARKETS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L29)

**코드의 선언:** `정의=("KRW-BTC", "KRW-SOL", "KRW-ETH", "KRW-XRP", "KRW-DOGE")`

**설정·옵션:** [`TIMEFRAMES`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L30)

**코드의 선언:** `정의=(1, 3, 5, 10, 15, 30, 60, 240, "day")`

**설정·옵션:** [`PAPER_ONLY`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L31)

**코드의 선언:** `정의=True`

**설정·옵션:** [`TRAINING_ETA`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L32)

**코드의 선언:** `정의=0.05`

**설정·옵션:** [`EXECUTION_POLICY`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L33)

**코드의 선언:** `정의="horizon_or_risk_v2"`

**설정·옵션:** [`PROTECTED_HOLDOUT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L34)

**코드의 선언:** `정의=("2026-10-19T00:00:00Z", "2026-10-20T00:00:00Z")`


**거부 조건:** [`sorted(config["markets"]) != sorted(MARKETS)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L106)

**오류 메시지:** 모의 대상은 BTC/SOL/ETH/XRP/DOGE 다섯 종목으로 고정됩니다

**거부 조건:** [`len(set(map(str, config["timeframes"]))) != len(config["timeframes"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L108)

**오류 메시지:** 중복 시간봉입니다

**거부 조건:** [`any(str(tf) not in set(map(str, TIMEFRAMES)) for tf in config["timeframes"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L110)

**오류 메시지:** 지원하지 않는 시간봉입니다

**거부 조건:** [`str(config["base_timeframe"]) not in set(map(str, config["timeframes"]))`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L112)

**오류 메시지:** 기본 시간봉이 수집 목록에 없습니다

**거부 조건:** [`tr["gap_bars"] &lt; tr["horizon"] + 1`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L125)

**오류 메시지:** 분할 간격은 다음봉 체결을 포함한 horizon+1 이상이어야 합니다

**거부 조건:** [`not tr["candidate_depths"] or any(not isinstance(d, int) or not 1 &lt;= d &lt;= 10 for d in tr["candidate_depths"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L127)

**오류 메시지:** 후보 깊이는 1~10 범위입니다

**거부 조건:** [`not 0 &lt; paper["initial_cash"] or paper["min_order"] &lt; 5000`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L132)

**오류 메시지:** 초기 자본과 최소 주문금액을 확인하세요

**거부 조건:** [`paper["max_allocation"] &gt; paper["max_total_allocation"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L134)

**오류 메시지:** 코인별 배분이 전체 한도를 초과합니다

**거부 조건:** [`isinstance(tr["class_weight_cap"], bool) or not math.isfinite(tr["class_weight_cap"]) or             not 1 &lt;= tr["class_weight_cap"] &lt;= 3`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L140)

**오류 메시지:** 클래스 가중치 상한은 1~3입니다

**거부 조건:** [`not tr["horizon_candidates"] or any(isinstance(h, bool) or not isinstance(h, int) or not 1 &lt;= h &lt;= 288                                            for h in tr["horizon_candidates"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L147)

**오류 메시지:** 지평 후보는 1~288봉 정수입니다

**거부 조건:** [`not tr["threshold_candidates"] or any(isinstance(t, bool) or not math.isfinite(t) or not 0 &lt; t &lt; 1                                               for t in [tr["label_threshold"], *tr["threshold_candidates"]])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L150)

**오류 메시지:** 비용 차감 후 라벨 임계값이 유효하지 않습니다

**거부 조건:** [`collection["request_interval"] &lt; 0.11 or not 1 &lt;= collection["retries"] &lt;= 8`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L153)

**오류 메시지:** 공유 요청 간격은 0.11초 이상, 재시도는 1~8회입니다

**거부 조건:** [`min(collection["recent_pages"], collection["backfill_pages_per_cycle"], collection["poll_seconds"], paper["poll_seconds"]) &lt;= 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L155)

**오류 메시지:** 수집 페이지와 대기 시간은 양수여야 합니다

**거부 조건:** [`sys.platform == "win32" and path.resolve().is_relative_to(RUNTIME_WORK)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L165)

**오류 메시지:** 운영 SQLite는 도커 내부에서만 접근하십시오

**거부 조건:** [`endpoint != "orderbook" and not endpoint.startswith("candles/")`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L251)

**오류 메시지:** 공개 캔들/호가 읽기만 허용됩니다

**거부 조건:** [`not np.isfinite(values).all() or (values[:, :4] &lt;= 0).any() or (values[:, 4] &lt; 0).any()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L293)

**오류 메시지:** 캔들 값이 유효하지 않습니다

**거부 조건:** [`((frame.high &lt; frame[["open", "close", "low"]].max(axis=1)) &#124;         (frame.low &gt; frame[["open", "close"]].min(axis=1))).any()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L295)

**오류 메시지:** OHLC 범위가 유효하지 않습니다

**거부 조건:** [`count &lt;= 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L328)

**오류 메시지:** 페이지 수는 양수여야 합니다

**거부 조건:** [`horizon &lt;= 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L402)

**오류 메시지:** 예측 지평은 양수여야 합니다

**거부 조건:** [`bar_length &lt;= pd.Timedelta(0) or not (lengths == bar_length).all()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L410)

**오류 메시지:** 라벨 봉 길이가 일관되지 않습니다

**거부 조건:** [`isinstance(sequence, bool) or not isinstance(sequence, int) or sequence &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L463)

**오류 메시지:** 연구 회차 커서가 손상되었습니다; 초기화하지 않습니다

**거부 조건:** [`p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p &lt; 0).any() or (p.sum(axis=1) &lt;= 0).any()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L484)

**오류 메시지:** 세 클래스 확률 배열이 유효하지 않습니다

**거부 조건:** [`not len(y) or len(y) != len(pred)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L561)

**오류 메시지:** 전체 표본 평가를 위한 행이 없습니다

**거부 조건:** [`side not in ("buy", "sell") or not math.isfinite(amount) or amount &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L584)

**오류 메시지:** 현물 모의체결 입력 오류

**거부 조건:** [`not math.isfinite(volatility) or volatility &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L612)

**오류 메시지:** 현재 변동성이 유효하지 않습니다

**거부 조건:** [`common is None or len(common) &lt; 2`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L820)

**오류 메시지:** 공통계좌 백테스트에 공통 완료 봉이 부족합니다

**거부 조건:** [`not xgb.build_info().get("USE_CUDA", False)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L915)

**오류 메시지:** 설치한 XGBoost가 CUDA를 지원하지 않습니다

**거부 조건:** [`not tr["holdout_start"] or not tr["holdout_end"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1060)

**오류 메시지:** training.holdout_start/holdout_end를 고정 UTC 시각으로 명시해야 합니다

**거부 조건:** [`he &lt;= hs`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1063)

**오류 메시지:** 잠금 시험 종료는 시작 이후여야 합니다

**거부 조건:** [`abs((he - hs).total_seconds() / 86400 - tr["holdout_days"]) &gt; 1e-6`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1065)

**오류 메시지:** holdout_days와 명시 시험 기간이 일치하지 않습니다

**거부 조건:** [`final and live`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1244)

**오류 메시지:** 실시간 학습은 잠금 시험을 평가할 수 없습니다

**거부 조건:** [`final and utc(None) &lt; plan["holdout"][1]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1252)

**오류 메시지:** 잠금 시험 기간이 끝나지 않았습니다

**거부 조건:** [`Path(pointer["run"]).name != pointer["run"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1441)

**오류 메시지:** 모델 포인터 경로 오류

**거부 조건:** [`stable_hash(manifest) != pointer["manifest_hash"] or not manifest["paper_only"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1445)

**오류 메시지:** 완성 모델 manifest가 일치하지 않습니다

**거부 조건:** [`manifest.get("run") != pointer["run"] or (package_markets != set(MARKETS) and not known_transition)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1457)

**오류 메시지:** 완성 모델의 실행 ID/다섯 코인 구성이 일치하지 않습니다

**거부 조건:** [`not isinstance(columns, list) or not columns or len(set(columns)) != len(columns)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1460)

**오류 메시지:** 완성 모델의 피처 목록이 유효하지 않습니다

**거부 조건:** [`not target.is_relative_to((ROOT / "work").resolve())`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L116)

**오류 메시지:** 유동 산출물은 프로젝트 work 내부에만 저장합니다

**거부 조건:** [`not isinstance(row["candle_date_time_utc"], str)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L279)

**오류 메시지:** 캔들 시각이 유효하지 않습니다

**거부 조건:** [`pd.isna(stamp)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L282)

**오류 메시지:** 캔들 시각이 유효하지 않습니다

**거부 조건:** [`not math.isfinite(price + size) or price &lt;= 0 or size &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L591)

**오류 메시지:** 호가 가격/잔량 오류

**거부 조건:** [`state is not None and (not isinstance(state, dict) or                                   state.get("initial_cash") != initial or state.get("paper_only") is not True)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L626)

**오류 메시지:** 복원 계좌의 초기 원금 또는 모의 전용 표시가 설정과 다릅니다

**거부 조건:** [`self.state["cash"] &lt; 0 or any(v["qty"] &lt; 0 for v in self.state["positions"].values())`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L635)

**오류 메시지:** 현물 계좌에 음수 잔액/공매도가 있습니다

**거부 조건:** [`any(m not in MARKETS for m in prices) or any(v not in (0, 1, 2) for v in signals.values())`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L655)

**오류 메시지:** 지원하지 않는 현물 시장/신호입니다

**거부 조건:** [`not all(math.isfinite(metrics[key]) for key in ("logloss", "net_return", "MDD"))`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1115)

**오류 메시지:** 모델 승격 비교 지표가 유한하지 않습니다

**거부 조건:** [`market not in MARKETS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1463)

**오류 메시지:** 지원하지 않는 모델 시장

**거부 조건:** [`not package["models"] or not len(package["models"]) == len(package["weights"]) == len(package["gpu"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1465)

**오류 메시지:** 완성 모델의 후보/가중치/적합 증거 수가 다릅니다

**거부 조건:** [`any(not math.isfinite(w) or w &lt; 0 for w in package["weights"]) or not math.isclose(sum(package["weights"]), 1, abs_tol=1e-6)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1467)

**오류 메시지:** 완성 모델의 확률 가중치가 유효하지 않습니다

**거부 조건:** [`set(quotes) != set(config["markets"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1572)

**오류 메시지:** 다섯 코인의 공개 호가가 모두 필요합니다

**거부 조건:** [`not isinstance(payload, list)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L260)

**오류 메시지:** 공개 시세 응답이 배열이 아닙니다

**거부 조건:** [`db.execute("SELECT 1 FROM final_tests WHERE start&lt;? AND end&gt;?", (end, start)).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1020)

**오류 메시지:** 이 시험 기간은 이미 소비 또는 평가 시도되었습니다. 반복 튜닝 금지

**거부 조건:** [`db.execute("SELECT 1 FROM exposures WHERE start&lt;? AND end&gt;?", (end, start)).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1022)

**오류 메시지:** 시험 기간이 이전 학습/검증에 노출되어 미사용 시험으로 평가할 수 없습니다

**거부 조건:** [`db.execute("SELECT 1 FROM final_tests WHERE start&lt;? AND end&gt;?", (iso(end), iso(start))).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1028)

**오류 메시지:** 이 시험 기간은 이미 소비 또는 평가 시도되었습니다. 반복 튜닝 금지

**거부 조건:** [`db.execute("SELECT 1 FROM exposures WHERE start&lt;? AND end&gt;?", (iso(end), iso(start))).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1030)

**오류 메시지:** 시험 기간이 이전 학습/검증에 노출되어 미사용 시험으로 평가할 수 없습니다

**거부 조건:** [`pending is None or market != expected`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1217)

**오류 메시지:** 특징 선행 준비의 시장 순서가 다릅니다

**거부 조건:** [`columns is not None and cols != columns`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1288)

**오류 메시지:** 코인별 피처 스키마가 다릅니다

**거부 조건:** [`Path(filename).name != filename or not (directory / filename).is_file()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1470)

**오류 메시지:** 완성 모델 파일이 없습니다

**거부 조건:** [`any(set(values) != set(MARKETS) for values in (frames, probabilities, rules))`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1147)

**오류 메시지:** 종목 전환 후보는 다섯 시장 모두의 후속 평가가 필요합니다

**거부 조건:** [`cooldowns[market] is None`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L737)

**오류 메시지:** 재진입 보류 만료시각이 없습니다

**거부 조건:** [`pd.isna(until)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L740)

**오류 메시지:** 재진입 보류 만료시각이 유효하지 않습니다


#### LEVEL 4 · [execution_probe.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py)


| 내용 |
| --- |
| **설정·옵션:** [`--config`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L60)<br>**코드의 선언:** `type=Path; default=engine.ROOT / 'engine-config.json'` |
| **설정·옵션:** [`SCOPE`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L13)<br>**코드의 선언:** `정의='체결 경로 진단 · 전략 성과 아님'` |

**거부 조건:** [`account.state['positions']`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L48)

**오류 메시지:** 별도 진단 계좌에 미청산 잔량이 남았습니다

**거부 조건:** [`len(payload) != 1 or payload[0].get('market') != market or not payload[0].get('orderbook_units')`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L28)

**오류 메시지:** 진단 시장의 호가가 없습니다

**거부 조건:** [`observed and now &lt;= observed[-1]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L35)

**오류 메시지:** 호가 관측 시각이 증가하지 않았습니다

**거부 조건:** [`not math.isfinite(bid + ask) or not 0 &lt; bid &lt;= ask`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L39)

**오류 메시지:** 호가 가격 또는 스프레드가 유효하지 않습니다


#### LEVEL 4 · [healthcheck.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py)


| 내용 |
| --- |
| **설정·옵션:** [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L18)<br>**코드의 선언:** `정의=Path(__file__).resolve().parent` |
| **설정·옵션:** [`INTEGRITY_TIMEOUT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L19)<br>**코드의 선언:** `정의=1800` |
| **설정·옵션:** [`INTEGRITY_MAX_AGE`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L20)<br>**코드의 선언:** `정의=21600` |
| **설정·옵션:** [`INTEGRITY_RETRY`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L21)<br>**코드의 선언:** `정의=900` |

**거부 조건:** [`database_identity(database) != identity`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L63)

**오류 메시지:** 검사 중 저장소 파일이 교체되었습니다

**거부 조건:** [`time.monotonic() &gt;= deadline`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L42)

**오류 메시지:** 전체 무결성 검사 시간 제한 초과

**거부 조건:** [`result['state'] != 'ok' or result['database'] != database_identity(database)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L86)

**오류 메시지:** 전체 검사 결과와 저장소가 일치하지 않습니다


#### LEVEL 4 · [supervisor.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py)


| 내용 |
| --- |
| **설정·옵션:** [`--config`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L176)<br>**코드의 선언:** `type=Path; default=ROOT / "service.json"` |
| **설정·옵션:** [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L19)<br>**코드의 선언:** `정의=Path(__file__).resolve().parent` |
| **설정·옵션:** [`STOP`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L20)<br>**코드의 선언:** `정의=threading.Event()` |

**거부 조건:** [`config.get("initial_capital") != 10_000_000 or paper.get("initial_cash") != config.get("initial_capital")             or config.get("market_mode") != "spot-paper" or config.get("notifications") is not False`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L162)

**오류 메시지:** 원금 1,000만원·현물 모의거래·알림 없음 설정을 확인하십시오.

**거부 조건:** [`start_at.tzinfo is None or engine_start.tzinfo is None or start_at != engine_start`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L167)

**오류 메시지:** 서비스와 엔진의 시작 시각·시간대가 일치해야 합니다.

**거부 조건:** [`config["trainer"]["interval_seconds"] != 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L169)

**오류 메시지:** 성공한 학습 회차의 고정 대기는 0초여야 합니다.

**거부 조건:** [`not STOP.is_set()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L128)

**오류 메시지:** 작업 시간 제한 초과


</details>

자료 크기·고유 아카이브 크기·DB 크기를 구분하고 준비/학습/승격 상태를 함께 확인합니다. 모의 손익은 실제 주문 성과가 아닙니다. GPU 사용률·처리량 개선도 수익 개선과 별개입니다.

**상황:** 연구가 대기함

**확인할 내용:** 자료량·최근 분봉 커버리지·최신성·GPU 사전 조건을 확인합니다. 대기 상태를 임의 학습 성공으로 바꾸지 않습니다.

**상황:** 외부 자료 검증 실패

**확인할 내용:** 공식 아카이브·체크섬·원문 체결 형식·완료봉을 확인합니다. 보간이나 기존 봉 덮어쓰기로 우회하지 않습니다.

**상황:** 차수 폴더만 실행할 수 없음

**확인할 내용:** 루트 공통 엔진·설정·검사와 차수 연결 파일의 실제 import 경로를 확인합니다.


### LEVEL 4 · 재현과 전체 비교

1. 모의매매와 GPU 연구·외부자료 수집의 독립 트랙을 확인합니다.
2. 설정·시점 경계·PAPER_ONLY와 실제 자료 준비 상태를 읽습니다.
3. 해당 요청의 수집·연구 조건이 충족될 때만 지정된 한 회차를 수행합니다.
4. 대시보드와 저장 지표에서 완료·대기·오류를 구분하고 기존 장부·모델을 보존합니다.

위 구성요소의 입력·출력과 실패 조건을 같은 차수 소스에 연결해 검토합니다. 전체 비교는 추가·수정·삭제와 원래 보관 범위를 구분하는 근거입니다.

[자료 폴더](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89) · [직전 비교·자료 범위](../revision-audit/GPU%EC%B2%98%EB%A6%AC%EB%9F%89.md) · [이 차수의 원래 전체 상세](#reading-level-5)
<a id="reading-level-5"></a>
## LEVEL 5 - Expert

기존의 전체 설명·표·코드·함수 참조·파일별 비교는 아래에 그대로 이어집니다. [자료 폴더](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89) · [직전 비교·자료 범위](../revision-audit/GPU%EC%B2%98%EB%A6%AC%EB%9F%89.md) · [이 차수의 원래 전체 상세](#reading-level-5)
<!-- reading-levels:end -->



[프로젝트](../../README.md) · [전체 차수](../REVISION_INDEX.md) · [원본 자료](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89)

<a name="-목적과-이-차수의-역할"></a>
## 목적과 이 차수의 역할

기존 공개 자료의 차수 표기는 `기본`이며 번호를 임의로 소급 부여하지 않습니다. 이번 추가 기록은 날짜로 구분합니다. 활성 실행 폴더를 복제하지 않고 기존 서비스에 적용하며 이전 코드는 Git 이력과 로컬 소스 전용 압축본으로 보존합니다.

<a name="features"></a>

<a name="-핵심-기능과-보존-범위"></a>
## 핵심 기능과 보존 범위

**항목:** 1

**이 차수에서 확인할 내용:** 기존 공개 자료의 차수 표기는 `기본`이며 번호를 임의로 소급 부여하지 않습니다. 이번 추가 기록은 날짜로 구분합니다. 활성 실행 폴더를 복제하지 않고 기존 서비스에 적용하며 이전 코드는 Git 이력과 로컬 소스 전용 압축본으로 보존합니다.

**항목:** 2

**이 차수에서 확인할 내용:** `CandleStore.read`는 완료 시각 범위 인덱스와 같은 순서로 조회해 임시 정렬을 제거합니다. 반환 행의 완료 시각 간격이 시간봉과 다르면 거부합니다. 기존 요청 범위·제한 수·최종 시작 시각 오름차순은 유지합니다.

**항목:** 3

**이 차수에서 확인할 내용:** `gpu_matrix_scope`는 같은 시장·같은 입력 객체·같은 열·같은 가중치 설정의 후보 묶음에서만 두 DMatrix를 재사용합니다. 범위 밖 호출과 다른 시장·가중치는 공유하지 않으며 종료 시 참조를 해제합니다. 해당 범위의 입력은 읽기 전용이며 제자리 변경은 지원하지 않습니다. 기존 CUDA 사전 검사와 매 반복 검사도 그대로 수행합니다.

**항목:** 4

**이 차수에서 확인할 내용:** `market_feature_loader`는 하나의 CPU 작업자로 다음 시장 특징만 선행 준비합니다. 누적 자료를 임의로 축소하지 않으며 학습·검증 범위는 같습니다. 준비 오류는 상위 호출로 전달하고, 본 작업 오류와 동시에 발생한 오류 원문도 예외에 남깁니다. 진행 중 SQL을 강제 종료하는 기능은 없습니다.


<a name="-입력처리출력"></a>
## 입력·처리·출력

**단계:** 입력

**내용:** 시세·호가·완료 분봉·외부 아카이브·연구 설정

**단계:** 처리

**내용:** 검증·저장 → 특징 준비·연구 → 정책 평가 → 모의체결·화면

**단계:** 출력

**내용:** 자료 준비 상태·연구 지표·모의계좌 장부·대시보드


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

| 내용 |
| --- |
| **경로:** [docs/](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/docs)<br>**자료 종류:** 하위 자료 폴더 |
| **경로:** [tests/](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/tests)<br>**자료 종류:** 하위 자료 폴더 |
| **경로:** [tools/](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/tools)<br>**자료 종류:** 하위 자료 폴더 |
| **경로:** [web/](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/web)<br>**자료 종류:** 하위 자료 폴더 |
| **경로:** [README.md](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/README.md)<br>**자료 종류:** MD 파일 |
| **경로:** [dashboard.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py)<br>**자료 종류:** PY 파일 |
| **경로:** [engine-config.json](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine-config.json)<br>**자료 종류:** JSON 파일 |
| **경로:** [engine.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py)<br>**자료 종류:** PY 파일 |
| **경로:** [execution_probe.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py)<br>**자료 종류:** PY 파일 |
| **경로:** [healthcheck.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py)<br>**자료 종류:** PY 파일 |
| **경로:** [package.json](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/package.json)<br>**자료 종류:** JSON 파일 |
| **경로:** [pnpm-lock.yaml](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/pnpm-lock.yaml)<br>**자료 종류:** YAML 파일 |
| **경로:** [pnpm-workspace.yaml](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/pnpm-workspace.yaml)<br>**자료 종류:** YAML 파일 |
| **경로:** [requirements.txt](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/requirements.txt)<br>**자료 종류:** TXT 파일 |
| **경로:** [service.json](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/service.json)<br>**자료 종류:** JSON 파일 |
| **경로:** [supervisor.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py)<br>**자료 종류:** PY 파일 |
| **경로:** [upbit_auto_trader_v5.ipynb](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/upbit_auto_trader_v5.ipynb)<br>**자료 종류:** IPYNB 파일 |
| **경로:** [운영.ps1](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/%EC%9A%B4%EC%98%81.ps1)<br>**자료 종류:** PS1 파일 |

<a name="-주요-모듈과-코드-읽기"></a>
## 주요 모듈과 코드 읽기

| 내용 |
| --- |
| **파일:** [engine.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py)<br>**원문에 정의된 역할·구성:** 공개 시세만 사용하는 GPU 학습 및 현물 모의투자 엔진. |
| **파일:** [dashboard.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py)<br>**원문에 정의된 역할·구성:** 로컬 전용 읽기 전용 현물 모의거래 대시보드입니다. |
| **파일:** [execution_probe.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py)<br>**원문에 정의된 역할·구성:** 실제 공개 호가의 체결 경로를 전략 계좌와 별도로 검증합니다. |
| **파일:** [healthcheck.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py)<br>**원문에 정의된 역할·구성:** 실제 운용 파일과 시세·현물 계좌의 불변 조건을 주기적으로 검사합니다. |
| **파일:** [supervisor.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py)<br>**원문에 정의된 역할·구성:** 수집·학습·현물 모의거래를 독립 실행하고 상태를 보존합니다. |
| **파일:** [upbit_auto_trader_v5.ipynb](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/upbit_auto_trader_v5.ipynb)<br>**원문에 정의된 역할·구성:** 노트북 16개 셀 |
| **파일:** [tests/test_artifact_scope.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/tests/test_artifact_scope.py)<br>**원문에 정의된 역할·구성:** Linux 실행 의존성을 프로젝트 직접 구현 소스와 구분하는지 검사합니다. |
| **파일:** [tests/test_capital_contract.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/tests/test_capital_contract.py)<br>**원문에 정의된 역할·구성:** 3차 원금·기존 장부 경계와 실제 대시보드 필드를 검사합니다. |
| **파일:** [tests/test_collection_recovery.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/tests/test_collection_recovery.py)<br>**원문에 정의된 역할·구성:** 손상된 원본 행을 보존하면서 과거 수집을 계속하는지 검사합니다. |

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

**진입 소스:** [dashboard.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py)

**정의된 옵션 이름:** `--host`, `--port`

**진입 소스:** [engine.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py)

**정의된 옵션 이름:** `--config`, `--backfill`, `--once`, `--pages`, `--paper-start`, `--final`

**진입 소스:** [execution_probe.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py)

**정의된 옵션 이름:** `--config`

**진입 소스:** [supervisor.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py)

**정의된 옵션 이름:** `--config`


각 옵션의 기본값·필수값·부작용은 해당 main과 인자 정의를 함께 확인합니다.


<details>
<summary>함수 역할·기본값·입력 거부 조건의 세부 참조</summary>

다음 항목은 이 자료의 Python 소스에 명시된 선언입니다. 전체 분기와 호출 순서는 각 행의 코드 위치에서 이어서 읽을 수 있습니다.

### [dashboard.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py)

| 내용 |
| --- |
| **정의:** [`diagnostic`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L73)<br>**소스에 적힌 역할:** 허용 키·깊이·항목 수·총 문자수로 로컬 진단을 제한합니다. |
| **정의:** [`execution_probe`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L105)<br>**소스에 적힌 역할:** 본 계좌와 합산하지 않는 공개호가 경로 검사만 투영합니다. |
| **정의:** [`project`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L123)<br>**소스에 적힌 역할:** 허용한 필드의 단순 값만 추출합니다. |
| **정의:** [`SnapshotAdapter`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L289)<br>**소스에 적힌 역할:** 경로를 외부 입력으로 받지 않는 제한된 스냅숏 어댑터입니다. |

| 내용 |
| --- |
| **설정·옵션:** [`--host`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L554)<br>**코드의 선언:** `default="127.0.0.1"` |
| **설정·옵션:** [`--port`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L555)<br>**코드의 선언:** `type=int; default=8765` |
| **설정·옵션:** [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L15)<br>**코드의 선언:** `정의=Path(__file__).resolve().parent` |
| **설정·옵션:** [`MAX_JSON_BYTES`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L16)<br>**코드의 선언:** `정의=4 * 1024 * 1024` |
| **설정·옵션:** [`COINS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L17)<br>**코드의 선언:** `정의=("BTC", "SOL", "ETH", "XRP", "DOGE")` |
| **설정·옵션:** [`MAX_DIAGNOSTIC_DEPTH`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L67)<br>**코드의 선언:** `정의=8` |
| **설정·옵션:** [`MAX_DIAGNOSTIC_ITEMS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L68)<br>**코드의 선언:** `정의=100` |
| **설정·옵션:** [`MAX_DIAGNOSTIC_NODES`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L69)<br>**코드의 선언:** `정의=1500` |
| **설정·옵션:** [`MAX_DIAGNOSTIC_CHARS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/dashboard.py#L70)<br>**코드의 선언:** `정의=24000` |

### [engine.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py)

| 내용 |
| --- |
| **정의:** [`sweep_book`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L582)<br>**소스에 적힌 역할:** 매수 amount는 수수료 포함 원화, 매도 amount는 현물 수량입니다. |
| **정의:** [`MarketUniverseTransitionError`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1431)<br>**소스에 적힌 역할:** 검사를 통과한 기존 네 시장 모델도 다섯 시장의 운용에는 사용할 수 없습니다. |

**설정·옵션:** [`--config`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1641)

**코드의 선언:** `type=Path; default=None`

**설정·옵션:** [`--backfill`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1644)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`--once`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1645)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`--pages`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1646)

**코드의 선언:** `type=int`

**설정·옵션:** [`--once`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1648)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`--once`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1650)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`--paper-start`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1651)

**코드의 선언:** ``

**설정·옵션:** [`--final`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1653)

**코드의 선언:** `action="store_true"`

**설정·옵션:** [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L27)

**코드의 선언:** `정의=Path(__file__).resolve().parent`

**설정·옵션:** [`RUNTIME_WORK`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L28)

**코드의 선언:** `정의=(ROOT / "work").resolve()`

**설정·옵션:** [`MARKETS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L29)

**코드의 선언:** `정의=("KRW-BTC", "KRW-SOL", "KRW-ETH", "KRW-XRP", "KRW-DOGE")`

**설정·옵션:** [`TIMEFRAMES`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L30)

**코드의 선언:** `정의=(1, 3, 5, 10, 15, 30, 60, 240, "day")`

**설정·옵션:** [`PAPER_ONLY`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L31)

**코드의 선언:** `정의=True`

**설정·옵션:** [`TRAINING_ETA`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L32)

**코드의 선언:** `정의=0.05`

**설정·옵션:** [`EXECUTION_POLICY`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L33)

**코드의 선언:** `정의="horizon_or_risk_v2"`

**설정·옵션:** [`PROTECTED_HOLDOUT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L34)

**코드의 선언:** `정의=("2026-10-19T00:00:00Z", "2026-10-20T00:00:00Z")`


**거부 조건:** [`sorted(config["markets"]) != sorted(MARKETS)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L106)

**오류 메시지:** 모의 대상은 BTC/SOL/ETH/XRP/DOGE 다섯 종목으로 고정됩니다

**거부 조건:** [`len(set(map(str, config["timeframes"]))) != len(config["timeframes"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L108)

**오류 메시지:** 중복 시간봉입니다

**거부 조건:** [`any(str(tf) not in set(map(str, TIMEFRAMES)) for tf in config["timeframes"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L110)

**오류 메시지:** 지원하지 않는 시간봉입니다

**거부 조건:** [`str(config["base_timeframe"]) not in set(map(str, config["timeframes"]))`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L112)

**오류 메시지:** 기본 시간봉이 수집 목록에 없습니다

**거부 조건:** [`tr["gap_bars"] &lt; tr["horizon"] + 1`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L125)

**오류 메시지:** 분할 간격은 다음봉 체결을 포함한 horizon+1 이상이어야 합니다

**거부 조건:** [`not tr["candidate_depths"] or any(not isinstance(d, int) or not 1 &lt;= d &lt;= 10 for d in tr["candidate_depths"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L127)

**오류 메시지:** 후보 깊이는 1~10 범위입니다

**거부 조건:** [`not 0 &lt; paper["initial_cash"] or paper["min_order"] &lt; 5000`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L132)

**오류 메시지:** 초기 자본과 최소 주문금액을 확인하세요

**거부 조건:** [`paper["max_allocation"] &gt; paper["max_total_allocation"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L134)

**오류 메시지:** 코인별 배분이 전체 한도를 초과합니다

**거부 조건:** [`isinstance(tr["class_weight_cap"], bool) or not math.isfinite(tr["class_weight_cap"]) or             not 1 &lt;= tr["class_weight_cap"] &lt;= 3`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L140)

**오류 메시지:** 클래스 가중치 상한은 1~3입니다

**거부 조건:** [`not tr["horizon_candidates"] or any(isinstance(h, bool) or not isinstance(h, int) or not 1 &lt;= h &lt;= 288                                            for h in tr["horizon_candidates"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L147)

**오류 메시지:** 지평 후보는 1~288봉 정수입니다

**거부 조건:** [`not tr["threshold_candidates"] or any(isinstance(t, bool) or not math.isfinite(t) or not 0 &lt; t &lt; 1                                               for t in [tr["label_threshold"], *tr["threshold_candidates"]])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L150)

**오류 메시지:** 비용 차감 후 라벨 임계값이 유효하지 않습니다

**거부 조건:** [`collection["request_interval"] &lt; 0.11 or not 1 &lt;= collection["retries"] &lt;= 8`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L153)

**오류 메시지:** 공유 요청 간격은 0.11초 이상, 재시도는 1~8회입니다

**거부 조건:** [`min(collection["recent_pages"], collection["backfill_pages_per_cycle"], collection["poll_seconds"], paper["poll_seconds"]) &lt;= 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L155)

**오류 메시지:** 수집 페이지와 대기 시간은 양수여야 합니다

**거부 조건:** [`sys.platform == "win32" and path.resolve().is_relative_to(RUNTIME_WORK)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L165)

**오류 메시지:** 운영 SQLite는 도커 내부에서만 접근하십시오

**거부 조건:** [`endpoint != "orderbook" and not endpoint.startswith("candles/")`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L251)

**오류 메시지:** 공개 캔들/호가 읽기만 허용됩니다

**거부 조건:** [`not np.isfinite(values).all() or (values[:, :4] &lt;= 0).any() or (values[:, 4] &lt; 0).any()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L293)

**오류 메시지:** 캔들 값이 유효하지 않습니다

**거부 조건:** [`((frame.high &lt; frame[["open", "close", "low"]].max(axis=1)) &#124;         (frame.low &gt; frame[["open", "close"]].min(axis=1))).any()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L295)

**오류 메시지:** OHLC 범위가 유효하지 않습니다

**거부 조건:** [`count &lt;= 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L328)

**오류 메시지:** 페이지 수는 양수여야 합니다

**거부 조건:** [`horizon &lt;= 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L402)

**오류 메시지:** 예측 지평은 양수여야 합니다

**거부 조건:** [`bar_length &lt;= pd.Timedelta(0) or not (lengths == bar_length).all()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L410)

**오류 메시지:** 라벨 봉 길이가 일관되지 않습니다

**거부 조건:** [`isinstance(sequence, bool) or not isinstance(sequence, int) or sequence &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L463)

**오류 메시지:** 연구 회차 커서가 손상되었습니다; 초기화하지 않습니다

**거부 조건:** [`p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p &lt; 0).any() or (p.sum(axis=1) &lt;= 0).any()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L484)

**오류 메시지:** 세 클래스 확률 배열이 유효하지 않습니다

**거부 조건:** [`not len(y) or len(y) != len(pred)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L561)

**오류 메시지:** 전체 표본 평가를 위한 행이 없습니다

**거부 조건:** [`side not in ("buy", "sell") or not math.isfinite(amount) or amount &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L584)

**오류 메시지:** 현물 모의체결 입력 오류

**거부 조건:** [`not math.isfinite(volatility) or volatility &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L612)

**오류 메시지:** 현재 변동성이 유효하지 않습니다

**거부 조건:** [`common is None or len(common) &lt; 2`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L820)

**오류 메시지:** 공통계좌 백테스트에 공통 완료 봉이 부족합니다

**거부 조건:** [`not xgb.build_info().get("USE_CUDA", False)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L915)

**오류 메시지:** 설치한 XGBoost가 CUDA를 지원하지 않습니다

**거부 조건:** [`not tr["holdout_start"] or not tr["holdout_end"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1060)

**오류 메시지:** training.holdout_start/holdout_end를 고정 UTC 시각으로 명시해야 합니다

**거부 조건:** [`he &lt;= hs`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1063)

**오류 메시지:** 잠금 시험 종료는 시작 이후여야 합니다

**거부 조건:** [`abs((he - hs).total_seconds() / 86400 - tr["holdout_days"]) &gt; 1e-6`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1065)

**오류 메시지:** holdout_days와 명시 시험 기간이 일치하지 않습니다

**거부 조건:** [`final and live`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1244)

**오류 메시지:** 실시간 학습은 잠금 시험을 평가할 수 없습니다

**거부 조건:** [`final and utc(None) &lt; plan["holdout"][1]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1252)

**오류 메시지:** 잠금 시험 기간이 끝나지 않았습니다

**거부 조건:** [`Path(pointer["run"]).name != pointer["run"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1441)

**오류 메시지:** 모델 포인터 경로 오류

**거부 조건:** [`stable_hash(manifest) != pointer["manifest_hash"] or not manifest["paper_only"]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1445)

**오류 메시지:** 완성 모델 manifest가 일치하지 않습니다

**거부 조건:** [`manifest.get("run") != pointer["run"] or (package_markets != set(MARKETS) and not known_transition)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1457)

**오류 메시지:** 완성 모델의 실행 ID/다섯 코인 구성이 일치하지 않습니다

**거부 조건:** [`not isinstance(columns, list) or not columns or len(set(columns)) != len(columns)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1460)

**오류 메시지:** 완성 모델의 피처 목록이 유효하지 않습니다

**거부 조건:** [`not target.is_relative_to((ROOT / "work").resolve())`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L116)

**오류 메시지:** 유동 산출물은 프로젝트 work 내부에만 저장합니다

**거부 조건:** [`not isinstance(row["candle_date_time_utc"], str)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L279)

**오류 메시지:** 캔들 시각이 유효하지 않습니다

**거부 조건:** [`pd.isna(stamp)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L282)

**오류 메시지:** 캔들 시각이 유효하지 않습니다

**거부 조건:** [`not math.isfinite(price + size) or price &lt;= 0 or size &lt; 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L591)

**오류 메시지:** 호가 가격/잔량 오류

**거부 조건:** [`state is not None and (not isinstance(state, dict) or                                   state.get("initial_cash") != initial or state.get("paper_only") is not True)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L626)

**오류 메시지:** 복원 계좌의 초기 원금 또는 모의 전용 표시가 설정과 다릅니다

**거부 조건:** [`self.state["cash"] &lt; 0 or any(v["qty"] &lt; 0 for v in self.state["positions"].values())`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L635)

**오류 메시지:** 현물 계좌에 음수 잔액/공매도가 있습니다

**거부 조건:** [`any(m not in MARKETS for m in prices) or any(v not in (0, 1, 2) for v in signals.values())`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L655)

**오류 메시지:** 지원하지 않는 현물 시장/신호입니다

**거부 조건:** [`not all(math.isfinite(metrics[key]) for key in ("logloss", "net_return", "MDD"))`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1115)

**오류 메시지:** 모델 승격 비교 지표가 유한하지 않습니다

**거부 조건:** [`market not in MARKETS`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1463)

**오류 메시지:** 지원하지 않는 모델 시장

**거부 조건:** [`not package["models"] or not len(package["models"]) == len(package["weights"]) == len(package["gpu"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1465)

**오류 메시지:** 완성 모델의 후보/가중치/적합 증거 수가 다릅니다

**거부 조건:** [`any(not math.isfinite(w) or w &lt; 0 for w in package["weights"]) or not math.isclose(sum(package["weights"]), 1, abs_tol=1e-6)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1467)

**오류 메시지:** 완성 모델의 확률 가중치가 유효하지 않습니다

**거부 조건:** [`set(quotes) != set(config["markets"])`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1572)

**오류 메시지:** 다섯 코인의 공개 호가가 모두 필요합니다

**거부 조건:** [`not isinstance(payload, list)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L260)

**오류 메시지:** 공개 시세 응답이 배열이 아닙니다

**거부 조건:** [`db.execute("SELECT 1 FROM final_tests WHERE start&lt;? AND end&gt;?", (end, start)).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1020)

**오류 메시지:** 이 시험 기간은 이미 소비 또는 평가 시도되었습니다. 반복 튜닝 금지

**거부 조건:** [`db.execute("SELECT 1 FROM exposures WHERE start&lt;? AND end&gt;?", (end, start)).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1022)

**오류 메시지:** 시험 기간이 이전 학습/검증에 노출되어 미사용 시험으로 평가할 수 없습니다

**거부 조건:** [`db.execute("SELECT 1 FROM final_tests WHERE start&lt;? AND end&gt;?", (iso(end), iso(start))).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1028)

**오류 메시지:** 이 시험 기간은 이미 소비 또는 평가 시도되었습니다. 반복 튜닝 금지

**거부 조건:** [`db.execute("SELECT 1 FROM exposures WHERE start&lt;? AND end&gt;?", (iso(end), iso(start))).fetchone()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1030)

**오류 메시지:** 시험 기간이 이전 학습/검증에 노출되어 미사용 시험으로 평가할 수 없습니다

**거부 조건:** [`pending is None or market != expected`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1217)

**오류 메시지:** 특징 선행 준비의 시장 순서가 다릅니다

**거부 조건:** [`columns is not None and cols != columns`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1288)

**오류 메시지:** 코인별 피처 스키마가 다릅니다

**거부 조건:** [`Path(filename).name != filename or not (directory / filename).is_file()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1470)

**오류 메시지:** 완성 모델 파일이 없습니다

**거부 조건:** [`any(set(values) != set(MARKETS) for values in (frames, probabilities, rules))`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L1147)

**오류 메시지:** 종목 전환 후보는 다섯 시장 모두의 후속 평가가 필요합니다

**거부 조건:** [`cooldowns[market] is None`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L737)

**오류 메시지:** 재진입 보류 만료시각이 없습니다

**거부 조건:** [`pd.isna(until)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/engine.py#L740)

**오류 메시지:** 재진입 보류 만료시각이 유효하지 않습니다


### [execution_probe.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py)


| 내용 |
| --- |
| **설정·옵션:** [`--config`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L60)<br>**코드의 선언:** `type=Path; default=engine.ROOT / 'engine-config.json'` |
| **설정·옵션:** [`SCOPE`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L13)<br>**코드의 선언:** `정의='체결 경로 진단 · 전략 성과 아님'` |

**거부 조건:** [`account.state['positions']`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L48)

**오류 메시지:** 별도 진단 계좌에 미청산 잔량이 남았습니다

**거부 조건:** [`len(payload) != 1 or payload[0].get('market') != market or not payload[0].get('orderbook_units')`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L28)

**오류 메시지:** 진단 시장의 호가가 없습니다

**거부 조건:** [`observed and now &lt;= observed[-1]`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L35)

**오류 메시지:** 호가 관측 시각이 증가하지 않았습니다

**거부 조건:** [`not math.isfinite(bid + ask) or not 0 &lt; bid &lt;= ask`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/execution_probe.py#L39)

**오류 메시지:** 호가 가격 또는 스프레드가 유효하지 않습니다


### [healthcheck.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py)


| 내용 |
| --- |
| **설정·옵션:** [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L18)<br>**코드의 선언:** `정의=Path(__file__).resolve().parent` |
| **설정·옵션:** [`INTEGRITY_TIMEOUT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L19)<br>**코드의 선언:** `정의=1800` |
| **설정·옵션:** [`INTEGRITY_MAX_AGE`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L20)<br>**코드의 선언:** `정의=21600` |
| **설정·옵션:** [`INTEGRITY_RETRY`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L21)<br>**코드의 선언:** `정의=900` |

**거부 조건:** [`database_identity(database) != identity`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L63)

**오류 메시지:** 검사 중 저장소 파일이 교체되었습니다

**거부 조건:** [`time.monotonic() &gt;= deadline`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L42)

**오류 메시지:** 전체 무결성 검사 시간 제한 초과

**거부 조건:** [`result['state'] != 'ok' or result['database'] != database_identity(database)`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/healthcheck.py#L86)

**오류 메시지:** 전체 검사 결과와 저장소가 일치하지 않습니다


### [supervisor.py](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py)


| 내용 |
| --- |
| **설정·옵션:** [`--config`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L176)<br>**코드의 선언:** `type=Path; default=ROOT / "service.json"` |
| **설정·옵션:** [`ROOT`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L19)<br>**코드의 선언:** `정의=Path(__file__).resolve().parent` |
| **설정·옵션:** [`STOP`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L20)<br>**코드의 선언:** `정의=threading.Event()` |

**거부 조건:** [`config.get("initial_capital") != 10_000_000 or paper.get("initial_cash") != config.get("initial_capital")             or config.get("market_mode") != "spot-paper" or config.get("notifications") is not False`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L162)

**오류 메시지:** 원금 1,000만원·현물 모의거래·알림 없음 설정을 확인하십시오.

**거부 조건:** [`start_at.tzinfo is None or engine_start.tzinfo is None or start_at != engine_start`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L167)

**오류 메시지:** 서비스와 엔진의 시작 시각·시간대가 일치해야 합니다.

**거부 조건:** [`config["trainer"]["interval_seconds"] != 0`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L169)

**오류 메시지:** 성공한 학습 회차의 고정 대기는 0초여야 합니다.

**거부 조건:** [`not STOP.is_set()`](../../%ED%9B%84%EC%86%8D%EB%B3%B4%EC%A1%B4/2026-10-30_GPU%EC%B2%98%EB%A6%AC%EB%9F%89/supervisor.py#L128)

**오류 메시지:** 작업 시간 제한 초과


</details>

<a name="results"></a>

<a name="-처리-결과와-출력-읽기"></a>
## 처리 결과와 출력 읽기

자료 크기·고유 아카이브 크기·DB 크기를 구분하고 준비/학습/승격 상태를 함께 확인합니다. 모의 손익은 실제 주문 성과가 아닙니다. GPU 사용률·처리량 개선도 수익 개선과 별개입니다.

<a name="️-제약과-문제-해결"></a>
## 제약과 문제 해결

**상황:** 연구가 대기함

**확인할 내용:** 자료량·최근 분봉 커버리지·최신성·GPU 사전 조건을 확인합니다. 대기 상태를 임의 학습 성공으로 바꾸지 않습니다.

**상황:** 외부 자료 검증 실패

**확인할 내용:** 공식 아카이브·체크섬·원문 체결 형식·완료봉을 확인합니다. 보간이나 기존 봉 덮어쓰기로 우회하지 않습니다.

**상황:** 차수 폴더만 실행할 수 없음

**확인할 내용:** 루트 공통 엔진·설정·검사와 차수 연결 파일의 실제 import 경로를 확인합니다.


<a name="changes"></a>

<a name="-직전-자료와-전체-비교"></a>
## 직전 자료와 전체 비교

이 자료의 보존 범위와 기준 파일을 연결합니다.

[전체 비교·자료 범위](../revision-audit/GPU%EC%B2%98%EB%A6%AC%EB%9F%89.md)

추가·수정·삭제, 주석·공백·개행과 바이너리 크기·해시는 연결된 원래 비교 자료에서 확인합니다.
