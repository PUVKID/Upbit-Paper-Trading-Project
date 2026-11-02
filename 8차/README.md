# 8차: 고정 자료 반복 학습

검증된 7차 완료 checkpoint에서 출발하여 동일한 고정 train/tune 자료로 `fit()`을 10번 호출합니다. 각 호출은 full fine-tuning 50 step이며 context 192, batch 8, learning rate 0.000001입니다. 500-step 단일 fit, 10 epoch, 독립평가 10회가 아닙니다.

새 독립평가는 수행하지 않습니다. 지표는 null이고 candidate_ready/promote는 false입니다. 실제 성능 향상·수익·방향 정확도는 이 실험으로 확인하지 않습니다. 모델은 미검증 train-only 연구 산출물이며 기존 모의계좌와 실매매에 적용하지 않습니다.

## 실제 결과

2026-11-03 RTX 4090의 CUDA에서 5코인 학습 자료 40,998행을 사용해 실제 fit 10회·총 500스텝을 실행했습니다. 첫 fit의 저장 검증 실패를 보존하고 추가 fit 없이 별도 재로드 복구 검사를 거친 뒤 나머지 9회만 추가 학습했습니다. 실제 완료 체크포인트 10개의 파일 해시·가중치 계보·각 50스텝을 대조했습니다.

학습 손실은 4.521545→3.839211로 감소했으나 고정 튜닝 손실은 7.453165→8.106275로 증가했습니다. 과적합 징후이며 성능 개선으로 판단하지 않습니다. 독립 미래 평가·정확도·수익은 미평가이고 매매 적용도 하지 않았습니다. [회차별 집계와 검증 범위](변경기록.md)를 함께 확인할 수 있습니다.

## 실행

구현 검사와 부모의 독립 사전 검토·실행 승인 후 Linux 컨테이너에서 실행합니다. 구현자는 실제 GPU 학습을 실행하지 않습니다.

```powershell
& 'C:/Program Files/Docker/Docker/resources/bin/docker.exe' exec -w '/workspace/2026/Upbit 5-Coin Project' 2026 /opt/venvs/torch/bin/python3 -B -X utf8 '8차/train_only_repeats.py' --help
& 'C:/Program Files/Docker/Docker/resources/bin/docker.exe' exec -w '/workspace/2026/Upbit 5-Coin Project' 2026 /opt/venvs/torch/bin/python3 -B -X utf8 '8차/train_only_repeats.py' --once --batch train-only-10-20261103 --config '8차/train-only-config.json'
```

시스템 torch와 기존 `7차/work/python-packages`의 고정 패키지를 7차 ChronosRuntime으로 재사용합니다. pip 설치, 새 가상환경·링크·다운로드, CPU 대체는 없습니다. `--help`는 ML 패키지를 import하지 않습니다.

## 재실행 정책

최소판은 중간 resume 대신 **예약 후 재실행 거부** 정책입니다. `8차/work/request.json` 또는 `8차/work/runs/*`가 있으면 같은 batch와 다른 이름의 batch 모두 거부합니다. 실패 checkpoint를 자동 삭제하거나 처음 가중치로 돌아가지 않습니다. 실패 전 완료된 회차의 manifest·checkpoint와 미완료 회차를 그대로 보존합니다. 미완료 fit을 자동으로 다시 수행하지 않습니다.

예약 전 준비 실패는 fit을 시작하지 않았으므로 재검사할 수 있습니다. 예약 후 중단된 실험의 별도 복구·새 실행 승인은 부모가 원본을 보존한 상태에서 판단해야 하며, 이 CLI에는 우회·초기화 옵션이 없습니다.

### 승인된 첫 회차 복구

수리71의 `--recover-first`는 수리70 원본 소스로 시작해 첫 fit 50 step 후 저장 검증에 실패한 **동일 batch만** 허용합니다. 기존 학습 1회를 소비한 채 두 번의 무학습 재로드 검사로 승계하고 rep2~10 최대 9회만 추가합니다. 원래 probe가 없어 당시 실패 원인을 소급 확정하거나 원래 저장 전후 검사를 PASS로 바꾸지 않습니다. `recovered_after_reload_check`라는 별도 복구 근거입니다. 진단 원문의 SHA256을 복구 origin·두 probe·manifest에 명시 결합합니다.

```powershell
& 'C:/Program Files/Docker/Docker/resources/bin/docker.exe' exec -w '/workspace/2026/Upbit 5-Coin Project' 2026 /opt/venvs/torch/bin/python3 -B -X utf8 '8차/train_only_repeats.py' --once --batch train-only-10-20261103 --config '8차/train-only-config.json' --recover-first
```

실행은 부모의 독립 사전 검토·승인 후 부모만 담당합니다. 첫 복구 예약 후에는 이 옵션도 재실행을 거부합니다. 원래 request/inputs/attempt/failure/미완료 completion은 보존하며 새 산출물은 `work/runs/<batch>/recovery-first/`에 기록합니다. 변경 전 status도 같은 경로의 `original-status.json`에 동일 바이트로 보존합니다. 새 aggregate에는 복구 사실과 실제 fit 소비 수를 표시합니다.

## 산출물

- `work/status.json`: 8차 진행 상태만 기록합니다. API 이름은 `training_repetition`이며 runner가 대시보드를 수정하지는 않습니다.
- `work/runs/<batch>/request.json`, `inputs.json`: 고정 기간·입력 해시·실제 내부 입력 tuple. 원시 입력은 공개하지 않습니다.
- `work/runs/<batch>/rep-01`부터 `rep-10`: 각 fit의 attempt, checkpoint, 완료 manifest. 직전 manifest 해시와 실제 가중치 해시를 연결합니다.
- `work/runs/<batch>/completion.json`: 실제 완료 manifest 수 0~10과 목록. 10개일 때만 complete입니다.
- `work/knowledge.json`: **미검증 train-only 전용** 포인터. 7차 knowledge와 분리됩니다.
- `work/evidence`: 실패 원문과 구현 검사 기록. 공개 대상이 아닙니다.

상세 경계는 [연구계약.md](연구계약.md)를 따릅니다. 공개 가능한 범위는 소스·설정·테스트·문서이며 work의 시세·모델·운영 원문은 제외합니다.
