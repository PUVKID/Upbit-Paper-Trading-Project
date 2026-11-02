"""고유 외부 자료의 초기 연구와 명시 요청 연구를 내구성 있게 제어합니다."""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import time
import uuid


INITIAL_BYTES = 50_000_000_000
INCREMENT_BYTES = 2_000_000_000
DEFAULT_SETTINGS = {"active": False, "continuous": False, "state_path": "4차/work/research_control.json"}
MARKETS = {"KRW-BTC", "KRW-SOL", "KRW-ETH", "KRW-XRP", "KRW-DOGE"}
BATCH_KEYS = {"id", "kind", "baseline_bytes", "target_bytes", "status", "requested_at",
              "attempts", "claim", "last_error", "run", "completed_at", "promoted"}
ATTEMPT_KEYS = {"schema_version", "batch_id", "claim", "attempt", "target_bytes", "run",
                "reports_dir", "models_dir", "previous_pointer_sha256", "previous_training_sha256",
                "phase", "report", "report_sha256", "manifest_sha256", "model_sha256", "error"}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def attempt_binding(record):
    return {key: record[key] for key in
            ("schema_version", "batch_id", "claim", "attempt", "target_bytes", "run")}


def validate_config(settings, root):
    if not isinstance(settings, dict) or set(settings) - set(DEFAULT_SETTINGS):
        raise ValueError("지원하지 않는 research 설정입니다")
    result = {**DEFAULT_SETTINGS, **settings}
    if type(result["active"]) is not bool:
        raise ValueError("research.active는 bool이어야 합니다")
    if type(result["continuous"]) is not bool:
        raise ValueError("research.continuous는 bool이어야 합니다")
    if result["continuous"] and not result["active"]:
        raise ValueError("연속 연구는 active 설정이 필요합니다")
    value = result["state_path"]
    if (not isinstance(value, str) or not value.strip() or "\x00" in value
            or value.startswith("file:") or (os.name != "nt" and (PureWindowsPath(value).drive or "\\" in value))):
        raise ValueError("research.state_path 경로가 유효하지 않습니다")
    root = Path(root).resolve()
    boundary = (root / "4차/work").resolve()
    target = (root / value).resolve()
    if (not boundary.is_relative_to(root) or not target.is_relative_to(boundary)
            or target == boundary or target.suffix != ".json"):
        raise ValueError("연구 상태는 프로젝트 4차/work 안의 JSON이어야 합니다")
    result["state_path"] = target.relative_to(root).as_posix()
    return result


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _utc_stamp(value):
    if not isinstance(value, str):
        raise ValueError("연구 시각 형식 오류")
    stamp = dt.datetime.fromisoformat(value)
    if stamp.tzinfo is None or stamp.utcoffset() != dt.timedelta(0):
        raise ValueError("연구 시각은 UTC여야 합니다")


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("연구 상태 JSON의 중복 키입니다")
        result[key] = value
    return result


@contextlib.contextmanager
def _file_lock(path, blocking=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("연구 잠금의 심볼릭 링크는 허용하지 않습니다")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "r+b") as stream:
        deadline = time.monotonic() + 30
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if not blocking or time.monotonic() >= deadline:
                    raise
                time.sleep(.05)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class ResearchControl:
    def __init__(self, root: Path, settings: dict):
        self.root = Path(root).resolve()
        self.settings = validate_config(settings, self.root)
        self.path = self.root / self.settings["state_path"]

    @contextlib.contextmanager
    def training_lock(self):
        """적합 중에도 Collector가 상태를 읽도록 짧은 상태 잠금과 분리합니다."""
        self._require_active()
        with _file_lock(self.path.with_suffix(".training.lock"), blocking=False):
            yield

    def _require_active(self):
        if not self.settings["active"]:
            raise ValueError("명시 요청 연구가 활성화되지 않았습니다")

    def _write(self, state):
        self._validate(state)
        self._atomic_write(self.path, state)

    @staticmethod
    def _atomic_write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise ValueError("연구 기록의 심볼릭 링크는 허용하지 않습니다")
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            if os.name != "nt":
                fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _new_batch(kind, baseline, target):
        return {"id": "bootstrap" if kind == "bootstrap" else uuid.uuid4().hex,
                "kind": kind, "baseline_bytes": baseline, "target_bytes": target,
                "status": "collecting", "requested_at": _now(), "attempts": 0,
                "claim": None, "last_error": None, "run": None,
                "completed_at": None, "promoted": None}

    @staticmethod
    def _validate(state):
        if (not isinstance(state, dict) or set(state) != {"schema_version", "batches"}
                or type(state["schema_version"]) is not int or state["schema_version"] != 1
                or not isinstance(state["batches"], list) or not state["batches"]):
            raise ValueError("연구 상태 schema가 잘못됐습니다")
        seen = set()
        previous_target = 0
        for index, batch in enumerate(state["batches"]):
            if not isinstance(batch, dict) or set(batch) != BATCH_KEYS:
                raise ValueError("연구 batch schema가 잘못됐습니다")
            name = batch["id"]
            if (not isinstance(name, str) or name in seen
                    or (index == 0 and (name != "bootstrap" or batch["kind"] != "bootstrap"))
                    or (index > 0 and (not re.fullmatch("[a-f0-9]{32}", name) or batch["kind"] != "increment"))):
                raise ValueError("연구 batch 식별자가 잘못됐습니다")
            seen.add(name)
            if any(type(batch[key]) is not int or batch[key] < 0
                   for key in ("baseline_bytes", "target_bytes", "attempts")):
                raise ValueError("연구 용량·시도 횟수는 비음수 정수여야 합니다")
            if ((index == 0 and (batch["baseline_bytes"] != 0 or batch["target_bytes"] != INITIAL_BYTES))
                    or (index > 0 and (batch["baseline_bytes"] < previous_target
                                       or batch["target_bytes"] != batch["baseline_bytes"] + INCREMENT_BYTES))):
                raise ValueError("연구 목표·기준 적재량이 계약과 다릅니다")
            previous_target = batch["target_bytes"]
            _utc_stamp(batch["requested_at"])
            status = batch["status"]
            if (status not in {"collecting", "training", "completed"}
                    or (index < len(state["batches"]) - 1 and status != "completed")
                    or (batch["last_error"] is not None and not isinstance(batch["last_error"], str))):
                raise ValueError("연구 batch 상태가 잘못됐습니다")
            if status == "completed":
                _utc_stamp(batch["completed_at"])
                if (batch["attempts"] < 1 or not isinstance(batch["run"], str) or not batch["run"]
                        or type(batch["promoted"]) is not bool or batch["claim"] is not None):
                    raise ValueError("학습 완료 근거가 잘못됐습니다")
            else:
                if any(batch[key] is not None for key in ("run", "completed_at", "promoted")):
                    raise ValueError("미완료 batch에 완료 근거가 있습니다")
                if status == "training":
                    if (batch["attempts"] < 1 or not isinstance(batch["claim"], str)
                            or not re.fullmatch("[a-f0-9]{32}", batch["claim"])):
                        raise ValueError("학습 소유권이 잘못됐습니다")
                elif batch["claim"] is not None:
                    raise ValueError("대기 batch에 학습 소유권이 있습니다")

    def _read(self):
        if self.path.is_symlink():
            raise ValueError("연구 상태의 심볼릭 링크는 허용하지 않습니다")
        if not self.path.exists():
            state = {"schema_version": 1, "batches": [self._new_batch("bootstrap", 0, INITIAL_BYTES)]}
            self._write(state)
        else:
            state = json.loads(self.path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
        self._validate(state)
        return state

    def snapshot(self):
        self._require_active()
        # 원자적으로 교체되는 JSON 자체가 아닌 고정 별도 파일을 잠급니다.
        with _file_lock(self.path.with_suffix(".lock")):
            return self._read()

    def pending_batch(self):
        if not self.settings["active"]:
            return None
        batch = self.snapshot()["batches"][-1]
        return batch if batch["status"] != "completed" else None

    def collection_target(self) -> int | None:
        batch = self.pending_batch()
        return batch["target_bytes"] if batch is not None else None

    def request_increment(self, read_actual_ingested):
        self._require_active()
        with _file_lock(self.path.with_suffix(".lock")):
            state = self._read()
            if state["batches"][0]["status"] != "completed":
                raise ValueError("초기 학습 완료 전에는 추가 연구를 요청할 수 없습니다")
            if state["batches"][-1]["status"] != "completed":
                raise ValueError("이미 활성 연구 요청이 있습니다")
            baseline = read_actual_ingested()
            if type(baseline) is not int or baseline < state["batches"][-1]["target_bytes"]:
                raise ValueError("현재 실제 고유 적재량이 유효하지 않습니다")
            batch = self._new_batch("increment", baseline, baseline + INCREMENT_BYTES)
            state["batches"].append(batch)
            self._write(state)
            return batch

    def ensure_continuous_batch(self, read_actual_ingested, stop=None):
        """완료 뒤 새 요청 하나만 생성하고 진행 중인 요청은 그대로 반환합니다."""
        self._require_active()
        if not self.settings["continuous"]:
            return self.pending_batch()
        # 느린 원본 재검증은 trainer가 읽는 상태 잠금을 점유하지 않습니다.
        with _file_lock(self.path.with_suffix(".collection.lock")):
            with _file_lock(self.path.with_suffix(".lock")):
                if stop is not None and stop.is_set():
                    return None
                state = self._read()
                previous = state["batches"][-1]
                if previous["status"] != "completed":
                    return previous
            baseline = read_actual_ingested()
            with _file_lock(self.path.with_suffix(".lock")):
                if stop is not None and stop.is_set():
                    return None
                state = self._read()
                previous = state["batches"][-1]
                if previous["status"] != "completed":
                    return previous
                if type(baseline) is not int or baseline < previous["target_bytes"]:
                    raise ValueError("현재 실제 고유 적재량이 유효하지 않습니다")
                batch = self._new_batch("increment", baseline, baseline + INCREMENT_BYTES)
                state["batches"].append(batch)
                self._write(state)
                return batch

    @staticmethod
    def _owned(state, batch_id, claim):
        batch = state["batches"][-1]
        if batch["id"] != batch_id or batch["status"] != "training" or batch["claim"] != claim:
            raise ValueError("현재 활성 학습의 소유권과 다릅니다")
        return batch

    def claim_training(self, batch_id):
        self._require_active()
        with _file_lock(self.path.with_suffix(".lock")):
            state = self._read()
            batch = state["batches"][-1]
            if batch["id"] != batch_id or batch["status"] != "collecting":
                raise ValueError("활성 연구 batch가 아닙니다")
            batch.update(status="training", claim=uuid.uuid4().hex, attempts=batch["attempts"] + 1)
            self._write(state)
            return batch

    def fail_training(self, batch_id, claim, error):
        with _file_lock(self.path.with_suffix(".lock")):
            state = self._read()
            batch = self._owned(state, batch_id, claim)
            if self.attempt_path(batch).exists() and self.read_attempt(batch)["phase"] == "complete":
                raise ValueError("완료 결과를 학습 실패로 되돌릴 수 없습니다")
            batch.update(status="collecting", claim=None, last_error=str(error))
            self._write(state)

    def attempt_path(self, batch):
        return self.path.parent / (self.path.stem + "_attempts") / f"{batch['id']}-{batch['attempts']}.json"

    def _validate_attempt(self, record, batch):
        if (not isinstance(record, dict) or set(record) != ATTEMPT_KEYS
                or type(record["schema_version"]) is not int or record["schema_version"] != 1
                or record["batch_id"] != batch["id"] or record["attempt"] != batch["attempts"]
                or type(record["attempt"]) is not int or record["attempt"] < 1
                or type(record["target_bytes"]) is not int or record["target_bytes"] != batch["target_bytes"]
                or not isinstance(record["claim"], str) or not re.fullmatch("[a-f0-9]{32}", record["claim"])
                or (batch["status"] == "training" and record["claim"] != batch["claim"])
                or not isinstance(record["run"], str) or not re.fullmatch(r"\d{8}T\d{6}-[a-f0-9]{8}", record["run"])):
            raise ValueError("연구 시도와 batch·소유권·목표의 연결이 잘못됐습니다")
        for key in ("reports_dir", "models_dir"):
            value = record[key]
            if not isinstance(value, str) or not value or Path(value).is_absolute() or "\\" in value:
                raise ValueError("연구 시도 산출물 경로가 잘못됐습니다")
            target = (self.root / value).resolve()
            if not target.is_relative_to(self.root / "work") or target == self.root / "work":
                raise ValueError("연구 시도 산출물이 work 경계를 벗어났습니다")
        for key in ("previous_pointer_sha256", "previous_training_sha256"):
            if record[key] is not None and (not isinstance(record[key], str) or not re.fullmatch("[a-f0-9]{64}", record[key])):
                raise ValueError("연구 이전 산출물 식별자가 잘못됐습니다")
        if record["phase"] not in {"running", "model_failed", "complete"}:
            raise ValueError("연구 시도 단계가 잘못됐습니다")
        if ((record["phase"] == "model_failed" and not isinstance(record["error"], str))
                or (record["phase"] != "model_failed" and record["error"] is not None)):
            raise ValueError("연구 시도 오류 근거가 단계와 다릅니다")
        if record["phase"] == "complete":
            self._result(record["report"], batch["target_bytes"])
            if (record["report"].get("research_batch") != attempt_binding(record)
                    or record["report"].get("run") != record["run"]
                    or digest(record["report"]) != record["report_sha256"]
                    or not isinstance(record["manifest_sha256"], str)
                    or not re.fullmatch("[a-f0-9]{64}", record["manifest_sha256"])
                    or not isinstance(record["model_sha256"], dict) or not record["model_sha256"]):
                raise ValueError("완료 결과의 batch 결합 또는 내용 해시가 다릅니다")
        elif any(record[key] is not None for key in ("report", "report_sha256", "manifest_sha256", "model_sha256")):
            raise ValueError("미완료 시도에 완료 결과가 섞였습니다")

    def read_attempt(self, batch):
        path = self.attempt_path(batch)
        if path.is_symlink() or not path.is_file():
            raise ValueError("미확인 연구 시도입니다. 소유권을 보존하고 복구를 차단합니다")
        record = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
        self._validate_attempt(record, batch)
        return record

    def start_attempt(self, batch, run, reports_dir, models_dir, previous_pointer, previous_training):
        record = {"schema_version": 1, "batch_id": batch["id"], "claim": batch["claim"],
                  "attempt": batch["attempts"], "target_bytes": batch["target_bytes"], "run": run,
                  "reports_dir": reports_dir, "models_dir": models_dir,
                  "previous_pointer_sha256": previous_pointer, "previous_training_sha256": previous_training,
                  "phase": "running", "report": None, "report_sha256": None,
                  "manifest_sha256": None, "model_sha256": None, "error": None}
        with _file_lock(self.path.with_suffix(".lock")):
            current = self._owned(self._read(), batch["id"], batch["claim"])
            self._validate_attempt(record, current)
            if self.attempt_path(current).exists():
                raise ValueError("기존 연구 시도 기록을 덮어쓸 수 없습니다")
            self._atomic_write(self.attempt_path(current), record)
        return record

    def store_result(self, batch_id, claim, report, manifest_sha256, model_sha256):
        with _file_lock(self.path.with_suffix(".lock")):
            batch = self._owned(self._read(), batch_id, claim)
            record = self.read_attempt(batch)
            if record["phase"] != "running":
                raise ValueError("실행 중인 시도의 결과만 저장할 수 있습니다")
            record.update(phase="complete", report=report, report_sha256=digest(report),
                          manifest_sha256=manifest_sha256, model_sha256=model_sha256)
            self._validate_attempt(record, batch)
            self._atomic_write(self.attempt_path(batch), record)

    def record_model_failure(self, batch_id, claim, error):
        with _file_lock(self.path.with_suffix(".lock")):
            state = self._read()
            batch = self._owned(state, batch_id, claim)
            record = self.read_attempt(batch)
            if record["phase"] != "running":
                raise ValueError("완료 저장 실패를 모델 실패로 기록할 수 없습니다")
            record.update(phase="model_failed", error=str(error))
            self._atomic_write(self.attempt_path(batch), record)
            batch.update(status="collecting", claim=None, last_error=str(error))
            self._write(state)

    def restore_claim(self, batch, record):
        with _file_lock(self.path.with_suffix(".lock")):
            state = self._read()
            current = state["batches"][-1]
            self._validate_attempt(record, current)
            if current["status"] != "collecting" or record["phase"] != "complete":
                raise ValueError("복구할 완료 시도의 소유권이 아닙니다")
            current.update(status="training", claim=record["claim"])
            self._write(state)

    def note_training_error(self, batch_id, claim, error):
        with _file_lock(self.path.with_suffix(".lock")):
            state = self._read()
            self._owned(state, batch_id, claim)["last_error"] = str(error)
            self._write(state)

    @staticmethod
    def _result(report, target):
        try:
            training, evidence = report["training"], report["external_resolution"]
            count = len(report["config"]["training"]["candidate_depths"])
            if (not count or set(report["markets"]) != MARKETS or training["status"] != "완료"
                    or type(training["completed_candidates"]) is not int
                    or training["completed_candidates"] != 5 * count
                    or training["total_candidates"] != 5 * count or len(report["rounds"]) != 5 * count
                    or not isinstance(report["run"], str) or not report["run"]
                    or type(report["promotion"]["promote"]) is not bool
                    or evidence["training_ready"] is not True):
                raise ValueError("다섯 시장 전체 학습의 완료 근거가 부족합니다")
            for key in ("target_archive_bytes", "ingested_archive_bytes", "verified_archive_bytes"):
                if type(evidence[key]) is not int or evidence[key] < target:
                    raise ValueError("학습 자료의 적재 목표를 충족하지 못했습니다")
            if evidence["verified_archive_bytes"] < evidence["ingested_archive_bytes"]:
                raise ValueError("검증되지 않은 적재 자료입니다")
            for market in MARKETS:
                rounds = [item for item in report["rounds"] if item["market"] == market]
                gpu = report["markets"][market]["gpu"]
                if (len(gpu) != count or sorted(item["candidate"] for item in rounds) != list(range(1, count + 1))
                        or any(not isinstance(item.get("actual_device"), str)
                               or not item["actual_device"].startswith("cuda:") for item in [*gpu, *rounds])):
                    raise ValueError("모든 후보의 GPU 학습 완료 근거가 부족합니다")
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("학습 완료 보고서 schema가 잘못됐습니다") from exc

    def complete_training(self, batch_id, claim, report):
        with _file_lock(self.path.with_suffix(".lock")):
            state = self._read()
            batch = self._owned(state, batch_id, claim)
            record = self.read_attempt(batch)
            if record["phase"] != "complete" or digest(report) != record["report_sha256"]:
                raise ValueError("내구성 있는 완료 결과와 제어 상태가 다릅니다")
            self._result(report, batch["target_bytes"])
            batch.update(status="completed", claim=None, run=report["run"], completed_at=_now(),
                         promoted=report["promotion"]["promote"])
            self._write(state)
