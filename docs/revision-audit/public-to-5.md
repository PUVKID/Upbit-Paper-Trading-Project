# 기존 공개본 → 4·5차 반영 전체 비교

대상은 아래 명시된 공개 코드·설정·검사 파일입니다. 텍스트 차이는 공백과 주석을 포함합니다. 문서와 4차 보관 설정의 차이는 [문서·설정 전체 비교](문서-변경.md)에서 확인합니다. 비교 문서 자체는 순환을 피하기 위해 별도 크기·해시 목록으로 기록합니다.

## 3차/external_data.py

이전: 31276 bytes / SHA-256 89f33fec5e8418d4f9ab8b2dd604afb01df3819ba65ed9260050f593356e4a37  
현재: 37586 bytes / SHA-256 308a74a56a162d20b0d01b2f86f202bbf6c36b1a6759319dc9ce08b80d7ad462

```diff
--- 이전/3차/external_data.py
+++ 현재/3차/external_data.py
@@ -4,6 +4,7 @@
 import argparse
 import datetime as dt
 import hashlib
+import importlib
 import json
 import logging
 from logging.handlers import RotatingFileHandler
@@ -40,6 +41,10 @@
 
 class ArchiveIntegrityError(ValueError):
     """안전한 일반 파일의 크기·해시 불일치만 복구 대상으로 구분합니다."""
+
+
+class ArchiveContentError(ValueError):
+    """파싱 실패만 격리하며 저장·권한·경로 오류와 구분합니다."""
 
 
 def file_sha256(path):
@@ -297,12 +302,17 @@
 
 def bootstrap_plan(now):
     now = pd.Timestamp(now)
+    if now.tzinfo is not None:
+        now = now.tz_convert('UTC')
     year, month = previous_month(now.year, now.month)
     for symbol in SYMBOLS:
         yield archive_spec(symbol, 'klines', f'{year:04d}-{month:02d}')
-    for day in range(1, now.day):
+    today = now.normalize()
+    # 월별 원본 공개 지연과 운용 중단 공백은 최근 완료 일별 원본으로 보충합니다.
+    first = min(today.replace(day=1), today - pd.Timedelta(days=14))
+    for day in reversed(pd.date_range(first, today, inclusive='left')):
         for symbol in SYMBOLS:
-            yield archive_spec(symbol, 'klines', f'{now.year:04d}-{now.month:02d}-{day:02d}', 'daily')
+            yield archive_spec(symbol, 'klines', day.strftime('%Y-%m-%d'), 'daily')
 
 
 def bulk_plan(now):
@@ -326,7 +336,7 @@
 
 
 class Collector:
-    def __init__(self, work=WORK, target=TARGET_BYTES):
+    def __init__(self, work=WORK, target=TARGET_BYTES, research_settings=None):
         self.work = Path(work)
         self.work.mkdir(parents=True, exist_ok=True)
         self.archives = self.work / 'archives'
@@ -341,6 +351,40 @@
         self.integrity_cache = {}
         self.integrity_errors = {}
         self.runtime = {'state': '시작', 'bulk_error': None, 'live_error': None, 'archive_current': None}
+        # 임시 수집기 검사는 실제 연구 상태를 자동으로 열지 않습니다.
+        if research_settings is None:
+            research_settings = {}
+            if self.work.resolve() == WORK.resolve():
+                config = json.loads((ROOT.parent / 'engine-config.json').read_text(encoding='utf-8'))
+                research_settings = config.get('research', {})
+        if not isinstance(research_settings, dict) or type(research_settings.get('active', False)) is not bool:
+            raise ValueError('연구 제어 설정 형식이 잘못됐습니다')
+        self.research_control = None
+        if research_settings.get('active', False):
+            status_path = self.work / 'status.json'
+            if status_path.is_symlink():
+                raise ValueError('이전 수집 상태의 심볼릭 링크는 허용하지 않습니다')
+            if status_path.exists():
+                previous = json.loads(status_path.read_text(encoding='utf-8')).get('target_archive_bytes', target)
+                if type(previous) is not int or previous <= 0:
+                    raise ValueError('이전 수집 목표가 양의 정수가 아닙니다')
+                self.target = previous
+            project_path = str(ROOT.parent)
+            if project_path not in sys.path:
+                sys.path.insert(0, project_path)
+            module = importlib.import_module('4차.research_control')
+            self.research_control = module.ResearchControl(ROOT.parent, research_settings)
+
+    def collection_target(self):
+        with self.lock:
+            if self.research_control is None:
+                return self.target
+            target = self.research_control.collection_target()
+            if target is not None:
+                if type(target) is not int or target <= 0:
+                    raise ValueError('연구 수집 목표는 양의 정수 또는 명시적 대기여야 합니다')
+                self.target = target
+            return target
 
     def verified_file(self, name, item):
         if (not isinstance(item, dict) or name != item.get('name') or item.get('symbol') not in SYMBOLS
@@ -382,13 +426,33 @@
                 verified.append(item)
         return verified
 
+    def content_rejection(self, item, digest=None):
+        records = item.get('content_rejections', {})
+        if not isinstance(records, dict):
+            raise ValueError('내용 격리 기록의 형식이 잘못됐습니다')
+        digest = item.get('sha256') if digest is None else digest
+        if digest not in records:
+            return None
+        record = records[digest]
+        if (not isinstance(digest, str) or not re.fullmatch('[a-f0-9]{64}', digest)
+                or not isinstance(record, dict) or record.get('sha256') != digest
+                or record.get('name') != item.get('name') or not isinstance(record.get('reason'), str)):
+            raise ValueError('내용 격리 기록의 원본·해시 결합이 잘못됐습니다')
+        return record
+
     def status(self, **fields):
         with self.lock:
             self.runtime.update(fields)
             stats = self.store.summary()
             verified = self.verified_items()
             total = sum(item['bytes'] for item in verified)
-            ingested = sum(item['bytes'] for item in verified if item.get('ingested'))
+            ingested = sum(item['bytes'] for item in verified
+                           if item.get('ingested') is True and self.content_rejection(item) is None)
+            warnings = []
+            for item in self.manifest.values():
+                if isinstance(item, dict):
+                    self.content_rejection(item)
+                    warnings.extend(self.content_rejection(item, digest) for digest in item.get('content_rejections', {}))
             self.store.archive_evidence(self.target, ingested, total)
             result = {'schema_version': 1, 'updated_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                       'training_ready': not self.integrity_errors and ingested >= self.target and self.store.ready(utc_ms()),
@@ -397,6 +461,8 @@
                       'unique_minute_rows': sum(v['rows'] for v in stats.values()), 'markets': stats,
                       'db_schema': 'binance_spot_v1', 'paper_only': True, **self.runtime}
             result['integrity_errors'] = dict(self.integrity_errors)
+            result['archive_warnings'] = warnings
+            result['archive_warning_count'] = len(warnings)
             result['error'] = result['bulk_error'] or result['live_error'] or ('원본 파일 무결성 확인 실패' if self.integrity_errors else None)
             atomic_json(self.work / 'status.json', result)
             return result
@@ -474,15 +540,23 @@
 
     def ingest(self, spec, path):
         count = 0
+
+        def frames(stream):
+            try:
+                chunks = pd.read_csv(stream, header=None, chunksize=200000)
+                if spec['kind'] == 'klines':
+                    yield from (parse_klines(chunk, utc_ms()) for chunk in chunks)
+                else:
+                    yield from aggregate_trade_chunks(chunks, utc_ms())
+            except ValueError as error:
+                failure = ArchiveContentError(f'{type(error).__name__}: {error}')
+                failure.partial_inserted_rows = count
+                raise failure from error
+
         with zipfile.ZipFile(path) as archive:
             member = archive_member(archive, spec['name'][:-4] + '.csv')
             with archive.open(member) as stream:
-                chunks = pd.read_csv(stream, header=None, chunksize=200000)
-                if spec['kind'] == 'klines':
-                    parsed = (parse_klines(chunk, utc_ms()) for chunk in chunks)
-                else:
-                    parsed = aggregate_trade_chunks(chunks, utc_ms())
-                for frame in parsed:
+                for frame in frames(stream):
                     if STOP.is_set():
                         raise InterruptedError('사용자 중단')
                     disk_budget(self.work, len(frame) * 1024)
@@ -490,6 +564,9 @@
         return count
 
     def archive(self, spec):
+        if self.collection_target() is None:
+            self.status(state='수집 요청 대기', archive_current=None)
+            return
         existing = self.manifest.get(spec['name'], {})
         path = self.archives / spec['name']
         self.status(state='외부 자료 수집', archive_current=spec['name'], download_bytes=0)
@@ -498,25 +575,52 @@
                 raise InterruptedError('사용자 중단')
             try:
                 path, digest = self.download(spec)
+                pending = {**spec, 'bytes': path.stat().st_size, 'sha256': digest,
+                           'sha256_verified': True, 'ingested': False,
+                           'content_rejections': existing.get('content_rejections', {})}
+                if self.content_rejection(existing, digest) is not None:
+                    with self.lock:
+                        if pending != existing:
+                            self.manifest[spec['name']] = pending
+                            atomic_json(self.manifest_path, self.manifest)
+                    logging.warning('동일 해시의 내용 검증 실패 원본은 미적재 유지 %s SHA256=%s', spec['name'], digest)
+                    self.status(state='내용 검증 실패 원본 격리 유지', archive_current=None)
+                    return
                 if existing.get('ingested') is True and existing.get('sha256') == digest:
                     self.integrity_errors.pop(spec['name'], None)
                     self.status(bulk_error=None, archive_current=None)
                     return
                 with self.lock:
-                    self.manifest[spec['name']] = {**spec, 'bytes': path.stat().st_size,
-                                                   'sha256': digest, 'sha256_verified': True, 'ingested': False}
+                    self.manifest[spec['name']] = pending
                     atomic_json(self.manifest_path, self.manifest)
-                inserted = self.ingest(spec, path)
+                try:
+                    inserted = self.ingest(spec, path)
+                except ArchiveContentError as error:
+                    # 정상 prefix는 그대로 두고 원본 해시의 실패를 영속적으로 보존합니다.
+                    with self.lock:
+                        self.manifest[spec['name']]['content_rejections'][digest] = {
+                            'name': spec['name'], 'sha256': digest, 'reason': str(error),
+                            'observed_at': dt.datetime.now(dt.timezone.utc).isoformat(),
+                            'partial_inserted_rows_this_attempt': error.partial_inserted_rows}
+                        atomic_json(self.manifest_path, self.manifest)
+                    logging.warning('내용 검증 실패 원본 미적재 격리 %s SHA256=%s', spec['name'], digest, exc_info=True)
+                    self.status(state='내용 검증 실패 원본 격리', archive_current=None,
+                                bulk_error=f'{type(error).__name__}: {error}')
+                    return
                 with self.lock:
                     self.manifest[spec['name']].update(ingested=True, inserted_rows=inserted)
                     atomic_json(self.manifest_path, self.manifest)
                 self.status(bulk_error=None, archive_current=None)
+                if self.research_control is not None:
+                    self.collection_target()
+                    self.status()
                 return
             except urllib.error.HTTPError as error:
                 if error.code == 404:
                     logging.warning('공식 미제공 파일 %s', spec['name'])
                     with self.lock:
-                        self.manifest[spec['name']] = {**spec, 'available': False, 'http_status': 404}
+                        self.manifest[spec['name']] = {**existing, **spec, 'available': False,
+                                                       'http_status': 404, 'ingested': False}
                         atomic_json(self.manifest_path, self.manifest)
                     return
                 if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
@@ -564,17 +668,20 @@
                 try:
                     now = pd.Timestamp.now(tz='UTC')
                     for spec in bootstrap_plan(now):
-                        if STOP.is_set():
+                        if STOP.is_set() or self.collection_target() is None:
                             break
                         self.archive(spec)
                     for spec in bulk_plan(now):
-                        if STOP.is_set() or self.status()['ingested_archive_bytes'] >= self.target:
+                        if (STOP.is_set() or self.collection_target() is None
+                                or self.status()['ingested_archive_bytes'] >= self.target):
                             break
                         self.archive(spec)
                     if not STOP.is_set():
-                        self.status(state='목표 자료 확보 후 실시간 갱신' if self.status()['ingested_archive_bytes'] >= self.target
+                        target = self.collection_target()
+                        self.status(state='수집 요청 대기' if target is None else
+                                    '목표 자료 확보 후 실시간 갱신' if self.status()['ingested_archive_bytes'] >= self.target
                                     else '공식 자료 범위 탐색 완료; 다음 갱신 대기')
-                        STOP.wait(3600)
+                        STOP.wait(15 if self.research_control is not None else 3600)
                 except InterruptedError:
                     break
                 except Exception as error:

```

## 4차/research_control.py

이전: 없음  
현재: 22332 bytes / SHA-256 9e5692d6926c4bd0964745be34785155b7e80c2323248fe44e5b8e990e468005

```diff
--- 이전/4차/research_control.py
+++ 현재/4차/research_control.py
@@ -0,0 +1,418 @@
+"""고유 외부 자료의 초기 연구와 명시 요청 연구를 내구성 있게 제어합니다."""
+from __future__ import annotations
+
+import contextlib
+import datetime as dt
+import hashlib
+import json
+import os
+from pathlib import Path, PureWindowsPath
+import re
+import time
+import uuid
+
+
+INITIAL_BYTES = 50_000_000_000
+INCREMENT_BYTES = 2_000_000_000
+DEFAULT_SETTINGS = {"active": False, "state_path": "4차/work/research_control.json"}
+MARKETS = {"KRW-BTC", "KRW-SOL", "KRW-ETH", "KRW-XRP", "KRW-DOGE"}
+BATCH_KEYS = {"id", "kind", "baseline_bytes", "target_bytes", "status", "requested_at",
+              "attempts", "claim", "last_error", "run", "completed_at", "promoted"}
+ATTEMPT_KEYS = {"schema_version", "batch_id", "claim", "attempt", "target_bytes", "run",
+                "reports_dir", "models_dir", "previous_pointer_sha256", "previous_training_sha256",
+                "phase", "report", "report_sha256", "manifest_sha256", "model_sha256", "error"}
+
+
+def digest(value):
+    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
+                                     allow_nan=False).encode("utf-8")).hexdigest()
+
+
+def attempt_binding(record):
+    return {key: record[key] for key in
+            ("schema_version", "batch_id", "claim", "attempt", "target_bytes", "run")}
+
+
+def validate_config(settings, root):
+    if not isinstance(settings, dict) or set(settings) - set(DEFAULT_SETTINGS):
+        raise ValueError("지원하지 않는 research 설정입니다")
+    result = {**DEFAULT_SETTINGS, **settings}
+    if type(result["active"]) is not bool:
+        raise ValueError("research.active는 bool이어야 합니다")
+    value = result["state_path"]
+    if (not isinstance(value, str) or not value.strip() or "\x00" in value
+            or value.startswith("file:") or (os.name != "nt" and (PureWindowsPath(value).drive or "\\" in value))):
+        raise ValueError("research.state_path 경로가 유효하지 않습니다")
+    root = Path(root).resolve()
+    boundary = (root / "4차/work").resolve()
+    target = (root / value).resolve()
+    if (not boundary.is_relative_to(root) or not target.is_relative_to(boundary)
+            or target == boundary or target.suffix != ".json"):
+        raise ValueError("연구 상태는 프로젝트 4차/work 안의 JSON이어야 합니다")
+    result["state_path"] = target.relative_to(root).as_posix()
+    return result
+
+
+def _now():
+    return dt.datetime.now(dt.timezone.utc).isoformat()
+
+
+def _utc_stamp(value):
+    if not isinstance(value, str):
+        raise ValueError("연구 시각 형식 오류")
+    stamp = dt.datetime.fromisoformat(value)
+    if stamp.tzinfo is None or stamp.utcoffset() != dt.timedelta(0):
+        raise ValueError("연구 시각은 UTC여야 합니다")
+
+
+def _pairs(items):
+    result = {}
+    for key, value in items:
+        if key in result:
+            raise ValueError("연구 상태 JSON의 중복 키입니다")
+        result[key] = value
+    return result
+
+
+@contextlib.contextmanager
+def _file_lock(path, blocking=True):
+    path.parent.mkdir(parents=True, exist_ok=True)
+    if path.is_symlink():
+        raise ValueError("연구 잠금의 심볼릭 링크는 허용하지 않습니다")
+    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
+    with os.fdopen(fd, "r+b") as stream:
+        deadline = time.monotonic() + 30
+        while True:
+            try:
+                if os.name == "nt":
+                    import msvcrt
+                    stream.seek(0)
+                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
+                else:
+                    import fcntl
+                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
+                break
+            except OSError:
+                if not blocking or time.monotonic() >= deadline:
+                    raise
+                time.sleep(.05)
+        try:
+            yield
+        finally:
+            if os.name == "nt":
+                import msvcrt
+                stream.seek(0)
+                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
+            else:
+                import fcntl
+                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
+
+
+class ResearchControl:
+    def __init__(self, root: Path, settings: dict):
+        self.root = Path(root).resolve()
+        self.settings = validate_config(settings, self.root)
+        self.path = self.root / self.settings["state_path"]
+
+    @contextlib.contextmanager
+    def training_lock(self):
+        """적합 중에도 Collector가 상태를 읽도록 짧은 상태 잠금과 분리합니다."""
+        self._require_active()
+        with _file_lock(self.path.with_suffix(".training.lock"), blocking=False):
+            yield
+
+    def _require_active(self):
+        if not self.settings["active"]:
+            raise ValueError("명시 요청 연구가 활성화되지 않았습니다")
+
+    def _write(self, state):
+        self._validate(state)
+        self._atomic_write(self.path, state)
+
+    @staticmethod
+    def _atomic_write(path, value):
+        path.parent.mkdir(parents=True, exist_ok=True)
+        if path.is_symlink():
+            raise ValueError("연구 기록의 심볼릭 링크는 허용하지 않습니다")
+        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
+        try:
+            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
+                json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
+                stream.write("\n")
+                stream.flush()
+                os.fsync(stream.fileno())
+            os.replace(temporary, path)
+            if os.name != "nt":
+                fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
+                try:
+                    os.fsync(fd)
+                finally:
+                    os.close(fd)
+        finally:
+            temporary.unlink(missing_ok=True)
+
+    @staticmethod
+    def _new_batch(kind, baseline, target):
+        return {"id": "bootstrap" if kind == "bootstrap" else uuid.uuid4().hex,
+                "kind": kind, "baseline_bytes": baseline, "target_bytes": target,
+                "status": "collecting", "requested_at": _now(), "attempts": 0,
+                "claim": None, "last_error": None, "run": None,
+                "completed_at": None, "promoted": None}
+
+    @staticmethod
+    def _validate(state):
+        if (not isinstance(state, dict) or set(state) != {"schema_version", "batches"}
+                or type(state["schema_version"]) is not int or state["schema_version"] != 1
+                or not isinstance(state["batches"], list) or not state["batches"]):
+            raise ValueError("연구 상태 schema가 잘못됐습니다")
+        seen = set()
+        previous_target = 0
+        for index, batch in enumerate(state["batches"]):
+            if not isinstance(batch, dict) or set(batch) != BATCH_KEYS:
+                raise ValueError("연구 batch schema가 잘못됐습니다")
+            name = batch["id"]
+            if (not isinstance(name, str) or name in seen
+                    or (index == 0 and (name != "bootstrap" or batch["kind"] != "bootstrap"))
+                    or (index > 0 and (not re.fullmatch("[a-f0-9]{32}", name) or batch["kind"] != "increment"))):
+                raise ValueError("연구 batch 식별자가 잘못됐습니다")
+            seen.add(name)
+            if any(type(batch[key]) is not int or batch[key] < 0
+                   for key in ("baseline_bytes", "target_bytes", "attempts")):
+                raise ValueError("연구 용량·시도 횟수는 비음수 정수여야 합니다")
+            if ((index == 0 and (batch["baseline_bytes"] != 0 or batch["target_bytes"] != INITIAL_BYTES))
+                    or (index > 0 and (batch["baseline_bytes"] < previous_target
+                                       or batch["target_bytes"] != batch["baseline_bytes"] + INCREMENT_BYTES))):
+                raise ValueError("연구 목표·기준 적재량이 계약과 다릅니다")
+            previous_target = batch["target_bytes"]
+            _utc_stamp(batch["requested_at"])
+            status = batch["status"]
+            if (status not in {"collecting", "training", "completed"}
+                    or (index < len(state["batches"]) - 1 and status != "completed")
+                    or (batch["last_error"] is not None and not isinstance(batch["last_error"], str))):
+                raise ValueError("연구 batch 상태가 잘못됐습니다")
+            if status == "completed":
+                _utc_stamp(batch["completed_at"])
+                if (batch["attempts"] < 1 or not isinstance(batch["run"], str) or not batch["run"]
+                        or type(batch["promoted"]) is not bool or batch["claim"] is not None):
+                    raise ValueError("학습 완료 근거가 잘못됐습니다")
+            else:
+                if any(batch[key] is not None for key in ("run", "completed_at", "promoted")):
+                    raise ValueError("미완료 batch에 완료 근거가 있습니다")
+                if status == "training":
+                    if (batch["attempts"] < 1 or not isinstance(batch["claim"], str)
+                            or not re.fullmatch("[a-f0-9]{32}", batch["claim"])):
+                        raise ValueError("학습 소유권이 잘못됐습니다")
+                elif batch["claim"] is not None:
+                    raise ValueError("대기 batch에 학습 소유권이 있습니다")
+
+    def _read(self):
+        if self.path.is_symlink():
+            raise ValueError("연구 상태의 심볼릭 링크는 허용하지 않습니다")
+        if not self.path.exists():
+            state = {"schema_version": 1, "batches": [self._new_batch("bootstrap", 0, INITIAL_BYTES)]}
+            self._write(state)
+        else:
+            state = json.loads(self.path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
+        self._validate(state)
+        return state
+
+    def snapshot(self):
+        self._require_active()
+        # 원자적으로 교체되는 JSON 자체가 아닌 고정 별도 파일을 잠급니다.
+        with _file_lock(self.path.with_suffix(".lock")):
+            return self._read()
+
+    def pending_batch(self):
+        if not self.settings["active"]:
+            return None
+        batch = self.snapshot()["batches"][-1]
+        return batch if batch["status"] != "completed" else None
+
+    def collection_target(self) -> int | None:
+        batch = self.pending_batch()
+        return batch["target_bytes"] if batch is not None else None
+
+    def request_increment(self, read_actual_ingested):
+        self._require_active()
+        with _file_lock(self.path.with_suffix(".lock")):
+            state = self._read()
+            if state["batches"][0]["status"] != "completed":
+                raise ValueError("초기 학습 완료 전에는 추가 연구를 요청할 수 없습니다")
+            if state["batches"][-1]["status"] != "completed":
+                raise ValueError("이미 활성 연구 요청이 있습니다")
+            baseline = read_actual_ingested()
+            if type(baseline) is not int or baseline < state["batches"][-1]["target_bytes"]:
+                raise ValueError("현재 실제 고유 적재량이 유효하지 않습니다")
+            batch = self._new_batch("increment", baseline, baseline + INCREMENT_BYTES)
+            state["batches"].append(batch)
+            self._write(state)
+            return batch
+
+    @staticmethod
+    def _owned(state, batch_id, claim):
+        batch = state["batches"][-1]
+        if batch["id"] != batch_id or batch["status"] != "training" or batch["claim"] != claim:
+            raise ValueError("현재 활성 학습의 소유권과 다릅니다")
+        return batch
+
+    def claim_training(self, batch_id):
+        self._require_active()
+        with _file_lock(self.path.with_suffix(".lock")):
+            state = self._read()
+            batch = state["batches"][-1]
+            if batch["id"] != batch_id or batch["status"] != "collecting":
+                raise ValueError("활성 연구 batch가 아닙니다")
+            batch.update(status="training", claim=uuid.uuid4().hex, attempts=batch["attempts"] + 1)
+            self._write(state)
+            return batch
+
+    def fail_training(self, batch_id, claim, error):
+        with _file_lock(self.path.with_suffix(".lock")):
+            state = self._read()
+            batch = self._owned(state, batch_id, claim)
+            if self.attempt_path(batch).exists() and self.read_attempt(batch)["phase"] == "complete":
+                raise ValueError("완료 결과를 학습 실패로 되돌릴 수 없습니다")
+            batch.update(status="collecting", claim=None, last_error=str(error))
+            self._write(state)
+
+    def attempt_path(self, batch):
+        return self.path.parent / (self.path.stem + "_attempts") / f"{batch['id']}-{batch['attempts']}.json"
+
+    def _validate_attempt(self, record, batch):
+        if (not isinstance(record, dict) or set(record) != ATTEMPT_KEYS
+                or type(record["schema_version"]) is not int or record["schema_version"] != 1
+                or record["batch_id"] != batch["id"] or record["attempt"] != batch["attempts"]
+                or type(record["attempt"]) is not int or record["attempt"] < 1
+                or type(record["target_bytes"]) is not int or record["target_bytes"] != batch["target_bytes"]
+                or not isinstance(record["claim"], str) or not re.fullmatch("[a-f0-9]{32}", record["claim"])
+                or (batch["status"] == "training" and record["claim"] != batch["claim"])
+                or not isinstance(record["run"], str) or not re.fullmatch(r"\d{8}T\d{6}-[a-f0-9]{8}", record["run"])):
+            raise ValueError("연구 시도와 batch·소유권·목표의 연결이 잘못됐습니다")
+        for key in ("reports_dir", "models_dir"):
+            value = record[key]
+            if not isinstance(value, str) or not value or Path(value).is_absolute() or "\\" in value:
+                raise ValueError("연구 시도 산출물 경로가 잘못됐습니다")
+            target = (self.root / value).resolve()
+            if not target.is_relative_to(self.root / "work") or target == self.root / "work":
+                raise ValueError("연구 시도 산출물이 work 경계를 벗어났습니다")
+        for key in ("previous_pointer_sha256", "previous_training_sha256"):
+            if record[key] is not None and (not isinstance(record[key], str) or not re.fullmatch("[a-f0-9]{64}", record[key])):
+                raise ValueError("연구 이전 산출물 식별자가 잘못됐습니다")
+        if record["phase"] not in {"running", "model_failed", "complete"}:
+            raise ValueError("연구 시도 단계가 잘못됐습니다")
+        if ((record["phase"] == "model_failed" and not isinstance(record["error"], str))
+                or (record["phase"] != "model_failed" and record["error"] is not None)):
+            raise ValueError("연구 시도 오류 근거가 단계와 다릅니다")
+        if record["phase"] == "complete":
+            self._result(record["report"], batch["target_bytes"])
+            if (record["report"].get("research_batch") != attempt_binding(record)
+                    or record["report"].get("run") != record["run"]
+                    or digest(record["report"]) != record["report_sha256"]
+                    or not isinstance(record["manifest_sha256"], str)
+                    or not re.fullmatch("[a-f0-9]{64}", record["manifest_sha256"])
+                    or not isinstance(record["model_sha256"], dict) or not record["model_sha256"]):
+                raise ValueError("완료 결과의 batch 결합 또는 내용 해시가 다릅니다")
+        elif any(record[key] is not None for key in ("report", "report_sha256", "manifest_sha256", "model_sha256")):
+            raise ValueError("미완료 시도에 완료 결과가 섞였습니다")
+
+    def read_attempt(self, batch):
+        path = self.attempt_path(batch)
+        if path.is_symlink() or not path.is_file():
+            raise ValueError("미확인 연구 시도입니다. 소유권을 보존하고 복구를 차단합니다")
+        record = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
+        self._validate_attempt(record, batch)
+        return record
+
+    def start_attempt(self, batch, run, reports_dir, models_dir, previous_pointer, previous_training):
+        record = {"schema_version": 1, "batch_id": batch["id"], "claim": batch["claim"],
+                  "attempt": batch["attempts"], "target_bytes": batch["target_bytes"], "run": run,
+                  "reports_dir": reports_dir, "models_dir": models_dir,
+                  "previous_pointer_sha256": previous_pointer, "previous_training_sha256": previous_training,
+                  "phase": "running", "report": None, "report_sha256": None,
+                  "manifest_sha256": None, "model_sha256": None, "error": None}
+        with _file_lock(self.path.with_suffix(".lock")):
+            current = self._owned(self._read(), batch["id"], batch["claim"])
+            self._validate_attempt(record, current)
+            if self.attempt_path(current).exists():
+                raise ValueError("기존 연구 시도 기록을 덮어쓸 수 없습니다")
+            self._atomic_write(self.attempt_path(current), record)
+        return record
+
+    def store_result(self, batch_id, claim, report, manifest_sha256, model_sha256):
+        with _file_lock(self.path.with_suffix(".lock")):
+            batch = self._owned(self._read(), batch_id, claim)
+            record = self.read_attempt(batch)
+            if record["phase"] != "running":
+                raise ValueError("실행 중인 시도의 결과만 저장할 수 있습니다")
+            record.update(phase="complete", report=report, report_sha256=digest(report),
+                          manifest_sha256=manifest_sha256, model_sha256=model_sha256)
+            self._validate_attempt(record, batch)
+            self._atomic_write(self.attempt_path(batch), record)
+
+    def record_model_failure(self, batch_id, claim, error):
+        with _file_lock(self.path.with_suffix(".lock")):
+            state = self._read()
+            batch = self._owned(state, batch_id, claim)
+            record = self.read_attempt(batch)
+            if record["phase"] != "running":
+                raise ValueError("완료 저장 실패를 모델 실패로 기록할 수 없습니다")
+            record.update(phase="model_failed", error=str(error))
+            self._atomic_write(self.attempt_path(batch), record)
+            batch.update(status="collecting", claim=None, last_error=str(error))
+            self._write(state)
+
+    def restore_claim(self, batch, record):
+        with _file_lock(self.path.with_suffix(".lock")):
+            state = self._read()
+            current = state["batches"][-1]
+            self._validate_attempt(record, current)
+            if current["status"] != "collecting" or record["phase"] != "complete":
+                raise ValueError("복구할 완료 시도의 소유권이 아닙니다")
+            current.update(status="training", claim=record["claim"])
+            self._write(state)
+
+    def note_training_error(self, batch_id, claim, error):
+        with _file_lock(self.path.with_suffix(".lock")):
+            state = self._read()
+            self._owned(state, batch_id, claim)["last_error"] = str(error)
+            self._write(state)
+
+    @staticmethod
+    def _result(report, target):
+        try:
+            training, evidence = report["training"], report["external_resolution"]
+            count = len(report["config"]["training"]["candidate_depths"])
+            if (not count or set(report["markets"]) != MARKETS or training["status"] != "완료"
+                    or type(training["completed_candidates"]) is not int
+                    or training["completed_candidates"] != 5 * count
+                    or training["total_candidates"] != 5 * count or len(report["rounds"]) != 5 * count
+                    or not isinstance(report["run"], str) or not report["run"]
+                    or type(report["promotion"]["promote"]) is not bool
+                    or evidence["training_ready"] is not True):
+                raise ValueError("다섯 시장 전체 학습의 완료 근거가 부족합니다")
+            for key in ("target_archive_bytes", "ingested_archive_bytes", "verified_archive_bytes"):
+                if type(evidence[key]) is not int or evidence[key] < target:
+                    raise ValueError("학습 자료의 적재 목표를 충족하지 못했습니다")
+            if evidence["verified_archive_bytes"] < evidence["ingested_archive_bytes"]:
+                raise ValueError("검증되지 않은 적재 자료입니다")
+            for market in MARKETS:
+                rounds = [item for item in report["rounds"] if item["market"] == market]
+                gpu = report["markets"][market]["gpu"]
+                if (len(gpu) != count or sorted(item["candidate"] for item in rounds) != list(range(1, count + 1))
+                        or any(not isinstance(item.get("actual_device"), str)
+                               or not item["actual_device"].startswith("cuda:") for item in [*gpu, *rounds])):
+                    raise ValueError("모든 후보의 GPU 학습 완료 근거가 부족합니다")
+        except (KeyError, TypeError, AttributeError) as exc:
+            raise ValueError("학습 완료 보고서 schema가 잘못됐습니다") from exc
+
+    def complete_training(self, batch_id, claim, report):
+        with _file_lock(self.path.with_suffix(".lock")):
+            state = self._read()
+            batch = self._owned(state, batch_id, claim)
+            record = self.read_attempt(batch)
+            if record["phase"] != "complete" or digest(report) != record["report_sha256"]:
+                raise ValueError("내구성 있는 완료 결과와 제어 상태가 다릅니다")
+            self._result(report, batch["target_bytes"])
+            batch.update(status="completed", claim=None, run=report["run"], completed_at=_now(),
+                         promoted=report["promotion"]["promote"])
+            self._write(state)

```

## engine.py

이전: 104789 bytes / SHA-256 bb132af274660fef567cc1bad5b7f4d7c1cf4427bbe9fcdb6c3701f8ac2896c5  
현재: 118210 bytes / SHA-256 2a799fc1c41c7a1eb1af9e23a74746c0420fbcc21930f6d4030061fbffd8a97d

```diff
--- 이전/engine.py
+++ 현재/engine.py
@@ -41,6 +41,7 @@
     "external": {"enabled": False, "db_path": "3차/work/data/external.sqlite3",
                  "status_path": "3차/work/status.json", "wait_for_dataset": False,
                  "target_bytes": 50000000000},
+    "research": {"active": False, "state_path": "4차/work/research_control.json"},
     "collection": {"recent_pages": 10, "backfill_pages_per_cycle": 5,
                    "poll_seconds": 30, "request_interval": 0.16, "retries": 4},
     "training": {"horizon": 12, "label_threshold": 0.006,
@@ -100,8 +101,8 @@
             if key not in config:
                 raise ValueError(f"지원하지 않는 설정: {key}")
             if isinstance(config[key], dict):
-                if key == "external" and not isinstance(value, dict):
-                    raise ValueError("external 설정은 객체여야 합니다")
+                if key in {"external", "research"} and not isinstance(value, dict):
+                    raise ValueError(f"{key} 설정은 객체여야 합니다")
                 unknown = set(value) - set(config[key])
                 if unknown:
                     raise ValueError(f"지원하지 않는 설정: {key}.{sorted(unknown)}")
@@ -163,6 +164,10 @@
         raise ValueError("수집 페이지와 대기 시간은 양수여야 합니다")
     utc(paper["start_time"])
     config["external"] = import_module("3차.external_features").validate_config(config["external"], ROOT)
+    config["research"] = import_module("4차.research_control").validate_config(config["research"], ROOT)
+    if config["research"]["active"] and (config["external"]["enabled"] is False
+                                         or not config["external"]["wait_for_dataset"]):
+        raise ValueError("명시 요청 연구에는 외부 자료 사용과 준비 대기 검사가 필요합니다")
     return config
 
 
@@ -249,7 +254,11 @@
 
     def summary(self):
         with connect(self.path) as db:
-            rows = db.execute("SELECT market,tf,COUNT(*),MIN(open_time),MAX(close_time) FROM candles GROUP BY market,tf").fetchall()
+            # 전체 행의 테이블 재접근을 피하고 같은 읽기 스냅샷에서 경계만 따로 조회합니다.
+            db.execute("BEGIN")
+            grouped = db.execute("SELECT market,tf,COUNT(*),MAX(close_time) FROM candles GROUP BY market,tf").fetchall()
+            rows = [(m, tf, n, db.execute("SELECT MIN(open_time) FROM candles WHERE market=? AND tf=?",
+                                         (m, tf)).fetchone()[0], end) for m, tf, n, end in grouped]
             cursors = {(r[0], r[1]): bool(r[2]) for r in db.execute("SELECT market,tf,done FROM cursors")}
         return [{"market": m, "timeframe": tf, "rows": n, "start": a, "end": b,
                  "backfill_exhausted": cursors.get((m, tf), False)} for m, tf, n, a, b in rows]
@@ -1047,8 +1056,13 @@
             db.execute("UPDATE final_tests SET status='complete' WHERE id=?", (run,))
 
 
+_research_progress = ContextVar("research_progress", default=None)
+_research_attempt = ContextVar("research_attempt", default=None)
+
+
 def progress(**fields):
-    atomic_json(ROOT / "work/service/training_progress.json", {"updated_at": iso(None), **fields})
+    atomic_json(ROOT / "work/service/training_progress.json",
+                {"updated_at": iso(None), **fields, **(_research_progress.get() or {})})
 
 
 def protected_periods(config, consumed=()):
@@ -1254,7 +1268,210 @@
                     active_error.add_note("선행 특징 준비 실패:\n" + traceback.format_exc())
 
 
+def _research_file_hash(path):
+    path = Path(path)
+    if path.is_symlink():
+        raise ValueError("연구 산출물의 심볼릭 링크는 허용하지 않습니다")
+    if not path.exists():
+        return None
+    digest = hashlib.sha256()
+    with path.open("rb") as stream:
+        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
+            digest.update(chunk)
+    return digest.hexdigest()
+
+
+def _research_json(path):
+    return json.loads(Path(path).read_text(encoding="utf-8"),
+                      object_pairs_hook=import_module("4차.research_control")._pairs)
+
+
+def _research_model_hashes(models_dir, run, manifest):
+    directory = Path(models_dir) / "versions" / run
+    if not directory.resolve().is_relative_to(Path(models_dir).resolve()):
+        raise ValueError("연구 모델 경계가 잘못됐습니다")
+    if set(manifest["packages"]) != set(MARKETS):
+        raise ValueError("완료 모델의 다섯 시장이 다릅니다")
+    hashes = {}
+    for package in manifest["packages"].values():
+        for name in package["models"]:
+            if (not isinstance(name, str) or Path(name).name != name or "\\" in name
+                    or not name.endswith(".ubj") or name in hashes):
+                raise ValueError("완료 모델 파일 경로·중복 오류")
+            hashes[name] = _research_file_hash(directory / name)
+            if hashes[name] is None:
+                raise ValueError("완료 모델 파일이 없습니다")
+    return hashes
+
+
+def validate_research_result(control, batch, record):
+    control._validate_attempt(record, batch)
+    if record["phase"] != "complete":
+        raise ValueError("내구성 있는 연구 완료 결과가 없습니다")
+    report = record["report"]
+    models = ROOT / record["models_dir"]
+    reports = ROOT / record["reports_dir"]
+    manifest = _research_json(models / "versions" / record["run"] / "manifest.json")
+    binding = import_module("4차.research_control").attempt_binding(record)
+    if (stable_hash(manifest) != record["manifest_sha256"] or manifest.get("research_batch") != binding
+            or manifest.get("run") != record["run"] or manifest.get("paper_only") is not True
+            or manifest.get("source_hash") != report.get("source_hash")
+            or manifest.get("config") != report["config"]
+            or manifest.get("schema_version") != 2 or manifest.get("execution_policy") != EXECUTION_POLICY
+            or _research_model_hashes(models, record["run"], manifest) != record["model_sha256"]
+            or len(record["model_sha256"]) != report["training"]["total_candidates"]):
+        raise ValueError("완료 보고서·manifest·모델 파일의 결합 또는 내용이 다릅니다")
+    result_path = reports / "experiments" / (record["run"] + ".json")
+    if result_path.exists() and stable_hash(_research_json(result_path)) != record["report_sha256"]:
+        raise ValueError("완료 보고서가 변경됐거나 다른 요청의 결과입니다")
+    latest = reports / "training.json"
+    if _research_file_hash(latest) != record["previous_training_sha256"]:
+        if not latest.is_file() or stable_hash(_research_json(latest)) != record["report_sha256"]:
+            raise ValueError("최신 학습 보고서가 복구 대상 또는 이전 보고서와 다릅니다")
+    pointer = models / "current.json"
+    if _research_file_hash(pointer) != record["previous_pointer_sha256"]:
+        expected = {"run": record["run"], "manifest_hash": record["manifest_sha256"]}
+        if (not report["promotion"]["promote"] or not pointer.is_file()
+                or _research_json(pointer) != expected):
+            raise ValueError("현재 모델 포인터가 복구 대상 또는 이전 포인터와 다릅니다")
+    return report
+
+
+def _durable_research_json(path, value):
+    atomic_json(path, value)
+    if os.name != "nt":
+        fd = os.open(Path(path).parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
+        try:
+            os.fsync(fd)
+        finally:
+            os.close(fd)
+
+
+def finish_research_outputs(record, report):
+    reports, models = ROOT / record["reports_dir"], ROOT / record["models_dir"]
+    _durable_research_json(reports / "experiments" / (record["run"] + ".json"), report)
+    _durable_research_json(reports / "training.json", report)
+    if report["promotion"]["promote"]:
+        _durable_research_json(models / "current.json",
+                               {"run": record["run"], "manifest_hash": record["manifest_sha256"]})
+    training = report["training"]
+    progress(status="complete", stage="complete", run=record["run"],
+             started_at=training.get("started_at"), elapsed_seconds=training.get("duration_seconds"),
+             market=None, candidate=None, completed_candidates=training["completed_candidates"],
+             total_candidates=training["total_candidates"], error=None,
+             rows=training.get("rows"), features=training.get("features"), promoted=report["promotion"]["promote"])
+
+
+def recover_research_result(control, batch):
+    record = control.read_attempt(batch)
+    if record["phase"] == "model_failed":
+        report_path = ROOT / record["reports_dir"] / "experiments" / (record["run"] + ".json")
+        if report_path.exists():
+            raise ValueError("모델 실패 기록과 완료 보고서가 충돌합니다")
+        if batch["status"] == "training":
+            control.fail_training(batch["id"], batch["claim"], record["error"])
+        return None
+    if record["phase"] != "complete":
+        raise ValueError("이전 시도의 완료 여부가 미확인입니다. claim을 보존하고 재적합을 차단합니다")
+    report = validate_research_result(control, batch, record)
+    if batch["status"] == "collecting":
+        control.restore_claim(batch, record)
+    finish_research_outputs(record, report)
+    control.complete_training(batch["id"], record["claim"], report)
+    return report
+
+
 def train_cycle(config, final=False, live=False):
+    settings = config.get("research", {})
+    if not settings.get("active", False):
+        return _train_cycle(config, final=final, live=live)
+    external = import_module("3차.external_features")
+    resolved = external.validate_config(config["external"], ROOT)
+    if resolved["enabled"] is False or not resolved["wait_for_dataset"]:
+        raise ValueError("명시 요청 연구의 외부 자료 준비 검사를 생략할 수 없습니다")
+    control = import_module("4차.research_control").ResearchControl(ROOT, settings)
+    token = _research_progress.set({"research_active": True, "pid": os.getpid()})
+    try:
+        with control.training_lock():
+            batch = wait_for_research_batch(control)
+            _research_progress.set({"research_active": True, "pid": os.getpid(), "research_batch_id": batch["id"]})
+            if batch["attempts"]:
+                recovered = recover_research_result(control, batch)
+                if recovered is not None:
+                    return recovered
+                batch = control.pending_batch()
+            batch_config = copy.deepcopy(config)
+            batch_config["external"] = {**resolved, "target_bytes": batch["target_bytes"]}
+            _research_progress.set({"research_active": True, "pid": os.getpid(), "research_batch_id": batch["id"]})
+            wait_for_training_data(batch_config)
+            claim = control.claim_training(batch["id"])
+            models, reports = Path(config["models_dir"]), Path(config["reports_dir"])
+            record = control.start_attempt(claim, utc(None).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8],
+                                           reports.relative_to(ROOT).as_posix(), models.relative_to(ROOT).as_posix(),
+                                           _research_file_hash(models / "current.json"),
+                                           _research_file_hash(reports / "training.json"))
+            attempt_token = _research_attempt.set({**record, "control": control, "result_started": False})
+            try:
+                report = _train_cycle(batch_config, final=final, live=live)
+                validate_research_result(control, claim, control.read_attempt(claim))
+                control.complete_training(batch["id"], claim["claim"], report)
+                return report
+            except BaseException as error:
+                try:
+                    saved = control.read_attempt(claim)
+                    if saved["phase"] == "complete" or _research_attempt.get()["result_started"]:
+                        control.note_training_error(batch["id"], claim["claim"], repr(error))
+                    else:
+                        control.record_model_failure(batch["id"], claim["claim"], repr(error))
+                except Exception as record_error:
+                    error.add_note(f"연구 실패 상태 저장 오류: {record_error!r}")
+                raise
+            finally:
+                _research_attempt.reset(attempt_token)
+    finally:
+        _research_progress.reset(token)
+
+
+def wait_for_research_batch(control, stop=None):
+    stop = threading.Event() if stop is None else stop
+    while not stop.is_set():
+        batch = control.pending_batch()
+        if batch is not None:
+            return batch
+        progress(status="waiting_request", stage="idle", reason="명시적인 다음 연구 요청 대기", error=None,
+                 run=None, started_at=None, elapsed_seconds=None, progress=None,
+                 completed_candidates=None, total_candidates=None, candidate=None, market=None,
+                 current_round=None, target_rounds=None, rows=None, features=None,
+                 research_batch_id=None)
+        if stop.wait(12):
+            break
+    raise KeyboardInterrupt("연구 요청 대기 중 중단 요청")
+
+
+def request_research_increment(config):
+    control = import_module("4차.research_control").ResearchControl(ROOT, config.get("research", {}))
+
+    def actual_ingested():
+        external = import_module("3차.external_features")
+        settings = external.validate_config(config["external"], ROOT)
+        # 과거 완료 목표나 캐시된 status 용량이 아니라 현재 Linux DB의 적재 근거를 읽습니다.
+        external.resolve_for_training(settings, ROOT, utc(None))
+        with external._database(settings, ROOT) as db:
+            values = dict(db.execute("SELECT key,value FROM metadata WHERE key IN "
+                                     "('ingested_archive_bytes','verified_archive_bytes')"))
+            for key in ("ingested_archive_bytes", "verified_archive_bytes"):
+                value = values.get(key)
+                if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
+                    raise ValueError("현재 외부 DB의 고유 적재량 근거가 유효하지 않습니다")
+            ingested, verified = int(values["ingested_archive_bytes"]), int(values["verified_archive_bytes"])
+            if verified < ingested:
+                raise ValueError("현재 외부 DB의 검증·적재량이 일치하지 않습니다")
+            return ingested
+
+    return control.request_increment(actual_ingested)
+
+
+def _train_cycle(config, final=False, live=False):
     started = time.monotonic()
     started_at = iso(None)
     if final and live:
@@ -1274,7 +1491,8 @@
         raise ValueError("잠금 시험 기간이 끝나지 않았습니다")
     if final:
         audit.assert_final_available(*plan["holdout"])
-    run = utc(None).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
+    attempt = _research_attempt.get()
+    run = attempt["run"] if attempt is not None else utc(None).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
 
     def update_progress(**fields):
         progress(run=run, started_at=started_at, elapsed_seconds=time.monotonic() - started, **fields)
@@ -1298,6 +1516,8 @@
                            "fold_id": stable_hash(report_plan),
                            "periods": report_plan},
               "markets": {}, "training": {"status": "학습 중", "gpu_required": True, "started_at": started_at}, "rounds": []}
+    if attempt is not None:
+        report["research_batch"] = import_module("4차.research_control").attempt_binding(attempt)
     with market_feature_loader(store, config, plan["train"][0], plan["validation"][1]) as read_market:
         for market in config["markets"]:
             update_progress(status="running", stage="features", market=market, candidate=None,
@@ -1393,6 +1613,8 @@
                 "run": run, "feature_columns": columns, "packages": packages, "config": config,
                 "source_hash": report["source_hash"], "paper_only": True,
                 "plan": report["periods"], "created_at": iso(None), "external_resolution": external_resolution}
+    if attempt is not None:
+        manifest["research_batch"] = report["research_batch"]
     atomic_json(directory / "manifest.json", manifest)
     manifest_hash = stable_hash(manifest)
     if final:
@@ -1426,6 +1648,15 @@
                               rows=sum(r["splits"]["train"]["rows"] for r in report["markets"].values()),
                               features=len(columns))
     report["class_distribution"] = {m: r["validation"]["class_distribution"] for m, r in report["markets"].items()}
+    if attempt is not None:
+        attempt["result_started"] = True
+        control = attempt["control"]
+        control.store_result(attempt["batch_id"], attempt["claim"], report, manifest_hash,
+                             _research_model_hashes(config["models_dir"], run, manifest))
+        saved = control.read_attempt(control.pending_batch())
+        validate_research_result(control, control.pending_batch(), saved)
+        finish_research_outputs(saved, report)
+        return report
     atomic_json(Path(config["reports_dir"]) / "experiments" / (run + ".json"), report)
     atomic_json(Path(config["reports_dir"]) / "training.json", report)
     # 다섯 코인의 모델과 보고서가 모두 완성된 뒤 포인터 하나만 바꿉니다. 이전 버전은 삭제하지 않습니다.
@@ -1455,7 +1686,9 @@
                  ingested_archive_bytes=evidence.get("ingested_archive_bytes", 0),
                  target_archive_bytes=evidence.get("target_archive_bytes", 50000000000),
                  checked_at=evidence["checked_at"], rows=None, features=None,
-                 run=None, started_at=None, elapsed_seconds=None)
+                 run=None, started_at=None, elapsed_seconds=None, progress=None,
+                 completed_candidates=None, total_candidates=None, candidate=None, market=None,
+                 current_round=None, target_rounds=None)
         if stop.wait(12):
             break
         pending = None
@@ -1723,6 +1956,7 @@
     experiment = sub.add_parser("experiment")
     experiment.add_argument("--final", action="store_true")
     sub.add_parser("check-gpu")
+    sub.add_parser("research-request", help="현재 실제 적재량에 새 고유 2GB를 더하는 연구 요청")
     return parser
 
 
@@ -1731,6 +1965,9 @@
     config = load_config(args.config)
     if args.command == "check-gpu":
         print(json.dumps(check_gpu(), ensure_ascii=False))
+        return 0
+    if args.command == "research-request":
+        print(json.dumps(request_research_increment(config), ensure_ascii=False))
         return 0
     if args.command == "experiment":
         with process_lock(ROOT / "work/service/trainer.lock"):

```

## supervisor.py

이전: 9599 bytes / SHA-256 4a2635c42145c201a4e57399df94dfd90149b0361b307d5c9607ff6d1ebab1a3  
현재: 11416 bytes / SHA-256 ed06606bd2f35a5788ce89aebdc49cf3509cd7dc8035bc8ad2698b0e75e69f57

```diff
--- 이전/supervisor.py
+++ 현재/supervisor.py
@@ -75,6 +75,42 @@
     handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
     logger.addHandler(handler)
     return logger
+
+
+def training_heartbeat(name, work):
+    if name != "trainer":
+        return {}
+    try:
+        value = json.loads((work / "service/training_progress.json").read_text(encoding="utf-8"))
+        return value if isinstance(value, dict) else {}
+    except (OSError, ValueError, UnicodeError):
+        return {}
+
+
+class TrainingBudget:
+    def __init__(self, limit, started_tick, started_utc, pid):
+        self.limit = limit
+        self.previous_tick = started_tick
+        self.started_utc = started_utc
+        self.pid = pid
+        self.active_seconds = 0.0
+
+    def expired(self, tick, now, heartbeat):
+        waiting = False
+        if isinstance(heartbeat, dict):
+            try:
+                stamp = dt.datetime.fromisoformat(heartbeat["updated_at"])
+                waiting = (heartbeat.get("research_active") is True and heartbeat.get("pid") == self.pid
+                           and self.started_utc <= stamp <= now
+                           and (now - stamp).total_seconds() <= 36
+                           and (heartbeat.get("status"), heartbeat.get("stage"))
+                           in {("waiting_request", "idle"), ("waiting_data", "download")})
+            except (KeyError, ValueError, TypeError):
+                waiting = False
+        if not waiting:
+            self.active_seconds += max(0, tick - self.previous_tick)
+        self.previous_tick = tick
+        return self.active_seconds >= self.limit
 
 
 def worker(name: str, settings: dict, work: Path, start_at: dt.datetime) -> None:
@@ -106,7 +142,9 @@
             process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding="utf-8", errors="strict", bufsize=1, start_new_session=os.name != "nt")
             limit = settings.get("timeout_seconds", 7200)
-            deadline = time.monotonic() + limit if limit is not None else float("inf")
+            started_tick = time.monotonic()
+            deadline = started_tick + limit if limit is not None else float("inf")
+            budget = TrainingBudget(limit, started_tick, started, process.pid) if name == "trainer" and limit is not None else None
             # 출력 배출은 별도 스레드가 맡아 중단과 시간 제한 감시를 막지 않습니다.
             def drain() -> None:
                 try:
@@ -117,8 +155,14 @@
 
             reader = threading.Thread(target=drain, daemon=True)
             reader.start()
+
+            def timed_out():
+                tick = time.monotonic()
+                return (budget.expired(tick, utcnow(), training_heartbeat(name, work))
+                        if budget is not None else tick >= deadline)
+
             while process.poll() is None:
-                if STOP.wait(0.5) or time.monotonic() >= deadline:
+                if STOP.wait(0.5) or timed_out():
                     process.terminate()
                     try:
                         process.wait(timeout=20)

```

## dashboard.py

이전: 33621 bytes / SHA-256 01b628748ce57c6686770898d0b5be2afa36d7671409970c3edb00094edb0ec5  
현재: 33663 bytes / SHA-256 42af6e83af9193a2e7d4104f26505893f060d5436f84810084dba10e7e726a50

```diff
--- 이전/dashboard.py
+++ 현재/dashboard.py
@@ -338,7 +338,7 @@
         current = mapping(training.get("training"))
         completed = progress.get("completed_candidates", current.get("completed_candidates"))
         total = progress.get("total_candidates", current.get("total_candidates"))
-        if progress.get("status") == "waiting_data":
+        if progress.get("status") in ("waiting_data", "waiting_request"):
             completed = total = None
         training_view = {**training, "state": progress.get("status", current.get("status", training.get("state"))),
                          "stage": progress.get("stage"), "market": progress.get("market"),
@@ -358,7 +358,7 @@
                          "learning_rate": progress.get("learning_rate", progress.get("eta"))}
         if finite(completed) and finite(total) and total > 0:
             training_view["progress"] = completed / total
-        if progress.get("status") == "waiting_data":
+        if progress.get("status") in ("waiting_data", "waiting_request"):
             training_view["progress"] = None
         # 보고서 설정 전체가 아닌 실제 사용 시간봉 한 항목만 표시합니다.
         training_view["timeframe"] = mapping(training.get("config")).get("base_timeframe")

```

## engine-config.json

이전: 1278 bytes / SHA-256 39099ca5e2d6655f7af6c855da67da2e09b2a239911a78bcb96242f4631485ad  
현재: 1359 bytes / SHA-256 f55d016ce131667b7f81a724eaaac5c11da7d2f90bfbc750672d57ba8dd5196f

```diff
--- 이전/engine-config.json
+++ 현재/engine-config.json
@@ -1,4 +1,5 @@
 {
+  "research": {"active": true, "state_path": "4차/work/research_control.json"},
   "collection": {"recent_pages": 2, "backfill_pages_per_cycle": 20, "poll_seconds": 30, "request_interval": 0.18, "retries": 4},
   "external": {"enabled": "auto", "wait_for_dataset": true, "target_bytes": 50000000000,
                "db_path": "3차/work/data/external.sqlite3", "status_path": "3차/work/status.json"},

```

## web/app.js

이전: 49154 bytes / SHA-256 4b4225e7b7bef75285a2504938c258dfa8a2ef32d49c8d27bd780e17320bf8ae  
현재: 49286 bytes / SHA-256 24fb606a6344b96fdc32d52554a3731b79bf637d39b1d4da60bc8f1934f9b226

```diff
--- 이전/web/app.js
+++ 현재/web/app.js
@@ -220,8 +220,8 @@
   const starts = trainSplits.map(split => split.period_start).filter(value => typeof value === 'string').sort();
   const ends = trainSplits.map(split => split.period_end).filter(value => typeof value === 'string').sort();
   const gpuFits = marketEntries.flatMap(([, report]) => report.gpu || []);
-  const states = {running: '진행 중', complete: '완료', failed: '오류', error: '오류', waiting_data: '50GB 자료 준비'};
-  const stages = {features: '피처 구성', fit: '후보 학습', calibration: '확률 보정', locked_test: '잠금 시험', complete: '완료', download: '수집 · 검증 · 처리'};
+  const states = {running: '진행 중', complete: '완료', failed: '오류', error: '오류', waiting_data: '자료 준비', waiting_request: '추가 요청 대기'};
+  const stages = {features: '피처 구성', fit: '후보 학습', calibration: '확률 보정', locked_test: '잠금 시험', complete: '완료', download: '수집 · 검증 · 처리', idle: '연구 대기'};
   const external = s.external_data || {};
   const gb = value => numeric(value) ? `${number(value / 1e9, 2)} GB` : '미기록';
   const stepIndex = t.state === 'running' ? ({features: 0, fit: 1, calibration: 2, locked_test: 2, complete: 3})[t.stage] : undefined;
@@ -232,7 +232,7 @@
     else item.removeAttribute('aria-current');
   });
   set('workflow-status', stepIndex === undefined ? `현재 단계: ${states[t.state] || '기록 대기'}` : `현재 단계: ${stepNames[stepIndex]} · ${stages[t.stage]}`);
-  set('workflow-title', t.state === 'running' ? '가격의 흐름을 배우고 있어요' : t.state === 'complete' ? '이번 학습을 마쳤어요' : ['failed', 'error'].includes(t.state) ? '학습 상태를 확인하고 있어요' : '가격의 흐름을 배우는 과정');
+  set('workflow-title', t.state === 'running' ? '가격의 흐름을 배우고 있어요' : t.state === 'complete' ? '이번 학습을 마쳤어요' : t.state === 'waiting_request' ? '다음 연구 요청을 기다려요' : ['failed', 'error'].includes(t.state) ? '학습 상태를 확인하고 있어요' : '가격의 흐름을 배우는 과정');
   details('learning-strip', [
     ['현재 단계', stepIndex === undefined ? (states[t.state] || '기록 대기') : stepNames[stepIndex]],
     t.state === 'waiting_data' ? ['처리 완료 / 목표', `${gb(external.ingested_archive_bytes)} / ${gb(external.target_archive_bytes)}`] : ['완료 후보', `${number(t.candidate_index)} / ${number(t.candidate_total)}`],

```

## tests/test_external_dashboard.py

이전: 1877 bytes / SHA-256 a2f4d5e0b3a5d2799143f700552e56c68f2bcee05d77b0033aa8c0ada421e905  
현재: 2866 bytes / SHA-256 19f1986ccafb717d2568936516153065436920bb64bc544cee07ebb08ad394d6

```diff
--- 이전/tests/test_external_dashboard.py
+++ 현재/tests/test_external_dashboard.py
@@ -8,6 +8,22 @@
 
 
 class ExternalDashboardTests(unittest.TestCase):
+    def test_waiting_request_does_not_reuse_completed_report_progress(self):
+        with tempfile.TemporaryDirectory() as temp:
+            root = Path(temp)
+            values = {'work/reports/training.json': {'progress': 1.0, 'training': {'completed_candidates': 15, 'total_candidates': 15}},
+                      'work/service/training_progress.json': {'status': 'waiting_request', 'stage': 'idle'}}
+            for name, value in values.items():
+                path = root / name
+                path.parent.mkdir(parents=True, exist_ok=True)
+                path.write_text(json.dumps(value), encoding='utf-8')
+            training = dashboard.SnapshotAdapter(root).status()['training']
+            self.assertEqual(training['state'], 'waiting_request')
+            self.assertEqual(training['stage'], 'idle')
+            self.assertIsNone(training['progress'])
+            self.assertIsNone(training['candidate_index'])
+            self.assertIsNone(training['candidate_total'])
+
     def test_waiting_data_exposes_volume_but_not_private_error_or_old_progress(self):
         with tempfile.TemporaryDirectory() as temp:
             root = Path(temp)

```

## tests/test_external_data.py

이전: 13400 bytes / SHA-256 b8ebe565850f6edfa7d62762323ce122f223cc51d58a61b631e723c54454826b  
현재: 34598 bytes / SHA-256 34c6c887e80386a59fbf6d8097542448fac77a83e2545b401b195e3d9abb721e

```diff
--- 이전/tests/test_external_data.py
+++ 현재/tests/test_external_data.py
@@ -2,6 +2,10 @@
 import importlib
 import hashlib
 import io
+import json
+import os
+import shutil
+import subprocess
 import sys
 import tempfile
 import unittest
@@ -170,6 +174,347 @@
             archive.assert_called_once_with(spec)
 
 
+@unittest.skipUnless(sys.platform == 'linux', '격리와 부분 저장은 Linux 임시 DB에서 검사합니다')
+class ExternalQuarantineTests(unittest.TestCase):
+    def zip_bytes(self, spec, rows, member=None):
+        data = io.BytesIO()
+        with zipfile.ZipFile(data, 'w') as archive:
+            archive.writestr(member or spec['name'][:-4] + '.csv',
+                             pd.DataFrame(rows).to_csv(index=False, header=False))
+        return data.getvalue()
+
+    def fixtures(self, collector):
+        start = 1769904000000
+        bad = external.archive_spec('BTCUSDT', 'aggTrades', '2026-02')
+        good = external.archive_spec('BTCUSDT', 'aggTrades', '2026-03')
+        rows = [[1, 100., 2., 1, 1, start, False, True],
+                [2, 101., 1., 2, 2, start + 60000, True, True],
+                [1, 100., 2., 1, 1, start, False, True],
+                [3, 102., 1., 3, 3, start + 120000, False, True]]
+        contents = {bad['name']: self.zip_bytes(bad, rows),
+                    good['name']: self.zip_bytes(good, [
+                        [10, 110., 3., 10, 10, 1772323200000, False, True]])}
+        for name, content in contents.items():
+            (collector.archives / name).write_bytes(content)
+        return bad, good, contents
+
+    def network(self, contents):
+        def get(url, timeout=30):
+            name = url.rsplit('/', 1)[-1]
+            if name.endswith('.CHECKSUM'):
+                name = name[:-9]
+                digest = hashlib.sha256(contents[name]).hexdigest()
+                return ExternalDownloadTests().response((digest + '  ' + name).encode())
+            return ExternalDownloadTests().response(contents[name])
+        return get
+
+    def parse_two_rows(self):
+        reader = pd.read_csv
+        return patch.object(external.pd, 'read_csv',
+                            side_effect=lambda *a, **kw: reader(*a, **{**kw, 'chunksize': 2}))
+
+    def reject_without_aborting(self, collector, spec):
+        try:
+            collector.archive(spec)
+        except ValueError as error:
+            self.fail('내용 오류는 명시적 미적재 격리 후 다음 파일 진행이 필요합니다: ' + str(error))
+
+    def test_rejection_preserves_prefix_excludes_bad_bytes_and_next_archive_succeeds(self):
+        with tempfile.TemporaryDirectory() as temp:
+            collector = external.Collector(Path(temp))
+            bad, good, contents = self.fixtures(collector)
+            with patch.object(external, 'open_public', side_effect=self.network(contents)), self.parse_two_rows():
+                self.reject_without_aborting(collector, bad)
+                rejected = collector.manifest[bad['name']]
+                digest = hashlib.sha256(contents[bad['name']]).hexdigest()
+                self.assertFalse(rejected['ingested'])
+                self.assertEqual(rejected['content_rejections'][digest]['sha256'], digest)
+                self.assertIn('역행', rejected['content_rejections'][digest]['reason'])
+                self.assertEqual(collector.status()['ingested_archive_bytes'], 0)
+                self.assertEqual(collector.status()['archive_warning_count'], 1)
+                self.assertEqual((collector.archives / bad['name']).read_bytes(), contents[bad['name']])
+                collector.archive(good)
+            state = collector.status()
+            self.assertEqual(state['ingested_archive_bytes'], len(contents[good['name']]))
+            self.assertEqual(state['verified_archive_bytes'], sum(map(len, contents.values())))
+            self.assertEqual(state['archive_warning_count'], 1)
+            self.assertIsNone(state['error'])
+            self.assertFalse(state['training_ready'])
+            self.assertEqual(state['target_archive_bytes'], 50_000_000_000)
+            with collector.store.connect() as db:
+                prefix = db.execute('SELECT open_time,volume,quote_volume,trade_count,source FROM bars '
+                                    'WHERE open_time<1772323200000').fetchall()
+            self.assertEqual(prefix, [(1769904000000, 2., 200., 1, bad['name'])])
+
+    def test_same_hash_restart_does_not_reparse_or_change_prefix(self):
+        with tempfile.TemporaryDirectory() as temp:
+            collector = external.Collector(Path(temp))
+            bad, _, contents = self.fixtures(collector)
+            with patch.object(external, 'open_public', side_effect=self.network(contents)), self.parse_two_rows():
+                self.reject_without_aborting(collector, bad)
+            before = collector.manifest[bad['name']]
+            restarted = external.Collector(Path(temp))
+            with patch.object(external, 'open_public', side_effect=self.network(contents)), \
+                 patch.object(restarted, 'ingest', side_effect=AssertionError('같은 격리 해시를 재파싱했습니다')):
+                restarted.archive(bad)
+            self.assertEqual(restarted.manifest[bad['name']], before)
+            self.assertEqual(restarted.status()['ingested_archive_bytes'], 0)
+            with restarted.store.connect() as db:
+                self.assertEqual(db.execute('SELECT COUNT(*),SUM(volume) FROM bars').fetchone(), (1, 2.))
+
+    def test_new_official_hash_can_ingest_while_old_file_and_rejection_are_preserved(self):
+        with tempfile.TemporaryDirectory() as temp:
+            collector = external.Collector(Path(temp))
+            bad, _, contents = self.fixtures(collector)
+            original = contents[bad['name']]
+            with patch.object(external, 'open_public', side_effect=self.network(contents)), self.parse_two_rows():
+                self.reject_without_aborting(collector, bad)
+            old_digest = hashlib.sha256(original).hexdigest()
+            contents[bad['name']] = self.zip_bytes(bad, [
+                [1, 100., 2., 1, 1, 1769904000000, False, True],
+                [2, 101., 1., 2, 2, 1769904060000, True, True]])
+            with patch.object(external, 'open_public', side_effect=self.network(contents)):
+                collector.archive(bad)
+            state = collector.status()
+            self.assertTrue(collector.manifest[bad['name']]['ingested'])
+            self.assertNotEqual(collector.manifest[bad['name']]['sha256'], old_digest)
+            self.assertIn(old_digest, collector.manifest[bad['name']]['content_rejections'])
+            self.assertEqual(state['ingested_archive_bytes'], len(contents[bad['name']]))
+            self.assertEqual(state['archive_warning_count'], 1)
+            preserved = list(collector.archives.glob(bad['name'] + '.rejected-*'))
+            self.assertTrue(any(p.read_bytes() == original for p in preserved if p.suffix != '.json'))
+            with collector.store.connect() as db:
+                self.assertEqual(db.execute('SELECT COUNT(*),SUM(volume) FROM bars').fetchone(), (2, 3.))
+            contents[bad['name']] = original
+            with patch.object(external, 'open_public', side_effect=self.network(contents)), \
+                 patch.object(collector, 'ingest', side_effect=AssertionError('격리 해시를 재적재했습니다')):
+                collector.archive(bad)
+            self.assertFalse(collector.manifest[bad['name']]['ingested'])
+            self.assertEqual(collector.status()['ingested_archive_bytes'], 0)
+
+    def test_run_continues_after_content_rejection(self):
+        with tempfile.TemporaryDirectory() as temp:
+            collector = external.Collector(Path(temp))
+            bad, good, contents = self.fixtures(collector)
+            stop = external.threading.Event()
+            with patch.object(external, 'STOP', stop), patch.object(external.threading, 'Thread') as thread, \
+                 patch.object(external, 'bootstrap_plan', return_value=[]), \
+                 patch.object(external, 'bulk_plan', return_value=[bad, good]), \
+                 patch.object(external, 'open_public', side_effect=self.network(contents)), \
+                 self.parse_two_rows(), patch.object(stop, 'wait', side_effect=lambda _: stop.set()):
+                thread.return_value.is_alive.return_value = False
+                collector.run()
+            self.assertIn(good['name'], collector.manifest)
+            self.assertTrue(collector.manifest[good['name']]['ingested'])
+            self.assertFalse(collector.manifest[bad['name']]['ingested'])
+            self.assertIsNone(collector.status()['error'])
+
+    def test_path_and_database_failures_are_not_quarantined_as_content(self):
+        with tempfile.TemporaryDirectory() as temp:
+            collector = external.Collector(Path(temp))
+            spec = external.archive_spec('BTCUSDT', 'aggTrades', '2026-02')
+            rows = [[1, 100., 1., 1, 1, 1769904000000, False, True]]
+            for invalid_path in (True, False):
+                content = self.zip_bytes(spec, rows, '../escape.csv' if invalid_path else None)
+                (collector.archives / spec['name']).write_bytes(content)
+                with self.subTest(invalid_path=invalid_path), \
+                     patch.object(external, 'open_public', side_effect=self.network({spec['name']: content})), \
+                     patch.object(collector.store, 'save', side_effect=ValueError('DB저장 실패')):
+                    with self.assertRaisesRegex(ValueError, '경계' if invalid_path else 'DB저장'):
+                        collector.archive(spec)
+                self.assertFalse(collector.manifest[spec['name']].get('content_rejections'))
+
+    def test_official_404_does_not_erase_existing_content_rejection(self):
+        with tempfile.TemporaryDirectory() as temp:
+            collector = external.Collector(Path(temp))
+            bad, _, contents = self.fixtures(collector)
+            with patch.object(external, 'open_public', side_effect=self.network(contents)), self.parse_two_rows():
+                self.reject_without_aborting(collector, bad)
+            digest = hashlib.sha256(contents[bad['name']]).hexdigest()
+            rejection = collector.manifest[bad['name']]['content_rejections'][digest].copy()
+            with patch.object(external, 'open_public', side_effect=external.urllib.error.HTTPError(
+                    bad['url'], 404, '미제공', {}, None)):
+                collector.archive(bad)
+            self.assertEqual(collector.status()['archive_warning_count'], 1)
+            self.assertEqual(collector.manifest[bad['name']]['content_rejections'][digest], rejection)
+            self.assertFalse(collector.manifest[bad['name']]['ingested'])
+
+    def test_mismatched_or_malformed_rejection_binding_is_not_silently_accepted(self):
+        with tempfile.TemporaryDirectory() as temp:
+            collector = external.Collector(Path(temp))
+            bad, _, contents = self.fixtures(collector)
+            with patch.object(external, 'open_public', side_effect=self.network(contents)), self.parse_two_rows():
+                self.reject_without_aborting(collector, bad)
+            digest = hashlib.sha256(contents[bad['name']]).hexdigest()
+            records = collector.manifest[bad['name']]['content_rejections']
+            original = records[digest].copy()
+            for record in (None, {**original, 'name': 'other.zip'}, {**original, 'sha256': '0' * 64}):
+                records[digest] = record
+                with self.subTest(record=record), self.assertRaises(ValueError):
+                    collector.status()
+
+
+class ExternalResearchTargetTests(unittest.TestCase):
+    def collector(self, folder):
+        control, store = Mock(), Mock()
+        control.collection_target.return_value = 50_000_000_000
+        store.summary.return_value = {}
+        store.ready.return_value = False
+        module = Mock()
+        module.ResearchControl.return_value = control
+        settings = {'active': True, 'state_path': '4차/work/research_control.json'}
+        with patch.object(external, 'ExternalStore', return_value=store), \
+             patch('importlib.import_module', return_value=module):
+            try:
+                collector = external.Collector(folder, research_settings=settings)
+            except TypeError as error:
+                self.fail('임시 연구 제어의 명시적 주입이 필요합니다: ' + str(error))
+        return collector, control, store
+
+    def run_once(self, collector):
+        stop = external.threading.Event()
+        with patch.object(external, 'STOP', stop), patch.object(external.threading, 'Thread') as thread, \
+             patch.object(stop, 'wait', side_effect=lambda _: stop.set()):
+            thread.return_value.is_alive.return_value = False
+            collector.run()
+
+    def test_temporary_work_default_does_not_load_actual_research_control(self):
+        with tempfile.TemporaryDirectory() as temp, \
+             patch('importlib.import_module', side_effect=AssertionError('실제 연구 상태에 접근했습니다')):
+            collector = ExternalDownloadTests().collector(Path(temp))
+            self.assertIsNone(getattr(collector, 'research_control', None))
+
+    def test_integer_target_and_last_target_on_none_are_written_consistently(self):
+        with tempfile.TemporaryDirectory() as temp:
+            collector, control, store = self.collector(Path(temp))
+            self.assertEqual(collector.collection_target(), 50_000_000_000)
+            control.collection_target.return_value = 52_000_000_000
+            self.assertEqual(collector.collection_target(), 52_000_000_000)
+            state = collector.status()
+            self.assertEqual(state['target_archive_bytes'], 52_000_000_000)
+            self.assertEqual(store.archive_evidence.call_args.args, (52_000_000_000, 0, 0))
+            control.collection_target.return_value = None
+            self.assertIsNone(collector.collection_target())
+            self.assertEqual(collector.status()['target_archive_bytes'], 52_000_000_000)
+            self.assertEqual(store.archive_evidence.call_args.args, (52_000_000_000, 0, 0))
+
+    def test_completed_restart_preserves_last_persisted_integer_target(self):
+        with tempfile.TemporaryDirectory() as temp:
+            folder = Path(temp)
+            (folder / 'status.json').write_text(json.dumps({'target_archive_bytes': 52_000_000_000}), encoding='utf-8')
+            collector, control, store = self.collector(folder)
+            control.collection_target.return_value = None
+            self.assertIsNone(collector.collection_target())
+            self.assertEqual(collector.status()['target_archive_bytes'], 52_000_000_000)
+            self.assertEqual(store.archive_evidence.call_args.args, (52_000_000_000, 0, 0))
+
+    def test_none_prevents_bootstrap_and_bulk_but_keeps_status(self):
+        with tempfile.TemporaryDirectory() as temp:
+            collector, control, _ = self.collector(Path(temp))
+            control.collection_target.return_value = None
+            spec = external.archive_spec('BTCUSDT', 'klines', '2026-02')
+            with patch.object(external, 'bootstrap_plan', return_value=[spec]), \
+                 patch.object(external, 'bulk_plan', return_value=[spec]), \
+                 patch.object(external, 'open_public', side_effect=AssertionError('대기 중 원본을 요청했습니다')):
+                self.run_once(collector)
+            self.assertEqual(collector.manifest, {})
+            state = json.loads((Path(temp) / 'status.json').read_text(encoding='utf-8'))
+            self.assertIsNone(state['error'])
+            self.assertEqual(state['target_archive_bytes'], 50_000_000_000)
+
+    def test_post_ingest_none_stops_remaining_bootstrap_and_bulk(self):
+        with tempfile.TemporaryDirectory() as temp:
+            collector, control, store = self.collector(Path(temp))
+            first = external.archive_spec('BTCUSDT', 'klines', '2026-02')
+            second = external.archive_spec('ETHUSDT', 'klines', '2026-02')
+            content = ExternalQuarantineTests().zip_bytes(first, ExternalParserTests().kline().values.tolist())
+            (collector.archives / first['name']).write_bytes(content)
+            def saved(*_):
+                control.collection_target.return_value = None
+                return 1
+            store.save.side_effect = saved
+            with patch.object(external, 'bootstrap_plan', return_value=[first, second]), \
+                 patch.object(external, 'bulk_plan', return_value=[second]), \
+                 patch.object(external, 'open_public', side_effect=ExternalQuarantineTests().network({first['name']: content})):
+                self.run_once(collector)
+            self.assertEqual(set(collector.manifest), {first['name']})
+            self.assertTrue(collector.manifest[first['name']]['ingested'])
+            self.assertIsNone(collector.status()['error'])
+
+    def test_reached_bytes_with_missing_coverage_does_not_skip_bootstrap(self):
+        with tempfile.TemporaryDirectory() as temp:
+            collector, control, _ = self.collector(Path(temp))
+            spec = external.archive_spec('BTCUSDT', 'klines', '2026-02')
+            visited = []
+            def archive(value):
+                visited.append(value['name'])
+                control.collection_target.return_value = None
+            with patch.object(external, 'bootstrap_plan', return_value=[spec]), \
+                 patch.object(external, 'bulk_plan', return_value=[]), \
+                 patch.object(collector, 'status', return_value={
+                     'ingested_archive_bytes': 50_000_000_000, 'training_ready': False}), \
+                 patch.object(collector, 'archive', side_effect=archive):
+                self.run_once(collector)
+            self.assertEqual(visited, [spec['name']])
+
+    def test_plan_exhaustion_refreshes_new_integer_target(self):
+        with tempfile.TemporaryDirectory() as temp:
+            collector, control, store = self.collector(Path(temp))
+            control.collection_target.return_value = 52_000_000_000
+            with patch.object(external, 'bootstrap_plan', return_value=[]), \
+                 patch.object(external, 'bulk_plan', return_value=[]):
+                self.run_once(collector)
+            self.assertEqual(collector.status()['target_archive_bytes'], 52_000_000_000)
+            self.assertEqual(store.archive_evidence.call_args.args, (52_000_000_000, 0, 0))
+
+    def test_control_error_is_not_hidden_as_none_or_collection_success(self):
+        with tempfile.TemporaryDirectory() as temp:
+            collector, control, _ = self.collector(Path(temp))
+            control.collection_target.side_effect = RuntimeError('연구 상태 읽기 실패')
+            with patch.object(external, 'bootstrap_plan', return_value=[]), \
+                 patch.object(external, 'bulk_plan', return_value=[]):
+                self.run_once(collector)
+            self.assertIn('연구 상태 읽기 실패', collector.status()['error'])
+            self.assertFalse(collector.status()['training_ready'])
+
+    def test_invalid_control_targets_are_rejected(self):
+        with tempfile.TemporaryDirectory() as temp:
+            collector, control, _ = self.collector(Path(temp))
+            for target in (True, 0, -1, '50000000000', 50.5):
+                control.collection_target.return_value = target
+                with self.subTest(target=target), self.assertRaises(ValueError):
+                    collector.collection_target()
+
+
+@unittest.skipUnless(sys.platform == 'linux', '실제 파일 진입점은 Linux 임시 프로젝트에서 검사합니다')
+class ExternalEntrypointTests(unittest.TestCase):
+    def test_absolute_script_entrypoint_loads_real_sibling_control_without_pythonpath(self):
+        with tempfile.TemporaryDirectory() as temp:
+            folder = Path(temp)
+            project, outside = folder / 'project', folder / 'outside'
+            (project / '3차').mkdir(parents=True)
+            (project / '4차').mkdir()
+            outside.mkdir()
+            script = project / '3차/external_data.py'
+            shutil.copyfile(external.ROOT / 'external_data.py', script)
+            shutil.copyfile(external.ROOT.parent / '4차/research_control.py', project / '4차/research_control.py')
+            (project / 'engine-config.json').write_text(json.dumps({
+                'research': {'active': True, 'state_path': '4차/work/research_control.json'}}), encoding='utf-8')
+            env = os.environ.copy()
+            env.pop('PYTHONPATH', None)
+            env['PYTHONNOUSERSITE'] = '1'
+            argv = [sys.executable, '-B', '-X', 'utf8', str(script), 'status']
+            # 운영 경로 대신 동일 소스의 격리 사본으로 실제 CLI 초기화를 검사합니다.
+            result = subprocess.run(argv, cwd=outside, env=env, capture_output=True, timeout=15, check=False)
+            self.assertEqual(result.returncode, 0,
+                             f'argv={argv!r}; cwd={outside}; stderr={result.stderr.decode("utf-8")}')
+            state = json.loads(result.stdout.decode('utf-8'))
+            self.assertEqual(state['target_archive_bytes'], 50_000_000_000)
+            self.assertFalse(state['training_ready'])
+            self.assertIsNone(state['error'])
+            self.assertTrue((project / '3차/work/data/external.sqlite3').is_file())
+
+
 class ExternalProcessBoundaryTests(unittest.TestCase):
     def test_only_exact_collector_script_and_cwd_can_match(self):
         manage = importlib.import_module('3차.manage')

```

## tests/test_research_control.py

이전: 없음  
현재: 33840 bytes / SHA-256 03b9139aa48bd716420fa80a1bdb4db6330ce6fcd9d516f1fdb314add18c1682

```diff
--- 이전/tests/test_research_control.py
+++ 현재/tests/test_research_control.py
@@ -0,0 +1,604 @@
+"""명시 요청 연구의 내구성·실패 경계와 엔진 공통 진입점을 검사합니다."""
+import importlib
+import json
+from pathlib import Path
+import tempfile
+import threading
+import unittest
+from unittest.mock import Mock, patch
+
+import engine
+
+
+class ResearchControlTests(unittest.TestCase):
+    def setUp(self):
+        self.assertTrue((engine.ROOT / "4차/research_control.py").is_file(), "연구 제어 모듈이 없습니다")
+        self.module = importlib.import_module("4차.research_control")
+        self.temp = tempfile.TemporaryDirectory()
+        self.addCleanup(self.temp.cleanup)
+        self.root = Path(self.temp.name)
+        self.settings = {"active": True, "state_path": "4차/work/research_control.json"}
+        self.control = self.module.ResearchControl(self.root, self.settings)
+        self.fit_calls = 0
+        self.last_report = None
+        self.promote = False
+
+    def active_config(self):
+        config = engine.load_config()
+        config["research"] = self.settings
+        config["external"].update(enabled="auto", wait_for_dataset=True)
+        config["training"]["candidate_depths"] = [3]
+        for key, folder in (("data_dir", "data"), ("models_dir", "models"), ("reports_dir", "reports")):
+            config[key] = str(self.root / "work" / folder)
+        return config
+
+    def completed_cycle(self, config, **kwargs):
+        return ResearchRecoveryTests.cycle(self, config, **kwargs)
+
+    def report(self, target=50_000_000_000, run="controlled-run"):
+        return {"run": run, "config": {"training": {"candidate_depths": [3]}},
+                "training": {"status": "완료", "completed_candidates": 5, "total_candidates": 5},
+                "markets": {m: {"gpu": [{"actual_device": "cuda:0"}]} for m in engine.MARKETS},
+                "rounds": [{"market": m, "candidate": 1, "actual_device": "cuda:0"} for m in engine.MARKETS],
+                "external_resolution": {"training_ready": True, "target_archive_bytes": target,
+                                        "ingested_archive_bytes": target + 3,
+                                        "verified_archive_bytes": target + 3},
+                "promotion": {"promote": False}}
+
+    def finish_initial(self):
+        with patch.object(engine, "ROOT", self.root), patch.object(engine, "wait_for_training_data"), \
+             patch.object(engine, "_train_cycle", side_effect=self.completed_cycle):
+            engine.train_cycle(self.active_config(), live=True)
+
+    def test_bootstrap_target_and_restart_are_exact(self):
+        self.assertEqual(self.control.collection_target(), 50_000_000_000)
+        again = self.module.ResearchControl(self.root, self.settings)
+        self.assertEqual(again.pending_batch(), self.control.pending_batch())
+        self.assertEqual(len(again.snapshot()["batches"]), 1)
+
+    def test_initial_request_rejected_before_reading_baseline(self):
+        reader = Mock(return_value=51_000_000_000)
+        with self.assertRaisesRegex(ValueError, "초기"):
+            self.control.request_increment(reader)
+        reader.assert_not_called()
+
+    def test_completed_unpromoted_batch_stays_idle_across_restart(self):
+        self.finish_initial()
+        # 예전 completed 이력은 새 시도 파일이 없어도 다시 적합하지 않습니다.
+        self.control.attempt_path(self.control.snapshot()["batches"][0]).unlink()
+        again = self.module.ResearchControl(self.root, self.settings)
+        self.assertIsNone(again.collection_target())
+        self.assertIsNone(again.pending_batch())
+        self.assertFalse(again.snapshot()["batches"][0]["promoted"])
+
+    def test_fresh_actual_baseline_excludes_last_archive_overshoot_and_is_frozen(self):
+        self.finish_initial()
+        reader = Mock(return_value=55_123_456_789)
+        batch = self.control.request_increment(reader)
+        self.assertEqual(batch["baseline_bytes"], 55_123_456_789)
+        self.assertEqual(batch["target_bytes"], 57_123_456_789)
+        reader.assert_called_once_with()
+        self.assertEqual(self.control.collection_target(), 57_123_456_789)
+        with self.assertRaisesRegex(ValueError, "활성"):
+            self.control.request_increment(reader)
+        reader.assert_called_once_with()
+
+    def test_invalid_or_failed_baseline_does_not_append_request(self):
+        self.finish_initial()
+        before = self.control.snapshot()
+        for invalid in (True, -1, 49_999_999_999, 50_000_000_000.0):
+            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
+                self.control.request_increment(lambda: invalid)
+        with self.assertRaises(OSError):
+            self.control.request_increment(Mock(side_effect=OSError("통제된 읽기 실패")))
+        self.assertEqual(before, self.control.snapshot())
+
+    def test_failure_and_interrupted_claim_retry_same_batch_not_completed(self):
+        batch = self.control.pending_batch()
+        first = self.control.claim_training(batch["id"])
+        self.control.fail_training(batch["id"], first["claim"], "통제된 학습 실패")
+        self.assertEqual(self.control.pending_batch()["id"], batch["id"])
+        self.assertEqual(self.control.pending_batch()["last_error"], "통제된 학습 실패")
+        second = self.control.claim_training(batch["id"])
+        again = self.module.ResearchControl(self.root, self.settings)
+        with again.training_lock():
+            with self.assertRaises(ValueError):
+                again.claim_training(batch["id"])
+            again.fail_training(batch["id"], second["claim"], "통제된 미완료 시도 종료")
+            third = again.claim_training(batch["id"])
+        self.assertEqual(third["attempts"], 3)
+        self.assertNotEqual(third["claim"], second["claim"])
+        with self.assertRaises(ValueError):
+            self.control.complete_training(batch["id"], second["claim"], self.report())
+        self.assertIsNotNone(self.control.collection_target())
+
+    def test_bad_result_cannot_complete_batch(self):
+        batch = self.control.pending_batch()
+        claim = self.control.claim_training(batch["id"])
+        run = "20261102T000000-01234567"
+        record = self.control.start_attempt(claim, run, "work/reports", "work/models", None, None)
+        invalids = []
+        for key, value in (("training_ready", False), ("ingested_archive_bytes", 49_999_999_999)):
+            report = self.report()
+            report["external_resolution"][key] = value
+            invalids.append(report)
+        report = self.report()
+        report["markets"].pop("KRW-DOGE")
+        invalids.append(report)
+        report = self.report()
+        report["training"]["completed_candidates"] = 4
+        invalids.append(report)
+        report = self.report()
+        report["rounds"][0]["actual_device"] = "cpu"
+        invalids.append(report)
+        for report in invalids:
+            report["run"] = run
+            report["research_batch"] = self.module.attempt_binding(record)
+            with self.subTest(report=report), self.assertRaises(ValueError):
+                self.control.store_result(batch["id"], claim["claim"], report, "0" * 64, {"model.ubj": "0" * 64})
+        self.assertEqual(self.control.pending_batch()["status"], "training")
+
+    def test_new_claim_cannot_complete_without_durable_bound_result(self):
+        batch = self.control.pending_batch()
+        claim = self.control.claim_training(batch["id"])
+        with self.assertRaises(ValueError):
+            self.control.complete_training(batch["id"], claim["claim"], self.report())
+
+    def test_corrupt_schema_and_duplicate_json_keys_fail_closed_without_rewrite(self):
+        self.control.collection_target()
+        path = self.root / self.settings["state_path"]
+        for raw in ('{"schema_version":99,"batches":[]}', '{"schema_version":1,"schema_version":1,"batches":[]}', '{broken'):
+            with self.subTest(raw=raw):
+                path.write_text(raw, encoding="utf-8")
+                with self.assertRaises(ValueError):
+                    self.control.collection_target()
+                self.assertEqual(path.read_text(encoding="utf-8"), raw)
+
+    def test_config_default_disabled_and_paths_confined(self):
+        self.assertFalse(self.module.validate_config({}, self.root)["active"])
+        for invalid in ({"active": 1}, {"unknown": True}, {"state_path": "../escape.json"},
+                        {"state_path": "work/paper/account.json"}):
+            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
+                self.module.validate_config(invalid, self.root)
+        disabled = self.module.ResearchControl(self.root / "disabled", {})
+        self.assertIsNone(disabled.collection_target())
+        self.assertFalse((self.root / "disabled/4차/work").exists())
+
+    def test_training_lock_excludes_another_owner(self):
+        another = self.module.ResearchControl(self.root, self.settings)
+        with self.control.training_lock():
+            with self.assertRaises((OSError, RuntimeError)):
+                with another.training_lock():
+                    self.fail("동시 학습 잠금이 허용됐습니다")
+
+    def test_atomic_write_failure_preserves_previous_state(self):
+        self.finish_initial()
+        before = self.control.snapshot()
+        with patch.object(self.module.os, "replace", side_effect=OSError("통제된 저장 실패")):
+            with self.assertRaises(OSError):
+                self.control.request_increment(lambda: 52_000_000_000)
+        self.assertEqual(self.control.snapshot(), before)
+
+    def test_two_concurrent_requests_accept_exactly_one_fresh_baseline(self):
+        self.finish_initial()
+        barrier = threading.Barrier(2)
+        results = []
+        reader = Mock(return_value=55_000_000_000)
+
+        def request():
+            barrier.wait(timeout=3)
+            try:
+                self.control.request_increment(reader)
+                results.append("accepted")
+            except ValueError:
+                results.append("rejected")
+
+        threads = [threading.Thread(target=request) for _ in range(2)]
+        for thread in threads:
+            thread.start()
+        for thread in threads:
+            thread.join(timeout=5)
+            self.assertFalse(thread.is_alive())
+        self.assertCountEqual(results, ["accepted", "rejected"])
+        reader.assert_called_once_with()
+
+    def test_request_cli_baseline_uses_current_database_not_previous_status_bytes(self):
+        self.finish_initial()
+        config = engine.load_config()
+        config["research"] = self.settings
+        config["external"].update(enabled="auto", wait_for_dataset=True)
+        external = importlib.import_module("3차.external_features")
+        db = Mock()
+        db.execute.return_value = [("ingested_archive_bytes", "57123456789"),
+                                   ("verified_archive_bytes", "57123456789")]
+        from contextlib import nullcontext
+        with patch.object(engine, "ROOT", self.root), \
+             patch.object(external, "resolve_for_training", return_value=(config["external"], {"ingested_archive_bytes": 50_000_000_000})), \
+             patch.object(external, "_database", return_value=nullcontext(db)):
+            batch = engine.request_research_increment(config)
+        self.assertEqual(batch["baseline_bytes"], 57_123_456_789)
+        self.assertEqual(batch["target_bytes"], 59_123_456_789)
+
+    def test_once_trainer_waits_twelve_seconds_before_fit_and_writes_owner_heartbeat(self):
+        config = self.active_config()
+        external = importlib.import_module("3차.external_features")
+        stop = Mock()
+        stop.is_set.return_value = False
+        stop.wait.return_value = False
+        train = Mock(side_effect=self.completed_cycle)
+        evidence = {"reason": "통제된 자료 미달", "checked_at": engine.iso(None),
+                    "target_archive_bytes": 50_000_000_000, "ingested_archive_bytes": 7_000_000_000}
+
+        def readiness(*args):
+            train.assert_not_called()
+            if not stop.wait.called:
+                raise external.DatasetNotReady(evidence)
+            return config["external"], {"training_ready": True}
+
+        with patch.object(engine, "ROOT", self.root), patch.object(engine, "load_config", return_value=config), \
+             patch.object(engine, "process_lock"), patch.object(engine, "_train_cycle", train), \
+             patch.object(engine, "atomic_json", wraps=engine.atomic_json) as writes, \
+             patch.object(engine.threading, "Event", return_value=stop), \
+             patch.object(external, "resolve_for_training", side_effect=readiness), patch("builtins.print"):
+            self.assertEqual(engine.main(["trainer", "--once"]), 0)
+        train.assert_called_once()
+        stop.wait.assert_called_once_with(12)
+        heartbeat = next(call.args[1] for call in writes.call_args_list
+                         if Path(call.args[0]).name == "training_progress.json"
+                         and call.args[1].get("status") == "waiting_data")
+        self.assertTrue(heartbeat["research_active"])
+        self.assertEqual(heartbeat["pid"], engine.os.getpid())
+        self.assertEqual(heartbeat["status"], "waiting_data")
+        self.assertIsNone(heartbeat["completed_candidates"])
+        self.assertIsNone(engine._research_progress.get())
+
+    def test_common_train_entry_waits_then_runs_all_candidates_once_and_goes_idle(self):
+        config = self.active_config()
+        with patch.object(engine, "ROOT", self.root), patch.object(engine, "_train_cycle", side_effect=self.completed_cycle) as train, \
+             patch.object(engine, "wait_for_training_data") as wait:
+            report = engine.train_cycle(config, live=True)
+        self.assertEqual(report["run"], self.control.snapshot()["batches"][0]["run"])
+        train.assert_called_once()
+        wait.assert_called_once()
+        self.assertEqual(train.call_args.args[0]["external"]["target_bytes"], 50_000_000_000)
+        self.assertIsNone(self.control.collection_target())
+
+    def test_train_failure_keeps_batch_and_claim_error(self):
+        config = self.active_config()
+        with patch.object(engine, "ROOT", self.root), patch.object(engine, "wait_for_training_data"), \
+             patch.object(engine, "_train_cycle", side_effect=RuntimeError("통제된 적합 실패")):
+            with self.assertRaisesRegex(RuntimeError, "적합 실패"):
+                engine.train_cycle(config, live=True)
+        self.assertEqual(self.control.pending_batch()["status"], "collecting")
+        self.assertIn("적합 실패", self.control.pending_batch()["last_error"])
+
+    def test_idle_has_heartbeat_no_fit_no_fake_candidates_and_no_busy_loop(self):
+        self.finish_initial()
+        config = engine.load_config()
+        config["research"] = self.settings
+        stop = Mock()
+        stop.is_set.return_value = False
+        stop.wait.return_value = True
+        records = []
+        with patch.object(engine, "ROOT", self.root), patch.object(engine, "_train_cycle") as train, \
+             patch.object(engine, "progress", side_effect=lambda **fields: records.append(fields)):
+            with self.assertRaises(KeyboardInterrupt):
+                engine.wait_for_research_batch(self.control, stop)
+        train.assert_not_called()
+        stop.wait.assert_called_once_with(12)
+        self.assertEqual(records[0]["status"], "waiting_request")
+        self.assertEqual(records[0]["stage"], "idle")
+        for key in ("run", "progress", "completed_candidates", "total_candidates", "candidate"):
+            self.assertIsNone(records[0][key])
+
+    def test_increment_target_is_used_for_training_gate(self):
+        self.finish_initial()
+        batch = self.control.request_increment(lambda: 56_000_000_000)
+        config = self.active_config()
+        with patch.object(engine, "ROOT", self.root), patch.object(engine, "wait_for_training_data") as wait, \
+             patch.object(engine, "_train_cycle", side_effect=self.completed_cycle) as train:
+            engine.train_cycle(config, live=True)
+        self.assertEqual(wait.call_args.args[0]["external"]["target_bytes"], 58_000_000_000)
+        self.assertEqual(train.call_args.args[0]["external"]["target_bytes"], 58_000_000_000)
+        self.assertIsNone(self.control.collection_target())
+
+    def test_request_cli_dispatch_and_experiment_share_public_train_entry(self):
+        self.assertEqual(engine.build_parser().parse_args(["research-request"]).command, "research-request")
+        config = engine.load_config()
+        config["research"] = self.settings
+        with patch.object(engine, "load_config", return_value=config), patch.object(engine, "request_research_increment", return_value={"id": "x"}) as request, \
+             patch.object(engine, "process_lock"), patch.object(engine, "train_cycle", return_value={"run": "x", "locked_test": {}}) as train, \
+             patch("builtins.print"):
+            self.assertEqual(engine.main(["research-request"]), 0)
+            self.assertEqual(engine.main(["experiment"]), 0)
+        request.assert_called_once_with(config)
+        train.assert_called_once_with(config, final=False)
+
+    def test_active_config_cannot_disable_external_gate(self):
+        path = self.root / "config.json"
+        path.write_text(json.dumps({"research": self.settings}), encoding="utf-8")
+        with self.assertRaises(ValueError):
+            engine.load_config(path)
+
+    def test_paper_dispatch_never_enters_research_gate(self):
+        config = engine.load_config()
+        config["research"] = self.settings
+        with patch.object(engine, "load_config", return_value=config), patch.object(engine, "process_lock"), \
+             patch.object(engine, "paper_cycle", return_value={"status": "통제"}) as paper, \
+             patch.object(engine, "train_cycle") as train, patch("builtins.print"):
+            self.assertEqual(engine.main(["paper", "--once"]), 0)
+        paper.assert_called_once_with(config)
+        train.assert_not_called()
+
+
+class ResearchRecoveryTests(unittest.TestCase):
+    def setUp(self):
+        self.temp = tempfile.TemporaryDirectory()
+        self.addCleanup(self.temp.cleanup)
+        self.root = Path(self.temp.name)
+        self.module = importlib.import_module("4차.research_control")
+        self.settings = {"active": True, "state_path": "4차/work/research_control.json"}
+        self.control = self.module.ResearchControl(self.root, self.settings)
+        self.config = engine.load_config()
+        self.config["research"] = self.settings
+        self.config["external"].update(enabled="auto", wait_for_dataset=True)
+        self.config["training"]["candidate_depths"] = [3]
+        for key, folder in (("data_dir", "data"), ("models_dir", "models"), ("reports_dir", "reports")):
+            self.config[key] = str(self.root / "work" / folder)
+        self.fit_calls = 0
+        self.last_report = None
+
+    def cycle(self, config, **kwargs):
+        self.fit_calls += 1
+        batch = self.control.pending_batch()
+        attempt = getattr(engine, "_research_attempt", None)
+        record = attempt.get() if attempt is not None else None
+        run = record["run"] if record else "20261102T000000-01234567"
+        binding = {"schema_version": 1, "batch_id": batch["id"], "claim": batch["claim"],
+                   "attempt": batch["attempts"], "target_bytes": batch["target_bytes"], "run": run}
+        directory = Path(config["models_dir"]) / "versions" / run
+        directory.mkdir(parents=True, exist_ok=True)
+        packages, hashes = {}, {}
+        for market in engine.MARKETS:
+            name = market + "-0.ubj"
+            (directory / name).write_bytes(("통제된 모델 " + market).encode("utf-8"))
+            hashes[name] = engine.hashlib.sha256((directory / name).read_bytes()).hexdigest()
+            packages[market] = {"models": [name], "gpu": [{"actual_device": "cuda:0"}]}
+        manifest = {"run": run, "schema_version": 2, "execution_policy": engine.EXECUTION_POLICY,
+                    "paper_only": True, "config": config, "packages": packages,
+                    "source_hash": "controlled-source", "research_batch": binding}
+        engine.atomic_json(directory / "manifest.json", manifest)
+        report = {"run": run, "config": config, "source_hash": "controlled-source",
+                  "research_batch": binding,
+                  "training": {"status": "완료", "completed_candidates": 5, "total_candidates": 5,
+                               "started_at": engine.iso(None), "duration_seconds": 1, "rows": 100, "features": 1},
+                  "markets": packages,
+                  "rounds": [{"market": m, "candidate": 1, "actual_device": "cuda:0"} for m in engine.MARKETS],
+                  "external_resolution": {"training_ready": True, "target_archive_bytes": batch["target_bytes"],
+                                          "ingested_archive_bytes": batch["target_bytes"],
+                                          "verified_archive_bytes": batch["target_bytes"]},
+                  "promotion": {"promote": getattr(self, "promote", True)}, "locked_test": {}}
+        self.last_report = report
+        if record is not None:
+            record["result_started"] = True
+            self.control.store_result(batch["id"], batch["claim"], report, engine.stable_hash(manifest), hashes)
+        engine.atomic_json(Path(config["reports_dir"]) / "experiments" / (run + ".json"), report)
+        engine.atomic_json(Path(config["reports_dir"]) / "training.json", report)
+        if report["promotion"]["promote"]:
+            engine.atomic_json(Path(config["models_dir"]) / "current.json",
+                               {"run": run, "manifest_hash": engine.stable_hash(manifest)})
+        engine.progress(status="complete", stage="complete", run=run)
+        return report
+
+    def run_cycle(self):
+        def pending(control):
+            batch = control.pending_batch()
+            if batch is None:
+                raise RuntimeError("검사 중 예상하지 않은 요청 대기")
+            return batch
+
+        with patch.object(engine, "ROOT", self.root), patch.object(engine, "wait_for_training_data"), \
+             patch.object(engine, "wait_for_research_batch", side_effect=pending), \
+             patch.object(engine, "_train_cycle", side_effect=self.cycle):
+            return engine.train_cycle(self.config, live=True)
+
+    def fail_completion(self, error=None):
+        with patch.object(self.module.ResearchControl, "complete_training", side_effect=error or OSError("통제된 완료 상태 저장 실패")):
+            with self.assertRaises(BaseException):
+                self.run_cycle()
+
+    def test_recovery_after_completion_state_write_failure_never_fits_again(self):
+        self.fail_completion()
+        self.run_cycle()
+        self.assertEqual(self.fit_calls, 1)
+        self.assertIsNone(self.control.collection_target())
+
+    def test_recovery_preserves_claim_on_completion_crash(self):
+        self.fail_completion(SystemExit("통제된 완료 직전 중단"))
+        batch = self.control.pending_batch()
+        self.assertEqual(batch["status"], "training")
+        self.assertIsNotNone(batch["claim"])
+        self.run_cycle()
+        self.assertEqual(self.fit_calls, 1)
+
+    def test_recovery_finishes_pointer_report_or_progress_failure_without_fit(self):
+        original_write = engine.atomic_json
+        for name in ("training.json", "current.json", "training_progress.json"):
+            with self.subTest(name=name):
+                # 각 장애는 서로 다른 임시 프로젝트에서 같은 초기 batch에 주입합니다.
+                with tempfile.TemporaryDirectory() as temp:
+                    self.root = Path(temp)
+                    self.control = self.module.ResearchControl(self.root, self.settings)
+                    for key, folder in (("data_dir", "data"), ("models_dir", "models"), ("reports_dir", "reports")):
+                        self.config[key] = str(self.root / "work" / folder)
+                    self.fit_calls = 0
+
+                    def write(path, payload):
+                        if Path(path).name == name:
+                            raise OSError("통제된 완료 후 파일 저장 실패")
+                        return original_write(path, payload)
+
+                    with patch.object(engine, "atomic_json", side_effect=write):
+                        with self.assertRaises(OSError):
+                            self.run_cycle()
+                    self.run_cycle()
+                    self.assertEqual(self.fit_calls, 1)
+                    pointer = json.loads((Path(self.config["models_dir"]) / "current.json").read_text(encoding="utf-8"))
+                    self.assertEqual(pointer["run"], self.last_report["run"])
+                    self.assertIsNone(self.control.pending_batch())
+
+    def test_recovery_rejects_changed_incomplete_or_foreign_report_without_fit(self):
+        self.fail_completion()
+        path = Path(self.config["reports_dir"]) / "experiments" / (self.last_report["run"] + ".json")
+        original = path.read_text(encoding="utf-8")
+        for field in ("status", "claim", "batch", "content"):
+            with self.subTest(field=field):
+                report = json.loads(original)
+                if field == "status":
+                    report["training"]["status"] = "학습 중"
+                elif field == "claim":
+                    report["research_batch"]["claim"] = "0" * 32
+                elif field == "batch":
+                    report["research_batch"]["batch_id"] = "다른 요청"
+                else:
+                    report["training"]["rows"] = 101
+                path.write_text(json.dumps(report), encoding="utf-8")
+                with self.assertRaises((ValueError, RuntimeError)):
+                    self.run_cycle()
+                self.assertEqual(self.fit_calls, 1)
+                self.assertIsNotNone(self.control.pending_batch())
+                path.write_text(original, encoding="utf-8")
+
+    def test_recovery_without_known_attempt_fails_closed(self):
+        batch = self.control.pending_batch()
+        self.control.claim_training(batch["id"])
+        with self.assertRaises((ValueError, RuntimeError)):
+            self.run_cycle()
+        self.assertEqual(self.fit_calls, 0)
+
+    def test_collecting_state_recovers_bound_result_without_new_claim_or_fit(self):
+        self.fail_completion()
+        state = self.control.snapshot()
+        state["batches"][-1].update(status="collecting", claim=None)
+        (self.root / self.settings["state_path"]).write_text(json.dumps(state), encoding="utf-8")
+        self.run_cycle()
+        self.assertEqual(self.fit_calls, 1)
+        self.assertEqual(self.control.snapshot()["batches"][-1]["attempts"], 1)
+
+    def test_durable_result_restores_missing_reports_and_pointer(self):
+        self.fail_completion()
+        reports, models = Path(self.config["reports_dir"]), Path(self.config["models_dir"])
+        for path in (reports / "experiments" / (self.last_report["run"] + ".json"),
+                     reports / "training.json", models / "current.json"):
+            path.unlink()
+        self.run_cycle()
+        self.assertEqual(self.fit_calls, 1)
+        self.assertEqual(json.loads((reports / "training.json").read_text(encoding="utf-8")), self.last_report)
+
+    def test_changed_models_manifest_or_attempt_record_are_rejected_without_fit(self):
+        self.fail_completion()
+        run = self.last_report["run"]
+        directory = Path(self.config["models_dir"]) / "versions" / run
+        paths = [directory / "KRW-BTC-0.ubj", directory / "manifest.json",
+                 self.control.attempt_path(self.control.pending_batch())]
+        for path in paths:
+            with self.subTest(path=path.name):
+                original = path.read_bytes()
+                if path.suffix == ".ubj":
+                    path.write_bytes(b"changed-model")
+                else:
+                    data = json.loads(original)
+                    if path.name == "manifest.json":
+                        data["source_hash"] = "changed-source"
+                    else:
+                        data["report"]["training"]["rows"] += 1
+                    path.write_text(json.dumps(data), encoding="utf-8")
+                with self.assertRaises((ValueError, RuntimeError)):
+                    self.run_cycle()
+                self.assertEqual(self.fit_calls, 1)
+                path.write_bytes(original)
+
+    def test_result_write_failure_preserves_claim_and_blocks_ambiguous_replay(self):
+        with patch.object(self.module.ResearchControl, "store_result", side_effect=OSError("통제된 완료 결과 저장 실패")):
+            with self.assertRaises(OSError):
+                self.run_cycle()
+        batch = self.control.pending_batch()
+        self.assertEqual(batch["status"], "training")
+        self.assertIsNotNone(batch["claim"])
+        with self.assertRaisesRegex(ValueError, "미확인"):
+            self.run_cycle()
+        self.assertEqual(self.fit_calls, 1)
+
+    def test_model_failure_is_recorded_separately_and_same_batch_can_retry(self):
+        with patch.object(engine, "ROOT", self.root), patch.object(engine, "wait_for_training_data"), \
+             patch.object(engine, "_train_cycle", side_effect=RuntimeError("통제된 모델 실패")):
+            with self.assertRaises(RuntimeError):
+                engine.train_cycle(self.config, live=True)
+        batch = self.control.pending_batch()
+        self.assertEqual(batch["status"], "collecting")
+        self.assertEqual(self.control.read_attempt(batch)["phase"], "model_failed")
+        self.run_cycle()
+        self.assertEqual(self.fit_calls, 1)
+        self.assertEqual(self.control.snapshot()["batches"][-1]["attempts"], 2)
+
+    def test_summary_output_failure_after_completion_cannot_replay_fit(self):
+        with patch.object(engine, "ROOT", self.root), patch.object(engine, "load_config", return_value=self.config), \
+             patch.object(engine, "process_lock"), patch.object(engine, "wait_for_training_data"), \
+             patch.object(engine, "_train_cycle", side_effect=self.cycle), \
+             patch("builtins.print", side_effect=OSError("통제된 요약 출력 실패")):
+            with self.assertRaises(OSError):
+                engine.main(["trainer", "--once"])
+        self.assertIsNone(self.control.pending_batch())
+        with self.assertRaisesRegex(RuntimeError, "요청 대기"):
+            self.run_cycle()
+        self.assertEqual(self.fit_calls, 1)
+
+    def test_actual_train_cycle_storage_branch_recovers_after_pointer_write_failure(self):
+        self.config["training"].update(holdout_start=engine.PROTECTED_HOLDOUT[0],
+                                       holdout_end=engine.PROTECTED_HOLDOUT[1], holdout_days=1)
+        from contextlib import nullcontext
+        import numpy as np
+        import pandas as pd
+
+        index = pd.date_range("2026-09-01", periods=3, freq="5min", tz="UTC")
+        frame = pd.DataFrame({"x": [1., 2., 3.], "label": [0, 1, 2],
+                              "label_end": index + pd.Timedelta(minutes=5)}, index=index)
+        split = {key: frame for key in ("train", "tune", "calibration", "rule", "validation")}
+        evidence = {"training_ready": True, "target_archive_bytes": 50_000_000_000,
+                    "verified_archive_bytes": 50_000_000_000, "ingested_archive_bytes": 50_000_000_000}
+        external = importlib.import_module("3차.external_features")
+        audit = Mock()
+        audit.consumed_periods.return_value = []
+        model = Mock()
+        model.save_model.side_effect = lambda path: Path(path).write_bytes(b"controlled-gpu-model")
+        original_write = engine.atomic_json
+
+        def fault(path, payload):
+            if Path(path).name == "current.json":
+                raise OSError("통제된 실제 저장 분기 포인터 실패")
+            return original_write(path, payload)
+
+        with patch.object(engine, "ROOT", self.root), patch.object(engine, "wait_for_training_data"), \
+             patch.object(external, "resolve_for_training", return_value=(self.config["external"], evidence)), \
+             patch.object(engine, "ExperimentAudit", return_value=audit), patch.object(engine, "CandleStore"), \
+             patch.object(engine, "market_feature_loader", return_value=nullcontext(lambda market: (frame, {}))), \
+             patch.object(engine, "label_frame", return_value=frame), patch.object(engine, "feature_columns", return_value=["x"]), \
+             patch.object(engine, "research_splits", return_value=split), \
+             patch.object(engine, "fit_gpu", return_value=(model, {"best_iteration": 0, "actual_device": "cuda:0"})) as fit, \
+             patch.object(engine, "predict_booster", return_value=np.tile([.1, .8, .1], (3, 1))), \
+             patch.object(engine, "tune_rule", side_effect=lambda *a, **k: {"confidence": 1.01, "edge": 1.01, "eligible": False, "trials": []}), \
+             patch.object(engine, "backtest_portfolio", return_value={"net_return": 0}), \
+             patch.object(engine, "assess_promotion", return_value={"promote": True, "eligibility": {"eligible": True}}):
+            with patch.object(engine, "atomic_json", side_effect=fault):
+                with self.assertRaises(OSError):
+                    engine.train_cycle(self.config, live=True)
+            self.assertEqual(fit.call_count, 5)
+            self.assertEqual(self.control.pending_batch()["status"], "training")
+            report = engine.train_cycle(self.config, live=True)
+            self.assertEqual(fit.call_count, 5)
+        self.assertEqual(report["training"]["completed_candidates"], 5)
+        self.assertIsNone(self.control.pending_batch())
+
+
+if __name__ == "__main__":
+    unittest.main()

```

## tests/test_research_supervisor.py

이전: 없음  
현재: 3740 bytes / SHA-256 a17e1dbcf8f2cca5cb960fddc2f207b64d50e3a2fccda78a664248de72a38287

```diff
--- 이전/tests/test_research_supervisor.py
+++ 현재/tests/test_research_supervisor.py
@@ -0,0 +1,67 @@
+"""대기 하트비트는 제외하고 실제 학습 시간 제한은 유지하는지 검사합니다."""
+import datetime as dt
+from pathlib import Path
+import tempfile
+import unittest
+from unittest.mock import patch
+
+import supervisor
+
+
+class ResearchTimeoutTests(unittest.TestCase):
+    def setUp(self):
+        self.assertTrue(hasattr(supervisor, "TrainingBudget"), "대기와 학습을 구분하는 시간 예산이 없습니다")
+        self.started = dt.datetime(2026, 11, 2, tzinfo=dt.timezone.utc)
+
+    def heartbeat(self, now, status="waiting_request", stage="idle", pid=17):
+        return {"updated_at": now.isoformat(), "pid": pid, "research_active": True,
+                "status": status, "stage": stage, "run": None}
+
+    def test_twelve_second_wait_heartbeats_survive_more_than_six_hours(self):
+        budget = supervisor.TrainingBudget(21600, 0, self.started, 17)
+        for second in range(12, 21600 + 120, 12):
+            now = self.started + dt.timedelta(seconds=second)
+            self.assertFalse(budget.expired(second, now, self.heartbeat(now)))
+        self.assertEqual(budget.active_seconds, 0)
+
+    def test_fit_and_evaluation_progress_never_reset_active_budget(self):
+        budget = supervisor.TrainingBudget(21600, 0, self.started, 17)
+        self.assertFalse(budget.expired(10000, self.started + dt.timedelta(seconds=10000),
+                                      self.heartbeat(self.started + dt.timedelta(seconds=10000), "running", "fit")))
+        self.assertTrue(budget.expired(21600, self.started + dt.timedelta(seconds=21600),
+                                     self.heartbeat(self.started + dt.timedelta(seconds=21600), "running", "calibration")))
+
+    def test_wait_exclusion_does_not_reset_accumulated_fit_time(self):
+        budget = supervisor.TrainingBudget(20, 0, self.started, 17)
+        self.assertFalse(budget.expired(8, self.started + dt.timedelta(seconds=8), {}))
+        now = self.started + dt.timedelta(seconds=100)
+        self.assertFalse(budget.expired(100, now, self.heartbeat(now, "waiting_data", "download")))
+        self.assertTrue(budget.expired(112, self.started + dt.timedelta(seconds=112), {}))
+
+    def test_stale_future_foreign_pid_and_wrong_stage_do_not_exempt_timeout(self):
+        now = self.started + dt.timedelta(seconds=100)
+        invalids = [self.heartbeat(self.started), self.heartbeat(now + dt.timedelta(seconds=1)),
+                    self.heartbeat(now, pid=18), self.heartbeat(now, stage="fit"),
+                    {**self.heartbeat(now), "research_active": False}, {}, None]
+        for record in invalids:
+            with self.subTest(record=record):
+                budget = supervisor.TrainingBudget(20, 0, self.started, 17)
+                self.assertTrue(budget.expired(100, now, record))
+
+    def test_wait_before_child_launch_cannot_exempt_timeout(self):
+        budget = supervisor.TrainingBudget(20, 0, self.started, 17)
+        self.assertTrue(budget.expired(20, self.started + dt.timedelta(seconds=20),
+                                     self.heartbeat(self.started - dt.timedelta(seconds=1))))
+
+    def test_only_trainer_reads_heartbeat_and_missing_file_is_not_wait(self):
+        with tempfile.TemporaryDirectory() as temp:
+            work = Path(temp)
+            self.assertEqual(supervisor.training_heartbeat("trainer", work), {})
+            with patch.object(supervisor.Path, "read_text", return_value='{"status":"waiting_request"}') as read:
+                self.assertEqual(supervisor.training_heartbeat("paper", work), {})
+                read.assert_not_called()
+                self.assertEqual(supervisor.training_heartbeat("trainer", work)["status"], "waiting_request")
+
+
+if __name__ == "__main__":
+    unittest.main()

```

## tests/test_summary_throughput.py

이전: 없음  
현재: 2453 bytes / SHA-256 c1b939cd2b2b6d845f047e993ad405425a126cd3ec7521f2200b2a735bed06bb

```diff
--- 이전/tests/test_summary_throughput.py
+++ 현재/tests/test_summary_throughput.py
@@ -0,0 +1,51 @@
+"""전체 통계의 결과를 유지하면서 포함 인덱스와 읽기 스냅샷을 검사합니다."""
+import contextlib
+from pathlib import Path
+import tempfile
+import unittest
+from unittest.mock import patch
+
+import engine
+
+
+class SummaryThroughputTests(unittest.TestCase):
+    def test_summary_keeps_exact_values_and_uses_covering_aggregate(self):
+        with tempfile.TemporaryDirectory() as folder:
+            store = engine.CandleStore(Path(folder) / 'test.sqlite3')
+            with engine.connect(store.path) as db:
+                db.executemany('INSERT INTO candles VALUES(?,?,?,?,?,?,?,?,?)', [
+                    ('KRW-BTC', '5', '2026-01-01T00:00Z', '2026-01-01T00:06Z', 1, 1, 1, 1, 1),
+                    ('KRW-BTC', '5', '2026-01-01T00:05Z', '2026-01-01T00:04Z', 1, 1, 1, 1, 1),
+                    ('KRW-SOL', 'day', '2025-12-31T00:00Z', '2026-01-01T00:00Z', 1, 1, 1, 1, 1)])
+                db.execute('INSERT INTO cursors VALUES(?,?,?,?)', ('KRW-BTC', '5', None, 1))
+                baseline = db.execute('SELECT market,tf,COUNT(*),MIN(open_time),MAX(close_time) '
+                                      'FROM candles GROUP BY market,tf').fetchall()
+            trace = []
+            original = engine.connect
+
+            @contextlib.contextmanager
+            def traced(path):
+                with original(path) as db:
+                    db.set_trace_callback(trace.append)
+                    yield db
+
+            with patch.object(engine, 'connect', traced):
+                actual = store.summary()
+            self.assertEqual([(r['market'], r['timeframe'], r['rows'], r['start'], r['end'])
+                              for r in actual], baseline)
+            self.assertEqual([r['backfill_exhausted'] for r in actual], [True, False])
+            aggregates = [q for q in trace if 'GROUP BY' in q.upper()]
+            self.assertEqual(len(aggregates), 1)
+            with engine.connect(store.path) as db:
+                plan = db.execute('EXPLAIN QUERY PLAN ' + aggregates[0]).fetchall()
+            self.assertTrue(all('COVERING INDEX' in str(row) for row in plan), plan)
+            self.assertIn('BEGIN', [q.upper() for q in trace])
+
+    def test_empty_candle_store_has_no_datasets(self):
+        with tempfile.TemporaryDirectory() as folder:
+            store = engine.CandleStore(Path(folder) / 'test.sqlite3')
+            self.assertEqual(store.summary(), [])
+
+
+if __name__ == '__main__':
+    unittest.main()

```

## tests/test_recent_archive_plan.py

이전: 없음  
현재: 2175 bytes / SHA-256 600694075ad2aaaeeb7fd32b88dc0896035cf7cb0fcfe8773044672470a5c00c

```diff
--- 이전/tests/test_recent_archive_plan.py
+++ 현재/tests/test_recent_archive_plan.py
@@ -0,0 +1,45 @@
+"""월 경계에서도 학습에 필요한 최근 완료 일별 자료를 확보합니다."""
+import importlib
+import unittest
+
+import pandas as pd
+
+external = importlib.import_module('3차.external_data')
+
+
+class RecentArchivePlanTests(unittest.TestCase):
+    def test_month_boundary_keeps_previous_month_recent_daily_fallback(self):
+        now = pd.Timestamp('2026-11-02T01:00:00Z')
+        plan = list(external.bootstrap_plan(now))
+        for symbol in external.SYMBOLS:
+            for date in pd.date_range('2026-10-19', '2026-11-01'):
+                expected = external.archive_spec(symbol, 'klines', date.strftime('%Y-%m-%d'), 'daily')
+                self.assertIn(expected, plan)
+
+    def test_new_year_and_leap_month_use_complete_utc_days_only(self):
+        for stamp in ('2027-01-01T00:01:00Z', '2028-03-01T12:00:00Z'):
+            now = pd.Timestamp(stamp)
+            plan = list(external.bootstrap_plan(now))
+            daily = [item for item in plan if item['frequency'] == 'daily']
+            self.assertEqual(len({item['name'] for item in plan}), len(plan))
+            self.assertEqual(len(daily), 14 * len(external.SYMBOLS))
+            self.assertTrue(all(pd.Timestamp(item['period'], tz='UTC') < now.normalize()
+                                for item in daily))
+
+    def test_current_month_history_is_kept_without_duplicate_requests(self):
+        now = pd.Timestamp('2026-11-25T12:00:00Z')
+        plan = list(external.bootstrap_plan(now))
+        self.assertEqual(len(plan), (1 + 24) * len(external.SYMBOLS))
+        self.assertEqual(len({item['name'] for item in plan}), len(plan))
+        self.assertTrue(all(item['kind'] == 'klines' for item in plan))
+        self.assertFalse(any('2026-11-25.zip' in item['name'] for item in plan))
+
+    def test_non_utc_input_cannot_request_incomplete_utc_day(self):
+        local = pd.Timestamp('2026-11-02T00:30:00+09:00')
+        expected = list(external.bootstrap_plan(local.tz_convert('UTC')))
+        self.assertEqual(list(external.bootstrap_plan(local)), expected)
+        self.assertFalse(any('2026-11-01.zip' in item['name'] for item in expected))
+
+
+if __name__ == '__main__':
+    unittest.main()

```

