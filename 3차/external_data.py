"""공식 외부 시세를 검증·청크 집계하여 연구 전용으로 저장합니다."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'work'
SYMBOLS = ('BTCUSDT', 'SOLUSDT', 'ETHUSDT', 'XRPUSDT', 'DOGEUSDT')
BASE = 'https://data.binance.vision/'
LIVE = 'https://data-api.binance.vision/api/v3/klines'
TARGET_BYTES = 50_000_000_000
RESERVE_BYTES = 20_000_000_000
MAX_ARCHIVE = 6_000_000_000
BAR_COLUMNS = ['open_time', 'open', 'high', 'low', 'close', 'volume',
               'quote_volume', 'trade_count', 'taker_quote_volume']
STOP = threading.Event()


class ArchiveIntegrityError(ValueError):
    """안전한 일반 파일의 크기·해시 불일치만 복구 대상으로 구분합니다."""


def file_sha256(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(fd, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
BAR_TYPES = {'symbol': 'TEXT', 'open_time': 'INTEGER', 'open': 'REAL', 'high': 'REAL', 'low': 'REAL',
             'close': 'REAL', 'volume': 'REAL', 'quote_volume': 'REAL', 'trade_count': 'INTEGER',
             'taker_quote_volume': 'REAL', 'source': 'TEXT'}


def disk_budget(path, additional=0):
    if shutil.disk_usage(path).free < RESERVE_BYTES + max(1_048_576, additional):
        raise OSError('외부 자료 쓰기를 중단합니다: 디스크 여유20GB보호선 부족')


def check_schema(db):
    kind = db.execute("SELECT type FROM sqlite_master WHERE name='bars'").fetchone()
    info = db.execute('PRAGMA table_info(bars)').fetchall()
    if (kind != ('table',) or [(r[1], r[2].upper()) for r in info] != list(BAR_TYPES.items())
            or {r[1]: r[5] for r in info if r[5]} != {'symbol': 1, 'open_time': 2}):
        raise ValueError('외부 bars 실제 스키마 또는 기본키가 계약과 다릅니다')
    if db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone() != ('binance_spot_v1',):
        raise ValueError('외부 DB 스키마 식별자가 계약과 다릅니다')


def utc_ms():
    return time.time_ns() // 1_000_000


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def timestamp_ms(values):
    values = pd.to_numeric(values, errors='raise')
    if not np.isfinite(values).all() or (values % 1 != 0).any():
        raise ValueError('UTC 시각은 유한한 정수여야 합니다')
    values = values.astype('int64')
    micros = values >= 100_000_000_000_000
    if micros.any() and not micros.all():
        raise ValueError('한 파일 안에서 시간 단위가 섞였습니다')
    result = values // 1000 if micros.all() else values
    if ((result < 1_400_000_000_000) | (result > 4_102_444_800_000)).any():
        raise ValueError('지원하지 않는 시세 시각입니다')
    return result


def validate_bars(frame):
    if frame.empty:
        return frame
    numeric = frame[BAR_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError('외부 분봉에 비유한 값이 있습니다')
    if (frame.open_time % 60000 != 0).any() or (frame.trade_count % 1 != 0).any():
        raise ValueError('분봉 시작 또는 체결수가 정수 경계와 다릅니다')
    if (frame[['open', 'high', 'low', 'close']] <= 0).any().any():
        raise ValueError('가격은 양수여야 합니다')
    if ((frame.high < frame[['open', 'close', 'low']].max(axis=1)) |
        (frame.low > frame[['open', 'close', 'high']].min(axis=1))).any():
        raise ValueError('외부 분봉 OHLC 범위 오류')
    if (frame[['volume', 'quote_volume', 'trade_count', 'taker_quote_volume']] < 0).any().any():
        raise ValueError('음수 거래량 또는 체결수입니다')
    if (frame.taker_quote_volume > frame.quote_volume * (1 + 1e-8) + 1e-8).any():
        raise ValueError('주도 매수 대금이 전체 거래대금을 넘습니다')
    return frame


def parse_klines(raw, now_ms):
    if raw.shape[1] != 12:
        raise ValueError('외부 kline 열은12개여야 합니다')
    raw = raw.copy()
    if len(raw) and str(raw.iloc[0, 0]).lower() in ('open_time', 'open time'):
        raw = raw.iloc[1:]
    starts = timestamp_ms(raw[0])
    ends = timestamp_ms(raw[6])
    if not (ends == starts + 59999).all():
        raise ValueError('외부 분봉 완료 시각이 간격과 다릅니다')
    frame = pd.DataFrame({'open_time': starts})
    for name, column in [('open', 1), ('high', 2), ('low', 3), ('close', 4), ('volume', 5),
                         ('quote_volume', 7), ('trade_count', 8), ('taker_quote_volume', 10)]:
        frame[name] = pd.to_numeric(raw[column], errors='raise')
    base_buy = pd.to_numeric(raw[9], errors='raise')
    if (base_buy < 0).any() or (base_buy > frame.volume * (1 + 1e-8) + 1e-8).any():
        raise ValueError('주도 매수 수량이 거래량 범위를 넘습니다')
    validate_bars(frame)
    if not frame.open_time.is_monotonic_increasing or frame.open_time.duplicated().any():
        raise ValueError('분봉 시각 중복 또는 역행입니다')
    return frame.loc[frame.open_time + 60000 <= now_ms, BAR_COLUMNS].reset_index(drop=True)


def aggregate_trade_chunks(chunks, now_ms):
    carry = None
    last_time = None
    last_id = None
    for raw in chunks:
        if raw.shape[1] != 8:
            raise ValueError('외부 aggTrades 열은8개여야 합니다')
        raw = raw.copy()
        if len(raw) and str(raw.iloc[0, 0]).lower() in ('agg_trade_id', 'aggregate_trade_id'):
            raw = raw.iloc[1:]
        if raw.empty:
            continue
        stamps = timestamp_ms(raw[5])
        ids = pd.to_numeric(raw[0], errors='raise').astype('int64')
        if (not stamps.is_monotonic_increasing or (last_time is not None and stamps.iloc[0] < last_time)
                or not ids.is_monotonic_increasing or ids.duplicated().any()
                or (last_id is not None and ids.iloc[0] <= last_id)):
            raise ValueError('체결 시각 또는 식별자 순서가 역행·중복됩니다')
        maker = raw[6].astype(str).str.lower()
        if not maker.isin(['true', 'false']).all():
            raise ValueError('매수 주도 방향 값이 올바르지 않습니다')
        price, quantity = pd.to_numeric(raw[1], errors='raise'), pd.to_numeric(raw[2], errors='raise')
        first, last = pd.to_numeric(raw[3], errors='raise'), pd.to_numeric(raw[4], errors='raise')
        if (not np.isfinite(price).all() or not np.isfinite(quantity).all()
                or (price <= 0).any() or (quantity <= 0).any()
                or (first % 1 != 0).any() or (last % 1 != 0).any() or (last < first).any()):
            raise ValueError('체결 가격·수량·개수 오류')
        values = pd.DataFrame({'open_time': stamps // 60000 * 60000, 'price': price, 'quantity': quantity,
                               'quote': price * quantity, 'trades': last - first + 1})
        values['buy'] = values.quote.where(maker.eq('false'), 0.)
        grouped = values.groupby('open_time', sort=True).agg(
            open=('price', 'first'), high=('price', 'max'), low=('price', 'min'), close=('price', 'last'),
            volume=('quantity', 'sum'), quote_volume=('quote', 'sum'),
            trade_count=('trades', 'sum'), taker_quote_volume=('buy', 'sum')).reset_index()
        if carry is not None:
            if grouped.open_time.iloc[0] == carry['open_time']:
                row = grouped.iloc[0].copy()
                row['open'] = carry['open']
                row['high'], row['low'] = max(row['high'], carry['high']), min(row['low'], carry['low'])
                for name in ('volume', 'quote_volume', 'trade_count', 'taker_quote_volume'):
                    row[name] += carry[name]
                grouped.iloc[0] = row
            else:
                grouped = pd.concat([pd.DataFrame([carry]), grouped], ignore_index=True)
        carry = grouped.iloc[-1].to_dict()
        finished = grouped.iloc[:-1]
        if not finished.empty:
            yield validate_bars(finished[finished.open_time + 60000 <= now_ms]).reset_index(drop=True)
        last_time, last_id = int(stamps.iloc[-1]), int(ids.iloc[-1])
    if carry is not None and carry['open_time'] + 60000 <= now_ms:
        yield validate_bars(pd.DataFrame([carry])[BAR_COLUMNS])


def archive_member(archive, expected):
    members = archive.infolist()
    if len(members) != 1:
        raise ValueError('아카이브에는 CSV 하나만 있어야 합니다')
    member = members[0]
    if (member.filename != expected or PurePosixPath(member.filename).name != member.filename
            or member.is_dir() or member.flag_bits & 1 or member.file_size > 128_000_000_000
            or member.file_size > max(1, member.compress_size) * 1000):
        raise ValueError('압축 경로·암호화·크기 경계 오류')
    return member


class ExternalStore:
    def __init__(self, path):
        if sys.platform != 'linux':
            raise RuntimeError('외부 운영 SQLite도 Linux 내부에서만 접근하십시오')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='bars'").fetchone():
                check_schema(db)
                return
            db.executescript('''CREATE TABLE IF NOT EXISTS bars(
                symbol TEXT NOT NULL, open_time INTEGER NOT NULL, open REAL, high REAL, low REAL, close REAL,
                volume REAL, quote_volume REAL, trade_count INTEGER, taker_quote_volume REAL, source TEXT,
                PRIMARY KEY(symbol,open_time));
                CREATE TABLE IF NOT EXISTS stats(symbol TEXT PRIMARY KEY, rows INTEGER, first_open_ms INTEGER, last_open_ms INTEGER);
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT);
                INSERT OR IGNORE INTO metadata VALUES('schema','binance_spot_v1');''')
            check_schema(db)

    def connect(self):
        from contextlib import contextmanager

        @contextmanager
        def connection():
            db = sqlite3.connect(self.path, timeout=30)
            try:
                db.execute('PRAGMA journal_mode=WAL')
                db.execute('PRAGMA busy_timeout=30000')
                with db:
                    yield db
            finally:
                db.close()
        return connection()

    def save(self, symbol, frame, source):
        if symbol not in SYMBOLS:
            raise ValueError('지원하지 않는 외부 시장입니다')
        frame = validate_bars(frame)
        if frame.empty:
            return 0
        disk_budget(self.path.parent, len(frame) * 1024)
        if (frame.open_time + 60000 > utc_ms()).any():
            raise ValueError('미완성 외부 분봉입니다')
        rows = [(symbol, int(r[0]), *map(float, r[1:7]), int(r[7]), float(r[8]), source)
                for r in frame[BAR_COLUMNS].itertuples(index=False, name=None)]
        with self.connect() as db:
            before = db.total_changes
            db.executemany('INSERT OR IGNORE INTO bars VALUES(?,?,?,?,?,?,?,?,?,?,?)', rows)
            inserted = db.total_changes - before
            db.execute('''INSERT INTO stats VALUES(?,?,?,?) ON CONFLICT(symbol) DO UPDATE SET
                          rows=rows+excluded.rows,first_open_ms=MIN(first_open_ms,excluded.first_open_ms),
                          last_open_ms=MAX(last_open_ms,excluded.last_open_ms)''',
                       (symbol, inserted, int(frame.open_time.min()), int(frame.open_time.max())))
        return inserted

    def summary(self):
        with self.connect() as db:
            return {s: {'rows': n, 'first_open_ms': first, 'last_open_ms': last}
                    for s, n, first, last in db.execute('SELECT * FROM stats')}

    def archive_evidence(self, target, ingested, verified):
        with self.connect() as db:
            db.executemany('INSERT OR REPLACE INTO metadata VALUES(?,?)',
                           [('target_archive_bytes', str(target)), ('ingested_archive_bytes', str(ingested)),
                            ('verified_archive_bytes', str(verified))])

    def ready(self, now_ms):
        stats = self.summary()
        cutoff = (now_ms // 60000 - 14 * 1440) * 60000
        with self.connect() as db:
            check_schema(db)
            recent = {s: db.execute('SELECT COUNT(*) FROM bars WHERE symbol=? AND open_time>=?',
                                     (s, cutoff)).fetchone()[0] for s in SYMBOLS}
        return all(s in stats and stats[s]['rows'] >= 20000
                   and stats[s]['last_open_ms'] - stats[s]['first_open_ms'] >= 30 * 86400000
                   and 0 <= now_ms - stats[s]['last_open_ms'] <= 300000
                   and recent.get(s, 0) >= 14 * 1440 * .98 for s in SYMBOLS)


def archive_spec(symbol, kind, period, frequency='monthly'):
    interval = '/1m' if kind == 'klines' else ''
    name = f'{symbol}-{kind + "-1m" if kind == "klines" else kind}-{period}.zip'
    if kind == 'klines':
        name = f'{symbol}-1m-{period}.zip'
    key = f'data/spot/{frequency}/{kind}/{symbol}{interval}/{name}'
    return {'symbol': symbol, 'kind': kind, 'period': period, 'frequency': frequency, 'name': name, 'url': BASE + key}


def previous_month(year, month):
    return (year - 1, 12) if month == 1 else (year, month - 1)


def bootstrap_plan(now):
    now = pd.Timestamp(now)
    year, month = previous_month(now.year, now.month)
    for symbol in SYMBOLS:
        yield archive_spec(symbol, 'klines', f'{year:04d}-{month:02d}')
    for day in range(1, now.day):
        for symbol in SYMBOLS:
            yield archive_spec(symbol, 'klines', f'{now.year:04d}-{now.month:02d}-{day:02d}', 'daily')


def bulk_plan(now):
    year, month = previous_month(*previous_month(now.year, now.month))
    while (year, month) >= (2017, 8):
        for symbol in SYMBOLS:
            yield archive_spec(symbol, 'aggTrades', f'{year:04d}-{month:02d}')
        year, month = previous_month(year, month)


def open_public(url, timeout=30):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname not in ('data.binance.vision', 'data-api.binance.vision'):
        raise ValueError('공식 공개 시세 HTTPS 주소만 허용합니다')
    response = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'personal-paper-research/3'}, method='GET'), timeout=timeout)
    final = urllib.parse.urlparse(response.geturl())
    if final.scheme != 'https' or final.hostname not in ('data.binance.vision', 'data-api.binance.vision'):
        response.close()
        raise ValueError('허용하지 않는 외부 리다이렉트입니다')
    return response


class Collector:
    def __init__(self, work=WORK, target=TARGET_BYTES):
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.archives = self.work / 'archives'
        self.archives.mkdir(exist_ok=True)
        self.store = ExternalStore(self.work / 'data/external.sqlite3')
        self.target = target
        self.lock = threading.RLock()
        self.manifest_path = self.work / 'archive_manifest.json'
        self.manifest = json.loads(self.manifest_path.read_text(encoding='utf-8')) if self.manifest_path.exists() else {}
        if not isinstance(self.manifest, dict):
            raise ValueError('외부 아카이브 목록 형식 오류')
        self.integrity_cache = {}
        self.integrity_errors = {}
        self.runtime = {'state': '시작', 'bulk_error': None, 'live_error': None, 'archive_current': None}

    def verified_file(self, name, item):
        if (not isinstance(item, dict) or name != item.get('name') or item.get('symbol') not in SYMBOLS
                or not re.fullmatch(r'(BTCUSDT|SOLUSDT|ETHUSDT|XRPUSDT|DOGEUSDT)-(1m|aggTrades)-\d{4}-\d{2}(-\d{2})?\.zip', name)
                or type(item.get('bytes')) is not int or not 0 < item['bytes'] <= MAX_ARCHIVE
                or not isinstance(item.get('sha256'), str) or not re.fullmatch('[a-f0-9]{64}', item['sha256'])):
            raise ValueError('아카이브 원본 목록·이름·크기·해시 형식이 잘못됐습니다')
        path = self.archives / name
        if not path.is_file() or path.is_symlink() or path.resolve().parent != self.archives.resolve():
            raise ValueError('검증 원본 파일이 없거나 경계를 벗어났습니다')
        info = path.stat()
        identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if info.st_size != item['bytes']:
            raise ArchiveIntegrityError('검증 원본의 현재 크기가 다릅니다')
        cached = self.integrity_cache.get(name)
        if (cached and cached['identity'] == identity and cached['sha256'] == item['sha256']
                and time.monotonic() - cached['checked_at'] < 3600):
            return True
        actual = file_sha256(path)
        after = path.stat()
        if ((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != identity
                or actual != item['sha256']):
            raise ArchiveIntegrityError('검증 원본의 현재 SHA256 또는 파일 동일성이 다릅니다')
        self.integrity_cache[name] = {'identity': identity, 'sha256': actual, 'checked_at': time.monotonic()}
        return True

    def verified_items(self):
        verified = []
        for name, item in self.manifest.items():
            if not isinstance(item, dict) or item.get('sha256_verified') is not True:
                continue
            try:
                self.verified_file(name, item)
            except (OSError, ValueError) as error:
                self.integrity_errors[name] = str(error)
                self.integrity_cache.pop(name, None)
            else:
                self.integrity_errors.pop(name, None)
                verified.append(item)
        return verified

    def status(self, **fields):
        with self.lock:
            self.runtime.update(fields)
            stats = self.store.summary()
            verified = self.verified_items()
            total = sum(item['bytes'] for item in verified)
            ingested = sum(item['bytes'] for item in verified if item.get('ingested'))
            self.store.archive_evidence(self.target, ingested, total)
            result = {'schema_version': 1, 'updated_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                      'training_ready': not self.integrity_errors and ingested >= self.target and self.store.ready(utc_ms()),
                      'target_archive_bytes': self.target, 'ingested_archive_bytes': ingested,
                      'verified_archive_bytes': total, 'verified_archives': len(verified),
                      'unique_minute_rows': sum(v['rows'] for v in stats.values()), 'markets': stats,
                      'db_schema': 'binance_spot_v1', 'paper_only': True, **self.runtime}
            result['integrity_errors'] = dict(self.integrity_errors)
            result['error'] = result['bulk_error'] or result['live_error'] or ('원본 파일 무결성 확인 실패' if self.integrity_errors else None)
            atomic_json(self.work / 'status.json', result)
            return result

    def download(self, spec):
        name, url = spec['name'], spec['url']
        if spec != archive_spec(spec['symbol'], spec['kind'], spec['period'], spec['frequency']):
            raise ValueError('외부 아카이브 요청 명세가 고정 계약과 다릅니다')
        if spec['symbol'] not in SYMBOLS or spec['kind'] not in ('klines', 'aggTrades'):
            raise ValueError('지원하지 않는 아카이브 시장 또는 종류입니다')
        if not re.fullmatch(r'\d{4}-\d{2}(-\d{2})?', spec['period']):
            raise ValueError('아카이브 날짜 경로 오류')
        path = self.archives / name
        if (self.archives.is_symlink() or self.archives.resolve().parent != self.work.resolve()
                or path.is_symlink() or (path.exists() and not path.is_file())
                or path.resolve().parent != self.archives.resolve()):
            raise ValueError('다운로드 원본 경로·링크 경계 오류')
        with open_public(url + '.CHECKSUM') as response:
            checksum = response.read(4097).decode('ascii')
        expected = re.fullmatch(r'([a-fA-F0-9]{64})\s+\*?' + re.escape(name) + r'\s*', checksum)
        if expected is None:
            raise ValueError('체크섬 형식 또는 파일명 불일치')
        digest = expected.group(1).lower()
        if path.exists():
            item = {**spec, 'bytes': path.stat().st_size, 'sha256': digest}
            try:
                self.verified_file(name, item)
                actual = digest
            except ArchiveIntegrityError:
                actual = file_sha256(path)
            if actual != digest:
                # 기존 잘못된 원본도 삭제하지 않고 격리합니다.
                rejected = path.with_name(path.name + '.rejected-' + uuid.uuid4().hex[:8])
                path.rename(rejected)
                atomic_json(rejected.with_name(rejected.name + '.json'),
                            {'source_url': url, 'original_name': name, 'preserved_file': rejected.name,
                             'actual_sha256': actual, 'current_official_sha256': digest,
                             'previous_manifest': self.manifest.get(name),
                             'observed_at': dt.datetime.now(dt.timezone.utc).isoformat()})
                logging.error('원본 변경/손상 보존 %s 실제SHA256=%s 기대=%s', rejected.name, actual, digest)
                self.integrity_cache.pop(name, None)
            else:
                return path, digest
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.part')
        created = False
        try:
            with open_public(url, timeout=45) as response, temporary.open('xb') as stream:
                created = True
                length = int(response.headers.get('Content-Length', '0'))
                if length > MAX_ARCHIVE:
                    raise OSError('외부 아카이브 크기 또는 디스크 여유 경계')
                disk_budget(self.work, length or MAX_ARCHIVE)
                actual, size = hashlib.sha256(), 0
                started = time.monotonic()
                while chunk := response.read(1024 * 1024):
                    if STOP.is_set():
                        raise InterruptedError('사용자 중단')
                    size += len(chunk)
                    disk_budget(self.work, len(chunk))
                    if size > MAX_ARCHIVE or time.monotonic() - started > 1800:
                        raise TimeoutError('파일 크기 또는 다운로드 시간 상한 초과')
                    if size % (32 * 1024 * 1024) == 0:
                        if shutil.disk_usage(self.work).free < RESERVE_BYTES:
                            raise OSError('디스크 여유가20GB미만입니다')
                        self.status(download_bytes=size)
                    stream.write(chunk)
                    actual.update(chunk)
                if (length and size != length) or actual.hexdigest() != digest:
                    raise ValueError('다운로드 크기 또는 SHA256 검증 실패')
            os.replace(temporary, path)
        finally:
            if created:
                temporary.unlink(missing_ok=True)
        return path, digest

    def ingest(self, spec, path):
        count = 0
        with zipfile.ZipFile(path) as archive:
            member = archive_member(archive, spec['name'][:-4] + '.csv')
            with archive.open(member) as stream:
                chunks = pd.read_csv(stream, header=None, chunksize=200000)
                if spec['kind'] == 'klines':
                    parsed = (parse_klines(chunk, utc_ms()) for chunk in chunks)
                else:
                    parsed = aggregate_trade_chunks(chunks, utc_ms())
                for frame in parsed:
                    if STOP.is_set():
                        raise InterruptedError('사용자 중단')
                    disk_budget(self.work, len(frame) * 1024)
                    count += self.store.save(spec['symbol'], frame, spec['name'])
        return count

    def archive(self, spec):
        existing = self.manifest.get(spec['name'], {})
        path = self.archives / spec['name']
        self.status(state='외부 자료 수집', archive_current=spec['name'], download_bytes=0)
        for attempt in range(3):
            if STOP.is_set():
                raise InterruptedError('사용자 중단')
            try:
                path, digest = self.download(spec)
                if existing.get('ingested') is True and existing.get('sha256') == digest:
                    self.integrity_errors.pop(spec['name'], None)
                    self.status(bulk_error=None, archive_current=None)
                    return
                with self.lock:
                    self.manifest[spec['name']] = {**spec, 'bytes': path.stat().st_size,
                                                   'sha256': digest, 'sha256_verified': True, 'ingested': False}
                    atomic_json(self.manifest_path, self.manifest)
                inserted = self.ingest(spec, path)
                with self.lock:
                    self.manifest[spec['name']].update(ingested=True, inserted_rows=inserted)
                    atomic_json(self.manifest_path, self.manifest)
                self.status(bulk_error=None, archive_current=None)
                return
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    logging.warning('공식 미제공 파일 %s', spec['name'])
                    with self.lock:
                        self.manifest[spec['name']] = {**spec, 'available': False, 'http_status': 404}
                        atomic_json(self.manifest_path, self.manifest)
                    return
                if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise
                STOP.wait(min(60, max(2 ** (attempt + 1), float(error.headers.get('Retry-After', 0)))))
            except InterruptedError:
                raise
            except (OSError, TimeoutError, urllib.error.URLError):
                if attempt == 2:
                    raise
                STOP.wait(2 ** (attempt + 1))
        raise RuntimeError('외부 자료 재시도가 완료되지 않았습니다')

    def live_cycle(self):
        for symbol in SYMBOLS:
            query = urllib.parse.urlencode({'symbol': symbol, 'interval': '1m', 'limit': 1000})
            with open_public(LIVE + '?' + query) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError('실시간 응답 크기 상한 초과')
            payload = json.loads(raw)
            if not isinstance(payload, list) or not payload:
                raise ValueError('실시간 공개 분봉이 비어 있거나 오류입니다')
            frame = parse_klines(pd.DataFrame(payload), utc_ms())
            disk_budget(self.work, len(frame) * 1024)
            self.store.save(symbol, frame, 'binance-public-live-klines')
            if STOP.wait(.25):
                return
        self.status(live_error=None, live_updated_at=dt.datetime.now(dt.timezone.utc).isoformat())

    def live_loop(self):
        while not STOP.is_set():
            try:
                self.live_cycle()
            except Exception as error:
                logging.exception('외부 실시간 수집 실패')
                self.status(live_error=f'{type(error).__name__}: {error}')
            STOP.wait(15)

    def run(self):
        thread = threading.Thread(target=self.live_loop, name='external-live')
        thread.start()
        try:
            while not STOP.is_set():
                try:
                    now = pd.Timestamp.now(tz='UTC')
                    for spec in bootstrap_plan(now):
                        if STOP.is_set():
                            break
                        self.archive(spec)
                    for spec in bulk_plan(now):
                        if STOP.is_set() or self.status()['ingested_archive_bytes'] >= self.target:
                            break
                        self.archive(spec)
                    if not STOP.is_set():
                        self.status(state='목표 자료 확보 후 실시간 갱신' if self.status()['ingested_archive_bytes'] >= self.target
                                    else '공식 자료 범위 탐색 완료; 다음 갱신 대기')
                        STOP.wait(3600)
                except InterruptedError:
                    break
                except Exception as error:
                    logging.exception('외부 대용량 수집 실패')
                    self.status(state='오류 기록 후 재시도', bulk_error=f'{type(error).__name__}: {error}')
                    STOP.wait(60)
        finally:
            STOP.set()
            thread.join(timeout=90)
            if thread.is_alive():
                raise RuntimeError('외부 실시간 수집 스레드가 종료되지 않았습니다')
            self.status(state='중지됨')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'live-once', 'status'])
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('외부 데이터 서비스는 기존 Linux2026 컨테이너에서 실행하십시오')
    WORK.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(WORK / 'collector.log', maxBytes=4_000_000, backupCount=3, encoding='utf-8')
    logging.basicConfig(level=logging.INFO, handlers=[handler], format='%(asctime)s %(levelname)s %(message)s')
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    import fcntl
    with (WORK / 'collector.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        collector = Collector()
        if args.action == 'run':
            collector.run()
        elif args.action == 'live-once':
            collector.live_cycle()
        else:
            print(json.dumps(collector.status(), ensure_ascii=False))


if __name__ == '__main__':
    main()
