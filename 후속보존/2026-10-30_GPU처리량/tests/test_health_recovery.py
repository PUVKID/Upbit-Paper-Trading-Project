"""운영 DB 대신 임시 WAL 저장소로 빠른 점검과 전체 검사를 검증합니다."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import healthcheck


class HealthRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.database = self.root / 'candles.sqlite3'
        self.service = self.root / 'service'
        self.service.mkdir()
        with closing(sqlite3.connect(self.database)) as db:
            db.executescript('''PRAGMA journal_mode=WAL;
                CREATE TABLE candles(market TEXT, tf TEXT, open_time TEXT, close_time TEXT,
                    PRIMARY KEY(market,tf,open_time));
                CREATE INDEX candle_close ON candles(market,tf,close_time);
                INSERT INTO candles VALUES('KRW-BTC',5,'2026-10-28T00:00:00Z','2026-10-28T00:05:00Z');''')
            db.commit()

    def test_latest_uses_index_without_aggregate(self):
        with closing(sqlite3.connect(self.database)) as db:
            statements = []
            db.set_trace_callback(statements.append)
            rows = healthcheck.latest_rows(db, ['KRW-BTC'], [5])
            self.assertEqual(rows[0][3], '2026-10-28T00:05:00Z')
            self.assertFalse(any('COUNT(' in sql.upper() or 'GROUP BY' in sql.upper()
                                 or 'QUICK_CHECK' in sql.upper() for sql in statements))
            plan = db.execute('EXPLAIN QUERY PLAN ' + statements[-1]).fetchall()
            self.assertIn('candle_close', str(plan))

    def test_full_check_records_real_counts_and_wal_rows(self):
        with closing(sqlite3.connect(self.database)) as writer:
            writer.execute("INSERT INTO candles VALUES('KRW-BTC',5,'b','2026-10-28T00:10:00Z')")
            writer.commit()
            result = healthcheck.full_integrity(self.database, time.monotonic() + 10)
        self.assertEqual(result['state'], 'ok')
        self.assertEqual(result['counts'], {'KRW-BTC/5': 2})
        self.assertEqual(result['check'], 'PRAGMA integrity_check')

    def test_backup_keeps_one_read_snapshot_while_writer_continues(self):
        original_connect = sqlite3.connect
        transactions = []

        class ObservedConnection(sqlite3.Connection):
            def backup(connection, target, **kwargs):
                transactions.append(connection.in_transaction)
                with closing(original_connect(self.database)) as writer:
                    writer.execute("INSERT INTO candles VALUES('KRW-BTC','5','later','2026-10-28T00:15:00Z')")
                    writer.commit()
                return super().backup(target, **kwargs)

        def connect(*args, **kwargs):
            return original_connect(*args, **kwargs, factory=ObservedConnection)

        with patch('healthcheck.sqlite3.connect', side_effect=connect):
            result = healthcheck.full_integrity(self.database, time.monotonic() + 10)
        self.assertEqual(transactions, [True])
        self.assertEqual(result['counts'], {'KRW-BTC/5': 1})

    def test_full_check_deadline_is_not_pass(self):
        with self.assertRaises(TimeoutError):
            healthcheck.full_integrity(self.database, time.monotonic() - 1)

    def test_corrupt_database_is_not_pass(self):
        self.database.write_bytes(b'not a sqlite database')
        with self.assertRaises(sqlite3.DatabaseError):
            healthcheck.full_integrity(self.database, time.monotonic() + 10)

    def test_missing_stale_and_failure_are_explicit(self):
        state = healthcheck.integrity_status(self.database, self.service, launch=False)
        self.assertEqual(state['state'], 'pending')
        self.assertIsNone(state['last_result'])
        result = healthcheck.full_integrity(self.database, time.monotonic() + 10)
        result['checked_utc'] = '2000-01-01T00:00:00Z'
        healthcheck.engine.atomic_json(self.service / 'integrity_result.json', result)
        self.assertEqual(healthcheck.integrity_status(self.database, self.service, launch=False)['state'], 'pending')
        healthcheck.engine.atomic_json(self.service / 'integrity_progress.json',
                                     {'state': 'failed', 'error': '시간 제한 초과'})
        self.assertEqual(healthcheck.integrity_status(self.database, self.service, launch=False)['state'], 'failed')

    @unittest.skipUnless(sys.platform == 'linux', '실제 flock 검사는 Linux 전용입니다')
    def test_lock_prevents_duplicate_worker(self):
        import fcntl
        with (self.service / 'integrity.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch('healthcheck.subprocess.Popen') as spawn:
                self.assertFalse(healthcheck.start_integrity(self.database, self.service))
                spawn.assert_not_called()

    def test_all_45_text_datasets_including_day(self):
        timeframes = ['1', '3', '5', '10', '15', '30', '60', '240', 'day']
        with closing(sqlite3.connect(self.database)) as db:
            db.execute('DELETE FROM candles')
            for market in healthcheck.engine.MARKETS:
                for tf in timeframes:
                    db.execute('INSERT INTO candles VALUES(?,?,?,?)',
                               (market, tf, 'a', '2026-10-28T00:05:00Z'))
                    db.execute('INSERT INTO candles VALUES(?,?,?,?)',
                               (market, tf, 'b', '2026-10-28T00:10:00Z'))
            db.commit()
            rows = healthcheck.latest_rows(db, healthcheck.engine.MARKETS, timeframes)
        self.assertEqual(len(rows), 45)
        self.assertEqual(len({(row[0], row[1]) for row in rows}), 45)
        self.assertTrue(all(row[3] == '2026-10-28T00:10:00Z' for row in rows))

    def test_success_cache_has_age_and_real_snapshot_counts(self):
        result = healthcheck.full_integrity(self.database, time.monotonic() + 10)
        healthcheck.engine.atomic_json(self.service / 'integrity_result.json', result)
        status = healthcheck.integrity_status(self.database, self.service, launch=False)
        self.assertEqual(status['state'], 'ok')
        self.assertEqual(status['last_result']['counts'], {'KRW-BTC/5': 1})
        self.assertGreaterEqual(status['age_seconds'], 0)

    def test_other_database_cache_is_not_accepted(self):
        result = healthcheck.full_integrity(self.database, time.monotonic() + 10)
        result['database']['inode'] += 1
        healthcheck.engine.atomic_json(self.service / 'integrity_result.json', result)
        status = healthcheck.integrity_status(self.database, self.service, launch=False)
        self.assertEqual(status['state'], 'pending')
        self.assertIsNone(status['last_result'])

    @unittest.skipUnless(sys.platform == 'linux', '실제 잠금 상속 검사는 Linux 전용입니다')
    def test_retry_keeps_previous_failure_visible(self):
        healthcheck.engine.atomic_json(self.service / 'integrity_progress.json',
                                     {'state': 'failed', 'error': '검사 실패',
                                      'finished_utc': '2000-01-01T00:00:00Z'})
        with patch('healthcheck.subprocess.Popen'):
            status = healthcheck.integrity_status(self.database, self.service)
        self.assertEqual(status['state'], 'failed')
        self.assertEqual(status['progress']['last_failure']['error'], '검사 실패')

    def test_real_worker_completes_on_fixture(self):
        healthcheck.integrity_worker(self.database, self.service, timeout=10)
        status = healthcheck.integrity_status(self.database, self.service, launch=False)
        self.assertEqual(status['state'], 'ok')
        self.assertEqual(status['progress']['state'], 'completed')

    def test_worker_hard_timeout_preserves_failure(self):
        with patch('healthcheck.scan_command', return_value=[sys.executable, '-c', 'import time; time.sleep(10)']):
            started = time.monotonic()
            healthcheck.integrity_worker(self.database, self.service, timeout=0.1)
        self.assertLess(time.monotonic() - started, 4)
        result = json.loads((self.service / 'integrity_progress.json').read_text())
        self.assertEqual(result['state'], 'failed')
        self.assertIn('TimeoutExpired', result['error'])
        self.assertFalse((self.service / 'integrity_result.json').exists())

    def test_account_boundaries_unchanged(self):
        state = {'initial_cash': 10_000_000, 'paper_only': True, 'cash': 10_000_000,
                 'positions': {}, 'trades': []}
        self.assertEqual(healthcheck.check_account(state, '2026-01-01Z', '2026-10-28Z'), [])
        state['paper_only'] = False
        self.assertTrue(healthcheck.check_account(state, '2026-01-01Z', '2026-10-28Z'))


if __name__ == '__main__':
    unittest.main()
