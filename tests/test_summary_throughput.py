"""전체 통계의 결과를 유지하면서 포함 인덱스와 읽기 스냅샷을 검사합니다."""
import contextlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import engine


class SummaryThroughputTests(unittest.TestCase):
    def test_summary_keeps_exact_values_and_uses_covering_aggregate(self):
        with tempfile.TemporaryDirectory() as folder:
            store = engine.CandleStore(Path(folder) / 'test.sqlite3')
            with engine.connect(store.path) as db:
                db.executemany('INSERT INTO candles VALUES(?,?,?,?,?,?,?,?,?)', [
                    ('KRW-BTC', '5', '2026-01-01T00:00Z', '2026-01-01T00:06Z', 1, 1, 1, 1, 1),
                    ('KRW-BTC', '5', '2026-01-01T00:05Z', '2026-01-01T00:04Z', 1, 1, 1, 1, 1),
                    ('KRW-SOL', 'day', '2025-12-31T00:00Z', '2026-01-01T00:00Z', 1, 1, 1, 1, 1)])
                db.execute('INSERT INTO cursors VALUES(?,?,?,?)', ('KRW-BTC', '5', None, 1))
                baseline = db.execute('SELECT market,tf,COUNT(*),MIN(open_time),MAX(close_time) '
                                      'FROM candles GROUP BY market,tf').fetchall()
            trace = []
            original = engine.connect

            @contextlib.contextmanager
            def traced(path):
                with original(path) as db:
                    db.set_trace_callback(trace.append)
                    yield db

            with patch.object(engine, 'connect', traced):
                actual = store.summary()
            self.assertEqual([(r['market'], r['timeframe'], r['rows'], r['start'], r['end'])
                              for r in actual], baseline)
            self.assertEqual([r['backfill_exhausted'] for r in actual], [True, False])
            aggregates = [q for q in trace if 'GROUP BY' in q.upper()]
            self.assertEqual(len(aggregates), 1)
            with engine.connect(store.path) as db:
                plan = db.execute('EXPLAIN QUERY PLAN ' + aggregates[0]).fetchall()
            self.assertTrue(all('COVERING INDEX' in str(row) for row in plan), plan)
            self.assertIn('BEGIN', [q.upper() for q in trace])

    def test_empty_candle_store_has_no_datasets(self):
        with tempfile.TemporaryDirectory() as folder:
            store = engine.CandleStore(Path(folder) / 'test.sqlite3')
            self.assertEqual(store.summary(), [])


if __name__ == '__main__':
    unittest.main()
