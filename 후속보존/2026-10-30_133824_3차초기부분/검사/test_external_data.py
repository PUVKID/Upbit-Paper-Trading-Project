"""외부 아카이브의 단위·정제·중복·집계 경계를 검사합니다."""
import importlib
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

external = importlib.import_module('3차.external_data')


class ExternalParserTests(unittest.TestCase):
    def kline(self, unit=1):
        start = 1735689600000
        return pd.DataFrame([[start * unit, 100., 103., 99., 102., 5.,
                              (start + 60000) * unit - 1, 510., 3, 2., 204., 0]])

    def test_millisecond_and_microsecond_archives_produce_same_completed_bar(self):
        now = 1735689720000
        a = external.parse_klines(self.kline(), now)
        b = external.parse_klines(self.kline(1000), now)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(a.open_time.iloc[0], 1735689600000)

    def test_unfinished_bar_is_not_available_early(self):
        self.assertTrue(external.parse_klines(self.kline(), 1735689659999).empty)

    def test_invalid_ohlc_or_buy_volume_is_rejected(self):
        for index, value in ((2, 101.), (9, 7.), (7, -1.)):
            frame = self.kline()
            frame.iloc[0, index] = value
            with self.subTest(column=index), self.assertRaises(ValueError):
                external.parse_klines(frame, 1735689720000)

    def test_aggregation_combines_chunks_without_double_counting(self):
        start = 1735689600000
        chunks = [pd.DataFrame([[1, 100., 2., 1, 2, start, False, True],
                                [2, 102., 1., 3, 3, start+1000, True, True]]),
                  pd.DataFrame([[3, 101., 3., 4, 5, start+2000, False, True],
                                [4, 99., 1., 6, 6, start+60000, True, True]])]
        result = pd.concat(list(external.aggregate_trade_chunks(chunks, start+120000)), ignore_index=True)
        self.assertEqual(len(result), 2)
        first = result.iloc[0]
        self.assertEqual((first.open, first.high, first.low, first.close), (100, 102, 100, 101))
        self.assertEqual(first.volume, 6)
        self.assertEqual(first.quote_volume, 605)
        self.assertEqual(first.taker_quote_volume, 503)
        self.assertEqual(first.trade_count, 5)

    def test_trade_timestamp_regression_and_invalid_boolean_are_rejected(self):
        for values in ([[1, 100, 1, 1, 1, 1735689600100, 'perhaps', True]],
                       [[1, 100, 1, 1, 1, 1735689600100, True, True],
                        [2, 100, 1, 2, 2, 1735689600000, True, True]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                list(external.aggregate_trade_chunks([pd.DataFrame(values)], 1735689720000))

    def test_zip_path_and_multiple_member_are_rejected(self):
        for names in (['../escape.csv'], ['a.csv', 'b.csv']):
            data = io.BytesIO()
            with zipfile.ZipFile(data, 'w') as archive:
                for name in names:
                    archive.writestr(name, 'x')
            data.seek(0)
            with zipfile.ZipFile(data) as archive, self.assertRaises(ValueError):
                external.archive_member(archive, 'a.csv')

    def test_bootstrap_does_not_request_incomplete_daily_or_monthly_archive(self):
        plan = list(external.bootstrap_plan(pd.Timestamp('2026-10-30T12:00:00Z')))
        self.assertEqual(len({p['symbol'] for p in plan}), 5)
        self.assertTrue(all('2026-10-30.zip' not in p['name'] for p in plan))
        self.assertTrue(all(p['kind'] == 'klines' for p in plan))


@unittest.skipUnless(sys.platform == 'linux', '외부 운영 SQLite도 Linux에서만 검사합니다')
class ExternalStoreTests(unittest.TestCase):
    def test_first_completed_bar_is_immutable_and_duplicate_count_stays_one(self):
        with tempfile.TemporaryDirectory() as temp:
            store = external.ExternalStore(Path(temp) / 'external.sqlite3')
            frame = external.parse_klines(ExternalParserTests().kline(), 1735689720000)
            self.assertEqual(store.save('BTCUSDT', frame, 'verified-source'), 1)
            self.assertEqual(store.save('BTCUSDT', frame, 'same-minute'), 0)
            self.assertEqual(store.summary()['BTCUSDT']['rows'], 1)


if __name__ == '__main__':
    unittest.main()
