"""월 경계에서도 학습에 필요한 최근 완료 일별 자료를 확보합니다."""
import importlib
import unittest

import pandas as pd

external = importlib.import_module('3차.external_data')


class RecentArchivePlanTests(unittest.TestCase):
    def test_month_boundary_keeps_previous_month_recent_daily_fallback(self):
        now = pd.Timestamp('2026-11-02T01:00:00Z')
        plan = list(external.bootstrap_plan(now))
        for symbol in external.SYMBOLS:
            for date in pd.date_range('2026-10-19', '2026-11-01'):
                expected = external.archive_spec(symbol, 'klines', date.strftime('%Y-%m-%d'), 'daily')
                self.assertIn(expected, plan)

    def test_new_year_and_leap_month_use_complete_utc_days_only(self):
        for stamp in ('2027-01-01T00:01:00Z', '2028-03-01T12:00:00Z'):
            now = pd.Timestamp(stamp)
            plan = list(external.bootstrap_plan(now))
            daily = [item for item in plan if item['frequency'] == 'daily']
            self.assertEqual(len({item['name'] for item in plan}), len(plan))
            self.assertEqual(len(daily), 14 * len(external.SYMBOLS))
            self.assertTrue(all(pd.Timestamp(item['period'], tz='UTC') < now.normalize()
                                for item in daily))

    def test_current_month_history_is_kept_without_duplicate_requests(self):
        now = pd.Timestamp('2026-11-25T12:00:00Z')
        plan = list(external.bootstrap_plan(now))
        self.assertEqual(len(plan), (1 + 24) * len(external.SYMBOLS))
        self.assertEqual(len({item['name'] for item in plan}), len(plan))
        self.assertTrue(all(item['kind'] == 'klines' for item in plan))
        self.assertFalse(any('2026-11-25.zip' in item['name'] for item in plan))

    def test_non_utc_input_cannot_request_incomplete_utc_day(self):
        local = pd.Timestamp('2026-11-02T00:30:00+09:00')
        expected = list(external.bootstrap_plan(local.tz_convert('UTC')))
        self.assertEqual(list(external.bootstrap_plan(local)), expected)
        self.assertFalse(any('2026-11-01.zip' in item['name'] for item in expected))


if __name__ == '__main__':
    unittest.main()
