"""운용 중 잘못된 잔고와 예약 시각 위반을 실제 입력으로 검사합니다."""

import copy
import unittest
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch
import healthcheck


class AccountInvariantTests(unittest.TestCase):
    def setUp(self):
        self.account = {"initial_cash": 10_000_000, "cash": 10_000_000, "paper_only": True,
                        "positions": {}, "trades": []}
        self.start = "2026-10-21T09:00:00Z"
        self.now = "2026-10-21T10:00:00Z"

    def test_funded_account_without_trades_is_valid(self):
        self.assertEqual(healthcheck.check_account(self.account, self.start, self.now), [])

    def test_negative_spot_balance_is_rejected(self):
        account = copy.deepcopy(self.account)
        account["positions"] = {"KRW-BTC": {"qty": -0.1, "cost": 0}}
        self.assertTrue(healthcheck.check_account(account, self.start, self.now))

    def test_nonfinite_balance_is_rejected(self):
        self.account["cash"] = float("nan")
        self.assertTrue(healthcheck.check_account(self.account, self.start, self.now))

    def test_trade_before_scheduled_start_is_rejected(self):
        self.account["trades"] = [{"timestamp": "2026-10-21T08:59:59Z"}]
        self.assertIn("예약 시작 전", healthcheck.check_account(self.account, self.start, self.now)[0])

    def test_future_trade_is_rejected(self):
        self.account["trades"] = [{"timestamp": "2026-10-21T10:00:06Z"}]
        self.assertIn("미래 시각", healthcheck.check_account(self.account, self.start, self.now)[0])


class InspectionTimeTests(unittest.TestCase):
    def inspect_at(self, latest):
        actual_utc = healthcheck.engine.utc
        clocks = iter([actual_utc('2026-10-26T00:04:59Z'), actual_utc('2026-10-26T00:05:01Z')])
        def clock(value=None):
            return next(clocks) if value is None else actual_utc(value)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('engine.py', 'supervisor.py', 'dashboard.py', 'healthcheck.py'):
                (root / name).write_text('', encoding='utf-8')
            (root / 'candles.sqlite3').touch()
            config = {'data_dir': str(root), 'models_dir': str(root / 'models'),
                      'base_timeframe': 5, 'timeframes': [5],
                      'paper': {'start_time': '2026-10-26T00:10:00Z', 'initial_cash': 10_000_000}}
            db = MagicMock()
            db.execute.return_value.fetchone.return_value = ('ok',)
            db.execute.return_value.fetchall.return_value = [('KRW-BTC', '5', 1, latest)]
            with patch('healthcheck.engine.load_config', return_value=config), \
                 patch('healthcheck.sqlite3.connect', return_value=db), \
                 patch('healthcheck.engine.utc', side_effect=clock):
                return healthcheck.inspect(root)

    def test_candle_completed_during_inspection_is_not_future(self):
        result = self.inspect_at('2026-10-26T00:05:00Z')
        self.assertFalse(any('미완성 미래 봉' in error for error in result['errors']))

    def test_candle_after_observation_is_still_rejected(self):
        result = self.inspect_at('2026-10-26T00:06:00Z')
        self.assertTrue(any('미완성 미래 봉' in error for error in result['errors']))


if __name__ == "__main__":
    unittest.main()
