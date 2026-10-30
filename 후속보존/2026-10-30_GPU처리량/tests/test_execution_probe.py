"""실제 호가 체결 진단이 전략 계좌와 분리되는지 검사합니다."""
import copy
import importlib.util
from pathlib import Path
import unittest

import engine


class ExecutionProbeTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'execution_probe.py'
        self.assertTrue(path.is_file(), '별도 체결 진단 모듈이 필요합니다')
        spec = importlib.util.spec_from_file_location('execution_probe', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def quote(self, stamp, price=100_000):
        return [{'market': 'KRW-BTC', 'timestamp': int(engine.utc(stamp).timestamp() * 1000),
                 'orderbook_units': [{'ask_price': price + 10, 'bid_price': price - 10,
                                      'ask_size': 100, 'bid_size': 100}]}]

    def test_round_trip_is_separate_and_costs_are_real(self):
        module = self.module()
        dates = iter(['2026-10-21T06:00:00Z', '2026-10-21T06:00:01Z'])
        quotes = iter([self.quote('2026-10-21T06:00:00Z'), self.quote('2026-10-21T06:00:01Z')])
        config = copy.deepcopy(engine.DEFAULT_CONFIG)
        result = module.run_probe(config, fetch=lambda: next(quotes), clock=lambda: engine.utc(next(dates)))
        self.assertTrue(result['passed'])
        self.assertEqual(result['scope'], '체결 경로 진단 · 전략 성과 아님')
        self.assertEqual(result['buy']['side'], 'buy')
        self.assertEqual(result['sell']['side'], 'sell')
        self.assertLess(result['round_trip_net_pnl'], 0)
        self.assertGreater(result['fees'], 0)
        self.assertLessEqual(result['buy']['cash'], 25_000)
        self.assertEqual(config['paper']['max_allocation'], 0.125)
        self.assertNotIn('strategy_return', result)

    def test_stale_quote_cannot_be_filled(self):
        module = self.module()
        with self.assertRaisesRegex(ValueError, '호가 시각'):
            module.run_probe(copy.deepcopy(engine.DEFAULT_CONFIG),
                             fetch=lambda: self.quote('2026-10-21T05:59:00Z'),
                             clock=lambda: engine.utc('2026-10-21T06:00:00Z'))

    def test_missing_quote_cannot_be_reported_as_success(self):
        module = self.module()
        with self.assertRaisesRegex(ValueError, '호가'):
            module.run_probe(copy.deepcopy(engine.DEFAULT_CONFIG), fetch=lambda: [], clock=engine.utc)


if __name__ == '__main__':
    unittest.main()
