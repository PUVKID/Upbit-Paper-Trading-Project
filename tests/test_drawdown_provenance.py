"""현재 관측 낙폭 제한과 거래 시점 근거 기록을 유한 대역으로 확인합니다."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import engine


BTC, ETH = 'KRW-BTC', 'KRW-ETH'
START = engine.utc('2026-11-03T00:00:00Z')


def settings():
    value = copy.deepcopy(engine.DEFAULT_CONFIG['paper'])
    value.update(initial_cash=10000, fee=0, slippage=0, max_allocation=0.5,
                 max_total_allocation=0.9, daily_loss_limit=0.9, max_drawdown=0.1)
    return value


class DrawdownRiskTests(unittest.TestCase):
    def breached(self):
        portfolio = engine.Portfolio(settings())
        portfolio.process(START, {BTC:100}, {BTC:2})
        events = portfolio.process(START + engine.pd.Timedelta(minutes=5), {BTC:60, ETH:100}, {BTC:1, ETH:2})
        return portfolio, events

    def test_breach_blocks_new_market_but_preserves_stop_sale(self):
        portfolio, events = self.breached()
        self.assertEqual([e['side'] for e in events], ['sell'])
        self.assertTrue(portfolio.state['drawdown_halt'])
        self.assertEqual(portfolio.state['decision_diagnostics'][ETH]['reason'], 'max_drawdown_limit')
        self.assertAlmostEqual(portfolio.state['MDD'], 0.2)

    def test_day_boundary_and_restore_cannot_reset_drawdown_halt(self):
        portfolio, _ = self.breached()
        restored = engine.Portfolio(settings(), json.loads(json.dumps(portfolio.state)))
        events = restored.process(START + engine.pd.Timedelta(days=1), {ETH:100}, {ETH:2})
        self.assertEqual(events, [])
        self.assertTrue(restored.state['drawdown_halt'])

    def test_exact_ten_percent_breach_is_latched(self):
        portfolio = engine.Portfolio(settings())
        portfolio.state['cash'] = 9000
        portfolio.process(START, {ETH:100}, {ETH:2})
        self.assertTrue(portfolio.state['drawdown_halt'])
        self.assertEqual(portfolio.state['positions'], {})

    def test_predecision_uses_current_prices_not_future_close(self):
        portfolio = engine.Portfolio(settings())
        events = portfolio.process(START, {BTC:100}, {BTC:2}, mark_prices={BTC:10})
        self.assertEqual([e['side'] for e in events], ['buy'])
        self.assertTrue(portfolio.state['drawdown_halt'])

    def test_observed_bid_risk_prices_can_block_before_midpoint_entry(self):
        portfolio = engine.Portfolio(settings())
        portfolio.process(START, {BTC:100}, {BTC:2})
        events = portfolio.process(START + engine.pd.Timedelta(minutes=5), {BTC:100, ETH:100},
                                   {BTC:1, ETH:2}, risk_prices={BTC:70, ETH:100})
        self.assertEqual(events, [])
        self.assertTrue(portfolio.state['drawdown_halt'])

    def test_invalid_restored_risk_state_is_not_silently_reset(self):
        for field, value in [('drawdown_halt', 'false'), ('peak_equity', float('nan')), ('MDD', -0.01)]:
            with self.subTest(field=field):
                state = engine.Portfolio(settings()).state
                state[field] = value
                with self.assertRaises(ValueError):
                    engine.Portfolio(settings(), state)

    def test_positive_profit_does_not_admit_excessive_drawdown_candidate(self):
        result = engine.trading_eligibility({'net_return':0.5, 'MDD':0.11, 'trades':30}, 3, 0.1)
        self.assertFalse(result['eligible'])
        self.assertIn('max_drawdown_exceeded', result['rejection_reasons'])

    def test_config_rejects_invalid_drawdown_limit_without_touching_live_config(self):
        for value in (True, 0, -0.1, 1, float('nan')):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'config.json'
                path.write_text(json.dumps({'paper':{'max_drawdown':value}}), encoding='utf-8')
                with self.assertRaises(ValueError):
                    engine.load_config(path)

    def test_config_accepts_explicit_valid_drawdown_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'config.json'
            path.write_text(json.dumps({'paper':{'max_drawdown':0.1}}), encoding='utf-8')
            self.assertEqual(engine.load_config(path)['paper']['max_drawdown'], 0.1)


class TradeProvenanceTests(unittest.TestCase):
    def test_partial_invalid_entry_context_stays_invalid_after_sale(self):
        portfolio = engine.Portfolio(settings())
        portfolio.state['inference_diagnostics'] = {BTC:{'model_run':'entry-model', 'signal':2,
                                                       'probabilities':[float('nan'),0,1]}}
        portfolio.process(START, {BTC:100}, {BTC:2})
        self.assertEqual(portfolio.state['positions'][BTC]['entry_context']['record_status'], 'partial_invalid')
        portfolio.state['inference_diagnostics'] = {BTC:{'model_run':'exit-model', 'signal':0}}
        sell = portfolio.process(START + engine.pd.Timedelta(minutes=5), {BTC:100}, {BTC:0})[0]
        self.assertTrue(sell['entry_context_invalid'])
        self.assertEqual(sell['entry_context']['record_status'], 'partial_invalid')
        self.assertIsNone(sell['entry_model_run'])
        self.assertEqual(sell['model_run'], 'exit-model')

    def test_oversized_json_integer_context_does_not_disable_stop_sale(self):
        for location in ('entry', 'current'):
            for key, value in (('confidence', 10**400), ('probabilities', [10**400,0,1])):
                with self.subTest(location=location, key=key):
                    portfolio = engine.Portfolio(settings())
                    portfolio.process(START, {BTC:100}, {BTC:2})
                    context = {key:value}
                    if location == 'entry':
                        portfolio.state['positions'][BTC]['entry_context'] = context
                    else:
                        portfolio.state['inference_diagnostics'] = {BTC:context}
                    sell = portfolio.process(START + engine.pd.Timedelta(minutes=5), {BTC:90}, {BTC:1})[0]
                    self.assertEqual(sell['reason'], 'stop_loss')
                    field = 'entry_context' if location == 'entry' else 'decision_context'
                    self.assertEqual(sell[field]['record_status'], 'partial_invalid')
                    self.assertNotIn(key, sell[field])

    def test_real_book_keeps_quote_and_observation_timestamps_distinct(self):
        portfolio = engine.Portfolio(settings())
        quoted = START - engine.pd.Timedelta(seconds=2)
        units = [{'ask_price':100, 'bid_price':100, 'ask_size':1000, 'bid_size':1000}]
        buy = portfolio.process(START, {BTC:100}, {BTC:2}, books={BTC:units}, risk_prices={BTC:100},
                                quote_timestamps={BTC:quoted})[0]
        context = buy['execution_context']
        self.assertEqual(context['quote_utc'], engine.iso(quoted))
        self.assertEqual(context['observed_utc'], engine.iso(START))
        self.assertEqual(context['quote_age_seconds'], 2)
        self.assertEqual(context['risk_price_source'], 'observed_bid')

    def test_missing_quote_time_is_not_fabricated_from_execution_time(self):
        portfolio = engine.Portfolio(settings())
        units = [{'ask_price':100, 'bid_price':100, 'ask_size':1000, 'bid_size':1000}]
        buy = portfolio.process(START, {BTC:100}, {BTC:2}, books={BTC:units})[0]
        self.assertIsNone(buy['execution_context']['quote_utc'])
        self.assertIsNone(buy['execution_context']['quote_age_seconds'])

    def test_invalid_old_entry_metadata_does_not_disable_stop_sale(self):
        portfolio = engine.Portfolio(settings())
        portfolio.process(START, {BTC:100}, {BTC:2})
        portfolio.state['positions'][BTC]['entry_context'] = 'invalid-legacy-context'
        sell = portfolio.process(START + engine.pd.Timedelta(minutes=5), {BTC:90}, {BTC:1})[0]
        self.assertEqual(sell['reason'], 'stop_loss')
        self.assertIsNone(sell['entry_model_run'])
        self.assertIsNone(sell['entry_context'])
        self.assertTrue(sell['entry_context_invalid'])

    def test_invalid_current_diagnostics_do_not_disable_stop_sale(self):
        for diagnostic in (None, {BTC:{'probabilities':[float('nan'),0,1]}}):
            with self.subTest(diagnostic=diagnostic):
                portfolio = engine.Portfolio(settings())
                portfolio.process(START, {BTC:100}, {BTC:2})
                portfolio.state['inference_diagnostics'] = diagnostic
                sell = portfolio.process(START + engine.pd.Timedelta(minutes=5), {BTC:90}, {BTC:1})[0]
                self.assertEqual(sell['reason'], 'stop_loss')
                self.assertIn(sell['decision_context']['record_status'], ('invalid','partial_invalid'))

    def test_buy_and_sell_keep_entry_and_exit_model_distinct(self):
        portfolio = engine.Portfolio(settings())
        portfolio.state['model_run'] = 'stale-state-model'
        diagnostic = {'model_run':'entry-model', 'signal_time':engine.iso(START), 'signal':2,
                      'probabilities':[0.1,0.1,0.8], 'reason':'signal_ready', 'private_secret':'excluded'}
        portfolio.state['inference_diagnostics'] = {BTC:diagnostic}
        buy = portfolio.process(START, {BTC:100}, {BTC:2})[0]
        self.assertEqual(buy['model_run'], 'entry-model')
        self.assertEqual(buy['entry_model_run'], 'entry-model')
        self.assertNotIn('private_secret', buy['decision_context'])
        diagnostic['probabilities'][2] = 0.6
        self.assertEqual(buy['decision_context']['probabilities'], [0.1,0.1,0.8])
        portfolio.state['inference_diagnostics'] = {BTC:{'model_run':'exit-model', 'signal':0}}
        sell = portfolio.process(START + engine.pd.Timedelta(minutes=5), {BTC:100}, {BTC:0})[0]
        self.assertEqual(sell['model_run'], 'exit-model')
        self.assertEqual(sell['entry_model_run'], 'entry-model')
        self.assertEqual(sell['entry_context']['probabilities'], [0.1,0.1,0.8])

    def test_legacy_entry_is_not_attributed_to_current_model(self):
        portfolio = engine.Portfolio(settings())
        portfolio.process(START, {BTC:100}, {BTC:2})
        portfolio.state['positions'][BTC].pop('entry_context', None)
        portfolio.state['inference_diagnostics'] = {BTC:{'model_run':'new-exit-model', 'signal':0}}
        sell = portfolio.process(START + engine.pd.Timedelta(minutes=5), {BTC:100}, {BTC:0})[0]
        self.assertIsNone(sell['entry_model_run'])
        self.assertIsNone(sell['entry_context'])


if __name__ == '__main__':
    unittest.main()
