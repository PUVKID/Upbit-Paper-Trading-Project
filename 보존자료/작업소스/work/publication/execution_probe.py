"""실제 공개 호가의 체결 경로를 전략 계좌와 별도로 검증합니다."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import engine


SCOPE = '체결 경로 진단 · 전략 성과 아님'


def run_probe(config, fetch=None, clock=engine.utc):
    market = 'KRW-BTC'
    if fetch is None:
        store = engine.CandleStore(Path(config['data_dir']) / 'candles.sqlite3')
        fetch = lambda: engine.public_get(store, 'orderbook', {'markets': market}, config['collection'])
    settings = copy.deepcopy(config['paper'])
    settings.update(initial_cash=25_000_000, max_allocation=0.001, max_total_allocation=0.001)
    account = engine.Portfolio(settings)
    observed, timestamps, events = [], [], []
    for side, signal in (('buy', 2), ('sell', 0)):
        payload = fetch()
        now = clock()
        if len(payload) != 1 or payload[0].get('market') != market or not payload[0].get('orderbook_units'):
            raise ValueError('진단 시장의 호가가 없습니다')
        quote = payload[0]
        stamp = engine.pd.Timestamp(quote['timestamp'], unit='ms', tz='UTC')
        age = (now - stamp).total_seconds()
        if not math.isfinite(age) or age > settings['max_quote_age_seconds'] or age < -5:
            raise ValueError(f'호가 시각이 유효하지 않습니다: age={age}')
        if observed and now <= observed[-1]:
            raise ValueError('호가 관측 시각이 증가하지 않았습니다')
        books = quote['orderbook_units']
        bid, ask = float(books[0]['bid_price']), float(books[0]['ask_price'])
        if not math.isfinite(bid + ask) or not 0 < bid <= ask:
            raise ValueError('호가 가격 또는 스프레드가 유효하지 않습니다')
        fills = account.process(now, {market: (bid + ask) / 2}, {market: signal},
                                books={market: books}, mark_prices={market: bid})
        if len(fills) != 1 or fills[0]['side'] != side:
            raise RuntimeError(f'별도 진단 계좌의 {side} 체결에 실패했습니다')
        observed.append(now)
        timestamps.append(engine.iso(stamp))
        events.append(fills[0])
    if account.state['positions']:
        raise RuntimeError('별도 진단 계좌에 미청산 잔량이 남았습니다')
    return {'checked_at': engine.iso(observed[-1]), 'market': market,
            'source': '업비트 공개 orderbook API', 'scope': SCOPE, 'passed': True,
            'buy': events[0], 'sell': events[1],
            'round_trip_net_pnl': account.state['cash'] - account.state['initial_cash'],
            'fees': account.state['fees_paid'], 'quote_timestamps': timestamps,
            'strategy_account_untouched': True, 'strategy_performance_evidence': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=engine.ROOT / 'engine-config.json')
    args = parser.parse_args()
    try:
        result = run_probe(engine.load_config(args.config))
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, IndexError) as error:
        result = {'checked_at': engine.iso(None), 'scope': SCOPE, 'passed': False,
                  'error': f'{type(error).__name__}: {error}', 'strategy_performance_evidence': False}
    engine.atomic_json(engine.ROOT / 'work/reports/execution_probe.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
