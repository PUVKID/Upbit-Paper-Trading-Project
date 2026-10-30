"""손상된 원본 행을 보존하면서 과거 수집을 계속하는지 검사합니다."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import engine


def candle(stamp, close=10):
    return {"candle_date_time_utc": stamp, "opening_price": 10, "high_price": 11,
            "low_price": 9, "trade_price": close, "candle_acc_trade_volume": 1}


class CollectionRecoveryTests(unittest.TestCase):
    def run_cycle(self, payloads, directory):
        config = engine.load_config()
        config.update(data_dir=directory, markets=["KRW-BTC"], timeframes=[3])
        with patch.object(engine, "public_get", side_effect=payloads):
            return engine.collect_cycle(config, backfill=True, pages=len(payloads))

    def test_invalid_ohlc_is_quarantined_and_cursor_advances(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [candle("2017-11-30T09:03:00"), candle("2017-11-30T09:00:00", 12)]
            self.run_cycle([rows], directory)
            store = engine.CandleStore(Path(directory) / "candles.sqlite3")
            frame = store.read("KRW-BTC", 3)
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.close.iloc[0], 10)
            self.assertEqual(store.cursor("KRW-BTC", 3), (engine.iso("2017-11-30T09:00:00"), False))
            saved = list((Path(directory) / "quarantine").glob("*.json"))
            self.assertEqual(len(saved), 1)
            record = json.loads(saved[0].read_text(encoding="utf-8"))
            self.assertEqual(record["raw"], rows[1])
            self.assertIn("OHLC", record["reason"])
            self.assertEqual(record["market"], "KRW-BTC")
            summary = json.loads((Path(directory) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["data"]["quarantined_count"], 1)
            self.assertIn("1개 격리", summary["data"]["status"])

    def test_fully_invalid_page_is_not_exhausted_and_next_page_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_cycle([[candle("2017-11-30T09:00:00", 12)],
                            [candle("2017-11-30T08:57:00")]], directory)
            store = engine.CandleStore(Path(directory) / "candles.sqlite3")
            self.assertEqual(len(store.read("KRW-BTC", 3)), 1)
            self.assertFalse(store.cursor("KRW-BTC", 3)[1])

    def test_unknown_schema_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(KeyError):
                self.run_cycle([[{"candle_date_time_utc": "2017-11-30T09:00:00"}]], directory)

    def test_quarantine_write_failure_does_not_advance_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(engine, "atomic_json", side_effect=OSError("격리 저장 실패")):
                with self.assertRaises(OSError):
                    self.run_cycle([[candle("2017-11-30T09:00:00", 12)]], directory)
            store = engine.CandleStore(Path(directory) / "candles.sqlite3")
            self.assertEqual(store.cursor("KRW-BTC", 3), (None, False))
            self.assertTrue(store.read("KRW-BTC", 3).empty)

    def test_retry_preserves_one_raw_quarantine_record(self):
        with tempfile.TemporaryDirectory() as directory:
            row = candle("2017-11-30T09:00:00", 12)
            for _ in range(2):
                frame = engine.collection_frame([row], 3, "KRW-BTC", {"count": 200}, directory)
                self.assertTrue(frame.empty)
            self.assertEqual(len(list((Path(directory) / "quarantine").glob("*.json"))), 1)

    def test_strict_parser_still_rejects_invalid_prices(self):
        with self.assertRaises(ValueError):
            engine.candles_from_api([candle("2017-11-30T09:00:00", 12)], 3)

    def test_missing_timestamp_is_not_empty_page(self):
        for value in (None, "NaT"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                engine.candles_from_api([candle(value)], 3)
