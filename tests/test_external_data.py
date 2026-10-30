"""외부 아카이브의 단위·정제·중복·집계 경계를 검사합니다."""
import importlib
import hashlib
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

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


class ExternalDownloadTests(unittest.TestCase):
    def collector(self, folder, target=50_000_000_000):
        store = Mock()
        store.summary.return_value = {}
        store.ready.return_value = True
        with patch.object(external, 'ExternalStore', return_value=store):
            return external.Collector(folder, target)

    def response(self, content):
        result = io.BytesIO(content)
        result.headers = {'Content-Length': str(len(content))}
        return result

    def test_download_checks_sha_before_accepting_archive(self):
        spec = external.archive_spec('BTCUSDT', 'klines', '2026-09')
        content = b'controlled-archive-bytes'
        for correct in (True, False):
            digest = hashlib.sha256(content).hexdigest() if correct else '0' * 64
            with self.subTest(correct=correct), tempfile.TemporaryDirectory() as temp:
                collector = self.collector(Path(temp))
                responses = [self.response((digest+'  '+spec['name']+'\n').encode()), self.response(content)]
                with patch.object(external, 'open_public', side_effect=responses):
                    if correct:
                        path, actual = collector.download(spec)
                        self.assertEqual(path.read_bytes(), content)
                        self.assertEqual(actual, digest)
                    else:
                        with self.assertRaisesRegex(ValueError, 'SHA256'):
                            collector.download(spec)
                        self.assertFalse((collector.archives / spec['name']).exists())
                self.assertFalse(list(collector.archives.glob('*.part')))

    def test_fifty_gb_gate_counts_only_verified_and_ingested_archives(self):
        with tempfile.TemporaryDirectory() as temp:
            collector = self.collector(Path(temp))
            collector.manifest = {'sample': {'bytes': 50_000_000_000, 'sha256_verified': True, 'ingested': False}}
            # 용량 계산만 격리하며 실제 파일 누락·손상은 아래 별도 시험에서 검사합니다.
            with patch.object(collector, 'verified_items', side_effect=lambda: [v for v in collector.manifest.values() if v['sha256_verified']]):
                self.assertFalse(collector.status()['training_ready'])
                collector.manifest['sample']['ingested'] = True
                self.assertTrue(collector.status()['training_ready'])
                collector.manifest['sample']['bytes'] -= 1
                self.assertFalse(collector.status()['training_ready'])
                collector.manifest['sample']['bytes'] = 50_000_000_000
                collector.manifest['sample']['sha256_verified'] = False
                self.assertFalse(collector.status()['training_ready'])

    def test_real_missing_or_same_size_corrupt_archive_cannot_open_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            collector = self.collector(Path(temp), target=8)
            spec = external.archive_spec('BTCUSDT', 'klines', '2026-09')
            item = {**spec, 'bytes': 8, 'sha256': hashlib.sha256(b'12345678').hexdigest(),
                    'sha256_verified': True, 'ingested': True}
            collector.manifest = {spec['name']: item}
            self.assertFalse(collector.status()['training_ready'])
            self.assertEqual(collector.status()['verified_archive_bytes'], 0)
            path = collector.archives / spec['name']
            path.write_bytes(b'12345678')
            self.assertTrue(collector.status()['training_ready'])
            path.write_bytes(b'87654321')
            self.assertFalse(collector.status()['training_ready'])
            self.assertEqual(collector.status()['ingested_archive_bytes'], 0)

    def test_ingest_stops_before_save_when_disk_reserve_is_exhausted(self):
        with tempfile.TemporaryDirectory() as temp:
            collector = self.collector(Path(temp))
            spec = external.archive_spec('BTCUSDT', 'klines', '2026-09')
            path = Path(temp) / spec['name']
            text = ExternalParserTests().kline().to_csv(index=False, header=False)
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr(spec['name'][:-4]+'.csv', text)
            with patch.object(external.shutil, 'disk_usage', return_value=Mock(free=0)):
                with self.assertRaisesRegex(OSError, '20GB'):
                    collector.ingest(spec, path)
            collector.store.save.assert_not_called()

    def test_verified_but_not_ingested_target_is_retried_not_declared_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            collector = self.collector(Path(temp))
            stop = external.threading.Event()
            spec = external.archive_spec('BTCUSDT', 'aggTrades', '2026-08')
            state = {'verified_archive_bytes': 50_000_000_000, 'ingested_archive_bytes': 0}
            with patch.object(external, 'STOP', stop), patch.object(external.threading, 'Thread') as thread, \
                 patch.object(external, 'bootstrap_plan', return_value=[]), \
                 patch.object(external, 'bulk_plan', return_value=[spec]), \
                 patch.object(collector, 'status', return_value=state), \
                 patch.object(collector, 'archive', side_effect=lambda value: stop.set()) as archive, \
                 patch.object(stop, 'wait', side_effect=lambda value: stop.set()):
                thread.return_value.is_alive.return_value = False
                collector.run()
            archive.assert_called_once_with(spec)


class ExternalProcessBoundaryTests(unittest.TestCase):
    def test_only_exact_collector_script_and_cwd_can_match(self):
        manage = importlib.import_module('3차.manage')
        for correct_cwd, script, action, expected in (
                (True, str(manage.SCRIPT), 'run', True),
                (False, str(manage.SCRIPT), 'run', False),
                (True, str(manage.PROJECT / 'supervisor.py'), 'run', False),
                (True, str(manage.SCRIPT), 'status', False)):
            proc, cwd, cmdline = Mock(), Mock(), Mock()
            cwd.resolve.return_value = manage.PROJECT.resolve() if correct_cwd else manage.ROOT.resolve()
            cmdline.read_bytes.return_value = ('python\0-u\0'+script+'\0'+action+'\0').encode()
            proc.joinpath.side_effect = lambda name: cwd if name == 'cwd' else cmdline
            with self.subTest(script=script, action=action), patch.object(manage, 'Path') as path:
                path.return_value.__truediv__.return_value = proc
                self.assertEqual(manage.matches(123), expected)


@unittest.skipUnless(sys.platform == 'linux', '심볼릭 링크 경계는 실제 Linux에서 검사합니다')
class ExternalPathBoundaryTests(unittest.TestCase):
    def test_archive_symlink_is_rejected_before_network_or_hash_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            helper = ExternalDownloadTests()
            collector = helper.collector(Path(temp) / 'work')
            spec = external.archive_spec('BTCUSDT', 'klines', '2026-09')
            outside = Path(temp) / 'outside'
            outside.write_bytes(b'keep-original')
            (collector.archives / spec['name']).symlink_to(outside)
            with patch.object(external, 'open_public') as network, self.assertRaisesRegex(ValueError, '경계'):
                collector.download(spec)
            network.assert_not_called()
            self.assertEqual(outside.read_bytes(), b'keep-original')

    def test_old_part_symlink_is_not_followed_or_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            helper = ExternalDownloadTests()
            collector = helper.collector(Path(temp) / 'work')
            spec = external.archive_spec('BTCUSDT', 'klines', '2026-09')
            outside = Path(temp) / 'outside'
            outside.write_bytes(b'keep-original')
            legacy = collector.archives / (spec['name'] + '.part')
            legacy.symlink_to(outside)
            content = b'archive-content'
            digest = hashlib.sha256(content).hexdigest()
            with patch.object(external, 'open_public', side_effect=[
                    helper.response((digest+'  '+spec['name']+'\n').encode()), helper.response(content)]):
                path, _ = collector.download(spec)
            self.assertEqual(path.read_bytes(), content)
            self.assertFalse(path.is_symlink())
            self.assertTrue(legacy.is_symlink())
            self.assertEqual(outside.read_bytes(), b'keep-original')


@unittest.skipUnless(sys.platform == 'linux', '외부 운영 SQLite도 Linux에서만 검사합니다')
class ExternalStoreTests(unittest.TestCase):
    def test_existing_wrong_schema_is_rejected_before_marker_normalization(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'external.sqlite3'
            db = sqlite3.connect(path)
            db.execute('CREATE TABLE bars(symbol TEXT,open_time INTEGER)')
            db.commit()
            db.close()
            with self.assertRaisesRegex(ValueError, '스키마'):
                external.ExternalStore(path)

    def test_first_completed_bar_is_immutable_and_duplicate_count_stays_one(self):
        with tempfile.TemporaryDirectory() as temp:
            store = external.ExternalStore(Path(temp) / 'external.sqlite3')
            frame = external.parse_klines(ExternalParserTests().kline(), 1735689720000)
            self.assertEqual(store.save('BTCUSDT', frame, 'verified-source'), 1)
            self.assertEqual(store.save('BTCUSDT', frame, 'same-minute'), 0)
            self.assertEqual(store.summary()['BTCUSDT']['rows'], 1)


if __name__ == '__main__':
    unittest.main()
