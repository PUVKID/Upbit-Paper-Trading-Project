"""외부 아카이브의 단위·정제·중복·집계 경계를 검사합니다."""
import importlib
import hashlib
import io
import json
import os
import shutil
import subprocess
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


@unittest.skipUnless(sys.platform == 'linux', '격리와 부분 저장은 Linux 임시 DB에서 검사합니다')
class ExternalQuarantineTests(unittest.TestCase):
    def zip_bytes(self, spec, rows, member=None):
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w') as archive:
            archive.writestr(member or spec['name'][:-4] + '.csv',
                             pd.DataFrame(rows).to_csv(index=False, header=False))
        return data.getvalue()

    def fixtures(self, collector):
        start = 1769904000000
        bad = external.archive_spec('BTCUSDT', 'aggTrades', '2026-02')
        good = external.archive_spec('BTCUSDT', 'aggTrades', '2026-03')
        rows = [[1, 100., 2., 1, 1, start, False, True],
                [2, 101., 1., 2, 2, start + 60000, True, True],
                [1, 100., 2., 1, 1, start, False, True],
                [3, 102., 1., 3, 3, start + 120000, False, True]]
        contents = {bad['name']: self.zip_bytes(bad, rows),
                    good['name']: self.zip_bytes(good, [
                        [10, 110., 3., 10, 10, 1772323200000, False, True]])}
        for name, content in contents.items():
            (collector.archives / name).write_bytes(content)
        return bad, good, contents

    def network(self, contents):
        def get(url, timeout=30):
            name = url.rsplit('/', 1)[-1]
            if name.endswith('.CHECKSUM'):
                name = name[:-9]
                digest = hashlib.sha256(contents[name]).hexdigest()
                return ExternalDownloadTests().response((digest + '  ' + name).encode())
            return ExternalDownloadTests().response(contents[name])
        return get

    def parse_two_rows(self):
        reader = pd.read_csv
        return patch.object(external.pd, 'read_csv',
                            side_effect=lambda *a, **kw: reader(*a, **{**kw, 'chunksize': 2}))

    def reject_without_aborting(self, collector, spec):
        try:
            collector.archive(spec)
        except ValueError as error:
            self.fail('내용 오류는 명시적 미적재 격리 후 다음 파일 진행이 필요합니다: ' + str(error))

    def test_rejection_preserves_prefix_excludes_bad_bytes_and_next_archive_succeeds(self):
        with tempfile.TemporaryDirectory() as temp:
            collector = external.Collector(Path(temp))
            bad, good, contents = self.fixtures(collector)
            with patch.object(external, 'open_public', side_effect=self.network(contents)), self.parse_two_rows():
                self.reject_without_aborting(collector, bad)
                rejected = collector.manifest[bad['name']]
                digest = hashlib.sha256(contents[bad['name']]).hexdigest()
                self.assertFalse(rejected['ingested'])
                self.assertEqual(rejected['content_rejections'][digest]['sha256'], digest)
                self.assertIn('역행', rejected['content_rejections'][digest]['reason'])
                self.assertEqual(collector.status()['ingested_archive_bytes'], 0)
                self.assertEqual(collector.status()['archive_warning_count'], 1)
                self.assertEqual((collector.archives / bad['name']).read_bytes(), contents[bad['name']])
                collector.archive(good)
            state = collector.status()
            self.assertEqual(state['ingested_archive_bytes'], len(contents[good['name']]))
            self.assertEqual(state['verified_archive_bytes'], sum(map(len, contents.values())))
            self.assertEqual(state['archive_warning_count'], 1)
            self.assertIsNone(state['error'])
            self.assertFalse(state['training_ready'])
            self.assertEqual(state['target_archive_bytes'], 50_000_000_000)
            with collector.store.connect() as db:
                prefix = db.execute('SELECT open_time,volume,quote_volume,trade_count,source FROM bars '
                                    'WHERE open_time<1772323200000').fetchall()
            self.assertEqual(prefix, [(1769904000000, 2., 200., 1, bad['name'])])

    def test_same_hash_restart_does_not_reparse_or_change_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            collector = external.Collector(Path(temp))
            bad, _, contents = self.fixtures(collector)
            with patch.object(external, 'open_public', side_effect=self.network(contents)), self.parse_two_rows():
                self.reject_without_aborting(collector, bad)
            before = collector.manifest[bad['name']]
            restarted = external.Collector(Path(temp))
            with patch.object(external, 'open_public', side_effect=self.network(contents)), \
                 patch.object(restarted, 'ingest', side_effect=AssertionError('같은 격리 해시를 재파싱했습니다')):
                restarted.archive(bad)
            self.assertEqual(restarted.manifest[bad['name']], before)
            self.assertEqual(restarted.status()['ingested_archive_bytes'], 0)
            with restarted.store.connect() as db:
                self.assertEqual(db.execute('SELECT COUNT(*),SUM(volume) FROM bars').fetchone(), (1, 2.))

    def test_new_official_hash_can_ingest_while_old_file_and_rejection_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            collector = external.Collector(Path(temp))
            bad, _, contents = self.fixtures(collector)
            original = contents[bad['name']]
            with patch.object(external, 'open_public', side_effect=self.network(contents)), self.parse_two_rows():
                self.reject_without_aborting(collector, bad)
            old_digest = hashlib.sha256(original).hexdigest()
            contents[bad['name']] = self.zip_bytes(bad, [
                [1, 100., 2., 1, 1, 1769904000000, False, True],
                [2, 101., 1., 2, 2, 1769904060000, True, True]])
            with patch.object(external, 'open_public', side_effect=self.network(contents)):
                collector.archive(bad)
            state = collector.status()
            self.assertTrue(collector.manifest[bad['name']]['ingested'])
            self.assertNotEqual(collector.manifest[bad['name']]['sha256'], old_digest)
            self.assertIn(old_digest, collector.manifest[bad['name']]['content_rejections'])
            self.assertEqual(state['ingested_archive_bytes'], len(contents[bad['name']]))
            self.assertEqual(state['archive_warning_count'], 1)
            preserved = list(collector.archives.glob(bad['name'] + '.rejected-*'))
            self.assertTrue(any(p.read_bytes() == original for p in preserved if p.suffix != '.json'))
            with collector.store.connect() as db:
                self.assertEqual(db.execute('SELECT COUNT(*),SUM(volume) FROM bars').fetchone(), (2, 3.))
            contents[bad['name']] = original
            with patch.object(external, 'open_public', side_effect=self.network(contents)), \
                 patch.object(collector, 'ingest', side_effect=AssertionError('격리 해시를 재적재했습니다')):
                collector.archive(bad)
            self.assertFalse(collector.manifest[bad['name']]['ingested'])
            self.assertEqual(collector.status()['ingested_archive_bytes'], 0)

    def test_run_continues_after_content_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            collector = external.Collector(Path(temp))
            bad, good, contents = self.fixtures(collector)
            stop = external.threading.Event()
            with patch.object(external, 'STOP', stop), patch.object(external.threading, 'Thread') as thread, \
                 patch.object(external, 'bootstrap_plan', return_value=[]), \
                 patch.object(external, 'bulk_plan', return_value=[bad, good]), \
                 patch.object(external, 'open_public', side_effect=self.network(contents)), \
                 self.parse_two_rows(), patch.object(stop, 'wait', side_effect=lambda _: stop.set()):
                thread.return_value.is_alive.return_value = False
                collector.run()
            self.assertIn(good['name'], collector.manifest)
            self.assertTrue(collector.manifest[good['name']]['ingested'])
            self.assertFalse(collector.manifest[bad['name']]['ingested'])
            self.assertIsNone(collector.status()['error'])

    def test_path_and_database_failures_are_not_quarantined_as_content(self):
        with tempfile.TemporaryDirectory() as temp:
            collector = external.Collector(Path(temp))
            spec = external.archive_spec('BTCUSDT', 'aggTrades', '2026-02')
            rows = [[1, 100., 1., 1, 1, 1769904000000, False, True]]
            for invalid_path in (True, False):
                content = self.zip_bytes(spec, rows, '../escape.csv' if invalid_path else None)
                (collector.archives / spec['name']).write_bytes(content)
                with self.subTest(invalid_path=invalid_path), \
                     patch.object(external, 'open_public', side_effect=self.network({spec['name']: content})), \
                     patch.object(collector.store, 'save', side_effect=ValueError('DB저장 실패')):
                    with self.assertRaisesRegex(ValueError, '경계' if invalid_path else 'DB저장'):
                        collector.archive(spec)
                self.assertFalse(collector.manifest[spec['name']].get('content_rejections'))

    def test_official_404_does_not_erase_existing_content_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            collector = external.Collector(Path(temp))
            bad, _, contents = self.fixtures(collector)
            with patch.object(external, 'open_public', side_effect=self.network(contents)), self.parse_two_rows():
                self.reject_without_aborting(collector, bad)
            digest = hashlib.sha256(contents[bad['name']]).hexdigest()
            rejection = collector.manifest[bad['name']]['content_rejections'][digest].copy()
            with patch.object(external, 'open_public', side_effect=external.urllib.error.HTTPError(
                    bad['url'], 404, '미제공', {}, None)):
                collector.archive(bad)
            self.assertEqual(collector.status()['archive_warning_count'], 1)
            self.assertEqual(collector.manifest[bad['name']]['content_rejections'][digest], rejection)
            self.assertFalse(collector.manifest[bad['name']]['ingested'])

    def test_mismatched_or_malformed_rejection_binding_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            collector = external.Collector(Path(temp))
            bad, _, contents = self.fixtures(collector)
            with patch.object(external, 'open_public', side_effect=self.network(contents)), self.parse_two_rows():
                self.reject_without_aborting(collector, bad)
            digest = hashlib.sha256(contents[bad['name']]).hexdigest()
            records = collector.manifest[bad['name']]['content_rejections']
            original = records[digest].copy()
            for record in (None, {**original, 'name': 'other.zip'}, {**original, 'sha256': '0' * 64}):
                records[digest] = record
                with self.subTest(record=record), self.assertRaises(ValueError):
                    collector.status()


class ExternalResearchTargetTests(unittest.TestCase):
    def collector(self, folder):
        control, store = Mock(), Mock()
        control.collection_target.return_value = 50_000_000_000
        store.summary.return_value = {}
        store.ready.return_value = False
        module = Mock()
        module.ResearchControl.return_value = control
        settings = {'active': True, 'state_path': '4차/work/research_control.json'}
        with patch.object(external, 'ExternalStore', return_value=store), \
             patch('importlib.import_module', return_value=module):
            try:
                collector = external.Collector(folder, research_settings=settings)
            except TypeError as error:
                self.fail('임시 연구 제어의 명시적 주입이 필요합니다: ' + str(error))
        return collector, control, store

    def run_once(self, collector):
        stop = external.threading.Event()
        with patch.object(external, 'STOP', stop), patch.object(external.threading, 'Thread') as thread, \
             patch.object(stop, 'wait', side_effect=lambda _: stop.set()):
            thread.return_value.is_alive.return_value = False
            collector.run()

    def test_temporary_work_default_does_not_load_actual_research_control(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch('importlib.import_module', side_effect=AssertionError('실제 연구 상태에 접근했습니다')):
            collector = ExternalDownloadTests().collector(Path(temp))
            self.assertIsNone(getattr(collector, 'research_control', None))

    def test_integer_target_and_last_target_on_none_are_written_consistently(self):
        with tempfile.TemporaryDirectory() as temp:
            collector, control, store = self.collector(Path(temp))
            self.assertEqual(collector.collection_target(), 50_000_000_000)
            control.collection_target.return_value = 52_000_000_000
            self.assertEqual(collector.collection_target(), 52_000_000_000)
            state = collector.status()
            self.assertEqual(state['target_archive_bytes'], 52_000_000_000)
            self.assertEqual(store.archive_evidence.call_args.args, (52_000_000_000, 0, 0))
            control.collection_target.return_value = None
            self.assertIsNone(collector.collection_target())
            self.assertEqual(collector.status()['target_archive_bytes'], 52_000_000_000)
            self.assertEqual(store.archive_evidence.call_args.args, (52_000_000_000, 0, 0))

    def test_completed_restart_preserves_last_persisted_integer_target(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / 'status.json').write_text(json.dumps({'target_archive_bytes': 52_000_000_000}), encoding='utf-8')
            collector, control, store = self.collector(folder)
            control.collection_target.return_value = None
            self.assertIsNone(collector.collection_target())
            self.assertEqual(collector.status()['target_archive_bytes'], 52_000_000_000)
            self.assertEqual(store.archive_evidence.call_args.args, (52_000_000_000, 0, 0))

    def test_none_prevents_bootstrap_and_bulk_but_keeps_status(self):
        with tempfile.TemporaryDirectory() as temp:
            collector, control, _ = self.collector(Path(temp))
            control.collection_target.return_value = None
            spec = external.archive_spec('BTCUSDT', 'klines', '2026-02')
            with patch.object(external, 'bootstrap_plan', return_value=[spec]), \
                 patch.object(external, 'bulk_plan', return_value=[spec]), \
                 patch.object(external, 'open_public', side_effect=AssertionError('대기 중 원본을 요청했습니다')):
                self.run_once(collector)
            self.assertEqual(collector.manifest, {})
            state = json.loads((Path(temp) / 'status.json').read_text(encoding='utf-8'))
            self.assertIsNone(state['error'])
            self.assertEqual(state['target_archive_bytes'], 50_000_000_000)

    def test_post_ingest_none_stops_remaining_bootstrap_and_bulk(self):
        with tempfile.TemporaryDirectory() as temp:
            collector, control, store = self.collector(Path(temp))
            first = external.archive_spec('BTCUSDT', 'klines', '2026-02')
            second = external.archive_spec('ETHUSDT', 'klines', '2026-02')
            content = ExternalQuarantineTests().zip_bytes(first, ExternalParserTests().kline().values.tolist())
            (collector.archives / first['name']).write_bytes(content)
            def saved(*_):
                control.collection_target.return_value = None
                return 1
            store.save.side_effect = saved
            with patch.object(external, 'bootstrap_plan', return_value=[first, second]), \
                 patch.object(external, 'bulk_plan', return_value=[second]), \
                 patch.object(external, 'open_public', side_effect=ExternalQuarantineTests().network({first['name']: content})):
                self.run_once(collector)
            self.assertEqual(set(collector.manifest), {first['name']})
            self.assertTrue(collector.manifest[first['name']]['ingested'])
            self.assertIsNone(collector.status()['error'])

    def test_reached_bytes_with_missing_coverage_does_not_skip_bootstrap(self):
        with tempfile.TemporaryDirectory() as temp:
            collector, control, _ = self.collector(Path(temp))
            spec = external.archive_spec('BTCUSDT', 'klines', '2026-02')
            visited = []
            def archive(value):
                visited.append(value['name'])
                control.collection_target.return_value = None
            with patch.object(external, 'bootstrap_plan', return_value=[spec]), \
                 patch.object(external, 'bulk_plan', return_value=[]), \
                 patch.object(collector, 'status', return_value={
                     'ingested_archive_bytes': 50_000_000_000, 'training_ready': False}), \
                 patch.object(collector, 'archive', side_effect=archive):
                self.run_once(collector)
            self.assertEqual(visited, [spec['name']])

    def test_plan_exhaustion_refreshes_new_integer_target(self):
        with tempfile.TemporaryDirectory() as temp:
            collector, control, store = self.collector(Path(temp))
            control.collection_target.return_value = 52_000_000_000
            with patch.object(external, 'bootstrap_plan', return_value=[]), \
                 patch.object(external, 'bulk_plan', return_value=[]):
                self.run_once(collector)
            self.assertEqual(collector.status()['target_archive_bytes'], 52_000_000_000)
            self.assertEqual(store.archive_evidence.call_args.args, (52_000_000_000, 0, 0))

    def test_control_error_is_not_hidden_as_none_or_collection_success(self):
        with tempfile.TemporaryDirectory() as temp:
            collector, control, _ = self.collector(Path(temp))
            control.collection_target.side_effect = RuntimeError('연구 상태 읽기 실패')
            with patch.object(external, 'bootstrap_plan', return_value=[]), \
                 patch.object(external, 'bulk_plan', return_value=[]):
                self.run_once(collector)
            self.assertIn('연구 상태 읽기 실패', collector.status()['error'])
            self.assertFalse(collector.status()['training_ready'])

    def test_invalid_control_targets_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            collector, control, _ = self.collector(Path(temp))
            for target in (True, 0, -1, '50000000000', 50.5):
                control.collection_target.return_value = target
                with self.subTest(target=target), self.assertRaises(ValueError):
                    collector.collection_target()


@unittest.skipUnless(sys.platform == 'linux', '실제 파일 진입점은 Linux 임시 프로젝트에서 검사합니다')
class ExternalEntrypointTests(unittest.TestCase):
    def test_absolute_script_entrypoint_loads_real_sibling_control_without_pythonpath(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            project, outside = folder / 'project', folder / 'outside'
            (project / '3차').mkdir(parents=True)
            (project / '4차').mkdir()
            outside.mkdir()
            script = project / '3차/external_data.py'
            shutil.copyfile(external.ROOT / 'external_data.py', script)
            shutil.copyfile(external.ROOT.parent / '4차/research_control.py', project / '4차/research_control.py')
            (project / 'engine-config.json').write_text(json.dumps({
                'research': {'active': True, 'state_path': '4차/work/research_control.json'}}), encoding='utf-8')
            env = os.environ.copy()
            env.pop('PYTHONPATH', None)
            env['PYTHONNOUSERSITE'] = '1'
            argv = [sys.executable, '-B', '-X', 'utf8', str(script), 'status']
            # 운영 경로 대신 동일 소스의 격리 사본으로 실제 CLI 초기화를 검사합니다.
            result = subprocess.run(argv, cwd=outside, env=env, capture_output=True, timeout=15, check=False)
            self.assertEqual(result.returncode, 0,
                             f'argv={argv!r}; cwd={outside}; stderr={result.stderr.decode("utf-8")}')
            state = json.loads(result.stdout.decode('utf-8'))
            self.assertEqual(state['target_archive_bytes'], 50_000_000_000)
            self.assertFalse(state['training_ready'])
            self.assertIsNone(state['error'])
            self.assertTrue((project / '3차/work/data/external.sqlite3').is_file())


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
