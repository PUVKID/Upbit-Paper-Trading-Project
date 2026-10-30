"""다른 커널의 운영 SQLite 혼합 접근을 엔진 진입점에서 거부합니다."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import engine


class RuntimeBoundaryTests(unittest.TestCase):
    def test_windows_cannot_open_live_work_database(self):
        for relative in ('data/candles.sqlite3', 'paper/ledger.sqlite3'):
            with self.subTest(relative=relative), patch('engine.sys.platform', 'win32'), \
                 patch('engine.sqlite3.connect') as connect:
                with self.assertRaisesRegex(RuntimeError, '도커'):
                    with engine.connect(engine.ROOT / 'work' / relative):
                        pass
                connect.assert_not_called()

    def test_windows_isolated_test_database_remains_available(self):
        with tempfile.TemporaryDirectory() as directory, patch('engine.sys.platform', 'win32'):
            with engine.connect(Path(directory) / 'isolated.sqlite3') as db:
                self.assertEqual(db.execute('SELECT 1').fetchone()[0], 1)

    def test_temporary_fixture_root_is_not_the_live_boundary(self):
        with tempfile.TemporaryDirectory() as directory, patch('engine.sys.platform', 'win32'), \
             patch('engine.ROOT', Path(directory)):
            with engine.connect(Path(directory) / 'work' / 'fixture.sqlite3') as db:
                self.assertEqual(db.execute('SELECT 1').fetchone()[0], 1)
