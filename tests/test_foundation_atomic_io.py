"""원자 JSON 교체의 Windows 공유 경합과 실패 보존을 검증합니다."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import call, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('foundation_atomic_io', ROOT/'7차/shadow_research.py')
f = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(f)


def sharing_error(code):
    error = PermissionError(13, '통제된 공유 오류 원문')
    error.winerror = code
    return error


class AtomicJsonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'status.json'
        self.original = b'{"version":"before"}\n'
        self.path.write_bytes(self.original)

    def test_windows_transient_sharing_errors_retry_same_payload(self):
        replace = f.os.replace
        for code in (5, 32, 33):
            with self.subTest(winerror=code):
                self.path.write_bytes(self.original)
                attempts = []

                def transient(source, target):
                    attempts.append(source)
                    if len(attempts) == 1:
                        self.assertEqual(self.path.read_bytes(), self.original)
                        raise sharing_error(code)
                    return replace(source, target)

                with patch.object(f, 'sys', SimpleNamespace(platform='win32')), \
                        patch.object(f.os, 'replace', side_effect=transient), \
                        patch.object(f.time, 'sleep') as sleep:
                    f.atomic_json(self.path, {'version': 'after'})
                self.assertEqual(f.read_json(self.path), {'version': 'after'})
                self.assertEqual(len(attempts), 2)
                self.assertEqual(attempts[0], attempts[1])
                sleep.assert_called_once_with(0.01)

    def test_windows_permanent_error_preserves_original_exception_and_files(self):
        for code in (5, 32, 33):
            with self.subTest(winerror=code):
                error = sharing_error(code)
                before = set(self.path.parent.glob('*.tmp'))
                with patch.object(f, 'sys', SimpleNamespace(platform='win32')), \
                        patch.object(f.os, 'replace', side_effect=error) as replace, \
                        patch.object(f.time, 'sleep') as sleep:
                    with self.assertRaises(OSError) as caught:
                        f.atomic_json(self.path, {'version': 'after'})
                self.assertIs(caught.exception, error)
                self.assertEqual(str(caught.exception), str(error))
                self.assertEqual(replace.call_count, 5)
                self.assertEqual(sleep.call_args_list, [call(0.01), call(0.02), call(0.03), call(0.04)])
                self.assertEqual(self.path.read_bytes(), self.original)
                remaining = set(self.path.parent.glob('*.tmp'))-before
                self.assertEqual(len(remaining), 1)
                self.assertEqual(f.read_json(remaining.pop()), {'version': 'after'})

    def test_nonsharing_windows_error_fails_without_retry(self):
        for error in (OSError(5, '일반 입출력 오류'), sharing_error(87)):
            with self.subTest(error=error):
                with patch.object(f, 'sys', SimpleNamespace(platform='win32')), \
                        patch.object(f.os, 'replace', side_effect=error) as replace, \
                        patch.object(f.time, 'sleep') as sleep:
                    with self.assertRaises(OSError) as caught:
                        f.atomic_json(self.path, {'version': 'after'})
                self.assertIs(caught.exception, error)
                self.assertEqual(replace.call_count, 1)
                sleep.assert_not_called()
                self.assertEqual(self.path.read_bytes(), self.original)

    def test_linux_sharing_error_fails_without_retry(self):
        error = sharing_error(5)
        with patch.object(f, 'sys', SimpleNamespace(platform='linux')), \
                patch.object(f.os, 'replace', side_effect=error) as replace, \
                patch.object(f.time, 'sleep') as sleep:
            with self.assertRaises(OSError) as caught:
                f.atomic_json(self.path, {'version': 'after'})
        self.assertIs(caught.exception, error)
        self.assertEqual(replace.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_link_appearing_before_retry_is_rejected(self):
        with patch.object(f, 'sys', SimpleNamespace(platform='win32')), \
                patch.object(Path, 'is_symlink', side_effect=[False, False, True]), \
                patch.object(f.os, 'replace', side_effect=sharing_error(5)) as replace, \
                patch.object(f.time, 'sleep'):
            with self.assertRaises(ValueError):
                f.atomic_json(self.path, {'version': 'after'})
        self.assertEqual(replace.call_count, 1)
        self.assertEqual(self.path.read_bytes(), self.original)

    @unittest.skipUnless(sys.platform == 'win32', 'Windows 실제 공유 핸들 시험입니다')
    def test_real_reader_release_after_observed_sharing_error(self):
        opened, release, closed = threading.Event(), threading.Event(), threading.Event()
        errors = []

        def reader():
            try:
                with self.path.open('rb'):
                    opened.set()
                    if not release.wait(5):
                        raise AssertionError('읽기 핸들 해제 신호가 없습니다')
            except BaseException as error:
                errors.append(error)
            finally:
                closed.set()

        thread = threading.Thread(target=reader)
        thread.start()
        replace = f.os.replace
        observed = []

        def observed_replace(source, target):
            try:
                return replace(source, target)
            except OSError as error:
                observed.append(error)
                self.assertIn(error.winerror, (5, 32, 33))
                self.assertEqual(self.path.read_bytes(), self.original)
                release.set()
                self.assertTrue(closed.wait(5), '읽기 핸들이 닫히지 않았습니다')
                raise

        try:
            self.assertTrue(opened.wait(5), '읽기 핸들이 열리지 않았습니다')
            with patch.object(f.os, 'replace', side_effect=observed_replace) as replacement:
                f.atomic_json(self.path, {'version': 'after'})
            self.assertEqual(replacement.call_count, 2)
            self.assertEqual(len(observed), 1)
            self.assertEqual(f.read_json(self.path), {'version': 'after'})
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main()
