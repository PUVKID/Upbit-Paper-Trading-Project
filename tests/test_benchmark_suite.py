"""고정 열 회차와 완료 증거 경계를 운영 실행 없이 검사합니다."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
import contextlib
import io


class SuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('test_suite_module',Path(__file__).resolve().parents[1]/'11차/benchmark_suite.py')
        cls.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.m)

    def test_fixed_ten_and_time_budget(self):
        config = self.m.f.read_json(Path(self.m.ROOT)/'11차/suite-config.json')
        self.assertEqual(self.m.settings(config)['experiments'],10)
        for key,value in (('experiments',11),('max_seconds',21601),('anchors',list(reversed(config['anchors'])))):
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.m.settings({**config,key:value})

    def test_periods_are_ordered_separate_and_outside_protected_dates(self):
        import engine
        anchors = [engine.utc(v) for v in self.m.ANCHORS]
        self.assertEqual(len(set(anchors)),10)
        for previous,current in zip(anchors,anchors[1:]):
            self.assertGreaterEqual((current-previous).total_seconds(),7*86400)
        self.assertLess(anchors[-1],engine.utc('2026-10-19T00:00:00Z'))

    def test_empty_artifacts_cannot_claim_completed_gpu_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root/'11차/work/runs/incomplete'
            folder.mkdir(parents=True)
            for name,value in (('manifest.json',{'artifacts':{}}),('request.json',{}),('report.json',{})):
                self.m.f.atomic_json(folder/name,value)
            with self.assertRaisesRegex(ValueError,'inventory'):
                self.m.verified_run(root,'incomplete',{})

    def test_resume_option_is_not_supported(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.m.main(['--suite','mock','--resume'])
        self.assertEqual(error.exception.code,2)


if __name__ == '__main__':
    unittest.main()
