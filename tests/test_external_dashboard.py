"""다운로드 대기를 완료된 학습으로 표시하지 않는지 검사합니다."""
import json
from pathlib import Path
import tempfile
import unittest

import dashboard


class ExternalDashboardTests(unittest.TestCase):
    def test_waiting_data_exposes_volume_but_not_private_error_or_old_progress(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            values = {'work/reports/training.json': {'progress': 1.0, 'training': {'completed_candidates': 15, 'total_candidates': 15}},
                      'work/service/training_progress.json': {'status': 'waiting_data', 'stage': 'download'},
                      '3차/work/status.json': {'schema_version': 1, 'state': '외부 자료 수집',
                          'verified_archive_bytes': 5000, 'ingested_archive_bytes': 4000,
                          'target_archive_bytes': 50000000000, 'unique_minute_rows': 100,
                          'training_ready': False, 'error': 'private-error-original', 'api_key': 'private-test-secret'}}
            for name, value in values.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding='utf-8')
            status = dashboard.SnapshotAdapter(root).status()
            self.assertEqual(status['training']['state'], 'waiting_data')
            self.assertIsNone(status['training']['progress'])
            self.assertIsNone(status['training']['candidate_index'])
            self.assertEqual(status['external_data']['ingested_archive_bytes'], 4000)
            self.assertTrue(status['external_data']['has_error'])
            self.assertNotIn('private-error-original', json.dumps(status))
            self.assertNotIn('private-test-secret', json.dumps(status))


if __name__ == '__main__':
    unittest.main()
