"""사전학습 모델의 예측을 추가 학습이나 운용 적용으로 오인하지 않는지 검사합니다."""
import unittest
import dashboard


def record(**changes):
    value = {'schema_version':1,'study':'pretrained_forecast','model_id':'google/timesfm-2.5-200m-pytorch',
             'run':'timesfm-test','state':'running','stage':'predict','windows_completed':16,'total_windows':100,
             'fit_calls':0,'promotion':False,'candidate_ready':False,'evaluation_performed':False,
             'updated_utc':'2026-11-03T00:00:00Z','comparisons':[]}
    return value|changes


class PretrainedDashboardTests(unittest.TestCase):
    def test_missing_or_invalid_cannot_override_current_research(self):
        for value in (None,{},record(fit_calls=1),record(windows_completed=True),record(total_windows=6001),
                      record(windows_completed=101),record(model_id='other'),record(promotion=True)):
            self.assertEqual(dashboard.pretrained_benchmark(value),{})

    def test_paths_and_error_text_are_private_and_device_needs_measurement(self):
        result=dashboard.pretrained_benchmark(record(error='/private/token',device='cuda:0',gpu_verified=False))
        self.assertIsNone(result['device'])
        self.assertTrue(result['has_error'])
        self.assertNotIn('private',str(result))
        self.assertEqual(result['fit_calls'],0)

    def test_complete_requires_prediction_and_evaluation(self):
        self.assertEqual(dashboard.pretrained_benchmark(record(state='complete',stage='complete')), {})
        result=dashboard.pretrained_benchmark(record(state='complete',stage='complete',windows_completed=100,
            evaluation_performed=True,gpu_verified=True,device='cuda:0'))
        self.assertEqual(result['device'],'cuda:0')
        self.assertIs(result['promotion'],False)

    def test_nonfinite_huge_or_nested_comparisons_do_not_break_api(self):
        result=dashboard.pretrained_benchmark(record(comparisons=[{'candidate':{}},
            {'candidate':'foundation','cost_multiplier':1,'net_return':10**400,'MDD':0,'trades':1}]))
        self.assertEqual(result['comparisons'],[])
