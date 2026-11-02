"""수익 예측 연구의 제한된 상태와 비교 결과만 공개하는지 검사합니다."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import dashboard


def fixture(**changes):
    value = {'schema_version':1, 'state':'running', 'stage':'fit', 'run':'return-test',
             'fit_calls':3, 'completed_fits':2, 'max_fits':30, 'fold_index':1, 'fold_total':2,
             'market':'KRW-BTC', 'objective':'net_forward_return', 'promotion':False,
             'candidate_ready':False, 'evaluation_performed':False, 'updated_utc':'2026-11-03T00:00:00Z', 'comparisons':[]}
    value.update(changes)
    return value


class ReturnResearchDashboardTests(unittest.TestCase):
    def view(self, value):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root/'10차/work/status.json'
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps(value), encoding='utf-8')
            return dashboard.SnapshotAdapter(root).status()['return_research']

    def test_missing_record_is_not_a_started_experiment(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(dashboard.SnapshotAdapter(Path(folder)).status()['return_research'], {})

    def test_actual_runner_failure_and_candidate_names_are_visible(self):
        result = self.view(fixture(state='error', stage='failed', comparisons=[
            {'fold':1, 'candidate':name, 'cost_multiplier':1, 'net_return':0, 'MDD':0, 'trades':0}
            for name in ('plain', 'selected')]))
        self.assertTrue(result['has_error'])
        self.assertEqual([row['candidate'] for row in result['comparisons']], ['plain', 'selected'])

    def test_unknown_keys_paths_and_error_text_never_escape(self):
        result = self.view(fixture(error='/private/token', checkpoint='/private/model', argv=['secret']))
        self.assertEqual(result['completed_fits'], 2)
        self.assertEqual(result['max_fits'], 30)
        self.assertTrue(result['has_error'])
        self.assertNotIn('private', json.dumps(result))
        self.assertNotIn('secret', json.dumps(result))

    def test_core_invalid_types_budgets_and_promotion_are_rejected(self):
        for change in ({'fit_calls':True}, {'fit_calls':31}, {'completed_fits':4}, {'max_fits':31},
                       {'fold_total':3}, {'fold_index':3}, {'run':'/private/path'}, {'promotion':True},
                       {'objective':'price_accuracy'}, {'state':{'secret':True}}, {'schema_version':True}):
            with self.subTest(change=change):
                self.assertEqual(self.view(fixture(**change)), {})

    def test_complete_is_research_only_and_has_no_accuracy_score(self):
        value = fixture(state='complete', stage='complete', fit_calls=30, completed_fits=30, fold_index=2,
                        accuracy=1, promoted=True, evaluation_performed=True)
        result = self.view(value)
        self.assertEqual(result['state'], 'complete')
        self.assertIs(result['promotion'], False)
        self.assertNotIn('accuracy', result)
        self.assertNotIn('promoted', result)

    def test_finished_fits_without_evaluation_cannot_be_complete(self):
        self.assertEqual(self.view(fixture(state='complete', stage='complete', fit_calls=30, completed_fits=30,
                                          fold_index=2, evaluation_performed=False)), {})

    def test_completed_unapplied_candidate_is_distinct_from_training_error(self):
        result = self.view(fixture(state='complete', stage='complete', fit_calls=30, completed_fits=30,
            fold_index=2, evaluation_performed=True, research_review_eligible=False, reason='research_review_rejected'))
        self.assertIs(result['candidate_ready'], False)
        self.assertNotIn('reason', result)
        self.assertFalse(result['has_error'])

    def test_nested_market_or_candidate_values_do_not_break_the_api(self):
        result = self.view(fixture(market={'private':'secret'}, comparisons=[{'candidate':{'private':'secret'}}]))
        self.assertIsNone(result['market'])
        self.assertEqual(result['comparisons'], [])

    def test_compact_comparisons_keep_only_valid_measured_rows(self):
        row = {'fold':1, 'candidate':'mean', 'cost_multiplier':2, 'net_return':-.01, 'MDD':.02,
               'trades':7, 'scope':'retrospective_research', 'model_path':'/secret'}
        result = self.view(fixture(comparisons=[row, {**row,'net_return':float('nan')}, {**row,'candidate':'/secret'}]))
        self.assertEqual(len(result['comparisons']), 1)
        self.assertEqual(result['comparisons'][0]['net_return'], -.01)
        self.assertNotIn('model_path', result['comparisons'][0])

    def test_raw_string_reason_and_bad_timestamp_are_not_public(self):
        result = self.view(fixture(reason='/private/secret', updated_utc='/private/time', market='/private/market'))
        self.assertNotIn('private', json.dumps(result))
        self.assertIsNone(result['updated_utc'])
        self.assertIsNone(result['market'])


if __name__ == '__main__':
    unittest.main()
