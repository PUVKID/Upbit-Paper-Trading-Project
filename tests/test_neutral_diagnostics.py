"""중립 편향의 학습 가중치와 확률 보정 진단 회귀검사."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import engine
from test_engine import candles
from test_research import research_config


class NeutralDiagnosticTests(unittest.TestCase):
    def test_weighted_class_mass_exposes_recency_without_changing_fit_weights(self):
        frame = pd.DataFrame({"label": [0, 1, 1, 2]},
                             index=pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC"))
        config = research_config()
        config["training"].update(class_weight_mode="recency", half_life_days=1)
        weights, evidence = engine.training_weights(frame, config["training"])
        np.testing.assert_allclose(weights, [4 / 15, 8 / 15, 16 / 15, 32 / 15])
        self.assertIn("effective_class_distribution", evidence,
                      "원시 표본 수만 있고 실제 적합 가중치의 클래스 비중이 없습니다")
        np.testing.assert_allclose(evidence["effective_class_distribution"], [1 / 15, 6 / 15, 8 / 15])
        np.testing.assert_allclose(evidence["class_weight_sums"], [4 / 15, 24 / 15, 32 / 15])
        self.assertEqual(evidence["scope"], "train_only")
        json.dumps(evidence, allow_nan=False)

    def test_stage_probabilities_include_only_bounded_extra_training_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = research_config()
            config["training"].update(candidate_depths=[3], min_split_rows=10)
            for key in ("data_dir", "models_dir", "reports_dir"):
                config[key] = str(root / "work" / key)
            raw = candles(10000, start="2026-09-20")
            prices = 100000 * (1 + .025 * np.sin(np.arange(len(raw)) / 10))
            raw["open"] = raw["close"] = prices
            raw["high"], raw["low"] = prices + 100, prices - 100
            features = engine.build_features({5: raw})
            actual_utc = engine.utc
            prediction_rows = []

            def fixture_clock(value=None):
                return actual_utc("2026-10-24T00:00:00Z" if value is None else value)

            def predict(model, frame, columns, best_iteration):
                prediction_rows.append(len(frame))
                return np.tile([.1, .8, .1], (len(frame), 1))

            def policy(frame, probabilities, settings, market, minimum, horizon):
                return {"confidence": 1.01, "edge": 1.01, "eligible": False,
                        "horizon_minutes": horizon, "trials": []}

            with patch.object(engine, "ROOT", root), patch("engine.CandleStore"), \
                 patch("engine.load_market_features", return_value=features), \
                 patch("engine.fit_gpu", return_value=(Mock(), {"best_iteration": 1, "actual_device": "cuda:0"})), \
                 patch("engine.predict_booster", side_effect=predict), \
                 patch("engine.tune_rule", side_effect=policy), patch("engine.utc", side_effect=fixture_clock):
                report = engine.train_cycle(config, live=True)
            self.assertEqual(len(prediction_rows), len(engine.MARKETS) * 5)
            self.assertEqual(prediction_rows.count(2048), len(engine.MARKETS))
            self.assertFalse(report["promotion"]["promote"])
            self.assertFalse((Path(config["models_dir"]) / "current.json").exists())
            for market in engine.MARKETS:
                result = report["markets"][market]
                self.assertIn("probability_stages", result["diagnostics"],
                              "튜닝·보정 전후 확률을 분리한 진단이 없습니다")
                stages = result["diagnostics"]["probability_stages"]
                self.assertEqual(set(stages), {"train", "tune", "calibration_before", "calibration_after"})
                for name, stage in stages.items():
                    split = name if name in {"train", "tune"} else "calibration"
                    rows = min(2048, result["splits"][split]["rows"]) if name == "train" else result["splits"][split]["rows"]
                    self.assertEqual(stage["rows"], rows)
                    self.assertEqual(sum(stage["class_distribution"]), rows)
                    if name != "train":
                        self.assertEqual(stage["class_distribution"], result["splits"][split]["class_distribution"])
                        self.assertEqual(stage["sample_scope"], "full_split")
                    else:
                        self.assertEqual(stage["sample_scope"], "recent_training_tail")
                        self.assertEqual(stage["split_rows"], result["splits"]["train"]["rows"])
                        self.assertEqual(stage["end"], result["splits"]["train"]["end"])
                    self.assertEqual(stage["argmax_counts"], [0, rows, 0])
                    self.assertEqual(stage["signal_counts"], [0, rows, 0])
                    self.assertEqual(stage["selection_scope"], "diagnostic_only")
                    self.assertTrue(np.isfinite(stage["logloss"]))
                self.assertAlmostEqual(stages["tune"]["probability_quantiles"]["2"]["max"], .1)
                self.assertAlmostEqual(stages["train"]["probability_quantiles"]["2"]["max"], .1)
                self.assertAlmostEqual(stages["calibration_before"]["probability_quantiles"]["2"]["max"], .1)
                expected_up = .1 ** (1 / result["temperature"]) / (
                    2 * .1 ** (1 / result["temperature"]) + .8 ** (1 / result["temperature"]))
                self.assertAlmostEqual(stages["calibration_after"]["probability_quantiles"]["2"]["max"], expected_up)
            json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
