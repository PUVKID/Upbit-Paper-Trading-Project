"""GPU 실험 없이 행렬 재사용·유한 선행 준비·장치 경계를 검증합니다."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import engine


def fake_xgb():
    model = Mock(best_iteration=1, best_score=.4)
    model.save_config.return_value = json.dumps({"learner": {"generic_param": {"device": "cuda:0"}}})
    model.num_boosted_rounds.return_value = 2
    matrices, fits = [], []

    def matrix(data, **kwargs):
        result = SimpleNamespace(data=data.copy(), **kwargs)
        matrices.append(result)
        return result

    def train(params, data, **kwargs):
        for callback in kwargs["callbacks"]:
            if hasattr(callback, "before_training"):
                callback.before_training(model)
            if hasattr(callback, "before_iteration"):
                callback.before_iteration(model, 0, {})
        fits.append((params.copy(), data, kwargs["evals"][0][0]))
        kwargs["evals_result"]["tune"] = {"mlogloss": [.5, .4]}
        return model

    return SimpleNamespace(build_info=lambda: {"USE_CUDA": True}, DMatrix=matrix,
                           Booster=Mock(return_value=model), train=Mock(side_effect=train),
                           callback=SimpleNamespace(TrainingCallback=object), __version__="통제된-대역"), matrices, fits


class MatrixReuseTests(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(engine.DEFAULT_CONFIG)
        self.frame = pd.DataFrame({"x": np.arange(12), "label": np.arange(12) % 3},
                                  index=pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC"))
        self.train, self.tune = self.frame.iloc[:9], self.frame.iloc[9:]

    def test_candidate_scope_reuses_exact_matrices_without_changing_training_contract(self):
        fake, matrices, fits = fake_xgb()
        with patch.dict("sys.modules", {"xgboost": fake}), engine.gpu_matrix_scope():
            evidence = [engine.fit_gpu(self.train, self.tune, ["x"], d, self.config)[1] for d in (3, 5, 7)]
        self.assertEqual(len(matrices), 2)
        self.assertTrue(all(f[1] is fits[0][1] and f[2] is fits[0][2] for f in fits))
        self.assertEqual([f[0]["max_depth"] for f in fits], [3, 5, 7])
        np.testing.assert_array_equal(matrices[0].label, np.arange(9) % 3)
        expected_weights, _ = engine.training_weights(self.train, self.config["training"])
        np.testing.assert_array_equal(matrices[0].weight, expected_weights)
        self.assertEqual([e["matrix_reused"] for e in evidence], [False, True, True])
        self.assertEqual(fake.Booster.call_count, 3)
        for item in evidence:
            for name in ("matrix_prepare_seconds", "cuda_preflight_seconds", "fit_seconds"):
                self.assertGreaterEqual(item["timings"][name], 0)

    def test_direct_calls_and_new_scopes_do_not_keep_old_matrices(self):
        fake, matrices, _ = fake_xgb()
        with patch.dict("sys.modules", {"xgboost": fake}):
            for _ in range(2):
                engine.fit_gpu(self.train, self.tune, ["x"], 3, self.config)
            for _ in range(2):
                with engine.gpu_matrix_scope():
                    engine.fit_gpu(self.train, self.tune, ["x"], 3, self.config)
        self.assertEqual(len(matrices), 8)

    def test_different_frames_and_weighting_cannot_reuse_previous_market(self):
        fake, matrices, _ = fake_xgb()
        with patch.dict("sys.modules", {"xgboost": fake}), engine.gpu_matrix_scope():
            engine.fit_gpu(self.train, self.tune, ["x"], 3, self.config)
            changed = self.train.copy()
            changed["x"] += 100
            engine.fit_gpu(changed, self.tune, ["x"], 3, self.config)
            self.config["training"]["half_life_days"] = 10
            engine.fit_gpu(changed, self.tune, ["x"], 3, self.config)
        self.assertEqual(len(matrices), 6)
        self.assertEqual(matrices[2].data[0, 0], 100)

    def test_reused_matrices_still_reject_cpu_before_training(self):
        fake, _, fits = fake_xgb()
        with patch.dict("sys.modules", {"xgboost": fake}), engine.gpu_matrix_scope():
            engine.fit_gpu(self.train, self.tune, ["x"], 3, self.config)
            fake.Booster.return_value.save_config.return_value = json.dumps(
                {"learner": {"generic_param": {"device": "cpu"}}})
            with self.assertRaisesRegex(RuntimeError, "CPU 대체 금지"):
                engine.fit_gpu(self.train, self.tune, ["x"], 5, self.config)
        self.assertEqual(len(fits), 1)


class FeaturePrefetchTests(unittest.TestCase):
    def test_only_one_market_is_ahead_and_identical_bounds_are_preserved(self):
        config = {"markets": ["A", "B", "C"]}
        calls, ready = [], threading.Event()
        frames = {m: pd.DataFrame({"x": [i]}) for i, m in enumerate(config["markets"])}

        def load(store, market, cfg, start, end):
            calls.append((market, start, end))
            if market == "B":
                ready.set()
            return frames[market]

        with patch("engine.load_market_features", side_effect=load):
            with engine.market_feature_loader(None, config, "start", "end") as read:
                first, timing = read("A")
                self.assertIs(first, frames["A"])
                self.assertTrue(ready.wait(2), "현재 시장 처리 중 다음 시장 준비가 시작되어야 합니다")
                self.assertEqual([c[0] for c in calls], ["A", "B"])
                self.assertGreaterEqual(timing["feature_seconds"], 0)
                self.assertGreaterEqual(timing["feature_wait_seconds"], 0)
                self.assertIs(read("B")[0], frames["B"])
                self.assertIs(read("C")[0], frames["C"])
        self.assertEqual(calls, [("A", "start", "end"), ("B", "start", "end"), ("C", "start", "end")])

    def test_worker_error_is_not_swallowed_or_followed_by_more_market_reads(self):
        config = {"markets": ["A", "B", "C"]}
        calls = []

        def load(store, market, cfg, start, end):
            calls.append(market)
            if market == "B":
                raise ValueError("선행 특징 원본 오류")
            return pd.DataFrame({"x": [1]})

        with patch("engine.load_market_features", side_effect=load):
            with self.assertRaisesRegex(ValueError, "선행 특징 원본 오류"):
                with engine.market_feature_loader(None, config, 1, 2) as read:
                    read("A")
                    read("B")
        self.assertEqual(calls, ["A", "B"])

    def test_main_error_keeps_concurrent_worker_traceback_and_joins_worker(self):
        ready = threading.Event()

        def load(store, market, cfg, start, end):
            if market == "B":
                ready.set()
                raise ValueError("선행 실패 원문")
            return pd.DataFrame({"x": [1]})

        with patch("engine.load_market_features", side_effect=load):
            with self.assertRaisesRegex(RuntimeError, "현재 시장 실패") as caught:
                with engine.market_feature_loader(None, {"markets": ["A", "B", "C"]}, 1, 2) as read:
                    read("A")
                    self.assertTrue(ready.wait(2))
                    raise RuntimeError("현재 시장 실패")
        self.assertIn("선행 실패 원문", "\n".join(caught.exception.__notes__))
        self.assertFalse(any(t.name.startswith("upbit-features") for t in threading.enumerate()))


@unittest.skipIf(sys.platform == "win32", "SQLite 검사는 Linux에서만 실행합니다")
class CandleReadThroughputTests(unittest.TestCase):
    def test_close_index_order_keeps_range_limit_and_ascending_result(self):
        with tempfile.TemporaryDirectory() as temp:
            store = engine.CandleStore(Path(temp) / "candles.sqlite3")
            index = pd.date_range("2026-01-01", periods=8, freq="5min", tz="UTC")
            frame = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100., "volume": 1.}, index=index)
            store.save_page("A", 5, frame)
            store.save_page("B", 5, frame)
            result = store.read("A", 5, start=index[2], end=index[6], limit=3)
            pd.testing.assert_frame_equal(result, frame.iloc[3:6].rename_axis("open_time"), check_freq=False)
            queries = []
            actual_read = pd.read_sql_query

            def measured_read(sql, db, **kwargs):
                queries.append(db.execute("EXPLAIN QUERY PLAN " + sql, kwargs["params"]).fetchall())
                return actual_read(sql, db, **kwargs)

            with patch("engine.pd.read_sql_query", side_effect=measured_read):
                store.read("A", 5, start=index[2], end=index[6], limit=3)
            self.assertFalse(any("TEMP B-TREE" in row[3] for row in queries[0]), queries)

    def test_corrupt_close_time_is_rejected_instead_of_silently_reordered(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "candles.sqlite3"
            store = engine.CandleStore(path)
            with engine.connect(path) as db:
                db.execute("INSERT INTO candles VALUES(?,?,?,?,?,?,?,?,?)",
                           ("A", "5", "2026-01-01T00:00:00+00:00", "2026-01-01T00:07:00+00:00", 100., 101., 99., 100., 1.))
            with self.assertRaisesRegex(ValueError, "캔들 종료 시각"):
                store.read("A", 5)

    def test_train_cycle_uses_one_pair_per_market_and_records_stage_times(self):
        from test_engine import candles
        from test_research import research_config

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = research_config()
            config["training"].update(candidate_depths=[3, 5, 7], min_split_rows=10)
            for key in ("data_dir", "models_dir", "reports_dir"):
                config[key] = str(root / key)
            raw = candles(10000, start="2026-09-20")
            price = 100000 * (1 + .025 * np.sin(np.arange(len(raw)) / 10))
            raw["open"] = raw["close"] = price
            raw["high"], raw["low"] = price + 100, price - 100
            features = engine.build_features({5: raw})
            fake, matrices, fits = fake_xgb()
            actual_utc = engine.utc

            def clock(value=None):
                return actual_utc("2026-10-24T00:00:00Z" if value is None else value)

            def predict(model, frame, columns, best_iteration):
                return np.tile([.05, .9, .05], (len(frame), 1))

            def policy(frame, probabilities, settings, market, minimum, horizon):
                return {"confidence": 1.01, "edge": 1.01, "eligible": False,
                        "horizon_minutes": horizon, "trials": []}

            with patch.dict("sys.modules", {"xgboost": fake}), patch.object(engine, "ROOT", root), \
                 patch("engine.utc", side_effect=clock), patch("engine.CandleStore"), \
                 patch("engine.load_market_features", return_value=features), \
                 patch("engine.predict_booster", side_effect=predict), patch("engine.tune_rule", side_effect=policy):
                report = engine.train_cycle(config, live=True)
            self.assertEqual(len(matrices), 10)
            self.assertEqual(len(fits), 15)
            self.assertFalse(report["promotion"]["promote"])
            self.assertFalse((Path(config["models_dir"]) / "current.json").exists())
            self.assertEqual(report["locked_test"]["status"], "미평가")
            for market in engine.MARKETS:
                item = report["markets"][market]
                self.assertEqual([e["matrix_reused"] for e in item["gpu"]], [False, True, True])
                for name in ("feature_seconds", "feature_wait_seconds", "label_split_audit_seconds",
                             "fit_and_tune_predict_seconds", "calibration_policy_validation_seconds"):
                    self.assertGreaterEqual(item["timings"][name], 0)


if __name__ == "__main__":
    unittest.main()
