# 전체 엔진 차이

직전 공개 기준과 보존 엔진은 논리적 텍스트가 같으며 개행 차이는 별도 해시로 구분합니다. 아래는 원본 엔진과 이번 엔진의 전체 텍스트 차이입니다. 추가 회귀 파일은 신규 파일 전체가 변경분입니다. 진단용 Git 경고도 출력에 포함될 수 있습니다.

```diff
warning: in the working copy of 'evidence/engine-gpu-baseline.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'engine.py', LF will be replaced by CRLF the next time Git touches it
diff --git before/evidence/engine-gpu-baseline.py after/engine.py
index bf51070..cbf2ffe 100644
--- before/evidence/engine-gpu-baseline.py
+++ after/engine.py
@@ -3,6 +3,8 @@ from __future__ import annotations
 
 import argparse
 import contextlib
+from concurrent.futures import ThreadPoolExecutor
+from contextvars import ContextVar
 import copy
 import hashlib
 import itertools
@@ -13,6 +15,7 @@ from pathlib import Path
 import sqlite3
 import sys
 import time
+import traceback
 import urllib.error
 import urllib.parse
 import urllib.request
@@ -224,12 +227,16 @@ class CandleStore:
             args.append(iso(end))
         args.append(int(limit))
         with connect(self.path) as db:
-            frame = pd.read_sql_query("SELECT open_time,open,high,low,close,volume FROM candles WHERE " +
-                                      " AND ".join(conditions) + " ORDER BY open_time DESC LIMIT ?", db, params=args)
+            # 고정 시간봉은 종료 시각 순서도 같으므로 범위 인덱스를 그대로 활용합니다.
+            frame = pd.read_sql_query("SELECT open_time,close_time,open,high,low,close,volume FROM candles WHERE " +
+                                      " AND ".join(conditions) + " ORDER BY close_time DESC LIMIT ?", db, params=args)
         if frame.empty:
             return pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                                 index=pd.DatetimeIndex([], tz="UTC", name="open_time"))
         frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
+        close_times = pd.to_datetime(frame.pop("close_time"), utc=True)
+        if not close_times.eq(frame["open_time"] + pd.Timedelta(minutes=minutes(tf))).all():
+            raise ValueError(f"{market}/{tf}: 캔들 종료 시각이 시간봉 간격과 다릅니다")
         return frame.set_index("open_time").sort_index()
 
     def summary(self):
@@ -888,6 +895,20 @@ def assert_cuda_config(config):
     return device
 
 
+_gpu_matrices = ContextVar("gpu_matrices", default=None)
+
+
+@contextlib.contextmanager
+def gpu_matrix_scope():
+    # 같은 시장의 읽기 전용 후보 묶음 안에서만 행렬을 보존합니다.
+    token = _gpu_matrices.set({})
+    try:
+        yield
+    finally:
+        _gpu_matrices.get().clear()
+        _gpu_matrices.reset(token)
+
+
 def fit_gpu(train, tune, columns, depth, config, on_progress=None):
     started = time.monotonic()
     import xgboost as xgb
@@ -898,14 +919,28 @@ def fit_gpu(train, tune, columns, depth, config, on_progress=None):
               "eta": TRAINING_ETA, "subsample": 0.9, "colsample_bytree": 0.9,
               "max_bin": 128, "seed": 42, "nthread": 4}
     tr = config["training"]
-    weights, weighting_evidence = training_weights(train, tr)
-    dtrain = xgb.DMatrix(train[columns].to_numpy(dtype=np.float32), label=train.label.to_numpy(dtype=int),
-                         weight=weights, feature_names=columns)
-    dtune = xgb.DMatrix(tune[columns].to_numpy(dtype=np.float32), label=tune.label.to_numpy(dtype=int), feature_names=columns)
+    matrix_started = time.monotonic()
+    cache = _gpu_matrices.get()
+    signature = (tuple(columns), tr["half_life_days"], tr["class_weight_mode"], tr["class_weight_cap"])
+    reused = (cache is not None and cache.get("train") is train and cache.get("tune") is tune
+              and cache.get("signature") == signature)
+    if reused:
+        dtrain, dtune, weighting_evidence = cache["matrices"]
+    else:
+        weights, weighting_evidence = training_weights(train, tr)
+        dtrain = xgb.DMatrix(train[columns].to_numpy(dtype=np.float32), label=train.label.to_numpy(dtype=int),
+                             weight=weights, feature_names=columns)
+        dtune = xgb.DMatrix(tune[columns].to_numpy(dtype=np.float32), label=tune.label.to_numpy(dtype=int), feature_names=columns)
+        if cache is not None:
+            cache.clear()
+            cache.update(train=train, tune=tune, signature=signature, matrices=(dtrain, dtune, weighting_evidence))
+    matrix_seconds = time.monotonic() - matrix_started
+    preflight_started = time.monotonic()
     # save_config는 실제 CUDA 컨텍스트를 구성하므로 학습 0회 상태에서 CPU 전환을 검출합니다.
     preflight = xgb.Booster(params, cache=[dtrain, dtune])
     assert_cuda_config(preflight.save_config())
     del preflight
+    preflight_seconds = time.monotonic() - preflight_started
 
     class CudaOnly(xgb.callback.TrainingCallback):
         def before_training(self, model):
@@ -926,9 +961,11 @@ def fit_gpu(train, tune, columns, depth, config, on_progress=None):
                     on_progress(rounds, float(evals_log["tune"]["mlogloss"][-1]))
                 return False
         callbacks.append(MeasuredProgress())
+    fit_started = time.monotonic()
     booster = xgb.train(params, dtrain, num_boost_round=tr["rounds"], evals=[(dtune, "tune")],
                         early_stopping_rounds=min(30, max(2, tr["rounds"] // 5)), verbose_eval=False,
                         callbacks=callbacks, evals_result=history)
+    fit_seconds = time.monotonic() - fit_started
     device = assert_cuda_config(booster.save_config())
     boosted_rounds = int(booster.num_boosted_rounds())
     last_loss = float(history["tune"]["mlogloss"][-1])
@@ -936,11 +973,15 @@ def fit_gpu(train, tune, columns, depth, config, on_progress=None):
         on_progress(boosted_rounds, last_loss)
     return booster, {"actual_device": device, "best_iteration": int(booster.best_iteration),
                      "requested_device": "cuda:0", "depth": depth, "xgboost_version": xgb.__version__,
+                     "parameters": params.copy(),
                      "eta": params["eta"],
                      "duration_seconds": time.monotonic() - started, "boosted_rounds": boosted_rounds,
                      "validation_logloss": float(booster.best_score), "last_validation_logloss": last_loss,
                      "validation_scope": "tune", "train_rows": len(train), "tune_rows": len(tune),
-                      "features": len(columns), "weighting": weighting_evidence}
+                      "features": len(columns), "weighting": weighting_evidence,
+                      "matrix_reused": reused,
+                      "timings": {"matrix_prepare_seconds": matrix_seconds,
+                                  "cuda_preflight_seconds": preflight_seconds, "fit_seconds": fit_seconds}}
 
 
 def predict_booster(booster, frame, columns, best_iteration):
@@ -1153,6 +1194,50 @@ def assess_promotion(config, frames, probabilities, rules):
                 "comparison_error": {"type": type(exc).__name__, "message": str(exc)}, "comparison_periods": periods}
 
 
+@contextlib.contextmanager
+def market_feature_loader(store, config, start, end):
+    # 완료 결과도 다음 시장 하나만 보존하며 DB 연결은 각 read 내부에서 닫습니다.
+    markets = iter(config["markets"])
+    pending = None
+    expected = None
+
+    def prepare(market):
+        started = time.monotonic()
+        frame = load_market_features(store, market, config, start, end)
+        return frame, time.monotonic() - started
+
+    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="upbit-features") as executor:
+        def schedule():
+            nonlocal pending, expected
+            expected = next(markets, None)
+            pending = executor.submit(prepare, expected) if expected is not None else None
+
+        def read(market):
+            nonlocal pending
+            if pending is None or market != expected:
+                raise ValueError("특징 선행 준비의 시장 순서가 다릅니다")
+            current, pending = pending, None
+            waited = time.monotonic()
+            frame, seconds = current.result()
+            wait_seconds = time.monotonic() - waited
+            schedule()
+            return frame, {"feature_seconds": seconds, "feature_wait_seconds": wait_seconds}
+
+        schedule()
+        try:
+            yield read
+        finally:
+            if pending is not None and not pending.cancel():
+                active_error = sys.exc_info()[1]
+                try:
+                    pending.result()
+                except Exception:
+                    if active_error is None:
+                        raise
+                    # 현재 시장 오류를 보존하면서 동시에 발생한 선행 오류 원문도 남깁니다.
+                    active_error.add_note("선행 특징 준비 실패:\n" + traceback.format_exc())
+
+
 def train_cycle(config, final=False, live=False):
     started = time.monotonic()
     started_at = iso(None)
@@ -1192,83 +1277,91 @@ def train_cycle(config, final=False, live=False):
                            "fold_id": stable_hash(report_plan),
                            "periods": report_plan},
               "markets": {}, "training": {"status": "학습 중", "gpu_required": True, "started_at": started_at}, "rounds": []}
-    for market in config["markets"]:
-        update_progress(status="running", stage="features", market=market, candidate=None,
-                 completed_candidates=completed, total_candidates=total, error=None)
-        raw = load_market_features(store, market, config, plan["train"][0], plan["validation"][1])
-        labelled = label_frame(raw, tr["horizon"], tr["label_threshold"], config["paper"]["fee"], config["paper"]["slippage"])
-        cols = feature_columns(labelled)
-        if columns is not None and cols != columns:
-            raise ValueError("코인별 피처 스키마가 다릅니다")
-        columns = cols
-        dropped_features = int(labelled[cols].isna().any(axis=1).sum())
-        labelled = labelled.dropna(subset=cols + ["label", "label_end"])
-        split = research_splits(labelled, plan, config, consumed)
-        forbidden = protected_periods(config, consumed)
-        for name in split:
-            for left, right in zip([None] + [b for a, b in forbidden], [a for a, b in forbidden] + [None]):
-                segment = split[name]
-                segment = segment[segment.index >= left] if left is not None else segment[segment.label_end < right]
-                if right is not None:
-                    segment = segment[segment.label_end < right]
-                if len(segment):
-                    audit.expose(run, segment.index.min(), segment.label_end.max(), name)
-        if set(split["train"].label.astype(int)) != {0, 1, 2}:
-            raise ValueError(f"{market}: 학습 구간에 세 클래스가 모두 필요합니다")
-        models, gpu_evidence, tune_probs, candidate_names = [], [], [], []
-        for idx, depth in enumerate(tr["candidate_depths"]):
-            def fit_progress(rounds, loss):
-                update_progress(status="running", stage="fit", market=market, candidate=idx + 1,
-                                completed_candidates=completed, total_candidates=total, error=None,
-                                rows=len(split["train"]), tune_rows=len(split["tune"]), features=len(cols),
-                                current_round=rounds, target_rounds=tr["rounds"], validation_logloss=loss,
-                                learning_rate=TRAINING_ETA)
-            fit_progress(0, None)
-            model, evidence = fit_gpu(split["train"], split["tune"], cols, depth, config, on_progress=fit_progress)
-            filename = market + f"-{idx}.ubj"
-            model.save_model(directory / filename)
-            models.append(model)
-            gpu_evidence.append(evidence)
-            report["rounds"].append({"market": market, "candidate": idx + 1, **evidence})
-            candidate_names.append(filename)
-            tune_probs.append(predict_booster(model, split["tune"], cols, evidence["best_iteration"]))
-            completed += 1
-        weights = ensemble_weights(tune_probs, split["tune"].label)
-        calibration = split["calibration"]
-        update_progress(status="running", stage="calibration", market=market, candidate=None,
-                        completed_candidates=completed, total_candidates=total, error=None,
-                        rows=len(calibration), features=len(cols))
-        mixed = sum(w * predict_booster(m, calibration, cols, e["best_iteration"]) for w, m, e in zip(weights, models, gpu_evidence))
-        temperature = min((0.75, 1.0, 1.25, 1.5, 2.0), key=lambda t: logloss(calibration.label, temperature_probs(mixed, t)))
-        calibrated = temperature_probs(mixed, temperature)
-        policy_frame, validation = split["rule"], split["validation"]
-
-        def probabilities(frame):
-            return temperature_probs(sum(w * predict_booster(m, frame, cols, e["best_iteration"])
-                                         for w, m, e in zip(weights, models, gpu_evidence)), temperature)
-
-        policy_probs, evaluated = probabilities(policy_frame), probabilities(validation)
-        rule = tune_rule(policy_frame, policy_probs, config["paper"], market, tr["min_closed_trades"],
-                         tr["horizon"] * minutes(config["base_timeframe"]))
-        rule.update(net_label_threshold=tr["label_threshold"], label_cost_factor=roundtrip_cost_factor(config["paper"]))
-        trials = rule.pop("trials")
-        majority = int(split["train"].label.astype(int).value_counts().idxmax())
-        metrics = classification_metrics(validation.label, evaluated, majority)
-        metrics.update(start=iso(validation.index.min()), end=iso(validation.index.max()))
-        diagnostics = {**probability_diagnostics(evaluated, rule), "scope": "validation_after_policy_selection",
-                       "rule_trials": trials, "weighting": training_weights(split["train"], tr)[1],
-                       "calibration": classification_metrics(calibration.label, calibrated, majority),
-                       "policy_selection": probability_diagnostics(policy_probs, rule),
-                       "feature_rows_dropped": dropped_features, "policy_eligible": rule["eligible"]}
-        report["markets"][market] = {"validation": metrics, "splits": {
-            k: {"rows": len(v), "start": iso(v.index.min()), "end": iso(v.index.max()),
-                "class_distribution": np.bincount(v.label.astype(int), minlength=3).tolist()} for k, v in split.items()},
-            "gpu": gpu_evidence, "rule": rule, "weights": weights.tolist(), "temperature": temperature,
-            "diagnostics": diagnostics}
-        packages[market] = {"models": candidate_names, "gpu": gpu_evidence, "weights": weights.tolist(),
-                            "temperature": temperature, "rule": rule, "majority_class": majority}
-        validation_frames[market], validation_probs[market], rules[market] = validation, evaluated, rule
-        del raw, labelled, split, models
+    with market_feature_loader(store, config, plan["train"][0], plan["validation"][1]) as read_market:
+        for market in config["markets"]:
+            update_progress(status="running", stage="features", market=market, candidate=None,
+                     completed_candidates=completed, total_candidates=total, error=None)
+            raw, stage_times = read_market(market)
+            stage_started = time.monotonic()
+            labelled = label_frame(raw, tr["horizon"], tr["label_threshold"], config["paper"]["fee"], config["paper"]["slippage"])
+            cols = feature_columns(labelled)
+            if columns is not None and cols != columns:
+                raise ValueError("코인별 피처 스키마가 다릅니다")
+            columns = cols
+            dropped_features = int(labelled[cols].isna().any(axis=1).sum())
+            labelled = labelled.dropna(subset=cols + ["label", "label_end"])
+            split = research_splits(labelled, plan, config, consumed)
+            forbidden = protected_periods(config, consumed)
+            for name in split:
+                for left, right in zip([None] + [b for a, b in forbidden], [a for a, b in forbidden] + [None]):
+                    segment = split[name]
+                    segment = segment[segment.index >= left] if left is not None else segment[segment.label_end < right]
+                    if right is not None:
+                        segment = segment[segment.label_end < right]
+                    if len(segment):
+                        audit.expose(run, segment.index.min(), segment.label_end.max(), name)
+            if set(split["train"].label.astype(int)) != {0, 1, 2}:
+                raise ValueError(f"{market}: 학습 구간에 세 클래스가 모두 필요합니다")
+            models, gpu_evidence, tune_probs, candidate_names = [], [], [], []
+            stage_times["label_split_audit_seconds"] = time.monotonic() - stage_started
+            fit_stage_started = time.monotonic()
+            with gpu_matrix_scope():
+                for idx, depth in enumerate(tr["candidate_depths"]):
+                    def fit_progress(rounds, loss):
+                        update_progress(status="running", stage="fit", market=market, candidate=idx + 1,
+                                        completed_candidates=completed, total_candidates=total, error=None,
+                                        rows=len(split["train"]), tune_rows=len(split["tune"]), features=len(cols),
+                                        current_round=rounds, target_rounds=tr["rounds"], validation_logloss=loss,
+                                        learning_rate=TRAINING_ETA)
+                    fit_progress(0, None)
+                    model, evidence = fit_gpu(split["train"], split["tune"], cols, depth, config, on_progress=fit_progress)
+                    filename = market + f"-{idx}.ubj"
+                    model.save_model(directory / filename)
+                    models.append(model)
+                    gpu_evidence.append(evidence)
+                    report["rounds"].append({"market": market, "candidate": idx + 1, **evidence})
+                    candidate_names.append(filename)
+                    tune_probs.append(predict_booster(model, split["tune"], cols, evidence["best_iteration"]))
+                    completed += 1
+            stage_times["fit_and_tune_predict_seconds"] = time.monotonic() - fit_stage_started
+            evaluation_started = time.monotonic()
+            weights = ensemble_weights(tune_probs, split["tune"].label)
+            calibration = split["calibration"]
+            update_progress(status="running", stage="calibration", market=market, candidate=None,
+                            completed_candidates=completed, total_candidates=total, error=None,
+                            rows=len(calibration), features=len(cols))
+            mixed = sum(w * predict_booster(m, calibration, cols, e["best_iteration"]) for w, m, e in zip(weights, models, gpu_evidence))
+            temperature = min((0.75, 1.0, 1.25, 1.5, 2.0), key=lambda t: logloss(calibration.label, temperature_probs(mixed, t)))
+            calibrated = temperature_probs(mixed, temperature)
+            policy_frame, validation = split["rule"], split["validation"]
+
+            def probabilities(frame):
+                return temperature_probs(sum(w * predict_booster(m, frame, cols, e["best_iteration"])
+                                             for w, m, e in zip(weights, models, gpu_evidence)), temperature)
+
+            policy_probs, evaluated = probabilities(policy_frame), probabilities(validation)
+            rule = tune_rule(policy_frame, policy_probs, config["paper"], market, tr["min_closed_trades"],
+                             tr["horizon"] * minutes(config["base_timeframe"]))
+            rule.update(net_label_threshold=tr["label_threshold"], label_cost_factor=roundtrip_cost_factor(config["paper"]))
+            trials = rule.pop("trials")
+            majority = int(split["train"].label.astype(int).value_counts().idxmax())
+            metrics = classification_metrics(validation.label, evaluated, majority)
+            metrics.update(start=iso(validation.index.min()), end=iso(validation.index.max()))
+            diagnostics = {**probability_diagnostics(evaluated, rule), "scope": "validation_after_policy_selection",
+                           "rule_trials": trials, "weighting": training_weights(split["train"], tr)[1],
+                           "calibration": classification_metrics(calibration.label, calibrated, majority),
+                           "policy_selection": probability_diagnostics(policy_probs, rule),
+                           "feature_rows_dropped": dropped_features, "policy_eligible": rule["eligible"]}
+            stage_times["calibration_policy_validation_seconds"] = time.monotonic() - evaluation_started
+            report["markets"][market] = {"validation": metrics, "splits": {
+                k: {"rows": len(v), "start": iso(v.index.min()), "end": iso(v.index.max()),
+                    "class_distribution": np.bincount(v.label.astype(int), minlength=3).tolist()} for k, v in split.items()},
+                "gpu": gpu_evidence, "rule": rule, "weights": weights.tolist(), "temperature": temperature,
+                "diagnostics": diagnostics, "timings": stage_times}
+            packages[market] = {"models": candidate_names, "gpu": gpu_evidence, "weights": weights.tolist(),
+                                "temperature": temperature, "rule": rule, "majority_class": majority}
+            validation_frames[market], validation_probs[market], rules[market] = validation, evaluated, rule
+            del raw, labelled, split, models
     report["validation"] = {"scope": "정책 선정 이후 시간순 후속 평가; 반복 연구이며 잠금 시험 아님", "portfolio":
                              backtest_portfolio(validation_frames, validation_probs, rules, config["paper"])}
     report["promotion"] = assess_promotion(config, validation_frames, validation_probs, rules)

```

