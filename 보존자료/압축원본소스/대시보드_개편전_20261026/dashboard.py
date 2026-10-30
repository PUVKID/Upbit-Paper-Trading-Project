"""로컬 전용 읽기 전용 현물 모의거래 대시보드입니다."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import heapq
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent
MAX_JSON_BYTES = 4 * 1024 * 1024
COINS = ("BTC", "SOL", "ETH", "XRP")
# 알 수 없는 필드와 실행 인자·오류 원문은 전달하지 않습니다.
FIELDS = {
    "account": "initial_capital cash equity realized_pnl unrealized_pnl total_pnl return_pct mdd_pct updated_utc quote_utc started_utc entry_guard model_run",
    "assets": "symbol quantity average_price price market_value unrealized_pnl quote_utc",
    "training": "state stage market round candidate candidate_index candidate_total progress gpu_name gpu_memory_used_mb gpu_memory_total_mb gpu_utilization_pct gpu_verified started_utc finished_utc updated_utc recorded_utc progress_started_utc duration_seconds elapsed_seconds rows tune_rows features current_round target_rounds validation_logloss period_start period_end samples timeframe model_id run learning_rate",
    "metrics": "accuracy macro_f1 balanced_accuracy baseline_accuracy samples period_start period_end",
    "class_distribution": "down flat up short neutral long sell hold buy 0 1 2",
    "data": "state total_rows total_bytes updated_utc period_start period_end progress completed_requests total_requests missing_count",
    "datasets": "symbol timeframe rows bytes period_start period_end updated_utc missing_count state progress backfill_exhausted",
    "trades": "timestamp symbol side quantity price fee realized_pnl reason",
    "equity": "timestamp equity cash",
    "rounds": "run round candidate state accuracy macro_f1 balanced_accuracy baseline_accuracy samples duration_seconds started_utc finished_utc recorded_utc split locked_accuracy locked_samples",
    "service": "state start_utc started_utc checked_utc finished_utc failed_utc stopped_utc iteration consecutive_failures exit_code previous_finished_utc success_gap_seconds successful_cycles next_iteration interval_seconds duration_seconds",
    "gpu": "state checked_utc gpu_name gpu_memory_total_mb gpu_memory_used_mb gpu_utilization_pct",
    "split": "rows period_start period_end",
    "gpu_fit": "actual_device requested_device best_iteration depth xgboost_version boosted_rounds duration_seconds validation_logloss last_validation_logloss validation_scope train_rows tune_rows features eta",
    "test_state": "status used_for_selection",
    "portfolio": "net_return MDD trades equity start end common_rows fees_paid",
    "model": "run",
    "promotion": "promote reason incumbent_run comparison_baseline excluded_incumbent_run excluded_incumbent_reason probability_comparable",
    "promotion_metrics": "logloss net_return MDD",
    "health": "state checked_utc checked_datasets total_rows errors_count pending_count",
}

# 연구 스키마가 아직 없는 항목은 만들지 않으며 알려진 키만 재귀 투영합니다.
DIAGNOSTIC_KEYS = frozenset("""
experiment_index experiment_id experiment candidate candidate_id candidate_index candidates
sequence candidate_total horizon_minutes class_weight_mode scope mode class_counts class_factors cap half_life_days
argmax_counts signal_counts filter_reasons no_trade_policy buy_confidence_below buy_edge_below neutral_signal
trials trading objective min_closed_trades validation_objective validation_trading selection_scope
rule_trials weighting feature_rows_dropped policy_eligible deployment eligibility schema_version
temperature weights candidate_objective incumbent_objective probability_comparable
logloss_not_worse trading_not_worse comparison_periods
executions realized_pnl exit_reasons stop_loss take_profit time_exit common_rows valuation fill
selected_candidate selected candidate_count completed_candidates total_candidates status stage
diagnostics research train tune calibration validation policy_selection forward_evaluation
prediction_distribution predicted_distribution class_distribution label_distribution actual_distribution
probabilities probability_quantiles quantiles down flat up short neutral long sell hold buy 0 1 2
p01 p05 p10 p25 p50 p75 p90 p95 p99 min max mean median std count samples rows support
rule rules rule_evaluations policy confidence edge threshold label_threshold class_weight class_weights
weighting label_mode label_multiplier weighting_mode parameters params depth horizon target rounds
trades trade_count buy_count sell_count signals signal_count buy_signals sell_signals hold_signals
fees fees_paid fee_paid costs cost turnover net_return net_pnl gross_pnl MDD equity win_rate
accuracy macro_f1 balanced_accuracy majority_baseline baseline_accuracy logloss brier_score
expected_return signal guard quote_age quote_age_seconds
reason reasons rejection_reason rejection_reasons rejected eligible promoted promote promotion
selection_reason promotion_reason incumbent_run candidate_run no_trade_reason start end duration_seconds
""".split())
MAX_DIAGNOSTIC_DEPTH = 8
MAX_DIAGNOSTIC_ITEMS = 100
MAX_DIAGNOSTIC_NODES = 1500
MAX_DIAGNOSTIC_CHARS = 24000


def diagnostic(value, allowed=DIAGNOSTIC_KEYS):
    """허용 키·깊이·항목 수·총 문자수로 로컬 진단을 제한합니다."""
    budget = [MAX_DIAGNOSTIC_NODES, MAX_DIAGNOSTIC_CHARS]

    def visit(item, depth):
        if depth > MAX_DIAGNOSTIC_DEPTH or budget[0] <= 0 or budget[1] <= 0:
            return None
        budget[0] -= 1
        if isinstance(item, dict):
            result = {}
            for key in item:
                if key not in allowed or len(result) >= MAX_DIAGNOSTIC_ITEMS or budget[0] <= 0 or budget[1] <= 0:
                    continue
                budget[1] -= len(key)
                result[key] = visit(item[key], depth + 1)
            return result
        if isinstance(item, list):
            result = []
            for entry in item[:MAX_DIAGNOSTIC_ITEMS]:
                if budget[0] <= 0 or budget[1] <= 0:
                    break
                result.append(visit(entry, depth + 1))
            return result
        if isinstance(item, str):
            result = item[:min(240, max(0, budget[1]))]
            budget[1] -= len(result)
            return result
        return item if item is None or isinstance(item, bool) or finite(item) else None

    return visit(value, 0)


def execution_probe(value):
    """본 계좌와 합산하지 않는 공개호가 경로 검사만 투영합니다."""
    if not isinstance(value, dict):
        return None
    allowed = frozenset("checked_at market source scope passed buy sell round_trip_net_pnl fees quote_timestamps error "
                        "side qty cash vwap fee_paid unfilled partial timestamp event_id paper_only "
                        "quantity price value notional fee fee_rate quote_timestamp "
                        "requested_krw requested_qty filled_qty filled_notional gross cost net proceeds slippage "
                        "spread spread_bps quote_age_seconds".split())
    # 예외 객체·원문은 비밀값을 포함할 수 있으므로 error는 확정된 코드만 허용합니다.
    safe = {key: value[key] for key in (
        "checked_at", "market", "source", "scope", "passed", "buy", "sell", "round_trip_net_pnl", "fees", "quote_timestamps")
            if key in value}
    if value.get("error") in ("quote_unavailable", "stale_quote", "insufficient_depth", "invalid_quote", "probe_failed"):
        safe["error"] = value["error"]
    return diagnostic(safe, allowed)


def project(value, kind: str):
    """허용한 필드의 단순 값만 추출합니다."""
    if not isinstance(value, dict):
        return None
    result = {}
    for key in FIELDS[kind].split():
        item = value.get(key)
        if item is None or isinstance(item, (str, bool, int, float)):
            if isinstance(item, float) and not math.isfinite(item):
                item = None
            result[key] = item[:240] if isinstance(item, str) else item
    return result


def rows(value, kind: str):
    if not isinstance(value, list):
        return []
    return [item for row in value[-1000:] if (item := project(row, kind)) is not None]


def mapping(value):
    return value if isinstance(value, dict) else {}


def finite(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def distribution(value):
    if isinstance(value, list) and len(value) == 3:
        return {key: item if finite(item) and item >= 0 else None
                for key, item in zip(("down", "flat", "up"), value)}
    return project(value, "class_distribution")


def metric(value):
    if not isinstance(value, dict):
        return None
    result = project({**value, "baseline_accuracy": value.get("baseline_accuracy", value.get("majority_baseline")),
                      "samples": value.get("samples", value.get("support")),
                      "period_start": value.get("period_start", value.get("start")),
                      "period_end": value.get("period_end", value.get("end"))}, "metrics")
    result["class_distribution"] = distribution(value.get("class_distribution"))
    return result


def market_report(report):
    result = {}
    for coin in COINS:
        key = "KRW-" + coin
        source = mapping(mapping(report.get("markets")).get(key))
        locked = mapping(mapping(report.get("locked_test")).get("markets")).get(key)
        if not source and not isinstance(locked, dict):
            continue
        item = {"validation": metric(source.get("validation")), "locked_test": metric(locked),
                "gpu": rows(source.get("gpu"), "gpu_fit")}
        for name in ("diagnostics", "research", "rule"):
            if isinstance(source.get(name), dict):
                item[name] = diagnostic(source[name])
        if isinstance(source.get("weights"), list):
            item["weights"] = diagnostic(source["weights"])
        if finite(source.get("temperature")):
            item["temperature"] = source["temperature"]
        item["splits"] = {}
        for name in ("train", "tune", "calibration", "rule", "validation"):
            split = mapping(mapping(source.get("splits")).get(name))
            if split:
                item["splits"][name] = project({**split, "period_start": split.get("start"),
                    "period_end": split.get("end")}, "split")
                item["splits"][name]["class_distribution"] = distribution(split.get("class_distribution"))
        if item["validation"] and not item["validation"].get("class_distribution"):
            item["validation"]["class_distribution"] = distribution(mapping(report.get("class_distribution")).get(key))
        for name in ("train", "tune", "calibration"):
            split = mapping(mapping(source.get("splits")).get(name))
            item[name] = project({**split, "period_start": split.get("start"),
                                  "period_end": split.get("end")}, "split")
            item[name]["class_distribution"] = distribution(split.get("class_distribution"))
        result[key] = item
    return result


def aggregate_distribution(markets):
    values = [mapping(item.get("validation")).get("class_distribution") for item in markets.values()]
    values = [value for value in values if isinstance(value, dict)]
    if not values:
        return None
    return {key: sum(value[key] for value in values) if all(finite(value.get(key)) for value in values) else None
            for key in ("down", "flat", "up")}


def validation_summary(markets):
    values = [item["validation"] for item in markets.values() if item.get("validation")]
    if not values:
        return None
    counts = [value.get("samples") for value in values]
    total = sum(counts) if all(finite(n) and n >= 0 for n in counts) else None
    result = project({}, "metrics")
    result["samples"] = total
    # 정확도와 기준선 외에 원자료 없이 결합할 수 없는 지표는 평균하지 않습니다.
    for key in ("accuracy", "baseline_accuracy"):
        if total and all(finite(value.get(key)) for value in values):
            result[key] = sum(value[key] * count for value, count in zip(values, counts)) / total
    for key, operation in (("period_start", min), ("period_end", max)):
        times = [value.get(key) for value in values]
        if times and all(isinstance(value, str) for value in times):
            result[key] = operation(times)
    return result


def adapt_account(raw, settings):
    result = {**raw, "initial_capital": raw.get("initial_capital", raw.get("initial_cash")),
              "updated_utc": raw.get("updated_utc", raw.get("updated_at")),
              "started_utc": raw.get("started_utc", raw.get("start_time"))}
    if finite(raw.get("net_return")):
        result["return_pct"] = raw["net_return"] * 100
    if finite(raw.get("MDD")):
        result["mdd_pct"] = raw["MDD"] * 100
    if finite(raw.get("equity")) and finite(result.get("initial_capital")):
        result["total_pnl"] = raw["equity"] - result["initial_capital"]
    positions = raw.get("positions")
    asset_values = raw.get("assets") if isinstance(raw.get("assets"), list) else []
    bid_assets = {item["market"]: item for item in asset_values
                  if isinstance(item, dict) and isinstance(item.get("market"), str)}
    fee, slip = settings.get("fee"), settings.get("slippage")
    factor = (1 - fee) * (1 - slip) if all(finite(v) and 0 <= v < 1 for v in (fee, slip)) else None
    unrealized = []
    complete = isinstance(positions, dict)
    for market, position in mapping(positions).items():
        p = mapping(position)
        asset = bid_assets.get(market, {})
        quantity, cost, price = asset.get("qty"), asset.get("cost", p.get("cost")), asset.get("price")
        # 자산 스냅숏의 매수호가만 사용하고 중간호가로 대체하지 않습니다.
        if (market not in {"KRW-" + coin for coin in COINS}
                or not all(finite(v) for v in (quantity, cost, price, factor))
                or quantity != p.get("qty")):
            complete = False
        else:
            unrealized.append(quantity * price * factor - cost)
    if complete:
        result["unrealized_pnl"] = sum(unrealized)
        if finite(result.get("total_pnl")):
            result["realized_pnl"] = result["total_pnl"] - result["unrealized_pnl"]
    guard = raw.get("entry_guard")
    result["entry_guard"] = ({"unvalidated_or_unprofitable_model": "검증 미달 또는 수익 기준 미달 · 신규 진입 보류",
                             "legacy_schema_requires_validation": "이전 스키마 재검증 필요 · 신규 진입 보류",
                             "model_load_error": "운용 모델 읽기 실패 · 신규 진입 보류"}.get(guard)
                            if isinstance(guard, str) and guard in ("unvalidated_or_unprofitable_model",
                                "legacy_schema_requires_validation", "model_load_error") else "추론 오류로 신규 진입 차단" if guard in (
        "추론 오류로 신규 진입 차단", "추론오류로신규진입차단") else "진입 제한 상태 확인 필요" if guard else None)
    assets = []
    for source in raw.get("assets", []) if isinstance(raw.get("assets"), list) else []:
        if not isinstance(source, dict):
            continue
        market = source.get("market", source.get("symbol"))
        item = {**source, "symbol": market, "quantity": source.get("quantity", source.get("qty")),
                "average_price": source.get("average_price", source.get("entry_price")),
                "market_value": source.get("market_value", source.get("value"))}
        if all(finite(v) for v in (source.get("qty"), source.get("price"), source.get("cost"), factor)):
            item["unrealized_pnl"] = source["qty"] * source["price"] * factor - source["cost"]
        assets.append(item)
    return project(result, "account"), assets


class SnapshotAdapter:
    """경로를 외부 입력으로 받지 않는 제한된 스냅숏 어댑터입니다."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.sources = {}

    def read(self, relative: str):
        path = self.root / relative
        try:
            resolved = path.resolve()
            if not resolved.is_relative_to(self.root) or not path.is_file():
                self.sources[relative] = "missing"
                return None
            if path.stat().st_size > MAX_JSON_BYTES:
                self.sources[relative] = "too_large"
                return None
            with path.open("rb") as stream:
                raw = stream.read(MAX_JSON_BYTES + 1)
            if len(raw) > MAX_JSON_BYTES:
                self.sources[relative] = "too_large"
                return None
            value = json.loads(raw.decode("utf-8-sig"))
            self.sources[relative] = "ok"
            return value
        except (OSError, ValueError, UnicodeError):
            self.sources[relative] = "unavailable"
            return None

    def status(self):
        self.sources = {}
        combined = self.read("work/reports/dashboard.json")
        combined = combined if isinstance(combined, dict) else {}
        sections = {}
        for name, path in {
            "account": "work/paper/state.json",
            "training": "work/reports/training.json",
            "data": "work/data/summary.json",
        }.items():
            value = self.read(path)
            fallback = combined.get(name)
            sections[name] = value if isinstance(value, dict) else fallback if isinstance(fallback, dict) else {}
        account = sections["account"]
        training = sections["training"]
        data = sections["data"]
        markets = market_report(training)
        progress = mapping(self.read("work/service/training_progress.json"))
        current = mapping(training.get("training"))
        completed = progress.get("completed_candidates", current.get("completed_candidates"))
        total = progress.get("total_candidates", current.get("total_candidates"))
        training_view = {**training, "state": progress.get("status", current.get("status", training.get("state"))),
                         "stage": progress.get("stage"), "market": progress.get("market"),
                         "candidate": progress.get("candidate", training.get("candidate")),
                         "candidate_index": completed, "candidate_total": total,
                         "updated_utc": progress.get("updated_at", training.get("updated_at")),
                         "model_id": training.get("run", training.get("model_id")),
                         "run": progress.get("run"), "started_utc": current.get("started_at"),
                         "finished_utc": current.get("completed_at"), "recorded_utc": training.get("updated_at"),
                         "progress_started_utc": progress.get("started_at"),
                         "duration_seconds": current.get("duration_seconds"),
                         "elapsed_seconds": progress.get("elapsed_seconds"),
                         "rows": progress.get("rows", current.get("rows")), "tune_rows": progress.get("tune_rows"),
                         "features": progress.get("features", current.get("features")),
                         "current_round": progress.get("current_round"), "target_rounds": progress.get("target_rounds"),
                         "validation_logloss": progress.get("validation_logloss"),
                         "learning_rate": progress.get("learning_rate", progress.get("eta"))}
        if finite(completed) and finite(total) and total > 0:
            training_view["progress"] = completed / total
        # 보고서 설정 전체가 아닌 실제 사용 시간봉 한 항목만 표시합니다.
        training_view["timeframe"] = mapping(training.get("config")).get("base_timeframe")
        locked = mapping(training.get("locked_test"))
        promotion_source = mapping(training.get("promotion"))
        promotion = project(promotion_source, "promotion")
        # 운영자가 볼 수 있는 확정 사유만 표시하고 비교 오류 원문은 전달하지 않습니다.
        reason = promotion_source.get("reason")
        promotion["reason"] = {
            "현재 모델이 없는 최초 완성 후보": "최초 완성 후보 승격",
            "동일 보정 표본에서 확률/거래 지표 악화 없이 개선": "동일 보정 표본에서 지표 개선",
            "확률/거래 지표 악화 또는 유의한 개선 없음; 현재 모델 유지": "지표 악화 또는 개선 부족 · 현재 모델 유지",
            "현재 모델 비교 실패; 포인터 유지": "현재 모델 비교 실패 · 기존 모델 유지",
            "후속 평가 통과 최초 후보": "후속 평가 통과 · 최초 후보 승격",
            "운용 불가 이전 모델 제외; 현금 기준선 통과": "운용 불가 이전 모델 제외 · 현금 기준선 통과",
            "후속 평가 수익/완료 거래 근거 부족; 운용 보류": "후속 평가 수익 또는 완료 거래 근거 부족 · 운용 보류",
            "라벨 의미가 달라 확률 오차는 비교하지 않고 동일 후속 구간 거래 비교": "라벨 의미 상이 · 확률 오차 비교 제외 · 동일 후속 구간 거래 비교",
        }.get(reason if isinstance(reason, str) else None, "사유 확인 대기")
        for name in ("candidate", "incumbent"):
            promotion[name] = project(promotion_source.get(name), "promotion_metrics")
        for name in ("eligibility", "comparison_periods"):
            if isinstance(promotion_source.get(name), dict):
                promotion[name] = diagnostic(promotion_source[name], DIAGNOSTIC_KEYS | {"KRW-" + coin for coin in COINS})
        config = mapping(self.read("engine-config.json"))
        account_view, assets = adapt_account(account, mapping(config.get("paper")))
        health_source = self.read("work/service/health_result.json")
        health = project(health_source, "health")
        if health is not None:
            health["state"] = {"검사 통과": "검사통과", "초기 준비 중": "초기준비중",
                               "검사 실패": "검사실패"}.get(health.get("state"), health.get("state"))
            if health.get("state") not in ("검사통과", "초기준비중", "검사실패"):
                health["state"] = "검사 상태 확인 필요"
            pending = mapping(health_source).get("pending")
            health["pending_count"] = len(pending) if isinstance(pending, list) else None
        trades = []
        trade_values = account.get("trades", combined.get("trades"))
        for trade in trade_values[-1000:] if isinstance(trade_values, list) else []:
            if isinstance(trade, dict):
                reason = trade.get("reason")
                trades.append({**trade, "symbol": trade.get("symbol", trade.get("market")),
                               "quantity": trade.get("quantity", trade.get("qty")),
                               "price": trade.get("price", trade.get("vwap")),
                               "fee": trade.get("fee", trade.get("fee_paid")),
                               "reason": {"signal": "모델 신호", "stop_loss": "손절", "take_profit": "익절"}.get(reason, "기록 없음" if reason is None else "기타 사유")})
        # 엔진의 수집 요약은 data 내부 통계와 최상위 datasets를 분리합니다.
        dataset_values = data.get("datasets", combined.get("datasets"))
        if isinstance(data.get("data"), dict):
            data = {**data["data"], "updated_utc": data.get("updated_at"),
                    "state": data["data"].get("status")}
        if isinstance(dataset_values, list):
            dataset_values = [
                {**row, "symbol": row.get("symbol", row.get("market")),
                 "period_start": row.get("period_start", row.get("start")),
                 "period_end": row.get("period_end", row.get("end")),
                 "state": row.get("state", "과거 탐색 완료" if row.get("backfill_exhausted") is True else "수집 상태 대기")}
                for row in dataset_values if isinstance(row, dict)
            ]
        result = {
            "schema_version": 1,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "policy": {"mode": "spot-paper", "real_orders": False, "notifications": False,
                       "initial_capital": mapping(config.get("paper")).get("initial_cash"),
                       "paper_start": mapping(config.get("paper")).get("start_time"),
                       "accuracy_target": 0.95, "symbols": list(COINS)},
            "account": account_view,
            "assets": rows(assets if "assets" in account else combined.get("assets"), "assets"),
            "equity": rows(account.get("equity_history", combined.get("equity")), "equity"),
            "trades": rows(trades, "trades"),
            "training": project(training_view, "training"),
            "current_model": project(self.read("work/models/current.json"), "model"),
            "promotion": promotion,
            "health": health,
            "gpu": project(self.read("work/service/gpu.json"), "gpu"),
            "validation": validation_summary(markets) if markets else metric(training.get("validation")),
            "locked_test": metric(locked.get("metrics", training.get("locked_test"))),
            "locked_test_state": project(locked, "test_state"),
            "validation_portfolio": project(mapping(training.get("validation")).get("portfolio"), "portfolio"),
            "locked_test_portfolio": project(locked.get("portfolio"), "portfolio"),
            "markets": markets,
            "research": diagnostic(training.get("research")),
            "execution_probe": execution_probe(self.read("work/reports/execution_probe.json")),
            "decision_diagnostics": {market: diagnostic(value, frozenset(
                "probabilities expected_return signal guard quote_age down flat up 0 1 2".split())) for coin in COINS
                if isinstance((value := mapping(account.get("decision_diagnostics")).get(market := "KRW-" + coin)), dict)},
            "inference_diagnostics": {market: diagnostic(value, frozenset(
                "checked_at model_run latest_raw_bar base_timeframe signal_time feature_age_seconds probabilities argmax signal confidence edge reason".split()))
                for coin in COINS if isinstance((value := mapping(account.get("inference_diagnostics")).get(market := "KRW-" + coin)), dict)},
            "class_distribution": aggregate_distribution(markets) if markets else distribution(training.get("class_distribution")),
            "rounds": self.experiments() or rows(training.get("rounds", combined.get("rounds")), "rounds"),
            "data": project(data, "data"),
            "datasets": rows(dataset_values, "datasets"),
            "services": {name: project(self.read(f"work/service/{name}.json"), "service")
                         for name in ("collector", "backfill", "trainer", "paper", "health")},
        }
        result["sources"] = dict(self.sources)
        return result

    def experiments(self):
        directory = self.root / "work/reports/experiments"
        try:
            if not directory.resolve().is_relative_to(self.root) or not directory.is_dir():
                return []
            # 엔진의 UTC run 이름 순서에서 마지막 20개만 읽습니다.
            paths = heapq.nlargest(20, (p for p in directory.glob("*.json") if p.is_file()), key=lambda p: p.name)
        except OSError:
            return []
        result = []
        for path in reversed(paths):
            report = self.read(path.relative_to(self.root).as_posix())
            if not isinstance(report, dict):
                continue
            markets = market_report(report)
            validation = validation_summary(markets) or {}
            locked = metric(mapping(report.get("locked_test")).get("metrics")) or {}
            timing = mapping(report.get("training"))
            item = project({**validation, "run": report.get("run"), "state": timing.get("status"),
                            "started_utc": timing.get("started_at"), "finished_utc": timing.get("completed_at"),
                            "recorded_utc": report.get("updated_at"), "duration_seconds": timing.get("duration_seconds"),
                            "split": "검증 · 보정 구간",
                            "locked_accuracy": locked.get("accuracy"), "locked_samples": locked.get("samples")}, "rounds")
            item["markets"] = markets
            result.append(item)
        return result


STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8"),
          "/fonts/NanumGothic-Bold.ttf": ("fonts/NanumGothic-Bold.ttf", "font/ttf"),
          "/fonts/OFL.txt": ("fonts/OFL.txt", "text/plain; charset=utf-8")}


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "LocalDashboard"
    sys_version = ""

    def __init__(self, *args, root=ROOT, **kwargs):
        self.root = Path(root).resolve()
        super().__init__(*args, **kwargs)

    def reply(self, code, body: bytes, content_type="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlsplit(self.path)
        path = unquote(url.path)
        if url.query:
            self.reply(400, b'{"error":"query_not_supported"}')
            return
        if path == "/api/status":
            value = SnapshotAdapter(self.root).status()
            self.reply(200, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))
            return
        asset = STATIC.get(path)
        if asset:
            target = (self.root / "web" / asset[0]).resolve()
            if target.is_relative_to(self.root / "web") and target.is_file():
                self.reply(200, target.read_bytes(), asset[1])
                return
        self.reply(404, b'{"error":"not_found"}')

    def method_not_allowed(self):
        self.reply(405, b'{"error":"read_only"}')

    do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = method_not_allowed

    def log_message(self, format, *args):
        # 요청 경로에 잘못 담긴 비밀값이 접근 로그로 복제되지 않게 합니다.
        return


def create_server(host="127.0.0.1", port=8765, root=ROOT):
    return ThreadingHTTPServer((host, port), partial(DashboardHandler, root=root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with create_server(args.host, args.port) as server:
        print(f"로컬 모의거래 대시보드 http://{args.host}:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
