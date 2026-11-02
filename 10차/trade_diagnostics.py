"""현물 모의 장부의 비례 평균원가를 읽기 전용으로 재구성합니다.

결과는 누적 장부 진단이며 현재 모델의 독립 성적이 아닙니다.
과거 호가가 없는 체결 가격차를 스프레드·시장변동·슬리피지로 나누지 않습니다.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
MARKETS = {"KRW-BTC", "KRW-ETH", "KRW-SOL", "KRW-XRP", "KRW-DOGE"}
REASONS = {"signal", "stop_loss", "take_profit", "time_exit"}
MAX_BYTES = 32 * 1024 * 1024
FILTER_METRICS = ("roundtrip_cost", "incremental_quote_drag", "net_label_budget", "net_label_budget_after_book_cost")


def number(value, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid_number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError("invalid_number") from None
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError("invalid_number")
    return result


def close(a, b, quantity=False):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12 if quantity else 1e-6)


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("invalid_timestamp")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("invalid_timestamp")
        return number(stamp.timestamp())
    except (ValueError, OverflowError, OSError):
        raise ValueError("invalid_timestamp") from None


def summary(values):
    return {"count": len(values), "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": math.fsum(values) / len(values) if values else None}


def bucket():
    return {"exits": 0, "wins": 0, "losses": 0, "net_pnl": 0.,
            "entry_fees": 0., "exit_fees": 0., "fees": 0., "fill_price_pnl": 0.}


def quote_status(event):
    context = event.get("execution_context")
    if not isinstance(context, dict):
        return "partial" if isinstance(event.get("quote"), dict) and event["quote"] else "unknown"
    scope = context.get("scope")
    if scope == "synthetic_candle":
        return "synthetic"
    if scope != "public_orderbook_at_execution":
        return "unknown"
    try:
        bid, ask = number(context.get("best_bid")), number(context.get("best_ask"))
        quoted, observed = timestamp(context.get("quote_utc")), timestamp(context.get("observed_utc"))
        age = number(context.get("quote_age_seconds"))
        rates = [number(context.get(key), 0) for key in ("fee_rate", "slippage_rate")]
        if (not 0 < bid <= ask or not close(observed - quoted, age)
                or any(rate >= 1 for rate in rates)
                or context.get("risk_price_source") not in {"observed_bid", "current_prices"}):
            return "partial"
    except (ValueError, TypeError):
        return "partial"
    return "complete"


def context_status(value):
    if value is None:
        return "missing"
    if not isinstance(value, dict):
        return "invalid"
    label = value.get("record_status")
    return label if isinstance(label, str) and label in {"recorded", "partial_invalid", "invalid", "missing"} else "unknown"


def metadata_present(event, key):
    value = event.get(key)
    if key == "model_run":
        return isinstance(value, str) and bool(value.strip())
    if key == "probabilities":
        context = event.get("decision_context")
        if isinstance(context, dict):
            value = context.get("probabilities", value)
        try:
            return (isinstance(value, list) and len(value) == 3
                    and all(0 <= number(p) <= 1 for p in value) and close(sum(value), 1))
        except ValueError:
            return False
    return quote_status(event) == "complete"


def diagnose(state):
    """불완전 이력은 완전한 계좌나 현재 모델 성적으로 승인하지 않습니다."""
    result = {"schema_version": 1, "scope": "cumulative_paper_ledger_not_current_model_performance",
              "status": "rejected", "reconciled": False, "issues": [], "totals": None,
              "cost_separation": {"slippage": "unavailable", "spread_vs_market_move": "unavailable"}}
    try:
        _reconstruct(state, result)
        # 파생 합계의 오버플로도 유효한 JSON 수치로 내보내지 않습니다.
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError, OverflowError, KeyError):
        result = {"schema_version": 1, "scope": result["scope"], "status": "rejected",
                  "reconciled": False, "issues": ["invalid_ledger"], "totals": None,
                  "cost_separation": result["cost_separation"]}
    return result


def _reconstruct(state, result):
    if not isinstance(state, dict) or state.get("paper_only") is not True:
        raise ValueError("not_paper_ledger")
    initial, actual_cash = number(state.get("initial_cash"), 0), number(state.get("cash"), 0)
    if initial <= 0 or not isinstance(state.get("positions"), dict) or not isinstance(state.get("trades"), list):
        raise ValueError("invalid_state")
    events = state["trades"]
    issues = result["issues"]
    if len(events) >= 2000 or state.get("history_truncated") is True:
        issues.append("possibly_truncated_history")
    positions, seen, stops = {}, set(), {}
    cash, paid_fees, recorded_pnl, closed = initial, 0., 0., 0
    previous, holdings, reentries = None, [], []
    totals, by_market, by_reason, by_model = bucket(), {}, {}, {}
    missing = {"model_run": 0, "probabilities": 0, "quote": 0}
    contexts = {"entry_model_run": 0, "entry_context": 0, "decision_context": 0, "execution_context": 0}
    quote_counts = {key: 0 for key in ("complete", "partial", "unknown", "synthetic")}
    context_labels = {key: {label: 0 for label in ("recorded", "partial_invalid", "invalid", "missing", "unknown")}
                      for key in ("entry_context", "decision_context")}
    filters = {key: [] for key in FILTER_METRICS}
    buys, unattributed, missing_filter, invalid_entries = 0, 0, 0, 0
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("invalid_event")
        identifier, market, side = event.get("event_id"), event.get("market"), event.get("side")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("invalid_event_id")
        seen.add(identifier)
        if market not in MARKETS or side not in {"buy", "sell"}:
            raise ValueError("invalid_market_or_side")
        stamp = timestamp(event.get("timestamp"))
        if previous is not None and stamp < previous:
            raise ValueError("nonchronological_events")
        previous = stamp
        qty, price = number(event.get("qty"), 0), number(event.get("vwap"), 0)
        fee, flow = number(event.get("fee_paid"), 0), number(event.get("cash"), 0)
        if qty <= 0 or price <= 0 or flow <= 0:
            raise ValueError("invalid_fill")
        gross = number(qty * price, 0)
        if not close(flow, gross + fee if side == "buy" else gross - fee):
            raise ValueError("fill_cash_mismatch")
        paid_fees += fee
        for key in missing:
            missing[key] += not metadata_present(event, key)
        quote_counts[quote_status(event)] += 1
        for key, labels in context_labels.items():
            labels[context_status(event.get(key))] += 1
        original_entry = event.get("entry_context")
        invalid_entry = (event.get("entry_context_invalid") is True
                         or context_status(original_entry) in {"invalid", "partial_invalid"})
        invalid_entries += invalid_entry
        entry_model = event.get("entry_model_run")
        entry_model = entry_model if isinstance(entry_model, str) and entry_model.strip() else None
        contexts["entry_model_run"] += entry_model is None
        for key in ("entry_context", "decision_context", "execution_context"):
            contexts[key] += not isinstance(event.get(key), dict) or not event[key]
        if side == "buy":
            buys += 1
            entry_filter = event.get("entry_filter")
            missing_filter += not isinstance(entry_filter, dict) or not entry_filter
            if isinstance(entry_filter, dict):
                for key in FILTER_METRICS:
                    try:
                        filters[key].append(number(entry_filter.get(key)))
                    except ValueError:
                        pass
            if flow > cash and not close(flow, cash):
                raise ValueError("cash_overspending")
            cash -= flow
            old = positions.get(market, {"qty": 0., "cost": 0., "entry_fees": 0., "entry_epoch": stamp})
            new_qty = number(old["qty"] + qty, 0)
            positions[market] = {"qty": new_qty, "cost": number(old["cost"] + flow, 0),
                                 "entry_fees": number(old["entry_fees"] + fee, 0),
                                 "entry_epoch": (old["entry_epoch"] * old["qty"] + stamp * qty) / new_qty}
            if market in stops:
                reentries.append(stamp - stops.pop(market))
            continue
        position = positions.get(market)
        if position is None or (qty > position["qty"] and not close(qty, position["qty"], True)):
            issues.append("missing_entry_or_oversell")
            result.update(status="incomplete", metadata_missing=missing)
            return
        ratio = min(1., qty / position["qty"])
        basis, entry_fee = position["cost"] * ratio, position["entry_fees"] * ratio
        pnl = flow - basis
        actual_pnl = event.get("realized_pnl")
        if actual_pnl is None:
            issues.append("missing_realized_pnl")
        else:
            actual_pnl = number(actual_pnl)
            recorded_pnl += actual_pnl
            if not close(actual_pnl, pnl):
                issues.append("realized_pnl_mismatch")
        reason = event.get("reason")
        reason = reason if isinstance(reason, str) and reason in REASONS else "other"
        if reason == "stop_loss":
            stops[market] = stamp
        holdings.append(stamp - position["entry_epoch"])
        groups = [totals, by_market.setdefault(market, bucket()), by_reason.setdefault(reason, bucket())]
        # 현재 판단 모델이나 계좌의 최신 모델로 과거 진입 모델을 메우지 않습니다.
        entry_context = event.get("entry_context")
        if invalid_entry or (isinstance(entry_context, dict) and entry_context.get("model_run") not in (None, entry_model)):
            entry_model = None
        if entry_model is None:
            unattributed += 1
        else:
            model_key = hashlib.sha256(entry_model.encode("utf-8")).hexdigest()
            groups.append(by_model.setdefault(model_key, bucket()))
        for group in groups:
            group["exits"] += 1
            group["wins"] += pnl > 0
            group["losses"] += pnl < 0
            group["net_pnl"] += pnl
            group["entry_fees"] += entry_fee
            group["exit_fees"] += fee
            group["fees"] += entry_fee + fee
            group["fill_price_pnl"] += pnl + entry_fee + fee
        cash += flow
        position["qty"] = max(0., position["qty"] - qty)
        position["cost"] -= basis
        position["entry_fees"] -= entry_fee
        if position["qty"] <= 1e-12:
            del positions[market]
            closed += 1
    actual_positions = state["positions"]
    positions_match = set(actual_positions) == set(positions)
    for market, actual in actual_positions.items():
        if market not in MARKETS or not isinstance(actual, dict):
            raise ValueError("invalid_position")
        qty, cost = number(actual.get("qty"), 0), number(actual.get("cost"), 0)
        if qty <= 0 or cost <= 0:
            raise ValueError("invalid_position")
        expected = positions.get(market)
        positions_match = positions_match and expected is not None and close(qty, expected["qty"], True) and close(cost, expected["cost"])
    checks = {"cash": close(cash, actual_cash), "positions": positions_match,
              "capital_identity": close(actual_cash + sum(p["cost"] for p in actual_positions.values()), initial + recorded_pnl)}
    if "fees_paid" in state:
        checks["fees_paid"] = close(number(state["fees_paid"], 0), paid_fees)
    else:
        checks["fees_paid"] = False
    completed = state.get("completed_trades")
    if completed is not None and (isinstance(completed, bool) or not isinstance(completed, int) or completed < 0):
        raise ValueError("invalid_completed_trades")
    checks["completed_trades"] = completed == closed
    if "realized_pnl" in state:
        checks["state_realized_pnl"] = close(number(state["realized_pnl"]), recorded_pnl)
    for key, matched in checks.items():
        if not matched:
            issues.append(key + "_mismatch_or_missing")
    issues[:] = sorted(set(issues))
    exits = totals["exits"]
    result.update(status="incomplete" if issues else "complete", reconciled=not issues,
                  reconciliation=checks, totals=totals, recorded_realized_pnl=recorded_pnl,
                  reconstructed_cash=cash, initial_cash=initial, event_count=len(events), buy_fills=buys,
                  by_market=by_market, by_reason=by_reason, metadata_missing=missing,
                  context_missing=contexts, entry_filter_missing_buy_fills=missing_filter,
                  quote_metadata=quote_counts, context_record_status=context_labels,
                  entry_context_invalid_events=invalid_entries,
                  entry_filter_metrics={key: summary(values) for key, values in filters.items()},
                  by_entry_model=by_model, entry_model_key_format="sha256_of_explicit_entry_model_run",
                  unattributed_exits=unattributed,
                  open_positions={m: {key: p[key] for key in ("qty", "cost", "entry_fees")} for m, p in positions.items()},
                  holding_seconds=summary(holdings), holding_method="quantity_weighted_entry_time_per_sell_fill",
                  stop_reentry_seconds=summary(reentries),
                  concentration={"largest_market_exit_share": max((p["exits"] for p in by_market.values()), default=0) / exits if exits else None,
                                 "signal_exit_share": by_reason.get("signal", {}).get("exits", 0) / exits if exits else None})


def safe_path(value, boundary, root):
    """resolve로 링크를 숨기지 않고 모든 기존 경로 요소를 검사합니다."""
    value, root, boundary = Path(value), Path(root).absolute(), Path(boundary).absolute()
    path = value if value.is_absolute() else root / value
    if ".." in path.parts or not path.is_relative_to(boundary) or path == boundary:
        raise ValueError("unsafe_path")
    for part in path.parts[1:]:
        if ":" in part:
            raise ValueError("unsafe_path")
    for current in reversed((path, *path.parents)):
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("linked_path")
        if current != path and not stat.S_ISDIR(info.st_mode):
            raise ValueError("invalid_parent")
        if current == path and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
            raise ValueError("invalid_file")
    return path


def parse_state(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite_json")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _read_snapshot(path, root):
    path = safe_path(path, root, root)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(MAX_BYTES + 1)
        after = os.fstat(stream.fileno())
    current = path.lstat()
    identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise ValueError("snapshot_changed")
    if len(raw) > MAX_BYTES:
        raise ValueError("input_too_large")
    return raw


def stable_read(path, root):
    """변하는 장부를 잠그거나 복제하지 않고 동일한 두 읽기를 유한 확인합니다."""
    for _ in range(3):
        try:
            first = _read_snapshot(path, root)
            second = _read_snapshot(path, root)
        except (OSError, ValueError):
            continue
        if first == second:
            return second, datetime.now(timezone.utc).isoformat()
    raise ValueError("stable_snapshot_unavailable")


def main(argv=None):
    parser = argparse.ArgumentParser(description="누적 현물 장부의 비례 원가 진단; 현재 모델 평가 아님")
    parser.add_argument("--state", default="work/paper/state.json")
    parser.add_argument("--output", default="10차/work/trade-diagnostics.json")
    args = parser.parse_args(argv)
    source = safe_path(args.state, ROOT, ROOT)
    output = safe_path(args.output, ROOT / "10차/work", ROOT)
    if source == output or output.suffix.lower() != ".json":
        raise ValueError("unsafe_output")
    raw, asof = stable_read(source, ROOT)
    result = diagnose(parse_state(raw.decode("utf-8", errors="strict")))
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    result["asof_utc"] = asof
    result["read_consistency"] = "two_identical_reads"
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    safe_path(output, ROOT / "10차/work", ROOT)
    temp = output.with_name("." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        safe_path(output, ROOT / "10차/work", ROOT)
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, RecursionError):
        print("진단 입력 또는 출력 경계를 거부했습니다", file=sys.stderr)
        raise SystemExit(2)
