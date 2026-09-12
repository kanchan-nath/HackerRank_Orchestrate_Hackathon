"""
Step 6 — Deterministic verification pass.
Checks every output row against the rulebook before writing.
"""

from datetime import date
from typing import Dict, List, Optional
from ingest import DataStore


def verify_row(row: dict, request: dict, data_store: DataStore) -> List[str]:
    """Verify a single output row against all invariants.

    Returns list of violation descriptions. Empty list = all OK.
    """
    violations = []
    req_id = row["request_id"]
    requested_amount = request["requested_amount"]
    request_date = request["request_date"]
    desired_completion = request["desired_completion_date"]
    allows_partial = request["allows_partial_payment"]

    profile = data_store.profiles[request["user_id"]]
    user_methods = set(profile["payment_methods_user_will_consider"])

    # 1. amount_safe_to_pay bounds
    safe = row["amount_safe_to_pay"]
    if safe < 0:
        violations.append(f"{req_id}: amount_safe_to_pay ({safe}) < 0")
    if safe > requested_amount:
        violations.append(f"{req_id}: amount_safe_to_pay ({safe}) > requested_amount ({requested_amount})")

    # 2. Valid affordability_status
    valid_statuses = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
    status = row["affordability_status"]
    if status not in valid_statuses:
        violations.append(f"{req_id}: invalid affordability_status: {status}")

    # 3. Valid recommended_payment_method
    valid_methods = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
    method = row["recommended_payment_method"]
    if method not in valid_methods:
        violations.append(f"{req_id}: invalid recommended_payment_method: {method}")

    # 4. affordable_now requires earliest_date == request_date
    if status == "affordable_now":
        earliest = row.get("earliest_date_for_full_payment", "")
        if earliest and earliest != str(request_date):
            violations.append(f"{req_id}: affordable_now but earliest_date ({earliest}) != request_date ({request_date})")

    # 5. Partial payment constraints
    if method == "partial_payment":
        if not allows_partial:
            violations.append(f"{req_id}: partial_payment but allows_partial_payment is False")
        if "partial_payment" not in user_methods:
            violations.append(f"{req_id}: partial_payment not in user's payment_methods_user_will_consider")
        if status != "affordable_with_plan":
            violations.append(f"{req_id}: partial_payment but status is {status}, should be affordable_with_plan")

        # Check two-payment constraint
        plan = row.get("payment_plan", "none")
        if plan and plan != "none":
            parts = plan.split("|")
            if len(parts) != 2:
                violations.append(f"{req_id}: partial_payment must have exactly 2 payments, got {len(parts)}")
            else:
                try:
                    amounts = [float(p.split(":")[1]) for p in parts]
                    total = sum(amounts)
                    if abs(total - requested_amount) > 0.02:
                        violations.append(f"{req_id}: partial payments sum ({total:.2f}) != requested_amount ({requested_amount})")
                except (ValueError, IndexError):
                    violations.append(f"{req_id}: cannot parse partial payment plan: {plan}")

    # 6. Installment plan must match a real payment_option_id
    if method == "installments":
        plan = row.get("payment_plan", "none")
        if plan == "none" or not plan:
            violations.append(f"{req_id}: installments but no payment_plan")
        else:
            options = data_store.options_by_request.get(req_id, [])
            installment_opts = [o for o in options if o["payment_method"] == "installments"]
            if not installment_opts:
                violations.append(f"{req_id}: installments but no installment options available")

    # 7. Spending changes validation
    changes = row.get("spending_changes_needed", "none")
    if changes and changes != "none":
        change_parts = changes.split("|")
        if len(change_parts) > 3:
            violations.append(f"{req_id}: more than 3 spending changes")

        seen_events = {}
        for change in change_parts:
            if change.startswith("stop:"):
                eid = change.split(":")[1]
                if eid in seen_events and seen_events[eid] == "reduce":
                    violations.append(f"{req_id}: both stop and reduce on same event {eid}")
                seen_events[eid] = "stop"

                # Check event exists and is flexible
                event = data_store.events_by_id.get(eid)
                if event:
                    if event["flexibility"] not in ("stoppable", "reducible_or_stoppable"):
                        violations.append(f"{req_id}: stop on non-stoppable event {eid}")
                    if event["category"] in profile["expense_categories_to_protect"]:
                        violations.append(f"{req_id}: stop on protected category event {eid}")

            elif change.startswith("reduce_to:"):
                parts = change.split(":")
                if len(parts) >= 3:
                    eid = parts[1]
                    if eid in seen_events and seen_events[eid] == "stop":
                        violations.append(f"{req_id}: both stop and reduce on same event {eid}")
                    seen_events[eid] = "reduce"

                    event = data_store.events_by_id.get(eid)
                    if event:
                        if event["flexibility"] not in ("reducible", "reducible_or_stoppable"):
                            violations.append(f"{req_id}: reduce on non-reducible event {eid}")
                        if event["category"] in profile["expense_categories_to_protect"]:
                            violations.append(f"{req_id}: reduce on protected category event {eid}")

    # 8. Payment plan chronological order
    plan = row.get("payment_plan", "none")
    if plan and plan != "none":
        parts = plan.split("|")
        dates_in_plan = []
        for p in parts:
            try:
                d_str = p.split(":")[0]
                from ingest import parse_date
                d = parse_date(d_str)
                if d:
                    dates_in_plan.append(d)
            except:
                pass
        for i in range(1, len(dates_in_plan)):
            if dates_in_plan[i] < dates_in_plan[i-1]:
                violations.append(f"{req_id}: payment_plan not in chronological order")
                break

    # 9. not_recommended should have payment_plan = none
    if method == "not_recommended" and plan != "none":
        violations.append(f"{req_id}: not_recommended but payment_plan is not 'none'")

    return violations


def verify_all(results: List[dict], requests: List[dict],
               data_store: DataStore) -> List[str]:
    """Verify all output rows. Returns list of all violations."""
    all_violations = []
    request_map = {r["request_id"]: r for r in requests}

    for row in results:
        req = request_map.get(row["request_id"])
        if not req:
            all_violations.append(f"{row['request_id']}: request not found")
            continue
        violations = verify_row(row, req, data_store)
        all_violations.extend(violations)

    # Check we have the right number of rows
    expected = len(requests)
    actual = len(results)
    if actual != expected:
        all_violations.append(f"Expected {expected} rows, got {actual}")

    return all_violations
