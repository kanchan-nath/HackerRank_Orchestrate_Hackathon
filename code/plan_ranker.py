"""
Step 4 — Enumerate eligible payment plans and rank them.

Deterministic plan selection logic — no LLM calls.
"""
import re

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from ingest import DataStore
from forecast import (
    build_forecast_timeline,
    compute_amount_safe_to_pay,
    compute_earliest_full_payment_date,
    get_flexible_recurring_events,
    compute_with_spending_changes,
)


def generate_installment_schedule(option: dict) -> List[Tuple[date, float]]:
    """Generate payment schedule from a payment option.

    Returns list of (date, amount) tuples.
    """
    first_date = option["first_payment_date"]
    n_payments = option["number_of_payments"]
    freq_days = option["payment_frequency_days"]
    per_payment = option["payment_amount"]

    schedule = []
    for i in range(n_payments):
        if freq_days:
            pay_date = first_date + timedelta(days=freq_days * i)
        else:
            pay_date = first_date
        schedule.append((pay_date, per_payment))

    return schedule


def check_installment_safety(schedule: List[Tuple[date, float]],
                              balance: float, min_balance: float,
                              timeline: Dict[date, float],
                              request_date: date) -> bool:
    """Check if an installment plan keeps balance safe throughout 90-day window."""
    end_date = request_date + timedelta(days=90)
    current = balance

    # Build combined timeline with installment payments
    d = request_date
    while d <= end_date:
        if d in timeline:
            current += timeline[d]
        # Check for installment payment on this day
        for pay_date, amount in schedule:
            if pay_date == d:
                current -= amount
        if current < min_balance:
            return False
        d += timedelta(days=1)

    return True


def rank_plans(plans: List[dict], desired_completion_date: date) -> List[dict]:
    """Rank eligible plans by preference criteria.

    1. Complete by desired_completion_date
    2. Require no spending changes
    3. Minimize total amount paid
    4. Start payment earliest
    5. Fewest payments
    6. Lowest payment_option_id
    """
    def sort_key(plan):
        # 1. Completes by deadline (True = completes, sort True first -> use False)
        completes = plan.get("completes_by_deadline", False)
        # 2. No spending changes (True = no changes, sort True first -> use False)
        no_changes = plan.get("spending_changes_needed", "none") == "none"
        # 3. Total amount (minimize)
        total = plan.get("total_payable", float("inf"))
        # 4. Start date (earlier is better)
        start = plan.get("first_payment_date", date.max)
        # 5. Fewer payments
        n_payments = plan.get("number_of_payments", 999)
        # 6. Lowest option ID

        opt_id_raw = plan.get("payment_option_id", "")
        nums = re.findall(r'\d+', opt_id_raw)
        opt_num = int(nums[0]) if nums else 999999

        opt_id = plan.get("payment_option_id", "zzz")

        return (
            not completes,      # False (completes) sorts before True
            not no_changes,     # False (no changes) sorts before True
            total,
            start,
            n_payments,
            opt_num,
        )

    return sorted(plans, key=sort_key)


def process_request(request: dict, data_store: DataStore,
                     messages: List[dict]) -> dict:
    """Process a single request and return the decision.

    This is the main entry point for the deterministic decision engine.
    """
    user_id = request["user_id"]
    request_date = request["request_date"]
    requested_amount = request["requested_amount"]
    desired_completion = request["desired_completion_date"]
    allows_partial = request["allows_partial_payment"]

    profile = data_store.profiles[user_id]
    home_ccy = profile["home_currency"]
    balance = profile["current_available_balance"]
    min_balance = profile["minimum_balance_to_keep"]
    user_methods = set(profile["payment_methods_user_will_consider"])
    max_installment_months = profile["max_installment_months"]

    # User messages
    user_msgs = [m for m in messages if m["user_id"] == user_id]

    # Build forecast timeline
    timeline = build_forecast_timeline(user_id, request_date, data_store, user_msgs)

    # Compute base metrics (before spending changes)
    amount_safe = compute_amount_safe_to_pay(
        balance, min_balance, timeline, request_date, requested_amount
    )
    earliest_date = compute_earliest_full_payment_date(
        balance, min_balance, timeline, request_date, requested_amount
    )

    # Get flexible events for potential spending changes
    flexible_events = get_flexible_recurring_events(user_id, request_date, data_store)

    # Try spending changes if needed
    spending_changes = "none"
    amount_safe_with_changes = amount_safe
    earliest_with_changes = earliest_date

    if amount_safe < requested_amount and flexible_events:
        new_safe, changes_str, new_earliest = compute_with_spending_changes(
            balance, min_balance, timeline, request_date, requested_amount,
            flexible_events, data_store, home_ccy
        )
        if changes_str:
            amount_safe_with_changes = new_safe if new_safe is not None else amount_safe
            if new_earliest:
                earliest_with_changes = new_earliest
            spending_changes = changes_str

    # Get payment options for this request
    options = data_store.options_by_request.get(request["request_id"], [])

    # Enumerate eligible plans
    eligible_plans = []

    # --- Plan A: Full payment ---
    if "full_payment" in user_methods:
        full_opts = [o for o in options if o["payment_method"] == "full_payment"]
        for opt in full_opts:
            # Check if full payment is safe on request_date
            if amount_safe >= requested_amount:
                eligible_plans.append({
                    "method": "full_payment",
                    "status": "affordable_now",
                    "amount_safe_to_pay": amount_safe,
                    "payment_plan": f"{request_date}:{requested_amount:.2f}",
                    "earliest_date": request_date,
                    "spending_changes_needed": "none",
                    "completes_by_deadline": True,
                    "total_payable": opt["total_payable_amount"],
                    "first_payment_date": request_date,
                    "number_of_payments": 1,
                    "payment_option_id": opt["payment_option_id"],
                })
            # Check with spending changes
            elif amount_safe_with_changes >= requested_amount and spending_changes != "none":
                eligible_plans.append({
                    "method": "full_payment",
                    "status": "affordable_with_plan",
                    "amount_safe_to_pay": amount_safe,  # Before changes
                    "payment_plan": f"{request_date}:{requested_amount:.2f}",
                    "earliest_date": earliest_with_changes or earliest_date,
                    "spending_changes_needed": spending_changes,
                    "completes_by_deadline": True,
                    "total_payable": opt["total_payable_amount"],
                    "first_payment_date": request_date,
                    "number_of_payments": 1,
                    "payment_option_id": opt["payment_option_id"],
                })

    # --- Plan B: Installments ---
    if "installments" in user_methods and max_installment_months is not None:
        installment_opts = [o for o in options if o["payment_method"] == "installments"]
        for opt in installment_opts:
            # Check max_installment_months constraint
            n_payments = opt["number_of_payments"]
            freq_days = opt["payment_frequency_days"] or 30
            total_duration_months = (n_payments * freq_days) / 30
            if total_duration_months > max_installment_months:
                continue  # User won't accept this many months

            schedule = generate_installment_schedule(opt)
            if not schedule:
                continue

            # Check if last payment is by desired_completion_date
            last_payment_date = schedule[-1][0]

            # Check safety
            is_safe = check_installment_safety(
                schedule, balance, min_balance, timeline, request_date
            )

            if is_safe:
                # Build payment plan string
                plan_parts = [f"{d}:{a:.2f}" for d, a in schedule]
                plan_str = "|".join(plan_parts)

                eligible_plans.append({
                    "method": "installments",
                    "status": "affordable_with_plan",
                    "amount_safe_to_pay": amount_safe,
                    "payment_plan": plan_str,
                    "earliest_date": earliest_date,
                    "spending_changes_needed": "none",
                    "completes_by_deadline": last_payment_date <= desired_completion,
                    "total_payable": opt["total_payable_amount"],
                    "first_payment_date": schedule[0][0],
                    "number_of_payments": n_payments,
                    "payment_option_id": opt["payment_option_id"],
                })

    # --- Plan C: Partial payment ---
    if ("partial_payment" in user_methods and allows_partial
            and 0 < amount_safe < requested_amount
            and earliest_date is not None
            and earliest_date <= desired_completion):
        remainder = requested_amount - amount_safe
        plan_str = f"{request_date}:{amount_safe:.2f}|{earliest_date}:{remainder:.2f}"

        eligible_plans.append({
            "method": "partial_payment",
            "status": "affordable_with_plan",
            "amount_safe_to_pay": amount_safe,
            "payment_plan": plan_str,
            "earliest_date": earliest_date,
            "spending_changes_needed": "none",
            "completes_by_deadline": earliest_date <= desired_completion,
            "total_payable": requested_amount,  # No financing fee
            "first_payment_date": request_date,
            "number_of_payments": 2,
            "payment_option_id": "partial",
        })

    # --- Plan D: Wait ---
    if "full_payment" in user_methods and earliest_date and earliest_date > request_date:
        # Wait is eligible when full payment becomes safe later
        completes = earliest_date <= desired_completion
        eligible_plans.append({
            "method": "wait",
            "status": "affordable_later",
            "amount_safe_to_pay": amount_safe,
            "payment_plan": f"{earliest_date}:{requested_amount:.2f}",
            "earliest_date": earliest_date,
            "spending_changes_needed": "none",
            "completes_by_deadline": completes,
            "total_payable": requested_amount,
            "first_payment_date": earliest_date,
            "number_of_payments": 1,
            "payment_option_id": "wait",
        })

    # Rank eligible plans
    if eligible_plans:
        ranked = rank_plans(eligible_plans, desired_completion)
        best = ranked[0]

        # Determine final values
        result = {
            "request_id": request["request_id"],
            "amount_safe_to_pay": round(amount_safe, 2),
            "affordability_status": best["status"],
            "recommended_payment_method": best["method"],
            "payment_plan": best["payment_plan"],
            "earliest_date_for_full_payment": str(earliest_date) if earliest_date else "",
            "spending_changes_needed": best["spending_changes_needed"],
            "requested_amount": requested_amount,
            "request_text": request.get("request_text", ""),
        }
    else:
        # Not recommended
        result = {
            "request_id": request["request_id"],
            "amount_safe_to_pay": round(amount_safe, 2),
            "affordability_status": "not_affordable",
            "recommended_payment_method": "not_recommended",
            "payment_plan": "none",
            "earliest_date_for_full_payment": str(earliest_date) if earliest_date else "",
            "spending_changes_needed": "none",
            "requested_amount": requested_amount,
            "request_text": request.get("request_text", ""),
        }

    # -- Post-processing invariants --

    # If affordable_now, earliest must equal request_date
    if result["affordability_status"] == "affordable_now":
        result["earliest_date_for_full_payment"] = str(request_date)

    # Ensure amount_safe_to_pay bounds
    result["amount_safe_to_pay"] = max(0, min(result["amount_safe_to_pay"], requested_amount))

    # If not_affordable and no earliest date, leave empty
    if result["affordability_status"] == "not_affordable" and not earliest_date:
        result["earliest_date_for_full_payment"] = ""

    # For wait, status should be affordable_later (not affordable_with_plan)
    if result["recommended_payment_method"] == "wait":
        if result["affordability_status"] != "affordable_later":
            result["affordability_status"] = "affordable_later"

    return result
