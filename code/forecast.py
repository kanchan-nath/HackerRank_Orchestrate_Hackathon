"""
Step 2+3 — Reconstruct each user's ground-truth cash position and
           build 90-day forward forecast.

DESIGN: The balance is a snapshot. We compute future net changes only:
1. Fixed recurring: project from last settled date at detected interval
2. Variable recurring (groceries etc): project from last settled date at detected interval
3. Scheduled/pending one-time: include if settlement_date >= request_date
4. Salary: from scheduled events or recurring pattern, with message amendments

Key: We group ALL recurring by (category, description) for fixed items,
and by (category ALONE) for variable items, computing an aggregate weekly/monthly rate.

However, the critical thing is: the balance already includes settled events.
So we ONLY project the NEXT occurrence forward from the LAST settled occurrence.

This is the DETERMINISTIC CORE. No LLM calls.
"""

import re
from datetime import date, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set
from ingest import DataStore


def detect_all_recurring(events: List[dict], request_date: date) -> Tuple[List[dict], Set[str]]:
    """Detect all recurring patterns from settled history.

    Groups by (category, description) and checks for regular intervals.
    For categories with multiple descriptions (groceries, transport),
    each description pattern is its own recurring stream.

    Returns:
        (recurring_templates, set_of_all_recurring_event_ids)
    """
    groups = defaultdict(list)
    for e in events:
        if e["status"] in ("cancelled", "failed", "unrealized"):
            continue
        if e["direction"] == "non_cash":
            continue
        if e["status"] != "settled":
            continue
        desc = e["description"].lower().strip()
        key = (e["category"], desc, e["direction"])
        groups[key].append(e)

    templates = []
    all_recurring_ids = set()

    for (cat, desc, direction), group in groups.items():
        sorted_events = sorted(group, key=lambda x: x["settlement_date"] or date.min)
        if len(sorted_events) < 2:
            continue

        dates = [e["settlement_date"] for e in sorted_events if e["settlement_date"]]
        if len(dates) < 2:
            continue

        intervals = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
        avg_interval = sum(intervals) / len(intervals)

        if avg_interval < 3 or avg_interval > 100:
            continue

        tolerance = max(5, avg_interval * 0.35)
        consistent = all(abs(iv - avg_interval) <= tolerance for iv in intervals)

        if not consistent:
            continue

        # Only project if the pattern is still active (last occurrence recent enough)
        last_occurrence = dates[-1]
        days_since_last = (request_date - last_occurrence).days
        # Allow up to 2.5x the interval — if longer, pattern likely stopped
        if days_since_last > avg_interval * 2.5 and days_since_last > 60:
            continue

        for e in sorted_events:
            all_recurring_ids.add(e["event_id"])

        template = dict(sorted_events[-1])
        template["_recurrence_interval_days"] = round(avg_interval)
        template["_recurrence_dates"] = dates
        template["_all_events"] = sorted_events

        recent = [e["amount"] for e in sorted_events[-3:] if e["amount"] is not None]
        if recent:
            template["_latest_amount"] = recent[-1]
            # Use average of recent amounts for variable items, max for conservative
            template["_avg_amount"] = sum(recent) / len(recent)
            template["_forecast_amount"] = template["_avg_amount"]
        templates.append(template)

    return templates, all_recurring_ids


def aggregate_category_recurring(templates: List[dict], request_date: date, 
                                  end_date: date, home_ccy: str,
                                  data_store: DataStore,
                                  salary_info: dict, rent_pct: Optional[float]) -> Dict[date, float]:
    """Project all recurring templates into timeline changes.
    
    Only projects FUTURE occurrences (after last settled date).
    """
    timeline: Dict[date, float] = defaultdict(float)
    
    for tmpl in templates:
        cat = tmpl["category"]
        direction = tmpl["direction"]
        interval = tmpl["_recurrence_interval_days"]
        amount = tmpl.get("_forecast_amount", tmpl["_latest_amount"])
        
        # Skip cancelled salary
        if cat == "salary" and salary_info.get("is_cancelled"):
            continue
        # Override salary amount
        if cat == "salary" and salary_info.get("amount_override") is not None:
            amount = salary_info["amount_override"]
        
        # Convert currency
        if tmpl["currency"] != home_ccy:
            amount = data_store.convert_currency(amount, tmpl["currency"],
                                                  home_ccy, request_date)
        
        # Apply rent increase
        if cat == "rent" and rent_pct:
            amount = amount * (1 + rent_pct / 100)
        
        # Find next occurrence after last settled
        last_settled = max(tmpl["_recurrence_dates"])
        
        if cat == "salary" and salary_info.get("date_override"):
            next_date = salary_info["date_override"]
        else:
            next_date = last_settled + timedelta(days=interval)
        
        # Advance past request_date if needed
        while next_date < request_date:
            next_date += timedelta(days=interval)
        
        # Project forward
        proj_date = next_date
        while proj_date <= end_date:
            if direction == "debit":
                timeline[proj_date] -= amount
            elif direction == "credit":
                timeline[proj_date] += amount
            proj_date += timedelta(days=interval)
    
    return dict(timeline)


def get_salary_info_from_messages(user_id: str, messages: List[dict],
                                   request_date: date) -> dict:
    result = {"amount_override": None, "date_override": None, "is_cancelled": False}
    for msg in messages:
        if msg["user_id"] != user_id:
            continue
        text = msg["message_text"]
        if msg.get("_is_cancellation"):
            if any(kw in text.lower() for kw in ["salary", "pay", "contract", "gaji", "payroll"]):
                result["is_cancelled"] = True
        if msg.get("_salary_amendment"):
            result["amount_override"] = msg["_salary_amendment"]
        if any(kw in text.lower() for kw in ["salary", "gaji", "pay"]):
            date_matches = re.findall(r'(\d{4}-\d{2}-\d{2})', text)
            for dm in date_matches:
                from ingest import parse_date
                d = parse_date(dm)
                if d and d > request_date - timedelta(days=60):
                    result["date_override"] = d
    return result


def get_rent_increase_from_messages(user_id: str, messages: List[dict]) -> Optional[float]:
    for msg in messages:
        if msg["user_id"] != user_id:
            continue
        if msg.get("_rent_increase_pct"):
            return msg["_rent_increase_pct"]
    return None


def get_pending_not_available(user_id: str, messages: List[dict]) -> Set[str]:
    result = set()
    for msg in messages:
        if msg["user_id"] != user_id:
            continue
        if msg.get("_pending_not_available"):
            related = msg.get("related_event_id")
            cls = msg.get("_classification", {})
            if related:
                result.add(related)
            if cls.get("affects_event_id"):
                result.add(cls["affects_event_id"])
    return result


def build_forecast_timeline(user_id: str, request_date: date,
                            data_store: DataStore,
                            messages: List[dict]) -> Dict[date, float]:
    """Build day-by-day balance changes for 90 days from request_date."""
    profile = data_store.profiles[user_id]
    home_ccy = profile["home_currency"]
    user_events = data_store.events_by_user.get(user_id, [])
    end_date = request_date + timedelta(days=90)
    timeline: Dict[date, float] = defaultdict(float)

    salary_info = get_salary_info_from_messages(user_id, messages, request_date)
    rent_increase_pct = get_rent_increase_from_messages(user_id, messages)
    pending_not_available = get_pending_not_available(user_id, messages)

    # ── 1. Detect and project recurring ──
    recurring_templates, recurring_ids = detect_all_recurring(user_events, request_date)
    
    recurring_timeline = aggregate_category_recurring(
        recurring_templates, request_date, end_date, home_ccy,
        data_store, salary_info, rent_increase_pct
    )
    
    for d, change in recurring_timeline.items():
        timeline[d] += change

    # ── 2. Non-recurring future events ──
    for e in user_events:
        if e["event_id"] in recurring_ids:
            continue
        if e["direction"] == "non_cash":
            continue
        if e["status"] in ("cancelled", "failed", "unrealized"):
            continue
        if not e["settlement_date"]:
            continue
        if e["settlement_date"] < request_date:
            continue
        if e["settlement_date"] > end_date:
            continue
        if e["event_id"] in pending_not_available and e["direction"] == "credit":
            continue

        amount = e["amount"]
        if amount is None:
            continue

        if e["currency"] != home_ccy:
            amount = data_store.convert_currency(amount, e["currency"],
                                                  home_ccy, e["settlement_date"])

        if e["direction"] == "debit":
            if e["status"] in ("settled", "pending", "scheduled"):
                timeline[e["settlement_date"]] -= amount
        elif e["direction"] == "credit":
            if e["status"] == "settled":
                timeline[e["settlement_date"]] += amount
            elif e["status"] in ("pending", "scheduled"):
                if e["category"] == "salary":
                    sal_amt = amount
                    if salary_info["amount_override"] is not None:
                        sal_amt = salary_info["amount_override"]
                        if e["currency"] != home_ccy:
                            sal_amt = data_store.convert_currency(
                                sal_amt, e["currency"], home_ccy, e["settlement_date"])
                    sdate = e["settlement_date"]
                    if salary_info["date_override"]:
                        sdate = salary_info["date_override"]
                    # Check not already covered by recurring projection
                    has_recurring_salary = any(t["category"] == "salary" 
                                               for t in recurring_templates
                                               if not salary_info.get("is_cancelled"))
                    if not has_recurring_salary:
                        if request_date <= sdate <= end_date:
                            timeline[sdate] += sal_amt
                # Other pending credits — don't count

    return dict(timeline)


def compute_amount_safe_to_pay(balance: float, min_balance: float,
                                timeline: Dict[date, float],
                                request_date: date,
                                requested_amount: float) -> float:
    """Max amount payable on request_date keeping balance >= min_balance for 90 days."""
    end_date = request_date + timedelta(days=90)
    sorted_dates = sorted(d for d in timeline.keys() if request_date <= d <= end_date)

    def is_safe(payment: float) -> bool:
        current = balance - payment
        if current < min_balance:
            return False
        for d in sorted_dates:
            current += timeline[d]
            if current < min_balance:
                return False
        return True

    if not is_safe(0.0):
        return 0.0
    if is_safe(requested_amount):
        return requested_amount

    lo, hi = 0.0, requested_amount
    for _ in range(60):
        mid = (lo + hi) / 2
        if is_safe(mid):
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.005:
            break

    return min(int(lo * 100) / 100.0, requested_amount)


def compute_earliest_full_payment_date(balance: float, min_balance: float,
                                        timeline: Dict[date, float],
                                        request_date: date,
                                        requested_amount: float) -> Optional[date]:
    """First date when paying full amount is safe for remaining 90 days."""
    end_date = request_date + timedelta(days=90)
    sorted_dates = sorted(d for d in timeline.keys() if request_date <= d <= end_date)

    def is_safe_on_date(pay_date: date) -> bool:
        current = balance
        paid = False
        all_dates = sorted(set(sorted_dates) | {pay_date})
        for d in all_dates:
            if d < request_date or d > end_date:
                continue
            if d in timeline:
                current += timeline[d]
            if d == pay_date:
                current -= requested_amount
                paid = True
            if paid and current < min_balance:
                return False
        return paid

    d = request_date
    while d <= end_date:
        if is_safe_on_date(d):
            return d
        d += timedelta(days=1)
    return None


def get_flexible_recurring_events(user_id: str, request_date: date,
                                   data_store: DataStore) -> List[dict]:
    """Get flexible recurring events that can be stopped or reduced."""
    profile = data_store.profiles[user_id]
    user_events = data_store.events_by_user.get(user_id, [])
    protect_cats = set(profile["expense_categories_to_protect"])
    reduce_cats = set(profile["expense_categories_user_is_willing_to_reduce"])
    stop_cats = set(profile["expense_categories_user_is_willing_to_stop"])

    recurring_templates, _ = detect_all_recurring(user_events, request_date)

    flexible = []
    for tmpl in recurring_templates:
        cat = tmpl["category"]
        flex = tmpl.get("flexibility")
        if not flex or cat in protect_cats or tmpl["direction"] != "debit":
            continue
        can_stop = flex in ("stoppable", "reducible_or_stoppable") and cat in stop_cats
        can_reduce = flex in ("reducible", "reducible_or_stoppable") and cat in reduce_cats
        if can_stop or can_reduce:
            tmpl["_can_stop"] = can_stop
            tmpl["_can_reduce"] = can_reduce
            tmpl["_min_amount"] = tmpl.get("minimum_allowed_amount")
            flexible.append(tmpl)

    return flexible


def compute_with_spending_changes(balance: float, min_balance: float,
                                   timeline: Dict[date, float],
                                   request_date: date,
                                   requested_amount: float,
                                   flexible_events: List[dict],
                                   data_store: DataStore,
                                   home_ccy: str) -> Tuple[Optional[float], Optional[str], Optional[date]]:
    """Try spending changes to improve affordability."""
    if not flexible_events:
        return None, None, None

    end_date = request_date + timedelta(days=90)

    scored = []
    for tmpl in flexible_events:
        amount = tmpl.get("_forecast_amount", tmpl["_latest_amount"])
        if tmpl["currency"] != home_ccy:
            amount = data_store.convert_currency(amount, tmpl["currency"],
                                                  home_ccy, request_date)
        interval = tmpl["_recurrence_interval_days"]
        occurrences = max(1, 90 // interval)
        if tmpl.get("_can_stop"):
            savings = amount * occurrences
        elif tmpl.get("_can_reduce"):
            min_amt = tmpl.get("_min_amount") or 0
            if tmpl["currency"] != home_ccy:
                min_amt = data_store.convert_currency(min_amt, tmpl["currency"],
                                                       home_ccy, request_date)
            savings = (amount - min_amt) * occurrences
        else:
            savings = 0
        scored.append((savings, amount, tmpl))

    scored.sort(reverse=True, key=lambda x: x[0])

    best_changes = []
    modified_timeline = dict(timeline)

    for savings, amount, tmpl in scored[:3]:
        if len(best_changes) >= 3:
            break
        event_id = tmpl["event_id"]
        interval = tmpl["_recurrence_interval_days"]
        profile = data_store.profiles[tmpl["user_id"]]
        cat = tmpl["category"]

        last_settled = max(tmpl["_recurrence_dates"])
        next_d = last_settled + timedelta(days=interval)
        while next_d < request_date:
            next_d += timedelta(days=interval)

        if tmpl.get("_can_stop") and cat in set(profile["expense_categories_user_is_willing_to_stop"]):
            d = next_d
            while d <= end_date:
                modified_timeline[d] = modified_timeline.get(d, 0) + amount
                d += timedelta(days=interval)
            best_changes.append(f"stop:{event_id}")
        elif tmpl.get("_can_reduce"):
            min_amt = tmpl.get("_min_amount") or 0
            orig_min = min_amt
            if tmpl["currency"] != home_ccy:
                min_amt = data_store.convert_currency(min_amt, tmpl["currency"],
                                                       home_ccy, request_date)
            reduction = amount - min_amt
            if reduction > 0:
                d = next_d
                while d <= end_date:
                    modified_timeline[d] = modified_timeline.get(d, 0) + reduction
                    d += timedelta(days=interval)
                best_changes.append(f"reduce_to:{event_id}:{orig_min}")

    if not best_changes:
        return None, None, None

    new_safe = compute_amount_safe_to_pay(balance, min_balance, modified_timeline,
                                           request_date, requested_amount)
    new_earliest = compute_earliest_full_payment_date(balance, min_balance, modified_timeline,
                                                       request_date, requested_amount)
    return new_safe, "|".join(best_changes), new_earliest
