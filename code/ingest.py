"""
Step 1 — Ingest & normalize all CSV datasets.
Builds lookup structures and currency conversion utilities.
"""

import csv
import os
from datetime import datetime, date, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any


def parse_date(s: str) -> Optional[date]:
    """Parse YYYY-MM-DD date string."""
    if not s or not s.strip():
        return None
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_float(s: str) -> Optional[float]:
    """Parse a float, returning None for blank/invalid."""
    if not s or not s.strip():
        return None
    try:
        return float(s.strip())
    except ValueError:
        return None


def parse_int(s: str) -> Optional[int]:
    """Parse an int, returning None for blank/invalid."""
    if not s or not s.strip():
        return None
    try:
        return int(s.strip())
    except ValueError:
        return None


def parse_pipe_list(s: str) -> List[str]:
    """Parse pipe-separated list like 'a|b|c' into ['a','b','c']."""
    if not s or not s.strip():
        return []
    return [x.strip() for x in s.split("|") if x.strip()]


class DataStore:
    """Central data store holding all loaded and indexed dataset."""

    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = dataset_dir

        # Raw data
        self.profiles: Dict[str, dict] = {}
        self.events: List[dict] = []
        self.events_by_user: Dict[str, List[dict]] = defaultdict(list)
        self.events_by_id: Dict[str, dict] = {}
        self.exchange_rates: Dict[Tuple[str, str, str], float] = {}  # (date_str, from, to) -> rate
        self.rate_dates_by_pair: Dict[Tuple[str, str], List[date]] = defaultdict(list)
        self.requests: List[dict] = []
        self.sample_requests: List[dict] = []
        self.payment_options: List[dict] = []
        self.options_by_request: Dict[str, List[dict]] = defaultdict(list)
        self.messages: List[dict] = []
        self.messages_by_user: Dict[str, List[dict]] = defaultdict(list)
        self.images: List[dict] = []
        self.images_by_event: Dict[str, dict] = {}

    def load_all(self):
        """Load all CSV files from dataset directory."""
        self._load_profiles()
        self._load_events()
        self._load_exchange_rates()
        self._load_requests()
        self._load_sample_requests()
        self._load_payment_options()
        self._load_messages()
        self._load_images()
        return self

    def _load_profiles(self):
        path = os.path.join(self.dataset_dir, "financial_profiles.csv")
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                uid = row["user_id"].strip()
                self.profiles[uid] = {
                    "user_id": uid,
                    "home_currency": row["home_currency"].strip(),
                    "current_available_balance": float(row["current_available_balance"]),
                    "minimum_balance_to_keep": float(row["minimum_balance_to_keep"]),
                    "financial_priorities": parse_pipe_list(row.get("financial_priorities", "")),
                    "expense_categories_to_protect": parse_pipe_list(row.get("expense_categories_to_protect", "")),
                    "expense_categories_user_is_willing_to_reduce": parse_pipe_list(row.get("expense_categories_user_is_willing_to_reduce", "")),
                    "expense_categories_user_is_willing_to_stop": parse_pipe_list(row.get("expense_categories_user_is_willing_to_stop", "")),
                    "payment_methods_user_will_consider": parse_pipe_list(row.get("payment_methods_user_will_consider", "")),
                    "max_installment_months": parse_int(row.get("max_installment_months", "")),
                }

    def _load_events(self):
        path = os.path.join(self.dataset_dir, "financial_events.csv")
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                evt = {
                    "event_id": row["event_id"].strip(),
                    "user_id": row["user_id"].strip(),
                    "event_type": row["event_type"].strip(),
                    "description": row["description"].strip(),
                    "category": row["category"].strip(),
                    "direction": row["direction"].strip(),
                    "amount": parse_float(row["amount"]),
                    "currency": row["currency"].strip(),
                    "event_date": parse_date(row["event_date"]),
                    "settlement_date": parse_date(row["settlement_date"]),
                    "status": row["status"].strip(),
                    "linked_event_id": row["linked_event_id"].strip() if row["linked_event_id"].strip() else None,
                    "flexibility": row["flexibility"].strip() if row["flexibility"].strip() else None,
                    "minimum_allowed_amount": parse_float(row.get("minimum_allowed_amount", "")),
                }
                self.events.append(evt)
                self.events_by_user[evt["user_id"]].append(evt)
                self.events_by_id[evt["event_id"]] = evt

    def _load_exchange_rates(self):
        path = os.path.join(self.dataset_dir, "exchange_rates.csv")
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rate_date = parse_date(row["rate_date"])
                from_ccy = row["from_currency"].strip()
                to_ccy = row["to_currency"].strip()
                rate = float(row["rate"])
                key = (row["rate_date"].strip(), from_ccy, to_ccy)
                self.exchange_rates[key] = rate
                pair = (from_ccy, to_ccy)
                if rate_date not in self.rate_dates_by_pair[pair]:
                    self.rate_dates_by_pair[pair].append(rate_date)

        # Sort rate dates for each pair
        for pair in self.rate_dates_by_pair:
            self.rate_dates_by_pair[pair].sort()

    def _load_requests(self):
        path = os.path.join(self.dataset_dir, "requests.csv")
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                req = {
                    "request_id": row["request_id"].strip(),
                    "user_id": row["user_id"].strip(),
                    "request_date": parse_date(row["request_date"]),
                    "request_type": row["request_type"].strip(),
                    "requested_amount": float(row["requested_amount"]),
                    "desired_completion_date": parse_date(row["desired_completion_date"]),
                    "allows_partial_payment": row["allows_partial_payment"].strip().lower() == "true",
                    "request_text": row["request_text"].strip(),
                }
                self.requests.append(req)

    def _load_sample_requests(self):
        path = os.path.join(self.dataset_dir, "sample_requests.csv")
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                req = {
                    "request_id": row["request_id"].strip(),
                    "user_id": row["user_id"].strip(),
                    "request_date": parse_date(row["request_date"]),
                    "request_type": row["request_type"].strip(),
                    "requested_amount": float(row["requested_amount"]),
                    "desired_completion_date": parse_date(row["desired_completion_date"]),
                    "allows_partial_payment": row["allows_partial_payment"].strip().lower() == "true",
                    "request_text": row["request_text"].strip(),
                    # Expected output columns
                    "expected_amount_safe_to_pay": parse_float(row.get("amount_safe_to_pay", "")),
                    "expected_affordability_status": row.get("affordability_status", "").strip(),
                    "expected_recommended_payment_method": row.get("recommended_payment_method", "").strip(),
                    "expected_payment_plan": row.get("payment_plan", "").strip(),
                    "expected_earliest_date_for_full_payment": row.get("earliest_date_for_full_payment", "").strip(),
                    "expected_spending_changes_needed": row.get("spending_changes_needed", "").strip(),
                    "expected_decision_explanation": row.get("decision_explanation", "").strip(),
                }
                self.sample_requests.append(req)

    def _load_payment_options(self):
        path = os.path.join(self.dataset_dir, "request_payment_options.csv")
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                opt = {
                    "payment_option_id": row["payment_option_id"].strip(),
                    "request_id": row["request_id"].strip(),
                    "payment_method": row["payment_method"].strip(),
                    "payment_amount": float(row["payment_amount"]),
                    "number_of_payments": int(row["number_of_payments"]),
                    "first_payment_date": parse_date(row["first_payment_date"]),
                    "payment_frequency_days": parse_int(row.get("payment_frequency_days", "")),
                    "financing_fee": float(row["financing_fee"]),
                    "total_payable_amount": float(row["total_payable_amount"]),
                }
                self.payment_options.append(opt)
                self.options_by_request[opt["request_id"]].append(opt)

    def _load_messages(self):
        path = os.path.join(self.dataset_dir, "messages.csv")
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                msg = {
                    "message_id": row["message_id"].strip(),
                    "user_id": row["user_id"].strip(),
                    "request_id": row["request_id"].strip() if row["request_id"].strip() else None,
                    "related_event_id": row["related_event_id"].strip() if row["related_event_id"].strip() else None,
                    "sent_at": row["sent_at"].strip(),
                    "source_type": row["source_type"].strip(),
                    "message_text": row["message_text"].strip(),
                }
                self.messages.append(msg)
                self.messages_by_user[msg["user_id"]].append(msg)

    def _load_images(self):
        path = os.path.join(self.dataset_dir, "images.csv")
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                img = {
                    "image_id": row["image_id"].strip(),
                    "user_id": row["user_id"].strip(),
                    "request_id": row["request_id"].strip() if row["request_id"].strip() else None,
                    "related_event_id": row["related_event_id"].strip() if row["related_event_id"].strip() else None,
                }
                self.images.append(img)
                if img["related_event_id"]:
                    self.images_by_event[img["related_event_id"]] = img

    def convert_currency(self, amount: float, from_ccy: str, to_ccy: str,
                         ref_date: date) -> float:
        """Convert amount from one currency to another using exchange_rates.csv.
        Uses the closest rate_date on or before ref_date. If exact pair not found,
        tries reverse rate (1/rate). Same currency returns amount unchanged.
        """
        if from_ccy == to_ccy:
            return amount
        if amount is None:
            return 0.0

        # Find rate for the closest date on or before ref_date
        rate = self._find_rate(from_ccy, to_ccy, ref_date)
        if rate is not None:
            return amount * rate

        # Try reverse
        reverse_rate = self._find_rate(to_ccy, from_ccy, ref_date)
        if reverse_rate is not None and reverse_rate != 0:
            return amount / reverse_rate

        # Try two-hop via USD
        if from_ccy != "USD" and to_ccy != "USD":
            # from_ccy -> USD -> to_ccy
            rate1 = self._find_rate(from_ccy, "USD", ref_date)
            if rate1 is None:
                r = self._find_rate("USD", from_ccy, ref_date)
                if r is not None and r != 0:
                    rate1 = 1.0 / r
            rate2 = self._find_rate("USD", to_ccy, ref_date)
            if rate2 is None:
                r = self._find_rate(to_ccy, "USD", ref_date)
                if r is not None and r != 0:
                    rate2 = 1.0 / r
            if rate1 is not None and rate2 is not None:
                return amount * rate1 * rate2

        # Try two-hop via EUR
        if from_ccy != "EUR" and to_ccy != "EUR":
            rate1 = self._find_rate(from_ccy, "EUR", ref_date)
            if rate1 is None:
                r = self._find_rate("EUR", from_ccy, ref_date)
                if r is not None and r != 0:
                    rate1 = 1.0 / r
            rate2 = self._find_rate("EUR", to_ccy, ref_date)
            if rate2 is None:
                r = self._find_rate(to_ccy, "EUR", ref_date)
                if r is not None and r != 0:
                    rate2 = 1.0 / r
            if rate1 is not None and rate2 is not None:
                return amount * rate1 * rate2

        # Fallback: log warning and return original amount
        print(f"WARNING: No exchange rate found for {from_ccy}->{to_ccy} on {ref_date}")
        return amount

    def _find_rate(self, from_ccy: str, to_ccy: str, ref_date: date) -> Optional[float]:
        """Find rate for from_ccy->to_ccy on or before ref_date."""
        pair = (from_ccy, to_ccy)
        dates = self.rate_dates_by_pair.get(pair, [])
        if not dates:
            return None

        # Find closest date on or before ref_date
        best_date = None
        for d in dates:
            if d <= ref_date:
                best_date = d
            else:
                break

        # If no date on or before, use closest after
        if best_date is None:
            best_date = dates[0]

        key = (best_date.strftime("%Y-%m-%d"), from_ccy, to_ccy)
        return self.exchange_rates.get(key)
