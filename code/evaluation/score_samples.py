"""
Evaluation script: run pipeline against sample_requests.csv and
compare predictions vs expected values field-by-field.

Usage:
    python3 code/evaluation/score_samples.py
"""

import os
import sys
import csv
from datetime import date

# Add code dir to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ingest import DataStore, parse_float, parse_date
from llm_extract import (
    extract_image_amounts, apply_image_extractions,
    classify_messages, apply_message_effects, _manual_message_parse, _manual_explanation
)
from plan_ranker import process_request
from verify import verify_all


def score():
    print("=" * 70)
    print("Scoring against sample_requests.csv (25 requests)")
    print("=" * 70)

    # Load data
    data_store = DataStore(dataset_dir="dataset")
    data_store.load_all()

    # Apply cached image extractions if available
    import json
    cache_path = os.path.join(".cache", "image_extractions.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        image_extractions = {}
        for img in data_store.images:
            if img["image_id"] in cached and img["related_event_id"]:
                image_extractions[img["related_event_id"]] = cached[img["image_id"]]
        apply_image_extractions(data_store, image_extractions)

    # Manual message classification
    msg_classifications = {}
    for msg in data_store.messages:
        result = _manual_message_parse(msg)
        msg_classifications[msg["message_id"]] = result
        msg["_classification"] = result
    apply_message_effects(data_store, msg_classifications)

    # Process each sample request
    results = []
    for req in data_store.sample_requests:
        user_msgs = [m for m in data_store.messages if m["user_id"] == req["user_id"]]
        try:
            decision = process_request(req, data_store, user_msgs)
            profile = data_store.profiles[req["user_id"]]
            decision["decision_explanation"] = _manual_explanation(decision, profile)
            results.append(decision)
        except Exception as e:
            print(f"  ERROR: {req['request_id']}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "request_id": req["request_id"],
                "amount_safe_to_pay": 0,
                "affordability_status": "not_affordable",
                "recommended_payment_method": "not_recommended",
                "payment_plan": "none",
                "earliest_date_for_full_payment": "",
                "spending_changes_needed": "none",
                "decision_explanation": "Error processing request.",
            })

    # Compare field by field
    fields = [
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
    ]
    numeric_fields = ["amount_safe_to_pay"]

    field_correct = {f: 0 for f in fields + numeric_fields}
    field_total = {f: 0 for f in fields + numeric_fields}
    mismatches = []

    for req, pred in zip(data_store.sample_requests, results):
        rid = req["request_id"]

        # Numeric: amount_safe_to_pay
        expected_safe = req.get("expected_amount_safe_to_pay")
        pred_safe = pred.get("amount_safe_to_pay", 0)
        if expected_safe is not None:
            field_total["amount_safe_to_pay"] += 1
            # Allow 1% tolerance or absolute difference of 1
            tolerance = max(1.0, abs(expected_safe) * 0.01)
            if abs(pred_safe - expected_safe) <= tolerance:
                field_correct["amount_safe_to_pay"] += 1
            else:
                mismatches.append(
                    f"{rid}: amount_safe_to_pay expected={expected_safe}, got={pred_safe} "
                    f"(diff={abs(pred_safe - expected_safe):.2f})"
                )

        # Categorical fields
        for field in fields:
            expected = req.get(f"expected_{field}", "")
            predicted = str(pred.get(field, ""))

            field_total[field] += 1

            if field == "payment_plan":
                # Compare plan structure, not exact amounts
                if expected == predicted:
                    field_correct[field] += 1
                elif expected == "none" and predicted == "none":
                    field_correct[field] += 1
                else:
                    # Check if dates match and amounts are close
                    if _plans_match(expected, predicted):
                        field_correct[field] += 1
                    else:
                        mismatches.append(f"{rid}: {field} expected='{expected}', got='{predicted}'")
            elif field == "earliest_date_for_full_payment":
                # Allow exact match or ±1 day
                if expected == predicted:
                    field_correct[field] += 1
                elif expected and predicted:
                    try:
                        e_date = parse_date(expected)
                        p_date = parse_date(predicted)
                        if e_date and p_date and abs((e_date - p_date).days) <= 1:
                            field_correct[field] += 1
                        else:
                            mismatches.append(f"{rid}: {field} expected='{expected}', got='{predicted}'")
                    except:
                        mismatches.append(f"{rid}: {field} expected='{expected}', got='{predicted}'")
                elif not expected and not predicted:
                    field_correct[field] += 1
                else:
                    mismatches.append(f"{rid}: {field} expected='{expected}', got='{predicted}'")
            else:
                if expected == predicted:
                    field_correct[field] += 1
                else:
                    mismatches.append(f"{rid}: {field} expected='{expected}', got='{predicted}'")

    # Print results
    print(f"\n{'Field':<40} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 70)
    all_fields = numeric_fields + fields
    total_correct = 0
    total_total = 0
    for f in all_fields:
        c = field_correct[f]
        t = field_total[f]
        acc = f"{c/t*100:.1f}%" if t > 0 else "N/A"
        print(f"{f:<40} {c:<10} {t:<10} {acc:<10}")
        total_correct += c
        total_total += t

    print("-" * 70)
    overall = f"{total_correct/total_total*100:.1f}%" if total_total > 0 else "N/A"
    print(f"{'OVERALL':<40} {total_correct:<10} {total_total:<10} {overall:<10}")

    if mismatches:
        print(f"\nMismatches ({len(mismatches)}):")
        for m in mismatches:
            print(f"  ✗ {m}")

    # Run verification
    violations = verify_all(results, data_store.sample_requests, data_store)
    if violations:
        print(f"\nVerification violations ({len(violations)}):")
        for v in violations:
            print(f"  ⚠ {v}")
    else:
        print("\n✓ All verification invariants passed!")

    return field_correct, field_total, mismatches


def _plans_match(expected: str, predicted: str) -> bool:
    """Check if two payment plans are approximately equal."""
    if not expected or not predicted:
        return expected == predicted

    e_parts = expected.split("|")
    p_parts = predicted.split("|")

    if len(e_parts) != len(p_parts):
        return False

    for ep, pp in zip(e_parts, p_parts):
        try:
            e_date, e_amt = ep.split(":")
            p_date, p_amt = pp.split(":")

            # Dates must match
            if e_date != p_date:
                return False

            # Amounts within 1% tolerance
            e_val = float(e_amt)
            p_val = float(p_amt)
            tol = max(1.0, abs(e_val) * 0.01)
            if abs(e_val - p_val) > tol:
                return False
        except:
            return ep == pp

    return True


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    score()
