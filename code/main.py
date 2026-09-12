"""
Buy or Wait? — AI-powered financial decision agent.
Main entry point: orchestrates the full pipeline.

Usage:
    python3 code/main.py                    # Process full requests.csv
    python3 code/main.py --sample           # Process sample_requests.csv only
    python3 code/main.py --no-llm           # Skip LLM calls (use cached/manual)
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()
import json
import time
import argparse
from datetime import datetime

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from ingest import DataStore
from llm_extract import (
    extract_image_amounts, apply_image_extractions,
    classify_messages, apply_message_effects,
    generate_explanation, get_token_usage,
)
from forecast import build_forecast_timeline, compute_amount_safe_to_pay, compute_earliest_full_payment_date
from plan_ranker import process_request
from verify import verify_all
from output_writer import write_output


def main():
    parser = argparse.ArgumentParser(description="Buy or Wait? Financial Decision Agent")
    parser.add_argument("--sample", action="store_true", help="Process sample_requests.csv only")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM calls, use cached/manual fallback")
    parser.add_argument("--output", default="output.csv", help="Output file path")
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 60)
    print("Buy or Wait? — Financial Decision Agent")
    print("=" * 60)

    # Step 1: Load data
    print("\n[Step 1] Loading and normalizing data...")
    data_store = DataStore(dataset_dir="dataset")
    data_store.load_all()
    print(f"  Loaded {len(data_store.profiles)} profiles")
    print(f"  Loaded {len(data_store.events)} financial events")
    print(f"  Loaded {len(data_store.exchange_rates)} exchange rates")
    print(f"  Loaded {len(data_store.requests)} requests")
    print(f"  Loaded {len(data_store.sample_requests)} sample requests")
    print(f"  Loaded {len(data_store.payment_options)} payment options")
    print(f"  Loaded {len(data_store.messages)} messages")
    print(f"  Loaded {len(data_store.images)} images")

    # Step 2: Extract from images and classify messages
    print("\n[Step 2] Extracting data from images and classifying messages...")

    if not args.no_llm:
        # Image extraction
        print("  Extracting amounts from images...")
        image_extractions = extract_image_amounts(data_store, cache_dir=".cache")
        apply_image_extractions(data_store, image_extractions)
        print(f"  Extracted amounts from {len(image_extractions)} images")

        # Message classification
        print("  Classifying messages...")
        msg_classifications = classify_messages(data_store, cache_dir=".cache")
        apply_message_effects(data_store, msg_classifications)
        print(f"  Classified {len(msg_classifications)} messages")
    else:
        print("  Skipping LLM calls (--no-llm mode)")
        # Still apply manual parsing for messages
        from llm_extract import _manual_message_parse
        msg_classifications = {}
        for msg in data_store.messages:
            result = _manual_message_parse(msg)
            msg_classifications[msg["message_id"]] = result
            msg["_classification"] = result
        apply_message_effects(data_store, msg_classifications)

        # For images, try to load from cache
        cache_path = os.path.join(".cache", "image_extractions.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            image_extractions = {}
            for img in data_store.images:
                if img["image_id"] in cached and img["related_event_id"]:
                    image_extractions[img["related_event_id"]] = cached[img["image_id"]]
            apply_image_extractions(data_store, image_extractions)
            print(f"  Loaded {len(image_extractions)} cached image extractions")

    # Select requests to process
    if args.sample:
        requests = data_store.sample_requests
        print(f"\n[Processing] {len(requests)} sample requests")
    else:
        requests = data_store.requests
        print(f"\n[Processing] {len(requests)} requests")

    # Steps 3-5: Process each request
    print("\n[Steps 3-5] Forecasting, plan ranking, and decision generation...")
    results = []
    for i, req in enumerate(requests):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  Processing request {i+1}/{len(requests)}: {req['request_id']}")

        try:
            user_msgs = [m for m in data_store.messages if m["user_id"] == req["user_id"]]
            decision = process_request(req, data_store, user_msgs)

            # Step 5: Generate explanation
            if not args.no_llm:
                profile = data_store.profiles[req["user_id"]]
                key_facts = f"Balance: {profile['home_currency']} {profile['current_available_balance']:,.2f}, Min balance: {profile['home_currency']} {profile['minimum_balance_to_keep']:,.2f}"
                explanation = generate_explanation(decision, profile, key_facts)
                decision["decision_explanation"] = explanation
            else:
                profile = data_store.profiles[req["user_id"]]
                from llm_extract import _manual_explanation
                decision["decision_explanation"] = _manual_explanation(decision, profile)

            results.append(decision)

        except Exception as e:
            print(f"  ERROR processing {req['request_id']}: {e}")
            import traceback
            traceback.print_exc()
            # Add fallback row
            results.append({
                "request_id": req["request_id"],
                "amount_safe_to_pay": 0,
                "affordability_status": "not_affordable",
                "recommended_payment_method": "not_recommended",
                "payment_plan": "none",
                "earliest_date_for_full_payment": "",
                "spending_changes_needed": "none",
                "decision_explanation": "Unable to process this request.",
            })

    # Step 6: Verification
    print(f"\n[Step 6] Verifying {len(results)} results...")
    violations = verify_all(results, requests, data_store)
    if violations:
        print(f"  WARNING: {len(violations)} violations found:")
        for v in violations[:20]:
            print(f"    - {v}")
        if len(violations) > 20:
            print(f"    ... and {len(violations) - 20} more")
    else:
        print("  All invariants passed!")

    # Step 7: Write output
    print(f"\n[Step 7] Writing output to {args.output}...")
    write_output(results, args.output)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Done! Processed {len(results)} requests in {elapsed:.1f}s")
    print(f"Output: {args.output}")

    # Print token usage
    usage = get_token_usage()
    total_calls = sum(v["calls"] for v in usage.values())
    total_input = sum(v["input_tokens"] for v in usage.values())
    total_output = sum(v["output_tokens"] for v in usage.values())
    print(f"\nToken usage:")
    print(f"  Total calls: {total_calls}")
    print(f"  Input tokens: {total_input}")
    print(f"  Output tokens: {total_output}")
    print(f"  Total tokens: {total_input + total_output}")
    if len(requests) > 0:
        print(f"  Avg tokens/request: {(total_input + total_output) / len(requests):.0f}")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    main()
