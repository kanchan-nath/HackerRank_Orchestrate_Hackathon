"""
Step 7 — Write output.csv with exact column order.
"""

import csv
import os
from typing import List


OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


def write_output(results: List[dict], output_path: str = "output.csv"):
    """Write output.csv with exact column order and formatting."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for row in results:
            # Ensure all required fields
            out = {}
            for col in OUTPUT_COLUMNS:
                val = row.get(col, "")
                if val is None:
                    val = ""
                out[col] = val

            # Format amount_safe_to_pay
            safe = out["amount_safe_to_pay"]
            if isinstance(safe, (int, float)):
                # Use minimal decimal representation
                if safe == int(safe):
                    out["amount_safe_to_pay"] = str(int(safe))
                else:
                    out["amount_safe_to_pay"] = f"{safe:.2f}"
                    # Remove trailing zeros but keep at least .X format
                    out["amount_safe_to_pay"] = out["amount_safe_to_pay"].rstrip("0").rstrip(".")
                    if "." not in str(out["amount_safe_to_pay"]) and safe != int(safe):
                        out["amount_safe_to_pay"] = f"{safe}"

            writer.writerow(out)

    print(f"Written {len(results)} rows to {output_path}")
    print(f"Columns: {','.join(OUTPUT_COLUMNS)}")
