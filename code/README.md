# Buy or Wait? — Financial Decision Agent

## Setup

```bash
# Install dependencies
pip3 install -r code/requirements.txt

# Set API key (optional — for LLM-enhanced explanations and image/message extraction)
export NVIDIA_API_KEY="your-key"   # NVIDIA NIM API
# OR
export GROQ_API_KEY="your-key"     # Groq API
# OR
export OPENAI_API_KEY="your-key"   # OpenAI API
```

## Run

```bash
# Full 250-request run (no LLM — deterministic only)
python3 code/main.py --no-llm

# Full run with LLM-enhanced extraction + explanations
python3 code/main.py

# Sample-only run (25 requests) for testing
python3 code/main.py --sample --no-llm

# Custom output path
python3 code/main.py --output my_output.csv
```

## Evaluate against samples

```bash
python3 code/evaluation/score_samples.py
```

## Architecture

**Deterministic financial-forecasting engine** with optional LLM for image/message extraction and explanation generation.

```
code/
├── main.py              # Entry point — orchestrates 7-step pipeline
├── ingest.py            # Step 1: Load & normalize CSVs, currency conversion
├── llm_extract.py       # Step 2: Image OCR + message classification (with caching)
├── forecast.py          # Step 3: Recurring detection, 90-day balance forecast
├── plan_ranker.py       # Step 4: Enumerate & rank eligible payment plans
├── verify.py            # Step 6: Deterministic invariant checks
├── output_writer.py     # Step 7: Write output.csv
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── evaluation/
    ├── score_samples.py  # Diff predictions vs sample_requests.csv
    └── usage_report.md   # Token usage summary
```

### Pipeline

1. **Ingest** — Load all CSVs, parse dates, build exchange rate lookup with multi-hop conversion
2. **Extract** — OCR images (16 pay slips/receipts), classify messages (215), apply amendments
3. **Forecast** — Detect recurring expenses/income, build 90-day balance forecast per user
4. **Rank** — Enumerate full_payment/installments/partial_payment/wait/not_recommended, rank by rules
5. **Explain** — Generate decision_explanation (LLM or template-based fallback)
6. **Verify** — Check all invariants before writing
7. **Output** — Write output.csv with exact column order

### Key design decisions

- **Recurring detection** groups by (category, description) with consistency checks on intervals
- **Recency filter** only projects patterns whose last occurrence is within ~2.5 intervals of request_date
- **Variable spending** (groceries, transport) detected per-description to avoid over-aggregation
- **Currency conversion** supports multi-hop (via USD or EUR intermediary)
- **Message effects** parsed for salary amendments, rent increases, cancellations, pending status
- **Binary search** for amount_safe_to_pay with 0.005 precision
- **All results cached** in .cache/ for deterministic re-runs
