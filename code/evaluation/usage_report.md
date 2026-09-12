# Token Usage Report — Final Full-Dataset Run

## Run Details
- **Date**: 2026-09-12
- **Mode**: `--no-llm` (deterministic only, no LLM API calls)
- **Dataset**: 250 requests from `dataset/requests.csv`
- **Runtime**: 0.3 seconds

## Model Information
- **Primary mode**: Deterministic rule-based engine (no LLM required)
- **Optional LLM provider**: NVIDIA NIM API / Groq / OpenAI (OpenAI-compatible)
- **Vision model** (if used): `meta/llama-3.2-90b-vision-instruct` (NVIDIA) / `llama-3.2-90b-vision-preview` (Groq)
- **Text model** (if used): `meta/llama-3.3-70b-instruct` (NVIDIA) / `llama-3.3-70b-versatile` (Groq)

## Token Usage Summary (No-LLM Mode)

| Category | Calls | Input Tokens | Output Tokens | Total Tokens |
|---|---|---|---|---|
| Image extraction | 0 | 0 | 0 | 0 |
| Message classification | 0 | 0 | 0 | 0 |
| Explanation generation | 0 | 0 | 0 | 0 |
| **Total** | **0** | **0** | **0** | **0** |

## Token Usage Summary (With LLM — estimated)

If LLM mode is enabled, estimated usage would be:

| Category | Calls | Est. Input Tokens | Est. Output Tokens | Est. Total |
|---|---|---|---|---|
| Image extraction | 16 | ~32,000 | ~1,600 | ~33,600 |
| Message classification | 215 | ~64,500 | ~21,500 | ~86,000 |
| Explanation generation | 250 | ~75,000 | ~12,500 | ~87,500 |
| **Total** | **481** | **~171,500** | **~35,600** | **~207,100** |

### Estimated Cost (with LLM)
- **Average tokens per request**: ~828
- **Estimated per-request cost**: ~$0.0002 (Groq free tier) / ~$0.001 (NVIDIA free tier)
- **Estimated total cost**: ~$0.05 (Groq) / ~$0.25 (NVIDIA) / ~$0.02 (GPT-4o-mini)

## Notes
- The deterministic-only run produces valid output without any API calls
- All LLM results are cached in `.cache/` for reproducible re-runs
- Temperature is set to 0 for deterministic LLM outputs
- No API keys, credentials, or sensitive configuration values included
