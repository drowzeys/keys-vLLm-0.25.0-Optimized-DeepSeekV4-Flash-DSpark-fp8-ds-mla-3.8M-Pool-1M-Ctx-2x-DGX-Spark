# Benchmarks & evals — 1M `nvfp4_ds_mla` DSV4-Flash-DSpark

## Re-run against a live serve

```bash
# stack already up via scripts/dsv4-nvfp4-1m-serve.sh
python3 benchmarks/bench_eval.py \
  --api http://<rank0-ip>:8000/v1 \
  --out benchmarks/results/$(date -u +%Y-%m-%dT%H%MZ)-nvfp4-1m.json
```

Outputs C1×5 pure/wall tok/s, C4 aggregate, math battery, tools smoke, code smoke.

## Published runs (this cluster, 2× GB10 TP=2 RoCE)

| id | when (UTC) | pure C1 mean / peak | C4 agg | math | tools | raw |
|---|---|---:|---:|---|---|---|
| Session A | 2026-07-09 ~22:32Z | **56.2 / 58.5** | — | easy ✓ hard ✗ | ✓ | [RESULTS-nvfp4-1m.md](../RESULTS-nvfp4-1m.md) |
| Session B | 2026-07-09 ~23:08Z | **50.1 / 51.5** | ~65 | easy ✓ hard ✗ | ✓ | [RESULTS-nvfp4-1m.md](../RESULTS-nvfp4-1m.md) |
| **Session C (publish re-measure)** | **2026-07-09 23:19Z** | **43.6 / 56.6** | **64.5** | **5/5 ✓** | **✓** | [json](results/2026-07-09T2319Z-nvfp4-1m-session-c.json) · [md](results/2026-07-09T2319Z-nvfp4-1m-session-c.md) |

Session spread is real (prompt mix, thermal, page cache, concurrent load). Publish band for pure C1: **~44–56 mean class, peak ~56–58**.

## Methodology

- Endpoint: OpenAI chat completions, `thinking=false`, `temperature=0`, `ignore_eos` not required when `max_tokens` hits length.
- **Pure decode:** `(completion_tokens − 1) / (t_end − t_first_content_token)` over SSE stream.
- **Wall:** `completion_tokens / (t_end − t_request_start)` (includes TTFT).
- C4: 4 concurrent 128-token streams; aggregate = total completion tokens / wall clock of the batch.
