# Results — 1M nvfp4_ds_mla + B12X DSpark (2× DGX Spark)

**Date:** 2026-07-09  
**Topology:** TP=2, nodes `10.100.10.3` (rank0 API :8000) + `10.100.10.4` (rank1), 200G RoCE  
**Image:** `vllm-dspark-runtime:dspark-nvfp4-stage-c`  
- local id `sha256:76532c4cc261…` (~22.7 GB)  
- GHCR: `ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10` (same id; see [image/STAGE-C.md](image/STAGE-C.md))  
**Engine:** vLLM `v0.21.1rc1.dev339+g1967a5627bc3` (aidendle/tonyd stage-c lineage, not stock 0.25 nightly)  
**Checkpoint:** DeepSeek-V4-Flash-DSpark safetensors (~157 GB) at `~/models/dsv4-flash-dspark`  
**Launcher:** `scripts/dsv4-nvfp4-1m-serve.sh`  
**Bench script:** `benchmarks/bench_eval.py`

## Serve config (standing)

| knob | value |
|---|---|
| `kv_cache_dtype` | **`nvfp4_ds_mla`** |
| MoE backend | **B12X** Mxfp4 (`VLLM_USE_B12X_MOE=1`) |
| Spec | DSpark `num_speculative_tokens=5` |
| CUDA graphs | FULL_AND_PIECEWISE (no `--enforce-eager`) |
| `max_model_len` | **1,048,576** |
| `max_num_seqs` | 12 |
| `gpu_memory_utilization` | 0.82 |
| `max_num_batched_tokens` | 8192 |
| `block_size` | 256 |
| thinking default | false |

Boot log markers (rank0, 2026-07-09 session):

```
Using probe DeepSeek V4 nvfp4_ds_mla KV cache format.
Using 'B12X' Mxfp4 MoE backend.
GPU KV cache size: 2,500,107 tokens
Maximum concurrency for 1,048,576 tokens per request: 2.38x
Graph capturing finished in 18 secs
Application startup complete.
```

## Throughput (C1, chat, `thinking=false`, 256 completion tokens)

Methodology: pure decode = `(n-1)/(t_end - t_first_content)` over streaming chat completions.
Reproduce: `python3 benchmarks/bench_eval.py --api http://<rank0>:8000/v1`.

### Session A (post-boot warm, 2026-07-09 ~22:32Z)

| run | pure tok/s | wall tok/s |
|---:|---:|---:|
| 1 | 55.62 | 52.25 |
| 2 | 58.49 | 54.76 |
| 3 | 55.00 | 52.65 |
| 4 | 55.76 | 53.46 |
| **mean / peak** | **56.22 / 58.49** | **53.3 / 54.8** |

### Session B (re-measure for publish, 2026-07-09 ~23:08Z)

| run | pure tok/s | wall tok/s | draft accept | τ (tok/draft) |
|---:|---:|---:|---:|---:|
| 1 | 48.08 | 45.97 | 0.622 | 4.06 |
| 2 | 50.90 | 48.16 | 0.664 | 4.34 |
| 3 | 51.51 | 48.95 | 0.690 | 4.41 |
| 4 | 50.64 | 48.85 | 0.663 | 4.27 |
| 5 | 49.46 | 47.17 | 0.652 | 4.20 |
| **mean / peak** | **50.11 / 51.51** | **47.8 / 49.0** | **0.658** | **4.26** |

### Session C (publish package re-measure, 2026-07-09 23:19Z)

Raw: [benchmarks/results/2026-07-09T2319Z-nvfp4-1m-session-c.json](benchmarks/results/2026-07-09T2319Z-nvfp4-1m-session-c.json)

| run | pure tok/s | wall tok/s | out_tok | prompt class |
|---:|---:|---:|---:|---|
| 1 | 44.42 | 42.46 | 253 | code (binary search) |
| 2 | 32.67 | 31.74 | 256 | prose (NCCL/RoCE) |
| 3 | 48.83 | 46.89 | 256 | bash/rsync |
| 4 | 56.60 | 54.52 | 256 | code (LRU) |
| 5 | 35.69 | 34.37 | 256 | systems (nvfp4 vs fp8) |
| **mean / peak** | **43.64 / 56.60** | **42.0 / 54.5** | | |

Session spread is real (prompt mix, load, thermal/memory pressure). Publish as:

- **Best observed pure C1: ~56–58 tok/s** (Sessions A/C peaks)
- **Typical re-measure pure C1: ~44–51 tok/s mean** depending on prompt mix
- **Code-heavy prompts** land in the high 50s; long prose can dip to low 30s
- **Draft acceptance ~66% @ temp 0** (Session B, code-completion prompts)

### Concurrency

| metric | value | session |
|---|---|---|
| C4 concurrent (4×128 tok, simultaneous) | **~65 tok/s aggregate** | B |
| C4 concurrent (4×128 tok) | **64.5 tok/s aggregate** (512 tok / 7.9s) | C |

## Quality / evals

### Session C battery (2026-07-09 23:19Z) — all green

| check | result |
|---|---|
| OpenAI tools (`get_weather` smoke) | **✓** `finish_reason=tool_calls` |
| Simple math `12×11` | **✓ 132** |
| Simple math `100−37` | **✓ 63** |
| Hard math `847×293` (expect 248171) | **✓ 248171** |
| Math `15+27` / `2**10` | **✓ 42 / 1024** |
| Code smoke (`s[::-1]`) | **✓** |

### Prior sessions (A/B) — hard-math flaky

| check | Session A | Session B |
|---|---|---|
| tools | ✓ | ✓ |
| easy math | ✓ | ✓ |
| hard `847×293` | ✗ 231371 | ✗ 247571 |

**Accuracy caveat (documented at boot):**

> Using nvfp4_ds_mla data type to store kv cache. It reduces the GPU memory footprint and boosts the performance. Meanwhile, **it may cause accuracy drop without a proper scaling factor**.

Session C shows hard math can pass; Sessions A/B show it can fail with small digit errors. Treat nvfp4_ds_mla as a **pool + long-context capacity + speed** win with **non-zero numeric risk**. For strict bit-stable arithmetic agents, dual-config against fp8_ds_mla / eugr graphs. Code/prose/tool paths were coherent in all smoke runs.

## Comparison vs prior champions in this repo

| stack | C1 pure/mean class | KV pool @1M | notes |
|---|---|---|---|
| Eager nightly + DSpark | ~33 | ~3.8M (fp8_ds_mla) | Wall-2 forces eager on stock nightly |
| eugr graphs + DSpark (fp8_ds_mla) | ~43 mean / 54 peak | ~3.08M | 0.25-line de-fork champion |
| **stage-c B12X + nvfp4_ds_mla 1M** | **~44–56 pure / ~58 peak** | **~2.5M** | This release; larger pool per-byte |

## Image / build notes

See **[image/STAGE-C.md](image/STAGE-C.md)** for GHCR pull, image identity, and lineage.

```bash
# both nodes
docker pull ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10
docker tag  ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10 \
  vllm-dspark-runtime:dspark-nvfp4-stage-c

# rank1 first, then rank0
bash scripts/dsv4-nvfp4-1m-serve.sh 1
bash scripts/dsv4-nvfp4-1m-serve.sh 0

# after Application startup complete:
python3 benchmarks/bench_eval.py --api http://<rank0>:8000/v1
```

Edit `MASTER` / `IF` / `HCA` at the top of the launcher for your fabric.
