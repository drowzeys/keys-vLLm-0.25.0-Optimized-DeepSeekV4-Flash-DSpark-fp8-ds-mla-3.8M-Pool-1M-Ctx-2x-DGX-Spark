# Results — 1M nvfp4_ds_mla + B12X DSpark (2× DGX Spark)

**Date:** 2026-07-09  
**Topology:** TP=2, nodes `10.100.10.3` (rank0 API :8000) + `10.100.10.4` (rank1), 200G RoCE  
**Image:** `vllm-dspark-runtime:dspark-nvfp4-stage-c` (local tag; digest lineage aidendle94/tonyd stage-c, ~22.7 GB)  
**Checkpoint:** DeepSeek-V4-Flash-DSpark safetensors (~157 GB) at `~/models/dsv4-flash-dspark`  
**Launcher:** `scripts/dsv4-nvfp4-1m-serve.sh`

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

Boot log markers:

```
Using probe DeepSeek V4 nvfp4_ds_mla KV cache format.
Using 'B12X' Mxfp4 MoE backend.
GPU KV cache size: 2,500,107 tokens
Maximum concurrency for 1,048,576 tokens per request: 2.38x
Application startup complete.
```

## Throughput (C1, chat, `thinking=false`, ignore_eos, 256 completion tokens)

Methodology: pure decode = `(n-1)/(t_end - t_first_content)` over streaming chat completions.

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

Session spread is real (load, page cache, thermal/memory pressure). Publish as:

- **Best observed pure C1: ~56–58 tok/s**
- **Typical re-measure pure C1: ~50–51 tok/s**
- **Draft acceptance ~66% @ temp 0** (code-completion prompts)

### Concurrency (Session B)

| metric | value |
|---|---|
| C4 concurrent (4×128 tok, simultaneous) | **~65 tok/s aggregate** (~16–20/stream) |

## Quality / evals

| check | result |
|---|---|
| OpenAI tools (`get_weather` smoke) | **✓** `finish_reason=tool_calls` |
| Simple math `12×11` | **✓ 132** |
| Simple math `100−37` | **✓ 63** |
| Hard math `847×293` (expect 248171) | **✗** Session A: 231371; Session B: 247571 |

**Accuracy caveat (documented at boot):**

> Using nvfp4_ds_mla data type to store kv cache. It reduces the GPU memory footprint and boosts the performance. Meanwhile, **it may cause accuracy drop without a proper scaling factor**.

Treat nvfp4_ds_mla as a **pool + long-context capacity** win. For strict numeric agent work, dual-config against fp8_ds_mla/eugr graphs if bit-exact digits matter. Code/prose/tool paths were coherent in all smoke runs.

## Comparison vs prior champions in this repo

| stack | C1 pure/mean class | KV pool @1M | notes |
|---|---|---|---|
| Eager nightly + DSpark | ~33 | ~3.8M (fp8_ds_mla) | Wall-2 forces eager on stock nightly |
| eugr graphs + DSpark (fp8_ds_mla) | ~43 mean / 54 peak | ~3.08M | 0.25-line de-fork champion |
| **stage-c B12X + nvfp4_ds_mla 1M** | **~50–56 pure / ~58 peak** | **~2.5M** | This release; larger pool per-byte; local image |

## Image / build notes

The stage-c image is **not** stock `vllm/vllm-openai:nightly`. It is the community aidendle94 / tonyd2wild GB10 stack with:

- DSpark speculative path + B12X Mxfp4 MoE
- `nvfp4_ds_mla` sparse-MLA KV probe path
- CUDA graphs that capture/replay on sm_121a

Local tag: `vllm-dspark-runtime:dspark-nvfp4-stage-c`  
Optional GHCR publish (if pushed): `ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10`

Reproduce serve:

```bash
# both nodes
# docker pull ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10   # if published
# docker tag ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10 vllm-dspark-runtime:dspark-nvfp4-stage-c

# rank1 first, then rank0
bash scripts/dsv4-nvfp4-1m-serve.sh 1
bash scripts/dsv4-nvfp4-1m-serve.sh 0
```

Edit `MASTER` / `IF` / `HCA` at the top of the launcher for your fabric.
