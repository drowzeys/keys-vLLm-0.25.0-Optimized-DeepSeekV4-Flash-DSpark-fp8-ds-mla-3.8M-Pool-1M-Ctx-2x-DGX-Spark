# DeepSeek-V4-Flash-DSpark on 2× DGX Spark — upstream vLLM 0.25, 33 tok/s C1, one 30-line patch

Serving recipe and measured benchmarks for **DeepSeek-V4-Flash-DSpark** (157 GB, NVIDIA's
DSpark speculative-decoding release of DSV4-Flash) on **2× NVIDIA DGX Spark (GB10, sm_121a,
128 GB unified each)** at tensor-parallel 2 — on the **upstream vLLM 0.25 line**, where DSpark
is now merged natively ([#46995](https://github.com/vllm-project/vllm/pull/46995),
[#47093](https://github.com/vllm-project/vllm/pull/47093),
[#47429](https://github.com/vllm-project/vllm/pull/47429)).

This retires the entire 0.24-era transplant stack (17-file overlay + hand-swapped flashinfer
`.so` + DeepGEMM transplant) in favor of: **stock nightly image + a 1-line pip bump + one
30-line, provably-safe patch** — and it's faster.

![C1 ladder](charts/c1-ladder.png)

## Measured (2× GB10, TP=2, `--enforce-eager`, DSpark k=5, fp8_ds_mla KV)

| Metric | Value |
|---|---|
| C1 decode (3× 512 tok, temp 0) | **37.0 / 33.2 / 29.1 tok/s (mean ~33)** |
| C1 no-spec baseline (same build) | 17.6 tok/s → **DSpark = +88%** |
| vs 0.24 transplant port (27 C1, graphs on) | **+22% — while running eager** |
| C16 aggregate (16× 256 tok, temp 0.7) | 69.0 tok/s (0.24 port: 77–106 — see graphs note) |
| DSpark acceptance | **42.0%** @ temp 0 (2.10 tok/draft) · 29.7% @ temp 0.7 under batch |
| KV pool | **2,838,963 tokens** @ 256K max ctx, GMU 0.85 |
| Coherence | ✓ all runs (base model: corpus-style continuations on bare prompts, clean prose in chat-shaped contexts) |

![DSpark speedup](charts/dspark-speedup.png)
![Acceptance](charts/acceptance.png)
![Runs](charts/runs.png)

## The two walls (root-caused — read before touching anything)

### Wall 1: warmup crash `Check failed: num_tokens > 64 (5 vs. 64)`

DSpark's non-causal draft pass sizes its SWA sparse-index width as
`cdiv(sliding_window + num_spec_tokens, 128) * 128`. DSV4-Flash has `sliding_window=128`, so
**any** `num_speculative_tokens` in 1..128 yields width **256**. flashinfer 0.6.14's standalone
sm120 decode kernel is an explicit instantiation switch over TOPK ∈ {128, 512, 1024} only
(`sparse_mla_sm120_decode_dsv4.cu` / `_DECODE_DSV4_DISPATCH`); an off-table width silently falls
through to the **prefill** orchestrator, which asserts `num_tokens > 64`. Bisecting
`SPEC_TOKENS` is analytically pointless — every k hits 256.

**Fix** (`patches/sparse_swa.py`, tag `PATCH(gb10-fi614)`): round the non-causal index width up
to the nearest instantiated width (256 → 512). Safe by construction: the Triton fill kernel
already writes `-1` beyond `swa_len` across the full width, and the decode kernel masks via
`topk_length`. (Equivalent upstream fix: add a `DSV4_DISPATCH(32, 256)` instantiation.)

### Wall 2: first sustained request hangs → `RPC call to sample_tokens timed out`

Reproduced with **spec decode fully off**: a 40-token smoke test works, the first 256-token
request wedges the rank-0 worker, and the engine dies at the 300 s RPC timeout. This is the
**CUDA-graphs decode path on sm_121a** (FULL_AND_PIECEWISE mode), not DSpark: `py-spy` shows the
surviving rank idle in `shm_broadcast` dequeue while the wedged rank sits in the full decode
graph replay. `--enforce-eager` eliminates it completely (256/512-tok and `ignore_eos` runs all
pass repeatedly). Untested lead for graph lovers: `-cc.cudagraph_mode=PIECEWISE` may salvage
graphs by skipping only the FULL decode graph — that experiment is the likely route past the
C16 gap vs the graphs-on 0.24 port.

## Step-by-step

### 1. Build the image (one pip bump over stock nightly)

The nightly ships flashinfer 0.6.13, but nightly vLLM's DSV4 sparse-MLA calls the 0.6.14 API
(`swa_topk_lens`, `extra_sparse_indices` — you'll get
`trtllm_batch_decode_sparse_mla_dsv4() got an unexpected keyword argument` on 0.6.13):

```bash
docker build -t vllm-dsv4-025:fi614 image/   # FROM vllm/vllm-openai:nightly-aarch64 + flashinfer 0.6.14
```

Tested against nightly `a23d8ade4ae3` (2026-07-08, vLLM `0.23.1rc1.dev925+g2afa3f7e9`).

### 2. Stage the model on both nodes

`~/models/dsv4-flash-dspark` (157 GB) — the DSpark speculators-format checkpoint. The launcher
bind-mounts it read-only at `/model`.

### 3. Launch (rank 1 on the worker node FIRST, then rank 0 = API node)

```bash
# worker:
IMG=vllm-dsv4-025:fi614 SPEC=dspark EAGER=1 PATCH_SWA=1 bash scripts/dsv4-025-serve-r34-mod.sh 1
# head (API on :8000):
IMG=vllm-dsv4-025:fi614 SPEC=dspark EAGER=1 PATCH_SWA=1 bash scripts/dsv4-025-serve-r34-mod.sh 0
```

`PATCH_SWA=1` bind-mounts `patches/sparse_swa.py` over
`vllm/v1/attention/backends/mla/sparse_swa.py`. Knobs: `SPEC(dspark|none)`, `SPEC_TOKENS(5)`,
`SEQS(16)`, `MAXLEN(262144)`, `KVD(fp8_ds_mla)`, `GMU(0.85)`. Adjust `MASTER`/`IF`/`HCA` to your
fabric (ours: 200G RoCE, NCCL IB GID 3).

### 4. Flags that matter

| Flag | Why |
|---|---|
| `--enforce-eager` | **mandatory** — Wall 2; the sm_121a full decode graph wedges the worker |
| `--speculative-config '{"method":"dspark","num_speculative_tokens":5}'` | upstream-native DSpark (forces Model-Runner-V2) |
| `--kv-cache-dtype fp8_ds_mla` | 2.84M-token KV pool at 256K ctx / GMU 0.85 |
| `--gpu-memory-utilization 0.85` | house rule for 128 GB unified GB10; higher risks OOM-livelock |
| `--no-enable-prefix-caching` | as benched |

### 5. Operational guardrails (unified-memory Sparks)

- Wait for ≥95 GB `MemAvailable` before launching after any teardown — GB10 reclaims a killed
  serve's pinned memory slowly, and loading 157 GB into a stale reclaim kernel-OOMs the node
  (the launcher enforces this).
- Run a watchdog that `docker kill`s the serve below ~4 GB `MemAvailable`; a kernel OOM here can
  livelock the node past SSH into a physical power cycle.
- Steady state is ~8–9 GB available/node at GMU 0.85 — that's normal, not a leak.

## What the 0.25 migration retired (vs our 0.24 port)

| 0.24-era transplant piece | 0.25 status |
|---|---|
| 17-file DSpark overlay (rafaelcaricio lineage) | upstream (#46995/#47093/#47429) |
| Hand-compiled flashinfer sparse-MLA `.so` swap + nsplit/out_lse fixes | stock `FLASHINFER_MLA_SPARSE_SM120` (pip 0.6.14) |
| DeepGEMM sm_121a transplant | official family-120 support |
| cooperative top-K disable hack | upstream (#47164) |
| Keys concurrency patch | superseded by upstream MRV2 DSpark (verified under C16) |

## Credits

- **[vLLM](https://github.com/vllm-project/vllm)** — upstream DSpark + sparse-MLA sm120 + DeepGEMM-120
- **[flashinfer](https://github.com/flashinfer-ai/flashinfer)** — the sm120 sparse-MLA kernels (0.6.14)
- **DeepSeek** — DSV4-Flash; **NVIDIA** — the DSpark speculative-decoding release
- Prior lineage: [our 0.24 transplant port](https://github.com/drowzeys/keys-vLLm-0.24.0-Optimized-DeepSeekV4-Flash-DSpark-NVFP4-KV-1.5M-CTX-3M-Pool-C-12-on-2-DGX-Spark), aidendle94's compiled-kernel stack
