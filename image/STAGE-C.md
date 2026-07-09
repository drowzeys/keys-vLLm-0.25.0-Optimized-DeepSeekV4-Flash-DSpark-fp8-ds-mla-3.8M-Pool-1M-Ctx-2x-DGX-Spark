# stage-c image (`vllm-dspark-runtime:dspark-nvfp4-stage-c`)

Used by `scripts/dsv4-nvfp4-1m-serve.sh` for the **nvfp4_ds_mla + B12X** recipe.

## Identity (publish cluster)

| field | value |
|---|---|
| Local tag | `vllm-dspark-runtime:dspark-nvfp4-stage-c` |
| Image id | `sha256:76532c4cc261afe7a7cad1d9731cd5123d0e14219c9a1d35a0ef6163fe67c5d4` |
| Size | ~22.7 GB |
| Created | 2026-07-01 |
| Engine banner | `v0.21.1rc1.dev339+g1967a5627bc3` |
| GHCR | `ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10` (also `:latest`) |

## What it is

A GB10 (sm_121a) serving image in the **aidendle94 / tonyd2wild / MiaAI-Lab** lineage:

- vLLM ~`0.21.1rc1.dev339` with DSpark speculative decode + B12X Mxfp4 MoE
- **`nvfp4_ds_mla`** sparse-MLA KV cache path (not in stock upstream 0.25)
- CUDA graphs that **capture and replay on GB10** (unlike stock `nightly-aarch64`, which wedges — see README Wall 2)
- Boot-proven on this cluster: `GPU KV cache size: 2,500,107 tokens` @ `max_model_len=1048576`, GMU 0.82

## Related images in this repo

| tag / path | role |
|---|---|
| `image/Dockerfile` | stock `vllm/vllm-openai:nightly-aarch64` + flashinfer 0.6.14 bump (`vllm-dsv4-025:fi614`) — eager DSpark path |
| `eugr/spark-vllm:latest` | public GB10-native playbook image — graphs + fp8_ds_mla 0.25-line recipe |
| `vllm-dspark-runtime:dspark-nvfp4-stage-c` | this file — 1M nvfp4_ds_mla speed recipe |

## Pull from GHCR

```bash
docker pull ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10
docker tag  ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10 \
  vllm-dspark-runtime:dspark-nvfp4-stage-c
```

If the package is private, authenticate first:

```bash
echo $GHCR_TOKEN | docker login ghcr.io -u drowzeys --password-stdin
# token needs read:packages (and write:packages to push)
```

## Re-publish / retag (maintainers)

```bash
# on a machine that already has the local tag
docker tag vllm-dspark-runtime:dspark-nvfp4-stage-c \
  ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10
docker tag vllm-dspark-runtime:dspark-nvfp4-stage-c \
  ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:latest
echo $GHCR_TOKEN | docker login ghcr.io -u drowzeys --password-stdin
docker push ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10
docker push ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:latest
```

Link the package to this GitHub repo (package settings → "Connect repository") so it shows under the repo Packages sidebar.

Make public (optional, after first successful push):

```bash
gh api -X PATCH \
  /user/packages/container/vllm-dspark-nvfp4-stage-c \
  -f visibility=public
```

## Credits

- aidendle94 / bjk110 unholy-fusion GB10 vLLM base
- tonyd2wild 1M NVFP4-KV DSpark recipes
- MiaAI-Lab two-node worker-first launch pattern
- DeepSeek DSpark / DeepSpec; NVIDIA DSV4-Flash-DSpark checkpoint
