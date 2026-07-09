# stage-c image (`vllm-dspark-runtime:dspark-nvfp4-stage-c`)

Used by `scripts/dsv4-nvfp4-1m-serve.sh` for the **nvfp4_ds_mla + B12X** recipe.

## What it is

A GB10 (sm_121a) serving image in the **aidendle94 / tonyd2wild / MiaAI-Lab** lineage:

- vLLM ~`0.21.1rc1.dev339` with DSpark speculative decode + B12X Mxfp4 MoE
- **`nvfp4_ds_mla`** sparse-MLA KV cache path (not in stock upstream 0.25)
- CUDA graphs that **capture and replay on GB10** (unlike stock `nightly-aarch64`, which wedges — see README Wall 2)

Local tag on the build cluster: `vllm-dspark-runtime:dspark-nvfp4-stage-c` (~22.7 GB).

## Related images in this repo

| tag / path | role |
|---|---|
| `image/Dockerfile` | stock `vllm/vllm-openai:nightly-aarch64` + flashinfer 0.6.14 bump (`vllm-dsv4-025:fi614`) — eager DSpark path |
| `eugr/spark-vllm:latest` | public GB10-native playbook image — graphs + fp8_ds_mla 0.25-line recipe |
| `vllm-dspark-runtime:dspark-nvfp4-stage-c` | this file — 1M nvfp4_ds_mla speed recipe |

## Publishing to GHCR (optional)

```bash
# on a machine that already has the local tag
docker tag vllm-dspark-runtime:dspark-nvfp4-stage-c \
  ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10
echo $GHCR_TOKEN | docker login ghcr.io -u drowzeys --password-stdin
docker push ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10
```

Consumers:

```bash
docker pull ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10
docker tag  ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10 \
  vllm-dspark-runtime:dspark-nvfp4-stage-c
```

## Credits

- aidendle94 / bjk110 unholy-fusion GB10 vLLM base
- tonyd2wild 1M NVFP4-KV DSpark recipes
- MiaAI-Lab two-node worker-first launch pattern
- DeepSeek DSpark / DeepSpec; NVIDIA DSV4-Flash-DSpark checkpoint
