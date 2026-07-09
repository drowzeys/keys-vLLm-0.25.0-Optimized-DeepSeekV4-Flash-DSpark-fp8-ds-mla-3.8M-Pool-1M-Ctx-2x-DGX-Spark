#!/usr/bin/env bash
# DeepSeek-V4-Flash-DSpark on UPSTREAM vLLM nightly (0.25 line) — MRV2-native DSpark (#46995/#47093/#47429),
# stock FLASHINFER_MLA_SPARSE_SM120 + DeepGEMM family-120. Retires vllm-dspark024:gb10 transplant stack.
# TP=2 across .1 (rank0, API :8000) + .2 (rank1 headless). Launch rank1 FIRST, then rank0.
# Usage: dsv4-025-serve.sh <rank 0|1>
# Knobs: SPEC(dspark|none) SPEC_TOKENS(5) SEQS(16) MAXLEN(1048576) KVD(fp8_ds_mla) GMU(0.85) IMG
#        EAGER(1)|CC_MODE(PIECEWISE) — Wall 2: FULL decode graph wedges sm_121a; PIECEWISE = the untested salvage lead
set -uo pipefail
RANK="${1:?usage: dsv4-025-serve.sh <rank 0|1>}"
MASTER=10.100.10.3; PORT=29551; IF=enp1s0f1np1; HCA=rocep1s0f1
IMG="${IMG:-vllm/vllm-openai:nightly-aarch64}"
SPEC="${SPEC:-dspark}"; SPEC_TOKENS="${SPEC_TOKENS:-5}"
SEQS="${SEQS:-16}"; MAXLEN="${MAXLEN:-1048576}"; KVD="${KVD:-fp8_ds_mla}"; GMU="${GMU:-0.85}"
EAGER="${EAGER:-0}"; PATCH_SWA="${PATCH_SWA:-0}"
EAGERARG=""; [ "$EAGER" = "1" ] && EAGERARG="--enforce-eager"
CC_MODE="${CC_MODE:-}"; [ "$EAGER" != "1" ] && [ -n "$CC_MODE" ] && EAGERARG="-cc.cudagraph_mode=$CC_MODE"
PATCHMOUNT=""; [ "$PATCH_SWA" = "1" ] && PATCHMOUNT="-v $HOME/dsv4-025-patches/sparse_swa.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/sparse_swa.py:ro"
# PATCH_DSPARK=1: graph-safety fixes for the MRV2 DSpark draft (clamp Markov feedback ids,
# pad-row input_id hygiene) — the [-1] draft corruption under CG replay (see repo Wall 2).
PATCH_DSPARK="${PATCH_DSPARK:-0}"
VSPEC=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode
[ "$PATCH_DSPARK" = "1" ] && PATCHMOUNT="$PATCHMOUNT -v $HOME/dsv4-025-patches/dspark_speculator.py:$VSPEC/dspark/speculator.py:ro -v $HOME/dsv4-025-patches/dflash_speculator.py:$VSPEC/dflash/speculator.py:ro"
# PATCH_ANCHOR=1: anchor RoPE-position probe (DeepSpec convention: anchor AT last verified pos).
# Overrides the dflash mount; implies the graph-safety patches too. (Probe result: no accept change.)
PATCH_ANCHOR="${PATCH_ANCHOR:-0}"
[ "$PATCH_ANCHOR" = "1" ] && PATCHMOUNT="$PATCHMOUNT -v $HOME/dsv4-025-patches/dspark_speculator.py:$VSPEC/dspark/speculator.py:ro -v $HOME/dsv4-025-patches/dflash_speculator_anchor.py:$VSPEC/dflash/speculator.py:ro"
# PATCH_DSPARK_EUGR=1: cg-clamp patches rebased onto the eugr/spark-vllm image's sources
# (dspark speculator identical across images; dflash has 2 trivial API diffs). No sparse_swa
# mount — eugr's build carries its own 512-width table.
PATCH_DSPARK_EUGR="${PATCH_DSPARK_EUGR:-0}"
[ "$PATCH_DSPARK_EUGR" = "1" ] && PATCHMOUNT="$PATCHMOUNT -v $HOME/dsv4-025-patches/dspark_speculator.py:$VSPEC/dspark/speculator.py:ro -v $HOME/dsv4-025-patches/dflash_speculator_eugr.py:$VSPEC/dflash/speculator.py:ro -v $HOME/dsv4-025-patches/sparse_swa_eugr.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/sparse_swa.py:ro"
# PATCH_CONF=1: stage-c confidence-head port (variable-length draft scheduling; the 60-67% accept lever).
# REQUIRES sync scheduling (adds --no-async-scheduling). Includes the cg-clamp patches. EAGER=1 advised.
PATCH_CONF="${PATCH_CONF:-0}"; CONF_THRESH="${CONF_THRESH:-0.4}"; CONF_SCHED="${CONF_SCHED:-threshold}"
NOASYNC=""
if [ "$PATCH_CONF" != "0" ]; then
  CD="$HOME/dsv4-025-patches/confidence"
  # eugr variant also needs the SWA width-pad (do NOT combine with PATCH_DSPARK_EUGR — duplicate mounts)
  [ "$PATCH_CONF" = "eugr" ] && { CD="$HOME/dsv4-025-patches/confidence-eugr"; PATCHMOUNT="$PATCHMOUNT -v $HOME/dsv4-025-patches/sparse_swa_eugr.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/sparse_swa.py:ro"; }
  PATCHMOUNT="$PATCHMOUNT -v $CD/dspark_confidence.py:$VSPEC/dspark/confidence.py:ro -v $CD/dspark_speculator.py:$VSPEC/dspark/speculator.py:ro -v $CD/dflash_speculator.py:$VSPEC/dflash/speculator.py:ro -v $CD/spec_decode_utils.py:$VSPEC/utils.py:ro -v $CD/model_runner.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py:ro -v $CD/model_dspark.py:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/dspark.py:ro"
  NOASYNC="--no-async-scheduling"
  CONFENV="-e VLLM_DSPARK_CONFIDENCE_THRESHOLD=$CONF_THRESH -e VLLM_DSPARK_CONFIDENCE_SCHEDULER=$CONF_SCHED -e VLLM_DSPARK_CONFIDENCE_LOG_EVERY=200"
fi
CONFENV="${CONFENV:-}"
MODELDIR="$HOME/models/dsv4-flash-dspark"
SELF=$(ip -4 addr show $IF 2>/dev/null|awk '/inet /{print $2}'|cut -d/ -f1); SELF=${SELF:-$MASTER}
HEADLESS=""; [ "$RANK" != "0" ] && HEADLESS="--headless"
SPECARG=""
[ "$SPEC" = "dspark" ] && SPECARG="--speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":$SPEC_TOKENS}'"

bash "$HOME/gpu-clear.sh" >/dev/null 2>&1 || true
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
docker rm -f dsv4_025 hy3_0xsero hy3_a4q hy3_nightly >/dev/null 2>&1 || true

# GB10 reclaim guard: require 95G available (78G shard + overhead) before launch
NEED_KB=$((95*1024*1024))
for i in $(seq 1 90); do
  avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  [ "$avail" -ge "$NEED_KB" ] && break
  sleep 2
done
[ "$avail" -lt "$NEED_KB" ] && { echo "ABORT: only $((avail/1048576))G avail — reclaim incomplete"; exit 1; }

docker run --gpus all -d --privileged --network host --ipc host --shm-size 10g \
  --memory 112g --memory-swap 112g --ulimit memlock=-1 --ulimit nofile=1048576 \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODELDIR:/model:ro" $PATCHMOUNT \
  --name dsv4_025 \
  -e VLLM_HOST_IP=$SELF -e NCCL_SOCKET_IFNAME=$IF -e GLOO_SOCKET_IFNAME=$IF -e TP_SOCKET_IFNAME=$IF \
  -e NCCL_IB_HCA=$HCA -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e HF_HUB_OFFLINE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $CONFENV \
  --entrypoint bash "$IMG" \
  -lc 'exec vllm serve /model \
    --served-model-name deepseek-v4-flash-dspark dsv4-dspark-025 \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 2 --pipeline-parallel-size 1 \
    --kv-cache-dtype '"$KVD"' \
    --max-model-len '"$MAXLEN"' --max-num-seqs '"$SEQS"' \
    --gpu-memory-utilization '"$GMU"' \
    --generation-config vllm --no-enable-prefix-caching '"$EAGERARG"' '"$NOASYNC"' \
    --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
    '"$SPECARG"' \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank '"$RANK"' --master-addr '"$MASTER"' --master-port '"$PORT"' '"$HEADLESS"''
echo "launched dsv4_025 rank=$RANK img=$IMG kv=$KVD spec=$SPEC/$SPEC_TOKENS seqs=$SEQS maxlen=$MAXLEN gmu=$GMU eager=$EAGER patch_swa=$PATCH_SWA rc=$?"
