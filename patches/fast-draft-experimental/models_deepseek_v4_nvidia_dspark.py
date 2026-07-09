# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft model for DeepSeek-V4 (semi-autoregressive speculative decoding).

See: qwen3_dspark.py for base architecture. This one is specialized to the DSV4 DSpark,
which reuses the target model's architecture similarly to MTP.

To implement non-causal attention, we leverage the sparse attention implementation to
include the future query tokens in the top-k indices for each query token.
"""

import os
from collections.abc import Iterable
from types import MethodType

import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_post_tilelang,
)
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_dspark import (
    DSparkMarkovHead,
)
from vllm.model_executor.models.utils import maybe_prefix

from .model import (
    DeepseekV4DecoderLayer,
    make_deepseek_v4_expert_params_mapping,
)

# PATCH(gb10-dspark-fast): stage-c Triton ring-attention kernels. Guarded so
# the legacy 2-file mount (without dspark_fast_kernels.py) still imports; the
# fast path asserts availability at enable time.
try:
    from .dspark_fast_kernels import dspark_fast_sparse_attention
except ImportError:
    dspark_fast_sparse_attention = None  # type: ignore[assignment]

logger = init_logger(__name__)

# MoE expert scale suffix differs by expert dtype (mirrors deepseek_v4 loaders):
# fp4 experts register ``.weight_scale``; block-fp8 experts ``.weight_scale_inv``.
_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


# PATCH(gb10-dspark-qatkv): QAT KV parity for the DSpark draft.
#
# With ``--kv-cache-dtype fp8_ds_mla`` the draft layers' private SWA cache is
# re-quantized into the UE8M0 fp8_ds_mla paged layout (uint8, 584B/token) —
# the dtype is resolved once per attention layer in
# ``deepseek_v4/attention.py::_resolve_dsv4_kv_cache_dtype`` (driven by the
# SM120 class's ``use_fp8_ds_mla_layout = True``) and the read is done by
# flashinfer's SM120 sparse-MLA kernel, which ONLY accepts the packed 584B
# uint8 pool (no bf16 variant exists in flashinfer 0.6.14). The fast 0.24
# stage-c stack instead kept the draft's context KV in a private bf16 ring
# buffer, matching the reference QAT training. This patch reproduces that:
#
#   * a per-layer bf16 ring buffer [max_num_reqs, window, head_dim], keyed by
#     the persistent request row (idx_mapping) and position % window;
#   * the context-KV precompute additionally RoPEs + stores bf16 rows into the
#     ring (rejected tokens are masked at write time, exactly like stage-c's
#     ``store_main_kv`` valid_mask — the ring aliases positions mod window, so
#     unlike the paged cache a rejected write would corrupt an in-window slot);
#   * the draft layers' ``forward_mqa`` is replaced with the stage-c torch
#     reference attention (fp32 scores, attn-sink in the softmax denominator,
#     non-causal over [ring window ; current query block]), reading K/V from
#     the bf16 ring + the current pass's bf16 kv instead of the fp8 pool.
#
# Enabled with VLLM_DSPARK_DRAFT_KV_BF16=1. Optionally
# VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT=1 additionally applies the reference
# QAT fp8 quant-dequant to the no-RoPE dims (stage-c's optional qdq), for A/B.
# The paged fp8 writes are left in place (harmless; the fp8 pool is simply no
# longer read by the draft layers), so the KV-cache spec, group layout and
# memory accounting are byte-identical to the unpatched build.
_FP8_E4M3_MAX = 448.0
_QATKV_NEG_INF = float("-inf")


def _qatkv_env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _qatkv_qdq_nope_torch(
    kv: torch.Tensor, rope_dim: int, group_size: int = 64
) -> torch.Tensor:
    """Reference in-place FP8 quant-dequant for the no-RoPE KV dims.

    Port of stage-c's ``dspark_quant_dequant_nope_torch`` (pow2 UE8M0-style
    scales over 64-wide groups, RoPE slice untouched).
    """
    head_dim = kv.shape[-1]
    nope_dim = head_dim - rope_dim
    if nope_dim <= 0:
        return kv
    assert nope_dim % group_size == 0
    nope = kv[..., :nope_dim]
    original_shape = nope.shape
    groups = nope.reshape(-1, nope_dim // group_size, group_size).float()
    amax = groups.abs().amax(dim=-1, keepdim=True).clamp_min_(1.0e-4)
    scale = torch.pow(
        torch.full((), 2.0, device=kv.device, dtype=torch.float32),
        torch.ceil(torch.log2(amax / _FP8_E4M3_MAX)),
    )
    quantized = torch.clamp(
        groups / scale, min=-_FP8_E4M3_MAX, max=_FP8_E4M3_MAX
    ).to(torch.float8_e4m3fn)
    nope.copy_((quantized.float() * scale).reshape(original_shape).to(kv.dtype))
    return kv


def _qatkv_forward_mqa(self, q, kv, positions, output):
    """PATCH(gb10-dspark-qatkv): bf16 draft attention (bound per draft layer).

    Replaces the SM120 flashinfer sparse read for the DSpark draft layers.
    Mirrors stage-c's ``dspark_sparse_attention_torch`` contract:
      K/V = [bf16 ring window (context) ; current query block kv (RoPE'd)],
      every query token attends non-causally to that whole set, ring slots
      beyond min(ctx_len, window) are masked, and the per-head attn_sink joins
      the softmax denominator (an extra logit with no value vector).

    ``q`` arrives post qnorm+RoPE (padded to padded_heads, bf16); ``kv``
    arrives post kv_norm but PRE-RoPE (the fused insert leaves it unchanged),
    so the query-block K gets RoPE'd here with the layer's rotary module —
    the same module stage-c used for both its ring writes and query block.
    """
    if get_forward_context().attn_metadata is None:
        # Profiling / warmup dummy run: nothing to read; mirror upstream.
        output.zero_()
        return

    ring = self._qatkv_ring  # [max_num_reqs, window, head_dim] bf16
    window = ring.shape[-2]
    head_dim = ring.shape[-1]
    n_query = self._qatkv_num_query
    num_tokens = q.shape[0]
    assert num_tokens % n_query == 0, (
        f"draft pass token count {num_tokens} not a multiple of "
        f"num_query_per_req {n_query}"
    )
    num_reqs = num_tokens // n_query
    n_heads = self.n_local_heads

    rows = self._qatkv_rows[:num_reqs]
    ring_kv = ring.index_select(0, rows)  # [B, W, D] bf16

    # Query-block K/V: RoPE the current pass's kv. Clone first — the CUDA
    # rope op is in-place and ``kv`` aliases the layer's activation.
    kv_q = kv[:num_tokens].unsqueeze(1).clone()
    kv_q, _ = self.rotary_emb(positions[:num_tokens], kv_q, None)
    kv_q = kv_q.squeeze(1)
    if self._qatkv_qdq:
        _qatkv_qdq_nope_torch(kv_q, self.rope_head_dim)
    draft_kv = kv_q.view(num_reqs, n_query, head_dim)

    kv_cat = torch.cat([ring_kv, draft_kv], dim=1).float()  # [B, W+N, D]
    q_f = q[:num_tokens, :n_heads].view(num_reqs, n_query, n_heads, head_dim)
    scores = torch.einsum("bqhd,bkd->bqhk", q_f.float(), kv_cat)
    scores.mul_(self.scale)

    # ctx_len == the anchor query position (prefix length): ring slots
    # [0, min(ctx_len, window)) hold the latest valid in-window positions.
    ctx_len = positions[:num_tokens].view(num_reqs, n_query)[:, 0]
    slot_idx = torch.arange(window, device=q.device)
    valid_main = slot_idx.unsqueeze(0) < ctx_len.clamp(max=window).unsqueeze(1)
    scores[..., :window].masked_fill_(
        ~valid_main[:, None, None, :], _QATKV_NEG_INF
    )

    sink = self.attn_sink[:n_heads].float().view(1, 1, n_heads, 1)
    normalizer = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
    weights = torch.exp(scores - normalizer)
    denom = weights.sum(dim=-1, keepdim=True) + torch.exp(sink - normalizer)
    out = torch.einsum("bqhk,bkd->bqhd", weights, kv_cat) / denom

    # Padded head columns of ``output`` are sliced off by the caller
    # (o = o_padded[:, :n_local_heads]) before the output projection.
    output[:num_tokens, :n_heads].copy_(
        out.reshape(num_tokens, n_heads, head_dim).to(output.dtype)
    )


# PATCH(gb10-dspark-fast): performance path on top of the QAT-KV ring.
#
# VLLM_DSPARK_FAST_DRAFT=1 turns the acceptance-neutral bf16-ring patch above
# into the stage-c persistent-ring fast path:
#   1. the ring attention read runs through stage-c's fused Triton kernels
#      (``dspark_fast_sparse_attention``) instead of eager torch einsums
#      (VLLM_DSPARK_FAST_DRAFT_TORCH=1 falls back to the torch read for A/B);
#   2. the paged fp8_ds_mla draft KV pool is no longer WRITTEN:
#      * the per-step context insert (``_insert_context_kv``) is skipped and
#        the context kv projection only runs on the ring-kept rows;
#      * the query-block insert inside the draft forward (the fused
#        qnorm+RoPE+quant+insert op) is redirected into a tiny private
#        throwaway block via an all-``arange`` private slot mapping, so the q
#        processing stays bit-identical while the real pool is untouched;
#   3. the speculator skips the per-step draft attention-metadata rebuild
#      (see dflash/speculator.py) — with 1+2 rebound, no draft-layer code
#      reads DeepseekSparseSWAMetadata any more.
#
# CUDA-graph safety: every tensor the captured region reads or writes is a
# persistent buffer allocated at init (ring / rows / gather / scores /
# valid-main / dummy slots) or lazily in an eager warmup pass (dummy cache);
# Triton launches replay with fixed shapes per captured batch size. Ring
# writes stay in the eager precompute, outside the capture.


def _fast_init_dummy_cache(attn: nn.Module) -> None:
    """Allocate the throwaway paged block for the draft query-KV write.

    Sized to hold ``max_num_reqs * num_query`` private slots in the same
    per-block layout/dtype as the real SWA cache. Called lazily on the first
    metadata-carrying forward, which is always an eager warmup or eager run
    (CudaGraphManager warms up eagerly before capturing), and eagerly from
    the speculator's ``capture()`` as a belt-and-braces.
    """
    real = attn.swa_cache_layer.kv_cache
    assert real is not None and real.numel() > 0, (
        "DSpark fast draft: SWA kv cache is not bound yet"
    )
    block_size = attn.swa_cache_layer.block_size
    n_slots = attn._fast_dummy_slots.shape[0]
    num_blocks = (n_slots + block_size - 1) // block_size + 1
    attn._fast_dummy_cache = torch.zeros(
        (num_blocks, *real.shape[1:]), dtype=real.dtype, device=real.device
    )


def _fast_fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
    """PATCH(gb10-dspark-fast): upstream fused q-qnorm+RoPE / kv-RoPE+insert,
    with the paged write redirected into a private throwaway block.

    Bound per draft attention layer. Mirrors
    ``DeepseekV4Attention._fused_qnorm_rope_kv_insert`` exactly (same ops,
    same q processing and return contract) except that ``slot_mapping`` is a
    persistent private ``arange`` (one distinct slot per query token — no
    write conflicts) into ``_fast_dummy_cache`` instead of the real pool.
    The ring-based ``forward_mqa`` never reads any paged pool, so the real
    draft pages are now neither written nor read.
    """
    if not isinstance(attn_metadata, dict):
        # Profile run: kernel doesn't fire; produce a padded tensor so
        # downstream gets the right shape (mirrors upstream).
        if self.n_local_heads < self.padded_heads:
            return F.pad(
                q,
                (0, 0, 0, self.padded_heads - self.n_local_heads),
                value=0.0,
            )
        return q

    if self._fast_dummy_cache is None:
        _fast_init_dummy_cache(self)
    swa_kv_cache = self._fast_dummy_cache
    num_tokens = q.shape[0]
    assert num_tokens <= self._fast_dummy_slots.shape[0], (
        f"draft query pass has {num_tokens} tokens but only "
        f"{self._fast_dummy_slots.shape[0]} private slots"
    )
    slot_mapping = self._fast_dummy_slots[:num_tokens]
    block_size = self.swa_cache_layer.block_size
    assert positions.dtype == torch.int64
    cos_sin_cache = self.rotary_emb.cos_sin_cache
    cache_dtype = swa_kv_cache.dtype

    if cache_dtype == torch.uint8:
        # fp8_ds_mla UE8M0 paged path (the champion config).
        swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)
        return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q,
            kv,
            swa_kv_cache_2d,
            slot_mapping,
            positions,
            cos_sin_cache,
            self.padded_heads,
            self.eps,
            block_size,
        )

    swa_kv_cache_3d = swa_kv_cache.view(-1, block_size, self.head_dim)
    if cache_dtype == torch.bfloat16:
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
            q,
            kv,
            swa_kv_cache_3d,
            slot_mapping,
            positions,
            cos_sin_cache,
            self.eps,
            block_size,
        )
        return q

    # per-tensor fp8 (torch.float8_e4m3fn). NOTE: upstream returns an fp8 q
    # here for the flashinfer read; the ring forward_mqa consumes q directly,
    # so this dtype is unsupported with the fast path (champion = fp8_ds_mla).
    raise NotImplementedError(
        "DSpark fast draft does not support per-tensor fp8 draft KV; use "
        "fp8_ds_mla or bf16."
    )


def _fast_forward_mqa(self, q, kv, positions, output):
    """PATCH(gb10-dspark-fast): Triton ring attention (bound per draft layer).

    Same contract as ``_qatkv_forward_mqa`` (which stays available via
    VLLM_DSPARK_FAST_DRAFT_TORCH=1), but reads run through stage-c's fused
    two-kernel Triton pipeline over persistent buffers:
      gather ring rows for this pass's request rows -> RoPE the query-block
      kv -> scores kernel (fp32, window masking by valid_main_lengths) ->
      out kernel (attn-sink softmax + PV), writing straight into the caller's
      padded output buffer.
    """
    if get_forward_context().attn_metadata is None:
        # Profiling / warmup dummy run: nothing to read; mirror upstream.
        output.zero_()
        return

    ring = self._qatkv_ring  # [max_num_reqs, window, head_dim] bf16
    head_dim = ring.shape[-1]
    n_query = self._qatkv_num_query
    num_tokens = q.shape[0]
    assert num_tokens % n_query == 0, (
        f"draft pass token count {num_tokens} not a multiple of "
        f"num_query_per_req {n_query}"
    )
    num_reqs = num_tokens // n_query
    n_heads = self.n_local_heads

    # Gather this pass's ring rows into the persistent [B, W, D] buffer the
    # captured Triton kernels read (rows is refreshed eagerly each step).
    rows = self._qatkv_rows[:num_reqs]
    gathered = self._fast_ring_gather[:num_reqs]
    torch.index_select(ring, 0, rows, out=gathered)

    # Query-block K/V: RoPE the current pass's kv. Clone first — the CUDA
    # rope op is in-place and ``kv`` aliases the layer's activation.
    kv_q = kv[:num_tokens].unsqueeze(1).clone()
    kv_q, _ = self.rotary_emb(positions[:num_tokens], kv_q, None)
    kv_q = kv_q.squeeze(1)
    if self._qatkv_qdq:
        _qatkv_qdq_nope_torch(kv_q, self.rope_head_dim)
    draft_kv = kv_q.view(num_reqs, n_query, head_dim)

    # ctx_len == the anchor query position (prefix length): ring slots
    # [0, min(ctx_len, window)) hold the latest valid in-window positions.
    # The kernel masks main slots at index >= this value (> window is
    # equivalent to window since the main region is only window wide).
    valid_main = self._fast_valid_main[:num_reqs]
    valid_main.copy_(positions[:num_tokens].view(num_reqs, n_query)[:, 0])

    # Padded-head strided views; the kernels only touch the first n_heads
    # head columns (padded output columns are sliced off by the caller).
    q4 = q[:num_tokens].view(num_reqs, n_query, q.shape[1], head_dim)
    out4 = output[:num_tokens].view(num_reqs, n_query, output.shape[1], head_dim)
    dspark_fast_sparse_attention(
        q4,
        draft_kv,
        gathered,
        valid_main,
        self.attn_sink,
        self.scale,
        self._fast_scores[:num_reqs],
        out4,
        n_heads,
    )


class DSparkDeepseekV4Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.num_hidden_layers = config.num_hidden_layers
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)

        self.num_dspark_layers = getattr(config, "n_mtp_layers", None) or 3

        # Shared with the target (aliased by the speculator's loading utility).
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        current_vllm_config = get_current_vllm_config()
        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(
                    current_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{self.num_hidden_layers + i}"),
                )
                for i in range(self.num_dspark_layers)
            ]
        )

        # Heads: final norm + hc_head, and the Markov head
        # Loaded from the "final" MTP layer weights (mtp.*) in the target checkpoint
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )

        # PATCH(gb10-dspark-qatkv): private bf16 context-KV ring (stage-c
        # parity). See the module-level comment above _qatkv_forward_mqa.
        # PATCH(gb10-dspark-fast): VLLM_DSPARK_FAST_DRAFT=1 implies the ring
        # and additionally enables the Triton attention read, the no-paged-
        # write mode and the speculator-side metadata skip (one switch).
        self.fast_draft = _qatkv_env_flag("VLLM_DSPARK_FAST_DRAFT")
        self.fast_draft_torch_attn = self.fast_draft and _qatkv_env_flag(
            "VLLM_DSPARK_FAST_DRAFT_TORCH"
        )
        self.qat_kv_bf16 = self.fast_draft or _qatkv_env_flag(
            "VLLM_DSPARK_DRAFT_KV_BF16"
        )
        self.qat_kv_qdq = self.qat_kv_bf16 and _qatkv_env_flag(
            "VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT"
        )
        if self.qat_kv_bf16:
            windows = {layer.attn.window_size for layer in self.layers}
            assert len(windows) == 1, (
                f"DSpark QAT-KV ring requires a uniform draft window, got {windows}"
            )
            self.qat_kv_window = windows.pop()
            head_dim = self.layers[0].attn.head_dim
            max_num_reqs = vllm_config.scheduler_config.max_num_seqs
            # num_query_per_req mirrors DSparkSpeculator: anchor-as-first (N)
            # unless the checkpoint uses the 1+N bonus-anchor block.
            num_spec = vllm_config.speculative_config.num_speculative_tokens
            sample_from_anchor = not getattr(config, "dspark_bonus_anchor", False)
            num_query = num_spec if sample_from_anchor else 1 + num_spec
            device = current_platform.device_type
            # Plain (unregistered) tensor attributes on purpose: they are not
            # weights (must stay out of state_dict / load_weights) and a
            # hypothetical post-init module .to() must not swap them out from
            # under the per-layer views cached below.
            self.qat_kv_ring = torch.zeros(
                self.num_dspark_layers,
                max_num_reqs,
                self.qat_kv_window,
                head_dim,
                dtype=vllm_config.model_config.dtype,
                device=device,
            )
            # Persistent request rows for the current draft pass (set by the
            # speculator each step from input_batch.idx_mapping).
            self.qat_kv_rows = torch.zeros(
                max_num_reqs, dtype=torch.int64, device=device
            )

            # PATCH(gb10-dspark-fast): persistent buffers for the Triton read
            # + the no-paged-write query insert. All fixed-address (CUDA-graph
            # replay safe); shared across the (sequentially executed) layers.
            if self.fast_draft:
                assert dspark_fast_sparse_attention is not None, (
                    "VLLM_DSPARK_FAST_DRAFT=1 requires dspark_fast_kernels.py "
                    "to be mounted next to this file"
                )
                n_local_heads = self.layers[0].attn.n_local_heads
                self.fast_ring_gather = torch.zeros(
                    max_num_reqs,
                    self.qat_kv_window,
                    head_dim,
                    dtype=vllm_config.model_config.dtype,
                    device=device,
                )
                self.fast_scores = torch.empty(
                    max_num_reqs,
                    num_query,
                    n_local_heads,
                    self.qat_kv_window + num_query,
                    dtype=torch.float32,
                    device=device,
                )
                self.fast_valid_main = torch.zeros(
                    max_num_reqs, dtype=torch.int64, device=device
                )
                # One private slot per possible query token: no conflicts, no
                # real-pool writes. The throwaway cache blocks are allocated
                # lazily once the real cache is bound (dtype/layout copied).
                self.fast_dummy_slots = torch.arange(
                    max_num_reqs * num_query, dtype=torch.int64, device=device
                )

            for i, layer in enumerate(self.layers):
                attn = layer.attn
                assert attn.compress_ratio <= 1, (
                    "DSpark QAT-KV ring only supports SWA-only draft layers"
                )
                attn._qatkv_ring = self.qat_kv_ring[i]
                attn._qatkv_rows = self.qat_kv_rows
                attn._qatkv_num_query = num_query
                attn._qatkv_qdq = self.qat_kv_qdq
                if self.fast_draft:
                    attn._fast_ring_gather = self.fast_ring_gather
                    attn._fast_scores = self.fast_scores
                    attn._fast_valid_main = self.fast_valid_main
                    attn._fast_dummy_slots = self.fast_dummy_slots
                    attn._fast_dummy_cache = None
                    attn._fused_qnorm_rope_kv_insert = MethodType(
                        _fast_fused_qnorm_rope_kv_insert, attn
                    )
                    attn.forward_mqa = MethodType(
                        _qatkv_forward_mqa
                        if self.fast_draft_torch_attn
                        else _fast_forward_mqa,
                        attn,
                    )
                else:
                    attn.forward_mqa = MethodType(_qatkv_forward_mqa, attn)
            if self.fast_draft:
                logger.info_once(
                    "DSpark FAST draft enabled: bf16 ring KV (%d layers x %d "
                    "reqs x %d window x %d dim), %s attention read, paged "
                    "draft KV writes disabled (query KV redirected to a "
                    "private throwaway block), draft attn-metadata rebuild "
                    "skipped%s.",
                    self.num_dspark_layers,
                    max_num_reqs,
                    self.qat_kv_window,
                    head_dim,
                    "torch-eager (A/B fallback)"
                    if self.fast_draft_torch_attn
                    else "Triton (stage-c dspark_sparse_attention)",
                    " + reference fp8 qdq on no-RoPE dims"
                    if self.qat_kv_qdq
                    else "",
                )
            else:
                logger.info_once(
                    "DSpark QAT-KV parity enabled: draft context KV stored/read in "
                    "%s ring buffers (%d layers x %d reqs x %d window x %d dim)%s; "
                    "the paged fp8_ds_mla draft pool is still written but no "
                    "longer read.",
                    vllm_config.model_config.dtype,
                    self.num_dspark_layers,
                    max_num_reqs,
                    self.qat_kv_window,
                    head_dim,
                    " + reference fp8 qdq on no-RoPE dims" if self.qat_kv_qdq else "",
                )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """main_x = main_norm(main_proj(concat of target aux hidden states)).

        ``aux_hidden_states`` is [T, hidden_size * len(target_layer_ids)].
        """
        return self.main_norm(self.main_proj(aux_hidden_states))

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        main_x: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
        qat_keep_idx: torch.Tensor | None = None,
        qat_ring_slots: torch.Tensor | None = None,
    ) -> None:
        """Insert the sliding-window context KV for every draft layer.

        Mirrors the reference DSparkAttention: each layer derives its context KV
        from the SAME projected target hidden ``main_x``, via that layer's own
        ``wkv`` + ``kv_norm`` + RoPE + quant, then writes it at the
        layer's context slots.

        ``context_slot_mappings`` is a per-layer list (each entry is the context
        slot mapping for that layer's kv-cache group, since the hybrid manager may
        place draft layers in different groups). ``None`` (or a ``None`` entry)
        runs the projection to reserve workspace but writes nothing (profiling).

        PATCH(gb10-dspark-qatkv): ``qat_keep_idx``/``qat_ring_slots`` (built by
        the speculator, eager path only) select the non-rejected, in-window
        context tokens and their bf16 ring slots (row * window + pos % window).
        When present, each layer additionally RoPEs those kv rows in bf16 and
        scatters them into its private ring. Runs eagerly (never captured).
        """
        # PATCH(gb10-dspark-fast): with the paged context insert disabled, only
        # the ring-kept rows (non-rejected, in-window: at most ~window per
        # request) need the kv projection at all. Gather them once and run the
        # per-layer wkv/kv_norm/RoPE on that subset only — a large cut on
        # prefill chunks, where main_x can be thousands of rows but the ring
        # keeps at most the trailing window. Dummy/profile passes
        # (qat_ring_slots is None) keep the full-width projection so memory
        # profiling still sees the worst-case workspace.
        fast_no_paged = self.fast_draft
        main_x_sel = pos_sel = None
        if fast_no_paged and qat_ring_slots is not None:
            main_x_sel = main_x[qat_keep_idx]
            pos_sel = context_positions[qat_keep_idx]

        for i, layer in enumerate(self.layers):
            slot_mapping = (
                None if context_slot_mappings is None else context_slot_mappings[i]
            )
            attn = layer.attn
            if main_x_sel is not None:
                # Fast path: project only the ring-kept rows; no paged insert.
                qr_kv, _ = attn.fused_wqa_wkv(main_x_sel)
                kv_sel = attn.kv_norm(qr_kv[..., attn.q_lora_rank :])
                kv_sel = kv_sel.unsqueeze(1)
                kv_sel, _ = attn.rotary_emb(pos_sel, kv_sel, None)
                kv_sel = kv_sel.squeeze(1)
                if self.qat_kv_qdq:
                    _qatkv_qdq_nope_torch(kv_sel, attn.rope_head_dim)
                ring = self.qat_kv_ring[i]
                ring.view(-1, ring.shape[-1]).index_copy_(
                    0, qat_ring_slots, kv_sel.to(ring.dtype)
                )
                continue
            # Optimized DSV4 MLA path: wkv part of the fused wq_a|wkv projection
            # (q_lora part discarded), then RoPE/quant/insert via the fused op.
            qr_kv, _ = attn.fused_wqa_wkv(main_x)
            kv = qr_kv[..., attn.q_lora_rank :]
            kv = attn.kv_norm(kv)
            # PATCH(gb10-dspark-qatkv): bf16 ring write (stage-c parity). The
            # gathered kv rows are a fresh copy, so the in-place CUDA rope op
            # cannot corrupt ``kv`` (still consumed by the paged insert below).
            if self.qat_kv_bf16 and qat_ring_slots is not None:
                kv_sel = kv[qat_keep_idx].unsqueeze(1)
                pos_sel_i = context_positions[qat_keep_idx]
                kv_sel, _ = attn.rotary_emb(pos_sel_i, kv_sel, None)
                kv_sel = kv_sel.squeeze(1)
                if self.qat_kv_qdq:
                    _qatkv_qdq_nope_torch(kv_sel, attn.rope_head_dim)
                ring = self.qat_kv_ring[i]
                ring.view(-1, ring.shape[-1]).index_copy_(
                    0, qat_ring_slots, kv_sel.to(ring.dtype)
                )
            # PATCH(gb10-dspark-fast): never write the real paged draft pool
            # in fast mode (it is never read; forward_mqa reads the ring).
            if slot_mapping is None or fast_no_paged:
                continue
            _insert_context_kv(attn, kv, context_positions, slot_mapping)

    # PATCH(gb10-dspark-fast): eager allocation hook for the throwaway
    # query-KV blocks (called from the speculator before CUDA-graph capture;
    # the per-layer insert rebind also lazily allocates on first eager use).
    def fast_draft_init_buffers(self) -> None:
        if not self.fast_draft:
            return
        for layer in self.layers:
            attn = layer.attn
            if (
                attn._fast_dummy_cache is None
                and attn.swa_cache_layer.kv_cache is not None
                and attn.swa_cache_layer.kv_cache.numel() > 0
            ):
                _fast_init_dummy_cache(attn)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        # Expand to hc_mult copies for hyper-connections ([T, H] -> [T, hc, H]).
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)

        residual = post_mix = res_mix = None
        for layer in self.layers:
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )
        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
        # hc_head reduces the hc copies; return the PRE-norm head hidden
        hidden_states = hc_head_fused_kernel_tilelang(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        return hidden_states


def _insert_context_kv(
    attn: nn.Module,
    kv: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """RoPE + quant + paged-cache insert of (already kv_norm'd) context KV.

    Reuses the DSV4 fused insert ops (which also process a query; we pass a dummy
    query and discard it, since context tokens have no query). Mirrors
    ``DeepseekV4Attention._fused_qnorm_rope_kv_insert``.
    """
    swa_cache = attn.swa_cache_layer.kv_cache
    block_size = attn.swa_cache_layer.block_size
    cos_sin_cache = attn.rotary_emb.cos_sin_cache
    cache_dtype = swa_cache.dtype
    n_ctx = kv.shape[0]
    dummy_q = torch.zeros(
        (n_ctx, attn.n_local_heads, attn.head_dim),
        dtype=kv.dtype,
        device=kv.device,
    )
    if cache_dtype == torch.uint8:
        # fp8_ds_mla UE8M0 paged layout
        swa_2d = swa_cache.view(swa_cache.shape[0], -1)
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            dummy_q,
            kv,
            swa_2d,
            slot_mapping,
            positions,
            cos_sin_cache,
            attn.padded_heads,
            attn.eps,
            block_size,
        )
    elif cache_dtype == torch.bfloat16:
        swa_3d = swa_cache.view(-1, block_size, attn.head_dim)
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
            dummy_q,
            kv,
            swa_3d,
            slot_mapping,
            positions,
            cos_sin_cache,
            attn.eps,
            block_size,
        )
    else:  # per-tensor fp8 (torch.float8_e4m3fn)
        # TODO(ben): double-check if this is being dispatched correctly for FI backend
        swa_3d = swa_cache.view(-1, block_size, attn.head_dim)
        dummy_q_fp8 = torch.zeros_like(dummy_q, dtype=torch.float8_e4m3fn)
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
            dummy_q,
            kv,
            dummy_q_fp8,
            swa_3d,
            slot_mapping,
            positions,
            cos_sin_cache,
            attn._flashinfer_fp8_kv_scale,
            attn._flashinfer_fp8_q_scale_inv,
            attn.eps,
            block_size,
        )


class DSparkDeepseekV4ForCausalLM(nn.Module):
    # Draft weights ship in the target checkpoint (mtp.*) without embed/head, so
    # load_dspark_model always aliases the target's.
    has_own_embed_tokens = False
    has_own_lm_head = False
    # Full-vocab draft: draft ids are target ids, no remapping needed.
    draft_id_to_target_id = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.model = DSparkDeepseekV4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Shared with the target (aliased by the speculator's load utility).
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    # --- Hooks used by the speculator -------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        # DSV4 MLA path: each draft layer's sliding-window cache is a separate
        # layer, named by its prefix.
        return [layer.attn.swa_cache_layer.prefix for layer in self.model.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
        qat_keep_idx: torch.Tensor | None = None,
        qat_ring_slots: torch.Tensor | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states,
            context_positions,
            context_slot_mappings,
            qat_keep_idx=qat_keep_idx,
            qat_ring_slots=qat_ring_slots,
        )

    # PATCH(gb10-dspark-qatkv): speculator-facing hooks for the bf16 ring.

    @property
    def qat_kv_bf16(self) -> bool:
        return self.model.qat_kv_bf16

    @property
    def qat_kv_window(self) -> int:
        return self.model.qat_kv_window

    # PATCH(gb10-dspark-fast): speculator-facing hooks for the fast path.

    @property
    def fast_draft(self) -> bool:
        return self.model.fast_draft

    def fast_draft_init_buffers(self) -> None:
        self.model.fast_draft_init_buffers()

    def qat_kv_set_rows(self, idx_mapping: torch.Tensor, num_reqs: int) -> None:
        """Record the persistent request row of each batch row for this pass."""
        self.model.qat_kv_rows[:num_reqs].copy_(idx_mapping[:num_reqs])

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Returns the pre-norm hc_head hidden ([T, hidden_size]).
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Base logits U_k = lm_head(norm(head_hidden))."""
        return self.logits_processor(self.lm_head, self.model.norm(hidden_states))

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Full-vocab draft: base logits, no d2t scatter.
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids  # full-vocab: draft ids are target ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    # --- Weight loading ----------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the ``mtp.{0,1,2}.*`` draft weights from the target checkpoint.

        Non-mtp weights (embed/head/main layers) belong to the target model and
        are skipped here. ``embed_tokens``/``lm_head`` are aliased from the target.
        """
        first_layer = self.model.layers[0]
        use_mega_moe = first_layer.ffn.use_mega_moe
        if use_mega_moe:
            expert_mapping = make_deepseek_v4_expert_params_mapping(
                self.config.n_routed_experts
            )
        else:
            expert_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.n_routed_experts,
            )
        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )

        # (param_name, ckpt_shard_name, shard_id) for non-expert stacked params.
        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // tp_size
        head_start = n_local_head * tp_rank
        head_end = n_local_head * (tp_rank + 1)

        for name, loaded_weight in weights:
            mapped = self._remap_dspark_name(name)
            if mapped is None:
                continue
            name = mapped

            # ``.scale`` -> per-method scale suffix.
            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else ".weight_scale_inv"
                )
                name = name.removesuffix(".scale") + suffix

            # E8M0 expert scales: keep raw exponent bytes.
            if ".experts." in name:
                if (
                    "weight_scale" in name
                    and loaded_weight.dtype == torch.float8_e8m0fnu
                ):
                    loaded_weight = loaded_weight.view(torch.uint8)
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name not in name:
                        continue
                    name_mapped = name.replace(weight_name, param_name)
                    param = params_dict[name_mapped]
                    success = param.weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                        break
                continue

            # Stacked rules only apply to decoder-layer weights. Head-stack params
            # (main_proj/norm/hc_head/markov_head) load directly — otherwise e.g.
            # "markov_w1" would collide with the "w1" shard rule.
            is_layer_param = name.startswith("model.layers.")
            for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
                if not is_layer_param or weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, stacked_shard_id)
                loaded_params.add(name)
                break
            else:
                if "attn_sink" in name:
                    narrow = loaded_weight[head_start:head_end]
                    params_dict[name][: narrow.shape[0]].copy_(narrow)
                    loaded_params.add(name)
                    continue
                if ".shared_experts.w2" in name:
                    name = name.replace(
                        ".shared_experts.w2", ".shared_experts.down_proj"
                    )
                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                    )
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        self._finalize_moe()
        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def _finalize_moe(self) -> None:
        for layer in self.model.layers:
            layer.ffn.finalize_mega_moe_weights()

    def _remap_dspark_name(self, name: str) -> str | None:
        """Map a checkpoint ``mtp.{i}.*`` name to this model's parameter path.

        Returns None for non-mtp weights (owned by the target model).
        """
        m = re.match(r"mtp\.(\d+)\.(.*)", name)
        if m is None:
            return None
        stage = int(m.group(1))
        rest = m.group(2)
        # The confidence head is not wired into inference yet; drop its weights.
        if rest.startswith("confidence_head."):
            return None
        # Head-stack params live at model level (mtp.last), context combiner at
        # model level (mtp.0); everything else is a per-layer decoder block.
        head_prefixes = (
            "norm.",
            "hc_head_fn",
            "hc_head_base",
            "hc_head_scale",
            "markov_head.",
        )
        if rest.startswith(("main_proj.", "main_norm.")) or rest.startswith(
            head_prefixes
        ):
            return f"model.{rest}"
        return f"model.layers.{stage}.{rest}"
