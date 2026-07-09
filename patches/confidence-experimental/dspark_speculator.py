# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly ``N =
    num_speculative_tokens`` query tokens (anchor + N-1 noise), NOT ``1 + N``.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and ``sample_pos = query_pos + 1``
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the ``sample_from_anchor`` path in the shared prepare-inputs kernel.
    Speculators-format checkpoints instead use the DFlash ``1 + N`` fill-in
    layout (anchor is the bonus token).
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.

PATCH(gb10-dspark-conf): confidence-head draft-length scheduling (stage-c port).
When the checkpoint carries a DSpark confidence head and
VLLM_DSPARK_CONFIDENCE_THRESHOLD / _SCHEDULER enables it, `_sample_sequential`
also evaluates ``sigmoid(confidence_head(head_hidden_i, markov_embed_i))`` per
draft position into a persistent ``[max_num_reqs, N]`` buffer. The model runner
hands that buffer plus `schedule_draft_lengths` to the DraftTokensHandler, which
converts confidences to per-request verification prefix lengths at
`take_draft_token_ids()` time (sync-scheduling only; see confidence.py).
"""

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.confidence import (
    DSparkConfidenceScheduler,
)
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

logger = init_logger(__name__)


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Anchor-as-first (N slots) unless the checkpoint uses the 1+N fill-in
        # block, where the anchor is a separate bonus token.
        self.sample_from_anchor = not getattr(
            self.draft_model_config.hf_config, "dspark_bonus_anchor", False
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps

        # DSpark consumes mean-pooled target aux hidden states at the target
        # layers, combined to hidden_size via main_proj. Store that combined
        # main_x (hidden_size wide). DSpark does not use the same pre-allocated buffer
        # that DeepSeek-V4's MTP uses.
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )

        self.dflash_causal = False

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        # Reduced-vocab probabilistic drafting only; set in load_draft_model.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None

        # PATCH(gb10-dspark-conf): persistent per-(req, position) confidence
        # buffer. Written inside _sample_sequential (fixed shapes / persistent
        # storage, so FULL-graph capture-safe like self.draft_tokens); read by
        # the model runner after propose() and async-copied to CPU by the
        # DraftTokensHandler. Scheduler wired in load_draft_model.
        self.draft_confidence = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.float32,
            device=device,
        )
        self._conf_sched: DSparkConfidenceScheduler | None = None

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            # -inf once; the per-step scatter overwrites the draft->target
            # columns. Kept separate from draft_logits to avoid aliasing.
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )

        # PATCH(gb10-dspark-conf): wire the confidence scheduler once the draft
        # model (and hence its loaded weights) is known.
        conf_sched = DSparkConfidenceScheduler(self.num_speculative_steps)
        if conf_sched.enabled:
            if not getattr(model, "has_confidence_head", False):
                logger.warning(
                    "DSpark confidence scheduling requested (%s) but the draft "
                    "model has no loaded confidence head; disabling.",
                    conf_sched.describe(),
                )
            elif self.vllm_config.scheduler_config.async_scheduling:
                logger.warning(
                    "DSpark confidence scheduling requested (%s) but async "
                    "scheduling is enabled. Per-request draft lengths only "
                    "reach the scheduler through the synchronous "
                    "take_draft_token_ids()/update_draft_token_ids() path; "
                    "relaunch with --no-async-scheduling. Disabling.",
                    conf_sched.describe(),
                )
            else:
                self._conf_sched = conf_sched
                logger.info(
                    "DSpark confidence-scheduled verification enabled: %s",
                    conf_sched.describe(),
                )
        return model

    # PATCH(gb10-dspark-conf): hooks consumed by the model runner.
    @property
    def confidence_scheduling_enabled(self) -> bool:
        return self._conf_sched is not None

    def get_draft_confidences(self, num_reqs: int) -> torch.Tensor:
        """[num_reqs, num_speculative_steps] confidences in draft batch order
        (identical to input_batch order)."""
        return self.draft_confidence[:num_reqs]

    def schedule_draft_lengths(self, confidence_rows: list[list[float]]) -> list[int]:
        assert self._conf_sched is not None
        return self._conf_sched.schedule(confidence_rows)

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)

        # PATCH(gb10-dspark-conf): the confidence head consumes the PRE-norm
        # head hidden at each draft position (stage-c's `dense`) together with
        # the markov embedding of the position's predecessor token — exactly
        # the `markov_embed` computed in the loop below.
        conf_enabled = self._conf_sched is not None
        sample_hidden_3d = None
        if conf_enabled:
            sample_hidden_3d = sample_hidden.view(num_reqs, n_spec, -1)

        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        # PATCH(gb10-dspark-cg): under CUDA-graph replay, rows padded beyond the
        # live request count carry stale/undefined ids; a negative or OOB id here
        # is an out-of-bounds embedding gather INSIDE the replayed graph (silent
        # corruption -> the [-1,...] draft-token dumps). Clamp everything that
        # feeds markov_embed to its valid range; padded rows' outputs are
        # discarded downstream, so any in-range id is safe.
        _emb_max = self.vocab_size - 1
        prev = torch.clamp(prev, 0, _emb_max)

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            if conf_enabled:
                # PATCH(gb10-dspark-conf): per-position conditional acceptance
                # confidence: sigmoid(confidence_head([dense_i, markov_embed])).
                # Fixed shapes + persistent buffer write: capture-safe. Padded
                # rows produce garbage but are sliced away at consume time.
                conf_i = self.model.confidence_logits(
                    sample_hidden_3d[:, i], markov_embed
                )
                conf_i = torch.sigmoid(conf_i.float())
                self.draft_confidence[:num_reqs, i] = torch.nan_to_num(
                    conf_i, nan=0.0
                ).clamp_(0.0, 1.0)
            if self.draft_logits is not None:
                # Probabilistic: sample in target vocab (a reduced draft vocab is
                # scattered into its target columns; full vocab is already there).
                if self._d2t_scatter_index is not None:
                    assert self._draft_scatter_buf is not None
                    buf = self._draft_scatter_buf[:num_reqs]
                    buf.index_copy_(1, self._d2t_scatter_index, logits_i.to(buf.dtype))
                    logits_i = buf
                # sample_pos is the predicted token's position Q; the target
                # verifies it with the predecessor's Gumbel key (Q-1). Pass Q-1.
                draft_sampled_i = gumbel_sample(
                    logits_i,
                    idx_map[:, i],
                    self.temperature,
                    self.seeds,
                    sample_pos[:, i] - 1,
                    apply_temperature=True,
                    output_processed_logits=self.draft_logits,
                    output_processed_logits_col=self._step_cols[i],
                    use_fp64=self.use_fp64_gumbel,
                )
            else:
                draft_sampled_i = self.model.map_draft_to_target(
                    logits_i.argmax(dim=-1)
                )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            # PATCH(gb10-dspark-cg): same clamp for the chained feedback — a
            # skipped (padded) row's sample return is undefined by contract.
            prev = torch.clamp(draft_sampled_i, 0, _emb_max)

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        # Full draft step (captured under CUDA graph): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._sample_sequential(num_reqs, head_hidden)
