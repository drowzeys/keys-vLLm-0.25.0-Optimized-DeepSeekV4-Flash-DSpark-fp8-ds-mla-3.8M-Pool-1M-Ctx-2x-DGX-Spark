# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# PATCH(gb10-dspark-conf): DraftTokensHandler learns optional per-request
# draft-length scheduling from confidence scores. Mount target:
#   /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/utils.py
from collections.abc import Callable

import numpy as np
import torch

from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch


class DraftTokensHandler:
    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0
        # PATCH(gb10-dspark-conf): per-request confidence rows + the scheduler
        # callback that converts them to verification prefix lengths.
        self.draft_confidences_np: np.ndarray | None = None
        self.length_scheduler: Callable[[list[list[float]]], list[int]] | None = None

    def set_draft_tokens(
        self,
        input_batch: InputBatch,
        draft_tokens: torch.Tensor,
        # PATCH(gb10-dspark-conf): optional [num_reqs, num_draft_tokens] float32
        # confidence tensor (GPU) and a CPU scheduler fn. When both are given,
        # get_draft_tokens() truncates each request's draft list to the
        # scheduled length, which (in synchronous scheduling) flows through
        # Scheduler.update_draft_token_ids -> request.spec_token_ids and makes
        # the next step verify only that prefix.
        draft_confidences: torch.Tensor | None = None,
        length_scheduler: Callable[[list[list[float]]], list[int]] | None = None,
    ) -> None:
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
        need_tokens = input_batch.has_structured_output_reqs
        need_confidences = (
            draft_confidences is not None and length_scheduler is not None
        )
        self.length_scheduler = length_scheduler if need_confidences else None
        if not need_tokens and not need_confidences:
            # No draft token validation or length scheduling needs to be
            # performed by the scheduler for this batch.
            self.draft_tokens_np = None
            self.draft_confidences_np = None
            return

        # For spec decoding + structured outputs, we must transfer the
        # draft tokens back to the scheduler for grammar validation.
        # PATCH(gb10-dspark-conf): confidences ride the same copy stream/event.
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            if need_tokens:
                self.draft_tokens_np = async_copy_to_np(draft_tokens)
                # draft_tokens is a temporary allocation on the main stream and read
                # here on copy_stream; without record_stream, the caching allocator
                # may reuse its memory before the async copy executes.
                draft_tokens.record_stream(self.copy_stream)
            else:
                self.draft_tokens_np = None
            if need_confidences:
                assert draft_confidences is not None
                # draft_confidences is a view of the speculator's persistent
                # buffer (never freed), so no record_stream is needed; the
                # wait_stream above orders the copy after the producer kernels.
                self.draft_confidences_np = async_copy_to_np(
                    draft_confidences.float()
                )
            else:
                self.draft_confidences_np = None
            self.copy_event.record()

    def get_draft_tokens(self) -> DraftTokenIds | None:
        if self.draft_tokens_np is not None or self.draft_confidences_np is not None:
            self.copy_event.synchronize()

        if self.draft_tokens_np is not None:
            draft_token_ids = self.draft_tokens_np.tolist()
        else:
            # This case only happens when async scheduling is disabled.
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]

        # PATCH(gb10-dspark-conf): convert confidences to per-request prefix
        # lengths and truncate. The scheduler only consumes list lengths (and,
        # for structured outputs, token values for grammar validation); the
        # actual draft input ids stay in the GPU-side req_states.draft_tokens
        # buffer, of which only the first len() slots per request are read.
        if self.draft_confidences_np is not None and self.length_scheduler is not None:
            lengths = self.length_scheduler(self.draft_confidences_np.tolist())
            if len(lengths) == len(draft_token_ids):
                draft_token_ids = [
                    row[: max(0, min(int(length), len(row)))]
                    for row, length in zip(draft_token_ids, lengths)
                ]
        return DraftTokenIds(self.req_ids, draft_token_ids)


def get_parallel_drafting_token_id(hf_config) -> int:
    """Resolve the mask token id used for parallel drafting slots.

    Checks (in order): `dflash_config.mask_token_id`, top-level `mask_token_id`,
    `dspark_noise_token_id`, `pard_token`, `ptd_token_id`. Raises ValueError if
    none are present.
    """
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    if "mask_token_id" in dflash_config:
        return int(dflash_config["mask_token_id"])
    if getattr(hf_config, "mask_token_id", None) is not None:
        return int(hf_config.mask_token_id)
    if hasattr(hf_config, "dspark_noise_token_id"):
        return int(hf_config.dspark_noise_token_id)
    if hasattr(hf_config, "pard_token"):
        return int(hf_config.pard_token)
    if hasattr(hf_config, "ptd_token_id"):
        return int(hf_config.ptd_token_id)
    raise ValueError(
        "Model config must specify `dflash_config.mask_token_id`,"
        " `mask_token_id`, `dspark_noise_token_id`, `pard_token`, or"
        " `ptd_token_id` for parallel drafting."
    )
