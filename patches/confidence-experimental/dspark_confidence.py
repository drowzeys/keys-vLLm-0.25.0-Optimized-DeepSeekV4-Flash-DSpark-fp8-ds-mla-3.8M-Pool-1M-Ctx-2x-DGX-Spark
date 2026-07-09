# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# PATCH(gb10-dspark-conf): NEW FILE.
# Confidence-head draft-length scheduling for DSpark on the MRV2 speculator
# stack, ported from the stage-c (vLLM 0.24 DSpark transplant) proposer:
#   - confidence_threshold_prefix_length / cumulative_survival /
#     score_prefix_lengths / hardware_aware_prefix_schedule are VERBATIM ports
#     of stage-c vllm/v1/spec_decode/dspark.py.
#   - DSparkConfidenceScheduler mirrors the stage-c proposer's
#     _schedule_from_confidence() dispatch and env-var knobs:
#       VLLM_DSPARK_CONFIDENCE_THRESHOLD   float in [0,1]; >0 enables scheduling
#       VLLM_DSPARK_CONFIDENCE_SCHEDULER   off|threshold|hardware|auto
#       VLLM_DSPARK_FORCE_DRAFT_LENGTH     int, fixed length (profiling)
#       VLLM_DSPARK_SPS_CURVE              "tokens:rate,tokens:rate,..."
#       VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP  bool (default 1)
#       VLLM_DSPARK_MIN_DRAFT_LENGTH       int floor on scheduled lengths (default 0)
#       VLLM_DSPARK_CONFIDENCE_LOG_EVERY   log diagnostics every N steps (0=off)
#
# Mount target:
#   /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dspark/confidence.py

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from vllm.logger import init_logger

logger = init_logger(__name__)

StepCurve = Callable[[int], float]


@dataclass(frozen=True)
class DSparkScheduleResult:
    """Selected DSpark verification lengths and their profiled throughput."""

    lengths: tuple[int, ...]
    expected_accepted_tokens: float
    batch_tokens: int
    expected_tokens_per_second: float


def _validate_probability(value: float, name: str) -> float:
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _sigmoid_scalar(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def cumulative_survival(confidences: Sequence[float]) -> tuple[float, ...]:
    """Convert conditional per-token confidences to prefix survival rates."""

    survivals: list[float] = []
    survival = 1.0
    for index, confidence in enumerate(confidences):
        survival *= _validate_probability(confidence, f"confidences[{index}]")
        survivals.append(survival)
    return tuple(survivals)


def confidence_threshold_prefix_length(
    confidences: Sequence[float],
    threshold: float,
) -> int:
    """Choose the longest prefix whose cumulative confidence stays above a threshold.

    The decision is non-anticipating: position n is admitted using only
    confidences from positions <= n. This keeps DSpark's confidence scheduling
    compatible with rejection sampling.
    """

    threshold = _validate_probability(threshold, "threshold")
    admitted = 0
    survival = 1.0
    for index, confidence in enumerate(confidences):
        survival *= _validate_probability(confidence, f"confidences[{index}]")
        if survival < threshold:
            break
        admitted += 1
    return admitted


def score_prefix_lengths(
    confidence_rows: Sequence[Sequence[float]],
    lengths: Sequence[int],
    *,
    steps_per_second: StepCurve,
) -> DSparkScheduleResult:
    """Score a fixed per-request prefix schedule against a profiled step curve."""

    if len(confidence_rows) != len(lengths):
        raise ValueError("confidence_rows and lengths must have the same length")

    expected_accepted_tokens = float(len(confidence_rows))
    batch_tokens = len(confidence_rows)
    normalized_lengths: list[int] = []

    for request_index, (confidences, length) in enumerate(
        zip(confidence_rows, lengths, strict=True)
    ):
        length = int(length)
        if length < 0 or length > len(confidences):
            raise ValueError(
                f"lengths[{request_index}] must be in [0, {len(confidences)}], "
                f"got {length}"
            )

        expected_accepted_tokens += sum(cumulative_survival(confidences)[:length])
        batch_tokens += length
        normalized_lengths.append(length)

    step_rate = float(steps_per_second(batch_tokens))
    if step_rate < 0.0:
        raise ValueError("steps_per_second must return a non-negative value")

    return DSparkScheduleResult(
        lengths=tuple(normalized_lengths),
        expected_accepted_tokens=expected_accepted_tokens,
        batch_tokens=batch_tokens,
        expected_tokens_per_second=expected_accepted_tokens * step_rate,
    )


def hardware_aware_prefix_schedule(
    confidence_rows: Sequence[Sequence[float]],
    *,
    steps_per_second: StepCurve,
    early_stop: bool = True,
) -> DSparkScheduleResult:
    """Choose DSpark verification prefix lengths for a batch.

    `confidence_rows[r][j]` is the conditional acceptance probability for
    request r at draft position j. The planner greedily admits prefix tokens by
    descending cumulative survival, then keeps the best point along that path
    after applying the supplied profiled engine step-rate curve.

    `early_stop=True` is intended for smooth capacity curves where adding more
    verification tokens cannot recover from the first throughput drop. Set it to
    `False` for exhaustive traversal of the greedy prefix path when the profiled
    curve has jagged capacity cliffs.
    """

    request_count = len(confidence_rows)
    lengths = [0] * request_count
    best_lengths = tuple(lengths)
    batch_tokens = request_count

    base_step_rate = float(steps_per_second(batch_tokens))
    if base_step_rate < 0.0:
        raise ValueError("steps_per_second must return a non-negative value")

    expected_accepted_tokens = float(request_count)
    best_batch_tokens = batch_tokens
    best_expected_accepted_tokens = expected_accepted_tokens
    best_tokens_per_second = expected_accepted_tokens * base_step_rate

    candidates: list[tuple[float, int, int]] = []
    for request_index, confidences in enumerate(confidence_rows):
        for position, survival in enumerate(cumulative_survival(confidences), start=1):
            if survival > 0.0:
                candidates.append((survival, request_index, position))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    for survival, request_index, position in candidates:
        if position != lengths[request_index] + 1:
            continue

        lengths[request_index] = position
        batch_tokens += 1
        expected_accepted_tokens += survival

        step_rate = float(steps_per_second(batch_tokens))
        if step_rate < 0.0:
            raise ValueError("steps_per_second must return a non-negative value")
        tokens_per_second = expected_accepted_tokens * step_rate

        if tokens_per_second > best_tokens_per_second:
            best_lengths = tuple(lengths)
            best_batch_tokens = batch_tokens
            best_expected_accepted_tokens = expected_accepted_tokens
            best_tokens_per_second = tokens_per_second
            continue

        if early_stop:
            break

    return DSparkScheduleResult(
        lengths=best_lengths,
        expected_accepted_tokens=best_expected_accepted_tokens,
        batch_tokens=best_batch_tokens,
        expected_tokens_per_second=best_tokens_per_second,
    )


# ---------------------------------------------------------------------------
# Env-var knobs (verbatim semantics from the stage-c DSparkProposer readers).
# ---------------------------------------------------------------------------


def _read_confidence_threshold() -> float:
    raw = os.getenv("VLLM_DSPARK_CONFIDENCE_THRESHOLD", "0.0")
    try:
        threshold = float(raw)
    except ValueError as exc:
        raise ValueError(
            "VLLM_DSPARK_CONFIDENCE_THRESHOLD must be a float in [0, 1], "
            f"got {raw!r}"
        ) from exc
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(
            f"VLLM_DSPARK_CONFIDENCE_THRESHOLD must be in [0, 1], got {threshold}"
        )
    return threshold


def _read_confidence_scheduler(confidence_threshold: float) -> str:
    raw = os.getenv("VLLM_DSPARK_CONFIDENCE_SCHEDULER", "auto")
    scheduler = raw.strip().lower()
    if scheduler in {"", "auto"}:
        return "threshold" if confidence_threshold > 0.0 else "off"
    if scheduler not in {"off", "threshold", "hardware"}:
        raise ValueError(
            "VLLM_DSPARK_CONFIDENCE_SCHEDULER must be one of "
            f"'off', 'threshold', 'hardware', or 'auto', got {raw!r}"
        )
    return scheduler


def _read_forced_draft_length(max_draft_length: int) -> int | None:
    raw = os.getenv("VLLM_DSPARK_FORCE_DRAFT_LENGTH", "").strip()
    if raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "VLLM_DSPARK_FORCE_DRAFT_LENGTH must be an integer in "
            f"[0, {max_draft_length}] or empty, got {raw!r}"
        ) from exc
    if value < 0 or value > max_draft_length:
        raise ValueError(
            "VLLM_DSPARK_FORCE_DRAFT_LENGTH must be in "
            f"[0, {max_draft_length}], got {value}"
        )
    return value


def _read_sps_curve() -> tuple[tuple[int, float], ...]:
    raw = os.getenv("VLLM_DSPARK_SPS_CURVE", "").strip()
    if not raw:
        return ()

    entries: dict[int, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            batch_tokens_raw, rate_raw = item.split(":", 1)
            batch_tokens = int(batch_tokens_raw)
            rate = float(rate_raw)
        except ValueError as exc:
            raise ValueError(
                "VLLM_DSPARK_SPS_CURVE must be a comma-separated table of "
                f"'<batch_tokens>:<steps_per_second>' entries, got {raw!r}"
            ) from exc
        if batch_tokens <= 0:
            raise ValueError(
                "VLLM_DSPARK_SPS_CURVE batch-token keys must be positive, "
                f"got {batch_tokens}"
            )
        if rate < 0.0:
            raise ValueError(
                f"VLLM_DSPARK_SPS_CURVE rates must be non-negative, got {rate}"
            )
        entries[batch_tokens] = rate
    return tuple(sorted(entries.items()))


def _read_hardware_scheduler_early_stop() -> bool:
    raw = os.getenv("VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP", "1")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _read_min_draft_length(max_draft_length: int) -> int:
    raw = os.getenv("VLLM_DSPARK_MIN_DRAFT_LENGTH", "0").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "VLLM_DSPARK_MIN_DRAFT_LENGTH must be an integer in "
            f"[0, {max_draft_length}], got {raw!r}"
        ) from exc
    if value < 0 or value > max_draft_length:
        raise ValueError(
            "VLLM_DSPARK_MIN_DRAFT_LENGTH must be in "
            f"[0, {max_draft_length}], got {value}"
        )
    return value


def _read_log_every() -> int:
    raw = os.getenv("VLLM_DSPARK_CONFIDENCE_LOG_EVERY", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Diagnostics (trimmed stage-c DSparkDiagnostics: length histogram + prune rate)
# ---------------------------------------------------------------------------


@dataclass
class DSparkDiagnostics:
    """Accumulate DSpark confidence-scheduler diagnostics."""

    max_spec_tokens: int
    num_steps: int = 0
    num_requests: int = 0
    num_possible_draft_tokens: int = 0
    num_scheduled_draft_tokens: int = 0
    total_expected_accepted_tokens: float = 0.0
    scheduled_length_histogram: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.max_spec_tokens = int(self.max_spec_tokens)
        if self.max_spec_tokens <= 0:
            raise ValueError(
                f"max_spec_tokens must be positive, got {self.max_spec_tokens}"
            )
        self.scheduled_length_histogram = [0] * (self.max_spec_tokens + 1)

    def observe(
        self,
        confidence_rows: Sequence[Sequence[float]],
        schedule_result: DSparkScheduleResult,
    ) -> None:
        self.num_steps += 1
        self.num_requests += len(confidence_rows)
        self.num_possible_draft_tokens += sum(len(row) for row in confidence_rows)
        self.num_scheduled_draft_tokens += sum(schedule_result.lengths)
        self.total_expected_accepted_tokens += schedule_result.expected_accepted_tokens
        for length in schedule_result.lengths:
            self.scheduled_length_histogram[min(length, self.max_spec_tokens)] += 1

    def format_line(self) -> str:
        avg_len = (
            self.num_scheduled_draft_tokens / self.num_requests
            if self.num_requests
            else 0.0
        )
        prune = (
            1.0 - self.num_scheduled_draft_tokens / self.num_possible_draft_tokens
            if self.num_possible_draft_tokens
            else 0.0
        )
        exp_acc = (
            self.total_expected_accepted_tokens / self.num_requests
            if self.num_requests
            else 0.0
        )
        return (
            f"steps={self.num_steps} reqs={self.num_requests} "
            f"avg_sched_len={avg_len:.3f} prune_rate={prune:.3f} "
            f"exp_accept_len={exp_acc:.3f} "
            f"len_hist={self.scheduled_length_histogram}"
        )


# ---------------------------------------------------------------------------
# Runtime scheduler used by DSparkSpeculator / DraftTokensHandler
# ---------------------------------------------------------------------------


class DSparkConfidenceScheduler:
    """CPU-side per-request draft-length scheduler.

    Mirrors the stage-c proposer's `_schedule_from_confidence` dispatch:
    forced length > hardware-aware planner > threshold prefix > off.

    `schedule(rows)` receives `[num_reqs][num_spec]` confidences (already
    sigmoid'ed and clamped to [0, 1] on GPU) and returns per-request
    verification prefix lengths. Runs at `take_draft_token_ids()` time on the
    driver worker; batch sizes here are tiny (max_num_seqs x num_spec), so the
    verbatim pure-python stage-c logic is used unmodified.
    """

    def __init__(self, num_speculative_tokens: int):
        self.num_speculative_tokens = int(num_speculative_tokens)
        self.confidence_threshold = _read_confidence_threshold()
        self.scheduler = _read_confidence_scheduler(self.confidence_threshold)
        self.forced_draft_length = _read_forced_draft_length(
            self.num_speculative_tokens
        )
        self.sps_curve = _read_sps_curve()
        self.hardware_early_stop = _read_hardware_scheduler_early_stop()
        self.min_draft_length = _read_min_draft_length(self.num_speculative_tokens)
        self.log_every = _read_log_every()
        self.diagnostics = DSparkDiagnostics(
            max_spec_tokens=self.num_speculative_tokens
        )

    @property
    def enabled(self) -> bool:
        return (
            self.forced_draft_length is not None
            or self.scheduler in ("threshold", "hardware")
        )

    def describe(self) -> str:
        return (
            f"scheduler={self.scheduler} threshold={self.confidence_threshold} "
            f"forced={self.forced_draft_length} min_len={self.min_draft_length} "
            f"sps_curve={self.sps_curve or 'constant'} "
            f"early_stop={self.hardware_early_stop}"
        )

    def _steps_per_second(self, batch_tokens: int) -> float:
        curve = self.sps_curve
        if not curve:
            return 1.0
        batch_tokens = int(batch_tokens)
        selected_rate = curve[0][1]
        for profiled_tokens, rate in curve:
            if batch_tokens < profiled_tokens:
                break
            selected_rate = rate
        return selected_rate

    @staticmethod
    def _sanitize_rows(rows: list[list[float]]) -> list[list[float]]:
        # GPU already clamps, but a NaN/garbage row must never raise inside the
        # engine step: clamp defensively before the strict verbatim functions.
        out: list[list[float]] = []
        for row in rows:
            out.append(
                [
                    0.0 if (c != c) else (0.0 if c < 0.0 else (1.0 if c > 1.0 else c))
                    for c in row
                ]
            )
        return out

    def _schedule_from_confidence(
        self, confidence_rows: list[list[float]]
    ) -> DSparkScheduleResult:
        confidence_rows = [
            row[: self.num_speculative_tokens] for row in confidence_rows
        ]
        if self.forced_draft_length is not None:
            lengths = [
                min(self.forced_draft_length, self.num_speculative_tokens, len(row))
                for row in confidence_rows
            ]
            return score_prefix_lengths(
                confidence_rows, lengths, steps_per_second=self._steps_per_second
            )

        if self.scheduler == "hardware":
            return hardware_aware_prefix_schedule(
                confidence_rows,
                steps_per_second=self._steps_per_second,
                early_stop=self.hardware_early_stop,
            )

        if self.scheduler == "off" or self.confidence_threshold <= 0.0:
            lengths = [
                min(self.num_speculative_tokens, len(row)) for row in confidence_rows
            ]
        else:
            lengths = [
                confidence_threshold_prefix_length(row, self.confidence_threshold)
                for row in confidence_rows
            ]
        return score_prefix_lengths(
            confidence_rows, lengths, steps_per_second=self._steps_per_second
        )

    def schedule(self, confidence_rows: list[list[float]]) -> list[int]:
        if not confidence_rows:
            return []
        confidence_rows = self._sanitize_rows(confidence_rows)
        result = self._schedule_from_confidence(confidence_rows)
        lengths = list(result.lengths)
        if self.min_draft_length > 0:
            lengths = [
                max(length, min(self.min_draft_length, len(row)))
                for length, row in zip(lengths, confidence_rows)
            ]
        self.diagnostics.observe(confidence_rows, result)
        if self.log_every and self.diagnostics.num_steps % self.log_every == 0:
            logger.info("DSpark confidence scheduler: %s", self.diagnostics.format_line())
        return lengths
