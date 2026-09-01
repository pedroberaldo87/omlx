# SPDX-License-Identifier: Apache-2.0
"""
Per-scheduler EWMA of bytes-per-prefill-token.

Used by the adaptive prefill throttle in Scheduler: when current memory
enters the caution zone (>= hard_cap * safe_zone_ratio), the next chunk
is sized so its predicted transient stays under the remaining headroom.

Owned by each Scheduler instance (one EWMA per loaded model), distinct
from the global PrefillProgressTracker which feeds the admin dashboard.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class PrefillTransientTracker:
    """EWMA estimator of MLX prefill chunk transient bytes per token.

    Updated post-chunk from `phys_footprint()` deltas. The first chunk
    has no measurement yet — callers fall back to a static estimate
    (MemoryMonitor.estimate_prefill_peak_bytes) until samples > 0.
    """

    _EWMA_ALPHA = 0.3  # weight on the most recent chunk
    # Candidates above this are rejected from the running max: a one-off
    # Metal/pool spike this large is not a repeatable chunk transient, and
    # charging it at admission would refuse prompts that always fit. A
    # genuinely recurring giant transient still reaches the guard through
    # the last-delta/EWMA terms of _predicted_chunk_transient.
    _OBSERVED_MAX_CLAMP_BYTES = 4 * 1024**3
    # A sample whose per_token exceeds the current EWMA by more than this
    # ratio is treated as measurement noise (a tail/residual prefill chunk,
    # not a genuine cost-per-token regime change) and excluded from the EWMA
    # blend. Chosen from a real incident (2026-07-29, Qwen3.6-35B-A3B):
    # baseline samples ranged ~525-1867 KB/token (largest legitimate
    # fluctuation ~1.7x the running EWMA) before a single n=185 tail chunk
    # measured 10497.1 KB/token — a ~13.6x jump off an EWMA of 773.3 KB/token
    # — and pushed the EWMA to 3690.5 KB/token in one update, poisoning every
    # later admission check for the rest of the process lifetime. 8x sits
    # above the largest observed legitimate fluctuation and below the
    # observed outlier.
    _EWMA_OUTLIER_RATIO = 8.0
    # Above this per-token rate a chunk delta is not a per-token cost at all
    # (the 128K-context SDPA row of a 64-head model is ~16MB/token; measured
    # real transients are 2–6MB/token). Such readings are buffer-pool growth
    # divided by a small chunk and must not steer the throttle.
    _PER_TOKEN_SANITY_BYTES = 32 * 1024**2

    def __init__(self, model_id: str = "") -> None:
        self._model_id = model_id
        self._ewma_per_token: float = 0.0
        self._samples: int = 0
        # Last observed delta for debug log inspection.
        self._last_delta_bytes: int = 0
        self._last_n_tokens: int = 0
        # Largest floor-size chunk transient seen this session: a stable
        # flat bound for admission and the pre-chunk guard's pass/abort
        # gates, matching the floor-chunk charge they price. Never used
        # for chunk sizing.
        self._observed_max_bytes: int = 0
        # Net process footprint released by negative post-chunk deltas. MLX may
        # need to allocate that pool again on the next chunk, so the scheduler
        # prices it once until a positive measurement confirms reallocation.
        self._recent_reclaim_bytes: int = 0

    def record_reclaim(self, reclaimed_bytes: int) -> None:
        """Hold the LARGEST footprint released since the last positive sample.

        The charge prices one reallocation of the pool MLX gave back, so it is
        bounded by the biggest single release — not by their sum. Summing made
        the charge grow without limit across a run of negative deltas: measured
        on GLM-5.3-Flash-oQ2e, a short conversation stacked 22.97GB of charge on
        top of a 1.02GB computed transient, and every long prompt after it was
        rejected by the pre-chunk guard at a 124GB ceiling.
        """
        if reclaimed_bytes > 0:
            self._recent_reclaim_bytes = max(
                self._recent_reclaim_bytes, int(reclaimed_bytes)
            )

    def clear_reclaim(self) -> None:
        """Drop the charge once any positive measurement confirms realloc.

        Callers invoke this for every positive delta, including samples the
        EWMA gates skip (sub-floor tails, speed-priority partials) — the
        footprint has grown back, so keeping the charge would double count
        against the guard's gates.
        """
        self._recent_reclaim_bytes = 0

    def update(
        self, n_tokens: int, transient_bytes: int, *, floor_sample: bool = False
    ) -> None:
        """Record one chunk observation.

        Negative deltas (MLX cache pool reclaim larger than this chunk's
        allocation) are skipped — they would bias the EWMA toward zero
        and underestimate the next chunk's footprint.

        ``floor_sample`` marks a chunk at the throttle's floor size. Only
        those feed the running max: admission charges the floor chunk, and
        chunk-transient maxima are NOT size-invariant across models
        (Qwen3.6 measured ~3.0GB at 2048-token chunks vs far less at the
        floor; charging the big-chunk max at admission rejected every
        prompt at a 21GB ceiling). Big-chunk transients stay the throttle's
        domain via the EWMA/last-delta terms.

        A sample whose per-token rate exceeds the current EWMA by more than
        ``_EWMA_OUTLIER_RATIO`` is excluded from the EWMA blend (see that
        constant's docstring) — it still counts toward ``samples`` and
        still updates ``last_delta_bytes``/``last_n_tokens`` raw, so a
        genuine regime change remains visible via those fields even while
        the accumulated EWMA is protected from a single noisy reading. A
        sample above ``_PER_TOKEN_SANITY_BYTES`` per token is dropped from
        both: no model costs that per token, so it is pool noise over a
        small chunk.
        """
        if n_tokens <= 0:
            return
        if transient_bytes <= 0:
            return

        self._recent_reclaim_bytes = 0

        # The very first sample after a model load carries weight page-fault
        # and load-residue noise, so it seeds the EWMA but is excluded from
        # the running max.
        if floor_sample and self._samples > 0:
            if transient_bytes <= self._OBSERVED_MAX_CLAMP_BYTES:
                if transient_bytes > self._observed_max_bytes:
                    self._observed_max_bytes = transient_bytes
            else:
                logger.debug(
                    "PrefillTransientTracker(%s): rejected %d-byte outlier "
                    "from observed max (clamp %d)",
                    self._model_id,
                    transient_bytes,
                    self._OBSERVED_MAX_CLAMP_BYTES,
                )

        per_token = transient_bytes / n_tokens
        if per_token > self._PER_TOKEN_SANITY_BYTES:
            # Physically impossible as a per-token cost (KV + SDPA transient
            # of any served model stays far below this): it is buffer-pool
            # growth or load residue divided by a small chunk. It must not
            # seed the EWMA, blend into it, nor land in
            # last_delta_bytes/last_n_tokens — the predictor
            # (_predicted_chunk_transient) reads that pair as a rate and takes
            # the MAX with the EWMA. Measured on GLM-5.3 oQ2e (01/09): the
            # 14-token warm-up after load left 2.2GB of residue, seeded the
            # EWMA at 157025KB/token (the 4GB seed clamp is on TOTAL bytes,
            # so it passed), and the 29K-token prompt ran at the 32-token
            # floor for 186s — real cost is 2.1–5.7MB/token.
            logger.debug(
                "PrefillTransientTracker(%s): dropped %.1f-byte/token sample "
                "above the per-token sanity bound (%d)",
                self._model_id,
                per_token,
                self._PER_TOKEN_SANITY_BYTES,
            )
            # Not counted as a sample: the next sane reading has to SEED the
            # EWMA (an EWMA of 0 would reject every later sample as an
            # outlier against 0 × ratio).
            return
        if self._samples == 0:
            # The seeding sample carries the same load-residue noise the
            # running max is already protected from, and until now it entered
            # raw: one contaminated reading became the EWMA and every later
            # sample was measured against it, so the outlier ratio below could
            # never pull the rate back down. Measured on GLM-5.3-Flash-oQ2e:
            # a 10GB seed at a 2048-token chunk set 4.9MB/token against a
            # 0.38MB/token static truth, and admission then charged 12.75GB
            # for a prompt that costs 1.02GB — rejecting every request.
            if transient_bytes <= self._OBSERVED_MAX_CLAMP_BYTES:
                self._ewma_per_token = per_token
            else:
                logger.debug(
                    "PrefillTransientTracker(%s): rejected %d-byte seeding "
                    "sample from EWMA (clamp %d); the static estimator holds "
                    "until a sample within the clamp arrives",
                    self._model_id,
                    transient_bytes,
                    self._OBSERVED_MAX_CLAMP_BYTES,
                )
                # last_delta_bytes is NOT recorded here on purpose: the
                # predictor reads it as a per-token rate of its own, so
                # keeping the contaminated reading would reintroduce the
                # exact number this branch exists to reject.
                return
        elif per_token > self._ewma_per_token * self._EWMA_OUTLIER_RATIO:
            # Reject from the EWMA blend: a single sample this far above
            # the running rate is more likely a noisy phys_footprint()
            # delta (see _record_chunk_transient's docstring on
            # buffer-pool-driven noise) than a real per-token cost jump.
            # last_delta_bytes/last_n_tokens below still record it raw: a
            # same-size spike within the sanity bound protects the next
            # full step (test_record_chunk_transient_keeps_full_speed_spike).
            logger.debug(
                "PrefillTransientTracker(%s): rejected %.1f-byte/token "
                "outlier from EWMA (current %.1f, ratio limit %.1fx)",
                self._model_id,
                per_token,
                self._ewma_per_token,
                self._EWMA_OUTLIER_RATIO,
            )
        else:
            self._ewma_per_token = (
                self._EWMA_ALPHA * per_token
                + (1.0 - self._EWMA_ALPHA) * self._ewma_per_token
            )
        self._samples += 1
        self._last_delta_bytes = transient_bytes
        self._last_n_tokens = n_tokens

    def predict(self, n_tokens: int, *, safety_factor: float = 1.2) -> int:
        """Predicted transient bytes for a chunk of `n_tokens`.

        Returns 0 when no samples have been observed yet — caller must
        fall back to a static estimator in that case.
        """
        if self._samples == 0 or n_tokens <= 0:
            return 0
        return int(self._ewma_per_token * n_tokens * safety_factor)

    @property
    def bytes_per_token(self) -> float:
        """Current EWMA value (bytes per prefill token). 0.0 if no samples."""
        return self._ewma_per_token

    @property
    def samples(self) -> int:
        """Number of chunks recorded since last reset."""
        return self._samples

    @property
    def last_delta_bytes(self) -> int:
        """Bytes added by the most recently measured chunk."""
        return self._last_delta_bytes

    @property
    def last_n_tokens(self) -> int:
        """Token count of the most recently measured chunk."""
        return self._last_n_tokens

    @property
    def observed_max_bytes(self) -> int:
        """Largest accepted chunk transient this session (0 if none yet)."""
        return self._observed_max_bytes

    @property
    def recent_reclaim_bytes(self) -> int:
        """Footprint released since the last positive chunk measurement."""
        return self._recent_reclaim_bytes

    def reset(self) -> None:
        """Drop all observations (e.g. on model reload or after a long idle)."""
        self._ewma_per_token = 0.0
        self._samples = 0
        self._last_delta_bytes = 0
        self._last_n_tokens = 0
        self._observed_max_bytes = 0
        self._recent_reclaim_bytes = 0
