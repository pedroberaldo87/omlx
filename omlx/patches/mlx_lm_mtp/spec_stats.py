# SPDX-License-Identifier: Apache-2.0
"""Process-global session accumulator for MTP/n-gram speculation stats.

``_MtpStats`` used to die inside ``_log_mtp_stats`` — the operator could
toggle speculation on but never SEE it working outside the raw log. This
module folds every finished sequence's counters into one lock-guarded
session total, in the exact ``{"last": ..., "totals": ...}`` shape the
admin bridge already duck-types for DFlash (routes.py picks it up via
``getattr(entry.engine, "get_speculation_stats", None)`` — no route
changes needed).

Process-global on purpose: the accumulator lives below the engine layer
(inside the mlx-lm patch), where no engine handle exists. With several
MTP-active models loaded at once the numbers aggregate across them; the
typical single-model deployment reads per-model.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

_LOCK = threading.Lock()
_TOTALS = {
    "requests": 0,
    "speculative_requests": 0,
    "fallback_requests": 0,
    "generation_tokens": 0,
    "accepted_draft_tokens": 0,
    "cycles": 0,
    "ngram_served_cycles": 0,
    "ngram_miss_cycles": 0,
}
_LAST: Optional[dict] = None


def record(stats: Any, finish_reason: str) -> None:
    """Fold one finished sequence's ``_MtpStats`` into the session totals."""
    global _LAST
    total_emits = (
        stats.init_emits + stats.draft_emits + stats.bonus_emits + stats.verify_emits
    )
    total_drafted = sum(stats.depth_drafted) or stats.cycles
    cycles = stats.cycles
    last = {
        "finish_reason": finish_reason,
        "generation_tokens": total_emits,
        "accepted_draft_tokens": stats.accepts,
        "cycles": cycles,
        # Share of output supplied by the draft — same semantics as the
        # DFlash panel (issue #2398), so the dashboard row reads uniformly.
        "acceptance_ratio": (stats.accepts / total_emits) if total_emits else None,
        "tokens_per_cycle": (total_emits / cycles) if cycles else None,
        "accepted_draft_tokens_per_cycle": (stats.accepts / cycles) if cycles else None,
        "draft_accept_rate": (
            stats.accepts / total_drafted if total_drafted else None
        ),
        "ngram_served_cycles": stats.ngram_cycles,
        "ngram_miss_cycles": stats.ngram_misses,
        "fallback_ar": False,
    }
    with _LOCK:
        _LAST = last
        _TOTALS["requests"] += 1
        if stats.accepts > 0:
            _TOTALS["speculative_requests"] += 1
        else:
            _TOTALS["fallback_requests"] += 1
        _TOTALS["generation_tokens"] += total_emits
        _TOTALS["accepted_draft_tokens"] += stats.accepts
        _TOTALS["cycles"] += cycles
        _TOTALS["ngram_served_cycles"] += stats.ngram_cycles
        _TOTALS["ngram_miss_cycles"] += stats.ngram_misses


def get_speculation_stats() -> Optional[dict]:
    """DFlash-shaped session snapshot; None before any MTP request finishes."""
    with _LOCK:
        if _TOTALS["requests"] == 0:
            return None
        totals = dict(_TOTALS)
        last = dict(_LAST) if _LAST else None
    gen = totals["generation_tokens"]
    cycles = totals["cycles"]
    totals["acceptance_ratio"] = (
        totals["accepted_draft_tokens"] / gen if gen > 0 else None
    )
    totals["tokens_per_cycle"] = gen / cycles if cycles > 0 else None
    totals["accepted_draft_tokens_per_cycle"] = (
        totals["accepted_draft_tokens"] / cycles if cycles > 0 else None
    )
    return {"last": last, "totals": totals}


def reset() -> None:
    global _LAST
    with _LOCK:
        for k in _TOTALS:
            _TOTALS[k] = 0
        _LAST = None
