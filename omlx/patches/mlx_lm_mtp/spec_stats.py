# SPDX-License-Identifier: Apache-2.0
"""Process-global session accumulator for MTP/n-gram speculation stats.

``_MtpStats`` used to die inside ``_log_mtp_stats``; this folds every
finished sequence into session totals in the ``{"last":..., "totals":...}``
shape the admin bridge already duck-types for DFlash.

v3 (plan F3.2): totals are kept PER MODEL. The recording side (the mlx-lm
patch) only has the model object, and the reading side (the engine) may
hold the same object or its wrapper — so each entry is keyed by the SET of
object identities that can name the model (the object itself plus its
language_model/inner attributes), and the engine reads by intersection
with its own candidates. Reading with no candidates aggregates everything,
preserving the v2 behavior.
"""

from __future__ import annotations

import threading
from typing import Any, Iterable, Optional

_LOCK = threading.Lock()
_ENTRIES: list[dict] = []  # each: {"keys": set[int], "totals": dict, "last": dict|None}

_TOTAL_FIELDS = (
    "requests",
    "speculative_requests",
    "fallback_requests",
    "generation_tokens",
    "accepted_draft_tokens",
    "cycles",
    "ngram_served_cycles",
    "ngram_miss_cycles",
    # slots the wide verify actually paid for (plan v8, F2.1) — the honest
    # denominator of the acceptance rate, comparable with llama.cpp's
    "drafted_slots",
    "ngram_drafted_slots",
)


def model_identity_keys(model: Any) -> set:
    """Every object identity that can name this model across layers."""
    keys = {id(model)}
    for attr in ("language_model", "_language_model", "model"):
        inner = getattr(model, attr, None)
        if inner is not None:
            keys.add(id(inner))
    return keys


def _new_totals() -> dict:
    return {f: 0 for f in _TOTAL_FIELDS}


def record(stats: Any, finish_reason: str, keys: Optional[set] = None) -> None:
    """Fold one finished sequence's ``_MtpStats`` into its model's totals."""
    total_emits = (
        stats.init_emits + stats.draft_emits + stats.bonus_emits + stats.verify_emits
    )
    total_drafted = sum(stats.depth_drafted) or stats.cycles
    slots = int(getattr(stats, "drafted_slots", 0) or 0)
    cycles = stats.cycles
    last = {
        "finish_reason": finish_reason,
        "generation_tokens": total_emits,
        "accepted_draft_tokens": stats.accepts,
        "cycles": cycles,
        "acceptance_ratio": (stats.accepts / total_emits) if total_emits else None,
        "tokens_per_cycle": (total_emits / cycles) if cycles else None,
        "accepted_draft_tokens_per_cycle": (stats.accepts / cycles) if cycles else None,
        # v8 F2.1: over every slot the verify paid for — comparable with the
        # 493/672 of llama.cpp. Falls back to the old denominator only when
        # the counter is absent (stubbed stats in tests).
        "draft_accept_rate": (
            stats.accepts / slots if slots
            else (stats.accepts / total_drafted if total_drafted else None)
        ),
        # the old ratio, kept under its honest name: how deep a cycle got
        # before the first rejection, never an acceptance rate
        "accept_depth_ratio": (
            stats.accepts / total_drafted if total_drafted else None
        ),
        "drafted_slots": slots,
        "ngram_drafted_slots": int(getattr(stats, "ngram_drafted_slots", 0) or 0),
        "ngram_served_cycles": stats.ngram_cycles,
        "ngram_miss_cycles": stats.ngram_misses,
        # per-position draft/accept counts (plan v5, F2.2): the A/B preset
        # diffs these per request to attribute depth telemetry per arm
        "depth_drafted": list(getattr(stats, "depth_drafted", []) or []),
        "depth_accepted": list(getattr(stats, "depth_accepted", []) or []),
        # v8 F2.1: a parked sequence hands over only its last stretch — the
        # A/B reads this dict, and reading a partial one as a whole request
        # silently under-reports every counter in it.
        "partial": finish_reason == "parked-at-depth-0",
        "fallback_ar": False,
    }
    # v8 F2.1: the copy drafter writes five counters that production never
    # read (grep outside ngram.py only hits tests). Surface them when the
    # sequence carried one.
    src = getattr(stats, "ngram_src_counters", None)
    if isinstance(src, dict):
        last.update({
            "ngram_hits": src.get("hits", 0),
            "ngram_lookup_misses": src.get("misses", 0),
            "ngram_drafted_tokens": src.get("drafted_tokens", 0),
            "ngram_accepted_tokens": src.get("accepted_tokens", 0),
            "ngram_frozen_keys": src.get("frozen_keys", 0),
        })
    keys = set(keys or ())
    with _LOCK:
        entry = next(
            (e for e in _ENTRIES if e["keys"] & keys), None
        ) if keys else (_ENTRIES[0] if _ENTRIES else None)
        if entry is None:
            entry = {"keys": set(keys), "totals": _new_totals(), "last": None}
            _ENTRIES.append(entry)
        else:
            entry["keys"] |= keys
        t = entry["totals"]
        entry["last"] = last
        t["requests"] += 1
        if stats.accepts > 0:
            t["speculative_requests"] += 1
        else:
            t["fallback_requests"] += 1
        t["generation_tokens"] += total_emits
        t["accepted_draft_tokens"] += stats.accepts
        t["cycles"] += cycles
        t["ngram_served_cycles"] += stats.ngram_cycles
        t["ngram_miss_cycles"] += stats.ngram_misses
        t["drafted_slots"] += slots
        t["ngram_drafted_slots"] += int(getattr(stats, "ngram_drafted_slots", 0) or 0)


def get_speculation_stats(candidates: Optional[Iterable] = None) -> Optional[dict]:
    """DFlash-shaped snapshot; None before any matching request finished.

    ``candidates`` narrows to one model's entries by object-identity
    intersection; None aggregates every entry (v2 behavior).
    """
    cand = set(candidates or ())
    with _LOCK:
        entries = [
            e for e in _ENTRIES if not cand or (e["keys"] & cand)
        ]
        if not entries or all(e["totals"]["requests"] == 0 for e in entries):
            return None
        totals = _new_totals()
        for e in entries:
            for f in _TOTAL_FIELDS:
                totals[f] += e["totals"][f]
        last = entries[-1]["last"] and dict(entries[-1]["last"])
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
    with _LOCK:
        _ENTRIES.clear()
