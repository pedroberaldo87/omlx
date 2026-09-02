# SPDX-License-Identifier: Apache-2.0
"""A/B benchmark for n-gram lookup drafting (plan v3, F1.3).

Runs N identical chat completions against this server's own OpenAI endpoint
with the n-gram drafter ON, then N with it OFF, and reports mean and stddev
of tokens/second per arm. The drafter toggle is a process-global read on
every draft cycle, so arms flip live — no engine reload between them.

Single-flight and synchronous inside a worker thread; results poll by id.
The prompt is the self-similar code-rewrite workload the drafter targets,
sampled per the model card (the operator's standing rule: never greedy
by default in benchmarks).
"""

from __future__ import annotations

import json
import statistics
import threading
import time
import urllib.request
import uuid
from typing import Any, Optional

_RUNS: dict[str, dict] = {}
_LOCK = threading.Lock()

# The valid flips. "none" runs two identical arms (plan v5, F0.1/F0.3): the
# measured "gain" between them is pure noise, which anchors the decision
# threshold for every real flip. An unknown flip is an error, never a silent
# fallback to "enabled" — that ran a whole benchmark measuring the wrong knob.
_FLIPS = ("enabled", "none", "freq_rule", "match_len", "draft_max",
          "draft_min", "chain", "hysteresis", "patient", "margin",
          "block_verify")

_CODE = '''def process_orders(orders):
    total = 0
    for order in orders:
        if order.status == "paid":
            total += order.amount
        elif order.status == "refunded":
            total -= order.amount
        elif order.status == "pending":
            continue
    return total

def summarize(orders):
    total = process_orders(orders)
    return {"count": len(orders), "total": total}'''

_PROMPT = (
    "Here is a Python module:\n\n```python\n" + _CODE + "\n```\n\n"
    "Rewrite the whole module EXACTLY as-is, changing only the variable "
    "name `total` to `balance` everywhere. Output only the code block."
)


_NOVEL_TOPICS = [
    "a lighthouse keeper who collects tide sounds",
    "the last tram ride across a city being renamed",
    "a cartographer mapping a river that moves at night",
    "two beekeepers arguing about the color of winter",
    "an archivist who files smells instead of letters",
    "a bridge painter who never saw the far bank",
    "the night shift at a museum of unfinished machines",
    "a typesetter setting the alphabet of a dying language",
    "a ferry pilot crossing between two time zones daily",
    "an astronomer cataloguing clouds instead of stars",
    "a locksmith retiring on an island without doors",
    "the gardener of a rooftop no one can reach",
]


def _novel_prompt(seq: int) -> str:
    # a fresh topic per request keeps every arm on unseen content
    topic = _NOVEL_TOPICS[seq % len(_NOVEL_TOPICS)]
    return (
        f"Write a short original story (about 350 words) about {topic}. "
        f"Do not repeat sentences. Variation seed: {uuid.uuid4().hex[:8]}."
    )


def _one_request(port: int, api_key: str, model_id: str, max_tokens: int,
                 prompt: str = _PROMPT, sampling: Optional[dict] = None) -> dict:
    # Card sampling by default (the operator's standing rule: never greedy
    # unless the sweep declares it). v8 F4.5 lets a run declare its own
    # sampling so the temperature curve can be measured — the flip that
    # separates "cost of sampling" from "loss of draft acceptance".
    amostra = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    if sampling:
        amostra.update(sampling)
    body = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            **amostra,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        payload = json.loads(resp.read())
    wall = time.perf_counter() - t0
    usage = payload.get("usage") or {}
    tokens = int(usage.get("completion_tokens") or 0)
    gen_s = usage.get("generation_duration") or usage.get("total_time") or wall
    return {
        "tokens": tokens,
        "seconds": round(float(gen_s), 3),
        "tok_s": round(tokens / float(gen_s), 2) if gen_s else 0.0,
    }


def _last_request_stats() -> dict:
    """The just-finished request's spec stats (``last``), or {}."""
    try:
        from ..patches.mlx_lm_mtp.spec_stats import get_speculation_stats

        snap = get_speculation_stats()
        return dict((snap or {}).get("last") or {})
    except Exception:  # noqa: BLE001 — telemetry must never break the bench
        return {}


def _tele_add(tele: dict, last: dict) -> None:
    """Fold one request's stats into an arm's telemetry accumulator."""
    for k in ("cycles", "generation_tokens", "accepted_draft_tokens",
              "ngram_served_cycles", "ngram_miss_cycles",
              # v8 F2.1: the slots the verify paid for, and the ngram counters
              "drafted_slots", "ngram_drafted_slots",
              "ngram_hits", "ngram_lookup_misses", "ngram_drafted_tokens",
              "ngram_accepted_tokens", "ngram_frozen_keys"):
        try:
            tele[k] = tele.get(k, 0) + int(last.get(k) or 0)
        except (TypeError, ValueError):
            pass
    for k in ("depth_drafted", "depth_accepted"):
        vals = last.get(k) or []
        if not isinstance(vals, list):
            continue
        acc = tele.setdefault(k, [])
        if len(acc) < len(vals):
            acc.extend([0] * (len(vals) - len(acc)))
        for j, v in enumerate(vals):
            acc[j] += int(v)


def _arm_summary(samples: list[dict]) -> dict:
    rates = [s["tok_s"] for s in samples]
    return {
        "runs": samples,
        "mean_tok_s": round(statistics.fmean(rates), 2) if rates else None,
        "stdev_tok_s": (
            round(statistics.stdev(rates), 2) if len(rates) > 1 else None
        ),
    }


def _worker(run: dict, port: int, api_key: str) -> None:
    from ..patches.mlx_lm_mtp import (
        get_ngram_spec_params,
        is_mtp_block_verify,
        is_mtp_hysteresis,
        is_ngram_spec_enabled,
        reset_ngram_pool,
        set_mtp_block_verify,
        set_mtp_hysteresis,
        set_ngram_spec,
    )

    flip = run.get("flip", "enabled")
    original = is_ngram_spec_enabled()
    original_hyst = is_mtp_hysteresis()
    original_block = is_mtp_block_verify()
    (original_match, original_max, original_min, original_freq,
     original_chain, original_patient, original_margin) = get_ngram_spec_params()
    if flip == "freq_rule":
        arms = (("freq_on", True), ("freq_off", False))
    elif flip == "match_len":
        # llama.cpp's ngram-mod lookup length (24) vs the v3 default (16)
        arms = (("match_24", 24), ("match_16", 16))
    elif flip == "draft_max":
        # llama.cpp's best run used n-max 112 (accept 0.73, mean len 83);
        # our route clamp is 64 — probe the next step up from 48
        arms = (("draft_64", 64), ("draft_48", 48))
    elif flip == "draft_min":
        # llama.cpp's all-or-nothing: chains shorter than n_min=24 draft
        # NOTHING, so wide verifies only run on high-confidence copies
        arms = (("min_24", 24), ("min_4", 4))
    elif flip == "chain":
        # v5 F1: chained walk stitching occurrences vs the block copy
        arms = (("chain_on", True), ("chain_off", False))
    elif flip == "hysteresis":
        # v5 F2: acceptance-ladder depth vs the measured controller
        arms = (("hyst_on", True), ("hyst_off", False))
    elif flip == "block_verify":
        # bloco (arXiv 2403.10444) vs a aceitacao token a token
        arms = (("block_on", True), ("block_off", False))
    elif flip == "patient":
        # v5 F3: patient index reset vs the per-key freeze
        arms = (("patient_on", True), ("patient_off", False))
    elif flip == "margin":
        # v5 F4: ambiguous-entry gate vs serving every match
        arms = (("margin_on", True), ("margin_off", False))
    elif flip == "none":
        # two identical arms on the untouched config: the "gain" between
        # them is the noise floor (plan v5, F0.3)
        arms = (("null_a", None), ("null_b", None))
    else:
        arms = (("ngram_on", True), ("ngram_off", False))

    def _apply(enabled) -> None:
        # arm isolation (plan v5, F0.5): the shared pool never carries
        # history — or copies — from one arm into the other
        reset_ngram_pool()
        if flip == "none":
            return  # both arms run the production config untouched
        if flip == "freq_rule":
            # drafter stays ON; only the copy-length rule flips
            set_ngram_spec(True, freq_rule=enabled)
        elif flip == "match_len":
            # drafter stays ON; only the lookup length flips
            set_ngram_spec(True, match_len=enabled)
        elif flip == "draft_max":
            set_ngram_spec(True, draft_max=enabled)
        elif flip == "draft_min":
            set_ngram_spec(True, draft_min=enabled)
        elif flip == "chain":
            set_ngram_spec(True, chain=enabled)
        elif flip == "hysteresis":
            # drafter config untouched; only the depth-controller mode flips
            set_mtp_hysteresis(enabled)
        elif flip == "block_verify":
            # so a regra de aceitacao da janela vira; o resto fica igual
            set_mtp_block_verify(enabled)
        elif flip == "patient":
            set_ngram_spec(True, patient=enabled)
        elif flip == "margin":
            set_ngram_spec(True, margin=enabled)
        else:
            set_ngram_spec(enabled)

    novel = run.get("workload") == "novel"
    seq = 0
    samples: dict[str, list[dict]] = {arm: [] for arm, _ in arms}
    tele: dict[str, dict] = {arm: {} for arm, _ in arms}
    try:
        # warmup: one request outside the tally so both arms start hot
        _one_request(port, api_key, run["model_id"], run["max_tokens"],
                     sampling=run.get("sampling"))
        # interleaved A,B,A,B (plan v5, F0.1): thermal/memory drift lands on
        # both arms instead of accumulating on whichever ran second
        for i in range(run["repeats"]):
            pair: dict[str, Any] = {}
            for arm, enabled in arms:
                _apply(enabled)
                prompt = _novel_prompt(seq) if novel else _PROMPT
                seq += 1
                sample = _one_request(port, api_key, run["model_id"],
                                      run["max_tokens"], prompt,
                                      sampling=run.get("sampling"))
                samples[arm].append(sample)
                # v5 F1.4/F2.2: the run is single-flight on an otherwise
                # idle server, so the model's "last" request stats ARE this
                # request's — fold them into this arm's telemetry
                _tele_add(tele[arm], _last_request_stats())
                run["sequence"].append(arm)
                run["progress"] = f"pair {i + 1}/{run['repeats']} · {arm}"
                pair[arm] = sample["tok_s"]
            a_rate, b_rate = pair[arms[0][0]], pair[arms[1][0]]
            pair["pair_gain_pct"] = (
                round((a_rate / b_rate - 1) * 100, 1) if b_rate else None
            )
            run["results"].setdefault("pairs", []).append(pair)
        for arm, _ in arms:
            run["results"][arm] = _arm_summary(samples[arm])
            run["results"][arm]["telemetry"] = tele[arm]
        on = run["results"][arms[0][0]]["mean_tok_s"]
        off = run["results"][arms[1][0]]["mean_tok_s"]
        if on and off:
            run["results"]["gain_pct"] = round((on / off - 1) * 100, 1)
        run["status"] = "done"
    except Exception as exc:  # noqa: BLE001 — surfaced to the poller
        run["status"] = "error"
        run["error"] = str(exc)
    finally:
        set_ngram_spec(original, match_len=original_match,
                       draft_max=original_max, draft_min=original_min,
                       freq_rule=original_freq, chain=original_chain,
                       patient=original_patient, margin=original_margin)
        set_mtp_hysteresis(original_hyst)
        set_mtp_block_verify(original_block)


def start(model_id: str, port: int, api_key: str, repeats: int = 5,
          max_tokens: int = 400, flip: str = "enabled",
          workload: str = "rewrite",
          sampling: Optional[dict] = None) -> dict:
    if flip not in _FLIPS:
        raise ValueError(
            f"unknown flip {flip!r}; valid: {', '.join(_FLIPS)}"
        )
    with _LOCK:
        active = next(
            (r for r in _RUNS.values() if r["status"] == "running"), None
        )
        if active is not None:
            return {"error": "spec-ab run already active", "id": active["id"]}
        run = {
            "id": uuid.uuid4().hex[:12],
            "model_id": model_id,
            "repeats": max(2, min(int(repeats), 20)),
            "max_tokens": max(64, min(int(max_tokens), 2048)),
            "flip": flip,
            "workload": workload if workload in ("rewrite", "novel") else "rewrite",
            "status": "running",
            "progress": "warmup",
            "sampling": sampling or None,
            "sequence": [],
            "results": {},
            "started_ts": time.time(),
        }
        _RUNS[run["id"]] = run
    threading.Thread(
        target=_worker, args=(run, port, api_key), daemon=True
    ).start()
    return {"id": run["id"], "status": "running"}


def get(run_id: str) -> Optional[dict]:
    return _RUNS.get(run_id)
