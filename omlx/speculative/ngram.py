# SPDX-License-Identifier: Apache-2.0
"""Self-speculative n-gram draft source (prompt/history lookup).

Drafts are produced by string matching over the request's own token
history — the prompt plus everything generated so far — with no model
forward involved. When the last ``match_len`` tokens have appeared
before, the tokens that followed that earlier occurrence are proposed
as the draft and verified by the target model in one wide forward.
Verification makes the scheme lossless regardless of draft quality
(a bad draft costs speed, never correctness).

This is the same family as llama.cpp's ``ngram-simple`` / ``ngram-mod``
(LLMA, arXiv:2304.04487; Prompt Lookup Decoding). The index is
request-scoped: it dies with the sequence and needs no locking.

Not to be confused with Qwen4-Exp's PLE "n-gram embedding", which is a
trained architectural table inside the model. This module never touches
model weights.
"""

from __future__ import annotations

from typing import List, Optional


class NGramDraftSource:
    """Hash-indexed lookup of the last ``match_len`` tokens in history.

    ``extend`` appends committed tokens and registers, for every new
    position, the ``match_len``-gram that ends there. ``lookup`` takes
    the current suffix and returns the continuation that followed its
    most recent *earlier* occurrence, or ``None`` when there is no
    occurrence or the continuation is shorter than ``draft_min``.
    """

    def __init__(
        self,
        match_len: int = 16,
        draft_max: int = 16,
        draft_min: int = 4,
    ):
        if match_len < 2:
            raise ValueError("match_len must be >= 2")
        if draft_min < 1 or draft_max < draft_min:
            raise ValueError("need 1 <= draft_min <= draft_max")
        self.n = int(match_len)
        self.draft_max = int(draft_max)
        self.draft_min = int(draft_min)
        self._tokens: List[int] = []
        # n-gram (ending at position p) -> p, plus the previous holder of
        # the same key so the suffix's own registration never shadows the
        # earlier occurrence the lookup needs.
        self._last: dict = {}
        self._prev: dict = {}
        # counters for the stats line / admin telemetry
        self.hits = 0
        self.misses = 0
        self.drafted_tokens = 0
        self.accepted_tokens = 0
        # Rejection cooloff: a wide verify that rejects at position 0 costs
        # ~2x a plain MTP cycle, so a source whose copies keep missing must
        # stop volunteering for a while instead of taxing every cycle.
        self._cooloff = 0
        self._fail_streak = 0

    def __len__(self) -> int:
        return len(self._tokens)

    def extend(self, tokens) -> None:
        toks = self._tokens
        n = self.n
        for t in tokens:
            toks.append(int(t))
            p = len(toks) - 1
            if p >= n - 1:
                key = tuple(toks[p - n + 1 : p + 1])
                old = self._last.get(key)
                if old is not None:
                    self._prev[key] = old
                self._last[key] = p

    def lookup(self) -> Optional[List[int]]:
        toks = self._tokens
        n = self.n
        if len(toks) < n:
            self.misses += 1
            return None
        end = len(toks) - 1
        key = tuple(toks[end - n + 1 :])
        p = self._last.get(key)
        if p == end:
            p = self._prev.get(key)
        if p is None:
            self.misses += 1
            return None
        cont = toks[p + 1 : p + 1 + self.draft_max]
        if len(cont) < self.draft_min:
            self.misses += 1
            return None
        if self._cooloff > 0:
            self._cooloff -= 1
            self.misses += 1
            return None
        self.hits += 1
        self.drafted_tokens += len(cont)
        return cont

    def feedback(self, accepted: int) -> None:
        """Report how many tokens of the last copied draft were accepted.

        Two consecutive drafts dying below ``draft_min`` put the source in
        an 8-lookup cooloff; one productive draft clears the streak.
        """
        self.accepted_tokens += int(accepted)
        if accepted < self.draft_min:
            self._fail_streak += 1
            if self._fail_streak >= 2:
                self._cooloff = 8
                self._fail_streak = 0
        else:
            self._fail_streak = 0
