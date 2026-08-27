# SPDX-License-Identifier: Apache-2.0
"""Self-speculative n-gram draft source (prompt/history lookup).

Drafts are produced by string matching over the request's own token
history — the prompt plus everything generated so far — with no model
forward involved. When a recent suffix has appeared before, the tokens
that followed that earlier occurrence are proposed as the draft and
verified by the target model in one wide forward. Verification makes
the scheme lossless regardless of draft quality.

v3 upgrades (plan 2026-08-27-ngram-v3-aprimoramentos):
- cascading windows: the lookup tries a long suffix first (high
  precision), then shorter ones (recall); the copy length is capped by
  the window that matched, so weak 8-token matches cannot request the
  widest — and most expensive — verify.
- per-key freeze: a key whose copies died twice below draft_min stops
  volunteering, while productive keys keep serving. Replaces the v2
  global cooloff, which silenced the whole source over one bad region.

v4 upgrade (plan 2026-08-27-ngram-v4-frequencia-e-sistema): frequency
rule. ``extend`` counts how many times each gram occurred; with
``freq_rule=True`` the copy cap follows that count — a suffix repeated
3+ times buys the full draft_max even through a short window, one seen
a single time is halved. SuffixDecoding's per-step scoring, without the
suffix tree. Off (the default) reproduces v3 byte for byte; the freeze
memory stays a separate mechanism (a brake, never a length signal).

Not to be confused with Qwen4-Exp's PLE "n-gram embedding", a trained
table inside the model. This module never touches weights.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# llama.cpp's ngram-mod lookup length — the F1.3 candidate the A/B flips
# against the v3 default of 16. match_len=24 yields windows (36, 24, 12).
NGRAM_MOD_MATCH_LEN = 24


class NGramDraftSource:
    """Cascading hash lookup of recent suffixes in the token history.

    ``match_len`` seeds the cascade: windows are ``(3n/2, n, n/2)``
    deduped — the default 16 yields (24, 16, 8), llama.cpp's documented
    operating point for MoE models. ``extend`` registers, per window,
    the gram ending at each new position; ``lookup`` walks windows long
    to short and returns the continuation after the most recent earlier
    occurrence, or ``None``.
    """

    def __init__(
        self,
        match_len: int = 16,
        draft_max: int = 16,
        draft_min: int = 4,
        freq_rule: bool = False,
        chain: bool = False,
        patient: bool = False,
        margin: bool = False,
        chain_min: int = 16,
        chain_max: int = 64,
    ):
        if match_len < 2:
            raise ValueError("match_len must be >= 2")
        if draft_min < 1 or draft_max < draft_min:
            raise ValueError("need 1 <= draft_min <= draft_max")
        n = int(match_len)
        self.match_len = n
        self.windows: Tuple[int, ...] = tuple(
            sorted({n * 3 // 2, n, max(2, n // 2)}, reverse=True)
        )
        self.draft_max = int(draft_max)
        self.draft_min = int(draft_min)
        self.freq_rule = bool(freq_rule)
        # Chained walk (plan v5, F1): instead of one block copy per cycle,
        # the draft grows token by token, re-consulting the index with the
        # rolling match_len key and jumping to the most recent occurrence —
        # llama.cpp's ngram-mod walk, which stitches occurrences into
        # 70+-token drafts. All-or-nothing gate (F1.2): a chain shorter
        # than chain_min drafts NOTHING (SuffixDecoding's port into
        # llama.cpp lost 26% on normal prompts until it gained this gate).
        # chain_max=64 is the route clamp for draft width on this machine.
        self.chain = bool(chain)
        self.chain_min = int(chain_min)
        self.chain_max = int(chain_max)
        # Patient brake (plan v5, F3.1): behind the knob, the per-key freeze
        # is replaced by one counter of consecutive bad rounds — the index
        # resets whole on the 5th round with acceptance under 25%, instead
        # of silencing productive regions after 2 failures.
        self.patient = bool(patient)
        self._bad_rounds = 0
        self._last_len = 0
        # Candidate margin (plan v5, F4.1): per key, up to 4 next-tokens
        # with counts; without a dominant candidate at 2x the sum of the
        # rivals, the key does not start a chain that step.
        self.margin = bool(margin)
        self._next: Dict[int, Dict[tuple, Dict[int, int]]] = {
            n: {} for n in self.windows
        }
        self._tokens: List[int] = []
        # per window size: gram-ending-at-p -> p, plus the previous holder
        # so the suffix's own registration never shadows what lookup needs
        self._last: Dict[int, dict] = {n: {} for n in self.windows}
        self._prev: Dict[int, dict] = {n: {} for n in self.windows}
        # per window size: gram -> occurrence count over the whole history.
        # Rebuilt histories (the pool's trim) re-derive it by re-extending,
        # so the count always reflects the tokens actually held.
        self._count: Dict[int, Dict[tuple, int]] = {n: {} for n in self.windows}
        # per-key acceptance memory: key -> consecutive sub-min failures;
        # at _FREEZE_AT the key stops volunteering (thawed by one success)
        self._fails: Dict[tuple, int] = {}
        self._last_key: Optional[tuple] = None
        self._last_region: Tuple[tuple, ...] = ()
        # counters for the stats line / admin telemetry
        self.hits = 0
        self.misses = 0
        self.drafted_tokens = 0
        self.accepted_tokens = 0
        self.frozen_keys = 0

    _FREEZE_AT = 2
    # freq_rule: this many earlier occurrences of the suffix buy draft_max
    _STRONG_AT = 3
    # patient brake (v5 F3.1): reset the index on the Nth consecutive round
    # with acceptance under this share of the served draft
    _PATIENT_ROUNDS = 5
    _PATIENT_ACCEPT = 0.25

    def __len__(self) -> int:
        return len(self._tokens)

    def extend(self, tokens) -> None:
        toks = self._tokens
        for t in tokens:
            toks.append(int(t))
            p = len(toks) - 1
            for n in self.windows:
                if p >= n - 1:
                    key = tuple(toks[p - n + 1 : p + 1])
                    last = self._last[n]
                    old = last.get(key)
                    if old is not None:
                        self._prev[n][key] = old
                    last[key] = p
                    cnt = self._count[n]
                    cnt[key] = cnt.get(key, 0) + 1
                if self.margin and p >= n:
                    # v5 F4.1: register this token as a next-token candidate
                    # of the gram that precedes it, up to 4 distinct rivals.
                    # ponytail: a 5th distinct candidate is ignored (no
                    # eviction); such a key is ambiguous far before that.
                    key_prev = tuple(toks[p - n : p])
                    cand = self._next[n].setdefault(key_prev, {})
                    ti = toks[p]
                    if ti in cand:
                        cand[ti] += 1
                    elif len(cand) < 4:
                        cand[ti] = 1

    def _copy_cap(self, window: int, reps: Optional[int] = None) -> int:
        # A short-window match is weaker evidence: cap its copy at the
        # window size so it cannot buy the widest verify forward.
        if window >= self.windows[len(self.windows) // 2]:
            base = self.draft_max
        else:
            base = min(self.draft_max, max(self.draft_min, window))
        if not self.freq_rule or reps is None:
            return base
        # Frequency rule (v4): boost-only. Strong repetition buys the full
        # draft even through a short window; weak evidence keeps the v3 cap.
        # The first cut also HALVED single-sighting copies and lost 3.4% on
        # the rewrite A/B (freq_on 25.84 vs freq_off 26.74) — never punish.
        if reps >= self._STRONG_AT:
            return self.draft_max
        return base

    def _reps(self, window: int, key: tuple) -> int:
        # Earlier occurrences of the gram, excluding the registration of
        # the suffix now being looked up (extend already counted it).
        return max(0, self._count[window].get(key, 0) - 1)

    def _ambiguous(self, window: int, key: tuple) -> bool:
        # v5 F4.1: a key without a dominant next-token (2x the sum of the
        # rivals) buys an expensive verify with no evidence — skip it.
        cand = self._next[window].get(key)
        if not cand or len(cand) < 2:
            return False
        top = max(cand.values())
        return top < 2 * (sum(cand.values()) - top)

    def _reset_index(self) -> None:
        # v5 F3.1: the patient brake drops the WHOLE index — history, maps,
        # freezes — instead of silencing regions early; the history rebuilds
        # from the tokens that follow.
        self._tokens.clear()
        for n in self.windows:
            self._last[n].clear()
            self._prev[n].clear()
            self._count[n].clear()
            self._next[n].clear()
        self._fails.clear()
        self._last_key = None
        self._last_region = ()
        self._bad_rounds = 0

    def _walk(self, entry_p: int, fence: Optional[int] = None) -> List[int]:
        """Chained walk from an entry match (plan v5, F1.1).

        Copies the token after ``entry_p``, then re-consults the index with
        the rolling ``match_len`` key and jumps to its most recent earlier
        occurrence — stitching occurrences into one draft. Stops on
        divergence (the rolling key only exists at the live tail), on
        ``chain_max``, at the end of history, or at ``fence`` (the pool's
        request boundary — a chain never splices two requests).
        Only the primary window walks; the cascade stays an entry-only
        mechanism.
        """
        n = self.match_len
        toks = self._tokens
        draft: List[int] = []
        cur = entry_p
        while len(draft) < self.chain_max and cur + 1 < len(toks):
            nxt = toks[cur + 1]
            if fence is not None and nxt == fence:
                break
            draft.append(nxt)
            cur += 1
            if cur >= n - 1:
                key = tuple(toks[cur - n + 1 : cur + 1])
                if fence is not None and fence in key:
                    break  # key spans two requests: never stitch across
                q = self._last[n].get(key)
                if q is None or q == len(toks) - 1:
                    # divergence: the key's most recent occurrence is the
                    # live tail — following _prev here would jump BACKWARD
                    # and cycle the same span until chain_max. Stop.
                    break
                cur = q
        return draft

    def lookup(self) -> Optional[List[int]]:
        toks = self._tokens
        end = len(toks) - 1
        for n in self.windows:
            if len(toks) < n:
                continue
            key = tuple(toks[end - n + 1 :])
            if self._fails.get(key, 0) >= self._FREEZE_AT:
                continue
            if self.margin and self._ambiguous(n, key):
                continue
            p = self._last[n].get(key)
            if p == end:
                p = self._prev[n].get(key)
            if p is None:
                continue
            if self.chain:
                # F1.1/F1.2: walk the chain from the entry match; a chain
                # below the gate discards the WHOLE draft — no shorter-window
                # retry, no partial serve (all-or-nothing).
                cont = self._walk(p)
                if len(cont) < self.chain_min:
                    break
                self.hits += 1
                self.drafted_tokens += len(cont)
                self._last_key = key
                self._last_len = len(cont)
                # punish only the entry key (plan v5, F1.1): the walk is
                # index-guided, so the entry is the only decision to blame
                self._last_region = (key,)
                return cont
            cont = toks[p + 1 : p + 1 + self._copy_cap(n, self._reps(n, key))]
            if len(cont) < self.draft_min:
                continue
            self.hits += 1
            self.drafted_tokens += len(cont)
            self._last_key = key
            self._last_len = len(cont)
            # Freezing must silence the REGION, not just the serving key:
            # otherwise the cascade re-offers the same bad copy through a
            # shorter window. Remember every window's key for this suffix.
            self._last_region = tuple(
                tuple(toks[end - w + 1 :]) for w in self.windows if len(toks) >= w
            )
            return cont
        self.misses += 1
        self._last_key = None
        return None

    def feedback(self, accepted: int) -> None:
        """Report how many tokens of the last served copy were accepted."""
        self.accepted_tokens += int(accepted)
        if self._last_key is None:
            return
        if self.patient:
            # v5 F3.1: one counter of consecutive bad rounds replaces the
            # per-key freeze; the 5th round under 25% acceptance resets the
            # whole index. A single good round clears the counter.
            served = self._last_len
            if served > 0 and accepted < self._PATIENT_ACCEPT * served:
                self._bad_rounds += 1
                if self._bad_rounds >= self._PATIENT_ROUNDS:
                    self._reset_index()
            else:
                self._bad_rounds = 0
            return
        region = getattr(self, "_last_region", (self._last_key,))
        if accepted < self.draft_min:
            for key in region:
                fails = self._fails.get(key, 0) + 1
                self._fails[key] = fails
                if fails == self._FREEZE_AT:
                    self.frozen_keys += 1
        else:
            for key in region:
                if key in self._fails:
                    if self._fails[key] >= self._FREEZE_AT:
                        self.frozen_keys -= 1
                    del self._fails[key]


class SharedNGramPool:
    """Cross-request n-gram pool (plan v3, F4.1) — llama.cpp's ngram-mod idea.

    One lock-guarded history shared by every request of the process: an
    agent loop that re-sends the same file gets its copies from the first
    request onward. Feeds from different requests are separated by a
    negative FENCE token, which can never equal a real token id — keys
    spanning a fence never match a real suffix, and continuations are cut
    at the fence, so no copy ever splices two unrelated requests.
    """

    FENCE = -1

    def __init__(self, match_len: int = 16, draft_max: int = 16,
                 draft_min: int = 4, freq_rule: bool = False,
                 chain: bool = False, patient: bool = False,
                 margin: bool = False, chain_min: int = 16,
                 chain_max: int = 64, max_tokens: int = 65536):
        # ponytail: global lock + full rebuild on trim; per-shard locks and
        # incremental eviction only if contention ever shows up in profiles
        import threading

        self._lock = threading.Lock()
        self._args = (match_len, draft_max, draft_min, freq_rule,
                      chain, patient, margin, chain_min, chain_max)
        self._src = NGramDraftSource(*self._args)
        self.max_tokens = int(max_tokens)

    def feed(self, tokens, new_segment: bool = True) -> None:
        """Append tokens; ``new_segment`` fences off the previous request.

        A request feeds its prompt with the fence and its incremental
        commits without it, so its own text stays contiguous while text
        from different requests can never splice.
        """
        with self._lock:
            src = self._src
            if new_segment and len(src) > 0:
                src.extend([self.FENCE])
            src.extend(tokens)
            if len(src) > self.max_tokens:
                tail = src._tokens[-self.max_tokens // 2 :]
                fresh = NGramDraftSource(*self._args)
                fresh.extend(tail)
                self._src = fresh

    def lookup_suffix(self, suffix) -> "Optional[List[int]]":
        with self._lock:
            src = self._src
            toks = src._tokens
            for n in src.windows:
                if len(suffix) < n or len(toks) < n:
                    continue
                key = tuple(int(t) for t in suffix[-n:])
                if src.margin and src._ambiguous(n, key):
                    # v5 F4.1: the same entry gate as the per-request path —
                    # a rule living only in one of them measures old code
                    continue
                p = src._last[n].get(key)
                if p is None or p == len(toks) - 1:
                    p = src._prev[n].get(key)
                if p is None:
                    continue
                if src.chain:
                    # plan v5, F1.3: the pool walks the same chain as the
                    # per-request drafter, fenced so a stitch never splices
                    # two requests; sub-gate discards the whole draft.
                    cont = src._walk(p, fence=self.FENCE)
                    if len(cont) < src.chain_min:
                        src.misses += 1
                        return None
                    src.hits += 1
                    src.drafted_tokens += len(cont)
                    return cont
                # Same frequency rule as the per-request path (F1.5). The
                # querying request usually feeds the pool too, so its own
                # registration is excluded the same way.
                cont = toks[p + 1 : p + 1 + src._copy_cap(n, src._reps(n, key))]
                if self.FENCE in cont:
                    cont = cont[: cont.index(self.FENCE)]
                if len(cont) < src.draft_min:
                    continue
                src.hits += 1
                src.drafted_tokens += len(cont)
                return cont
            src.misses += 1
            return None

    def __len__(self) -> int:
        with self._lock:
            return len(self._src)
