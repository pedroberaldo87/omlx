# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the self-speculative n-gram draft source."""

from omlx.speculative.ngram import NGramDraftSource


def test_repeating_sequence_returns_continuation():
    src = NGramDraftSource(match_len=4, draft_max=6, draft_min=2)
    seq = [1, 2, 3, 4, 5, 6, 7, 8]
    src.extend(seq)
    src.extend(seq)
    # suffix (5,6,7,8) first occurred at the end of the first pass;
    # what followed it there was the second pass's opening tokens
    assert src.lookup() == [1, 2, 3, 4, 5, 6]
    assert src.hits == 1


def test_unique_suffix_does_not_match_itself():
    src = NGramDraftSource(match_len=4)
    src.extend([9, 9, 9, 9])
    assert src.lookup() is None
    assert src.misses == 1


def test_short_history_misses():
    src = NGramDraftSource(match_len=8)
    src.extend([1, 2, 3])
    assert src.lookup() is None


def test_continuation_shorter_than_min_misses():
    src = NGramDraftSource(match_len=3, draft_max=8, draft_min=6)
    # suffix (1,2,3) repeats; the earlier occurrence is followed by only
    # 5 tokens of history (7,8,1,2,3), below the draft_min of 6
    src.extend([1, 2, 3, 7, 8, 1, 2, 3])
    assert src.lookup() is None


def test_draft_max_truncates():
    src = NGramDraftSource(match_len=3, draft_max=4, draft_min=1)
    src.extend([1, 2, 3] + list(range(10, 20)) + [1, 2, 3])
    assert src.lookup() == [10, 11, 12, 13]


def test_most_recent_earlier_occurrence_wins():
    src = NGramDraftSource(match_len=3, draft_max=3, draft_min=1)
    # (1,2,3) appears three times with different continuations; the copy
    # must come from the most recent occurrence BEFORE the suffix itself
    src.extend([1, 2, 3, 50, 0, 1, 2, 3, 60, 0, 1, 2, 3])
    assert src.lookup()[0] == 60


def test_two_failed_drafts_trigger_cooloff():
    src = NGramDraftSource(match_len=3, draft_max=6, draft_min=4)
    src.extend([1, 2, 3] + list(range(10, 16)) + [1, 2, 3])
    assert src.lookup() is not None
    src.feedback(0)
    src.extend([9])
    src.extend([1, 2, 3])
    assert src.lookup() is not None
    src.feedback(1)  # segunda falha seguida -> cooloff
    src.extend([8])
    src.extend([1, 2, 3])
    assert src.lookup() is None  # em cooloff, nao se oferece
    # um acerto produtivo depois do cooloff zera a sequencia de falhas
    src.feedback(6)
    assert src._fail_streak == 0
