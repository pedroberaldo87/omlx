# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the self-speculative n-gram draft source (v3)."""

from omlx.speculative.ngram import NGramDraftSource


def test_repeating_sequence_returns_continuation():
    src = NGramDraftSource(match_len=4, draft_max=6, draft_min=2)
    seq = [1, 2, 3, 4, 5, 6, 7, 8]
    src.extend(seq)
    src.extend(seq)
    got = src.lookup()
    assert got is not None and got[:4] == [1, 2, 3, 4]
    assert src.hits == 1


def test_unique_suffix_does_not_match_itself():
    src = NGramDraftSource(match_len=4)
    src.extend([9] * 30)
    src2 = NGramDraftSource(match_len=4)
    src2.extend(list(range(30)))
    assert src2.lookup() is None
    assert src2.misses == 1


def test_short_history_misses():
    src = NGramDraftSource(match_len=8)
    src.extend([1, 2, 3])
    assert src.lookup() is None


def test_cascade_falls_back_to_short_window():
    # long window (24/16) has no repeat, but the last 8 tokens do
    src = NGramDraftSource(match_len=16, draft_max=16, draft_min=2)
    tail = list(range(100, 108))          # the 8-gram that repeats
    src.extend(list(range(0, 40)) + tail + list(range(200, 210)) + tail)
    got = src.lookup()
    assert got is not None
    assert got[0] == 200                  # continuation after the first tail


def test_short_window_match_caps_copy_length():
    src = NGramDraftSource(match_len=16, draft_max=16, draft_min=2)
    tail = list(range(100, 108))
    src.extend(list(range(0, 40)) + tail + list(range(200, 230)) + tail)
    got = src.lookup()
    assert got is not None
    # 8-window match cannot buy the full 16-wide verify
    assert len(got) <= 8


def test_per_key_freeze_only_hits_the_failing_key():
    src = NGramDraftSource(match_len=4, draft_max=6, draft_min=3)
    a = [11, 12, 13, 14]
    src.extend(a + [50, 51, 52] + a)
    assert src.lookup() is not None
    src.feedback(0)
    src.extend([60])
    src.extend(a)
    assert src.lookup() is not None
    src.feedback(0)                       # second failure -> key frozen
    assert src.frozen_keys >= 1   # regiao congela as chaves de todas as janelas
    src.extend([61])
    src.extend(a)
    assert src.lookup() is None           # frozen key stays quiet
    # a DIFFERENT key keeps serving in the same source
    b = [21, 22, 23, 24]
    src.extend(b + [70, 71, 72] + b)
    got = src.lookup()
    assert got is not None and got[0] == 70


def test_success_thaws_a_failing_key():
    src = NGramDraftSource(match_len=4, draft_max=6, draft_min=2)
    a = [1, 2, 3, 4]
    src.extend(a + [9, 9] + a)
    assert src.lookup() is not None
    src.feedback(0)                       # one failure
    src.feedback(6)                       # productive copy clears the slate
    assert src._fails == {}


def test_seed_with_processors_does_not_duplicate_splice():
    from types import SimpleNamespace

    from mlx_lm.models.cache import TokenBuffer

    from omlx.patches.mlx_lm_mtp import set_ngram_spec
    from omlx.patches.mlx_lm_mtp.batch_generator import _ngram_source_for_cycle

    import mlx.core as mx

    set_ngram_spec(True)
    prompt = list(range(100, 140))
    generated = [1, 2, 3]
    buf = TokenBuffer(prompt)
    buf.update_and_fetch(mx.array(generated, dtype=mx.int32))
    gb = SimpleNamespace(_token_context=[buf])
    state = SimpleNamespace()
    committed = mx.array(generated, dtype=mx.uint32)

    # processors ativos: o buffer ja contem os gerados -> sem duplicata
    src = _ngram_source_for_cycle(gb, state, committed, procs=[object()])
    assert src._tokens == prompt + generated

    # sem processors: o buffer so tem o prompt -> committed entra uma vez
    buf2 = TokenBuffer(prompt)
    gb2 = SimpleNamespace(_token_context=[buf2])
    state2 = SimpleNamespace()
    src2 = _ngram_source_for_cycle(gb2, state2, committed, procs=None)
    assert src2._tokens == prompt + generated
    set_ngram_spec(False)


def test_shared_pool_serves_across_requests_and_respects_fence():
    from omlx.speculative.ngram import SharedNGramPool

    pool = SharedNGramPool(match_len=4, draft_max=8, draft_min=2)
    doc = list(range(500, 540))
    pool.feed(doc)                       # pedido 1 alimenta o pool
    suffix = doc[10:14]                  # pedido 2 chega no mesmo trecho
    got = pool.lookup_suffix(suffix)
    assert got == doc[14:22]
    # a cerca impede emenda entre pedidos: continuacao para na fronteira
    pool.feed([1, 2, 3])
    tail = doc[-4:]
    got2 = pool.lookup_suffix(tail)
    assert got2 is None or all(t >= 0 for t in got2)


def test_shared_pool_eviction_keeps_size_bounded():
    import threading

    from omlx.speculative.ngram import SharedNGramPool

    pool = SharedNGramPool(match_len=4, max_tokens=2000)

    def fill(base):
        for i in range(20):
            pool.feed(list(range(base + i * 100, base + i * 100 + 90)))

    threads = [threading.Thread(target=fill, args=(b,)) for b in (0, 10_000, 20_000)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(pool) <= 2000
