# SPDX-License-Identifier: Apache-2.0
"""Per-model speculation accumulator (plan v3, F3.2)."""

from types import SimpleNamespace

from omlx.patches.mlx_lm_mtp import spec_stats


def _stats(accepts, cycles, ngram=0, miss=0):
    return SimpleNamespace(
        init_emits=2, draft_emits=accepts, bonus_emits=0, verify_emits=cycles,
        accepts=accepts, cycles=cycles, depth_drafted=[cycles],
        ngram_cycles=ngram, ngram_misses=miss,
    )


def test_two_models_keep_separate_totals():
    spec_stats.reset()
    model_a, model_b = object(), object()
    ka = spec_stats.model_identity_keys(model_a)
    kb = spec_stats.model_identity_keys(model_b)
    spec_stats.record(_stats(accepts=10, cycles=5), "stop", keys=ka)
    spec_stats.record(_stats(accepts=2, cycles=8), "stop", keys=kb)

    a = spec_stats.get_speculation_stats(ka)
    b = spec_stats.get_speculation_stats(kb)
    assert a["totals"]["accepted_draft_tokens"] == 10
    assert b["totals"]["accepted_draft_tokens"] == 2
    assert a["totals"]["requests"] == 1 and b["totals"]["requests"] == 1

    # sem candidatos: agrega tudo (comportamento v2)
    both = spec_stats.get_speculation_stats()
    assert both["totals"]["accepted_draft_tokens"] == 12
    spec_stats.reset()


def test_wrapper_and_inner_model_share_an_entry():
    spec_stats.reset()
    inner = object()
    wrapper = SimpleNamespace(language_model=inner)
    spec_stats.record(
        _stats(accepts=4, cycles=3), "stop",
        keys=spec_stats.model_identity_keys(wrapper),
    )
    # o engine pode enxergar so o objeto interno e ainda achar a entrada
    got = spec_stats.get_speculation_stats({id(inner)})
    assert got is not None and got["totals"]["accepted_draft_tokens"] == 4
    spec_stats.reset()


def test_drafted_slots_e_denominador_do_aceite():
    # v8 F2.1: depth_drafted para na 1a rejeicao; drafted_slots conta TODAS as
    # posicoes que o verify pagou — vermelho se a taxa usar o denominador velho
    from types import SimpleNamespace

    from omlx.patches.mlx_lm_mtp import spec_stats

    spec_stats.reset()
    stats = SimpleNamespace(
        init_emits=0, draft_emits=1, bonus_emits=1, verify_emits=0,
        accepts=1, cycles=1, rejects=1,
        depth_drafted=[1], depth_accepted=[1],   # parou na 1a rejeicao
        drafted_slots=16, ngram_drafted_slots=16,  # o verify pagou 16
        ngram_cycles=1, ngram_misses=0, model_keys=None,
        ngram_src_counters={"hits": 3, "misses": 1, "drafted_tokens": 40,
                            "accepted_tokens": 9, "frozen_keys": 2},
    )
    spec_stats.record(stats, "length")
    snap = spec_stats.get_speculation_stats()
    last = snap["last"]

    assert last["drafted_slots"] == 16
    assert last["draft_accept_rate"] == 1 / 16       # honesto: 6,25%
    assert last["accept_depth_ratio"] == 1 / 1       # o antigo, sob outro nome
    assert last["ngram_frozen_keys"] == 2            # contador que ninguem lia
    assert last["partial"] is False
    assert snap["totals"]["drafted_slots"] == 16
    spec_stats.reset()


def test_last_de_sequencia_estacionada_vem_marcado_parcial():
    from types import SimpleNamespace

    from omlx.patches.mlx_lm_mtp import spec_stats

    spec_stats.reset()
    stats = SimpleNamespace(
        init_emits=0, draft_emits=0, bonus_emits=0, verify_emits=1,
        accepts=0, cycles=1, rejects=1, depth_drafted=[1], depth_accepted=[0],
        drafted_slots=1, ngram_drafted_slots=0,
        ngram_cycles=0, ngram_misses=1, model_keys=None,
    )
    spec_stats.record(stats, "parked-at-depth-0")
    assert spec_stats.get_speculation_stats()["last"]["partial"] is True
    spec_stats.reset()
