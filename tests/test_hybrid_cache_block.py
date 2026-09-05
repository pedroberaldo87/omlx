# SPDX-License-Identifier: Apache-2.0
"""O bloco do cache do modelo híbrido como ajuste do painel.

Hoje só entra por OMLX_ARRAYS_CACHE_BLOCK na subida, e o app da barra não passa variável de
ambiente. Medido em 05/09 no GLM-5.3-Flash oQ2e a 110k tokens: 1024 deu 181 tok/s (94 pedaços
inteiros, zero cortes) contra 163 com 512. O padrão 0 preserva o comportamento de hoje.
"""
import os
from types import SimpleNamespace

from omlx.scheduler import Scheduler, SchedulerConfig
from omlx.settings import SchedulerSettings


def test_padrao_zero_e_ida_e_volta():
    assert SchedulerSettings().hybrid_cache_block == 0
    assert SchedulerSettings.from_dict({}).hybrid_cache_block == 0
    assert SchedulerSettings.from_dict({"hybrid_cache_block": 1024}).hybrid_cache_block == 1024
    assert SchedulerConfig().hybrid_cache_block == 0


def _agendador(bloco, monkeypatch):
    monkeypatch.delenv("OMLX_ARRAYS_CACHE_BLOCK", raising=False)
    from mlx_lm.models.cache import ArraysCache

    modelo = SimpleNamespace(make_cache=lambda: [ArraysCache(size=2)])
    ns = SimpleNamespace(
        config=SchedulerConfig(paged_ssd_cache_dir="/tmp/x", paged_cache_block_size=256,
                               prefill_step_size=512, hybrid_cache_block=bloco),
        model=modelo,
        _detect_rotating_window_sizes=lambda: [],
        _cache_tree_has_arrays_cache=lambda c: isinstance(c, ArraysCache),
        _ARRAYS_CACHE_BLOCK_SIZE=Scheduler._ARRAYS_CACHE_BLOCK_SIZE,
        _qwen35_prefill_floor=0,
        _model_type_name=lambda: "glm5_next",
        _GLM5_NEXT_BLOCK_SIZE=Scheduler._GLM5_NEXT_BLOCK_SIZE,
    )
    ns._enlarge_block_size_for_arrays_cache = Scheduler._enlarge_block_size_for_arrays_cache.__get__(ns, Scheduler)
    ns._enlarge_block_size_for_arrays_cache()
    return ns.config


def test_o_ajuste_manda_no_bloco_e_no_passo(monkeypatch):
    cfg = _agendador(1024, monkeypatch)
    assert cfg.paged_cache_block_size == 1024
    assert cfg.prefill_step_size == 1024


def test_zero_mantem_o_512_do_glm5(monkeypatch):
    cfg = _agendador(0, monkeypatch)
    assert cfg.paged_cache_block_size == Scheduler._GLM5_NEXT_BLOCK_SIZE == 512
    assert cfg.prefill_step_size == 512


def test_a_variavel_de_ambiente_ainda_vence(monkeypatch):
    monkeypatch.setenv("OMLX_ARRAYS_CACHE_BLOCK", "2048")
    from mlx_lm.models.cache import ArraysCache

    ns = SimpleNamespace(
        config=SchedulerConfig(paged_ssd_cache_dir="/tmp/x", paged_cache_block_size=256,
                               prefill_step_size=512, hybrid_cache_block=1024),
        model=SimpleNamespace(make_cache=lambda: [ArraysCache(size=2)]),
        _detect_rotating_window_sizes=lambda: [],
        _cache_tree_has_arrays_cache=lambda c: isinstance(c, ArraysCache),
        _ARRAYS_CACHE_BLOCK_SIZE=Scheduler._ARRAYS_CACHE_BLOCK_SIZE,
        _qwen35_prefill_floor=0,
        _model_type_name=lambda: "glm5_next",
        _GLM5_NEXT_BLOCK_SIZE=Scheduler._GLM5_NEXT_BLOCK_SIZE,
    )
    ns._enlarge_block_size_for_arrays_cache = Scheduler._enlarge_block_size_for_arrays_cache.__get__(ns, Scheduler)
    ns._enlarge_block_size_for_arrays_cache()
    assert ns.config.paged_cache_block_size == 2048 and ns.config.prefill_step_size == 2048
