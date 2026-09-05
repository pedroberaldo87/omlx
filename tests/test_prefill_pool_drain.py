# SPDX-License-Identifier: Apache-2.0
"""O pool de buffers do MLX é devolvido ao sistema no meio de um preparo longo.

Medido em 05/09 no GLM-5.3-Flash oQ2e, um prompt de 229k tokens: o pool cresceu de 2,71 para
5,03 GB ao longo do prompt (o KV, de 0,15 para 2,53), o processo encostou no alvo de 112,48 GB
perto dos 181k e o estrangulador pausou duas vezes e encolheu os pedaços. A limpeza periódica
conta passos do agendador, e um preparo inteiro é um passo só — por isso nunca disparava.
"""
from types import SimpleNamespace

import mlx.core as mx

from omlx.scheduler import Scheduler


def _stub():
    return SimpleNamespace(_PREFILL_POOL_DRAIN_BYTES=Scheduler._PREFILL_POOL_DRAIN_BYTES)


def test_drena_acima_do_limiar(monkeypatch):
    chamadas = []
    monkeypatch.setattr(mx, "get_cache_memory", lambda: 5 * 1024**3)
    monkeypatch.setattr(mx, "clear_cache", lambda: chamadas.append("clear"))
    monkeypatch.delenv("OMLX_PREFILL_POOL_DRAIN", raising=False)
    Scheduler._drain_prefill_pool_if_bloated(_stub(), "r", 181_248)
    assert chamadas == ["clear"]


def test_nao_drena_o_pool_quente_normal(monkeypatch):
    chamadas = []
    monkeypatch.setattr(mx, "get_cache_memory", lambda: int(2.7 * 1024**3))
    monkeypatch.setattr(mx, "clear_cache", lambda: chamadas.append("clear"))
    monkeypatch.delenv("OMLX_PREFILL_POOL_DRAIN", raising=False)
    Scheduler._drain_prefill_pool_if_bloated(_stub(), "r", 1024)
    assert chamadas == []


def test_o_ambiente_desliga(monkeypatch):
    chamadas = []
    monkeypatch.setattr(mx, "get_cache_memory", lambda: 9 * 1024**3)
    monkeypatch.setattr(mx, "clear_cache", lambda: chamadas.append("clear"))
    monkeypatch.setenv("OMLX_PREFILL_POOL_DRAIN", "0")
    Scheduler._drain_prefill_pool_if_bloated(_stub(), "r", 1024)
    assert chamadas == []
