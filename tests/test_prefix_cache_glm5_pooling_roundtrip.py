# SPDX-License-Identifier: Apache-2.0
"""O cache guarda e devolve inteiro o estado de compressão do GLM-5.x?

O GLM-5.3-Flash monta suas camadas esparsas como
``CacheList(KVCache, PoolingCache)`` e entra no caminho de guarda por membro
— o modo ``pm``, confirmado na marca dos blocos em disco
(``layer_3_storage_mode = pm``, ``layer_3_sub_1_state_class_name = PoolingCache``).

Esse caminho tinha guarda para o par ``CacheList(KVCache, ArraysCache)``, mas
nenhuma para o par com ``PoolingCache`` — que é justamente o do GLM. O que a
compressão acumula (``pooled``) é o que o seletor de tokens usa para decidir
QUAIS pedaços do contexto o modelo vai ler; se ele voltar diferente do que foi
guardado, o modelo lê o pedaço errado e responde com gramática correta sobre o
assunto errado.

Aqui a razão de compressão é 4, a mesma do modelo (``index_kpool = 4``).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache, cachelist_pm_member_plan
from omlx.cache.type_registry import CacheTypeRegistry

try:
    import mlx.core as mx
    from mlx_vlm.models.cache import CacheList, KVCache

    # A que o servidor de fato usa: o patch do DeepSeek-V4 injeta esta classe
    # em mlx_lm.models.cache, e é dela que o glm5_next constrói o cache.
    # A injeção é obrigatória — sem ela a reconstrução falha por classe
    # desconhecida, o que seria um defeito do arreio, não do produto.
    from omlx.patches.deepseek_v4 import _inject_cache_extras

    _inject_cache_extras()
    from mlx_lm.models.cache import PoolingCache

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

BLOCK_SIZE = 8
RATIO = 4          # index_kpool do GLM-5.3-Flash
HEAD_DIM = 8
GATE_DIM = 4


class MockModel:
    def __init__(self, num_layers: int = 1):
        self._num_layers = num_layers
        self.layers = [MagicMock() for _ in range(num_layers)]

    @property
    def args(self):
        a = MagicMock()
        a.num_hidden_layers = self._num_layers
        return a


def _make_cache(tmp_path):
    paged_cache = PagedCacheManager(
        block_size=BLOCK_SIZE,
        max_blocks=100,
        model_name="test-model",
        initial_blocks=100,
    )
    ssd = PagedSSDCacheManager(
        cache_dir=tmp_path / "ssd_cache",
        max_size_bytes=100 * 1024**2,
        hot_cache_max_bytes=10 * 1024**2,
        hot_cache_only=True,
        expected_model_name="test-model",
    )
    return BlockAwarePrefixCache(
        model=MockModel(),
        paged_cache_manager=paged_cache,
        paged_ssd_cache_manager=ssd,
    ), ssd


def _position_kv(seq_len):
    pos = mx.arange(seq_len, dtype=mx.float32).reshape(1, 1, seq_len, 1)
    keys = mx.broadcast_to(pos, (1, 2, seq_len, HEAD_DIM))
    values = keys + 1000.0
    return mx.contiguous(keys), mx.contiguous(values)


def _build_glm_cachelist(seq_len):
    """CacheList(KVCache, PoolingCache) alimentado como o GLM alimenta."""
    kv = KVCache()
    keys, values = _position_kv(seq_len)
    kv.update_and_fetch(keys, values)

    pool = PoolingCache(RATIO)
    # o indexer entrega kv/gate por token e a compressão acumula janelas
    kv_tok = mx.arange(seq_len * HEAD_DIM, dtype=mx.float32).reshape(1, seq_len, HEAD_DIM)
    gate_tok = mx.arange(seq_len * GATE_DIM, dtype=mx.float32).reshape(1, seq_len, GATE_DIM)
    ready_k, ready_g, _ = pool.accumulate_windows(kv_tok, gate_tok, 0)
    if ready_k.shape[1] > 0:
        # comprime cada janela em uma linha, como o compressor do modelo faz
        janelas = ready_k.reshape(1, ready_k.shape[1] // RATIO, RATIO, HEAD_DIM)
        pool.update_and_fetch(mx.mean(janelas, axis=2))

    cl = CacheList(kv, pool)
    mx.eval([keys, values] + [t for t in pool.state if t is not None])
    return cl


def _layer_dict(cache_list):
    handler = CacheTypeRegistry.get_handler_by_class_name("CacheList")
    sd = handler.extract_state(cache_list)
    return {
        "state": list(sd["sub_states"]),
        "meta_state": (list(sd["sub_class_names"]), list(sd["sub_meta_states"])),
        "class_name": "CacheList",
        "cache_type": "CacheList",
    }


def _cache_data(seq_len):
    return [_layer_dict(_build_glm_cachelist(seq_len))]


def _store_blocks(cache, num_blocks, request_id="req-glm"):
    tokens = list(range(num_blocks * BLOCK_SIZE))
    snaps = {
        BLOCK_SIZE * (i + 1): _cache_data(BLOCK_SIZE * (i + 1))
        for i in range(num_blocks)
    }
    return cache.store_cache(request_id, tokens, _cache_data(len(tokens)),
                             boundary_snapshots=snaps)


def test_o_par_do_glm_entra_no_caminho_por_membro():
    """Antes de tudo: o layout do GLM é elegível ao modo pm? (é o que o disco mostra)"""
    cl = _build_glm_cachelist(BLOCK_SIZE)
    handler = CacheTypeRegistry.get_handler_by_class_name("CacheList")
    sd = handler.extract_state(cl)
    plano = cachelist_pm_member_plan(list(sd["sub_class_names"]), list(sd["sub_states"]))
    assert plano == ["slice", "boundary"], (
        "o par KVCache+PoolingCache do GLM deveria entrar no modo por membro; "
        f"veio {plano}"
    )


def test_o_estado_de_compressao_volta_igual(tmp_path):
    """O que a compressão acumulou tem que voltar idêntico do cache."""
    cache, _ = _make_cache(tmp_path)
    seq = 3 * BLOCK_SIZE
    esperado = _build_glm_cachelist(seq)
    pool_esperado = list(esperado.caches)[1].pooled

    tabela = _store_blocks(cache, num_blocks=3)
    assert tabela is not None, "o cache recusou guardar o layout do GLM"

    resultado = cache.reconstruct_cache(tabela)
    assert resultado is not None, "o cache não devolveu nada para o layout do GLM"

    restaurado = resultado[0]
    assert type(restaurado).__name__ == "CacheList"
    pool_restaurado = list(restaurado.caches)[1].pooled

    assert pool_esperado is not None, "o arreio não acumulou compressão nenhuma"
    assert pool_restaurado is not None, (
        "o estado de compressão voltou VAZIO — o seletor de tokens perde a "
        "referência de quais pedaços do contexto ler"
    )
    assert tuple(pool_restaurado.shape) == tuple(pool_esperado.shape), (
        f"forma diferente: guardou {pool_esperado.shape}, "
        f"voltou {pool_restaurado.shape}"
    )
    assert mx.max(mx.abs(pool_restaurado - pool_esperado)).item() == 0.0, (
        "o estado de compressão voltou com CONTEÚDO diferente do guardado"
    )


def test_o_kv_volta_no_tamanho_certo(tmp_path):
    cache, _ = _make_cache(tmp_path)
    tabela = _store_blocks(cache, num_blocks=3, request_id="req-glm-kv")
    resultado = cache.reconstruct_cache(tabela)
    assert resultado is not None
    kv = list(resultado[0].caches)[0]
    assert kv.state[0].shape[2] == 3 * BLOCK_SIZE


def _store_blocks_compactados(cache, num_blocks, request_id="req-glm-delta"):
    """Como o servidor guarda de fato: cada retrato de fronteira passa pela
    compactação e leva só as linhas de compressão que o bloco acrescentou
    (``pooling_delta_ranges``), não o acumulado inteiro."""
    from omlx.cache.pooling_delta import compact_pooling_cache_snapshot

    tokens = list(range(num_blocks * BLOCK_SIZE))
    snaps = {}
    for i in range(num_blocks):
        tc = BLOCK_SIZE * (i + 1)
        snaps[tc] = compact_pooling_cache_snapshot(_cache_data(tc), tc, BLOCK_SIZE)
        assert snaps[tc][0].get("pooling_delta_ranges"), "o arreio não compactou"
    return cache.store_cache(request_id, tokens, _cache_data(len(tokens)),
                             boundary_snapshots=snaps)


def test_o_estado_de_compressao_volta_inteiro_com_retratos_compactados(tmp_path):
    """Reprodução do defeito medido em 01/09 (cache_det2.py, turnos 6–11
    DIFERENTES): com os retratos compactados, o modo por membro guardava só o
    pedaço do bloco e, na volta, devolvia só o do ÚLTIMO bloco — a compressão
    voltava com 1/3 das linhas, e o seletor de tokens lia o contexto errado."""
    cache, _ = _make_cache(tmp_path)
    seq = 3 * BLOCK_SIZE
    esperado = _build_glm_cachelist(seq)
    pool_esperado = list(esperado.caches)[1].pooled

    tabela = _store_blocks_compactados(cache, num_blocks=3)
    assert tabela is not None, "o cache recusou guardar o layout do GLM compactado"

    resultado = cache.reconstruct_cache(tabela)
    assert resultado is not None, "o cache não devolveu nada para o layout do GLM"
    pool_restaurado = list(resultado[0].caches)[1].pooled

    assert tuple(pool_restaurado.shape) == tuple(pool_esperado.shape), (
        f"a compressão voltou com {pool_restaurado.shape[1]} linhas em vez de "
        f"{pool_esperado.shape[1]}: só o pedaço do último bloco"
    )
    assert mx.max(mx.abs(pool_restaurado - pool_esperado)).item() == 0.0, (
        "o estado de compressão voltou com CONTEÚDO diferente do guardado"
    )
