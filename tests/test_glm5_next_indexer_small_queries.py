# SPDX-License-Identifier: Apache-2.0
"""O seletor top-k do GLM-5.x usa os núcleos nativos só no preparo (≥ 64 consultas)?

Os núcleos ``dsa_indexer_scores`` e ``dsa_topk_indices`` são ladrilhados para o
preparo: a pontuação acolchoa as consultas até 64 linhas. Em decode (1 consulta)
e na janela de verificação da previsão múltipla (2 a 8) eles saem mais caros que
o caminho de MLX — medido no oQ2e com 3000 tokens de contexto, por camada
esparsa em T=1: nativo 0,96 ms, MLX 0,58; o passo inteiro 54,0 -> 49,0 ms.
"""
from __future__ import annotations

import pytest

try:
    import mlx.core as mx

    from omlx.patches.mlx_vlm_glm5_next_compat import (
        apply_mlx_vlm_glm5_next_compat_patch,
    )

    apply_mlx_vlm_glm5_next_compat_patch()
    from mlx_vlm.models.glm5_next import language as lang

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


class _Cfg:
    hidden_size = 64
    index_n_heads = 32
    index_head_dim = 128
    index_topk = 8
    index_kpool = 2
    index_kpool_always_select_tail = True
    q_lora_rank = 16


def _indexer_com_espioes(monkeypatch):
    ix = lang.Glm5NextIndexer(_Cfg())
    # pesos em fp16, como no checkpoint real: o nativo só é considerado quando
    # consultas e pool têm o mesmo dtype de meia precisão
    from mlx.utils import tree_map

    ix.update(tree_map(lambda v: v.astype(mx.float16), ix.parameters()))
    mx.eval(ix.parameters())
    chamadas = {"scores": 0, "topk": 0}

    def scores_nativo(*a, **k):
        chamadas["scores"] += 1
        raise RuntimeError("nativo não deveria ser chamado neste teste")

    def topk_nativo(*a, **k):
        chamadas["topk"] += 1
        raise RuntimeError("nativo não deveria ser chamado neste teste")

    from omlx.custom_kernels.glm_moe_dsa import fast

    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(fast, "dsa_indexer_scores", scores_nativo, raising=False)
    monkeypatch.setattr(fast, "dsa_topk_indices", topk_nativo, raising=False)
    monkeypatch.setattr(lang, "_NATIVE_INDEXER_WARNED", False)
    return ix, chamadas


@pytest.mark.parametrize("consultas", [2, 4, 8, 63])
def test_poucas_consultas_ficam_no_caminho_de_mlx(monkeypatch, consultas):
    ix, chamadas = _indexer_com_espioes(monkeypatch)
    x = mx.random.normal((1, consultas, _Cfg.hidden_size)).astype(mx.float16)
    qr = mx.random.normal((1, consultas, _Cfg.q_lora_rank)).astype(mx.float16)
    ix.bypass_short = False
    topk = ix(x, qr, None, cache=None, kv_cache=None)
    mx.eval(topk)
    assert chamadas == {"scores": 0, "topk": 0}, chamadas
    assert topk.shape[2] == consultas


def test_a_partir_de_64_consultas_tenta_o_nativo(monkeypatch):
    ix, chamadas = _indexer_com_espioes(monkeypatch)
    x = mx.random.normal((1, 64, _Cfg.hidden_size)).astype(mx.float16)
    qr = mx.random.normal((1, 64, _Cfg.q_lora_rank)).astype(mx.float16)
    ix.bypass_short = False
    topk = ix(x, qr, None, cache=None, kv_cache=None)
    mx.eval(topk)
    # o espião levanta, o indexer cai no caminho de MLX e segue; o que importa
    # e que ele TENTOU o nativo no preparo
    assert chamadas["scores"] == 1
