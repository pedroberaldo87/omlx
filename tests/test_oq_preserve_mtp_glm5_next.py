# SPDX-License-Identifier: Apache-2.0
"""Pedir para preservar a cabeça de previsão múltipla tem que preservá-la.

O GLM-5.3-Flash guarda essa cabeça como uma camada extra de transformador
(``model.language_model.layers.45.*``, o estilo do DeepSeek-V3), e não sob o
nome ``mtp.*``. O dono ligou a opção de preservar, e o resultado saiu sem ela:
o modelo de origem tem 46 camadas, o quantizado tem 45.

Nada avisou. O nome do arquivo ganhou o sufixo que só é aposto quando a opção
está ligada, e a contagem de camadas no config de saída não foi zerada — as
duas marcas de que a opção seguiu ligada até o fim. O que corta é a limpeza de
nomes do mlx-vlm, que descarta tudo além de ``num_hidden_layers``; a preservação
depende de uma limpeza própria por família de modelo, e não existe uma para
``glm5_next``.

Medido em 31/08/2026: 1760 pesos da camada 45 na origem, zero no resultado.
"""

from __future__ import annotations

import json
import os

import pytest

from omlx.utils.model_loading import (
    _checkpoint_has_mtp_weights,
    _nextn_weight_prefixes_from_config,
)

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

ORIGEM = os.path.expanduser("~/.omlx/models/zai-org/GLM-5.3-Flash")

pytestmark = pytest.mark.skipif(
    not HAS_MLX or not os.path.isdir(ORIGEM),
    reason="precisa do checkpoint de origem do GLM-5.3-Flash em disco",
)


def _config():
    return json.load(open(os.path.join(ORIGEM, "config.json")))


def test_a_origem_declara_e_carrega_a_cabeca():
    """Antes de acusar a quantização: a origem realmente tem a cabeça?"""
    cfg = _config()
    prefixos = _nextn_weight_prefixes_from_config(cfg)
    assert prefixos, "o config da origem não declara camada de previsão múltipla"

    mapa = json.load(
        open(os.path.join(ORIGEM, "model.safetensors.index.json"))
    )["weight_map"]
    pesos = [k for k in mapa if k.startswith(prefixos)]
    assert len(pesos) > 1000, (
        f"a origem deveria trazer a cabeça inteira; achei {len(pesos)} pesos"
    )
    assert _checkpoint_has_mtp_weights(ORIGEM) is True, (
        "o detector não enxerga a cabeça na origem — a opção seria desligada "
        "em silêncio antes mesmo de começar"
    )


@pytest.mark.xfail(
    strict=True,
    reason="defeito aberto: nao existe limpeza de nomes propria para "
    "glm5_next, entao a de estoque do mlx-vlm corta a camada extra. "
    "Quando alguem escrever essa limpeza, este teste passa e o aviso "
    "de XPASS cobra a remocao desta marca.",
)
def test_a_limpeza_de_nomes_nao_pode_descartar_a_cabeca():
    """O elo que quebra: com a opção LIGADA, a camada extra é descartada."""
    from omlx.oq import _build_model_sanitizer

    cfg = _config()
    mapa = json.load(
        open(os.path.join(ORIGEM, "model.safetensors.index.json"))
    )["weight_map"]

    limpeza = _build_model_sanitizer(cfg, model_path=ORIGEM, preserve_mtp=True)
    assert limpeza is not None, "sem limpeza de nomes não há o que testar"

    ultima_normal = sorted([k for k in mapa if ".layers.44." in k])[:6]
    cabeca = sorted([k for k in mapa if ".layers.45." in k])[:6]
    assert ultima_normal and cabeca

    entrada = {k: mx.zeros((2, 2), dtype=mx.float16) for k in ultima_normal + cabeca}
    saida = limpeza(dict(entrada))

    sobreviveu_normal = [k for k in saida if "layers.44" in k]
    sobreviveu_cabeca = [
        k for k in saida if "layers.45" in k or ".mtp." in k or k.startswith("mtp.")
    ]

    assert len(sobreviveu_normal) == len(ultima_normal), (
        "a camada comum deveria atravessar intacta"
    )
    assert sobreviveu_cabeca, (
        "com preservar LIGADO, a camada de previsão múltipla foi descartada pela "
        f"limpeza de nomes: entraram {len(cabeca)} pesos dela, sobraram 0. "
        "É por isso que o modelo sai com o sufixo no nome e sem a cabeça dentro."
    )


@pytest.mark.xfail(
    strict=True,
    reason="defeito aberto: o portao de compatibilidade aceita glm_moe_dsa "
    "(o GLM-5.2) e nao glm5_next (o 5.3), entao a previsao multipla nao liga "
    "nem no modelo ORIGINAL — observado pelo dono na tela de configuracoes.",
)
def test_o_portao_reconhece_o_tipo_do_glm_5_3():
    """Segundo bloqueio, independente da quantizacao.

    O dono relatou que, abrindo o modelo ORIGINAL nas configuracoes, a previsao
    multipla tambem nao fica ativada. Confere: o portao enxerga a cabeca no
    config e mesmo assim recusa, porque decide pelo nome da familia.
    """
    from omlx.utils.model_loading import _is_mtp_compatible

    cfg = _config()
    tipo = cfg.get("model_type")
    assert tipo == "glm5_next", f"o tipo mudou: {tipo}"
    assert _has_mtp_heads(cfg) is True, "o config precisa declarar a cabeca"
    assert _is_mtp_compatible(cfg, tipo) is True, (
        f"o portao ve a cabeca no config mas recusa o tipo {tipo!r}; a lista "
        "aceita glm_moe_dsa, que e a geracao anterior"
    )
