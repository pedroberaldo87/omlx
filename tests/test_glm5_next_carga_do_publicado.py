"""Os elos que faltavam para o GLM-5.3 publicado carregar pelo caminho de texto.

Cada um foi medido em 01/09/2026 carregando
``Vontra/GLM-5.3-Flash-MLX-2bit-MTP`` de verdade, e cada um só aparece depois
que o anterior sai da frente — por isso eles vêm juntos aqui.
"""

from __future__ import annotations

import json
import os

import mlx.core as mx
import pytest

ORIGEM = os.path.expanduser("~/.omlx/models/zai-org/GLM-5.3-Flash")


def test_o_portao_de_esquecimento_leva_os_companheiros_de_quantizacao():
    """Num checkpoint já quantizado, `.scales` e `.biases` acompanham o peso.

    A limpeza MOVE `self_attn.f_a_proj` para dentro de `forget_gate`. Ela casava
    só o sufixo `.weight`, então num checkpoint quantizado os dois companheiros
    ficavam para trás: 136 chaves recusadas por não existirem no modelo.
    """
    from omlx.patches.mlx_vlm_glm5_next_compat import (
        apply_mlx_vlm_glm5_next_compat_patch,
    )

    apply_mlx_vlm_glm5_next_compat_patch()
    from mlx_vlm.models.glm5_next.language import LanguageModel

    entrada = {
        f"model.layers.0.self_attn.{proj}{sufixo}": mx.zeros((2, 2))
        for proj in ("f_a_proj", "f_b_proj")
        for sufixo in (".weight", ".scales", ".biases")
    }
    # A limpeza lê a contagem de camadas antes de renomear; um objeto mínimo
    # basta, `None` não.
    import types

    contexto = types.SimpleNamespace(
        args=types.SimpleNamespace(num_hidden_layers=45, quantization=None)
    )
    saida = LanguageModel.sanitize(contexto, dict(entrada))

    for proj in ("f_a_proj", "f_b_proj"):
        for sufixo in (".weight", ".scales", ".biases"):
            alvo = f"model.layers.0.self_attn.forget_gate.{proj}{sufixo}"
            assert alvo in saida, (
                f"{proj}{sufixo} não entrou no portão de esquecimento; num "
                f"checkpoint quantizado ele fica órfão"
            )


def test_a_regra_de_quantizacao_segue_a_projecao_para_dentro_do_portao():
    """A regra por camada aponta o caminho ANTIGO da projeção.

    Ela vale para o nome publicado, mas o módulo do modelo está em
    `self_attn.forget_gate.f_a_proj`. Sem a variante, a busca erra, o global de
    2 bits é usado, e a carga morre com "Expected shape (128, 256) but received
    shape (128, 1024)".
    """
    from omlx.utils.model_loading import expand_per_layer_quant_keys

    cfg = {
        "model_type": "glm5_next",
        "quantization": {
            "bits": 2,
            "group_size": 64,
            "language_model.model.layers.0.self_attn.f_a_proj": {
                "bits": 8,
                "group_size": 64,
                "mode": "affine",
            },
        },
    }
    quant = expand_per_layer_quant_keys(cfg)["quantization"]

    esperado = "model.layers.0.self_attn.forget_gate.f_a_proj"
    assert esperado in quant, (
        f"falta a variante {esperado}; ela é o caminho que o modelo de texto "
        f"usa — prefixo trocado E projeção movida, as duas coisas juntas"
    )
    assert quant[esperado]["bits"] == 8


@pytest.mark.skipif(
    not os.path.exists(os.path.join(ORIGEM, "model.safetensors.index.json")),
    reason="o checkpoint de origem não está em disco",
)
def test_o_checkpoint_nao_traz_hiperconexao_na_camada_da_cabeca():
    """O fato do checkpoint que obriga o preenchimento, medido no disco.

    As camadas comuns trazem os seis coeficientes; a da cabeça não traz nenhum.
    Não é peso perdido: a referência os deixa no valor de fábrica, que é o
    neutro. Se um dia o checkpoint passar a trazê-los, este teste avisa que o
    preenchimento virou desnecessário.
    """
    idx = json.load(
        open(os.path.join(ORIGEM, "model.safetensors.index.json"), encoding="utf-8")
    )
    chaves = list(idx["weight_map"])
    comum = [k for k in chaves if ".layers.43." in k and ".hc_" in k]
    cabeca = [k for k in chaves if ".layers.45." in k and ".hc_" in k]

    assert len(comum) == 6, f"a camada comum deveria ter 6 coeficientes, tem {len(comum)}"
    assert not cabeca, (
        f"a camada da cabeça passou a trazer {len(cabeca)} coeficientes; o "
        f"preenchimento com o valor de fábrica pode não ser mais o certo"
    )


def test_a_limpeza_preenche_a_hiperconexao_que_falta_na_cabeca():
    """Sem isto a carga estrita morre com "Missing 6 parameters"."""
    import types

    from mlx.utils import tree_flatten

    from omlx.patches.mlx_lm_mtp.glm5_next_model import (
        _completa_hiperconexao_da_cabeca,
    )

    class _BlocoFalso:
        def parameters(self):
            return {
                "block": {
                    "attn_hc": {
                        "base": mx.zeros((24,)),
                        "fn": mx.zeros((24, 16)),
                        "scale": mx.ones((3,)),
                    },
                    "ffn_hc": {
                        "base": mx.zeros((24,)),
                        "fn": mx.zeros((24, 16)),
                        "scale": mx.ones((3,)),
                    },
                    "input_layernorm": {"weight": mx.ones((8,))},
                }
            }

    modelo = types.SimpleNamespace(mtp=[_BlocoFalso()])
    weights = {"mtp.0.block.input_layernorm.weight": mx.ones((8,))}
    saida = _completa_hiperconexao_da_cabeca(modelo, dict(weights))

    for grupo in ("attn_hc", "ffn_hc"):
        for parte in ("base", "fn", "scale"):
            chave = f"mtp.0.block.{grupo}.{parte}"
            assert chave in saida, f"faltou preencher {chave}"
    # os valores de fábrica são o neutro: mistura zerada, escala unitária
    assert float(mx.abs(saida["mtp.0.block.attn_hc.fn"]).max().item()) == 0.0
    assert float(saida["mtp.0.block.attn_hc.scale"].min().item()) == 1.0
    # o que já veio no checkpoint não é tocado
    assert saida["mtp.0.block.input_layernorm.weight"] is weights[
        "mtp.0.block.input_layernorm.weight"
    ]


def test_a_limpeza_de_texto_converte_os_nomes_crus_do_publicado():
    """Só trocar o prefixo não basta — os nomes ainda são os crus.

    Quem sabe convertê-los é o renomeador do modelo vendorado, e ele NÃO roda
    sozinho neste caminho: a classe do mlx-lm não segura o `LanguageModel` (ela
    pega os submódulos direto, de propósito) e o mlx-lm chama a limpeza uma vez
    só. Sem a delegação, medido no Vontra: 112.180 parâmetros recusados por não
    existirem no modelo.

    (Que a cabeça sobrevive à delegação — o renomeador vendorado descarta toda
    chave com `mtp.` — é medido em ``test_mtp_glm5_next_runtime.py``, com um
    modelo que TEM cabeça. Num objeto sem cabeça, descartar é o certo.)
    """
    import types

    from omlx.patches.mlx_lm_glm5_next import Model

    contexto = types.SimpleNamespace(
        args=types.SimpleNamespace(num_hidden_layers=45, quantization=None)
    )
    entrada = {
        "model.language_model.layers.0.hc_attn_base": mx.zeros((4,)),
        "model.language_model.layers.0.hc_ffn_base": mx.zeros((4,)),
        "model.language_model.layers.0.self_attn.f_a_proj.weight": mx.zeros((2, 2)),
    }
    saida = Model.sanitize(contexto, dict(entrada))

    assert "model.layers.0.attn_hc.base" in saida, (
        "o coeficiente de hiperconexão não foi convertido; ele chega como "
        "`hc_attn_base` e o modelo tem `attn_hc.base`"
    )
    assert "model.layers.0.ffn_hc.base" in saida
    assert "model.layers.0.self_attn.forget_gate.f_a_proj.weight" in saida, (
        "a projeção não entrou no portão de esquecimento"
    )
