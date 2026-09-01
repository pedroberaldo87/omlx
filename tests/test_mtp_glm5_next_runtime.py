# SPDX-License-Identifier: Apache-2.0
"""O runtime de previsão múltipla do GLM-5.3 (``glm5_next``).

A cabeça é uma camada extra além do tronco (a 45, num modelo de 45 camadas
0..44), e construí-la esbarra em três coisas que este runtime existe para
tratar. A que mais dói é a primeira: ``Glm5NextDecoderLayer.__init__`` lê
``config.layer_types[layer_idx]`` e ``config.mlp_layer_types[layer_idx]``, e as
duas listas têm exatamente ``num_hidden_layers`` entradas — a camada da cabeça
cai fora do índice e o modelo nem chega a carregar.

Estes testes rodam com o config REAL do checkpoint de origem quando ele está em
disco, e com um config sintético quando não está, porque o que eles exercitam é
a construção, não os pesos.
"""

from __future__ import annotations

import os

import pytest

try:
    import mlx.core as mx  # noqa: F401

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="precisa de MLX")

ORIGEM = os.path.expanduser("~/.omlx/models/zai-org/GLM-5.3-Flash")


def _glm_com_remendo():
    """Registra a família no mlx-lm e aplica o remendo, como o servidor faz."""
    import sys

    from omlx.patches.mlx_lm_glm5_next import register_into_mlx_lm
    from omlx.patches.mlx_lm_mtp import glm5_next_model

    register_into_mlx_lm()
    aplicou = glm5_next_model.apply()
    return sys.modules.get("mlx_lm.models.glm5_next"), aplicou


def _config():
    """O config de origem se estiver em disco; senão um sintético equivalente."""
    import json

    caminho = os.path.join(ORIGEM, "config.json")
    if os.path.isfile(caminho):
        return json.load(open(caminho))
    # sintético: o que decide a construção é o padrão de tipos e a contagem
    n = 8
    return {
        "model_type": "glm5_next",
        "text_config": {
            "model_type": "glm5_next_text",
            "num_hidden_layers": n,
            "num_nextn_predict_layers": 1,
            "hidden_size": 128,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "rms_norm_eps": 1e-5,
            "vocab_size": 512,
            "layer_types": [
                "linear_attention" if (i % 4) != 3 else "deepseek_sparse_attention"
                for i in range(n)
            ],
            "mlp_layer_types": ["dense"] * 3 + ["sparse"] * (n - 3),
            "indexer_types": ["full"] * n,
            "first_k_dense_replace": 3,
        },
    }


def _args_enxutos(glm, cfg):
    """ModelArgs do config, com as dimensões encolhidas para o teste caber."""
    args = glm.ModelArgs.from_dict(cfg)
    args.hidden_size = 128
    args.num_attention_heads = 4
    args.n_routed_experts = 4
    args.moe_intermediate_size = 64
    args.intermediate_size = 64
    return args


def test_o_remendo_aplica_e_registra_o_bloco():
    glm, aplicou = _glm_com_remendo()
    assert aplicou is True, "o remendo do glm5_next não aplicou"
    assert hasattr(glm, "Glm5NextMTPBlock"), "o bloco da cabeça não foi registrado"


def test_as_listas_de_tipo_cobrem_a_camada_da_cabeca():
    """Sem isto, construir a camada da cabeça levanta IndexError."""
    glm, _ = _glm_com_remendo()
    args = glm.ModelArgs.from_dict(_config())
    n = args.num_hidden_layers
    n_mtp = getattr(args, "num_nextn_predict_layers", 0)

    assert n_mtp > 0, "o config precisa declarar a camada de previsão múltipla"
    for nome in ("layer_types", "mlp_layer_types"):
        lista = getattr(args, nome, None)
        assert isinstance(lista, list), f"{nome} não é lista"
        assert len(lista) >= n + n_mtp, (
            f"{nome} tem {len(lista)} entradas para {n} camadas comuns + {n_mtp} "
            f"da cabeça; construir a camada {n} daria IndexError"
        )


def test_a_camada_da_cabeca_espelha_a_ultima_comum():
    """O config não declara o tipo dela; espelhar a última comum é a regra."""
    glm, _ = _glm_com_remendo()
    args = glm.ModelArgs.from_dict(_config())
    n = args.num_hidden_layers
    assert args.layer_types[n] == args.layer_types[n - 1], (
        "a camada da cabeça consome o hidden que sai da última camada comum, "
        "então assume o mesmo regime dela"
    )


def test_o_bloco_da_cabeca_constroi_de_verdade():
    """O teste que importa: a camada extra vira objeto, sem IndexError."""
    glm, _ = _glm_com_remendo()
    args = _args_enxutos(glm, _config())
    n = args.num_hidden_layers

    bloco = glm.Glm5NextMTPBlock(args, n)

    for nome in ("enorm", "hnorm", "eh_proj", "norm", "block"):
        assert hasattr(bloco, nome), f"o bloco não tem {nome}"
    assert type(bloco.block).__name__ == "Glm5NextDecoderLayer"
    # a fusão recebe embedding + hidden concatenados, então entra com 2*dim
    assert bloco.eh_proj.weight.shape[1] == 2 * args.hidden_size


def test_o_cache_da_cabeca_segue_o_tipo_da_camada():
    """Camada linear pede estado recorrente; esparsa pede o par KV+compressão."""
    from omlx.patches.mlx_lm_mtp.glm5_next_model import _cache_para

    glm, _ = _glm_com_remendo()
    args = _args_enxutos(glm, _config())
    n = args.num_hidden_layers
    bloco = glm.Glm5NextMTPBlock(args, n)

    cache = _cache_para(bloco.block)
    esperado = "ArraysCache" if bloco.block.is_linear else "CacheList"
    assert type(cache).__name__ == esperado, (
        f"camada {'linear' if bloco.block.is_linear else 'esparsa'} recebeu "
        f"{type(cache).__name__}, esperado {esperado} — cache de formato errado "
        "faz a cabeça ler estado que não é dela"
    )


def test_o_renomeador_leva_a_camada_extra_para_o_prefixo_da_cabeca():
    """``model[.language_model].layers.<n>.*`` vira ``mtp.0.*``."""
    from omlx.patches.mlx_lm_mtp.glm5_next_model import _renomeia_para_mtp

    n = 45
    entrada = {
        f"model.language_model.layers.{n}.eh_proj.weight": 1,
        f"model.language_model.layers.{n}.enorm.weight": 2,
        f"model.language_model.layers.{n}.shared_head.norm.weight": 3,
        f"model.language_model.layers.{n}.self_attn.q_proj.weight": 4,
        f"model.language_model.layers.{n - 1}.self_attn.q_proj.weight": 5,
    }
    saida = _renomeia_para_mtp(entrada, n, 1)

    assert "mtp.0.eh_proj.weight" in saida
    assert "mtp.0.enorm.weight" in saida
    assert "mtp.0.norm.weight" in saida, "shared_head.norm tem que virar norm"
    assert "mtp.0.block.self_attn.q_proj.weight" in saida, (
        "o que não é peso especial da cabeça entra sob block.*"
    )
    assert f"model.language_model.layers.{n - 1}.self_attn.q_proj.weight" in saida, (
        "a camada comum não pode ser renomeada"
    )
