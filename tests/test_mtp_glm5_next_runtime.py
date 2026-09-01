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
import re

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


def test_a_camada_da_cabeca_usa_atencao_esparsa():
    """A cabeça NÃO espelha a última camada comum — ela é sempre esparsa.

    Espelhar a 44 (linear) era o palpite, e o checkpoint de referência o
    derruba: a camada 45 de ``zai-org/GLM-5.3-Flash`` traz
    ``self_attn.indexer.*``, ``kv_a_proj_with_mqa`` e ``q_a_proj`` — pesos que
    só a esparsa tem — e nenhum ``conv1d``/``forget_gate`` da linear. O config
    diz o mesmo por outro caminho: ``index_share_for_mtp_iteration`` liga o
    indexer à iteração da cabeça, e indexer é peça exclusiva da esparsa.

    Construir linear não levanta erro nenhum — só monta o bloco errado, e aí
    nenhum peso da cabeça encontra destino no carregamento.
    """
    glm, _ = _glm_com_remendo()
    args = glm.ModelArgs.from_dict(_config())
    n = args.num_hidden_layers
    assert "linear" not in args.layer_types[n], (
        f"a camada da cabeça saiu {args.layer_types[n]!r}; o checkpoint tem "
        f"indexer e projeções q_a/kv_a nela, que só a atenção esparsa monta"
    )
    esparsas = [t for t in args.layer_types[:n] if "linear" not in t]
    assert args.layer_types[n] == esparsas[-1], (
        "o tipo tem que ser o mesmo nome que o tronco já usa para a esparsa, "
        "não uma string inventada aqui"
    )


_ORIGEM = os.path.expanduser("~/.omlx/models/zai-org/GLM-5.3-Flash")


@pytest.mark.skipif(
    not os.path.exists(os.path.join(_ORIGEM, "model.safetensors.index.json")),
    reason="o checkpoint de origem não está em disco",
)
def test_o_bloco_construido_bate_com_os_pesos_do_checkpoint():
    """A prova contra o dado real, com a régua calibrada pelo próprio modelo.

    O que se exige não é casamento perfeito de nomes — nem a camada COMUM tem
    isso: o modelo cria coeficientes de hiperconexão sob outro nome, gera
    ``embed_q``/``unembed_out`` a partir do ``kv_b_proj`` do disco, e o
    renomeador acerta tudo no carregamento.

    O que se exige é que a camada da cabeça **não divirja mais que uma camada
    comum do mesmo tipo**. Assim a régua acompanha o modelo em vez de virar
    uma lista de exceções escrita à mão — e ela morde no que importa: com o
    tipo errado a cabeça vinha com doze pesos da atenção linear que não
    existem em camada nenhuma deste checkpoint.
    """
    import json
    from mlx.utils import tree_flatten

    from omlx.patches.mlx_lm_mtp.glm5_next_model import _vendored

    glm, _ = _glm_com_remendo()
    cfg = json.load(open(os.path.join(_ORIGEM, "config.json"), encoding="utf-8"))
    texto = cfg.get("text_config", cfg)
    params = dict(texto)
    params["num_nextn_predict_layers"] = cfg.get(
        "num_nextn_predict_layers", texto.get("num_nextn_predict_layers", 1)
    )
    args = _args_enxutos(glm, params)
    n = int(texto["num_hidden_layers"])

    idx = json.load(
        open(os.path.join(_ORIGEM, "model.safetensors.index.json"), encoding="utf-8")
    )

    def no_disco(camada):
        alvo = f".layers.{camada}."
        nomes = {k.split(alvo, 1)[1] for k in idx["weight_map"] if alvo in k}
        # o disco guarda um expert por vez; o MLX os empilha num tensor só
        nomes = {re.sub(r"experts\.\d+\.", "switch_mlp.", k) for k in nomes}
        # e guarda os fatores de escala da quantização de origem à parte
        return {k for k in nomes if not k.endswith("_scale_inv")}

    # a última camada COMUM do mesmo tipo é a régua
    tipos = list(texto["layer_types"])
    comum = max(i for i, t in enumerate(tipos) if "linear" not in t)
    DecoderLayer, _lf = _vendored()
    criados_comum = {k for k, _ in tree_flatten(DecoderLayer(args, comum).parameters())}
    tolerado = criados_comum - no_disco(comum)

    bloco = glm.Glm5NextMTPBlock(args, n)
    criados = {k for k, _ in tree_flatten(bloco.parameters())}
    # os nomes do bloco vêm prefixados por "block." fora do trio da cabeça
    criados = {k[len("block.") :] if k.startswith("block.") else k for k in criados}
    # a norma final tem nome próprio no disco
    criados = {"shared_head.norm.weight" if k == "norm.weight" else k for k in criados}

    orfaos = criados - no_disco(n) - tolerado
    assert not orfaos, (
        f"o bloco da cabeça cria {len(orfaos)} pesos que a camada {n} do "
        f"checkpoint não tem e que uma camada comum ({comum}) também não cria: "
        f"{sorted(orfaos)}"
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


def test_o_bloco_da_cabeca_roda_de_ponta_a_ponta():
    """O teste que só um forward de verdade pega.

    O tronco não entrega o hidden cru às camadas: repete-o em ``hc_mult``
    cópias antes do laço, porque a hiperconexão lê quatro eixos, e recolhe
    pela média ao sair. A cabeça é uma camada do mesmo tipo e precisa do mesmo
    par — sem ele o forward estoura na primeira linha da hiperconexão, e
    nenhuma checagem de nome de peso ou de tipo de camada percebe.

    Roda os dois regimes: o preenchimento (várias posições de uma vez) e o
    passo encadeado de uma posição, que é como a cabeça é usada no rascunho.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.base import create_attention_mask

    from omlx.patches.mlx_lm_mtp.glm5_next_model import _cache_para

    glm, _ = _glm_com_remendo()
    cfg = _config()
    cfg["text_config"]["hc_mult"] = 4
    args = _args_enxutos(glm, cfg)
    args.num_experts_per_tok = 2
    n = args.num_hidden_layers

    bloco = glm.Glm5NextMTPBlock(args, n)
    cache = _cache_para(bloco.block)
    embed = nn.Embedding(args.vocab_size, args.hidden_size)

    ids = mx.array([[3, 9, 17, 42, 8, 1]])
    h = mx.random.normal((1, ids.shape[1], args.hidden_size)).astype(mx.float32)
    mask = create_attention_mask(h, None)

    saida = bloco(h, embed, ids, mask, cache)
    mx.eval(saida)
    assert saida.shape == h.shape, (
        f"a cabeça devolveu {saida.shape} para uma entrada {h.shape}; ela tem "
        f"que recolher as cópias da hiperconexão antes de sair"
    )
    assert bool(mx.all(mx.isfinite(saida)).item()), "a cabeça devolveu não-finito"

    # o passo encadeado: uma posição, aproveitando o cache do preenchimento
    ids2 = mx.array([[77]])
    h2 = mx.random.normal((1, 1, args.hidden_size)).astype(mx.float32)
    saida2 = bloco(h2, embed, ids2, None, cache)
    mx.eval(saida2)
    assert saida2.shape == h2.shape
    assert bool(mx.all(mx.isfinite(saida2)).item()), "o passo encadeado saiu não-finito"


def _modelo_com_cabeca(glm):
    """Um Model de verdade, tronco curto, com a cabeça anexada e ligada."""
    import mlx.core as mx  # noqa: F401

    from omlx.patches import mlx_lm_mtp

    cfg = _config()
    texto = cfg.get("text_config", cfg)
    p = dict(texto)
    p["num_nextn_predict_layers"] = 1
    # tronco curto: o padrão três-lineares-uma-esparsa cabe inteiro em 8
    p["num_hidden_layers"] = 8
    p["layer_types"] = list(texto["layer_types"])[:8]
    p["mlp_layer_types"] = list(texto["mlp_layer_types"])[:8]

    args = _args_enxutos(glm, p)
    args.num_experts_per_tok = 2
    args.hc_mult = int(texto.get("hc_mult", 4) or 4)

    antes = mlx_lm_mtp.is_mtp_active()
    mlx_lm_mtp.set_mtp_active(True)
    try:
        modelo = glm.Model(args)
    finally:
        mlx_lm_mtp.set_mtp_active(antes)
    return modelo, args


def test_a_cabeca_enxerga_o_historico_ao_rascunhar_varias_posicoes():
    """O regime que a previsão múltipla de fato usa, e o único que pega o erro.

    A máscara da cabeça sai do cache da camada dela. Numa camada esparsa esse
    cache é um PAR — o KV e o acumulado de compressão do seletor — e quem
    conta posições é o KV de dentro. É de lá que o tronco tira a máscara dele
    também. Passar o par inteiro faz a máscara nascer sem o histórico: larga
    só o bastante para as posições novas.

    Os outros dois regimes escondem isso por acidente: no preenchimento o
    cache está vazio, e num passo de uma posição só não há máscara nenhuma.
    Rascunhar várias posições com histórico é o que expõe.
    """
    import mlx.core as mx

    glm, _ = _glm_com_remendo()
    modelo, args = _modelo_com_cabeca(glm)
    assert hasattr(modelo, "mtp") and modelo.mtp, "a cabeça não foi anexada"

    ids = mx.array([[3, 9, 17, 42, 8, 1, 55, 2]])
    _logits, h = modelo(ids, cache=modelo.make_cache(), return_hidden=True)
    cache = modelo.make_mtp_cache()

    # preenchimento: enche o cache da cabeça com as 8 posições
    saida = modelo.mtp_forward(h, ids, cache=cache)
    mx.eval(saida)
    assert saida.shape[:2] == ids.shape
    assert cache[0][0].offset == ids.shape[1]

    # o caso que importa: rascunhar 4 posições com as 8 anteriores no cache
    ids2 = mx.array([[11, 12, 13, 14]])
    h2 = mx.random.normal((1, 4, args.hidden_size)).astype(h.dtype)
    saida2 = modelo.mtp_forward(h2, ids2, cache=cache)
    mx.eval(saida2)
    assert saida2.shape[:2] == ids2.shape, (
        "rascunhar várias posições com histórico no cache tem que funcionar — "
        "é o regime encadeado da previsão múltipla"
    )
    assert bool(mx.all(mx.isfinite(saida2)).item())
    assert cache[0][0].offset == 12, (
        f"o cache da cabeça ficou em {cache[0][0].offset}; 8 do preenchimento "
        f"mais 4 do rascunho são 12"
    )
