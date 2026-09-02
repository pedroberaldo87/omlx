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
import types

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


@pytest.mark.skipif(
    not os.path.exists(os.path.join(_ORIGEM, "model.safetensors.index.json")),
    reason="o checkpoint de origem não está em disco",
)
def test_o_prefixo_do_checkpoint_publicado_e_normalizado():
    """Os checkpoints desta família usam DOIS arranjos de prefixo.

    O REAP37, em que a limpeza foi medida, pendura a torre de texto na raiz
    (``language_model.*``). O publicado pela zai-org e o do Vontra a penduram
    sob ``model.`` (``model.language_model.*``), e nesse arranjo NENHUM dos
    nomes de camada casava com os do modelo — 221 esperados, 0 recebidos.

    A torre de visão tem que sair junto: ela vem sob ``model.visual.*`` neste
    arranjo, e não sob ``vision_model.*``.
    """
    import json

    from omlx.patches.mlx_lm_glm5_next import Model

    idx = json.load(
        open(os.path.join(_ORIGEM, "model.safetensors.index.json"), encoding="utf-8")
    )
    chaves = list(idx["weight_map"])
    assert any(k.startswith("model.language_model.") for k in chaves), (
        "o checkpoint de origem mudou de arranjo; este teste precisa ser revisto"
    )
    visao = [k for k in chaves if k.startswith("model.visual.")]
    assert visao, "o checkpoint de origem deveria trazer a torre de visão"

    # Um objeto mínimo, não `None`: quando o remendo de previsão múltipla já
    # foi aplicado por outro teste, `sanitize` é o invólucro dele, que lê
    # `self.mtp` e `self.args` antes de delegar. Sem cabeça, ele delega direto
    # para o de baixo, que é o que este teste mede.
    class _Cru:
        args = types.SimpleNamespace(num_hidden_layers=45)

    # Poucas chaves representativas, com arrays de verdade: a limpeza delega ao
    # renomeador do modelo vendorado, que opera nos valores, então uma entrada
    # vazia não atravessa. O que este teste mede é o PREFIXO.
    import mlx.core as mx

    amostra = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "model.language_model.layers.0.input_layernorm.weight",
        visao[0],
    ]
    entrada = {k: mx.zeros((4, 4)) for k in amostra}
    saida = Model.sanitize(_Cru(), dict(entrada))

    assert not [k for k in saida if "language_model" in k], (
        "sobrou chave com o prefixo de linguagem; ela não casa com o modelo"
    )
    assert not [k for k in saida if k.startswith(("model.visual.", "visual."))], (
        "a torre de visão atravessou a limpeza"
    )
    assert "model.embed_tokens.weight" in saida
    assert "model.layers.0.input_layernorm.weight" in saida
    assert len(saida) == len(amostra) - 1, (
        f"entraram {len(amostra)} (uma de visão), saíram {len(saida)}"
    )


def test_a_limpeza_do_modelo_completa_a_hiperconexao_da_cabeca():
    """A CHAMADA do preenchimento, não só a função.

    A camada da cabeça não traz os seis coeficientes de hiperconexão no
    checkpoint, e a carga estrita morre com "Missing 6 parameters" se a limpeza
    não os completar com o valor de fábrica — que é o neutro que a referência
    usa. Testar a função sozinha não cobre a linha que a invoca.
    """
    import mlx.core as mx

    glm, _ = _glm_com_remendo()
    modelo, args = _modelo_com_cabeca(glm)
    n = args.num_hidden_layers

    # os pesos da camada da cabeça como o checkpoint os traz: SEM hiperconexão
    entrada = {
        f"model.layers.{n}.eh_proj.weight": mx.zeros((args.hidden_size, 2 * args.hidden_size)),
        f"model.layers.{n}.enorm.weight": mx.ones((args.hidden_size,)),
        f"model.layers.{n}.hnorm.weight": mx.ones((args.hidden_size,)),
    }
    entrada["mtp.0.block.input_layernorm.weight"] = mx.ones((args.hidden_size,))
    saida = modelo.sanitize(dict(entrada))

    assert "mtp.0.block.input_layernorm.weight" in saida, (
        "a chave da cabeça foi descartada: o renomeador do modelo vendorado "
        "joga fora tudo que contém `mtp.`, e é por isso que ela tem que passar "
        "por FORA dele"
    )

    faltando = [
        k
        for k, _ in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(
            modelo.mtp[0].parameters()
        )
        if "_hc." in k and f"mtp.0.{k}" not in saida
    ]
    assert not faltando, (
        f"a limpeza não completou {len(faltando)} coeficientes de hiperconexão "
        f"da cabeça: {faltando}; a carga estrita morre neles"
    )


def test_o_desfazer_parcial_recusa_quando_ha_camada_recorrente():
    """Desfazer parcial com camada recorrente no tronco CORROMPE a saída.

    A esparsa guarda KV e `trim(n)` tira n posições, deixando as confirmadas.
    A linear guarda estado recorrente, que não tem posições para tirar: o par
    guardado a leva de volta ao ponto anterior a TODAS elas.

    Misturar as duas desalinha o tronco. Medido em 01/09 com temperatura zero:
    a saída trazia "26, 26" onde a geração sem a cabeça produzia a sequência
    correta. Sem o replay guardado por um forward ARMADO (aqui o cache está
    recém-criado), recusar é o que preserva a resposta.
    """
    from mlx_lm.models.cache import ArraysCache

    glm, _ = _glm_com_remendo()
    modelo, _args = _modelo_com_cabeca(glm)

    tronco = modelo.make_cache()
    assert any(isinstance(c, ArraysCache) for c in tronco), (
        "o tronco do GLM-5.3 tem camadas recorrentes; sem elas este teste não "
        "mede nada"
    )

    # com algo a tirar e camada recorrente presente: recusa
    assert modelo.mtp_partial_rollback(tronco, 0, 2) is False

    # nada a tirar (tudo aceito): aceita, porque não há desfazer a fazer
    assert modelo.mtp_partial_rollback(tronco, 2, 2) is True
    assert modelo.mtp_partial_rollback(tronco, 3, 2) is True


def test_o_desfazer_encadeado_confere_tudo_antes_de_mexer():
    """Desfazer metade deixa as camadas com comprimentos diferentes.

    A máscara de atenção é montada a partir da primeira camada, então um tronco
    meio desfeito quebra no forward seguinte, longe da causa.
    """

    class _Apáravel:
        """Uma camada que se apara, e anota se alguém a aparou."""

        rollback_state = None
        aparada = 0

        def is_trimmable(self):
            return True

        def trim(self, n):
            self.aparada += n
            return n

    class _Recusa:
        """Uma camada que não se apara e não guarda o par."""

        rollback_state = None

        def is_trimmable(self):
            return False

    glm, _ = _glm_com_remendo()
    modelo, _args = _modelo_com_cabeca(glm)

    boa = _Apáravel()
    assert modelo.mtp_partial_rollback([boa, _Recusa()], 0, 2) is False, (
        "com uma camada recusando, o desfazer inteiro tem que recusar"
    )
    assert boa.aparada == 0, (
        f"a camada apárável foi aparada {boa.aparada} vez(es) mesmo com outra "
        f"recusando; o desfazer tem que conferir TODAS antes de tocar em "
        f"qualquer uma, senão o tronco fica com comprimentos diferentes"
    )


def test_o_desfazer_nunca_mistura_restaurar_com_aparar():
    """A regra que impede a corrupção de saída, travada por construção.

    Um desfazer que RESTAURA umas camadas e APARA outras deixa o tronco com
    comprimentos diferentes: a esparsa fica com `accepted + 1` posições e a
    recorrente com zero. O forward seguinte monta a máscara a partir da
    primeira camada, e a geração sai com token repetido.

    Este teste falha se alguém reintroduzir a mistura — foi exatamente o que
    eu fiz em 01/09, e o sintoma ("26, 26" numa sequência de pares) só apareceu
    comparando a saída com e sem a cabeça em temperatura zero.
    """
    import mlx.core as mx

    glm, _ = _glm_com_remendo()
    modelo, _args = _modelo_com_cabeca(glm)

    class _Recorrente:
        """Guarda estado que não tem posições para tirar."""

        rollback_state = (None, None)
        restaurada = 0

        def is_trimmable(self):
            return False

        def __setitem__(self, i, v):
            self.restaurada += 1

        def __getitem__(self, i):
            return None

    class _Esparsa:
        rollback_state = None
        aparada = 0

        def is_trimmable(self):
            return True

        def trim(self, n):
            self.aparada += n
            return n

    rec, esp = _Recorrente(), _Esparsa()
    assert modelo.mtp_partial_rollback([esp, rec], 0, 2) is False, (
        "com camada recorrente e algo a tirar, o desfazer tem que RECUSAR"
    )
    assert esp.aparada == 0, (
        f"a esparsa foi aparada {esp.aparada} vez(es) numa recusa; isso deixa "
        f"o tronco desalinhado"
    )
    assert rec.restaurada == 0, (
        f"a recorrente foi restaurada {rec.restaurada} vez(es) numa recusa"
    )


def _logits_seguintes(modelo, cache, proximo):
    """Os logits do token seguinte, dado o cache como está."""
    import mlx.core as mx

    saida = modelo(mx.array([[proximo]]), cache=cache)
    mx.eval(saida)
    return saida


def test_o_desfazer_parcial_reprocessa_as_posicoes_aceitas():
    """Aceitar 1 de 3 rascunhos tem que deixar o tronco IGUAL a quem só viu as
    2 primeiras posições do bloco — nas 34 recorrentes e nas esparsas.

    A prova é o passo seguinte: com o mesmo próximo token, os logits saídos do
    tronco desfeito e os do tronco de referência têm que coincidir. Antes
    deste conserto o desfazer recusava sempre que havia camada recorrente, o
    ciclo nunca completava (`cycles=0`) e a geração caía para 2,1 tok/s.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    glm, _ = _glm_com_remendo()
    modelo, _args = _modelo_com_cabeca(glm)

    prefixo = mx.array([[3, 9, 17, 42, 8]])
    bloco = mx.array([[1, 55, 2, 7]])  # confirmado + 3 rascunhos
    aceitos = 1  # fica o confirmado e o 1º rascunho

    # referência: o prefixo e só as posições aceitas
    ref = modelo.make_cache()
    modelo(prefixo, cache=ref)
    modelo(bloco[:, : aceitos + 1], cache=ref)
    esperado = _logits_seguintes(modelo, ref, 99)

    # o caminho real: verificação armada do bloco inteiro, depois o desfazer
    cache = modelo.make_cache()
    modelo(prefixo, cache=cache)
    assert any(isinstance(c, ArraysCache) for c in cache)
    modelo(bloco, cache=cache, return_hidden=True)  # arma e captura
    for c in cache:
        if isinstance(c, ArraysCache):
            assert c.rollback_replay is not None, (
                "a camada recorrente não guardou o que o replay precisa"
            )
    assert modelo.mtp_partial_rollback(cache, aceitos, 3) is True, (
        "com o replay guardado, o desfazer parcial tem que ACEITAR"
    )
    for c in cache:
        if isinstance(c, ArraysCache):
            assert c.rollback_replay is None, "o replay é de uso único"
    obtido = _logits_seguintes(modelo, cache, 99)

    assert obtido.shape == esperado.shape
    dif = float(mx.max(mx.abs(obtido - esperado)).item())
    assert dif < 1e-3, (
        f"o tronco desfeito diverge da referência em {dif:.2e}; o desfazer não "
        f"deixou as duas famílias de camada no mesmo ponto"
    )


def test_o_desfazer_parcial_sem_replay_e_recusado_e_com_replay_errado_diverge():
    """A mutação que prova que o teste anterior morde.

    (1) Forward NÃO armado: nada guardado → o desfazer recusa, e o chamador cai
    no passo comum (correto, lento). (2) Um replay que só restaura o estado
    anterior — o desfazer antigo — deixa a recorrente em zero posições do
    bloco enquanto a esparsa fica com duas: os logits seguintes DIVERGEM.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    glm, _ = _glm_com_remendo()
    modelo, _args = _modelo_com_cabeca(glm)
    prefixo = mx.array([[3, 9, 17, 42, 8]])
    bloco = mx.array([[1, 55, 2, 7]])

    # (1) sem armar
    cache = modelo.make_cache()
    modelo(prefixo, cache=cache)
    modelo(bloco, cache=cache)
    assert modelo.mtp_partial_rollback(cache, 1, 3) is False

    # (2) replay mutado: devolve o estado anterior sem reprocessar nada
    ref = modelo.make_cache()
    modelo(prefixo, cache=ref)
    modelo(bloco[:, :2], cache=ref)
    esperado = _logits_seguintes(modelo, ref, 99)

    cache = modelo.make_cache()
    modelo(prefixo, cache=cache)
    modelo(bloco, cache=cache, return_hidden=True)
    for c in cache:
        if isinstance(c, ArraysCache):
            anterior, recorrente = c.rollback_state
            c.rollback_replay = lambda n_keep, a=anterior, r=recorrente: (a, r)
    assert modelo.mtp_partial_rollback(cache, 1, 3) is True
    obtido = _logits_seguintes(modelo, cache, 99)
    dif = float(mx.max(mx.abs(obtido - esperado)).item())
    assert dif > 1e-3, (
        f"restaurar sem reprocessar deu diferença {dif:.2e}: o teste de "
        f"identidade não estaria medindo nada"
    )


def test_o_desfazer_parcial_funciona_com_o_cache_em_lote_do_gerador():
    """O servidor não usa ``modelo.make_cache()``: o BatchGenerator do mlx-lm
    monta o cache com ``left_padding=[0]`` (a API em lote, mesmo para uma
    sequência só), e a camada recorrente então recebe uma MÁSCARA toda
    verdadeira em vez de ``None``. A captura do replay exigia ``mask is None``
    e nunca acontecia: cada rascunho recusado caía no passo comum, que
    RE-PREFILLAVA o contexto inteiro num único forward (`_reconcile_mtp_to_
    standard`) — medido em 01/09 no oQ2e: a cada 3 tokens, um forward de 2600
    posições; e esse prefill de um golpe diverge do prefill em pedaços (KV da
    camada 7 já difere em 38 posições), o que num agente com 30 mil tokens de
    contexto terminou em lixo ("NoNo", "DEDEDE").
    """
    import sys

    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    glm, _ = _glm_com_remendo()
    modelo, _args = _modelo_com_cabeca(glm)
    make_cache = sys.modules["mlx_lm.generate"]._make_cache

    prefixo = mx.array([[3, 9, 17, 42, 8]])
    bloco = mx.array([[1, 55, 2, 7]])
    aceitos = 1

    ref = make_cache(modelo, [0], None)
    modelo(prefixo, cache=ref)
    modelo(bloco[:, : aceitos + 1], cache=ref)
    esperado = _logits_seguintes(modelo, ref, 99)

    cache = make_cache(modelo, [0], None)
    modelo(prefixo, cache=cache)
    lineares = [c for c in cache if isinstance(c, ArraysCache)]
    assert lineares and all(c.make_mask(4) is not None for c in lineares), (
        "o arreio tem que reproduzir o cache em lote: máscara não-None"
    )
    modelo(bloco, cache=cache, return_hidden=True)
    for c in lineares:
        assert c.rollback_replay is not None, (
            "com o cache em lote (máscara toda verdadeira) a recorrente não "
            "guardou o replay — é o que derruba o ciclo para o re-prefill"
        )
    assert modelo.mtp_partial_rollback(cache, aceitos, 3) is True
    obtido = _logits_seguintes(modelo, cache, 99)
    dif = float(mx.max(mx.abs(obtido - esperado)).item())
    assert dif < 1e-3, dif
