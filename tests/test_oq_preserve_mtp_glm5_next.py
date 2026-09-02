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
    _has_mtp_heads,
    _nextn_weight_prefixes_from_config,
)

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

ORIGEM = os.path.expanduser("~/.omlx/models/zai-org/GLM-5.3-Flash")

# A pasta pode existir vazia depois de o checkpoint ser apagado — o que decide
# e o indice de pesos, nao o diretorio.
#
# O pulo vale SO para os testes que leem o checkpoint real. Os que exercitam a
# protecao da quantizacao rodam sempre: eram eles que precisavam existir quando o
# defeito aconteceu, e um pytestmark de modulo os apagava justo na maquina onde o
# checkpoint nao esta.
precisa_do_checkpoint = pytest.mark.skipif(
    not HAS_MLX or not os.path.isfile(
        os.path.join(ORIGEM, "model.safetensors.index.json")
    ),
    reason="precisa do checkpoint de origem do GLM-5.3-Flash em disco",
)


def _config():
    return json.load(open(os.path.join(ORIGEM, "config.json")))


@precisa_do_checkpoint
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


@precisa_do_checkpoint
def test_a_limpeza_de_nomes_nao_pode_descartar_a_cabeca():
    """Com a opcao LIGADA, a camada extra atravessa a limpeza de nomes.

    Este teste nasceu VERMELHO, marcado como falha esperada estrita, travando
    o defeito: a limpeza de visao descartava a cabeca. Consertado em f98c67fb
    (familias so-texto sao roteadas para a limpeza de texto quando a
    preservacao esta ligada), e o proprio aviso de XPASS cobrou a remocao da
    marca — que era exatamente para isso que ela existia.
    """
    from omlx.oq import _build_model_sanitizer

    cfg = _config()
    mapa = json.load(
        open(os.path.join(ORIGEM, "model.safetensors.index.json"))
    )["weight_map"]

    limpeza = _build_model_sanitizer(cfg, model_path=ORIGEM, preserve_mtp=True)
    assert limpeza is not None, "sem limpeza de nomes não há o que testar"

    # So pesos NAO quantizados: este teste mede NOMES, e a limpeza desfaz a
    # quantizacao de origem quando ve um `_scale_inv` ao lado — o que cobraria
    # dtype e forma reais de um arreio que so tem nomes.
    escalados = {k[: -len("_scale_inv")] for k in mapa if k.endswith("_scale_inv")}

    def limpos(camada):
        return sorted(
            k
            for k in mapa
            if f".layers.{camada}." in k
            and not k.endswith("_scale_inv")
            and k not in escalados
        )

    ultima_normal = limpos(44)[:6]
    cabeca = limpos(45)[:6]
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


@precisa_do_checkpoint
def test_o_portao_reconhece_o_tipo_do_glm_5_3():
    """Segundo bloqueio, agora fechado.

    O dono relatou que, abrindo o modelo ORIGINAL nas configuracoes, a previsao
    multipla tambem nao ficava ativada — o portao enxergava a cabeca no config e
    recusava pelo nome da familia. Aberto para glm5_next junto com o runtime que
    a cabeca precisa; abrir sem ele armaria o ciclo de rascunho sem cabeca.

    Este teste nasceu como falha esperada estrita e ficou verde com o conserto.
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


# ---------------------------------------------------------------------------
# A protecao: quantizar sem a cabeca, com a opcao ligada, tem que FALHAR ALTO.
# Estes rodam sem o checkpoint em disco — nao dependem do pytestmark acima.
# ---------------------------------------------------------------------------


def _checkpoint_falso(pasta, com_cabeca: bool):
    """Escreve o minimo que _checkpoint_has_mtp_weights le: config + indice."""
    import json

    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm5_next",
                "text_config": {
                    "num_hidden_layers": 2,
                    "num_nextn_predict_layers": 1,
                },
            }
        )
    )
    pesos = {"model.language_model.layers.0.self_attn.q_proj.weight": "s0.safetensors"}
    if com_cabeca:
        pesos["model.language_model.layers.2.eh_proj.weight"] = "s0.safetensors"
    (pasta / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": pesos})
    )
    return pasta


@pytest.mark.skipif(not HAS_MLX, reason="precisa de MLX")
def test_o_detector_separa_com_cabeca_de_sem_cabeca(tmp_path):
    """Rede do proprio arreio: se o detector nao separa, o teste abaixo nao vale."""
    from omlx.utils.model_loading import _checkpoint_has_mtp_weights

    assert _checkpoint_has_mtp_weights(_checkpoint_falso(tmp_path / "com", True))
    assert not _checkpoint_has_mtp_weights(_checkpoint_falso(tmp_path / "sem", False))


def test_resultado_sem_cabeca_com_a_opcao_ligada_nao_passa_calado(tmp_path):
    """O modelo de 31/08 saiu com '-mtp' no nome e sem a cabeca dentro.

    A conferencia de saida existe para que isso pare de ser possivel: com a opcao
    ligada e o resultado sem a cabeca, a quantizacao levanta erro em vez de gravar.
    """
    import inspect

    from omlx import oq

    fonte = inspect.getsource(oq)
    assert "resultado saiu SEM a cabeca" in fonte, (
        "a conferencia de saida sumiu de omlx/oq.py — sem ela a quantizacao volta a "
        "gravar um modelo com o sufixo '-mtp' e nada dentro"
    )
    # a conferencia tem que rodar DEPOIS de escrever o resultado (ela le o output),
    # e antes de anunciar sucesso
    pos_conf = fonte.index("resultado saiu SEM a cabeca")
    pos_ok = fonte.index("Quantized model saved")
    assert pos_conf < pos_ok, (
        "a conferencia precisa vir antes de anunciar 'Quantized model saved'"
    )


def test_a_estimativa_avisa_quando_desliga_a_preservacao():
    """O caminho que so estima desligava a opcao calado."""
    import inspect

    from omlx import oq

    fonte = inspect.getsource(oq)
    assert "prices the model WITHOUT" in fonte, (
        "o aviso do caminho de estimativa sumiu — ele desligava preserve_mtp em "
        "silencio, e a estimativa que o dono le antes de mandar quantizar ja "
        "contava sem a cabeca"
    )


# ---------------------------------------------------------------------------
# O conserto: com a opcao ligada, a limpeza de nomes preserva a cabeca.
# Precisa so do config.json do original (nao do checkpoint inteiro), entao roda
# enquanto os pesos ainda estao baixando.
# ---------------------------------------------------------------------------

precisa_do_config = pytest.mark.skipif(
    not HAS_MLX or not os.path.isfile(os.path.join(ORIGEM, "config.json")),
    reason="precisa do config.json do GLM-5.3-Flash de origem",
)


def _amostra_com_a_cabeca(n_camadas):
    """Um peso da ultima camada comum + os tres que formam a cabeca."""
    import mlx.core as _mx

    pesos = {
        f"model.language_model.layers.{n_camadas - 1}.input_layernorm.weight":
            _mx.zeros((2, 2), dtype=_mx.float16)
    }
    for nome in ("eh_proj", "enorm", "hnorm"):
        pesos[f"model.language_model.layers.{n_camadas}.{nome}.weight"] = _mx.zeros(
            (2, 2), dtype=_mx.float16
        )
    return pesos


@precisa_do_config
def test_com_a_opcao_ligada_a_cabeca_atravessa_a_limpeza():
    """Com a opcao ligada a cabeca atravessa a limpeza — a de VISAO.

    O conserto de 31/08 desviava o GLM-5.x para a limpeza de texto, sob a
    premissa de que a familia nao tinha peso de visao a proteger. Tinha: 347
    tensores `model.visual.*`, que a limpeza de texto jogava fora (o oQ2e saiu
    so-texto). Desde 02/09 o modelo fica na limpeza de visao e e ela que
    preserva a cabeca, sob `language_model.mtp.<i>.*`.
    """
    import json

    from omlx.oq import _build_model_sanitizer, _is_mtp_tensor

    cfg = json.load(open(os.path.join(ORIGEM, "config.json")))
    n = (cfg.get("text_config") or {}).get("num_hidden_layers")
    assert n, "o config de origem precisa declarar num_hidden_layers"

    limpeza = _build_model_sanitizer(cfg, model_path=ORIGEM, preserve_mtp=True)
    assert limpeza is not None, "sem limpeza de nomes nao ha o que testar"

    saida = limpeza(_amostra_com_a_cabeca(n))
    # A cabeca pode atravessar com o nome de origem (``layers.<n>.*``) ou ja
    # renomeada para ``mtp.<i>.*`` — as duas contam, porque as duas sao formas
    # que o detector de pesos reconhece (_MTP_WEIGHT_PREFIXES inclui "mtp.").
    # Quem renomeia e o runtime de previsao multipla, quando ele esta aplicado
    # no processo; sem ele os nomes de origem seguem intactos.
    sobrou = [k for k in saida if f"layers.{n}." in k or _is_mtp_tensor(k)]
    # Os tres que ENTRARAM tem que sair. Podem sair mais: desde 01/09 a limpeza
    # PREENCHE os seis coeficientes de hiperconexao que o checkpoint nao traz na
    # camada da cabeca, com o valor de fabrica que a referencia usa — sem isso a
    # carga estrita morre com "Missing 6 parameters".
    trio = {"eh_proj.weight", "enorm.weight", "hnorm.weight"}
    achados = {k.split(".")[-2] + "." + k.split(".")[-1] for k in sobrou}
    assert trio <= achados, (
        f"com preserve_mtp ligado a cabeca tem que atravessar inteira; faltaram "
        f"{sorted(trio - achados)} em {sorted(sobrou)}. Sem isso a quantizacao "
        f"grava um modelo com o sufixo '-mtp' no nome e sem a cabeca dentro."
    )
    extras = achados - trio
    assert all("_hc." in k or k.startswith("attn_hc") or k.startswith("ffn_hc")
               for k in extras), (
        f"sobrou coisa que nao e nem o trio da cabeca nem hiperconexao: "
        f"{sorted(extras)}"
    )


@precisa_do_config
def test_com_a_opcao_ligada_a_torre_de_visao_tambem_atravessa():
    """Preservar a cabeca nao pode custar a visao.

    O oQ2e de 31/08 saiu com 2716 pesos e nenhum `model.visual.*` (a origem
    tem 347), com o config ainda anunciando `vision_config`: o servidor
    tentava o motor de visao, falhava ("2713 parameters not in model") e caia
    no de texto. Agora a torre sai sob `vision_model.*`, com os nomes que o
    carregador de visao espera, ao lado da cabeca.
    """
    import json

    import mlx.core as _mx

    from omlx.oq import _build_model_sanitizer, _is_mtp_tensor

    cfg = json.load(open(os.path.join(ORIGEM, "config.json")))
    n = (cfg.get("text_config") or {}).get("num_hidden_layers")
    limpeza = _build_model_sanitizer(cfg, model_path=ORIGEM, preserve_mtp=True)
    assert limpeza is not None

    pesos = _amostra_com_a_cabeca(n)
    pesos["model.visual.blocks.0.attn.proj.weight"] = _mx.zeros((2, 2), dtype=_mx.float16)
    pesos["model.visual.merger.proj.weight"] = _mx.zeros((2, 2), dtype=_mx.float16)
    pesos["lm_head.weight"] = _mx.zeros((2, 2), dtype=_mx.float16)
    saida = limpeza(dict(pesos))

    visao = sorted(k for k in saida if k.startswith("vision_model."))
    assert visao == [
        "vision_model.blocks.0.attn.proj.weight",
        "vision_model.merger.proj.weight",
    ], f"a torre de visao nao atravessou com os nomes do carregador: {sorted(saida)}"
    assert any(_is_mtp_tensor(k) for k in saida), "e a cabeca tem que seguir junto"
    assert "language_model.lm_head.weight" in saida
    assert "language_model.model.layers.%d.input_layernorm.weight" % (n - 1) in saida


@precisa_do_config
def test_sem_a_opcao_o_caminho_de_antes_nao_muda():
    """Conserto cirurgico: quem nao pediu preservacao nao ve diferenca."""
    import json

    from omlx.oq import _build_model_sanitizer

    cfg = json.load(open(os.path.join(ORIGEM, "config.json")))
    n = (cfg.get("text_config") or {}).get("num_hidden_layers")
    limpeza = _build_model_sanitizer(cfg, model_path=ORIGEM, preserve_mtp=False)
    assert limpeza is not None
    saida = limpeza(_amostra_com_a_cabeca(n))
    assert not [k for k in saida if f"layers.{n}." in k or k.startswith("mtp.")], (
        "sem a opcao, o roteamento tem que continuar o de antes (a limpeza de "
        "visao, que descarta a cabeca) — senao o conserto deixou de ser cirurgico"
    )


@precisa_do_config
def test_a_limpeza_da_cabeca_sabotada_para_a_quantizacao(monkeypatch):
    """Sem a limpeza que preserva a cabeça, a de fábrica descarta toda chave
    `mtp.` e o resultado sai com o sufixo "-mtp" no nome e sem a cabeça dentro.

    O erro que avisa disso vivia dentro de um try cujo except só escreve em
    depuração (oq.py:4084 antes de 02/09), então a quantização seguia por
    horas e só a conferência final pegava. Agora ele interrompe na hora.
    """
    import json

    from omlx.oq import _build_model_sanitizer
    from omlx.patches.mlx_vlm_mtp import glm5_next_vlm_runtime

    monkeypatch.setattr(glm5_next_vlm_runtime, "apply_sanitize", lambda: False)
    cfg = json.load(open(os.path.join(ORIGEM, "config.json")))
    with pytest.raises(RuntimeError, match="cabeca de previsao multipla"):
        _build_model_sanitizer(cfg, model_path=ORIGEM, preserve_mtp=True)


@precisa_do_checkpoint
def test_a_limpeza_de_visao_quebrada_para_quando_a_origem_tem_visao(monkeypatch):
    """Cair no ramo de texto DESCARTA os pesos de visao — foi assim que 347
    tensores `model.visual.*` sumiram do GLM-5.3 em 31/08, com a linha do
    desvio escrita em nivel de depuracao. Com pesos de visao na origem, a
    falha da limpeza passa a interromper."""
    import json

    import mlx_vlm.utils as vlm_utils

    from omlx.oq import _build_model_sanitizer

    def _quebra(*a, **k):
        raise RuntimeError("get_model_and_args sabotado no teste")

    monkeypatch.setattr(vlm_utils, "get_model_and_args", _quebra)
    cfg = json.load(open(os.path.join(ORIGEM, "config.json")))
    with pytest.raises(RuntimeError, match="a origem traz pesos de visao"):
        _build_model_sanitizer(cfg, model_path=ORIGEM, preserve_mtp=False)


@precisa_do_checkpoint
def test_a_mesma_falha_com_text_only_segue_no_caminho_de_antes(monkeypatch):
    """Quem pediu so-texto ja abriu mao da visao: o desvio continua valendo."""
    import json

    import mlx_vlm.utils as vlm_utils

    from omlx.oq import _build_model_sanitizer

    def _quebra(*a, **k):
        raise RuntimeError("get_model_and_args sabotado no teste")

    monkeypatch.setattr(vlm_utils, "get_model_and_args", _quebra)
    cfg = json.load(open(os.path.join(ORIGEM, "config.json")))
    _build_model_sanitizer(cfg, model_path=ORIGEM, preserve_mtp=False, text_only=True)


@precisa_do_config
def test_sem_preservar_a_cabeca_a_limpeza_sabotada_nao_atrapalha(monkeypatch):
    """O conserto é cirúrgico: quem não pediu a cabeça não vê diferença."""
    import json

    from omlx.oq import _build_model_sanitizer
    from omlx.patches.mlx_vlm_mtp import glm5_next_vlm_runtime

    monkeypatch.setattr(glm5_next_vlm_runtime, "apply_sanitize", lambda: False)
    cfg = json.load(open(os.path.join(ORIGEM, "config.json")))
    assert _build_model_sanitizer(cfg, model_path=ORIGEM, preserve_mtp=False) is not None


@precisa_do_config
def test_a_lista_de_modulos_nao_quantizaveis_nao_sai_vazia_para_o_glm_5_3():
    """O mlx-lm só conhece o GLM-5.x depois do registro; sem ele a lista saía
    vazia com "Model type glm5_next not supported" em DEBUG, e o guarda não
    guardava nada. A instanciação é preguiçosa: nenhum peso é materializado."""
    from omlx.oq import _build_non_quantizable_set, universal_quant_predicate

    cfg = _config()
    lista = _build_non_quantizable_set(cfg)
    conv = {p for p in lista if p.endswith("self_attn.conv1d")}
    assert conv, "a lista não trouxe nenhum conv1d"
    assert "model.layers.0.self_attn.conv1d" in conv

    # No caminho de visão os nomes chegam com language_model. na frente; o
    # predicado tem que recusar os dois.
    cfg["_oq_non_quantizable"] = lista
    assert universal_quant_predicate("model.layers.0.self_attn.conv1d.weight", None, cfg, 2) is False
    assert universal_quant_predicate(
        "language_model.model.layers.0.self_attn.conv1d.weight", None, cfg, 2
    ) is False
