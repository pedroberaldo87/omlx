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
@precisa_do_checkpoint
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
    """O conserto de 31/08: rotear para a limpeza que preserva.

    A limpeza de visao descarta a cabeca — esta escrito na propria descricao de
    ``_build_model_sanitizer``. O GLM-5.x e uma familia SO TEXTO implementada em
    mlx-vlm, entao cai nela sem ter peso de visao a proteger. Com a opcao ligada,
    passa a usar a limpeza de texto, que preserva.
    """
    import json

    from omlx.oq import _build_model_sanitizer

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
    sobrou = [k for k in saida if f"layers.{n}." in k or k.startswith("mtp.")]
    assert len(sobrou) == 3, (
        f"com preserve_mtp ligado a cabeca tem que atravessar inteira; sobraram "
        f"{len(sobrou)} de 3: {sorted(sobrou)}. Sem isso a quantizacao grava um "
        f"modelo com o sufixo '-mtp' no nome e sem a cabeca dentro."
    )


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
