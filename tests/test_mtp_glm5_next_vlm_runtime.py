"""A previsão múltipla do GLM-5.3 no caminho de VISÃO — o que o servidor usa.

O servidor descobre o GLM-5.3 como ``vlm`` e o carrega pelo VLMBatchedEngine.
O runtime irmão (``mlx_lm_mtp/glm5_next_model.py``) atende o caminho de TEXTO,
por onde passam a quantização e o rascunhador — não o servidor. Sem este, o
portão abria e nada acontecia: o registro mostrava "Lightning MTP active"
seguido dos remendos de Qwen3.5, Gemma 4 e inkling, e de nenhum para o GLM-5.3.
"""

from __future__ import annotations

import json
import os

import pytest

ORIGEM = os.path.expanduser("~/.omlx/models/zai-org/GLM-5.3-Flash")


def _glm_com_runtime():
    from omlx.patches.mlx_vlm_mtp import glm5_next_vlm_runtime

    aplicou = glm5_next_vlm_runtime.apply()
    from mlx_vlm.models.glm5_next import language as glm_lang

    return glm_lang, aplicou


def _config():
    """O config de origem se estiver em disco; senão um sintético equivalente."""
    caminho = os.path.join(ORIGEM, "config.json")
    if os.path.isfile(caminho):
        return json.load(open(caminho, encoding="utf-8"))
    n = 8
    return {
        "model_type": "glm5_next",
        "text_config": {
            "model_type": "glm5_next",
            "num_hidden_layers": n,
            "num_nextn_predict_layers": 1,
            "hidden_size": 128,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "rms_norm_eps": 1e-5,
            "vocab_size": 512,
            "hc_mult": 4,
            "layer_types": [
                "linear_attention" if (i % 4) != 3 else "deepseek_sparse_attention"
                for i in range(n)
            ],
            "mlp_layer_types": ["dense"] * 3 + ["sparse"] * (n - 3),
            "first_k_dense_replace": 3,
        },
    }


def _args_enxutos(glm_lang, tronco=8):
    """TextConfig do config real, com as dimensões encolhidas para caber."""
    from mlx_vlm.models.glm5_next.config import TextConfig

    cfg = _config()
    texto = cfg.get("text_config", cfg)
    p = dict(texto)
    p["num_nextn_predict_layers"] = 1
    p["num_hidden_layers"] = tronco
    p["layer_types"] = list(texto["layer_types"])[:tronco]
    p["mlp_layer_types"] = list(texto["mlp_layer_types"])[:tronco]

    args = TextConfig.from_dict(p)
    args.hidden_size = 128
    args.num_attention_heads = 4
    args.n_routed_experts = 4
    args.moe_intermediate_size = 64
    args.intermediate_size = 64
    args.vocab_size = 512
    args.num_experts_per_tok = 2
    args.hc_mult = int(texto.get("hc_mult", 4) or 4)
    return args


def _modelo_com_cabeca(glm_lang):
    """Um LanguageModel de verdade, com a cabeça anexada e ligada."""
    from omlx.patches import mlx_lm_mtp, mlx_vlm_mtp

    args = _args_enxutos(glm_lang)
    antes_ativo = mlx_lm_mtp.is_mtp_active()
    antes_anexa = mlx_vlm_mtp.is_mtp_attach_enabled()
    mlx_lm_mtp.set_mtp_active(True)
    mlx_vlm_mtp.set_mtp_attach_enabled(True)
    try:
        modelo = glm_lang.LanguageModel(args)
    finally:
        mlx_lm_mtp.set_mtp_active(antes_ativo)
        mlx_vlm_mtp.set_mtp_attach_enabled(antes_anexa)
    return modelo, args


def test_o_runtime_de_visao_aplica():
    _glm, aplicou = _glm_com_runtime()
    assert aplicou is True, "o runtime de visão do glm5_next não aplicou"


def test_as_listas_de_tipo_cobrem_a_camada_da_cabeca_no_config_de_visao():
    """Sem isto, construir a camada da cabeça levanta IndexError.

    O ``TextConfig`` do lado de visão é outra classe que a do lado de texto, e
    tem o próprio ``from_dict`` — remendar um não remenda o outro.
    """
    glm_lang, _ = _glm_com_runtime()
    args = _args_enxutos(glm_lang)
    n = args.num_hidden_layers
    assert len(args.layer_types) >= n + 1
    assert len(args.mlp_layer_types) >= n + 1
    assert "linear" not in args.layer_types[n], (
        f"a camada da cabeça saiu {args.layer_types[n]!r}; o checkpoint tem "
        f"indexer e projeções q_a/kv_a nela, que só a esparsa monta"
    )


def test_a_cabeca_e_anexada_ao_modelo_de_visao():
    glm_lang, _ = _glm_com_runtime()
    modelo, args = _modelo_com_cabeca(glm_lang)
    assert hasattr(modelo, "mtp") and modelo.mtp, "a cabeça não foi anexada"
    assert modelo._omlx_mtp_decode_enabled is True
    bloco = modelo.mtp[0]
    for nome in ("enorm", "hnorm", "eh_proj", "norm", "block"):
        assert hasattr(bloco, nome), f"o bloco não tem {nome}"


def test_o_cache_da_cabeca_e_o_par_da_camada_esparsa():
    """Camada esparsa pede KV + acumulado de compressão, não um par de KV.

    As outras famílias devolvem ``[KVCache() ...]``; aqui isso entregaria à
    cabeça um cache de formato errado.
    """
    glm_lang, _ = _glm_com_runtime()
    modelo, _args = _modelo_com_cabeca(glm_lang)
    caches = modelo.make_mtp_cache()
    assert len(caches) == len(modelo.mtp)
    assert type(caches[0]).__name__ == "CacheList", (
        f"o cache da cabeça saiu {type(caches[0]).__name__}; a camada é esparsa "
        f"e pede o par KV + compressão"
    )


def test_o_tronco_devolve_o_hidden_que_a_cabeca_consome():
    """``return_hidden`` é o que liga o tronco ao ciclo de rascunho."""
    import mlx.core as mx

    glm_lang, _ = _glm_com_runtime()
    modelo, args = _modelo_com_cabeca(glm_lang)
    ids = mx.array([[3, 9, 17, 42]])

    saida = modelo(ids, return_hidden=True)
    mx.eval(saida.logits)
    assert saida.logits.shape == (1, 4, args.vocab_size)
    assert saida.hidden_states, "o tronco não devolveu o hidden"
    h = saida.hidden_states[0]
    assert h.shape == (1, 4, args.hidden_size)
    assert bool(mx.all(mx.isfinite(h)).item())

    # sem o pedido, a forma antiga tem que continuar valendo
    normal = modelo(ids)
    mx.eval(normal.logits)
    assert normal.logits.shape == (1, 4, args.vocab_size)


def test_o_ciclo_de_rascunho_roda_e_enxerga_o_historico():
    """O teste que fecha: preenchimento e depois rascunho de várias posições.

    O segundo regime é o que a previsão múltipla de fato usa, e é o único que
    pega a máscara saindo do par de cache em vez do KV de dentro — ela nasceria
    sem o histórico e a atenção estouraria contra um KV mais comprido.
    """
    import mlx.core as mx

    glm_lang, _ = _glm_com_runtime()
    modelo, args = _modelo_com_cabeca(glm_lang)

    ids = mx.array([[3, 9, 17, 42, 8, 1, 55, 2]])
    saida = modelo(ids, return_hidden=True)
    h = saida.hidden_states[0]
    cache = modelo.make_mtp_cache()

    logits = modelo.mtp_forward(h, ids, cache)
    mx.eval(logits)
    assert logits.shape == (1, 8, args.vocab_size)
    assert bool(mx.all(mx.isfinite(logits)).item())
    assert cache[0][0].offset == 8

    ids2 = mx.array([[11, 12, 13, 14]])
    h2 = mx.random.normal((1, 4, args.hidden_size)).astype(h.dtype)
    logits2 = modelo.mtp_forward(h2, ids2, cache)
    mx.eval(logits2)
    assert logits2.shape == (1, 4, args.vocab_size), (
        "rascunhar várias posições com histórico no cache tem que funcionar — "
        "é o regime encadeado da previsão múltipla"
    )
    assert bool(mx.all(mx.isfinite(logits2)).item())
    assert cache[0][0].offset == 12, (
        f"o cache ficou em {cache[0][0].offset}; 8 do preenchimento mais 4 do "
        f"rascunho são 12"
    )

    # e o encadeamento: devolver o hidden da própria cabeça para a volta seguinte
    logits3, h3 = modelo.mtp_forward(h2, ids2, cache, return_hidden=True)
    mx.eval(logits3, h3)
    assert h3.shape == h2.shape


def test_logits_keep_encolhe_a_projecao_de_vocabulario():
    """A projeção é o passo caro; o ciclo só precisa das últimas posições."""
    import mlx.core as mx

    glm_lang, _ = _glm_com_runtime()
    modelo, args = _modelo_com_cabeca(glm_lang)
    ids = mx.array([[3, 9, 17, 42, 8, 1]])
    saida = modelo(ids, return_hidden=True)
    cache = modelo.make_mtp_cache()

    logits = modelo.mtp_forward(saida.hidden_states[0], ids, cache, logits_keep=2)
    mx.eval(logits)
    assert logits.shape == (1, 2, args.vocab_size), (
        f"com logits_keep=2 a saída deveria ter 2 posições, tem {logits.shape[1]}"
    )


@pytest.mark.skipif(
    not os.path.exists(os.path.join(ORIGEM, "model.safetensors.index.json")),
    reason="o checkpoint de origem não está em disco",
)
def test_a_limpeza_de_visao_preserva_a_cabeca():
    """O ``sanitize`` do vendorado descarta toda chave com ``mtp.``.

    Ele nasceu para um caminho sem cabeça. As chaves dela têm que passar por
    fora, e os seis coeficientes de hiperconexão que o checkpoint não traz na
    camada da cabeça têm que ser completados com o valor de fábrica — senão a
    carga estrita morre com "Missing 6 parameters".
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten

    glm_lang, _ = _glm_com_runtime()
    modelo, args = _modelo_com_cabeca(glm_lang)

    entrada = {
        "mtp.0.eh_proj.weight": mx.zeros((args.hidden_size, 2 * args.hidden_size)),
        "mtp.0.enorm.weight": mx.ones((args.hidden_size,)),
        "mtp.0.hnorm.weight": mx.ones((args.hidden_size,)),
    }
    saida = modelo.sanitize(dict(entrada))

    for k in entrada:
        assert k in saida, f"a limpeza de visão descartou {k}"

    faltando = [
        f"mtp.0.{k}"
        for k, _ in tree_flatten(modelo.mtp[0].parameters())
        if "_hc." in k and f"mtp.0.{k}" not in saida
    ]
    assert not faltando, (
        f"a limpeza não completou {len(faltando)} coeficientes de hiperconexão: "
        f"{faltando}; a carga estrita morre neles"
    )


def test_o_desfazer_de_rascunho_recusado_nao_recusa():
    """Um rascunho RECUSADO tem que ter como voltar, ou o ciclo não rende.

    ``_restore_or_trim_caches`` desfaz restaurando ``rollback_state`` nas
    camadas que o têm e aparando as demais. As 34 camadas lineares do GLM-5.3
    não oferecem nem um nem outro — ``is_trimmable`` é False e nada escreve
    ``rollback_state`` neste caminho —, então sem armar o desfazer ele RECUSA e
    o ciclo inteiro cai no passo padrão: o trabalho da rodada vai fora toda vez
    que o rascunho erra, que é justamente quando o custo importa.

    Quem arma é o próprio ``__call__`` da verificação, e é assim que este teste
    exercita — chamar o armador à mão não cobriria a linha que o invoca.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    from omlx.patches.mlx_lm_mtp.batch_generator import _restore_or_trim_caches

    glm_lang, _ = _glm_com_runtime()
    modelo, _args = _modelo_com_cabeca(glm_lang)

    tronco = modelo.make_cache()
    lineares = [i for i, c in enumerate(tronco) if isinstance(c, ArraysCache)]
    assert lineares, "o tronco deveria ter camadas lineares"

    # cache virgem: o desfazer recusa, e é esse o defeito que o armar fecha
    assert _restore_or_trim_caches(list(modelo.make_cache())) is False, (
        "num tronco virgem o desfazer deveria recusar; se ele já aceita, o "
        "mecanismo mudou e este teste precisa ser revisto"
    )

    # primeira passada: deixa estado recorrente de verdade no cache
    modelo(mx.array([[3, 9, 17, 42]]), cache=tronco, return_hidden=True)
    marco = [(tronco[i][0], tronco[i][1]) for i in lineares]

    # segunda passada: o __call__ arma o desfazer com o estado do marco acima,
    # e depois a camada o substitui
    modelo(mx.array([[8, 1]]), cache=tronco, return_hidden=True)
    mudou = any(
        not bool(mx.array_equal(marco[k][1], tronco[i][1]).item())
        for k, i in enumerate(lineares)
        if marco[k][1] is not None and tronco[i][1] is not None
    )
    assert mudou, "a segunda passada deveria ter mexido no estado recorrente"

    # rascunho recusado
    assert _restore_or_trim_caches(list(tronco)) is True, (
        "com o desfazer armado pela verificação, a rejeição tem que ser "
        "desfeita em vez de derrubar o ciclo para o passo padrão"
    )

    for k, i in enumerate(lineares):
        for eixo, nome in ((0, "convolução"), (1, "recorrente")):
            a, d = marco[k][eixo], tronco[i][eixo]
            if a is None:
                assert d is None, f"camada {i}: {nome} nasceu do nada"
                continue
            assert d is not None, f"camada {i}: {nome} sumiu no desfazer"
            assert bool(mx.array_equal(a, d).item()), (
                f"camada {i}: o {nome} não voltou ao estado de antes da "
                f"verificação"
            )


def test_armar_o_desfazer_nao_toca_no_cache_da_camada_esparsa():
    """A esparsa já é apáravel; escrever nela seria mexer no que funciona."""
    from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

    from omlx.patches.mlx_vlm_mtp.glm5_next_vlm_runtime import _arma_desfazer

    esparsa = CacheList(KVCache(), KVCache())
    linear = ArraysCache(size=2)
    _arma_desfazer([linear, esparsa])

    assert getattr(esparsa, "rollback_state", None) is None, (
        "a camada esparsa não deveria receber rollback_state: ela é apáravel "
        "e o desfazer já sabe lidar com ela"
    )
