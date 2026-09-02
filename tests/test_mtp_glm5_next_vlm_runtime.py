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

    Este teste cobre só o ARMAR. O ciclo encadeado não usa mais este desfazer:
    ele chama ``mtp_partial_rollback`` (replay das posições aceitas), e a
    prova de alinhamento está em
    ``test_o_desfazer_parcial_de_visao_reprocessa_as_posicoes_aceitas``.
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


def test_a_limpeza_de_visao_renomeia_a_camada_crua_para_o_prefixo_da_cabeca():
    """O checkpoint guarda a cabeça como camada 45; o bloco mora em ``mtp.0``.

    Sem o renomeio os pesos chegam com o nome de uma camada que o modelo não
    tem, e a carga estrita os recusa. O renomeador é o MESMO do caminho de
    texto, de propósito: duas cópias divergiram antes — uma casava prefixo
    exato, a outra usava expressão regular mais permissiva — e nada apontava
    isso, porque nenhum teste cobria o renomeio deste lado.
    """
    import mlx.core as mx

    glm_lang, _ = _glm_com_runtime()
    modelo, args = _modelo_com_cabeca(glm_lang)
    n = args.num_hidden_layers

    entrada = {
        f"model.language_model.layers.{n}.eh_proj.weight": mx.zeros((4, 8)),
        f"model.language_model.layers.{n}.enorm.weight": mx.ones((4,)),
        f"model.language_model.layers.{n}.hnorm.weight": mx.ones((4,)),
        f"model.language_model.layers.{n}.shared_head.norm.weight": mx.ones((4,)),
        f"model.language_model.layers.{n}.input_layernorm.weight": mx.ones((4,)),
    }
    saida = modelo.sanitize(dict(entrada))

    # o trio da cabeça fica na raiz do bloco
    for nome in ("eh_proj.weight", "enorm.weight", "hnorm.weight"):
        assert f"mtp.0.{nome}" in saida, f"faltou mtp.0.{nome} em {sorted(saida)[:8]}"
    # a norma final tem nome próprio no checkpoint e vira a `norm` do bloco
    assert "mtp.0.norm.weight" in saida, (
        "shared_head.norm deveria virar a norm do bloco"
    )
    # o resto é a camada decoder, que vive sob `block.`
    assert "mtp.0.block.input_layernorm.weight" in saida
    # e nada sobrou com o nome da camada crua
    assert not [k for k in saida if f"layers.{n}." in k], (
        f"sobrou chave com o nome da camada crua: "
        f"{[k for k in saida if f'layers.{n}.' in k]}"
    )


def test_a_limpeza_levanta_a_contagem_para_a_camada_da_cabeca_nao_ser_descartada():
    """A limpeza de baixo joga fora toda camada de índice >= num_hidden_layers.

    Está em ``glm_moe_dsa/deepseek_v32.py`` (``parts[1] == "layers" and
    int(parts[2]) >= mpt_layer``), e a camada da cabeça é exatamente a primeira
    além do fim. Levantar a contagem durante a chamada faz cada transformação da
    camada comum valer igual para a dela; sem isso ela é descartada antes de
    qualquer renomeio.

    O filtro só casa o formato ``<raiz>.layers.N.*`` — com o prefixo de
    linguagem no meio ele não pega, e foi por isso que o primeiro teste do
    renomeio não cobria esta linha.
    """
    import mlx.core as mx

    glm_lang, _ = _glm_com_runtime()
    modelo, args = _modelo_com_cabeca(glm_lang)
    n = args.num_hidden_layers

    entrada = {
        f"model.layers.{n}.eh_proj.weight": mx.zeros((4, 8)),
        f"model.layers.{n}.enorm.weight": mx.ones((4,)),
        f"model.layers.{n}.hnorm.weight": mx.ones((4,)),
    }
    saida = modelo.sanitize(dict(entrada))

    for nome in ("eh_proj.weight", "enorm.weight", "hnorm.weight"):
        assert f"mtp.0.{nome}" in saida, (
            f"mtp.0.{nome} não sobreviveu: a camada da cabeça foi descartada "
            f"pelo filtro de índice antes de chegar ao renomeio. "
            f"Saíram: {sorted(saida)[:6]}"
        )
    # e a contagem tem que voltar ao que era, ou o modelo passa a mentir o
    # próprio tamanho para todo o resto do processo
    assert modelo.args.num_hidden_layers == n, (
        f"a contagem ficou em {modelo.args.num_hidden_layers} em vez de voltar "
        f"para {n}"
    )


def test_o_tronco_aceita_os_dois_jeitos_de_receber_os_tokens():
    """O motor chama ora por posição, ora por ``input_ids`` no dicionário.

    O ``__call__`` de fábrica trata os dois; o nosso, que o envolve para o
    ciclo de rascunho, tem que tratar igual — senão a verificação quebra
    exatamente na chamada que o motor faz.
    """
    import mlx.core as mx

    glm_lang, _ = _glm_com_runtime()
    modelo, args = _modelo_com_cabeca(glm_lang)
    ids = mx.array([[3, 9, 17, 42]])

    por_posicao = modelo(ids, return_hidden=True)
    por_nome = modelo(None, return_hidden=True, input_ids=ids)
    mx.eval(por_posicao.logits, por_nome.logits)

    assert por_nome.logits.shape == por_posicao.logits.shape
    assert por_nome.hidden_states, "o hidden sumiu quando os tokens vêm por nome"


def test_o_tronco_encolhe_a_projecao_quando_o_motor_pede():
    """``num_logits_to_keep`` existe para pular a projeção de vocabulário.

    Ela é o passo caro, e nas posições descartadas do preenchimento o resultado
    não é usado. O ``__call__`` de fábrica corta antes de projetar; o nosso
    tinha que manter isso, ou o preenchimento paga a projeção inteira.
    """
    import mlx.core as mx

    glm_lang, _ = _glm_com_runtime()
    modelo, args = _modelo_com_cabeca(glm_lang)
    ids = mx.array([[3, 9, 17, 42, 8, 1]])

    saida = modelo(ids, return_hidden=True, num_logits_to_keep=2)
    mx.eval(saida.logits)
    assert saida.logits.shape == (1, 2, args.vocab_size), (
        f"com num_logits_to_keep=2 deveriam sair 2 posições, saíram "
        f"{saida.logits.shape[1]}"
    )
    # o hidden NÃO é cortado: a cabeça consome todas as posições
    assert saida.hidden_states[0].shape == (1, 6, args.hidden_size), (
        "o hidden foi cortado junto; a cabeça precisa dele inteiro"
    )


def test_a_cabeca_roda_sem_cache_nenhum():
    """Uma primeira volta pode chegar sem cache; não pode explodir."""
    import mlx.core as mx

    glm_lang, _ = _glm_com_runtime()
    modelo, args = _modelo_com_cabeca(glm_lang)
    ids = mx.array([[3, 9, 17, 42]])
    saida = modelo(ids, return_hidden=True)

    logits = modelo.mtp_forward(saida.hidden_states[0], ids, None)
    mx.eval(logits)
    assert logits.shape == (1, 4, args.vocab_size)
    assert bool(mx.all(mx.isfinite(logits)).item())


def test_sem_cabeca_o_cache_dela_e_vazio_e_nao_levanta():
    """Modelo carregado sem previsão múltipla ainda responde ao motor.

    O motor chama ``make_mtp_cache`` antes de saber se o ciclo vale; devolver
    lista vazia é o contrato, levantar seria derrubar a requisição.
    """
    from omlx.patches import mlx_lm_mtp, mlx_vlm_mtp

    glm_lang, _ = _glm_com_runtime()
    args = _args_enxutos(glm_lang)
    antes_ativo = mlx_lm_mtp.is_mtp_active()
    antes_anexa = mlx_vlm_mtp.is_mtp_attach_enabled()
    mlx_lm_mtp.set_mtp_active(False)
    mlx_vlm_mtp.set_mtp_attach_enabled(False)
    try:
        sem_cabeca = glm_lang.LanguageModel(args)
    finally:
        mlx_lm_mtp.set_mtp_active(antes_ativo)
        mlx_vlm_mtp.set_mtp_attach_enabled(antes_anexa)

    assert not hasattr(sem_cabeca, "mtp")
    assert sem_cabeca._omlx_mtp_decode_enabled is False
    assert sem_cabeca.make_mtp_cache() == []


def test_a_cabeca_recebe_o_hidden_ANTES_da_norma_final():
    """A cabeça consome o hidden PRÉ-norma, e o `hnorm` dela normaliza.

    O tronco termina com `return self.norm(h)`, e o bloco da cabeça começa com
    `eh_proj([enorm(embedding), hnorm(h)])`. Entregar o pós-norma faz o dado
    passar por DUAS normalizações, cada uma com peso próprio e diferente — a
    cabeça recebe uma escala que nunca viu no treino.

    O efeito medido não é rascunho ruim, é rascunho SEMPRE recusado: a tela do
    agente mostrou `draft share 0%` e `0 draft tok`, com a geração em 1,6 tokens
    por segundo contra 18,8 sem a cabeça.

    Como este teste prova que é o pré-norma: aplicar a norma final ao hidden
    devolvido tem que reproduzir o que gera os logits. Se o hidden já viesse
    normalizado, aplicar a norma de novo mudaria o valor.
    """
    import mlx.core as mx

    glm_lang, _ = _glm_com_runtime()
    modelo, args = _modelo_com_cabeca(glm_lang)
    ids = mx.array([[3, 9, 17, 42]])

    saida = modelo(ids, return_hidden=True)
    h = saida.hidden_states[0]
    mx.eval(h, saida.logits)

    # o que o tronco faria com o pré-norma para chegar aos logits
    pos = modelo.model.norm(h)
    from mlx_vlm.models.glm5_next.language import linear_forward

    esperado = (
        modelo.model.embed_tokens.as_linear(pos)
        if args.tie_word_embeddings
        else linear_forward(modelo.lm_head, pos)
    )
    mx.eval(esperado)

    assert bool(mx.allclose(esperado, saida.logits, atol=1e-2).item()), (
        "aplicar a norma final ao hidden devolvido não reproduz os logits; "
        "o hidden entregue à cabeça não é o pré-norma"
    )

    # e o pós-norma tem que ser DIFERENTE do que foi entregue — se fossem
    # iguais, a norma não estaria fazendo nada e o teste não provaria nada
    assert not bool(mx.allclose(h, pos, atol=1e-3).item()), (
        "o hidden entregue é igual ao pós-norma; ou a norma é identidade "
        "(e este teste não prova nada), ou ainda estamos entregando o errado"
    )


def test_o_desfazer_parcial_de_visao_reprocessa_as_posicoes_aceitas():
    """No caminho de VISÃO o ciclo caía em ``_restore_or_trim_caches``: as
    recorrentes voltavam para ANTES do bloco e as esparsas ficavam com a
    confirmada — o mesmo desalinhamento que no texto produzia token repetido.
    Agora o modelo de visão expõe ``mtp_partial_rollback`` com o replay do
    irmão, e a prova é a mesma: os logits do passo seguinte têm que coincidir
    com os de um tronco que só viu as posições aceitas.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    glm_lang, _ = _glm_com_runtime()
    modelo, _args = _modelo_com_cabeca(glm_lang)

    def _logits(cache, proximo):
        saida = modelo(mx.array([[proximo]]), cache=cache)
        saida = getattr(saida, "logits", saida)
        mx.eval(saida)
        return saida

    prefixo = mx.array([[3, 9, 17, 42, 8]])
    bloco = mx.array([[1, 55, 2, 7]])

    ref = modelo.make_cache()
    modelo(prefixo, cache=ref)
    modelo(bloco[:, :2], cache=ref)
    esperado = _logits(ref, 99)

    cache = modelo.make_cache()
    modelo(prefixo, cache=cache)
    modelo(bloco, cache=cache, return_hidden=True)
    assert all(
        c.rollback_replay is not None for c in cache if isinstance(c, ArraysCache)
    ), "a verificação armada tem que deixar o replay guardado"
    assert modelo.mtp_partial_rollback(cache, 1, 3) is True
    obtido = _logits(cache, 99)
    dif = float(mx.max(mx.abs(obtido - esperado)).item())
    assert dif < 1e-3, f"tronco de visão desfeito diverge da referência em {dif:.2e}"


def test_o_desfazer_da_cadeia_acha_o_metodo_atras_do_adaptador_de_visao():
    """No servidor o lote não vê o modelo de língua: vê o ``VLMModelAdapter``,
    que não repassa ``mtp_partial_rollback``. O desfazer da cadeia procurava só
    no embrulho, não achava, e TODA recusa caía em "cache layer rejects chain
    rollback" + reconcile — medido no Vontra pela visão: 2 tok/s, MTP com
    cycles=0. Agora ele olha também o modelo interno.
    """
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    chamadas = []

    class _Lingua:
        def mtp_partial_rollback(self, cache, accepted, num_drafts):
            chamadas.append((accepted, num_drafts))
            return True

    class _Adaptador:
        def __init__(self):
            self._language_model = _Lingua()

    assert bg._chain_rollback(_Adaptador(), [], 1, 3, None) is True
    assert chamadas == [(1, 3)]


def test_a_camada_linear_compilada_em_t1_da_o_mesmo_resultado_do_caminho_comum():
    """No decode (T=1) a camada linear roda hiperconexão + norma + atenção +
    expansão num grafo compilado, com os dois estados entrando e saindo como
    arrays (o caminho comum escreve no cache dentro do forward). Os logits e os
    estados têm que coincidir com o caminho comum, token a token.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    glm_lang, _ = _glm_com_runtime()
    modelo, _args = _modelo_com_cabeca(glm_lang)
    camadas = modelo.model.layers
    assert any(c.is_linear for c in camadas)

    def roda(compilar):
        for c in camadas:
            c.compile_ffn = compilar
            c._attn_c = None
            c._ffn_c = None
        cache = modelo.make_cache()
        modelo(mx.array([[3, 9, 17, 42, 8]]), cache=cache)
        saidas = []
        for tok in (1, 55, 2):
            s = modelo(mx.array([[tok]]), cache=cache)
            s = getattr(s, "logits", s)
            mx.eval(s)
            saidas.append(s)
        estados = [c.state for c in cache if isinstance(c, ArraysCache)]
        return saidas, estados

    ref, est_ref = roda(False)
    obt, est_obt = roda(True)
    assert all(c._attn_c is not None for c in camadas if c.is_linear), (
        "o caminho compilado da atenção não foi usado"
    )
    for a, b in zip(ref, obt):
        dif = float(mx.max(mx.abs(a - b)).item())
        assert dif < 1e-2, f"logits divergem entre compilado e comum: {dif:.2e}"
    for (c0, s0), (c1, s1) in zip(est_ref, est_obt):
        assert float(mx.max(mx.abs(c0 - c1)).item()) < 1e-2
        assert float(mx.max(mx.abs(s0 - s1)).item()) < 1e-2
