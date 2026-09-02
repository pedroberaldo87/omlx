"""Previsão múltipla do GLM-5.3 no caminho de VISÃO.

O GLM-5.3 é uma família só-texto implementada em mlx-vlm: o servidor a descobre
como ``vlm`` (``VLM_NATIVE_TEXT_MODEL_TYPES`` em ``omlx/model_discovery.py``) e a
carrega pelo ``VLMBatchedEngine``. O runtime irmão
(``omlx/patches/mlx_lm_mtp/glm5_next_model.py``) atende o caminho de TEXTO, que é
por onde a quantização e o rascunhador passam — não este.

Sem este módulo o portão abre e nada acontece: o registro mostrava
``Speculative backend selected: Lightning MTP (model_type=glm5_next, active)``
seguido dos remendos de Qwen3.5, Gemma 4 e inkling, e de nenhum para o GLM-5.3.
A cabeça ficava fora do ciclo de rascunho.

**O que é COMPARTILHADO com o irmão, e por quê.** A camada decoder vem do modelo
vendorado nos dois caminhos, então o bloco da cabeça é literalmente o mesmo
objeto: ele nasce em ``fabrica_bloco_da_cabeca()`` e é servido aos dois, em vez
de existirem duas cópias que envelhecem em separado. O mesmo vale para o tipo da
camada da cabeça, a extensão das listas, o cache que ela pede e o preenchimento
da hiperconexão — cada um desses foi um defeito medido contra o checkpoint real,
e duplicá-los seria pedir para consertar cada um duas vezes.

**O que é PRÓPRIO daqui:** o modelo de visão expõe ``LanguageModel``, não
``Model``; o ``__call__`` devolve ``LanguageModelOutput`` e precisa de um
``return_hidden`` que entregue o hidden que a cabeça consome; e o ``sanitize``
do vendorado descarta toda chave que contenha ``mtp.``, de modo que as da cabeça
têm de passar por fora dele.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_APPLIED = False


def _vendored():
    """O módulo de linguagem do GLM-5.3 no mlx-vlm, já remendado."""
    from ..mlx_vlm_glm5_next_compat import apply_mlx_vlm_glm5_next_compat_patch

    apply_mlx_vlm_glm5_next_compat_patch()
    from mlx_vlm.models.glm5_next import language as glm_lang

    return glm_lang


def apply() -> bool:
    """Instala o runtime. Idempotente — o carregamento chama a cada modelo."""
    global _APPLIED
    if _APPLIED:
        return True

    try:
        glm_lang = _vendored()
    except Exception as e:
        logger.debug("GLM-5.3 VLM runtime MTP patch skipped: %s", e)
        return False

    try:
        _patch_text_config(glm_lang)
        _patch_vlm_language_model(glm_lang)
        _patch_vlm_sanitize(glm_lang)
    except Exception as e:
        logger.warning("GLM-5.3 VLM runtime MTP patch failed: %s", e)
        return False

    _APPLIED = True
    logger.info("mlx-vlm GLM-5.3 (glm5_next) runtime MTP patch applied")
    return True


def apply_sanitize() -> bool:
    """Só a limpeza de nomes que preserva a cabeça (e o TextConfig que a
    sustenta), sem o runtime de decodificação.

    É o que a QUANTIZAÇÃO precisa: o oQ limpa os nomes pelo caminho de visão
    para manter a torre de visão, e a limpeza de fábrica desse caminho
    descarta a cabeça. Aplicar o runtime inteiro ali embrulharia ``__init__``
    e ``__call__`` de uma classe que a quantização nunca instancia.
    """
    try:
        glm_lang = _vendored()
        _patch_text_config(glm_lang)
        _patch_vlm_sanitize(glm_lang)
    except Exception as e:
        logger.debug("GLM-5.3 VLM MTP sanitize patch skipped: %s", e)
        return False
    return True


# ---------------------------------------------------------------------------
# TextConfig — as listas de tipo têm que cobrir a camada da cabeça.
# ---------------------------------------------------------------------------


def _patch_text_config(glm_lang: Any) -> None:
    """Estende ``layer_types`` e ``mlp_layer_types`` para a camada extra.

    As duas listas têm exatamente ``num_hidden_layers`` entradas, e construir a
    camada da cabeça com elas dá ``IndexError``. A regra de qual tipo ela assume
    é a mesma do caminho de texto — atenção esparsa, não a linear que a última
    camada comum usa —, então vem de lá.
    """
    from ..mlx_lm_mtp.glm5_next_model import _estende, _tipo_da_camada_da_cabeca

    from mlx_vlm.models.glm5_next.config import TextConfig

    if getattr(TextConfig, "_omlx_mtp_from_dict_patched", False):
        return

    original = TextConfig.from_dict.__func__

    def from_dict_remendado(cls, params):
        args = original(cls, params)
        texto = params.get("text_config")
        fonte = texto if isinstance(texto, dict) else params
        n_mtp = int(fonte.get("num_nextn_predict_layers", 0) or 0)
        args.num_nextn_predict_layers = n_mtp

        if n_mtp > 0:
            n_main = int(args.num_hidden_layers)
            alvo = n_main + n_mtp

            estendida = _estende(
                getattr(args, "layer_types", None),
                alvo,
                lambda atual: _tipo_da_camada_da_cabeca(atual[:n_main]),
            )
            if estendida is not None:
                args.layer_types = estendida

            mlp = _estende(
                getattr(args, "mlp_layer_types", None),
                alvo,
                lambda atual: atual[n_main - 1] if n_main else "dense",
            )
            if mlp is not None:
                args.mlp_layer_types = mlp

        return args

    TextConfig.from_dict = classmethod(from_dict_remendado)
    TextConfig._omlx_mtp_from_dict_patched = True


# ---------------------------------------------------------------------------
# LanguageModel — anexar a cabeça e expor o ciclo de rascunho.
# ---------------------------------------------------------------------------


def _patch_vlm_language_model(glm_lang: Any) -> None:
    cls = glm_lang.LanguageModel
    if "_omlx_mtp_runtime_patched" in cls.__dict__:
        return

    from ..mlx_lm_mtp.glm5_next_model import _cache_para, fabrica_bloco_da_cabeca

    Glm5NextMTPBlock = fabrica_bloco_da_cabeca()

    original_init = cls.__init__
    original_call = cls.__call__

    def __init__(self, args, config=None):
        from . import is_mtp_attach_enabled
        from ..mlx_lm_mtp import get_mtp_depth, is_mtp_active

        original_init(self, args, config)

        n_mtp = int(getattr(args, "num_nextn_predict_layers", 0) or 0)
        anexar = bool(is_mtp_attach_enabled())
        # Anexar e DECODIFICAR são coisas separadas: o bloco precisa existir
        # para o carregamento estrito achar destino para os pesos da camada da
        # cabeça, mesmo com a previsão múltipla desligada no ciclo.
        self._omlx_mtp_decode_enabled = bool(
            n_mtp > 0 and anexar and is_mtp_active()
        )
        if n_mtp > 0 and anexar:
            n_main = int(args.num_hidden_layers)
            self.mtp = [
                Glm5NextMTPBlock(args, n_main + i) for i in range(n_mtp)
            ]
        if self._omlx_mtp_decode_enabled:
            self._omlx_mtp_chain = True
            self._omlx_mtp_depth = get_mtp_depth()
            # A cadeia de rascunhos roda numa CÓPIA do cache da cabeça, por
            # ciclo: o cache persistente só recebe os tokens confirmados. Sem
            # a cópia, os rascunhos recusados ficavam para sempre no cache da
            # cabeça (CacheList(KVCache, PoolingCache) não expõe `offset`, e o
            # aparador lia 0) — medido em 02/09: 7 tokens confirmados, 12 no
            # KV da cabeça — e a cabeça passava a rascunhar sobre um histórico
            # de rascunhos recusados, a aceitação caía e o controlador
            # estacionava (11 vezes numa jornada de agente).
            self._omlx_mtp_head_clone = True
            # Custo de UMA POSIÇÃO A MAIS na janela de verificação do TRONCO, em
            # milissegundos — é assim que o controlador consome este valor
            # (`_DepthController.MARGINAL_MS`: "one extra verify token's cost"),
            # até ter medida própria da inclinação entre profundidades.
            #
            # MEDIDO em 01/09 no servidor: cada linha a mais no forward custa
            # 19–23 ms (decode em lote B=1/2/4: 48,2 / 71,6 / 110,8 ms por passo;
            # verify com barreira: +22 ms por posição). O valor anterior (2,5)
            # era o custo da CABEÇA por rascunho — unidade errada, 8x baixo — e
            # o de antes dele (35) o de outro modelo. Como o prior só governa o
            # aquecimento, o efeito é nas corridas curtas.
            self._omlx_mtp_marginal_ms = 21.0

    def __call__(self, inputs=None, inputs_embeds=None, cache=None, mask=None,
                 **kwargs):
        """O tronco, com a forma extra que o ciclo de rascunho pede.

        Com ``return_hidden``, devolve também o hidden que a cabeça consome. No
        GLM-5.3 esse hidden é o PÓS-norma final — é o que sai de
        ``Glm5NextModel.__call__`` antes da projeção de vocabulário —, e o
        ``hnorm`` de dentro do bloco normaliza de novo. O irmão de texto usa
        exatamente o mesmo ponto.
        """
        return_hidden = kwargs.pop("return_hidden", False)
        kwargs.pop("n_confirmed", None)
        if not return_hidden:
            return original_call(self, inputs, inputs_embeds, cache, mask, **kwargs)

        # Antes de rodar a verificação, guardar de onde o estado recorrente
        # veio. Sem isto o desfazer de um rascunho RECUSADO não tem como
        # voltar: as 34 camadas lineares do tronco não são apáraveis
        # (``is_trimmable`` é False) e não trazem ``rollback_state``, então o
        # desfazer recusa e o ciclo inteiro cai no passo padrão — medido: o
        # tronco de 45 camadas devolvia False.
        #
        # Custa quase nada: os arrays do MLX são imutáveis e a camada os
        # SUBSTITUI em vez de mutar, então guardar a referência anterior não
        # copia dado nenhum.
        _arma_desfazer(cache)

        from mlx_vlm.models.base import LanguageModelOutput
        from mlx_vlm.models.glm5_next.language import linear_forward

        if inputs is None:
            inputs = kwargs.get("input_ids")
        # O tronco termina com `self.norm(h)`, e o bloco da cabeça começa com
        # `hnorm`. Entregar o pós-norma faz o dado passar por DUAS
        # normalizações, cada uma com peso próprio — a cabeça recebe uma escala
        # que nunca viu no treino, e o rascunho dela é sempre recusado.
        # Medido: draft share 0%, e 1,6 tokens por segundo contra 18,8 sem ela.
        h_pre = _tronco_ate_a_norma(self.model, inputs, cache, inputs_embeds)
        h = self.model.norm(h_pre)
        fonte = h
        nlk = kwargs.get("num_logits_to_keep", 0)
        if nlk:
            fonte = fonte[:, -nlk:, :]
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(fonte)
        else:
            logits = linear_forward(self.lm_head, fonte)
        # o hidden que sai é o PRÉ-norma, que é o que a cabeça consome
        return LanguageModelOutput(logits=logits, hidden_states=[h_pre])

    def mtp_forward(self, hidden_states, next_token_ids, mtp_cache,
                    return_hidden: bool = False, logits_keep: int = 0):
        """A cabeça, a norma dela e o ``lm_head`` compartilhado.

        A máscara sai do KV de DENTRO do par de cache da camada esparsa, não do
        par inteiro — quem conta posições é o KV. Passar o par faz a máscara
        nascer sem o histórico, o que só aparece ao rascunhar várias posições
        encadeadas (é o mesmo defeito que o irmão de texto já pagou).
        """
        import mlx.core as mx  # noqa: F401
        from mlx_lm.models.base import create_attention_mask
        from mlx_vlm.models.glm5_next.language import linear_forward

        if mtp_cache is None:
            mtp_cache = [None] * len(self.mtp)

        conta_posicoes = mtp_cache[0]
        if conta_posicoes is not None and not hasattr(conta_posicoes, "offset"):
            conta_posicoes = conta_posicoes[0]
        mask = create_attention_mask(hidden_states, conta_posicoes, return_array=True)

        h = hidden_states
        ultimo = None
        for i, bloco in enumerate(self.mtp):
            h = bloco(h, self.model.embed_tokens, next_token_ids, mask, mtp_cache[i])
            ultimo = bloco

        fonte = h
        if logits_keep and fonte.shape[1] > logits_keep:
            fonte = fonte[:, -logits_keep:]
        saida = ultimo.norm(fonte)
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(saida)
        else:
            logits = linear_forward(self.lm_head, saida)
        if return_hidden:
            return logits, h
        return logits

    def make_mtp_cache(self):
        """O cache no formato que o TIPO da camada da cabeça pede.

        Ela é esparsa, então pede o par KV + acumulado de compressão do seletor,
        e não o par de KV que as outras famílias assumem.
        """
        if not hasattr(self, "mtp"):
            return []
        return [_cache_para(bloco.block) for bloco in self.mtp]

    def mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
        """Desfaz a janela de verificação até ``accepted`` rascunhos.

        Sem este método o ciclo caía em ``_restore_or_trim_caches``, que
        RESTAURA as recorrentes ao ponto anterior ao bloco e APARA as esparsas
        deixando a confirmada — as duas famílias ficavam em posições
        diferentes, o mesmo desalinhamento que no caminho de texto produzia
        token repetido. A regra e o replay são os do irmão.
        """
        from ..mlx_lm_mtp.glm5_next_model import desfaz_parcial

        return desfaz_parcial(cache, accepted, num_drafts)

    cls.__init__ = __init__
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.mtp_partial_rollback = mtp_partial_rollback
    cls._omlx_mtp_runtime_patched = True


def _tronco_ate_a_norma(modelo, inputs, cache, inputs_embeds):
    """Roda o tronco e para ANTES da norma final, devolvendo o hidden cru.

    É uma cópia do laço de ``Glm5NextModel.__call__`` sem a última linha. Não dá
    para pedir isso ao tronco: ele devolve só o resultado já normalizado, e a
    normalização não se desfaz.

    Quem consome é a cabeça de previsão múltipla, cujo ``hnorm`` normaliza este
    valor com o peso próprio dela.
    """
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask
    from mlx_vlm.models.glm5_next.language import _mascara_da_recorrente

    h = modelo.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
    if cache is None:
        cache = [None] * len(modelo.layers)

    fa_cache = cache[modelo.fa_idx]
    fa_mask = create_attention_mask(
        h, fa_cache[0] if fa_cache else None, return_array=True
    )
    ssm_mask = _mascara_da_recorrente(h, cache[modelo.ssm_idx])

    h = mx.contiguous(
        mx.broadcast_to(
            h[:, :, None, :],
            (h.shape[0], h.shape[1], modelo.hc_mult, h.shape[2]),
        )
    )
    for camada, c in zip(modelo.layers, cache):
        mask = ssm_mask if camada.is_linear else fa_mask
        h = camada(h, mask=mask, cache=c)
    return h.mean(axis=2)


def _arma_desfazer(cache) -> None:
    """Guarda o estado recorrente de cada camada linear antes da verificação.

    ``_restore_or_trim_caches`` (em ``mlx_lm_mtp/batch_generator.py``) desfaz
    uma rejeição restaurando ``cache.rollback_state`` nas camadas que o têm e
    aparando as demais. As camadas lineares do GLM-5.3 não oferecem nem um nem
    outro, e sem este par o desfazer recusa e o ciclo perde a rodada inteira.

    O par é ``(convolução, recorrente)`` — os dois arrays que a camada linear
    guarda e que só fazem sentido juntos.
    """
    if not cache:
        return
    for c in cache:
        if c is None:
            continue
        # SEMPRE sobrescreve. O par tem que ser o estado imediatamente antes
        # DESTA verificação: quando o rascunho é aceito ninguém consome o par
        # anterior, e mantê-lo faria a verificação seguinte voltar para o
        # estado de duas rodadas atrás. Neste caminho nenhum outro mecanismo
        # escreve o campo — no de texto quem escreve é a camada, e lá este
        # código não roda.
        # O critério é o que o desfazer SABE FAZER, não o tipo da camada: se
        # ela já é apárável, ele a trata sozinho e escrever aqui seria mexer no
        # que funciona. Sobram exatamente as lineares.
        if hasattr(c, "is_trimmable") and c.is_trimmable():
            continue
        try:
            anterior, recorrente = c[0], c[1]
        except (TypeError, IndexError, KeyError):
            continue
        try:
            c.rollback_state = (anterior, recorrente)
        except AttributeError:
            # cache que não aceita o atributo: o desfazer o trata pelo aparo
            continue
        # A camada guarda, neste forward, o que o desfazer PARCIAL precisa
        # para reprocessar as posições aceitas (`desfaz_parcial`, no irmão).
        c.rollback_replay = None
        c._omlx_captura_desfazer = True


# ---------------------------------------------------------------------------
# sanitize — a cabeça não pode cair no filtro do vendorado.
# ---------------------------------------------------------------------------


def _patch_vlm_sanitize(glm_lang: Any) -> None:
    """Preserva as chaves da cabeça e completa o que o checkpoint não traz.

    O ``sanitize`` do modelo vendorado começa descartando toda chave que
    contenha ``mtp.`` — ele nasceu para um caminho sem cabeça. Aqui elas passam
    por fora dele e voltam depois.

    E a camada da cabeça é a única sem os seis coeficientes de hiperconexão no
    disco: as comuns trazem os seis, ela não traz nenhum. Não é peso perdido, a
    referência os deixa no valor de fábrica — que é o neutro. Sem completá-los,
    a carga estrita morre com "Missing 6 parameters".
    """
    cls = glm_lang.LanguageModel
    if "_omlx_mtp_sanitize_patched" in cls.__dict__:
        return

    from ..mlx_lm_mtp.glm5_next_model import (
        _completa_hiperconexao_da_cabeca,
        _renomeia_para_mtp,
    )

    original_sanitize = cls.sanitize

    def sanitize(self, weights):
        n_main = int(self.args.num_hidden_layers)
        n_mtp = len(getattr(self, "mtp", ()) or ())
        if not n_mtp:
            # Sem bloco anexado nao ha onde pendurar a cabeca: as chaves dela
            # ficariam sem parametro e a carga estrita recusaria o modelo
            # inteiro. Cai na limpeza de fabrica, que as descarta. E o
            # remendo e de CLASSE: quem quantiza sem preservar a cabeca no
            # mesmo processo em que outro pedido preservou tem que continuar
            # vendo a limpeza de fabrica.
            return original_sanitize(self, weights)

        cabeca = {k: v for k, v in weights.items() if "mtp." in k}
        corpo = {k: v for k, v in weights.items() if "mtp." not in k}

        # O checkpoint guarda a cabeça como `...layers.<n_main + i>.*`, e a
        # limpeza de fábrica descarta camadas a partir de `num_hidden_layers`.
        # Levantar a contagem durante a chamada faz cada transformação da camada
        # comum valer igual para a da cabeça; só depois as chaves são renomeadas.
        prefixos = tuple(
            f".layers.{n_main + i}." for i in range(max(n_mtp, 1))
        )
        cru = any(p in k for k in corpo for p in prefixos)

        if cru and n_mtp:
            self.args.num_hidden_layers = n_main + n_mtp
            try:
                limpo = original_sanitize(self, corpo)
            finally:
                self.args.num_hidden_layers = n_main
            limpo = _renomeia_para_mtp(limpo, n_main, n_mtp)
        else:
            limpo = original_sanitize(self, corpo)

        limpo.update(cabeca)
        if n_mtp:
            limpo = _completa_hiperconexao_da_cabeca(self, limpo)
        return limpo

    cls.sanitize = sanitize
    cls._omlx_mtp_sanitize_patched = True
