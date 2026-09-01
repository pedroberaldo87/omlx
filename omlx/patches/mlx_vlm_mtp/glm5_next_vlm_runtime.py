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
            self._omlx_mtp_head_clone = False
            # Mesma ordem de grandeza declarada no irmão: com roteamento esparso
            # cada linha extra de verificação puxa um conjunto de especialistas
            # quase disjunto. Medir e ajustar quando houver número desta família.
            self._omlx_mtp_marginal_ms = 35.0

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

        from mlx_vlm.models.base import LanguageModelOutput
        from mlx_vlm.models.glm5_next.language import linear_forward

        if inputs is None:
            inputs = kwargs.get("input_ids")
        h = self.model(inputs, cache=cache, inputs_embeds=inputs_embeds)
        fonte = h
        nlk = kwargs.get("num_logits_to_keep", 0)
        if nlk:
            fonte = fonte[:, -nlk:, :]
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(fonte)
        else:
            logits = linear_forward(self.lm_head, fonte)
        return LanguageModelOutput(logits=logits, hidden_states=[h])

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

    cls.__init__ = __init__
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls._omlx_mtp_runtime_patched = True


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

    from ..mlx_lm_mtp.glm5_next_model import _completa_hiperconexao_da_cabeca

    original_sanitize = cls.sanitize

    def sanitize(self, weights):
        cabeca = {k: v for k, v in weights.items() if "mtp." in k}
        corpo = {k: v for k, v in weights.items() if "mtp." not in k}

        n_main = int(self.args.num_hidden_layers)
        n_mtp = len(getattr(self, "mtp", ()) or ())

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


def _renomeia_para_mtp(weights, n_main: int, n_mtp: int):
    """``...layers.<n_main+i>.X`` vira ``mtp.<i>.X``, que é onde o bloco mora.

    O trio da cabeça (``eh_proj``, ``enorm``, ``hnorm``) fica na raiz do bloco;
    o resto é a camada decoder, que vive sob ``block.``. A norma final tem nome
    próprio no checkpoint e vira a ``norm`` do bloco.
    """
    import re

    saida = {}
    padrao = re.compile(r"^(.*?)layers\.(\d+)\.(.*)$")
    raiz = ("eh_proj", "enorm", "hnorm")
    for chave, valor in weights.items():
        m = padrao.match(chave)
        if not m:
            saida[chave] = valor
            continue
        idx = int(m.group(2))
        if not (n_main <= idx < n_main + n_mtp):
            saida[chave] = valor
            continue
        resto = m.group(3)
        i = idx - n_main
        if resto.startswith("shared_head.norm."):
            novo = f"mtp.{i}.norm." + resto[len("shared_head.norm."):]
        elif resto.split(".")[0] in raiz:
            novo = f"mtp.{i}.{resto}"
        else:
            novo = f"mtp.{i}.block.{resto}"
        saida[novo] = valor
    return saida
