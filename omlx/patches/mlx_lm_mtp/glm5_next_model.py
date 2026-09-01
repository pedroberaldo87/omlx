# SPDX-License-Identifier: Apache-2.0
"""Previsão múltipla (Lightning MTP) para o GLM-5.3-Flash (``glm5_next``).

Irmão de ``glm_moe_dsa_model.py``, que faz o mesmo para o GLM-5.2. O bloco da
cabeça é idêntico entre as duas gerações — ``enorm``/``hnorm`` fundidos por
``eh_proj``, mais uma camada decoder da própria família e o ``shared_head.norm``
antes do ``lm_head`` compartilhado. O que muda é a arquitetura do tronco, e são
três diferenças que este arquivo existe para tratar:

1. **As listas de tipo não cobrem a camada extra.** ``Glm5NextDecoderLayer``
   lê ``config.layer_types[layer_idx]`` e ``config.mlp_layer_types[layer_idx]``,
   e as duas têm exatamente ``num_hidden_layers`` entradas (45, índices 0..44).
   Construir a camada 45 dá ``IndexError``. O irmão estende ``indexer_types``
   pelo mesmo motivo; aqui são duas listas.

2. **A assinatura da camada decoder difere.** O GLM-5.2 faz
   ``x, _ = self.block(x, mask, cache, None)`` (quatro argumentos, devolve
   tupla); o GLM-5.3 faz ``x = self.block(x, mask, cache)`` (três, devolve o
   valor).

3. **O tronco alterna camadas.** ``layer_types`` é três ``linear_attention``
   para uma ``deepseek_sparse_attention``, repetido. O tipo da camada da cabeça
   **não está declarado no config**, então ela espelha a última camada comum —
   ver ``_tipo_da_camada_da_cabeca``.

O cache também difere: as camadas lineares do GLM-5.3 usam ``ArraysCache`` (estado
recorrente) e as esparsas ``CacheList(KVCache, PoolingCache)``, em vez do par de
``KVCache`` do irmão. ``make_mtp_cache`` monta o que o tipo da camada da cabeça
pedir, em vez de assumir.

A implementação de texto do ``glm5_next`` é registrada no mlx-lm por
``omlx/patches/mlx_lm_glm5_next.py::register_into_mlx_lm()`` — o mesmo registro
de que a quantização depende para preservar a cabeça.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Nomes de peso da cabeça que não seguem o padrão da camada comum. Mesmo mapa do
# irmão: o checkpoint chama o último de ``shared_head.norm`` e o bloco o expõe
# como ``norm``.
_ESPECIAIS = {
    "eh_proj.weight": "eh_proj.weight",
    "enorm.weight": "enorm.weight",
    "hnorm.weight": "hnorm.weight",
    "shared_head.norm.weight": "norm.weight",
}


def _vendored():
    """A camada decoder e o ``linear_forward`` da árvore vendorada do GLM-5.x.

    Delega ao módulo que REGISTRA a família (``omlx/patches/mlx_lm_glm5_next.py``)
    em vez de importar ``mlx_vlm`` aqui. Não é rodeio: a análise estática de
    ``omlx/cluster/autoconfigure.py`` lê o fonte de cada patch do despacho para
    saber o que um nó vai importar, e um nó de mlx-lm que serve OUTRA família
    carrega este despacho inteiro sem nunca chegar aqui. Citar ``mlx_vlm`` neste
    arquivo faria todo nó de mlx-lm passar a exigir mlx-vlm — foi o que
    aconteceu, e dois testes de cluster pegaram.

    Quem já tem essa dependência declarada é o módulo de registro, e é dele que
    o caminho sai.
    """
    from omlx.patches import mlx_lm_glm5_next

    _, language_model = mlx_lm_glm5_next._vendored()
    # ``language_model`` já É a classe — ``type()`` dela devolveria ``type``,
    # cujo módulo é ``builtins``.
    modulo = sys.modules[language_model.__module__]
    return modulo.Glm5NextDecoderLayer, modulo.linear_forward


def _e_nosso(cls: Any, attr: str, marca: str) -> bool:
    """O método já é o nosso? (marcador na função viva, não na classe)"""
    fn = cls.__dict__.get(attr)
    return fn is not None and getattr(fn, marca, False)


def apply() -> bool:
    """Aplica os remendos do lado do modelo, se a família estiver registrada."""
    glm = sys.modules.get("mlx_lm.models.glm5_next")
    if glm is None or not hasattr(glm, "Model"):
        logger.debug(
            "glm5_next não registrado no mlx-lm; pulando o remendo de previsão "
            "múltipla (esperado para modelos de outra família)"
        )
        return False

    _patch_model_args(glm)
    _register_mtp_block(glm)
    _patch_model(glm)

    if not hasattr(glm.Model, "_omlx_mtp_patched"):
        glm.Model._omlx_mtp_patched = "patch"
        logger.info("GLM-5.3 (glm5_next) MTP model patch applied")
    return True


# ---------------------------------------------------------------------------
# ModelArgs — carregar a contagem da cabeça e estender as listas de tipo.
# ---------------------------------------------------------------------------


def _tipo_da_camada_da_cabeca(tipos: list) -> str:
    """Que tipo a camada da cabeça assume: a atenção ESPARSA, sempre.

    ``layer_types`` para na última camada comum e não diz o tipo da cabeça, mas
    o config diz por outro caminho: ``index_share_for_mtp_iteration`` liga o
    indexer à iteração da cabeça, e indexer é peça exclusiva da esparsa. O
    checkpoint de referência confirma — a camada 45 de ``zai-org/GLM-5.3-Flash``
    traz ``self_attn.indexer.*``, ``kv_a_proj_with_mqa`` e ``q_a_proj``, e
    nenhum ``conv1d``/``forget_gate``.

    Espelhar a última camada comum era o palpite anterior, e ele erra: no
    GLM-5.3-Flash a 44 é linear. Construir linear não levanta erro nenhum —
    monta o bloco errado em silêncio, e aí nenhum peso da cabeça acha destino.

    O nome do tipo sai da própria lista, não de uma constante escrita aqui:
    se o tronco mudar de nomenclatura, a cabeça acompanha.
    """
    esparsas = [t for t in tipos if "linear" not in t]
    if esparsas:
        return esparsas[-1]
    return tipos[-1] if tipos else "linear_attention"


def _estende(tipos: Any, ate: int, preenche: Any) -> list | None:
    """Copia a lista e a estende até ``ate``, sem mutar a do chamador.

    A lista vem do dicionário de config e é compartilhada — crescer no lugar
    contamina quem a passou.
    """
    if not isinstance(tipos, list):
        return None
    saida = list(tipos)
    while len(saida) < ate:
        saida.append(preenche(saida))
    return saida


def _patch_model_args(glm: Any) -> None:
    """Envolve ``ModelArgs.from_dict`` para a camada da cabeça ser construível.

    ``num_nextn_predict_layers`` não é campo declarado (o filtro de campos
    desconhecidos o descarta), e as duas listas de tipo cobrem só o tronco.
    """
    args_cls = glm.ModelArgs
    if "_omlx_mtp_args_patched" in args_cls.__dict__:
        return

    original = args_cls.from_dict.__func__

    def from_dict_remendado(cls, params):
        args = original(cls, params)
        # a contagem vive sob text_config num checkpoint de visão, e solta num
        # export só-texto — o mesmo lugar de onde `from_dict` leu o resto.
        texto = params.get("text_config")
        fonte = texto if isinstance(texto, dict) else params
        n_mtp = int(fonte.get("num_nextn_predict_layers", 0) or 0)
        args.num_nextn_predict_layers = n_mtp

        if n_mtp > 0:
            n_main = int(args.num_hidden_layers)
            alvo = n_main + n_mtp

            estendida = _estende(
                getattr(args, "layer_types", None), alvo,
                lambda atual: _tipo_da_camada_da_cabeca(atual[:n_main]),
            )
            if estendida is not None:
                args.layer_types = estendida

            # o tipo de FFN da cabeça acompanha o da última camada comum pelo
            # mesmo motivo: é o regime em que o hidden que ela recebe foi gerado.
            mlp = _estende(
                getattr(args, "mlp_layer_types", None), alvo,
                lambda atual: atual[n_main - 1] if n_main else "dense",
            )
            if mlp is not None:
                args.mlp_layer_types = mlp

            indexer = _estende(
                getattr(args, "indexer_types", None), alvo,
                lambda atual: atual[n_main - 1] if n_main else "full",
            )
            if indexer is not None:
                args.indexer_types = indexer

        return args

    args_cls.from_dict = classmethod(from_dict_remendado)
    args_cls._omlx_mtp_args_patched = True


# ---------------------------------------------------------------------------
# O bloco da cabeça.
# ---------------------------------------------------------------------------


def _register_mtp_block(glm: Any) -> None:
    if hasattr(glm, "Glm5NextMTPBlock"):
        return

    import mlx.core as mx
    import mlx.nn as nn

    Glm5NextDecoderLayer, _ = _vendored()

    class Glm5NextMTPBlock(nn.Module):
        """Uma camada de previsão múltipla: fusão enorm/hnorm + camada decoder.

        ``norm`` guarda o ``shared_head.norm`` do checkpoint — aplicado à saída
        antes do ``lm_head`` compartilhado, que ``mtp_forward`` executa para
        poder encolher a projeção de vocabulário.
        """

        def __init__(self, config, layer_idx: int):
            super().__init__()
            dim = config.hidden_size
            self.enorm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
            self.hnorm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
            self.eh_proj = nn.Linear(2 * dim, dim, bias=False)
            self.norm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
            self.block = Glm5NextDecoderLayer(config, layer_idx)

        def __call__(self, h, embed_tokens, input_ids, mask, cache):
            e = self.enorm(embed_tokens(input_ids))
            x = self.eh_proj(mx.concatenate([e, self.hnorm(h)], axis=-1))
            # Diferença 2: a camada do GLM-5.3 recebe três argumentos e devolve
            # o valor, não uma tupla.
            return self.block(x, mask, cache)

    glm.Glm5NextMTPBlock = Glm5NextMTPBlock


# ---------------------------------------------------------------------------
# Model — anexar a cabeça, o cache dela, o passo de rascunho e a limpeza.
# ---------------------------------------------------------------------------


def _patch_model(glm: Any) -> None:
    cls = glm.Model
    init_envolvido = getattr(cls, "_omlx_mtp_init_wrapped", False)
    call_nosso = _e_nosso(cls, "__call__", "_omlx_mtp_call_marker")
    if init_envolvido and call_nosso:
        return

    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    original_init = cls.__init__
    original_sanitize = cls.sanitize

    def __init__(self, config):
        original_init(self, config)
        n_mtp = int(getattr(config, "num_nextn_predict_layers", 0) or 0)
        # Preso ao interruptor global, como as outras famílias: com a previsão
        # múltipla desligada o modelo fica indistinguível do de fábrica.
        from . import is_mtp_active

        ligado = bool(n_mtp > 0 and is_mtp_active())
        self._omlx_mtp_decode_enabled = ligado
        if ligado:
            n_main = int(config.num_hidden_layers)
            self.mtp = [
                glm.Glm5NextMTPBlock(config, n_main + i) for i in range(n_mtp)
            ]
            from . import get_mtp_depth

            self._omlx_mtp_chain = True
            self._omlx_mtp_depth = get_mtp_depth()
            self._omlx_mtp_head_clone = False
            # Mesma ordem de grandeza do irmão: com roteamento esparso cada
            # linha extra de verificação puxa um conjunto de especialistas quase
            # disjunto. Medir e ajustar quando houver número desta família.
            self._omlx_mtp_marginal_ms = 35.0

    __init__._omlx_mtp_init_marker = True

    def __call__(self, inputs, cache=None, return_hidden: bool = False,
                 n_confirmed: int = 0):
        # ``n_confirmed`` só importa a modelos com estado recorrente de módulo
        # que o cache não cobre; aqui o estado todo vive em cache aparável, então
        # é aceito e ignorado — mesma postura do irmão.
        out = self.model(inputs, cache=cache)
        if return_hidden:
            # ``out`` já é o hidden pós-norma final, que é exatamente o que a
            # cabeça consome — o irmão precisa de um `return_raw_hidden` porque
            # a estrutura dele é outra.
            return self._lm_head(out), out
        return self._lm_head(out)

    __call__._omlx_mtp_call_marker = True

    def _lm_head(self, out):
        _, linear_forward = _vendored()

        if getattr(self.args, "tie_word_embeddings", False):
            return self.model.embed_tokens.as_linear(out)
        return linear_forward(self.lm_head, out)

    def make_mtp_cache(self):
        """O cache da cabeça, no formato que o tipo da camada dela pedir.

        As camadas lineares do GLM-5.3 guardam estado recorrente
        (``ArraysCache``); as esparsas pareiam KV com o acumulado de compressão
        do seletor (``CacheList(KVCache, PoolingCache)``). Montar o que o tipo
        pede, em vez de assumir um par de KV como o irmão, é o que evita que a
        cabeça receba um cache de formato errado.
        """
        if not hasattr(self, "mtp"):
            return None
        caches = []
        for bloco in self.mtp:
            caches.append(_cache_para(bloco.block))
        return caches

    def mtp_forward(self, h, input_ids, cache=None, return_hidden: bool = False,
                    logits_keep: int = 0):
        """Roda a cabeça + a norma dela + o ``lm_head`` compartilhado.

        ``h`` é o hidden pós-norma do tronco na primeira dobra, ou a saída crua
        da própria cabeça nos passos encadeados — o ``hnorm`` de dentro do bloco
        normaliza os dois. ``logits_keep`` limita a norma e a projeção de
        vocabulário às últimas N posições (0 = todas).
        """
        if cache is None:
            cache = [None] * len(self.mtp)

        mask = create_attention_mask(h, cache[0], return_array=True)

        ultimo = None
        for i, bloco in enumerate(self.mtp):
            h = bloco(h, self.model.embed_tokens, input_ids, mask, cache[i])
            ultimo = bloco

        fonte = h
        if logits_keep and fonte.shape[1] > logits_keep:
            fonte = fonte[:, -logits_keep:]
        logits = self._lm_head(ultimo.norm(fonte))
        if return_hidden:
            return logits, h
        return logits

    def mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
        """Desfaz a janela de verificação até ``accepted`` rascunhos.

        Devolve False quando algum cache não é aparável — o chamador então
        descarta a janela inteira em vez de continuar com estado meio desfeito.
        """
        n = num_drafts - accepted
        if n <= 0:
            return True
        alvos = [c for c in _achata(cache) if c is not None]
        if not all(getattr(c, "is_trimmable", lambda: False)() for c in alvos):
            return False
        for c in alvos:
            if c.trim(n) != n:
                logger.warning(
                    "GLM-5.3: desfazer parcial não aparou tudo em %s",
                    type(c).__name__,
                )
                return False
        return True

    def sanitize(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """Mantém e renomeia a camada extra do checkpoint.

        O checkpoint guarda a cabeça como ``model.layers.<n_main + i>.*`` (o
        estilo DeepSeek-V3). A limpeza de fábrica descarta camadas a partir de
        ``num_hidden_layers``, então a contagem é levantada durante a chamada
        para que cada transformação da camada comum se aplique igual à da
        cabeça, e só depois as chaves são renomeadas sob ``mtp.<i>.*``.
        """
        tem_cabeca = hasattr(self, "mtp")
        n_mtp = len(self.mtp) if tem_cabeca else 0
        n_main = int(self.args.num_hidden_layers)

        tem_pesos = any(
            k.startswith("mtp.") or k.startswith(f"model.layers.{n_main}.")
            or k.startswith(f"model.language_model.layers.{n_main}.")
            for k in weights
        )
        if tem_cabeca and not tem_pesos:
            # Checkpoint que perdeu a cabeça: solta a nossa em vez de falhar a
            # carga estrita por parâmetro faltando.
            try:
                del self.mtp
            except AttributeError:
                pass
            self._omlx_mtp_decode_enabled = False
            tem_cabeca = False
            n_mtp = 0

        if not tem_cabeca:
            weights = {k: v for k, v in weights.items() if not k.startswith("mtp.")}
            return original_sanitize(self, weights)

        prefixos = tuple(
            p
            for i in range(n_mtp)
            for p in (
                f"model.layers.{n_main + i}.",
                f"model.language_model.layers.{n_main + i}.",
            )
        )
        cru = any(k.startswith(prefixos) for k in weights)

        if cru:
            self.args.num_hidden_layers = n_main + n_mtp
            try:
                weights = original_sanitize(self, weights)
            finally:
                self.args.num_hidden_layers = n_main
            weights = _renomeia_para_mtp(weights, n_main, n_mtp)
            return weights

        return original_sanitize(self, weights)

    if not init_envolvido:
        cls.__init__ = __init__
        cls._omlx_mtp_init_wrapped = True
    cls.__call__ = __call__
    cls._lm_head = _lm_head
    cls.make_mtp_cache = make_mtp_cache
    cls.mtp_forward = mtp_forward
    cls.mtp_partial_rollback = mtp_partial_rollback
    cls.sanitize = sanitize


# ---------------------------------------------------------------------------
# Auxiliares.
# ---------------------------------------------------------------------------


def _cache_para(camada: Any) -> Any:
    """O objeto de cache que uma camada do GLM-5.3 pede."""
    _vendored()  # garante a árvore exposta antes de tocar nos caches
    from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, PoolingCache

    if getattr(camada, "is_linear", False):
        return ArraysCache(size=2)
    return CacheList(KVCache(), PoolingCache(camada.self_attn.indexer.index_kpool))


def _achata(cache: Any) -> list:
    """Percorre CacheList aninhado e devolve as folhas aparáveis."""
    saida = []
    for c in cache or []:
        subs = getattr(c, "caches", None)
        if subs:
            saida.extend(subs)
        else:
            saida.append(c)
    return saida


def _renomeia_para_mtp(weights: Dict[str, Any], n_main: int, n_mtp: int) -> Dict[str, Any]:
    """``model[.language_model].layers.<n_main+i>.*`` vira ``mtp.<i>.*``."""
    saida: Dict[str, Any] = {}
    for k, v in weights.items():
        novo = k
        for i in range(n_mtp):
            for prefixo in (
                f"model.layers.{n_main + i}.",
                f"model.language_model.layers.{n_main + i}.",
            ):
                if not k.startswith(prefixo):
                    continue
                resto = k[len(prefixo):]
                resto = _ESPECIAIS.get(resto, f"block.{resto}")
                novo = f"mtp.{i}.{resto}"
                break
            if novo != k:
                break
        saida[novo] = v
    return saida
