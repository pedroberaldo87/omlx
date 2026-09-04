# SPDX-License-Identifier: Apache-2.0
"""Make mlx-lm able to load GLM-5.x (glm5_next), so DFlash can target it.

DFlash loads the *target* model through ``mlx_lm.utils.load``, and mlx-lm
resolves an architecture by importing ``mlx_lm.models.<model_type>``. There is
no ``glm5_next`` module there, so the load dies with

    Model type glm5_next not supported.

before the drafter is ever read, and engine_pool falls back to the VLM engine.
GLM-5.x does exist in this repo, but as an *mlx-vlm* model: the vendored
package under ``omlx/patches/mlx_vlm_glm5_next_compat/vendor``. This module is
the adapter between the two, following the pattern dflash-mlx itself uses for
Muse Glimmer (``dflash_mlx.models.muse_glimmer.register_into_mlx_lm``): seed
``sys.modules`` with a module that exposes what mlx-lm asks for.

mlx-lm asks for exactly two names, and calls them like this:

    model_args = ModelArgs.from_dict(config)     # the WHOLE config.json
    model = Model(model_args)
    ...
    logits = model(inputs)                       # expects an array

Two things differ from the vendored language model, and each is bridged below:

1. GLM-5.x ships a VLM config, so the text fields live under ``text_config``.
   mlx-lm hands over the top-level dict, so ``ModelArgs.from_dict`` has to
   descend into it.
2. The vendored ``LanguageModel`` returns ``LanguageModelOutput``, a wrapper
   with a ``.logits`` attribute. mlx-lm indexes the return value as an array.

Registration yields to a real upstream module: if mlx-lm ever ships
``mlx_lm.models.glm5_next``, this does nothing.
"""

import sys
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

_MODULE_NAME = "mlx_lm.models.glm5_next"


def _vendored():
    """Import the vendored GLM-5.x text stack, applying the compat patch first."""
    from omlx.patches.mlx_vlm_glm5_next_compat import (
        apply_mlx_vlm_glm5_next_compat_patch,
    )

    apply_mlx_vlm_glm5_next_compat_patch()
    from mlx_vlm.models.glm5_next.config import TextConfig
    from mlx_vlm.models.glm5_next.language import LanguageModel

    return TextConfig, LanguageModel


class ModelArgs:
    """mlx-lm's args object, built from a GLM-5.x VLM config.json."""

    @classmethod
    def from_dict(cls, params: dict) -> Any:
        TextConfig, _ = _vendored()
        # The text fields live under text_config on a VLM checkpoint; accept a
        # flat text-only config too, so a text-only export still loads.
        text = params.get("text_config")
        if not isinstance(text, dict):
            text = params
        args = TextConfig.from_dict(text)
        # mlx-lm's load_model looks the per-tensor quantization table up by
        # MODULE path (``model.layers.3.self_attn.embed_q``) — the same
        # ``config`` dict it hands us here. A VLM-layout oQ checkpoint keys
        # that table ``language_model.model.layers...`` (sanitize strips the
        # prefix from the WEIGHTS, but nothing stripped it from the table), so
        # every override missed, fell to the default bits, and an 8-bit tensor
        # died in dequantize with a shape mismatch. Measured on
        # GLM-5.3-Flash-oQ2e (03/09); the 31/08 build had raw names and slid by.
        quant = params.get("quantization")
        if isinstance(quant, dict):
            for chave in [k for k in quant if isinstance(quant[k], dict)]:
                nova = _module_key(chave)
                if nova != chave:
                    quant[nova] = quant.pop(chave)
        # mlx-lm reads model_type off the args to pick cache and prompt paths.
        if not getattr(args, "model_type", None):
            args.model_type = "glm5_next"
        return args


def _strip_language_prefix(chave: str) -> str:
    """``language_model.X`` / ``model.language_model.X`` -> the text-only path."""
    if chave.startswith("model.language_model."):
        return "model." + chave[len("model.language_model."):]
    if chave.startswith("language_model."):
        return chave[len("language_model."):]
    return chave


def _module_key(chave: str) -> str:
    """A quantization-table key as the MODULE path the loaded model has.

    Mirrors what the vendored sanitize does to the weights: the language
    prefix goes, and a raw-layout checkpoint's forget gate
    (``self_attn.f_a_proj`` / ``f_b_proj``) is nested under
    ``self_attn.forget_gate.``. Measured on the Vontra 2-bit build (04/09):
    its table names the gate raw, the module is nested, the lookup missed,
    and the 8-bit f_a_proj died in quantized_matmul with
    "w.shape() == (128,1024) and scales.shape() == (128,64) ... bits=2".
    """
    chave = _strip_language_prefix(chave)
    for parte in ("f_a_proj", "f_b_proj"):
        sufixo = ".self_attn." + parte
        if chave.endswith(sufixo):
            return chave[: -len(parte)] + "forget_gate." + parte
    return chave


class Model(nn.Module):
    """mlx-lm's model object, wrapping the vendored GLM-5.x language model."""

    def __init__(self, args: Any):
        super().__init__()
        _, LanguageModel = _vendored()
        self.args = args
        self.model_type = getattr(args, "model_type", "glm5_next")
        lm = LanguageModel(args)
        # Hold the submodules DIRECTLY, not the LanguageModel wrapper. Nesting
        # it would put the head at `_lm.lm_head` while the checkpoint names it
        # `lm_head`, and the load fails with "Received 3 parameters not in
        # model" (weight/scales/biases) -- measured on GLM-5.3-Flash-REAP37.
        self.model = lm.model
        if not getattr(args, "tie_word_embeddings", False):
            self.lm_head = lm.lm_head

    @property
    def layers(self):
        return self.model.layers

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        # Same tail as the vendored LanguageModel.__call__, inlined because the
        # wrapper itself is not held (see __init__). Returns a bare array:
        # mlx-lm indexes the return value, it does not unwrap a container.
        from mlx_vlm.models.glm5_next.language import linear_forward

        out = self.model(inputs, cache=cache, inputs_embeds=inputs_embeds)
        if getattr(self.args, "tie_word_embeddings", False):
            return self.model.embed_tokens.as_linear(out)
        return linear_forward(self.lm_head, out)

    def make_cache(self):
        """The hybrid cache, same shape as the vendored LanguageModel builds.

        Inlined rather than delegated: make_cache lives on the vendored
        LanguageModel wrapper, which this class deliberately does not hold
        (see __init__), and NOT on Glm5NextModel underneath it. The linear
        layers keep a recurrent state; the sparse-attention ones pair a KV
        cache with the DSA indexer's pooling cache.
        """
        from mlx_lm.models.cache import PoolingCache
        from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache

        caches = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=2))
            else:
                caches.append(
                    CacheList(
                        KVCache(),
                        PoolingCache(layer.self_attn.indexer.index_kpool),
                    )
                )
        return caches

    def sanitize(self, weights: dict) -> dict:
        """Turn VLM checkpoint keys into the text-only names this model has.

        The published GLM-5.x checkpoints are VLM exports: every tensor is
        under ``language_model.*`` or ``vision_model.*`` (measured on
        GLM-5.3-Flash-REAP37: 2651 and 347). This model is the text stack
        alone, so the vision tower is dropped and the language prefix
        stripped. Without this the load fails with "Received 2998 parameters
        not in model" and DFlash falls back to the VLM engine.
        """
        renomeados = {}
        for chave, valor in weights.items():
            if chave.startswith(("vision_model.", "visual.", "model.visual.",
                                 "model.vision_model.")):
                continue
            # Os dois arranjos que os checkpoints desta família usam. O REAP37
            # pendura a torre de texto na raiz; o publicado pela zai-org e o do
            # Vontra a penduram sob `model.`, e medir só o primeiro deixava
            # NENHUM dos 221 nomes de camada casar com os do modelo.
            if chave.startswith("model.language_model."):
                chave = "model." + chave[len("model.language_model."):]
            elif chave.startswith("language_model."):
                chave = chave[len("language_model."):]
            renomeados[chave] = valor

        # Só o prefixo não basta para o checkpoint publicado: ele chega com os
        # nomes crus — `hc_attn_base` por `attn_hc.base`, três convoluções
        # separadas em vez de uma fundida, o portão de esquecimento solto — e a
        # quantização de origem por desfazer. Medido em 01/09 no Vontra: 112.180
        # parâmetros recusados por não existirem no modelo.
        #
        # Quem sabe desfazer tudo isso é o `sanitize` do modelo vendorado, e ele
        # NÃO roda sozinho aqui: esta classe não segura o `LanguageModel` (pega
        # os submódulos direto, ver __init__) e o mlx-lm chama `sanitize` uma
        # vez só. Daí a delegação.
        #
        # As chaves da cabeça de previsão múltipla passam POR FORA: o sanitize
        # vendorado descarta toda chave que contenha `mtp.`, e é justamente ela
        # que não pode se perder.
        cabeca = {k: v for k, v in renomeados.items() if "mtp." in k}
        corpo = {k: v for k, v in renomeados.items() if "mtp." not in k}

        _, LanguageModel = _vendored()
        limpo = LanguageModel.sanitize(self, corpo)
        limpo.update(cabeca)
        return limpo


def register_into_mlx_lm() -> bool:
    """Seed ``mlx_lm.models.glm5_next``. Yields to a real upstream module."""
    import importlib.util

    if _MODULE_NAME in sys.modules:
        return False
    try:
        if importlib.util.find_spec(_MODULE_NAME) is not None:
            return False
    except (ImportError, ModuleNotFoundError):
        pass
    sys.modules[_MODULE_NAME] = sys.modules[__name__]
    return True


__all__ = ["Model", "ModelArgs", "register_into_mlx_lm"]
