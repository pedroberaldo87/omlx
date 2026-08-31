# SPDX-License-Identifier: Apache-2.0
"""DFlash target backend for GLM-5.x (glm5_next).

dflash-mlx ships three target backends and none of them claims glm5_next, so
`resolve_target_ops` raises before the drafter is ever loaded. The drafter
itself needs no work: GLM-5.3-Flash-DFlash2 declares
``architectures: ["DFlash2DraftModel"]``, which dflash-mlx already resolves
in `_get_dflash_model_classes`.

What the target side needs is small but not nothing. GLM-5.x is a hybrid
stack shaped exactly like the Qwen GDN one the stock backend drives -- the
text model exposes ``fa_idx``/``ssm_idx``, its layers carry ``is_linear``,
and each block takes ``(x, mask, cache)`` -- so QwenGdnTargetOps is the right
base. Four things differ, and each is overridden below:

1. the gate keys on the literal substring "qwen";
2. GLM broadcasts the hidden state across a hyper-connection axis before the
   layer walk (B, S, hc_mult, D) and averages it away afterwards, so a
   forward that skips the tile feeds every layer the wrong rank;
3. GLM names its linear attention module ``self_attn``, not ``linear_attn``,
   and pairs its sparse-attention layers with
   ``CacheList(KVCache, PoolingCache)`` for the DSA indexer -- neither of
   which the stock `make_cache` builds;
4. because of (3) the stock speculative hooks would install a GQA hook on a
   sparse-attention module and no hook at all on the linear ones. Both are
   wrong here, so the hooks are declared unavailable rather than mis-installed.

Consequence of (4): this backend runs DFlash without the speculative linear
cache. Verification is still correct -- rejected drafts fall back to the
model's own cache -- but the recurrent-rollback fast path is off until the
GLM linear-attention module gets a hook of its own.
"""

from typing import Any, Optional

import mlx.core as mx

_BACKEND_PATH = "omlx.patches.dflash_glm5_next:Glm5NextTargetOps"


def _base_class():
    """Import the stock hybrid backend lazily (dflash-mlx is an optional dep)."""
    from dflash_mlx.engine.target_qwen_gdn import QwenGdnTargetOps

    return QwenGdnTargetOps


try:  # pragma: no cover - exercised only where dflash-mlx is installed
    _Base = _base_class()
except Exception:  # pragma: no cover - keeps the module importable without dflash
    _Base = object


class Glm5NextTargetOps(_Base):  # type: ignore[misc,valid-type]
    """Target ops for GLM-5.x, on top of the stock hybrid GDN backend."""

    def supports_model(self, target_model: Any) -> bool:
        model_type = self.model_type(target_model)
        if not model_type.startswith("glm5_next"):
            return False
        try:
            inner = self.text_model(target_model)
        except AttributeError:
            return False
        # The tile and the mask schedule both depend on these three.
        return (
            hasattr(inner, "layers")
            and hasattr(inner, "embed_tokens")
            and hasattr(inner, "hc_mult")
        )

    def capabilities_for(self, target_model: Any):
        # Start from the base capabilities so a dflash-mlx upgrade that adds a
        # field does not silently drop it here, then turn off the one path
        # this backend cannot honour yet (see point 4 in the module docstring).
        caps = super().capabilities_for(target_model)
        return caps.__class__(
            **{
                **{
                    field: getattr(caps, field)
                    for field in getattr(caps, "__dataclass_fields__", {})
                },
                "supports_recurrent_rollback": False,
            }
        )

    def install_speculative_hooks(self, target_model: Any) -> None:
        # Deliberately a no-op: the stock installer keys on `linear_attn` and
        # would put a GQA hook on GLM's sparse-attention module. Marking the
        # model as installed keeps callers from retrying on every generation.
        text_model = self.text_model(target_model)
        text_model._dflash_speculative_hooks_installed = True

    def make_cache(
        self,
        target_model: Any,
        *,
        enable_speculative_linear_cache: bool = False,
        quantize_kv_cache: bool = False,
        target_fa_window: Optional[int] = None,
    ) -> list[Any]:
        # The GLM text model builds its own hybrid cache: ArraysCache for the
        # linear layers, CacheList(KVCache, PoolingCache) for the sparse ones.
        # Rebuilding that here would duplicate the indexer's pooling geometry,
        # which lives on the layer, so delegate.
        #
        # Look on the WRAPPER first: make_cache sits next to lm_head, not on
        # the inner Glm5NextModel that text_model() returns. Checking only the
        # inner one raised AttributeError at the first generation.
        for candidate in (
            self.text_wrapper(target_model),
            self.text_model(target_model),
        ):
            make_cache = getattr(candidate, "make_cache", None)
            if callable(make_cache):
                return make_cache()
        raise AttributeError(
            "GLM-5.x DFlash target needs make_cache() on the text wrapper or "
            f"the text model; {type(self.text_wrapper(target_model))!r} and "
            f"{type(self.text_model(target_model))!r} provide neither"
        )

    def forward_with_hidden_capture(
        self,
        target_model: Any,
        *,
        input_ids: Optional[mx.array] = None,
        cache: Optional[list[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
        capture_layer_ids: Optional[set[int]] = None,
        logits_last_only: bool = False,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        """Same walk as the stock hybrid path, plus GLM's hyper-connection tile.

        The captured hidden states are the (B, S, D) view the drafter
        consumes, averaged over the hc axis exactly as the model's own
        forward does at the end -- capturing the 4D tile instead would hand
        the drafter a tensor of the wrong rank.
        """
        from mlx_vlm.models.base import create_attention_mask, create_ssm_mask

        inner = self.text_model(target_model)
        hidden_states = (
            input_embeddings
            if input_embeddings is not None
            else inner.embed_tokens(input_ids)
        )
        if cache is None:
            cache = [None] * len(inner.layers)

        capture_all = capture_layer_ids is None
        if capture_all:
            captured: list[mx.array] | dict[int, mx.array] = [hidden_states]
        else:
            capture_layer_ids = set(capture_layer_ids)
            captured = {0: hidden_states} if 0 in capture_layer_ids else {}

        fa_cache = cache[inner.fa_idx]
        fa_mask = create_attention_mask(
            hidden_states, fa_cache[0] if fa_cache else None, return_array=True
        )
        ssm_mask = create_ssm_mask(hidden_states, cache[inner.ssm_idx])

        h = mx.broadcast_to(
            hidden_states[:, :, None, :],
            (
                hidden_states.shape[0],
                hidden_states.shape[1],
                inner.hc_mult,
                hidden_states.shape[2],
            ),
        )
        h = mx.contiguous(h)

        for layer_index, (layer, layer_cache) in enumerate(
            zip(inner.layers, cache, strict=True)
        ):
            mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
            h = layer(h, mask=mask, cache=layer_cache)
            capture_key = layer_index + 1
            if capture_all:
                captured.append(h.mean(axis=2))
            elif capture_layer_ids is not None and capture_key in capture_layer_ids:
                captured[capture_key] = h.mean(axis=2)

        normalized = inner.norm(h.mean(axis=2))
        if logits_last_only and isinstance(captured, dict):
            captured[-1] = normalized
        logits_hidden = normalized[:, -1:, :] if logits_last_only else normalized
        logits = self.logits_from_hidden(target_model, logits_hidden)
        return logits, captured


def install_dflash_glm5_next_backend() -> bool:
    """Register the GLM-5.x target ops in dflash-mlx. Idempotent."""
    from dflash_mlx.engine import target_ops

    if _BACKEND_PATH in target_ops.TARGET_BACKENDS:
        return False
    # Ahead of the stock backends: resolve_target_ops takes the first match,
    # and none of them claims glm5_next, so order only matters for clarity.
    target_ops.TARGET_BACKENDS.insert(0, _BACKEND_PATH)
    return True


__all__ = ["Glm5NextTargetOps", "install_dflash_glm5_next_backend"]
