# SPDX-License-Identifier: Apache-2.0
"""DFlash support for GLM-5.x (glm5_next): the gate and the target backend.

Two halves have to agree, and the reason this file exists is that they once
did not for another architecture: the dashboard offered the DFlash toggle for
qwen4_exp, the loader then raised "Model type not supported", and engine_pool
swallowed it, so the operator saw the switch turn on and silently fall back
(v8 F2.7). Opening `is_dflash_compatible` for a model that
`resolve_target_ops` cannot serve reproduces exactly that.

So these tests pin both sides together:

- the compatibility gate accepts glm5_next
- omlx's own target backend claims glm5_next, and claims ONLY glm5_next, so
  registering it cannot steal a model from the stock backends

The drafter side needs no test here: GLM-5.3-Flash-DFlash2 declares
``architectures: ["DFlash2DraftModel"]`` and dflash-mlx already resolves that
in `_get_dflash_model_classes`.
"""

import json

import pytest

from omlx.engine.dflash import is_dflash_compatible


def _write_config(tmp_path, **cfg):
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return tmp_path


# --- the compatibility gate ---------------------------------------------------


@pytest.mark.parametrize("model_type", ["glm5_next", "glm5_next_text"])
def test_gate_accepts_glm5(tmp_path, model_type):
    ok, reason = is_dflash_compatible(_write_config(tmp_path, model_type=model_type))
    assert ok is True
    assert reason == ""


def test_gate_still_refuses_an_unsupported_architecture(tmp_path):
    ok, reason = is_dflash_compatible(_write_config(tmp_path, model_type="qwen4_exp"))
    assert ok is False
    assert "qwen4_exp" in reason


def test_the_refusal_message_names_glm(tmp_path):
    """A refusal should list what IS supported, GLM included."""
    _, reason = is_dflash_compatible(_write_config(tmp_path, model_type="llama"))
    assert "GLM" in reason


# --- the target backend -------------------------------------------------------

dflash_mlx = pytest.importorskip("dflash_mlx")


class _Camada:
    def __init__(self, is_linear):
        self.is_linear = is_linear


class _Inner:
    def __init__(self):
        self.layers = [_Camada(True), _Camada(False)]
        self.embed_tokens = object()
        self.hc_mult = 4
        self.fa_idx = 1
        self.ssm_idx = 0


class _Wrapper:
    def __init__(self):
        self.model = _Inner()


class _Alvo:
    """Minimal stand-in with the attribute shape the backend reads."""

    def __init__(self, model_type):
        self.language_model = _Wrapper()
        self.config = {"model_type": model_type}


def _ops():
    from omlx.patches.dflash_glm5_next import Glm5NextTargetOps

    return Glm5NextTargetOps()


def test_backend_claims_glm5_next():
    assert _ops().supports_model(_Alvo("glm5_next")) is True


def test_backend_does_not_claim_other_architectures():
    ops = _ops()
    for outro in ("qwen3_next", "gemma4", "laguna", "muse_glimmer", "llama"):
        assert ops.supports_model(_Alvo(outro)) is False, outro


def test_backend_refuses_a_model_without_the_hyper_connection_width():
    """hc_mult drives the tile; without it the forward would feed the wrong rank."""
    alvo = _Alvo("glm5_next")
    del alvo.language_model.model.hc_mult
    assert _ops().supports_model(alvo) is False


def test_registration_is_idempotent():
    from dflash_mlx.engine import target_ops

    from omlx.patches.dflash_glm5_next import (
        _BACKEND_PATH,
        install_dflash_glm5_next_backend,
    )

    original = list(target_ops.TARGET_BACKENDS)
    try:
        install_dflash_glm5_next_backend()
        install_dflash_glm5_next_backend()
        assert target_ops.TARGET_BACKENDS.count(_BACKEND_PATH) == 1
    finally:
        target_ops.TARGET_BACKENDS[:] = original


def test_registration_keeps_the_stock_backends():
    from dflash_mlx.engine import target_ops

    from omlx.patches.dflash_glm5_next import install_dflash_glm5_next_backend

    original = list(target_ops.TARGET_BACKENDS)
    try:
        install_dflash_glm5_next_backend()
        for stock in original:
            assert stock in target_ops.TARGET_BACKENDS
    finally:
        target_ops.TARGET_BACKENDS[:] = original


def test_speculative_linear_cache_is_declared_unavailable():
    """The stock hooks key on `linear_attn`; GLM names it `self_attn`.

    Rather than mis-installing a GQA hook on a sparse-attention module, this
    backend turns the recurrent-rollback path off. Verification still works.
    """
    ops = _ops()
    alvo = _Alvo("glm5_next")
    caps = ops.capabilities_for(alvo)
    assert caps.supports_recurrent_rollback is False
    assert caps.supports_dflash is True


def test_installing_hooks_is_a_no_op_that_does_not_repeat():
    ops = _ops()
    alvo = _Alvo("glm5_next")
    ops.install_speculative_hooks(alvo)
    assert alvo.language_model.model._dflash_speculative_hooks_installed is True
