# SPDX-License-Identifier: Apache-2.0
"""Streamed oQe calibration boundary for GLM-5.x (glm5_next).

GLM-5.3-Flash is 305.8 GB on disk, so the auto rule always wants streaming
for it. Before glm5_next joined the supported set the decision fell back to
the RAM-safe proxy, which builds a uniform 4-bit copy on disk (166.5 GB) and
is then rejected against the live memory limit -- the failure the dashboard
reported. These tests pin the two things that make the streamed path correct
for this layout:

- glm5_next is inside the supported set, so the auto rule turns streaming on
- the per-layer mask schedule is INDEXED, not shared: the 34 linear-attention
  layers take None and the 11 sparse-attention layers take the causal array
  mask. Reusing one mask for every layer would calibrate the sparse-attention
  layers on the wrong forward, silently.

Pure boundary tests: no checkpoint and no model load, so they run in CI where
the GLM-5.3 weights are absent.
"""

import pytest

from omlx.oq import (
    _STREAM_CALIBRATION_SUPPORTED_MODEL_TYPES,
    _resolve_stream_calibration,
    _stream_calibration_supported,
    _streamed_glm5_next_state,
)

mx = pytest.importorskip("mlx.core")

HC_MULT = 4
HIDDEN = 8
# The real GLM-5.3-Flash schedule shape in miniature: linear layers with two
# sparse-attention layers interleaved, exactly how config.layer_types reads.
LAYER_TYPES = [
    "linear_attention",
    "linear_attention",
    "deepseek_sparse_attention",
    "linear_attention",
    "deepseek_sparse_attention",
]


def _config(**overrides):
    text_config = {
        "hc_mult": HC_MULT,
        "hidden_size": HIDDEN,
        "layer_types": list(LAYER_TYPES),
    }
    text_config.update(overrides)
    return {"model_type": "glm5_next", "text_config": text_config}


# --- the gate ----------------------------------------------------------------


def test_glm5_next_is_in_the_supported_set():
    assert "glm5_next" in _STREAM_CALIBRATION_SUPPORTED_MODEL_TYPES
    assert _stream_calibration_supported("glm5_next") is True


def test_auto_streams_glm5_next_when_the_source_exceeds_ram():
    assert (
        _resolve_stream_calibration(
            None, model_exceeds_ram=True, model_type="glm5_next"
        )
        is True
    )


def test_auto_keeps_proxy_for_glm5_next_that_fits_in_ram():
    assert (
        _resolve_stream_calibration(
            None, model_exceeds_ram=False, model_type="glm5_next"
        )
        is False
    )


def test_explicit_streaming_request_is_accepted_for_glm5_next():
    assert (
        _resolve_stream_calibration(
            True, model_exceeds_ram=False, model_type="glm5_next"
        )
        is True
    )


# --- the boundary state ------------------------------------------------------


def test_inputs_are_hc_tiled_to_four_dimensions():
    embedded = mx.zeros((2, 6, HIDDEN), dtype=mx.bfloat16)
    hidden, _ = _streamed_glm5_next_state(_config(), embedded)
    mx.eval(hidden)
    assert hidden.shape == (2, 6, HC_MULT, HIDDEN)
    assert hidden.dtype == mx.bfloat16


def test_the_tile_repeats_the_embedding_across_the_hc_axis():
    embedded = mx.arange(2 * 3 * HIDDEN, dtype=mx.float32).reshape(2, 3, HIDDEN)
    hidden, _ = _streamed_glm5_next_state(_config(), embedded)
    mx.eval(hidden)
    for slot in range(HC_MULT):
        assert mx.array_equal(hidden[:, :, slot, :], embedded)


def test_mask_schedule_has_one_entry_per_layer():
    embedded = mx.zeros((1, 4, HIDDEN), dtype=mx.bfloat16)
    _, layer_masks = _streamed_glm5_next_state(_config(), embedded)
    assert len(layer_masks) == len(LAYER_TYPES)


def test_linear_layers_take_no_mask_and_sparse_layers_take_the_causal_mask():
    embedded = mx.zeros((1, 4, HIDDEN), dtype=mx.bfloat16)
    _, layer_masks = _streamed_glm5_next_state(_config(), embedded)
    for layer_type, mask in zip(LAYER_TYPES, layer_masks):
        if layer_type == "linear_attention":
            assert mask is None
        else:
            assert mask is not None
            assert mask.shape == (4, 4)


def test_the_two_layer_kinds_do_not_share_one_mask():
    """The regression this file exists for: an indexed schedule, not a shared mask."""
    embedded = mx.zeros((1, 4, HIDDEN), dtype=mx.bfloat16)
    _, layer_masks = _streamed_glm5_next_state(_config(), embedded)
    distintas = {mask is None for mask in layer_masks}
    assert distintas == {True, False}


def test_missing_hc_mult_is_refused():
    embedded = mx.zeros((1, 4, HIDDEN), dtype=mx.bfloat16)
    with pytest.raises(RuntimeError, match="hc_mult"):
        _streamed_glm5_next_state(_config(hc_mult=0), embedded)


def test_missing_layer_types_is_refused():
    embedded = mx.zeros((1, 4, HIDDEN), dtype=mx.bfloat16)
    with pytest.raises(RuntimeError, match="layer_types"):
        _streamed_glm5_next_state(_config(layer_types=[]), embedded)


# --- o teto do micro-lote -----------------------------------------------------
#
# The estimator prices one hidden state per routed expert, which is far below
# what a 288-expert top-8 layer actually materializes: a micro-batch of 6 was
# measured here peaking at 69.4 GB against a 402 MB estimate, which swapped the
# machine out. The streaming budget check only looks at memory left BETWEEN
# layers, so nothing else bounds the in-layer peak.


def _moe_config():
    return {
        "model_type": "glm5_next",
        "text_config": {
            "hidden_size": 4096,
            "n_routed_experts": 288,
            "num_experts_per_tok": 8,
            "layer_types": list(LAYER_TYPES),
            "hc_mult": HC_MULT,
        },
    }


def test_env_cap_lowers_the_micro_batch(monkeypatch):
    from omlx.oq import _oqe_calibration_batch_plan

    monkeypatch.delenv("OMLX_OQ_MAX_MICRO_BATCH", raising=False)
    livre = _oqe_calibration_batch_plan(
        _moe_config(), requested_samples=128, seq_length=512
    )["micro_batch_size"]

    monkeypatch.setenv("OMLX_OQ_MAX_MICRO_BATCH", "2")
    limitado = _oqe_calibration_batch_plan(
        _moe_config(), requested_samples=128, seq_length=512
    )["micro_batch_size"]

    assert limitado == 2
    assert limitado < livre


def test_env_cap_never_raises_the_micro_batch(monkeypatch):
    """A cap above the estimator's choice must not widen the batch."""
    from omlx.oq import _oqe_calibration_batch_plan

    monkeypatch.delenv("OMLX_OQ_MAX_MICRO_BATCH", raising=False)
    livre = _oqe_calibration_batch_plan(
        _moe_config(), requested_samples=128, seq_length=512
    )["micro_batch_size"]

    monkeypatch.setenv("OMLX_OQ_MAX_MICRO_BATCH", str(livre + 50))
    com_teto = _oqe_calibration_batch_plan(
        _moe_config(), requested_samples=128, seq_length=512
    )["micro_batch_size"]

    assert com_teto == livre


@pytest.mark.parametrize("valor", ["0", "-3", "abc", "  "])
def test_invalid_env_cap_is_ignored(monkeypatch, valor):
    from omlx.oq import _oqe_calibration_batch_plan

    monkeypatch.delenv("OMLX_OQ_MAX_MICRO_BATCH", raising=False)
    livre = _oqe_calibration_batch_plan(
        _moe_config(), requested_samples=128, seq_length=512
    )["micro_batch_size"]

    monkeypatch.setenv("OMLX_OQ_MAX_MICRO_BATCH", valor)
    assert (
        _oqe_calibration_batch_plan(
            _moe_config(), requested_samples=128, seq_length=512
        )["micro_batch_size"]
        == livre
    )
