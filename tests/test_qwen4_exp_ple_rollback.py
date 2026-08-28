# SPDX-License-Identifier: Apache-2.0
"""Speculative rollback oracle for Qwen4-Exp: verify+rollback == plain decode.

Qwen4-Exp keeps FOUR recurrent slots on the linear layer's ``ArraysCache``
(``LanguageModel.make_cache``: ``size=4 if "ple" in layer``):

    [0] GDN conv state          [1] GDN recurrent state
    [2] PLE short-conv state    [3] PLE n-gram token history

A verify forward writes all four for every token in the window, drafts
included. ``rollback_speculative_cache`` (inherited from qwen3_5) restores
only [0] and [1].

``test_partial_accept_restores_gdn_state`` guards the half that works.
``test_rejected_draft_does_not_poison_ple_state`` is the failing half: it
commits one token through verify+rollback and compares against committing
that token on its own.
"""

import mlx.core as mx
import pytest

from test_mlx_vlm_qwen4_exp_compat import _tiny_config

PREFIX = [2, 3, 4, 5]
WINDOW = [6, 7, 8, 9]  # [confirmed, draft1, draft2, draft3]


def _ple_cache(caches):
    """The linear layer's ArraysCache — the one carrying the PLE slots."""
    for c in caches:
        if len(getattr(c, "cache", ())) == 4:
            return c
    raise AssertionError("no size-4 ArraysCache in this model's cache list")


def _model():
    config = _tiny_config()
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    mx.random.seed(0)
    return LanguageModel(config.text_config, config)


def _spec_and_ref(model, accepted, k):
    """Verify a (k+1)-token window, roll back to ``accepted`` drafts.

    Returns the speculative cache list and a reference cache list that saw
    only the ``accepted + 1`` committed tokens, one forward each.
    """
    spec, ref = model.make_cache(), model.make_cache()
    prefix = mx.array([PREFIX], dtype=mx.int32)
    model(prefix, cache=spec)
    model(prefix, cache=ref)

    window = WINDOW[: k + 1]
    verified = model(
        mx.array([window], dtype=mx.int32), cache=spec, return_hidden=True
    )
    model.rollback_speculative_cache(
        spec, verified.gdn_states, accepted, block_size=k + 1
    )
    for token in window[: accepted + 1]:
        model(mx.array([[token]], dtype=mx.int32), cache=ref)
    return spec, ref


@pytest.mark.parametrize("accepted,k", [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)])
def test_partial_accept_restores_gdn_state(accepted, k):
    """GDN conv + recurrent state after rollback == plain decode (no off-by-one)."""
    model = _model()
    spec, ref = _spec_and_ref(model, accepted, k)
    a, b = _ple_cache(spec), _ple_cache(ref)
    mx.eval(a[0], a[1], b[0], b[1])

    assert mx.allclose(a[0], b[0], atol=1e-5).item(), "GDN conv state drifted"
    assert mx.allclose(a[1], b[1], atol=1e-5).item(), "GDN recurrent state drifted"
    # KV/QSA layers must land on the same length too.
    assert [getattr(c, "offset", None) for c in spec] == [
        getattr(c, "offset", None) for c in ref
    ]


def test_rejected_draft_does_not_poison_ple_state():
    """PLE state after rollback must not contain the rejected draft."""
    model = _model()
    spec, ref = _spec_and_ref(model, accepted=0, k=1)
    a, b = _ple_cache(spec), _ple_cache(ref)
    mx.eval(a[2], a[3], b[2], b[3])

    assert mx.array_equal(a[3], b[3]).item(), (
        f"PLE n-gram history keeps the rejected draft: "
        f"{a[3].tolist()} != {b[3].tolist()}"
    )
    assert mx.allclose(a[2], b[2], atol=1e-5).item(), (
        "PLE short-conv state keeps the rejected draft: max |delta| = "
        f"{mx.abs(a[2] - b[2]).max().item():.4g}"
    )

    # End to end: the next token's logits must match the non-speculative path.
    nxt = mx.array([[11]], dtype=mx.int32)
    spec_logits = model(nxt, cache=spec).logits
    ref_logits = model(nxt, cache=ref).logits
    mx.eval(spec_logits, ref_logits)
    assert mx.allclose(spec_logits, ref_logits, atol=1e-4).item(), (
        "post-rollback logits differ from plain decode: max |delta| = "
        f"{mx.abs(spec_logits - ref_logits).max().item():.4g}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
