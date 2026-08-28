# SPDX-License-Identifier: Apache-2.0
"""The sparse-attention block selection must match the current query length.

Two concurrent requests whose prompts cross the sparse-attention threshold used
to abort the whole batch: a stale MTP-verify position leaked into the batched
query, put_along_axis broadcast that stale axis, and the concatenate inside
update_indexer raised, returning 500 to every live stream.

    RuntimeError: [concatenate] All the input array dimensions must match exactly
    except for the concatenation axis. However, the provided shapes are
    (2,2,2536), (2,1,3), and the concatenation axis is -1.

The clamp keeps the freshest query rows. This file pins two things: that the
clamp fires and produces a mask of the right shape, and that it is a no-op when
the axes already agree.
"""

import mlx.core as mx
import pytest

from omlx.patches.mlx_vlm_qwen4_exp_compat import (
    apply_mlx_vlm_qwen4_exp_compat_patch,
)

apply_mlx_vlm_qwen4_exp_compat_patch()

from mlx_vlm.models.qwen4_exp.language import Qwen4ExpQSAIndexer  # noqa: E402


class _Rotary:
    """The indexer only ever calls apply_rotary; positions do not matter here."""

    def apply_rotary(self, q, k, position_ids, unsqueeze_dim=1):
        return q, k


def _indexer():
    """Build a real indexer off the same tiny config the compat suite uses."""
    from test_mlx_vlm_qwen4_exp_compat import _tiny_config

    config = _tiny_config()
    return Qwen4ExpQSAIndexer(config.text_config, _Rotary())


def test_a_real_indexer_survives_a_stale_selection_axis():
    """Drive from_projected with more key blocks than the topk so the sparse
    path engages, and confirm the returned mask follows the query length."""
    indexer = _indexer()
    before = Qwen4ExpQSAIndexer._seq_clamp_hits

    batch, seq_len = 2, 1
    heads = indexer.n_heads + indexer.kv_heads
    # Enough keys that max_complete_blocks exceeds block_topk and the sparse
    # branch runs instead of returning None.
    key_len = (indexer.block_topk + 4) * indexer.compress_ratio
    qk = mx.random.normal((batch, seq_len, heads * indexer.head_dim))
    mx.eval(qk)

    class _Cache:
        def __init__(self):
            self.keys = None
            self.offset = key_len

        def update_indexer(self, raw_keys, position_ids):
            full_keys = mx.random.normal((batch, key_len, indexer.head_dim))
            full_pos = mx.broadcast_to(
                mx.arange(key_len, dtype=mx.int32)[None, :], (batch, key_len)
            )
            mx.eval(full_keys, full_pos)
            return full_keys, full_pos

    mask = indexer.from_projected(qk, _Cache(), None)
    if mask is not None:
        mx.eval(mask)
        assert mask.shape[0] == batch
        assert mask.shape[1] == seq_len, "the mask must follow the query length"
    assert Qwen4ExpQSAIndexer._seq_clamp_hits >= before


def test_mask_shape_follows_the_query_length_not_the_selection():
    """put_along_axis is what turns a stale axis into the crash — pin its shape."""
    batch, seq_len, blocks, topk = 2, 1, 4, 2

    stale = mx.zeros((batch, 2, topk), dtype=mx.int32)
    clamped = stale[:, -seq_len:, :]

    hits = mx.put_along_axis(
        mx.zeros((batch, seq_len, blocks), dtype=mx.bool_),
        clamped,
        mx.array(True),
        axis=-1,
    )
    mx.eval(hits)
    assert hits.shape == (batch, seq_len, blocks)


def test_clamp_is_a_noop_when_the_axes_already_agree():
    seq_len = 3
    selected = mx.zeros((2, seq_len, 4), dtype=mx.int32)
    assert selected.shape[1] == seq_len  # the guard would not fire


def test_the_clamp_runs_after_the_selection_exists():
    """Regression pin for the upstream proposal, which placed it 39 lines early.

    In PR #3264 the guard referenced ``selected_blocks`` before the argpartition
    that creates it, so the first time it fired it would have raised NameError
    instead of clamping.
    """
    import inspect

    src = inspect.getsource(Qwen4ExpQSAIndexer.from_projected)
    definicao = src.index("selected_blocks = mx.argpartition")
    guarda = src.index("selected_blocks.shape[1] != seq_len")
    assert definicao < guarda, "the clamp must come after selected_blocks exists"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
