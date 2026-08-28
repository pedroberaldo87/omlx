# SPDX-License-Identifier: Apache-2.0
"""A second request joining an in-flight one must not abort the batch.

qwen4_exp's singleton QSA cache has no extend(); it offers to_batch() instead.
Before the fix _to_batched_cache_layer left it as a singleton and the join died
with "'QSAKVCache' object has no attribute 'extend'".
"""

import pytest

from omlx.scheduler import _extend_cache_layer, _to_batched_cache_layer


class _FakeBatchCache:
    """Stands in for BatchQSAKVCache: knows how to absorb another batch row."""

    def __init__(self, rows):
        self.rows = list(rows)

    def extend(self, other):
        if not isinstance(other, _FakeBatchCache):
            raise TypeError(f"Cannot extend _FakeBatchCache with {type(other)}")
        self.rows.extend(other.rows)


class _FakeSingletonCache:
    """Stands in for QSAKVCache: no extend(), but a model-owned to_batch()."""

    def __init__(self, token):
        self.token = token

    def to_batch(self, left_padding):
        assert len(left_padding) == 1
        return _FakeBatchCache([self.token])


class _NoConversionCache:
    pass


def test_singleton_with_to_batch_is_converted():
    converted = _to_batched_cache_layer(_FakeSingletonCache("a"))
    assert isinstance(converted, _FakeBatchCache)
    assert converted.rows == ["a"]


def test_two_singletons_join_into_one_batch():
    joined = _extend_cache_layer(_FakeSingletonCache("a"), _FakeSingletonCache("b"))
    assert isinstance(joined, _FakeBatchCache)
    assert joined.rows == ["a", "b"]


def test_cache_without_to_batch_is_left_alone():
    cache = _NoConversionCache()
    assert _to_batched_cache_layer(cache) is cache


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
