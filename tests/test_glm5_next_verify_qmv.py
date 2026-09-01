# SPDX-License-Identifier: Apache-2.0
"""O matmul de 8 bits para 2–8 linhas (a janela da verificação) é bit a bit igual
ao de fábrica e é o que `linear_forward` usa nesse regime."""
import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except Exception:  # noqa: BLE001
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="precisa de MLX")


def _mod():
    from omlx.patches.mlx_lm_glm5_next import register_into_mlx_lm

    register_into_mlx_lm()
    import sys

    from omlx.patches import mlx_lm_glm5_next

    _, language_model = mlx_lm_glm5_next._vendored()
    modulo = sys.modules[language_model.__module__]
    linear = sys.modules[modulo.linear_forward.__module__]
    return linear, sys.modules[linear.verify_qmv8.__module__]


@pytest.mark.parametrize("m", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("k_n", [(4096, 2048), (1536, 4096), (4096, 8)])
def test_bit_a_bit_igual_ao_de_fabrica(m, k_n):
    _linear, vq = _mod()
    k, n = k_n
    w = mx.random.normal((n, k)).astype(mx.float16)
    wq, sc, bi = mx.quantize(w, group_size=64, bits=8)
    x = (mx.random.normal((m, k)) * 0.5).astype(mx.float16)
    assert vq.elegivel(x, wq, sc, bi, bits=8, group_size=64)
    ref = mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=64, bits=8)
    novo = vq.verify_qmv8(x, wq, sc, bi, group_size=64)
    mx.eval(ref, novo)
    assert novo.shape == ref.shape
    assert bool(mx.array_equal(ref, novo).item()), (
        "a saída tem que ser IDÊNTICA à de fábrica — mesma conta, mesma ordem"
    )


def test_fora_do_regime_nao_e_elegivel():
    _linear, vq = _mod()
    w = mx.random.normal((2048, 4096)).astype(mx.float16)
    wq, sc, bi = mx.quantize(w, group_size=64, bits=8)
    um = mx.zeros((1, 4096), dtype=mx.float16)
    nove = mx.zeros((9, 4096), dtype=mx.float16)
    assert not vq.elegivel(um, wq, sc, bi, bits=8, group_size=64), "1 linha é do qmv de fábrica"
    assert not vq.elegivel(nove, wq, sc, bi, bits=8, group_size=64), "acima de 8 é do qmm_t de fábrica"
    wq2, sc2, bi2 = mx.quantize(w, group_size=64, bits=2)
    assert not vq.elegivel(mx.zeros((4, 4096), dtype=mx.float16), wq2, sc2, bi2, bits=2, group_size=64)


def test_linear_forward_usa_o_kernel_na_janela(monkeypatch):
    """Mutação: se o kernel não for chamado com 4 linhas, o teste não mede nada."""
    linear, vq = _mod()
    camada = nn.QuantizedLinear(4096, 2048, bias=False, group_size=64, bits=8)
    camada.set_dtype(mx.float16)  # como no checkpoint: escalas e vieses em fp16
    mx.eval(camada.parameters())
    chamadas = []

    def espiao(x, w, s, b, *, group_size):
        chamadas.append(x.shape)
        return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=group_size, bits=8)

    monkeypatch.setattr(linear, "verify_qmv8", espiao)
    x4 = mx.zeros((1, 4, 4096), dtype=mx.float16)
    mx.eval(linear.linear_forward(camada, x4))
    assert chamadas == [(1, 4, 4096)], "4 linhas de 8 bits têm que passar pelo kernel"
    chamadas.clear()
    mx.eval(linear.linear_forward(camada, mx.zeros((1, 1, 4096), dtype=mx.float16)))
    assert chamadas == [], "1 linha fica com o qmv de fábrica"
