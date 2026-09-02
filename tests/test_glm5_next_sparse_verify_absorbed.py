# SPDX-License-Identifier: Apache-2.0
"""A atenção esparsa do GLM-5.x na janela de verificação (2 a 8 posições, contexto
abaixo do limiar do seletor) usa a forma ABSORVIDA, como o decode de 1 posição?

Medido no modelo real (oQ2e, 01/09/2026): com o contexto em 1500 tokens — abaixo
de ``index_topk`` = 2048, onde o seletor é pulado e ``topk_indices`` é None — a
camada esparsa custava 1,05 ms em 1 posição e 5,9 ms em 2 posições. O caminho de
L > 1 projetava o CONTEXTO INTEIRO pelas 64 cabeças (``embed_q(kv)`` e
``unembed_out(kv)``) a cada passo, quando bastava projetar as L consultas para
o espaço latente, como o caminho de L = 1 já fazia. Nas 11 camadas esparsas isso
somava ~53 ms por passo de verificação — mais que o passo inteiro de decode.

A conta é a mesma (``q·(Wk) = (Wᵀq)·k`` e ``Σp·(Uk) = U·Σp·k``); o que se confere
aqui é que os dois caminhos batem dentro do ruído de fp16 e que a projeção do
contexto não é mais chamada para L pequeno.
"""
from __future__ import annotations

import pytest

try:
    import mlx.core as mx

    from omlx.patches.mlx_vlm_glm5_next_compat import (
        apply_mlx_vlm_glm5_next_compat_patch,
    )

    apply_mlx_vlm_glm5_next_compat_patch()
    from mlx_vlm.models.base import create_attention_mask
    from mlx_vlm.models.glm5_next import language as lang
    from tests.test_mlx_vlm_glm5_next_compat import _tiny_config

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


def _camada_esparsa_com_contexto(ctx: int):
    mx.random.seed(3)
    cfg = _tiny_config().text_config
    cfg.index_topk = 2048  # o seletor fica pulado: contexto curto = topk None
    attn = lang.Glm5NextSparseAttention(cfg)
    mx.eval(attn.parameters())
    from mlx_lm.models.cache import PoolingCache

    cache = lang.CacheList(lang.KVCache(), PoolingCache(attn.indexer.index_kpool))
    x0 = mx.random.normal((1, ctx, cfg.hidden_size)).astype(mx.float16)
    mx.eval(attn(x0, create_attention_mask(x0, cache[0], return_array=True), cache))
    return attn, cache, cfg


def _referencia_projetando_o_contexto(attn, x, mask, cache):
    """O caminho antigo de L > 1, escrito por extenso: k e v projetados do contexto."""
    B, L, _ = x.shape
    qr = attn.q_a_layernorm(lang.linear_forward(attn.q_a_proj, x))
    q = lang.linear_forward(attn.q_b_proj, qr)
    q = q.reshape(B, L, attn.num_heads, attn.q_head_dim).transpose(0, 2, 1, 3)
    kv_latent = attn.kv_a_layernorm(lang.linear_forward(attn.kv_a_proj_with_mqa, x))
    kv_latent = mx.expand_dims(kv_latent, axis=1)
    kv_latent, _ = cache[0].update_and_fetch(
        kv_latent, mx.zeros((B, 1, L, 0), dtype=kv_latent.dtype)
    )
    k = attn.embed_q(kv_latent, transpose=False)
    v = attn.unembed_out(kv_latent)
    out = lang.scaled_dot_product_attention(
        q, k, v, cache=cache, scale=attn.scale, mask=mask
    )
    out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return lang.linear_forward(attn.o_proj, out)


@pytest.mark.parametrize("L", [2, 4, 8])
def test_a_janela_de_verificacao_bate_com_a_projecao_do_contexto(L):
    attn, cache, cfg = _camada_esparsa_com_contexto(40)
    x = mx.random.normal((1, L, cfg.hidden_size)).astype(mx.float16)
    mask = create_attention_mask(x, cache[0], return_array=True)

    ref = _referencia_projetando_o_contexto(attn, x, mask, cache)
    cache[0].trim(L)
    cache[1].pooled = None  # o arreio da referência não toca o seletor; zera o que sobrou
    got = attn(x, mask, cache)
    mx.eval(ref, got)

    assert got.shape == ref.shape
    escala = float(mx.max(mx.abs(ref.astype(mx.float32))).item())
    dif = float(mx.max(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32))).item())
    assert dif <= 2e-2 * max(escala, 1.0), (dif, escala)


def test_a_janela_de_verificacao_nao_projeta_o_contexto(monkeypatch):
    attn, cache, cfg = _camada_esparsa_com_contexto(40)
    x = mx.random.normal((1, 4, cfg.hidden_size)).astype(mx.float16)
    mask = create_attention_mask(x, cache[0], return_array=True)

    chamadas = []
    original = attn.embed_q.__call__

    def espiao(inp, transpose=True):
        chamadas.append((tuple(inp.shape), transpose))
        return original(inp, transpose=transpose)

    monkeypatch.setattr(type(attn.embed_q), "__call__", lambda self, inp, transpose=True: espiao(inp, transpose))
    mx.eval(attn(x, mask, cache))
    assert chamadas, "embed_q nunca foi chamado"
    assert all(t for _, t in chamadas), (
        "embed_q(kv, transpose=False) projeta o contexto inteiro por passo: "
        f"{chamadas}"
    )
