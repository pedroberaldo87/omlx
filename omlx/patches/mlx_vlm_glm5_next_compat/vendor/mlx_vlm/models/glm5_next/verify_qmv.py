"""Matmul quantizado afim de 8 bits para POUCAS linhas (2 a 8), lendo o peso UMA vez.

Por que existe. O ``mx.quantized_matmul`` escolhe o kernel pelo número de linhas
de entrada: abaixo de um limite (12 para o ``lm_head``, 18 para as projeções
deste modelo, no M1 Ultra) ele usa o ``qmv``, que trata cada linha como uma
multiplicação matriz-vetor separada e RELÊ o peso inteiro para cada uma. Medido
nesta máquina (01/09/2026, ms por chamada, peso de 8 bits g64):

    in_proj 4096x16384 (71 MB)   M=1 0,36  M=2 0,43  M=4 0,67  M=8 1,13  M=12 1,15
    lm_head 4096x154880 (674 MB) M=1 1,20  M=2 1,54  M=4 2,68  M=8 4,92  M=12 4,00

Cada linha a mais custa um peso inteiro relido (0,11 ms para 71 MB, que é
exatamente a banda de 645 GB/s). É o regime da VERIFICAÇÃO da previsão múltipla,
que roda 2 a 8 posições por passo: os 8,7 GB de pesos de 8 bits do GLM-5.3 eram
lidos uma vez por posição — ~13 ms por posição extra.

Este kernel espelha o ``qmv_fast`` do MLX (``quantized.h``: 2 simdgroups por
threadgroup, 4 linhas de saída por simdgroup, 8 valores consecutivos de K por
lane, blocos de 256) e acrescenta o laço sobre as M linhas de entrada, que ficam
em registrador em meia precisão. A conta é a mesma do ``qdot`` de fábrica —
``s * sum(wq*x) + b * sum(x)`` por bloco, acumulação em float — e a saída é
BIT A BIT igual à do ``mx.quantized_matmul`` (conferido para M de 1 a 8 nas
formas deste modelo). Medido (ms por chamada, fábrica → este):

    in_proj  M=2 0,52→0,46  M=4 0,59→0,47  M=6 0,90→0,71  M=8 1,25→1,00
    lm_head  M=2 1,91→1,42  M=4 2,63→1,80  M=6 3,77→2,39  M=8 4,91→3,70

Ainda longe da banda (a primeira linha custa o peso; cada linha extra devia
custar quase nada e custa 0,06–0,35 ms): a versão de ladrilhos 8x8 com
``simdgroup_matrix`` ficou plana em M mas com constante alta (in_proj 0,75,
lm_head 2,1) — está em ``scratchpad/bench_qmm8x8.py`` para a segunda rodada.

Só 8 bits, afim, ``transpose=True`` (o peso é (N, K)), sem viés de camada,
grupo 64 ou 128, K múltiplo de 256 e N múltiplo de 8 — o que todas as
projeções não-especialistas deste modelo cumprem. Fora disso ``elegivel`` diz
não e o chamador segue pelo caminho de fábrica. ``OMLX_GLM5_VERIFY_QMV=0``
desliga (para o A/B).
"""

from __future__ import annotations

import os

import mlx.core as mx

MIN_ROWS = 2
MAX_ROWS = 8
_ROWS_PER_TG = 8   # 2 simdgroups × 4 linhas de saída, como o qmv_fast do MLX
_BLOCK = 256       # colunas de K por bloco: 32 lanes × 8 valores
LIGADO = os.environ.get("OMLX_GLM5_VERIFY_QMV", "1") != "0"

_kernels: dict = {}

_SOURCE = """
    uint simd_gid = thread_index_in_threadgroup / 32;
    uint lane = thread_index_in_simdgroup;
    uint out_row = threadgroup_position_in_grid.y * 8 + simd_gid * 4;
    constexpr uint KG = K / GS;
    const device uchar* ws = ((const device uchar*)w) + out_row * K + lane * 8;
    const device T* sp = scales + out_row * KG + lane / (GS / 8);
    const device T* bp = biases + out_row * KG + lane / (GS / 8);
    const device T* xp = x + lane * 8;
    float result[4][M];
    #pragma unroll
    for (uint r = 0; r < 4; ++r)
        #pragma unroll
        for (uint m = 0; m < M; ++m) result[r][m] = 0.0f;
    for (uint k = 0; k < K; k += 256) {
        // as M linhas de entrada desta fatia ficam em meia precisao: metade
        // dos registradores, e e isso que evita o despenhadeiro em M >= 6
        T xt[M][8]; float xsum[M];
        #pragma unroll
        for (uint m = 0; m < M; ++m) {
            xsum[m] = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 8; ++i) { xt[m][i] = xp[m * K + i]; xsum[m] += float(xt[m][i]); }
        }
        #pragma unroll
        for (uint r = 0; r < 4; ++r) {
            const device uchar* wl = ws + r * K;
            uchar wq[8];
            #pragma unroll
            for (uint i = 0; i < 8; ++i) wq[i] = wl[i];
            float s = float(sp[r * KG]); float b = float(bp[r * KG]);
            #pragma unroll
            for (uint m = 0; m < M; ++m) {
                float acc = 0.0f;
                #pragma unroll
                for (uint i = 0; i < 8; ++i) acc += float(wq[i]) * float(xt[m][i]);
                result[r][m] += s * acc + b * xsum[m];
            }
        }
        ws += 256; sp += 256 / GS; bp += 256 / GS; xp += 256;
    }
    #pragma unroll
    for (uint r = 0; r < 4; ++r)
        #pragma unroll
        for (uint m = 0; m < M; ++m) {
            float v = simd_sum(result[r][m]);
            if (lane == 0) out[m * N + out_row + r] = static_cast<T>(v);
        }
"""


def _kernel():
    k = _kernels.get("k")
    if k is None:
        k = mx.fast.metal_kernel(
            name="omlx_glm5_verify_qmv8",
            input_names=["x", "w", "scales", "biases"],
            output_names=["out"],
            source=_SOURCE,
        )
        _kernels["k"] = k
    return k


def elegivel(x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array,
             *, bits: int, group_size: int) -> bool:
    if not LIGADO or bits != 8 or group_size not in (64, 128):
        return False
    if x.ndim < 2 or weight.ndim != 2 or weight.dtype != mx.uint32:
        return False
    if x.dtype not in (mx.float16, mx.bfloat16) or scales.dtype != x.dtype or biases.dtype != x.dtype:
        return False
    m = 1
    for d in x.shape[:-1]:
        m *= d
    if not (MIN_ROWS <= m <= MAX_ROWS):
        return False
    K = x.shape[-1]
    N = weight.shape[0]
    return K % _BLOCK == 0 and N % _ROWS_PER_TG == 0 and weight.shape[1] * 4 == K


def verify_qmv8(x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array,
                *, group_size: int) -> mx.array:
    """``x @ W.T`` para x com 2 a 8 linhas, lendo W uma vez. Quem chama confere `elegivel`."""
    forma = x.shape
    K = forma[-1]
    N = weight.shape[0]
    x2 = mx.contiguous(x.reshape(-1, K))
    M = x2.shape[0]
    out = _kernel()(
        inputs=[x2, weight, scales, biases],
        template=[("T", x.dtype), ("M", M), ("K", K), ("N", N), ("GS", group_size)],
        grid=(64, N // _ROWS_PER_TG, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(*forma[:-1], N)
