# SPDX-License-Identifier: Apache-2.0
"""Uma pausa por expulsão no MEIO do prefill externo não pode perder o progresso.

O laço do prefill externo dimensiona cada pedaço com o estrangulamento
adaptativo, que pode levantar ``_PrefillEvictionNeeded`` — inclusive DEPOIS
de pedaços já terem entrado no cache (o objeto ``prompt_cache`` é avançado no
lugar). A pausa devolvia o pedido à fila com ``remaining_tokens`` e
``cached_tokens`` do ponto de partida; na retomada, os mesmos tokens eram
alimentados de novo sobre o cache já avançado: um trecho DUPLICADO no KV, os
retratos de fronteira rotulados um bloco atrás do que o cache tinha
(``captura tc=7680 … kv=8192``, medido em 02/09 no oQ2e) e todos os blocos
gravados dali em diante desalinhados — a próxima rodada que acertasse o cache
saía lixo ("NoNo", "DEDEDE"). Só aparecia com a cabeça MTP ligada porque só
ela apertava a memória a ponto de estrangular.
"""
from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig, _PrefillEvictionNeeded


class _ModeloDeContagem:
    """Só empurra posições num KVCache: o que importa é o offset."""

    def __init__(self):
        self.layers = [SimpleNamespace()]
        self.args = SimpleNamespace(num_hidden_layers=1)

    def __call__(self, inputs, cache=None, **kwargs):
        S = inputs.shape[1]
        cache[0].update_and_fetch(mx.zeros((1, 1, S, 4)), mx.zeros((1, 1, S, 4)))
        return mx.zeros((1, S, 8))

    def make_cache(self):
        return [KVCache()]

    def parameters(self):
        return {}


def test_pausa_no_meio_do_prefill_externo_guarda_o_progresso(mock_tokenizer):
    modelo = _ModeloDeContagem()
    scheduler = Scheduler(
        model=modelo, tokenizer=mock_tokenizer, config=SchedulerConfig(prefill_step_size=4)
    )
    # prefixo restaurado: 4 tokens já no cache
    cache = modelo.make_cache()
    modelo(mx.zeros((1, 4), dtype=mx.int32), cache=cache)
    prompt = list(range(100, 116))  # 16 tokens; faltam 12 (o último fica para o gerador)
    request = Request(request_id="req-pausa", prompt=prompt, sampling_params=SamplingParams())
    request.prompt_token_ids = prompt
    request.num_prompt_tokens = len(prompt)
    request.cached_tokens = 4
    request.remaining_tokens = prompt[4:]
    request.prompt_cache = cache
    scheduler.requests[request.request_id] = request

    chamadas = {"n": 0}
    original = scheduler._adaptive_chunk_size

    def estrangula_no_segundo(requested, **kw):
        chamadas["n"] += 1
        if chamadas["n"] == 2:
            raise _PrefillEvictionNeeded(
                SimpleNamespace(reason="adaptive_prefill_throttle", request_id=request.request_id)
            )
        return original(requested, **kw)

    scheduler._adaptive_chunk_size = estrangula_no_segundo

    with pytest.raises(_PrefillEvictionNeeded):
        scheduler._do_external_prefill(request, request.remaining_tokens, cache)

    # o cache avançou UM pedaço (4 tokens) antes da pausa
    assert cache[0].offset == 8
    # e o pedido tem que saber disso: o que falta começa DEPOIS do que já entrou
    assert request.cached_tokens == 8, request.cached_tokens
    assert request.remaining_tokens == prompt[8:], request.remaining_tokens
    assert request.prompt_cache is cache


def test_pausa_num_prompt_FRIO_tambem_guarda_o_progresso(mock_tokenizer):
    """O caso que faltou: sem cache de prefixo, a pausa recomeçava do token zero.

    Medido em 05/09 no GLM-5.3-Flash oQ2e: um prompt frio de 229.923 tokens foi
    pausado aos 206.336 por falta de folga (111,76 GB contra o alvo de 112,48),
    o despejo recuperou 1,56 GB, e o pedido voltou como novo (new=229923,
    kv_exact=0) — 19 minutos de preparo jogados fora, sem nada que impedisse o
    mesmo aos 206k de novo. O cache construído localmente tem que ir para o
    pedido, que é de onde a retomada o lê.
    """
    modelo = _ModeloDeContagem()
    scheduler = Scheduler(
        model=modelo, tokenizer=mock_tokenizer, config=SchedulerConfig(prefill_step_size=4)
    )
    prompt = list(range(100, 116))
    request = Request(request_id="req-frio", prompt=prompt, sampling_params=SamplingParams())
    request.prompt_token_ids = prompt
    request.num_prompt_tokens = len(prompt)
    request.cached_tokens = 0
    request.remaining_tokens = prompt
    request.prompt_cache = None
    scheduler.requests[request.request_id] = request

    chamadas = {"n": 0}
    original = scheduler._adaptive_chunk_size

    def estrangula_no_terceiro(requested, **kw):
        chamadas["n"] += 1
        if chamadas["n"] == 3:
            raise _PrefillEvictionNeeded(
                SimpleNamespace(reason="adaptive_prefill_throttle", request_id=request.request_id)
            )
        return original(requested, **kw)

    scheduler._adaptive_chunk_size = estrangula_no_terceiro
    with pytest.raises(_PrefillEvictionNeeded):
        scheduler._do_external_prefill(request, prompt, None)

    # dois pedaços de 4 entraram antes da pausa; o cache tem que estar NO PEDIDO
    assert request.prompt_cache is not None
    assert request.prompt_cache[0].offset == 8
    assert request.cached_tokens == 8, request.cached_tokens
    assert request.remaining_tokens == prompt[8:], request.remaining_tokens
