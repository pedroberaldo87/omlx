# SPDX-License-Identifier: Apache-2.0
"""A barreira de ordenação precisa valer para TODOS os caminhos de atenção.

A camada esparsa do GLM-5.3 mantém dois estados que precisam andar juntos: o KV
da atenção e o acumulado de compressão que o seletor usa para decidir QUAIS
pedaços do contexto o modelo vai ler. Uma barreira amarra os dois, obrigando o
KV a depender do acumulado.

Ela estava depois de três saídas antecipadas, então três dos quatro caminhos
saíam sem passar por ela. Enquanto nada avalia o grafo no meio da geração isso
não aparece; ligar o cache de prefixo liga a captura de retratos de fronteira,
que avalia no meio — e aí o seletor passa a escolher com base num acumulado
dessincronizado do KV.

Medido em 31/08/2026, ligando e desligando o cache: 4 falhas de envelope de
ferramenta em 6 oportunidades com ele ligado, 0 em 21 com ele desligado, 2 em 3
ao religar (Fisher exato bicaudal, p = 0,00014).

A implementação irmã (omlx/patches/glm_moe_dsa/deepseek_v32.py) põe a barreira
antes do seu único retorno. Este teste cobra o mesmo aqui.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ALVO = pathlib.Path(__file__).resolve().parents[1] / (
    "omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next/language.py"
)

pytestmark = pytest.mark.skipif(not ALVO.exists(), reason="vendor glm5_next ausente")


def _funcao_da_barreira() -> ast.FunctionDef:
    """A função de atenção que contém a barreira mx.depends."""
    arvore = ast.parse(ALVO.read_text(encoding="utf-8"))
    achadas = []
    for no in ast.walk(arvore):
        if not isinstance(no, ast.FunctionDef):
            continue
        for interno in ast.walk(no):
            if (
                isinstance(interno, ast.Call)
                and isinstance(interno.func, ast.Attribute)
                and interno.func.attr == "depends"
            ):
                achadas.append((no, interno))
                break
    assert achadas, "nenhuma barreira mx.depends encontrada no arquivo"
    # a barreira do cache esparso vive numa função só
    return achadas[0]


def test_a_barreira_existe():
    fn, chamada = _funcao_da_barreira()
    assert chamada.lineno > 0
    assert fn.name == "__call__", f"a barreira mudou de função: {fn.name}"


def test_nenhuma_saida_antecipada_pula_a_barreira():
    """O invariante: nenhum retorno pode acontecer antes da barreira."""
    fn, chamada = _funcao_da_barreira()
    linha_barreira = chamada.lineno

    antes = [
        no.lineno
        for no in ast.walk(fn)
        if isinstance(no, ast.Return) and no.lineno < linha_barreira
    ]
    assert not antes, (
        f"{len(antes)} saída(s) da função acontecem ANTES da barreira "
        f"(linha {linha_barreira}): linhas {sorted(antes)}. Esses caminhos "
        "devolvem sem amarrar o KV ao acumulado de compressão, e com o cache "
        "de prefixo ligado a captura de retratos avalia o grafo no meio da "
        "geração — o seletor passa a ler o pedaço errado do contexto. "
        "A implementação irmã em deepseek_v32.py põe a barreira antes do seu "
        "único retorno; faça o mesmo aqui."
    )


def test_a_irma_continua_correta():
    """Regressão do molde: na irmã, a barreira vem antes do retorno."""
    irma = pathlib.Path(__file__).resolve().parents[1] / (
        "omlx/patches/glm_moe_dsa/deepseek_v32.py"
    )
    if not irma.exists():
        pytest.skip("deepseek_v32.py ausente")
    arvore = ast.parse(irma.read_text(encoding="utf-8"))
    for no in ast.walk(arvore):
        if not isinstance(no, ast.FunctionDef):
            continue
        barreira = None
        for interno in ast.walk(no):
            if (
                isinstance(interno, ast.Call)
                and isinstance(interno.func, ast.Attribute)
                and interno.func.attr == "depends"
            ):
                barreira = interno.lineno
                break
        if barreira is None:
            continue
        antes = [
            n.lineno
            for n in ast.walk(no)
            if isinstance(n, ast.Return) and n.lineno < barreira
        ]
        assert not antes, (
            f"a irmã {no.name} regrediu: retornos {sorted(antes)} antes da "
            f"barreira na linha {barreira}"
        )
