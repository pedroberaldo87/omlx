"""O cache de bytecode tem que apontar para arquivos que existem.

02/09: seis testes falhavam ha semanas e a doc do projeto os listava como
"anteriores, nao sao regressao". Nao eram falha nenhuma: o repositorio foi
copiado de outro caminho COM a pasta __pycache__ junto e as datas
preservadas. O Python valida o cache por data e tamanho, os dois batiam, e
ele usava o cache — que carrega dentro de si o caminho antigo. Quem le o
proprio texto-fonte para se validar (omlx/cluster/tensor_strategies.py e
runtime_optimizations.py usam inspect.getsource) nao achava o arquivo e
recusava por precaucao; os testes viam a recusa e falhavam.

O git ignora __pycache__, entao nada disso aparecia. Este cobrador aparece.
"""

import os

import pytest


def test_nenhum_modulo_do_omlx_aponta_para_arquivo_que_nao_existe():
    import importlib
    import sys

    # Importar alguns modulos que leem o proprio fonte em producao.
    for nome in (
        "omlx.cluster.tensor_strategies",
        "omlx.cluster.runtime_optimizations",
        "omlx.patches.qwen3_6_nested_visual",
    ):
        try:
            importlib.import_module(nome)
        except Exception as e:  # pragma: no cover - ambiente sem a dependencia
            pytest.skip(f"{nome} nao importa neste ambiente: {e}")

    fantasmas = []
    for nome, modulo in list(sys.modules.items()):
        if not nome.startswith("omlx"):
            continue
        arquivo = getattr(modulo, "__file__", None)
        if not arquivo or not arquivo.endswith(".py"):
            continue
        gravado = getattr(getattr(modulo, "__loader__", None), "path", None) or arquivo
        codigo = getattr(getattr(modulo, "__spec__", None), "loader", None)
        del codigo
        if not os.path.exists(gravado):
            fantasmas.append(f"{nome}: {gravado}")

    assert not fantasmas, (
        "modulo(s) carregado(s) de caminho inexistente — cache de bytecode de "
        "outro checkout. Apague: find . -name __pycache__ -type d -prune "
        f"-exec rm -rf {{}} +\n" + "\n".join(fantasmas)
    )


def test_o_texto_fonte_dos_modulos_que_se_inspecionam_esta_legivel():
    """A trava do fatiamento por camada le o proprio fonte antes de deixar
    rodar. Sem fonte legivel ela recusa — e foi isso que quebrou seis testes."""
    import inspect

    alvos = []
    try:
        from omlx.cluster import runtime_optimizations, tensor_strategies

        alvos = [tensor_strategies, runtime_optimizations]
    except Exception as e:  # pragma: no cover
        pytest.skip(f"modulos de cluster nao importam: {e}")

    for modulo in alvos:
        fonte = inspect.getsource(modulo)
        assert len(fonte) > 100, f"{modulo.__name__}: fonte vazio ou ilegivel"
