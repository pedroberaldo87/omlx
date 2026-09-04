# SPDX-License-Identifier: Apache-2.0
"""A fila de acurácia espera o ledger do kernel assentar entre um item e o seguinte.

Medido em 03/09: a descarga reportou `freed=106,72GB, active_memory: 433,66MB
(settled)` às 15:06:25,491 e a fila pediu o modelo seguinte no MESMO
milissegundo. A admissão compara o teto contra
`max(active_memory, phys_footprint, contabilizado)`, e o phys_footprint — o
ledger do macOS — ainda contava 75,91 GB de páginas recuperáveis do modelo que
acabara de sair. Dois itens morreram com InsufficientMemoryError, com a máquina
de fato vazia; 92 s depois a mesma carga passou sozinha.
"""

import asyncio

import pytest

from omlx.admin import accuracy_benchmark as ab


@pytest.mark.asyncio
async def test_espera_ate_o_ledger_cair(monkeypatch):
    """Enquanto o footprint estiver alto, a fila segura; quando cai, ela segue."""
    leituras = iter([80 * 1024**3, 60 * 1024**3, 30 * 1024**3, 1 * 1024**3])
    vistos = []

    def _footprint():
        v = next(leituras)
        vistos.append(v)
        return v

    monkeypatch.setattr("omlx.utils.proc_memory.get_phys_footprint", _footprint)
    _sleep_real = asyncio.sleep
    monkeypatch.setattr(ab.asyncio, "sleep", lambda _s: _sleep_real(0))

    class _MX:
        @staticmethod
        def get_active_memory():
            return 400 * 1024**2

    monkeypatch.setitem(__import__("sys").modules, "mlx.core", _MX)
    await ab._esperar_memoria_assentar(None, teto_s=5.0)
    # esperou as leituras altas e parou na que assentou (1 GB <= 0,4 + 8)
    assert len(vistos) == 4, vistos


@pytest.mark.asyncio
async def test_nao_espera_quando_ja_esta_baixo(monkeypatch):
    """Com o ledger já baixo, não custa nada — a fila não perde tempo."""
    chamadas = []

    def _footprint():
        chamadas.append(1)
        return 500 * 1024**2

    monkeypatch.setattr("omlx.utils.proc_memory.get_phys_footprint", _footprint)

    class _MX:
        @staticmethod
        def get_active_memory():
            return 400 * 1024**2

    monkeypatch.setitem(__import__("sys").modules, "mlx.core", _MX)
    await ab._esperar_memoria_assentar(None, teto_s=5.0)
    assert len(chamadas) == 1


@pytest.mark.asyncio
async def test_desiste_quando_para_de_cair(monkeypatch):
    """Se o ledger estabiliza alto, não adianta insistir: a admissão tem o
    próprio caminho de despejo, e segurar a fila para sempre é pior."""
    leituras = iter([80 * 1024**3, 79 * 1024**3, 79 * 1024**3, 79 * 1024**3])
    vistos = []

    def _footprint():
        v = next(leituras)
        vistos.append(v)
        return v

    monkeypatch.setattr("omlx.utils.proc_memory.get_phys_footprint", _footprint)
    _sleep_real = asyncio.sleep
    monkeypatch.setattr(ab.asyncio, "sleep", lambda _s: _sleep_real(0))

    class _MX:
        @staticmethod
        def get_active_memory():
            return 400 * 1024**2

    monkeypatch.setitem(__import__("sys").modules, "mlx.core", _MX)
    await ab._esperar_memoria_assentar(None, teto_s=5.0)
    assert len(vistos) == 3, vistos
