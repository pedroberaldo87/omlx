"""O piso de bits da cabeça de previsão múltipla tem que dizer o preço.

`_MTP_MIN_BITS` mantém a cabeça em 4 bits mesmo quando o tronco vai a 2. A
razão está escrita e é boa: a cabeça só desenha rascunhos, todo token emitido é
verificado pelo tronco, e a aceitação que ela compra sai barata — "~2 GB no
DeepSeek-V4-Flash, ~15 MB no Qwen3.6".

Só que o piso era aplicado em SILÊNCIO, e a premissa "custa quase nada em
relação ao modelo" não vale sempre. Medido em 01/09 no GLM-5.3-Flash-oQ2e: a
cabeça saiu com 4,30 GB contra 2,43 GB de uma camada esparsa comum, o modelo
carregou a 101,90 GB contra um teto de aborto de 102,10, e o guarda de memória
passou a recusar prompt de 24 tokens. Os 1,89 GB que o piso acrescentou eram
exatamente a folga que faltava.

Quem liga "preservar MTP" não tem como adivinhar isso. O aviso não muda o piso
— muda o que a pessoa sabe ao escolher.
"""

from __future__ import annotations

import logging


def test_o_piso_eleva_os_bits_da_cabeca_acima_do_tronco():
    """O comportamento que a família precisa, e que fica de pé."""
    from omlx.oq import _MTP_MIN_BITS, _mtp_bits_override

    assert _mtp_bits_override(2) == _MTP_MIN_BITS, (
        "num tronco de 2 bits a cabeça tem que subir ao piso: ela desenha os "
        "rascunhos, e rascunho ruim é rascunho recusado"
    )
    assert _mtp_bits_override(8) == 8, "acima do piso, a escolha da calibração manda"
    assert _mtp_bits_override(_MTP_MIN_BITS) == _MTP_MIN_BITS


def test_o_piso_avisa_quanto_acrescentou(caplog):
    """Sem o aviso, o custo do piso é invisível até o guarda recusar o prompt.

    No GLM-5.3-Flash-oQ2e foram 1,89 GB — a diferença entre ter 2,08 GB de
    folga e ter 0,20 GB.
    """
    import omlx.oq as oq
    from omlx.oq import _mtp_bits_override

    # o aviso sai UMA vez por conversão; sem rearmar, quem rodar antes o consome
    oq._reinicia_aviso_do_piso_mtp()

    with caplog.at_level(logging.INFO, logger="omlx.oq"):
        _mtp_bits_override(2)

    ditas = " ".join(r.getMessage() for r in caplog.records)
    assert ditas.strip(), (
        "o piso subiu os bits da cabeça e não disse nada; quem ligou preservar "
        "MTP não tem como saber que o modelo acabou de engordar"
    )
    assert "4" in ditas and "2" in ditas, (
        f"o aviso tem que nomear de quanto para quanto subiu; saiu: {ditas!r}"
    )


def test_o_aviso_sai_uma_vez_so_e_nao_por_tensor():
    """A cabeça tem centenas de tensores; um aviso por tensor é ruído.

    O que interessa é a decisão, e ela é uma só por quantização.
    """
    from omlx.oq import _mtp_bits_override

    import omlx.oq as oq

    if hasattr(oq, "_reinicia_aviso_do_piso_mtp"):
        oq._reinicia_aviso_do_piso_mtp()

    registros = []

    class _Coletor(logging.Handler):
        def emit(self, r):
            registros.append(r.getMessage())

    lg = logging.getLogger("omlx.oq")
    h = _Coletor()
    nivel = lg.level
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    try:
        for _ in range(50):
            _mtp_bits_override(2)
    finally:
        lg.removeHandler(h)
        lg.setLevel(nivel)

    assert len(registros) == 1, (
        f"o aviso saiu {len(registros)} vezes; a cabeça tem centenas de "
        f"tensores e a decisão é uma só"
    )
