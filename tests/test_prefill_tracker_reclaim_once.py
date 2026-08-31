"""A devolução de memória tem que ser cobrada UMA vez, não somada sem fim.

Quando um pedaço do preparo termina ocupando MENOS memória do que antes, o
medidor guarda essa diferença: o desenho de memória pode precisar pedir aquele
espaço de volta no pedaço seguinte, e o guarda cobra esse risco adiantado.

O comentário do campo diz, com todas as letras, que isso se cobra **uma vez**
("prices it once until a positive measurement confirms reallocation"). O código
somava. Cada devolução empilhava na anterior, e como a soma só zera quando um
pedaço cresce, uma conversa curta que devolve memória várias vezes deixava a
cobrança armada para a conversa seguinte.

Medido no GLM-5.3-Flash-oQ2e em 31/08: com o mesmo pedaço de 2048 palavras e o
mesmo tamanho de contexto, a cobrança prevista foi de 1,02 GB (limpa) a 23,99 GB
(suja). O termo calculado vale 1,02 GB; os outros 22,97 GB eram devoluções
empilhadas. Com o teto da máquina em 124 GB e o modelo ocupando 107 GB, isso
recusa todo prompt longo.
"""

from omlx.prefill_transient_tracker import PrefillTransientTracker

GB = 1024**3
MB = 1024**2


def _medidor():
    return PrefillTransientTracker(model_id="teste")


def test_uma_devolucao_e_cobrada_pelo_proprio_valor():
    m = _medidor()
    m.record_reclaim(512 * MB)
    assert m.recent_reclaim_bytes == 512 * MB


def test_devolucoes_seguidas_nao_se_somam():
    """Vinte devoluções de 512 MB não viram 10 GB a devolver de uma vez."""
    m = _medidor()
    for _ in range(20):
        m.record_reclaim(512 * MB)
    assert m.recent_reclaim_bytes == 512 * MB


def test_a_maior_devolucao_e_a_que_vale():
    m = _medidor()
    m.record_reclaim(200 * MB)
    m.record_reclaim(1500 * MB)
    m.record_reclaim(300 * MB)
    assert m.recent_reclaim_bytes == 1500 * MB


def test_um_pedaco_que_cresce_zera_a_cobranca():
    """Regressão: o comportamento que já existia continua valendo."""
    m = _medidor()
    m.record_reclaim(800 * MB)
    m.clear_reclaim()
    assert m.recent_reclaim_bytes == 0


def test_pedaco_positivo_zera_a_cobranca_pelo_update():
    m = _medidor()
    m.record_reclaim(800 * MB)
    m.update(2048, 512 * MB)
    assert m.recent_reclaim_bytes == 0


def test_devolucao_negativa_ou_zero_nao_conta():
    m = _medidor()
    m.record_reclaim(0)
    m.record_reclaim(-100 * MB)
    assert m.recent_reclaim_bytes == 0


def test_o_caso_real_de_31_08():
    """A conversa curta devolve memória várias vezes; a longa vinha depois.

    Antes: 22,97 GB de cobrança. Depois: o maior evento único, 1,52 GB —
    que é o que o registro do servidor mostrou ser reclamado de uma vez.
    """
    m = _medidor()
    for devolucao in (526 * MB, 1520 * MB, 900 * MB, 480 * MB, 1200 * MB):
        m.record_reclaim(devolucao)
    assert m.recent_reclaim_bytes == 1520 * MB
    assert m.recent_reclaim_bytes < 2 * GB
