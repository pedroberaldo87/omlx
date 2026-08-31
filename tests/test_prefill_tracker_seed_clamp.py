"""A primeira amostra do medidor de transiente também precisa de teto.

O medidor aprende quantos bytes cada palavra do prompt custa, observando o
crescimento de memória durante o preparo. A primeira observação depois de
carregar um modelo carrega o resíduo do próprio carregamento — o código já
protege o máximo observado dela, mas deixava essa mesma leitura definir a
média, e aí nenhuma leitura seguinte conseguia puxar a média de volta.

Medido no GLM-5.3-Flash-oQ2e: uma semente de 10 GB num pedaço de 2048 palavras
fixou 4,9 MB por palavra contra 0,38 MB reais, e a admissão passou a cobrar
12,75 GB por um prompt que custa 1,02 GB — recusando toda requisição.
"""

from omlx.prefill_transient_tracker import PrefillTransientTracker

GB = 1024**3
MB = 1024**2


def _medidor():
    return PrefillTransientTracker(model_id="teste")


def test_semente_dentro_do_teto_entra_normalmente():
    m = _medidor()
    m.update(2048, 512 * MB)
    assert m.bytes_per_token == 512 * MB / 2048
    assert m.samples == 1


def test_semente_acima_do_teto_nao_define_a_media():
    m = _medidor()
    m.update(2048, 10 * GB)
    assert m.bytes_per_token == 0.0
    assert m.samples == 0


def test_semente_rejeitada_nao_deixa_rastro_no_ultimo_delta():
    """O previsor lê o último delta como taxa própria — o valor sujo não pode ficar."""
    m = _medidor()
    m.update(2048, 10 * GB)
    assert m.last_delta_bytes == 0
    assert m.last_n_tokens == 0


def test_depois_da_rejeicao_a_proxima_amostra_semeia():
    m = _medidor()
    m.update(2048, 10 * GB)
    m.update(2048, 400 * MB)
    assert m.bytes_per_token == 400 * MB / 2048
    assert m.samples == 1


def test_a_protecao_antiga_contra_fora_da_curva_continua_valendo():
    m = _medidor()
    m.update(2048, 100 * MB)
    base = m.bytes_per_token
    m.update(2048, 3 * GB)
    assert m.bytes_per_token == base
