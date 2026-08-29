# SPDX-License-Identifier: Apache-2.0
"""Os tensores de fusao do MTP do qwen4_exp NAO ficam fora da quantizacao.

A protecao existia por analogia com as outras familias, nunca por medicao.
Um checkpoint que carrega as quatro camadas comprimidas aceita entre 69,6%
e 80,1% em oito amostras, entao o colapso de ~0% medido num Qwen3.5-27B nao
vale aqui.
"""

from omlx.oq import _is_mtp_protected_tensor


def test_qwen4_exp_fusion_tensors_nao_sao_protegidos():
    nao_protegidos = [
        "language_model.model.mtp.fc_embedding.weight",
        "language_model.model.mtp.fc_hidden.weight",
        "language_model.model.mtp.hyper_connection_mixer.input_mix_weight_a",
    ]
    for nome in nao_protegidos:
        assert not _is_mtp_protected_tensor(nome), nome


def test_as_outras_familias_seguem_protegidas():
    protegidos = [
        "model.mtp.fc.weight",
        "model.mtp.e_proj.weight",
        "model.mtp.h_proj.weight",
        "model.mtp.eh_proj.weight",
        "model.mtp.main_proj.weight",
        "model.mtp.hc_head.weight",
        "model.mtp.markov_head.weight",
        "model.mtp.confidence_head.weight",
    ]
    for nome in protegidos:
        assert _is_mtp_protected_tensor(nome), nome


def test_tensor_comum_do_tronco_nao_e_protegido():
    assert not _is_mtp_protected_tensor("language_model.model.layers.0.mlp.gate.weight")
    assert not _is_mtp_protected_tensor("language_model.model.fc_embedding.weight")
