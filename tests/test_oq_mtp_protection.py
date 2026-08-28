# SPDX-License-Identifier: Apache-2.0
"""v8 F5.1 — os tensores de fusao do MTP do qwen4_exp ficam fora da
quantizacao agressiva, como os equivalentes das outras familias."""

from omlx.oq import _is_mtp_protected_tensor


def test_qwen4_exp_fusion_tensors_sao_protegidos():
    protegidos = [
        "language_model.model.mtp.fc_embedding.weight",
        "language_model.model.mtp.fc_hidden.weight",
        "language_model.model.mtp.hyper_connection_mixer.input_mix_weight_a",
    ]
    for nome in protegidos:
        assert _is_mtp_protected_tensor(nome), nome


def test_tensor_comum_do_tronco_nao_e_protegido():
    assert not _is_mtp_protected_tensor("language_model.model.layers.0.mlp.gate.weight")
    # e um nome parecido FORA do bloco mtp tambem nao
    assert not _is_mtp_protected_tensor("language_model.model.fc_embedding.weight")
