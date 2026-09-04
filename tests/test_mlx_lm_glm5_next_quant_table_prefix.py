# SPDX-License-Identifier: Apache-2.0
"""A tabela de bits por tensor de um checkpoint no layout de visão (``language_model.*``)
tem que chegar ao mlx-lm pelo caminho do módulo, senão todo override é ignorado.

Medido em 03/09 no GLM-5.3-Flash-oQ2e: o predicado do mlx-lm não achava
``model.layers.3.self_attn.embed_q`` (a tabela dizia ``language_model.model.layers.3...``),
caía no padrão de 2 bits e um tensor de 8 bits morria no dequantize com
"Shape of scales and biases does not match the matrix (1,512,1024)".
"""
import json
from pathlib import Path

import pytest

from omlx.patches.mlx_lm_glm5_next import ModelArgs, _module_key, _strip_language_prefix


def test_prefixo_e_tirado_da_tabela_de_bits():
    real = Path.home() / ".omlx/models/GLM-5.3-Flash-oQ2e-fp16-mtp/config.json"
    if not real.exists():
        pytest.skip("checkpoint real ausente")
    config = {
        "text_config": json.load(open(real))["text_config"],
        "quantization": {
            "bits": 2, "group_size": 64, "mode": "affine",
            "language_model.lm_head": {"bits": 8, "group_size": 64},
            "language_model.model.layers.3.self_attn.embed_q": {"bits": 8, "group_size": 64},
            "model.language_model.model.embed_tokens": {"bits": 8, "group_size": 64},
            "language_model.model.layers.0.self_attn.f_a_proj": {"bits": 8, "group_size": 64},
        },
    }
    ModelArgs.from_dict(config)
    q = config["quantization"]
    assert "lm_head" in q and "model.layers.3.self_attn.embed_q" in q and "model.model.embed_tokens" in q
    assert "model.layers.0.self_attn.forget_gate.f_a_proj" in q, "o portão cru do Vontra vira o módulo aninhado"
    assert not any(k.startswith("language_model.") for k in q)
    assert q["bits"] == 2 and q["group_size"] == 64, "os campos globais ficam"


def test_nomes_crus_ficam_como_estao():
    assert _strip_language_prefix("model.layers.3.self_attn.embed_q") == "model.layers.3.self_attn.embed_q"
    assert _strip_language_prefix("language_model.lm_head") == "lm_head"
    assert _strip_language_prefix("model.language_model.model.embed_tokens") == "model.model.embed_tokens"


def test_portao_de_esquecimento_cru_vira_o_modulo_aninhado():
    assert _module_key("language_model.model.layers.0.self_attn.f_a_proj") == "model.layers.0.self_attn.forget_gate.f_a_proj"
    assert _module_key("model.layers.0.self_attn.f_b_proj") == "model.layers.0.self_attn.forget_gate.f_b_proj"
    assert _module_key("model.layers.0.self_attn.forget_gate.f_a_proj") == "model.layers.0.self_attn.forget_gate.f_a_proj"
