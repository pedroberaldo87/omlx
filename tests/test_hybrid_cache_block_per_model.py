# SPDX-License-Identifier: Apache-2.0
"""O bloco do cache paginado por modelo vence o do servidor.

Medido em 05/09 no M1 Ultra: o GLM-5.3-Flash quer 1024, e o Qwen3.8-Flash-Next fica 18% mais
rápido com os 4096 nativos dele (551 contra 466 tok/s a 30k) — que o 1024 do servidor, escolhido
para o GLM, limitava. O valor certo é do modelo, não do servidor.
"""
import copy
from types import SimpleNamespace

from omlx.model_settings import ModelSettings
from omlx.scheduler import SchedulerConfig


def _config_do_motor(config_do_servidor, ajustes_do_modelo):
    """Reproduz o trecho que os motores rodam ao montar o config do agendador."""
    scheduler_config = copy.copy(config_do_servidor)
    model_block = getattr(ajustes_do_modelo, "hybrid_cache_block", None)
    if model_block is not None:
        scheduler_config.hybrid_cache_block = int(model_block)
    return scheduler_config


def test_padrao_e_nulo_e_o_servidor_manda():
    assert ModelSettings().hybrid_cache_block is None
    servidor = SchedulerConfig(hybrid_cache_block=1024)
    cfg = _config_do_motor(servidor, ModelSettings())
    assert cfg.hybrid_cache_block == 1024


def test_o_modelo_vence_o_servidor():
    servidor = SchedulerConfig(hybrid_cache_block=1024)
    cfg = _config_do_motor(servidor, ModelSettings(hybrid_cache_block=4096))
    assert cfg.hybrid_cache_block == 4096
    # e não contamina o config compartilhado do servidor
    assert servidor.hybrid_cache_block == 1024


def test_sem_ajustes_de_modelo_nao_quebra():
    servidor = SchedulerConfig(hybrid_cache_block=512)
    assert _config_do_motor(servidor, None).hybrid_cache_block == 512


def test_ida_e_volta_pelo_disco():
    d = ModelSettings(hybrid_cache_block=4096).to_dict()
    assert d["hybrid_cache_block"] == 4096
    assert ModelSettings.from_dict(d).hybrid_cache_block == 4096
    assert ModelSettings.from_dict({}).hybrid_cache_block is None
