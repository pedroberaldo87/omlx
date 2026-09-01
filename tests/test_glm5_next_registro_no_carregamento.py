"""O GLM-5.3 tem que existir no mlx-lm quando o carregamento comum roda.

O GLM-5.2 (`glm_moe_dsa`) vem de fábrica no mlx-lm, então o runtime de previsão
múltipla dele encontra o modelo já lá. O 5.3 (`glm5_next`) é inteiramente nosso:
só aparece no mlx-lm quando alguém chama o registro. Até 01/09 quem chamava era
só a quantização e o rascunhador — o carregamento comum não, e por isso ele
morria com "Model type glm5_next not supported" antes de ler um peso.

Os dois testes rodam em SUBPROCESSO: `maybe_apply_pre_load_patches` aplica
remendos globais e irreversíveis no processo (medido: rodá-la aqui derrubava 9
testes do inkling mais adiante na suíte). Isolar é o que permite exercitar a
função de verdade em vez de imitá-la.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

ORIGEM = os.path.expanduser("~/.omlx/models/zai-org/GLM-5.3-Flash")
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _modelo_minimo(tmp_path):
    """Um diretório de modelo só com o que o carregador lê antes dos pesos."""
    if os.path.isfile(os.path.join(ORIGEM, "config.json")):
        cfg = json.load(open(os.path.join(ORIGEM, "config.json"), encoding="utf-8"))
    else:
        cfg = {"model_type": "glm5_next", "num_hidden_layers": 45}
    destino = tmp_path / "modelo"
    destino.mkdir()
    (destino / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return str(destino)


def _registrou(caminho, for_vlm):
    """Roda a função de verdade num processo limpo e devolve se registrou."""
    programa = textwrap.dedent(
        f"""
        import importlib.util, sys
        from omlx.utils.model_loading import maybe_apply_pre_load_patches
        maybe_apply_pre_load_patches({caminho!r}, for_vlm={for_vlm!r})
        nome = "mlx_lm.models.glm5_next"
        achou = nome in sys.modules or importlib.util.find_spec(nome) is not None
        print("REGISTROU" if achou else "NAO")
        """
    )
    saida = subprocess.run(
        [sys.executable, "-c", programa],
        capture_output=True,
        text=True,
        cwd=RAIZ,
        timeout=300,
    )
    assert saida.returncode == 0, (
        f"o subprocesso falhou:\n{saida.stdout[-2000:]}\n{saida.stderr[-2000:]}"
    )
    return saida.stdout.strip().splitlines()[-1] == "REGISTROU"


def test_o_carregamento_de_texto_registra_o_glm5_next(tmp_path):
    """Sem isto o mlx-lm nem chega a olhar os pesos."""
    assert _registrou(_modelo_minimo(tmp_path), for_vlm=False), (
        "o carregamento de texto não registrou o glm5_next no mlx-lm; ele morre "
        'com "Model type glm5_next not supported" antes de ler um peso'
    )


def test_o_caminho_de_visao_nao_registra_a_familia_no_mlx_lm(tmp_path):
    """O ramo de visão tem o próprio remendo e não deve mexer no mlx-lm.

    Registrar dos dois lados faria o modelo de texto vencer o de visão em
    processos que carregam os dois, que é o oposto do que o ramo de visão quer.
    """
    assert not _registrou(_modelo_minimo(tmp_path), for_vlm=True), (
        "o caminho de visão registrou a família de texto no mlx-lm"
    )
