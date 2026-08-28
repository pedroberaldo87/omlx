# SPDX-License-Identifier: Apache-2.0
"""The PLE probe must read the counter of the module the model actually uses.

The compat patch exposes the vendored tree as ``mlx_vlm.models.qwen4_exp``.
Importing the same file through ``omlx.patches...vendor...`` produces a second
module object with its own accumulator, so the probe reported "chamadas=0" for a
lookup that runs on every cycle.
"""

import importlib
import sys

from omlx.patches.mlx_vlm_qwen4_exp_compat import (
    apply_mlx_vlm_qwen4_exp_compat_patch,
)


def test_probe_reads_the_same_module_the_model_loads():
    apply_mlx_vlm_qwen4_exp_compat_patch()

    # The module the model loads is the mlx_vlm.* one, and it is the only one
    # importable at all: the omlx.patches...vendor... path cannot even resolve
    # its own relative imports.
    assert "mlx_vlm.models.qwen4_exp.language" in sys.modules

    fonte = (
        importlib.import_module("omlx.patches.mlx_lm_mtp.batch_generator")
        .__file__
    )
    with open(fonte, encoding="utf-8") as fh:
        texto = fh.read()
    assert "from mlx_vlm.models.qwen4_exp.language import ple_clock_read" in texto
    assert (
        "vendor.mlx_vlm.models.qwen4_exp.language import (  # noqa: E501"
        not in texto
    )


def test_counter_survives_a_round_trip_through_the_loaded_module():
    apply_mlx_vlm_qwen4_exp_compat_patch()
    lang = sys.modules["mlx_vlm.models.qwen4_exp.language"]

    lang.ple_clock_read(zerar=True)
    lang._ple_clock_add(0.005)
    lido = lang.ple_clock_read(zerar=True)
    assert lido["chamadas"] == 1
    assert lido["segundos"] > 0
    assert lang.ple_clock_read()["chamadas"] == 0


if __name__ == "__main__":
    test_probe_reads_the_same_module_the_model_loads()
    test_counter_survives_a_round_trip_through_the_loaded_module()
    print("OK")
