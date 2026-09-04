# SPDX-License-Identifier: Apache-2.0
"""Tests for the OQManager admin component."""

import asyncio
import json
from pathlib import Path

import pytest

from omlx.admin.oq_manager import OQManager, QuantStatus, QuantTask


@pytest.fixture
def fp_model_dir(tmp_path):
    """One directory with a full-precision (quantizable) source model."""
    d = tmp_path / "models1"
    d.mkdir()
    model = d / "Llama-3B"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "num_hidden_layers": 32,
            }
        )
    )
    (model / "model.safetensors").write_bytes(b"\x00" * 4096)
    return d


@pytest.fixture
def second_fp_model_dir(tmp_path):
    """A second directory holding a different full-precision model."""
    d = tmp_path / "models2"
    d.mkdir()
    model = d / "Qwen-7B"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "num_hidden_layers": 28,
            }
        )
    )
    (model / "model.safetensors").write_bytes(b"\x00" * 4096)
    return d


class TestOQManagerUpdateModelDirs:
    @pytest.mark.asyncio
    async def test_picks_up_added_dir(self, fp_model_dir, second_fp_model_dir):
        # Mirrors the real Settings UI flow: server starts with one model
        # directory, the user adds a second one at runtime via Settings, and
        # _apply_model_dirs_runtime calls update_model_dirs(). Without that
        # call, models in the newly added directory never show up in the oQ
        # Quantization "Source Model" dropdown.
        manager = OQManager(model_dirs=[str(fp_model_dir)])
        source_before, _ = await manager.list_quantizable_models()
        names_before = {m["name"] for m in source_before}
        assert "Llama-3B" in names_before
        assert "Qwen-7B" not in names_before

        manager.update_model_dirs([str(fp_model_dir), str(second_fp_model_dir)])

        source_after, _ = await manager.list_quantizable_models()
        names_after = {m["name"] for m in source_after}
        assert "Llama-3B" in names_after
        assert "Qwen-7B" in names_after

    def test_output_dir_tracks_primary_dir(self, fp_model_dir, second_fp_model_dir):
        # Output is always written to the primary (first) directory.
        manager = OQManager(model_dirs=[str(fp_model_dir)])
        assert manager._output_dir == fp_model_dir

        manager.update_model_dirs([str(second_fp_model_dir), str(fp_model_dir)])
        assert manager._output_dir == second_fp_model_dir


class TestOQManagerMxfp8Discovery:
    @pytest.mark.asyncio
    async def test_mxfp8_source_is_available_for_quantization(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        model = root / "MiniMax-M3-MXFP8"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "minimax_m3_vl",
                    "text_config": {
                        "num_hidden_layers": 60,
                        "num_local_experts": 128,
                        "num_mtp_modules": 1,
                    },
                    "vision_config": {"num_hidden_layers": 32},
                    "quantization_config": {
                        "quant_method": "mxfp8",
                        "activation_scheme": "dynamic",
                        "weight_block_size": [1, 32],
                    },
                }
            ),
            encoding="utf-8",
        )
        (model / "model.safetensors").write_bytes(b"\x00" * 4096)

        manager = OQManager(model_dirs=[str(root)])
        source_models, all_models = await manager.list_quantizable_models()

        assert [entry["name"] for entry in source_models] == ["MiniMax-M3-MXFP8"]
        assert source_models[0]["is_quantized"] is False
        assert source_models[0]["is_vlm"] is True
        # The published checkpoint advertises this training metadata but has
        # no MTP/nextn tensors, so it must not offer fake MTP preservation.
        assert source_models[0]["has_mtp_heads"] is False
        assert source_models[0]["num_layers"] == 60
        assert [entry["name"] for entry in all_models] == ["MiniMax-M3-MXFP8"]


class TestOQManagerMtpDetection:
    def _write_model(self, root, name, *, index_weight_map=None):
        model = root / name
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5",
                    "text_config": {
                        "model_type": "qwen3_5_text",
                        "num_hidden_layers": 32,
                        "mtp_num_hidden_layers": 1,
                    },
                }
            )
        )
        (model / "model.safetensors").write_bytes(b"\x00" * 4096)
        if index_weight_map is not None:
            (model / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "weight_map": index_weight_map,
                    }
                )
            )
        return model

    @pytest.mark.asyncio
    async def test_config_only_mtp_is_not_reported_as_preservable(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        self._write_model(root, "QwenPawLike")

        manager = OQManager(model_dirs=[str(root)])
        source_models, _ = await manager.list_quantizable_models()

        [model] = source_models
        assert model["has_mtp_heads"] is False

    @pytest.mark.asyncio
    async def test_mtp_weight_index_is_reported_as_preservable(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        self._write_model(
            root,
            "QwenMtp",
            index_weight_map={
                "language_model.mtp.fc.weight": "model.safetensors",
            },
        )

        manager = OQManager(model_dirs=[str(root)])
        source_models, _ = await manager.list_quantizable_models()

        [model] = source_models
        assert model["has_mtp_heads"] is True

    @pytest.mark.asyncio
    async def test_inkling_mtp_config_is_reported_as_preservable(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        model = root / "Inkling-Small"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "inkling_mm_model",
                    "text_config": {
                        "hidden_size": 4096,
                        "num_hidden_layers": 42,
                    },
                    "vision_config": {},
                    "mtp_config": {"num_nextn_predict_layers": 8},
                }
            )
        )
        (model / "model.safetensors").write_bytes(b"\x00" * 4096)
        (model / "mtp.safetensors").write_bytes(b"\x00" * 4096)
        (model / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {},
                    "weight_map": {
                        "model.mtp.layers.0.input_proj.weight": "mtp.safetensors",
                    },
                }
            )
        )

        manager = OQManager(model_dirs=[str(root)])
        source_models, _ = await manager.list_quantizable_models()

        [model_info] = source_models
        assert model_info["has_mtp_heads"] is True

    @pytest.mark.asyncio
    async def test_start_quantization_disables_preserve_mtp_without_weights(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "models"
        root.mkdir()
        self._write_model(root, "QwenPawLike")

        manager = OQManager(model_dirs=[str(root)])

        async def _noop_run(task_id):
            return None

        monkeypatch.setattr(manager, "_run_quantization", _noop_run)

        task = await manager.start_quantization(
            str(root / "QwenPawLike"),
            4,
            preserve_mtp=True,
        )
        await manager._active_tasks[task.task_id]

        assert task.preserve_mtp is False
        assert task.output_name == "QwenPawLike-oQ4"


class TestOQManagerAssistantCombine:
    """Gemma 4 assistant MTP combine wiring through start/run."""

    def _write_gemma4_base(self, root):
        model = root / "gemma-4-test"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "gemma4",
                    "vision_config": {},
                    "text_config": {"model_type": "gemma4_text", "hidden_size": 24},
                }
            )
        )
        (model / "model.safetensors").write_bytes(b"\x00" * 4096)
        return model

    def _write_assistant(self, root, backbone_hidden=24):
        model = root / "gemma-4-test-assistant"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "gemma4_assistant",
                    "backbone_hidden_size": backbone_hidden,
                    "text_config": {
                        "model_type": "gemma4_text",
                        "num_hidden_layers": 4,
                    },
                }
            )
        )
        (model / "model.safetensors").write_bytes(b"\x00" * 512)
        return model

    @pytest.mark.asyncio
    async def test_start_names_output_with_mtp_suffix(self, tmp_path, monkeypatch):
        root = tmp_path / "models"
        root.mkdir()
        base = self._write_gemma4_base(root)
        assistant = self._write_assistant(root)

        manager = OQManager(model_dirs=[str(root)])

        async def _noop_run(task_id):
            return None

        monkeypatch.setattr(manager, "_run_quantization", _noop_run)

        task = await manager.start_quantization(
            str(base),
            4,
            mtp_assistant_model_path=str(assistant),
        )
        await manager._active_tasks[task.task_id]

        assert task.output_name == "gemma-4-test-oQ4-mtp"
        assert task.mtp_assistant_model_path == str(assistant)

    @pytest.mark.asyncio
    async def test_start_rejects_mismatched_assistant(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        base = self._write_gemma4_base(root)
        assistant = self._write_assistant(root, backbone_hidden=32)

        manager = OQManager(model_dirs=[str(root)])

        with pytest.raises(ValueError, match="backbone_hidden_size"):
            await manager.start_quantization(
                str(base),
                4,
                mtp_assistant_model_path=str(assistant),
            )
        assert manager._tasks == {}

    @pytest.mark.asyncio
    async def test_run_invokes_combine_after_quantization(self, tmp_path, monkeypatch):
        root = tmp_path / "models"
        root.mkdir()
        base = self._write_gemma4_base(root)
        assistant = self._write_assistant(root)

        manager = OQManager(model_dirs=[str(root)])

        def _fake_quantize(model_path, output_path, *args, **kwargs):
            from pathlib import Path

            Path(output_path).mkdir(parents=True)

        combine_calls = []
        monkeypatch.setattr("omlx.oq.quantize_oq_streaming", _fake_quantize)
        monkeypatch.setattr(
            "omlx.oq.combine_mtp_into_output",
            lambda out, asst: combine_calls.append((out, asst)),
        )

        task = await manager.start_quantization(
            str(base),
            4,
            mtp_assistant_model_path=str(assistant),
        )
        await manager._active_tasks[task.task_id]

        assert task.status is QuantStatus.COMPLETED
        assert combine_calls == [(task.output_path, str(assistant))]

    @pytest.mark.asyncio
    async def test_run_dispatches_gemma4_assistant_to_legacy_combine(
        self, tmp_path, monkeypatch
    ):
        # The real dispatcher must route a gemma4_assistant donor to the
        # legacy assistant merge.
        root = tmp_path / "models"
        root.mkdir()
        base = self._write_gemma4_base(root)
        assistant = self._write_assistant(root)

        manager = OQManager(model_dirs=[str(root)])

        def _fake_quantize(model_path, output_path, *args, **kwargs):
            from pathlib import Path

            Path(output_path).mkdir(parents=True)

        legacy_calls = []
        monkeypatch.setattr("omlx.oq.quantize_oq_streaming", _fake_quantize)
        monkeypatch.setattr(
            "omlx.oq.combine_gemma4_assistant_mtp",
            lambda out, asst: legacy_calls.append((out, asst)),
        )

        task = await manager.start_quantization(
            str(base),
            4,
            mtp_assistant_model_path=str(assistant),
        )
        await manager._active_tasks[task.task_id]

        assert task.status is QuantStatus.COMPLETED
        assert legacy_calls == [(task.output_path, str(assistant))]


class TestOQManagerMtpDonorCombine:
    """Native Qwen3.5/3.6 donor head graft wiring through start/run."""

    _GEOMETRY = {
        "vocab_size": 16,
        "hidden_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 16,
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000,
    }

    def _write_source(self, root, *, with_mtp=False):
        model = root / "Qwen-Test"
        model.mkdir()
        config = {"model_type": "qwen3_5", "num_hidden_layers": 2, **self._GEOMETRY}
        if with_mtp:
            config["mtp_num_hidden_layers"] = 1
        (model / "config.json").write_text(json.dumps(config))
        (model / "model.safetensors").write_bytes(b"\x00" * 4096)
        if with_mtp:
            (model / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "weight_map": {"mtp.fc.weight": "model.safetensors"},
                    }
                )
            )
        (model / "tokenizer.json").write_bytes(b'{"v": "tok"}')
        return model

    def _write_donor(self, root, *, model_type="qwen3_5"):
        model = root / "Qwen-Test-Donor"
        model.mkdir()
        config = {
            "model_type": model_type,
            "num_hidden_layers": 2,
            "mtp_num_hidden_layers": 1,
            **self._GEOMETRY,
        }
        (model / "config.json").write_text(json.dumps(config))
        (model / "model.safetensors").write_bytes(b"\x00" * 512)
        (model / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {},
                    "weight_map": {
                        "mtp.fc.weight": "model.safetensors",
                        "mtp.norm.weight": "model.safetensors",
                    },
                }
            )
        )
        (model / "tokenizer.json").write_bytes(b'{"v": "tok"}')
        return model

    @pytest.mark.asyncio
    async def test_start_names_output_with_mtp_suffix(self, tmp_path, monkeypatch):
        root = tmp_path / "models"
        root.mkdir()
        source = self._write_source(root)
        donor = self._write_donor(root)

        manager = OQManager(model_dirs=[str(root)])

        async def _noop_run(task_id):
            return None

        monkeypatch.setattr(manager, "_run_quantization", _noop_run)

        task = await manager.start_quantization(
            str(source),
            4,
            mtp_assistant_model_path=str(donor),
        )
        await manager._active_tasks[task.task_id]

        assert task.output_name == "Qwen-Test-oQ4-mtp"
        assert task.mtp_assistant_model_path == str(donor)

    @pytest.mark.asyncio
    async def test_start_rejects_preserve_mtp_with_donor(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        # Source ships its own mtp weights so the preserve flag survives the
        # auto-disable and hits the mutual-exclusion check.
        source = self._write_source(root, with_mtp=True)
        donor = self._write_donor(root)

        manager = OQManager(model_dirs=[str(root)])

        with pytest.raises(ValueError, match="not both"):
            await manager.start_quantization(
                str(source),
                4,
                preserve_mtp=True,
                mtp_assistant_model_path=str(donor),
            )
        assert manager._tasks == {}

    @pytest.mark.asyncio
    async def test_start_rejects_family_mismatch_donor(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        source = self._write_source(root)
        donor = self._write_donor(root, model_type="qwen3_6")

        manager = OQManager(model_dirs=[str(root)])

        with pytest.raises(ValueError, match="does not match the recipient"):
            await manager.start_quantization(
                str(source),
                4,
                mtp_assistant_model_path=str(donor),
            )
        assert manager._tasks == {}

    @pytest.mark.asyncio
    async def test_run_invokes_donor_combine_after_quantization(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "models"
        root.mkdir()
        source = self._write_source(root)
        donor = self._write_donor(root)

        manager = OQManager(model_dirs=[str(root)])

        def _fake_quantize(model_path, output_path, *args, **kwargs):
            from pathlib import Path

            Path(output_path).mkdir(parents=True)

        combine_calls = []
        monkeypatch.setattr("omlx.oq.quantize_oq_streaming", _fake_quantize)
        monkeypatch.setattr(
            "omlx.oq.combine_mtp_into_output",
            lambda out, donor_path: combine_calls.append((out, donor_path)),
        )

        task = await manager.start_quantization(
            str(source),
            4,
            mtp_assistant_model_path=str(donor),
        )
        await manager._active_tasks[task.task_id]

        assert task.status is QuantStatus.COMPLETED
        assert combine_calls == [(task.output_path, str(donor))]

    @pytest.mark.asyncio
    async def test_list_models_includes_hidden_size(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        self._write_source(root)

        manager = OQManager(model_dirs=[str(root)])
        source_models, all_models = await manager.list_quantizable_models()

        [model] = source_models
        assert model["hidden_size"] == 8
        assert all_models[0]["hidden_size"] == 8


class TestOQManagerDtypeSupport:
    @pytest.mark.asyncio
    async def test_start_quantization_rejects_deepseek_v4_float16(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        model = root / "DeepSeek-V4-Flash"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps({"model_type": "deepseek_v4"}),
            encoding="utf-8",
        )

        manager = OQManager(model_dirs=[str(root)])

        with pytest.raises(ValueError, match="dtype=float16.*deepseek_v4"):
            await manager.start_quantization(str(model), 4, dtype="float16")

        assert manager._tasks == {}
        assert not (root / "DeepSeek-V4-Flash-oQ4-fp16").exists()


class TestOQManagerProgress:
    def test_byte_level_quant_progress_disables_time_estimator(self):
        task = QuantTask(
            task_id="task",
            model_name="Model",
            model_path="/tmp/Model",
            oq_level=2.5,
            output_name="Model-oQ2.5e",
            output_path="/tmp/Model-oQ2.5e",
            status=QuantStatus.QUANTIZING,
            progress=39.0,
            progress_meta={"processed_bytes": 31, "total_bytes": 100},
        )

        assert OQManager._has_explicit_quant_progress(task) is True

    def test_non_byte_quant_progress_can_use_time_estimator(self):
        task = QuantTask(
            task_id="task",
            model_name="Model",
            model_path="/tmp/Model",
            oq_level=2.5,
            output_name="Model-oQ2.5e",
            output_path="/tmp/Model-oQ2.5e",
            status=QuantStatus.QUANTIZING,
            progress=30.0,
            progress_meta={},
        )

        assert OQManager._has_explicit_quant_progress(task) is False


class TestOQManagerEnhanced:
    @pytest.mark.asyncio
    async def test_start_quantization_uses_enhanced_name_and_cache_path(
        self, fp_model_dir, monkeypatch
    ):
        manager = OQManager(model_dirs=[str(fp_model_dir)])

        async def _noop_run(task_id):
            return None

        monkeypatch.setattr(manager, "_run_quantization", _noop_run)

        task = await manager.start_quantization(
            str(fp_model_dir / "Llama-3B"),
            4,
            enhanced=True,
            imatrix_num_samples=8,
            imatrix_seq_length=128,
        )
        await manager._active_tasks[task.task_id]

        assert task.enhanced is True
        assert task.output_name == "Llama-3B-oQ4e"
        assert ".oqe_imatrix" in task.imatrix_cache_path
        assert task.imatrix_cache_path.endswith("-s8-l128.npz")


class TestOQManagerOpcoesDeMemoria:
    """As tres opcoes de memoria chegam ao pedido e ao nome do checkpoint.

    Sao opcoes, nao mudanca fixa: com os valores de fabrica o pedido tem que
    sair identico ao de antes delas existirem.
    """

    @pytest.mark.asyncio
    async def test_o_padrao_nao_muda_o_pedido_nem_o_nome(
        self, fp_model_dir, monkeypatch
    ):
        manager = OQManager(model_dirs=[str(fp_model_dir)])

        async def _noop_run(task_id):
            return None

        monkeypatch.setattr(manager, "_run_quantization", _noop_run)
        task = await manager.start_quantization(str(fp_model_dir / "Llama-3B"), 4)
        await manager._active_tasks[task.task_id]

        assert task.attention_bits_cap == 0
        assert task.vision_dtype == "auto"
        assert task.mtp_bits_floor == 4
        assert task.group_size == 64
        assert task.output_name == "Llama-3B-oQ4"

    @pytest.mark.asyncio
    async def test_as_opcoes_chegam_ao_pedido_e_ao_nome(
        self, fp_model_dir, monkeypatch
    ):
        manager = OQManager(model_dirs=[str(fp_model_dir)])

        async def _noop_run(task_id):
            return None

        monkeypatch.setattr(manager, "_run_quantization", _noop_run)
        task = await manager.start_quantization(
            str(fp_model_dir / "Llama-3B"),
            2,
            group_size=128,
            attention_bits_cap=4,
            vision_dtype="float16",
            mtp_bits_floor=0,
        )
        await manager._active_tasks[task.task_id]

        assert task.attention_bits_cap == 4
        assert task.vision_dtype == "float16"
        assert task.mtp_bits_floor == 0
        assert task.group_size == 128
        assert task.output_name == "Llama-3B-oQ2-a4-g128"
        # e o painel enxerga as quatro pelo to_dict
        d = task.to_dict()
        assert d["attention_bits_cap"] == 4
        assert d["vision_dtype"] == "float16"
        assert d["mtp_bits_floor"] == 0


class TestOQManagerHfCacheDiscovery:
    """HF cache models (non-MLX) should appear as quantization sources."""

    @pytest.mark.asyncio
    async def test_hf_cache_model_is_available_for_quantization(self, tmp_path):
        """Models stored in HF Hub cache layout should be discoverable
        for oQ quantization, even when they are non-MLX PyTorch checkpoints."""
        hf_cache = tmp_path / "hf_cache"
        # Create HF cache layout: models--Org--Repo/snapshots/<hash>/
        hf_entry = hf_cache / "models--Org--MyModel"
        snapshots = hf_entry / "snapshots"
        commit_hash = "abc123def456"
        model_dir = snapshots / commit_hash
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "llama",
                    "num_hidden_layers": 32,
                    "hidden_size": 4096,
                }
            )
        )
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 4096)
        # Create refs/main to point to the commit hash
        refs = hf_entry / "refs"
        refs.mkdir()
        (refs / "main").write_text(commit_hash)

        manager = OQManager(model_dirs=[str(hf_cache)])
        source_models, all_models = await manager.list_quantizable_models()

        assert len(source_models) == 1
        assert source_models[0]["name"] == "Org--MyModel"
        assert source_models[0]["source_repo_id"] == "Org/MyModel"
        assert commit_hash in source_models[0]["path"]
        assert source_models[0]["num_layers"] == 32

    @pytest.mark.asyncio
    async def test_hf_cache_quantization_uses_repo_identity(
        self, tmp_path, monkeypatch
    ):
        output_dir = tmp_path / "models"
        output_dir.mkdir()
        hf_cache = tmp_path / "hf_cache"
        hf_entry = hf_cache / "models--Org--MyModel"
        commit_hash = "abc123def456"
        model_dir = hf_entry / "snapshots" / commit_hash
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(
            json.dumps({"model_type": "llama", "num_hidden_layers": 32})
        )
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 4096)

        manager = OQManager(model_dirs=[str(output_dir), str(hf_cache)])

        async def _noop_run(task_id):
            return None

        monkeypatch.setattr(manager, "_run_quantization", _noop_run)
        source_models, _ = await manager.list_quantizable_models()
        assert [model["name"] for model in source_models] == ["Org--MyModel"]

        task = await manager.start_quantization(
            source_models[0]["path"],
            4,
            enhanced=True,
            imatrix_num_samples=8,
            imatrix_seq_length=128,
        )
        await manager._active_tasks[task.task_id]

        assert task.model_name == "Org/MyModel"
        # Output name must use the bare repo name (no double-dash org prefix):
        # huggingface_hub rejects repo_ids containing "--" on upload.
        assert task.output_name == "MyModel-oQ4e"
        assert Path(task.output_path) == output_dir / "MyModel-oQ4e"
        imatrix_path = Path(task.imatrix_cache_path)
        assert imatrix_path.parent == output_dir / ".oqe_imatrix"
        assert imatrix_path.name.startswith("MyModel-")

    @pytest.mark.asyncio
    async def test_hf_cache_excludes_bin_only_checkpoint(self, tmp_path):
        hf_cache = tmp_path / "hf_cache"
        hf_entry = hf_cache / "models--Org--BinOnly"
        commit_hash = "abc123"
        model_dir = hf_entry / "snapshots" / commit_hash
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(
            json.dumps({"model_type": "llama", "num_hidden_layers": 32})
        )
        (model_dir / "pytorch_model.bin").write_bytes(b"\x00" * 4096)
        refs = hf_entry / "refs"
        refs.mkdir()
        (refs / "main").write_text(commit_hash)

        manager = OQManager(model_dirs=[str(hf_cache)])
        source_models, all_models = await manager.list_quantizable_models()

        assert source_models == []
        assert all_models == []
        with pytest.raises(ValueError, match=r"No \.safetensors files found"):
            await manager.start_quantization(str(model_dir), 4)
        assert manager._tasks == {}

    @pytest.mark.asyncio
    async def test_hf_cache_fallback_without_refs(self, tmp_path):
        """HF cache entry without refs/main should still be discovered
        via the latest-by-mtime fallback."""
        hf_cache = tmp_path / "hf_cache"
        hf_entry = hf_cache / "models--Meta--Llama"
        snapshots = hf_entry / "snapshots"
        commit_hash = "deadbeef"
        model_dir = snapshots / commit_hash
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "llama",
                    "num_hidden_layers": 16,
                }
            )
        )
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 4096)
        # No refs/main file — fallback to latest snapshot by mtime

        manager = OQManager(model_dirs=[str(hf_cache)])
        source_models, _ = await manager.list_quantizable_models()

        assert len(source_models) == 1
        assert source_models[0]["name"] == "Meta--Llama"
        assert source_models[0]["source_repo_id"] == "Meta/Llama"

    @pytest.mark.asyncio
    async def test_hf_cache_excludes_models_without_model_type(self, tmp_path):
        """HF cache models without model_type in config should be excluded
        because MLX cannot resolve the model class for quantization."""
        hf_cache = tmp_path / "hf_cache"
        hf_entry = hf_cache / "models--Org--NoType"
        snapshots = hf_entry / "snapshots"
        commit_hash = "abc123"
        model_dir = snapshots / commit_hash
        model_dir.mkdir(parents=True)
        # Config with no model_type
        (model_dir / "config.json").write_text(
            json.dumps({"architectures": ["LlamaForCausalLM"]})
        )
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 4096)
        refs = hf_entry / "refs"
        refs.mkdir()
        (refs / "main").write_text(commit_hash)

        manager = OQManager(model_dirs=[str(hf_cache)])
        source_models, all_models = await manager.list_quantizable_models()

        assert len(source_models) == 0
        assert len(all_models) == 0


class TestRelatoriosSobrevivemAoRmtree:
    """02/09: toda excecao apagava a pasta inteira, inclusive os oq_*_report.json
    que a propria mensagem de erro mandava o dono ler."""

    def test_move_os_relatorios_para_a_pasta_rejeitado(self, tmp_path):
        from omlx.admin.oq_manager import _preserve_oq_reports

        out = tmp_path / "GLM-oQ2e"
        out.mkdir()
        (out / "oq_imatrix_report.json").write_text("{}")
        (out / "oq_verify_report.json").write_text("{}")
        (out / "model-00001-of-00002.safetensors").write_bytes(b"x")
        destino = _preserve_oq_reports(out)
        assert destino == tmp_path / "GLM-oQ2e.rejeitado"
        assert sorted(p.name for p in destino.iterdir()) == [
            "oq_imatrix_report.json", "oq_verify_report.json",
        ]
        assert not (out / "oq_imatrix_report.json").exists()
        assert (out / "model-00001-of-00002.safetensors").exists()

    def test_sem_relatorio_nao_cria_pasta(self, tmp_path):
        from omlx.admin.oq_manager import _preserve_oq_reports

        out = tmp_path / "vazio"
        out.mkdir()
        assert _preserve_oq_reports(out) is None
        assert not (tmp_path / "vazio.rejeitado").exists()

    @pytest.mark.asyncio
    async def test_falha_na_quantizacao_preserva_o_diagnostico_e_aponta_no_erro(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "models"
        root.mkdir()
        base = TestOQManagerAssistantCombine._write_gemma4_base(TestOQManagerAssistantCombine(), root)
        manager = OQManager(model_dirs=[str(root)])

        def _quant_que_recusa_no_fim(model_path, output_path, *args, **kwargs):
            out = Path(output_path)
            out.mkdir(parents=True)
            (out / "oq_imatrix_report.json").write_text('{"missing": ["x"]}')
            (out / "model-00001-of-00001.safetensors").write_bytes(b"x")
            raise RuntimeError("oQe: 1 peso sem entrada. Nao gravo: " + output_path)

        monkeypatch.setattr("omlx.oq.quantize_oq_streaming", _quant_que_recusa_no_fim)
        task = await manager.start_quantization(str(base), 4)
        await manager._active_tasks[task.task_id]

        assert task.status is QuantStatus.FAILED
        rejeitado = Path(task.output_path).parent / (Path(task.output_path).name + ".rejeitado")
        assert (rejeitado / "oq_imatrix_report.json").exists()
        assert not Path(task.output_path).exists()
        assert str(rejeitado) in task.error


class TestCancelarParaDeVerdade:
    """03/09: o dono disparou uma oQ4e por engano, cancelou e tirou da lista.
    Ela seguiu rodando por 12 minutos, e ao 'cancelar' liberou a vaga para a
    oQ2e da fila: duas coletas de 35 GB juntas, memoria estourada, e a guarda
    derrubou justamente a que ele queria."""

    def _gerente_com_trabalho_vivo(self, tmp_path):
        """Um gerente com trabalho num thread real, com portao controlado pelo
        teste: o thread so consulta a bandeira quando o teste deixa."""
        import threading

        from omlx.admin.oq_manager import QuantTask, _QuantCancelled

        manager = OQManager(model_dirs=[str(tmp_path)])
        task_id = "tarefa-viva"
        task = QuantTask(
            task_id=task_id, model_name="fake", model_path=str(tmp_path / "src"),
            output_path=str(tmp_path / "out"), output_name="out",
            oq_level=4, group_size=64,
        )
        manager._tasks[task_id] = task
        e = {
            "voltas": 0,
            "parou": False,
            "pode_seguir": threading.Event(),
            "fim": threading.Event(),
        }

        def trabalho(callback):
            for _ in range(40):
                # espera o teste liberar a proxima "camada"
                e["pode_seguir"].wait(timeout=5.0)
                e["pode_seguir"].clear()
                try:
                    callback("imatrix", 13.0, "layer", {})
                except _QuantCancelled:
                    e["parou"] = True
                    break
                e["voltas"] += 1
            e["fim"].set()

        async def roda():
            async with manager._quant_sem:
                def _progress_cb(phase, pct, detail="", meta=None):
                    if task_id in manager._cancelled:
                        raise _QuantCancelled("cancelled")

                task.status = QuantStatus.LOADING
                await asyncio.to_thread(trabalho, _progress_cb)

        return manager, task_id, task, e, roda

    @pytest.mark.asyncio
    async def test_remover_da_lista_nao_apaga_a_bandeira_do_cancelamento(self, tmp_path):
        manager, task_id, task, e, roda = self._gerente_com_trabalho_vivo(tmp_path)
        manager._active_tasks[task_id] = asyncio.create_task(roda())
        e["pode_seguir"].set()
        await asyncio.sleep(0.05)

        cancelar = asyncio.create_task(manager.cancel_quantization(task_id))
        await asyncio.sleep(0.05)
        # o clique seguinte, em "remover da lista", com o trabalho ainda vivo
        assert manager.remove_task(task_id) is False, (
            "remover foi aceito com o trabalho ainda vivo — foi assim que a "
            "bandeira sumiu e a quantizacao seguiu por 12 minutos"
        )
        assert task_id in manager._cancelled
        assert task.status is QuantStatus.CANCELLING

        e["pode_seguir"].set()               # a proxima camada: o thread ve a bandeira
        await asyncio.wait_for(cancelar, timeout=5.0)
        assert e["parou"] is True
        assert task.status is QuantStatus.CANCELLED
        # so agora sai da lista
        assert manager.remove_task(task_id) is True
        assert task_id in manager._cancelled, "a bandeira nao pode ser apagada"

    @pytest.mark.asyncio
    async def test_a_vaga_so_abre_quando_o_trabalho_sai(self, tmp_path):
        """Matar a casca assincrona liberava o semaforo com o trabalho ainda
        residente, e a proxima da fila entrava por cima — foi o estouro de
        memoria de 03/09."""
        manager, task_id, task, e, roda = self._gerente_com_trabalho_vivo(tmp_path)
        manager._active_tasks[task_id] = asyncio.create_task(roda())
        e["pode_seguir"].set()
        await asyncio.sleep(0.05)
        assert manager._quant_sem.locked()

        cancelar = asyncio.create_task(manager.cancel_quantization(task_id))
        await asyncio.sleep(0.05)
        assert manager._quant_sem.locked(), (
            "a vaga abriu com o trabalho ainda rodando: a proxima quantizacao "
            "entraria com esta na memoria"
        )
        e["pode_seguir"].set()
        await asyncio.wait_for(cancelar, timeout=5.0)
        assert e["parou"] is True
        assert not manager._quant_sem.locked()

    @pytest.mark.asyncio
    async def test_cancelando_conta_como_ativa(self, tmp_path):
        """CANCELLING e um estado ATIVO: e isso que faz remove_task recusar e
        o botao de remover sumir do painel."""
        from omlx.admin.oq_manager import _ACTIVE_STATUSES

        assert QuantStatus.CANCELLING in _ACTIVE_STATUSES
        assert QuantStatus.CANCELLED not in _ACTIVE_STATUSES
