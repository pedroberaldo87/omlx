# SPDX-License-Identifier: Apache-2.0
"""Tests for the spec-ab bench (plan v5, F0.1): unknown flip is an error,
flip=none yields a gain_pct (the noise floor), arms run interleaved."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.admin import routes as admin_routes
from omlx.admin import spec_ab


@pytest.fixture(autouse=True)
def _clean_runs():
    spec_ab._RUNS.clear()
    yield
    spec_ab._RUNS.clear()


@pytest.fixture
def client(monkeypatch):
    async def _fake_require_admin():
        return True

    monkeypatch.setattr(admin_routes, "_get_server_state", MagicMock())
    app = FastAPI()
    app.include_router(admin_routes.router)
    app.dependency_overrides[admin_routes.require_admin] = _fake_require_admin
    return TestClient(app)


def _run_dict(flip: str, repeats: int = 3, workload: str = "rewrite") -> dict:
    return {
        "id": "test-run",
        "model_id": "m",
        "repeats": repeats,
        "max_tokens": 64,
        "flip": flip,
        "workload": workload,
        "status": "running",
        "progress": "warmup",
        "sequence": [],
        "results": {},
        "started_ts": 0.0,
    }


def _stub_requests(monkeypatch):
    rates = iter([50.0, 100.0, 90.0, 102.0, 88.0, 101.0, 91.0])

    def _fake(port, api_key, model_id, max_tokens, prompt=spec_ab._PROMPT):
        r = next(rates)
        return {"tokens": 100, "seconds": 1.0, "tok_s": r}

    monkeypatch.setattr(spec_ab, "_one_request", _fake)


def test_unknown_flip_raises_value_error():
    with pytest.raises(ValueError, match="unknown flip"):
        spec_ab.start("m", 8000, "k", flip="bogus")
    assert not spec_ab._RUNS  # nothing was registered


def test_unknown_flip_returns_400(client):
    resp = client.post(
        "/admin/api/bench/spec-ab/start",
        json={"model_id": "m", "flip": "bogus"},
    )
    assert resp.status_code == 400
    assert "unknown flip" in resp.json()["detail"]


def test_flip_none_yields_gain_pct_and_pairs(monkeypatch):
    _stub_requests(monkeypatch)
    run = _run_dict("none")
    spec_ab._worker(run, 8000, "k")
    assert run["status"] == "done"
    # two identical arms still produce a gain_pct — the noise floor
    assert isinstance(run["results"]["gain_pct"], float)
    assert run["results"]["null_a"]["mean_tok_s"] is not None
    assert run["results"]["null_b"]["mean_tok_s"] is not None
    pairs = run["results"]["pairs"]
    assert len(pairs) == 3
    assert all("pair_gain_pct" in p for p in pairs)


def test_arms_run_interleaved(monkeypatch):
    _stub_requests(monkeypatch)
    run = _run_dict("enabled")
    spec_ab._worker(run, 8000, "k")
    assert run["status"] == "done"
    # A,B,A,B,A,B — never A,A,A then B,B,B
    assert run["sequence"] == ["ngram_on", "ngram_off"] * 3
    assert run["progress"] == "pair 3/3 · ngram_off"


# --- plan v5, F0.2: every knob the preset touches is saved and restored ---


@pytest.mark.parametrize("flip", spec_ab._FLIPS)
def test_params_restored_after_any_flip(monkeypatch, flip):
    from omlx.patches import mlx_lm_mtp as mtp

    before_params = mtp.get_ngram_spec_params()
    before_enabled = mtp.is_ngram_spec_enabled()
    before_hyst = mtp.is_mtp_hysteresis()
    _stub_requests(monkeypatch)
    run = _run_dict(flip)
    spec_ab._worker(run, 8000, "k")
    assert run["status"] == "done"
    assert mtp.get_ngram_spec_params() == before_params
    assert mtp.is_ngram_spec_enabled() == before_enabled
    assert mtp.is_mtp_hysteresis() == before_hyst


def test_worker_zeroes_pool_between_arms(monkeypatch):
    # plan v5, F0.5: every arm switch resets the shared cross-request pool
    from omlx.patches import mlx_lm_mtp as mtp

    calls: list[int] = []
    old = mtp._NGRAM_POOL_RESET
    mtp.set_ngram_pool_reset(lambda: calls.append(1))
    try:
        _stub_requests(monkeypatch)
        run = _run_dict("none")
        spec_ab._worker(run, 8000, "k")
        assert run["status"] == "done"
        assert len(calls) == 6  # one reset per arm-request, A,B interleaved
    finally:
        mtp.set_ngram_pool_reset(old)


def test_params_restored_after_mid_run_error(monkeypatch):
    from omlx.patches import mlx_lm_mtp as mtp

    before_params = mtp.get_ngram_spec_params()
    before_enabled = mtp.is_ngram_spec_enabled()
    calls = {"n": 0}

    def _dies_on_third(port, api_key, model_id, max_tokens,
                      prompt=spec_ab._PROMPT):
        calls["n"] += 1
        if calls["n"] >= 3:
            raise RuntimeError("boom")
        return {"tokens": 100, "seconds": 1.0, "tok_s": 50.0}

    monkeypatch.setattr(spec_ab, "_one_request", _dies_on_third)
    run = _run_dict("match_len")
    spec_ab._worker(run, 8000, "k")
    assert run["status"] == "error"
    assert mtp.get_ngram_spec_params() == before_params
    assert mtp.is_ngram_spec_enabled() == before_enabled
