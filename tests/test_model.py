from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from app import model


def test_offline_flags_are_forced():
    import os
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


@pytest.mark.parametrize("value, expected", [(None, None), ("", None), ("1", True), ("true", True), ("0", False), ("False", False)])
def test_four_bit_switch(value, expected):
    assert model.optional_bool(value) is expected


def test_invalid_four_bit_switch():
    with pytest.raises(ValueError):
        model.optional_bool("perhaps")


def test_local_base_override_preserves_bundle(monkeypatch, tmp_path):
    monkeypatch.delenv("MOCK_MODEL")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    decision = bundle / "decision.json"
    decision.write_text('{"qkeys":[]}', encoding="utf-8")
    original = '{"cfg":{"model_id":"cached/model"},"tau":0.5}'
    config = bundle / "pipeline_config.json"
    config.write_text(original, encoding="utf-8")
    base = tmp_path / "local-model"
    base.mkdir()
    monkeypatch.setenv("FORTE_BUNDLE_DIR", str(bundle))
    monkeypatch.setenv("BASE_MODEL_PATH", str(base))
    monkeypatch.setenv("LOAD_4BIT", "0")
    calls = []

    def fake_load(path, **kwargs):
        path = Path(path)
        calls.append(path)
        assert path == bundle.resolve()
        assert json.loads((path / "pipeline_config.json").read_text())["cfg"]["model_id"] == "cached/model"
        assert (path / "decision.json").is_file()
        assert os.environ["BASE_MODEL_PATH"] == str(base)
        assert not (path / "adapter").exists()
        assert kwargs == {"load_4bit": False, "merge": True}
        return "loaded"

    monkeypatch.setattr(model, "load_bundle", fake_load)
    assert model.load_analyzer() == "loaded"
    assert config.read_text(encoding="utf-8") == original
    assert len(calls) == 1
    assert calls[0].is_dir()
    assert decision.read_text(encoding="utf-8") == '{"qkeys":[]}'


@pytest.mark.parametrize("setting", ["FORTE_BUNDLE_DIR", "BUNDLE_DIR", "default"])
def test_v3_bundle_setting_without_loading_model(monkeypatch, tmp_path, setting):
    monkeypatch.delenv("MOCK_MODEL")
    for name in ("FORTE_BUNDLE_DIR", "BUNDLE_DIR", "BASE_MODEL_PATH", "LOAD_4BIT"):
        monkeypatch.delenv(name, raising=False)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "pipeline_config.json").write_text('{"cfg":{}}', encoding="utf-8")
    (bundle / "decision.json").write_text("{}", encoding="utf-8")
    if setting == "default":
        monkeypatch.chdir(tmp_path)
    else:
        monkeypatch.setenv(setting, str(bundle))
    if setting == "FORTE_BUNDLE_DIR":
        monkeypatch.setenv("BUNDLE_DIR", str(tmp_path / "wrong-legacy-path"))
    calls = []

    def fake_load(path, **kwargs):
        calls.append(path)
        assert kwargs == {"load_4bit": None, "merge": True}
        return "loaded"

    monkeypatch.setattr(model, "load_bundle", fake_load)
    assert model.load_analyzer() == "loaded"
    assert calls == [bundle.resolve()]


def test_v3_missing_decision_is_rejected_before_model_loading(monkeypatch, tmp_path):
    monkeypatch.delenv("MOCK_MODEL")
    monkeypatch.setenv("FORTE_BUNDLE_DIR", str(tmp_path))
    (tmp_path / "pipeline_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(model, "load_bundle", lambda *a, **kw: pytest.fail("Weights must not be loaded"))
    with pytest.raises(RuntimeError, match="decision.json"):
        model.load_analyzer()


@pytest.mark.parametrize("mock", [False, True])
def test_pipeline_import_needs_no_gpu_packages(mock):
    script = '''
import importlib.abc, os, sys
class BlockGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = {"torch", "torchvision", "transformers", "peft", "accelerate", "bitsandbytes"}
        if os.environ.get("MOCK_MODEL") == "1":
            forbidden.add("fc_pipeline")
        if fullname.split(".")[0] in forbidden:
            raise AssertionError("Unexpected import: " + fullname)
sys.meta_path.insert(0, BlockGPU())
if os.environ.get("MOCK_MODEL") == "1":
    from app.model import pipeline_module, load_analyzer
    assert pipeline_module().__name__ == "fc_pipeline_mock"
    assert load_analyzer().__class__.__name__ == "MockAnalyzer"
else:
    import fc_pipeline
    assert fc_pipeline.__file__.endswith("fc_pipeline.py")
    for name in ("load_bundle", "Video", "ffprobe_duration", "extract_frames", "to_api"):
        assert callable(getattr(fc_pipeline, name))
    assert hasattr(fc_pipeline, "ChecklistAnalyzer")
assert "torch" not in sys.modules
'''
    environment = dict(os.environ)
    if mock:
        environment["MOCK_MODEL"] = "1"
    else:
        environment.pop("MOCK_MODEL", None)
    subprocess.run([sys.executable, "-X", "utf8", "-c", script], env=environment,
                   cwd=Path(__file__).resolve().parents[1], capture_output=True, check=True, timeout=20)


def test_v3_internal_temp_directories_use_existing_job_workspace(monkeypatch, tmp_path):
    import tempfile
    monkeypatch.delenv("MOCK_MODEL")
    original = tempfile.tempdir
    nested = None
    with pytest.raises(RuntimeError, match="test failure"):
        with model.pipeline_temp_directory(str(tmp_path)):
            with tempfile.TemporaryDirectory() as directory:
                nested = Path(directory)
                assert nested.parent == tmp_path
                (nested / "frame.jpg").write_bytes(b"test")
                raise RuntimeError("test failure")
    assert not nested.exists()
    assert tempfile.tempdir == original


def test_requested_api_key_is_used_without_changing_auth(client_factory, monkeypatch):
    monkeypatch.setenv("API_KEY", "test12345")
    client, _, _ = client_factory()
    assert client.get("/api/v1/jobs/unknown", headers={"X-API-Key": "test12345"}).status_code == 404
    assert client.get("/api/v1/jobs/unknown", headers={"X-API-Key": "wrong"}).status_code == 401
