from __future__ import annotations

import json
from pathlib import Path

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
    (bundle / "adapter").mkdir(parents=True)
    (bundle / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    original = '{"cfg":{"model_id":"cached/model"},"tau":0.5}'
    config = bundle / "pipeline_config.json"
    config.write_text(original, encoding="utf-8")
    base = tmp_path / "local-model"
    base.mkdir()
    monkeypatch.setenv("BUNDLE_DIR", str(bundle))
    monkeypatch.setenv("BASE_MODEL_PATH", str(base))
    monkeypatch.setenv("LOAD_4BIT", "0")
    calls = []

    def fake_load(path, **kwargs):
        path = Path(path)
        calls.append(path)
        assert json.loads((path / "pipeline_config.json").read_text())["cfg"]["model_id"] == str(base.resolve())
        assert (path / "adapter" / "adapter_config.json").is_file()
        assert kwargs == {"load_4bit": False, "merge": True}
        return "loaded"

    monkeypatch.setattr(model, "load_bundle", fake_load)
    assert model.load_analyzer() == "loaded"
    assert config.read_text(encoding="utf-8") == original
    assert len(calls) == 1
    assert not calls[0].exists()
