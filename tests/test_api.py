from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import re
import time

from fastapi.testclient import TestClient
from PIL import Image
import pytest

import app.main as main
from app.main import create_app
from conftest import ControlledAnalyzer, submit, wait_for


@pytest.mark.parametrize("key", [None, "wrong", "test-secret-extra", "тест"])
def test_auth_required_before_parsing(client_factory, key):
    client, _, _ = client_factory()
    headers = {"X-API-Key": key.encode("utf-8")} if key is not None else {}
    for method, path in [("POST", "/api/v1/analyze"), ("GET", "/api/v1/jobs/unknown")]:
        response = client.request(method, path, headers=headers, content=b"malformed multipart")
        assert response.status_code == 401
        assert set(response.json()) == {"error"}


def test_health_and_static_are_public_and_local(client_factory):
    client, _, _ = client_factory()
    assert client.get("/health").json() == {"model_loaded": True, "gpu_name": None, "queue_length": 0}
    page = client.get("/")
    assert page.status_code == 200
    assert 'lang="ru"' in page.text
    assert "test-secret" not in page.text
    assert "connect-src 'self'" in page.headers["content-security-policy"]
    assert client.get("/ui/config").json() == {"mock_model": True}
    assert client.get("/docs").status_code == 404
    for path in (main.STATIC).iterdir():
        assert not re.search(r"https?://", path.read_text(encoding="utf-8")), path


@pytest.mark.parametrize("job_id", ["does-not-exist", "a" * 32, "../outside", "DEADBEEF"])
def test_unknown_id(client_factory, job_id):
    client, _, _ = client_factory()
    assert client.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": "test-secret"}).status_code == 404
    assert client.get(f"/ui/jobs/{job_id}").status_code == 404


def test_exact_api_schema_and_ui_only_extras(client_factory, mov_path):
    client, app, _ = client_factory()
    job_id = submit(client, mov_path)
    assert re.fullmatch(r"[0-9a-f]{32}", job_id)
    result = wait_for(client, job_id)
    assert result["status"] == "done"
    assert set(result) == {"job_id", "status", "verdict", "violations"}
    assert result["verdict"] == "violation"
    for violation in result["violations"]:
        assert set(violation) == {"type", "timecode_sec", "description", "evidence_frames"}
        assert isinstance(violation["timecode_sec"], float)
        assert 1 <= len(violation["evidence_frames"]) <= 3
        for frame in violation["evidence_frames"]:
            assert set(frame) == {"timestamp_sec", "image_base64"}
            with Image.open(io.BytesIO(base64.b64decode(frame["image_base64"], validate=True))) as image:
                assert image.format == "JPEG"
    ui = client.get(f"/ui/jobs/{job_id}").json()
    assert set(ui["violations"][0]) == {"type", "timecode_sec", "description", "evidence_frames", "score", "rule_ref"}
    assert 0 <= ui["violations"][0]["score"] <= 100
    assert ui["violations"][0]["rule_ref"]
    directory = app.state.jobs.root / job_id
    assert not (directory / "video.mov").exists()
    assert list(directory.iterdir()) == [directory / "state.json"]
    assert not list(app.state.jobs.upload_root.iterdir())
    assert json.loads((directory / "state.json").read_text(encoding="utf-8"))["status"] == "done"
    assert not list(directory.rglob("*.jpg"))


def test_no_violation(client_factory, mov_path):
    client, _, _ = client_factory(ControlledAnalyzer(no_violation=True))
    result = wait_for(client, submit(client, mov_path, ui=True), ui=True)
    assert result["verdict"] == "no_violation"
    assert result["violations"] == []


@pytest.mark.parametrize("name", ["cash.avi", "cash.mkv", "cash.txt", "video", "cash.mov.exe"])
@pytest.mark.parametrize("path", ["/api/v1/analyze", "/ui/analyze"])
def test_wrong_extension(client_factory, name, path):
    client, _, _ = client_factory()
    response = client.post(path, headers={"X-API-Key": "test-secret"}, files={"file": (name, b"video")})
    assert response.status_code == 400


@pytest.mark.parametrize("filename", ["video.MOV", "video.mp4"])
def test_supported_extensions(client_factory, mov_path, filename):
    client, _, _ = client_factory()
    result = wait_for(client, submit(client, mov_path, name=filename))
    assert result["status"] == "done"


def test_missing_empty_extra_and_oversized_file(client_factory, monkeypatch):
    client, _, _ = client_factory()
    assert client.post("/ui/analyze").status_code == 400
    assert client.post("/ui/analyze", files={"other": ("x.mov", b"x")}).status_code == 400
    assert client.post("/ui/analyze", files={"file": ("x.mov", b"")}).status_code == 400
    assert client.post("/ui/analyze", files=[("file", ("a.mov", b"x")), ("file", ("b.mov", b"x"))]).status_code == 400
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 16)
    assert client.post("/ui/analyze", files={"file": ("x.mov", b"x" * 17)}).status_code == 413
    # Exercise the receive-stream limit, independent of Content-Length.
    assert client.post("/ui/analyze", files={"file": ("x.mov", b"x" * 70000)}).status_code == 413


def test_corrupt_video_fails_and_worker_recovers(client_factory, mov_path):
    client, app, _ = client_factory()
    response = client.post("/ui/analyze", files={"file": ("broken.mov", b"not a video")})
    failed = wait_for(client, response.json()["job_id"], ui=True)
    assert set(failed) == {"job_id", "status", "error"}
    assert failed["status"] == "failed"
    assert app.state.jobs.worker.is_alive()
    assert wait_for(client, submit(client, mov_path))["status"] == "done"


def test_model_exception_fails_job_but_worker_survives(client_factory, mov_path):
    client, app, analyzer = client_factory(ControlledAnalyzer(fail_first=True))
    bad = wait_for(client, submit(client, mov_path))
    assert bad["status"] == "failed"
    assert set(bad) == {"job_id", "status", "error"}
    assert "RuntimeError" in bad["error"]
    assert app.state.jobs.worker.is_alive()
    assert wait_for(client, submit(client, mov_path))["status"] == "done"
    assert len(analyzer.calls) == 2
    assert not list(app.state.jobs.root.rglob("*.jpg"))


def test_fifo_and_concurrent_uploads(client_factory, mov_path):
    analyzer = ControlledAnalyzer(hold=True)
    client, app, _ = client_factory(analyzer)
    first = submit(client, mov_path)
    assert analyzer.entered.wait(3)
    second = submit(client, mov_path, ui=True)
    third = submit(client, mov_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        others = list(pool.map(lambda _: submit(client, mov_path), range(4)))
    ids = [first, second, third, *others]
    assert len(set(ids)) == 7
    for job_id in ids:
        assert client.get(f"/ui/jobs/{job_id}").json() == {"job_id": job_id, "status": "processing"}
    assert client.get("/health").json()["queue_length"] == 6
    with app.state.jobs.queue.mutex:
        enqueued_order = [job.job_id for job in app.state.jobs.queue.queue]
    analyzer.release.set()
    for job_id in ids:
        assert wait_for(client, job_id)["status"] == "done"
    assert analyzer.calls == [first, *enqueued_order]
    assert analyzer.calls[:3] == [first, second, third]
    assert analyzer.maximum_active == 1


def test_required_api_key(client_factory, monkeypatch, tmp_path):
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API_KEY"):
        with TestClient(create_app(jobs_dir=tmp_path / "jobs")):
            pass


def test_model_loaded_once(client_factory, mov_path):
    client, app, analyzer = client_factory()
    first = submit(client, mov_path)
    second = submit(client, mov_path)
    assert wait_for(client, first)["status"] == "done"
    assert wait_for(client, second)["status"] == "done"
    assert app.state.jobs.analyzer is analyzer
    assert len(analyzer.calls) == 2


def test_full_mock_real_jpegs(tmp_path, mov_path):
    started = time.monotonic()
    with TestClient(create_app(jobs_dir=tmp_path / "real-mock")) as client:
        result = wait_for(client, submit(client, mov_path), timeout=12)
        assert result["status"] == "done"
        assert 1 <= len(result["violations"]) <= 2
        assert len(result["violations"][0]["evidence_frames"]) == 3
        assert time.monotonic() - started >= 3
        for frame in result["violations"][0]["evidence_frames"]:
            with Image.open(io.BytesIO(base64.b64decode(frame["image_base64"]))) as image:
                assert image.format == "JPEG"
                assert image.width == 640

