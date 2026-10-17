from __future__ import annotations

import base64
import copy
from pathlib import Path
import subprocess
import threading
import time

from fastapi.testclient import TestClient
import pytest

from app.main import create_app
from app.model import MockAnalyzer


@pytest.fixture(autouse=True)
def offline_mock(monkeypatch):
    monkeypatch.setenv("MOCK_MODEL", "1")
    monkeypatch.setenv("API_KEY", "test-secret")


@pytest.fixture(scope="session")
def mov_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "sample.mov"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
        "-i", "testsrc2=size=320x180:rate=10", "-t", "2",
        "-c:v", "mpeg4", "-q:v", "6", str(path),
    ], check=True, timeout=20)
    return path


class ControlledAnalyzer(MockAnalyzer):
    """Real pipeline frame extraction; deterministic model outcomes for races."""

    def __init__(self, *, hold=False, fail_first=False, no_violation=False):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        if not hold:
            self.release.set()
        self.fail_first, self.no_violation = fail_first, no_violation
        self.calls = []
        self.active = self.maximum_active = 0
        self.lock = threading.Lock()

    def analyze(self, v):
        with self.lock:
            self.calls.append(Path(v.path).parent.name.split("-")[0])
            number = len(self.calls)
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        self.entered.set()
        try:
            if not self.release.wait(timeout=10):
                raise RuntimeError("Test gate was not released")
            if self.fail_first and number == 1:
                raise RuntimeError("Intentional model failure")
            if self.no_violation:
                return {"verdict": "no_violation", "violations": []}
            image = base64.b64encode(v.files[0].read_bytes()).decode("ascii")
            return {"verdict": "violation", "violations": [{
                "type_key": "test", "type": "Проверка наличных", "timecode_sec": 0.5,
                "score": 76, "description": "Проверьте указанный момент.",
                "description_en": "Review this moment.", "rule_ref": "Регламент: заполняет команда",
                "evidence_frames": [{"timestamp_sec": 0.5, "image_base64": image}],
            }]}
        finally:
            with self.lock:
                self.active -= 1


@pytest.fixture
def client_factory(tmp_path):
    contexts = []
    analyzers = []

    def make(analyzer=None, *, timeout=270):
        analyzer = analyzer or ControlledAnalyzer()
        analyzers.append(analyzer)
        app = create_app(model_loader=lambda: analyzer,
                         jobs_dir=tmp_path / f"jobs-{len(contexts)}", job_timeout=timeout)
        context = TestClient(app)
        client = context.__enter__()
        contexts.append(context)
        return client, app, analyzer

    yield make
    for analyzer in analyzers:
        analyzer.release.set()
    for context in reversed(contexts):
        context.__exit__(None, None, None)


def submit(client, mov_path, *, ui=False, name="sample.mov"):
    response = client.post("/ui/analyze" if ui else "/api/v1/analyze",
                           headers={} if ui else {"X-API-Key": "test-secret"},
                           files={"file": (name, mov_path.read_bytes(), "video/quicktime")})
    assert response.status_code == 200, response.text
    assert set(response.json()) == {"job_id"}
    return response.json()["job_id"]


def wait_for(client, job_id, *, status=None, ui=False, timeout=8):
    deadline = time.monotonic() + timeout
    path = f"/ui/jobs/{job_id}" if ui else f"/api/v1/jobs/{job_id}"
    while time.monotonic() < deadline:
        response = client.get(path, headers={} if ui else {"X-API-Key": "test-secret"})
        assert response.status_code == 200, response.text
        data = response.json()
        if data["status"] == status if status else data["status"] != "processing":
            return copy.deepcopy(data)
        time.sleep(0.02)
    raise AssertionError(f"Job {job_id} did not finish: {data}")

