from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading
import time

from fastapi.testclient import TestClient
import psutil
import pytest

from app.jobs import JobManager
from app.main import create_app
import app.worker as worker_module
from conftest import ControlledAnalyzer, submit, wait_for


def assert_media_removed(manager, job_id):
    assert not list(manager.upload_root.iterdir())
    assert not manager.jobs[job_id].path.exists()
    assert list((manager.root / job_id).iterdir()) == [manager.root / job_id / "state.json"]


def test_real_mock_done_and_corrupt_failed_remove_all_media(tmp_path, mov_path, caplog, capfd):
    with TestClient(create_app(jobs_dir=tmp_path / "jobs")) as client:
        manager = client.app.state.jobs
        model_pid = manager.analyzer.process.pid
        job_id = submit(client, mov_path)
        result = wait_for(client, job_id, timeout=15)
        assert result["status"] == "done"
        assert_media_removed(manager, job_id)
        assert client.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": "test-secret"}).json() == result
        assert manager.analyzer.process.pid == model_pid

        response = client.post("/api/v1/analyze", headers={"X-API-Key": "test-secret"},
                               files={"file": ("SECRET_JURY_FILENAME.mov", b"PRIVATE_VIDEO_CONTENT", "video/quicktime")})
        failed_id = response.json()["job_id"]
        failed = wait_for(client, failed_id)
        assert failed["status"] == "failed"
        assert_media_removed(manager, failed_id)
        assert manager.analyzer.process.pid == model_pid  # A bad file doesn't reload the model.
        captured = capfd.readouterr()
        assert "SECRET_JURY_FILENAME" not in captured.out + captured.err + caplog.text
        assert "PRIVATE_VIDEO_CONTENT" not in captured.out + captured.err + caplog.text
        job_logs = [record.getMessage() for record in caplog.records if record.name == "app.jobs"]
        assert job_logs == [f"job_id={failed_id}"]


def test_real_mock_timeout_deletes_immediately_and_worker_recovers(tmp_path, mov_path):
    with TestClient(create_app(jobs_dir=tmp_path / "jobs", job_timeout=0.8)) as client:
        manager = client.app.state.jobs
        previous = manager.analyzer.process.pid
        job_id = submit(client, mov_path)
        result = wait_for(client, job_id, timeout=5)
        assert result["status"] == "failed"
        assert "0.8" in result["error"]
        assert_media_removed(manager, job_id)
        assert not psutil.pid_exists(previous)
        assert manager.worker.is_alive()
        manager.timeout = 20  # Only the test shortens/changes the production limit.
        next_id = submit(client, mov_path)
        assert wait_for(client, next_id, timeout=15)["status"] == "done"
        assert_media_removed(manager, next_id)
        assert manager.analyzer.process.pid != previous


def test_restart_removes_garbage_legacy_media_and_preserves_json(tmp_path, mov_path):
    root = tmp_path / "jobs"
    with TestClient(create_app(model_loader=ControlledAnalyzer, jobs_dir=root)) as client:
        job_id = submit(client, mov_path)
        result = wait_for(client, job_id)
    (root / "uploads" / "garbage.mov").write_bytes(b"private old upload")
    leftover = root / "uploads" / "unfinished" / "frames"
    leftover.mkdir(parents=True)
    (leftover / "00001.jpg").write_bytes(b"old frame")
    # Migration also removes videos left by the previous service layout.
    (root / job_id / "video.mov").write_bytes(b"legacy video")
    with TestClient(create_app(model_loader=ControlledAnalyzer, jobs_dir=root)) as restarted:
        manager = restarted.app.state.jobs
        assert_media_removed(manager, job_id)
        assert restarted.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": "test-secret"}).json() == result


def test_result_ttl_does_not_retain_media(tmp_path, mov_path):
    with TestClient(create_app(model_loader=ControlledAnalyzer, jobs_dir=tmp_path / "jobs", result_ttl=0.4)) as client:
        manager = client.app.state.jobs
        job_id = submit(client, mov_path)
        assert wait_for(client, job_id)["status"] == "done"
        assert_media_removed(manager, job_id)
        time.sleep(0.45)
        assert client.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": "test-secret"}).status_code == 404
        assert not (manager.root / job_id).exists()


def test_dead_fifo_thread_is_detected_and_upload_deleted(tmp_path, mov_path):
    manager = JobManager(ControlledAnalyzer(), tmp_path / "jobs")
    manager.worker = threading.Thread(target=lambda: None)
    manager.start()
    try:
        job = manager.reserve(".mov", time.monotonic())
        job.path.write_bytes(mov_path.read_bytes())
        manager.enqueue(job)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and manager.snapshot(job.job_id)["status"] == "processing":
            time.sleep(0.02)
        assert manager.snapshot(job.job_id)["status"] == "failed"
        assert_media_removed(manager, job.job_id)
    finally:
        manager.close()


def _native_holder(connection):
    """Test worker owning a long-running ffmpeg which keeps input/output open."""
    import os
    worker_module._silence_output()
    if os.name == "posix":
        os.setsid()
    assert connection.recv() == "start"
    connection.send(("ready", None))
    path, workspace = connection.recv()
    frames = Path(workspace) / "native-frames"
    frames.mkdir()
    child = subprocess.Popen([
        "ffmpeg", "-nostdin", "-v", "error", "-re", "-stream_loop", "-1", "-i", path,
        "-vf", "fps=2", str(frames / "%05d.jpg"),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    (Path(workspace) / "native.pid").write_text(str(child.pid), encoding="utf-8")
    while True:
        time.sleep(1)


@pytest.mark.parametrize("cause", ["timeout", "process_crash"])
def test_native_handles_closed_before_media_deleted(tmp_path, mov_path, monkeypatch, cause):
    monkeypatch.setattr(worker_module, "_serve", _native_holder)
    with TestClient(create_app(jobs_dir=tmp_path / "jobs", job_timeout=3)) as client:
        manager = client.app.state.jobs
        job_id = submit(client, mov_path)
        job = manager.jobs[job_id]
        marker = job.path.parent / "native.pid"
        deadline = time.monotonic() + 2
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists()
        child_pid = int(marker.read_text(encoding="utf-8"))
        assert psutil.pid_exists(child_pid)
        if cause == "process_crash":
            manager.analyzer.process.kill()
        result = wait_for(client, job_id, timeout=6)
        assert result["status"] == "failed"
        assert_media_removed(manager, job_id)
        assert manager.worker.is_alive()
        deadline = time.monotonic() + 2
        while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not psutil.pid_exists(child_pid)


def test_video_is_not_served(client_factory, mov_path):
    client, _, _ = client_factory()
    job_id = submit(client, mov_path)
    assert wait_for(client, job_id)["status"] == "done"
    for path in (f"/jobs/{job_id}/video.mov", f"/api/v1/jobs/{job_id}/video", f"/ui/jobs/{job_id}/video", f"/uploads/{job_id}/video.mov"):
        assert client.get(path, headers={"X-API-Key": "test-secret"}).status_code == 404


def test_multipart_spool_closed_on_stream_limit(monkeypatch):
    import asyncio
    import tempfile
    import app.main as main_module
    import starlette.formparsers as formparsers

    spools = []

    def create_spool(*args, **kwargs):
        spool = tempfile.SpooledTemporaryFile(*args, **kwargs)
        spool.rollover()
        spools.append(spool)
        return spool

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", create_spool)
    header = b'--privacy-check\r\nContent-Disposition: form-data; name="file"; filename="private.mov"\r\nContent-Type: video/quicktime\r\n\r\n'

    async def chunks():
        yield header
        yield b"x" * 100
        raise main_module.UploadLimitExceeded()

    async def parse():
        parser = main_module.ClosingMultiPartParser(
            {"Content-Type": "multipart/form-data; boundary=privacy-check"}, chunks())
        with pytest.raises(main_module.UploadLimitExceeded):
            await parser.parse()

    asyncio.run(parse())
    assert len(spools) == 1
    assert all(spool.closed for spool in spools)
