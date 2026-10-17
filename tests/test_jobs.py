from __future__ import annotations

import json
import os
from pathlib import Path
import time

from app.jobs import JobManager
from conftest import ControlledAnalyzer, submit, wait_for


def test_running_timeout_is_hard_status_deadline_and_late_result_discarded(client_factory, mov_path):
    analyzer = ControlledAnalyzer(hold=True)
    client, app, _ = client_factory(analyzer, timeout=1.2)
    started = time.monotonic()
    job_id = submit(client, mov_path)
    assert analyzer.entered.wait(1)
    failed = wait_for(client, job_id, timeout=3)
    assert failed["status"] == "failed"
    assert set(failed) == {"job_id", "status", "error"}
    assert "1.2" in failed["error"]
    assert time.monotonic() - started < 2.5
    assert analyzer.active == 1  # The GPU owner is never replaced by another thread.
    assert not list(app.state.jobs.upload_root.iterdir())
    analyzer.release.set()
    app.state.jobs.queue.join()
    assert client.get(f"/ui/jobs/{job_id}").json() == failed
    assert app.state.jobs.worker.is_alive()
    assert wait_for(client, submit(client, mov_path))["status"] == "done"
    assert not list(app.state.jobs.root.rglob("*.jpg"))


def test_queued_job_expires_without_running(client_factory, mov_path):
    analyzer = ControlledAnalyzer(hold=True)
    client, app, _ = client_factory(analyzer, timeout=1.5)
    first = submit(client, mov_path)
    assert analyzer.entered.wait(1)
    second = submit(client, mov_path)
    assert wait_for(client, first, timeout=3)["status"] == "failed"
    assert wait_for(client, second, timeout=3)["status"] == "failed"
    analyzer.release.set()
    app.state.jobs.queue.join()
    assert analyzer.calls == [first]
    assert analyzer.maximum_active == 1
    assert client.get("/health").json()["queue_length"] == 0


def test_deadline_includes_upload_time(tmp_path):
    analyzer = ControlledAnalyzer()
    manager = JobManager(analyzer, tmp_path / "jobs", timeout=0.2)
    manager.start()
    try:
        job = manager.reserve(".mov", time.monotonic() - 1)
        job.path.write_bytes(b"unused")
        manager.enqueue(job)
        assert manager.snapshot(job.job_id)["status"] == "failed"
        assert analyzer.calls == []
    finally:
        manager.close()


def test_old_job_cleanup_and_unrelated_directory_preserved(tmp_path):
    root = tmp_path / "jobs"
    root.mkdir()
    old = root / ("a" * 32)
    old.mkdir()
    (old / "video.mov").write_bytes(b"old")
    os.utime(old, (time.time() - 3601, time.time() - 3601))
    unrelated = root / "keep-me"
    unrelated.mkdir()
    recently_modified_old_job = root / ("b" * 32)
    recently_modified_old_job.mkdir()
    (recently_modified_old_job / "state.json").write_text(
        json.dumps({"created_at": time.time() - 3601}), encoding="utf-8"
    )
    manager = JobManager(ControlledAnalyzer(), root, retention=3600)
    manager.start()
    try:
        assert not old.exists()
        assert not recently_modified_old_job.exists()
        assert unrelated.exists()
        job = manager.reserve(".mov", time.monotonic())
        job.uploading = False
        job.created_wall = time.time() - 3601
        job.status = "failed"
        manager.cleanup()
        assert manager.snapshot(job.job_id) is None
        assert not job.directory.exists()
    finally:
        manager.close()

