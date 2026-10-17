"""Live mock HTTP checks; all server interpreters explicitly use -X utf8."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import httpx


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "jobs")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--limit", type=float, default=270)
    return parser.parse_args()


@contextmanager
def server(root, limit=270):
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    capture = tempfile.TemporaryFile()
    process = subprocess.Popen([
        sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--serve",
        "--root", str(root), "--port", str(port), "--limit", str(limit),
    ], stdout=capture, stderr=subprocess.STDOUT)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", headers={"X-API-Key": "data-smoke-key"}, timeout=15) as client:
            deadline = time.monotonic() + 25
            while True:
                if process.poll() is not None:
                    raise AssertionError("Mock server did not start")
                try:
                    if client.get("/health").status_code == 200:
                        break
                except httpx.ConnectError:
                    pass
                if time.monotonic() >= deadline:
                    raise AssertionError("Mock startup timed out")
                time.sleep(0.05)
            yield client
    finally:
        process.terminate()
        process.wait(timeout=10)
        capture.seek(0)
        output = capture.read()
        capture.close()
        assert b"PRIVATE_UPLOAD_NAME" not in output
        assert b"PRIVATE_VIDEO_CONTENT" not in output


def poll(client, job_id):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        if data["status"] != "processing":
            return data
        time.sleep(0.05)
    raise AssertionError("Job did not finish")


def check_media(root, job_id):
    assert not list((root / "uploads").iterdir())
    assert [p.name for p in (root / job_id).iterdir()] == ["state.json"]


def main(args):
    root = args.root.resolve()
    output = Path(__file__).resolve().parents[1] / ".verification"
    output.mkdir(exist_ok=True)
    video = output / "data-smoke.mov"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x180:rate=10", "-t", "2", "-c:v", "mpeg4", "-q:v", "6", str(video)],
                   check=True, capture_output=True, timeout=20)
    report = {}
    with server(root) as client:
        response = client.post("/api/v1/analyze", files={"file": ("PRIVATE_UPLOAD_NAME.mov", video.read_bytes(), "video/quicktime")})
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        result = poll(client, job_id)
        assert result["status"] == "done"
        check_media(root, job_id)
        assert client.get(f"/api/v1/jobs/{job_id}").json() == result
        report["done"] = {"job_id": job_id, "media_deleted": True, "result_available": True}
        failed_id = client.post("/api/v1/analyze", files={"file": ("PRIVATE_UPLOAD_NAME.mov", b"PRIVATE_VIDEO_CONTENT")}).json()["job_id"]
        assert poll(client, failed_id)["status"] == "failed"
        check_media(root, failed_id)
        report["failed"] = {"job_id": failed_id, "media_deleted": True}

    (root / "uploads" / "garbage.mov").write_bytes(b"old upload")
    leftovers = root / "uploads" / "old-workspace" / "frames"
    leftovers.mkdir(parents=True)
    (leftovers / "00001.jpg").write_bytes(b"old frame")
    with server(root) as client:
        assert not list((root / "uploads").iterdir())
        assert client.get(f"/api/v1/jobs/{job_id}").json() == result
        report["restart"] = {"garbage_deleted": True, "previous_result_available": True}

    with server(root, limit=0.8) as client:
        timeout_id = client.post("/api/v1/analyze", files={"file": ("PRIVATE_UPLOAD_NAME.mov", video.read_bytes(), "video/quicktime")}).json()["job_id"]
        timed_out = poll(client, timeout_id)
        assert timed_out["status"] == "failed" and "0.8" in timed_out["error"]
        check_media(root, timeout_id)
        report["timeout"] = {"job_id": timeout_id, "media_deleted": True, "status": "failed"}
    (output / "data-lifecycle-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    options = arguments()
    if options.serve:
        # Keep imports after the project root is set when invoked as a script.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        os.environ["MOCK_MODEL"], os.environ["API_KEY"] = "1", "data-smoke-key"
        import uvicorn
        from app.main import create_app
        uvicorn.run(create_app(jobs_dir=options.root, job_timeout=options.limit), host="127.0.0.1", port=options.port)
    else:
        main(options)
