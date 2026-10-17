"""A single FIFO inference thread plus an independent deadline/retention monitor."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import io
import json
import logging
import math
from pathlib import Path
import queue
import re
import shutil
import tempfile
import threading
import time
import uuid

from PIL import Image

from app.model import VideoAnalyzer, analyze_video
from app.worker import ProcessAnalyzer, clear_orphan

logger = logging.getLogger(__name__)
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
JOB_TIMEOUT = 270.0
RETENTION_SECONDS = 6 * 3600.0


@dataclass
class Job:
    job_id: str
    directory: Path
    path: Path
    created: float
    created_wall: float = field(default_factory=time.time)
    status: str = "processing"
    api: dict | None = None
    raw: dict | None = None
    error: str | None = None
    active: bool = False
    uploading: bool = True
    temporary: tempfile.TemporaryDirectory | None = None
    finished_wall: float | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)


def validate_result(result: dict) -> tuple[dict, dict]:
    """Explicit whitelists prevent raw model metadata leaking into the API."""
    api, raw = result["api"], result["raw"]
    if api["verdict"] not in {"violation", "no_violation"}:
        raise ValueError("Модель вернула неизвестный verdict")
    if len(api["violations"]) != len(raw["violations"]):
        raise ValueError("Модель вернула несогласованный результат")
    if (api["verdict"] == "violation") != bool(api["violations"]):
        raise ValueError("verdict не соответствует списку нарушений")
    clean, ui = [], []
    for violation, original in zip(api["violations"], raw["violations"]):
        timestamp = float(violation["timecode_sec"])
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Некорректный таймкод модели")
        if not 1 <= len(violation["evidence_frames"]) <= 3:
            raise ValueError("Требуется от 1 до 3 кадров на нарушение")
        frames = []
        for frame in violation["evidence_frames"]:
            at, encoded = float(frame["timestamp_sec"]), frame["image_base64"]
            if not math.isfinite(at) or at < 0:
                raise ValueError("Некорректная отметка времени кадра")
            data = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "JPEG":
                    raise ValueError("Кадр должен быть JPEG")
                image.verify()
            frames.append({"timestamp_sec": at, "image_base64": encoded})
        item = {
            "type": str(violation["type"]), "timecode_sec": timestamp,
            "description": str(violation["description"]), "evidence_frames": frames,
        }
        score = float(original["score"])
        if not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError("Оценка модели должна находиться в диапазоне 0–100")
        clean.append(item)
        ui.append({**item, "description": str(original["description"]),
                   "score": score, "rule_ref": str(original.get("rule_ref") or "")})
    return {"verdict": api["verdict"], "violations": clean}, {"verdict": api["verdict"], "violations": ui}


class JobManager:
    def __init__(self, analyzer: VideoAnalyzer | ProcessAnalyzer | None, root: Path, *, timeout: float = JOB_TIMEOUT,
                 retention: float = RETENTION_SECONDS, monitor_interval: float = 0.1) -> None:
        self.analyzer = analyzer
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.upload_root = self.root / "uploads"
        if self.upload_root.is_symlink():
            raise RuntimeError("Папка загрузок не должна быть символической ссылкой")
        self.upload_root.mkdir(exist_ok=True)
        if self.upload_root.resolve().parent != self.root:
            raise RuntimeError("Папка загрузок должна находиться внутри jobs")
        self.timeout, self.retention = timeout, retention
        self.monitor_interval = monitor_interval
        self.queue: queue.Queue[Job | None] = queue.Queue()
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.worker = threading.Thread(target=self._work, name="gpu-fifo-worker", daemon=True)
        self.monitor = threading.Thread(target=self._monitor, name="job-deadlines", daemon=True)
        self.prepared = False

    def _remove(self, path: Path) -> None:
        # Check the absolute target before every recursive removal. In particular,
        # never traverse a Windows junction or a symlink into someone else's data.
        if path.absolute() == self.root or not path.absolute().is_relative_to(self.root):
            raise ValueError("Удаление вне папки jobs запрещено")
        if path.is_symlink():
            path.unlink()
        elif not path.resolve().is_relative_to(self.root):
            path.rmdir()  # Remove a directory junction itself, without traversal.
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

    def prepare_storage(self) -> None:
        """Before model loading: remove uploads, including previous-version media."""
        clear_orphan(self.root / ".worker.json")
        (self.root / ".worker.json").unlink(missing_ok=True)
        for entry in self.upload_root.iterdir():
            self._remove(entry)
        for directory in self.root.iterdir():
            if not JOB_ID.fullmatch(directory.name):
                continue
            if directory.is_symlink() or not directory.is_dir() or directory.resolve().parent != self.root:
                self._remove(directory)
                continue
            # Migrate the old jobs/<id> layout: preserve only the JSON result.
            for entry in directory.iterdir():
                if entry.name != "state.json" or entry.is_symlink():
                    self._remove(entry)
            try:
                state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
                created = float(state["created_at"])
                finished = float(state.get("finished_at") or created)
                if not math.isfinite(finished) or time.time() - finished >= self.retention:
                    self._remove(directory)
                    continue
                status = state["status"]
                api, ui = None, None
                if status == "done":
                    api, ui = validate_result({"api": state, "raw": state["ui_result"]})
                elif status != "failed":
                    status = "failed"
                    state["error"] = "Обработка прервана перезапуском сервиса. Повторите попытку."
                    finished = time.time()
                job = Job(directory.name, directory, self.upload_root / directory.name / "video.mov",
                          time.monotonic(), created_wall=created, status=status, api=api, raw=ui,
                          error=state.get("error"), uploading=False, finished_wall=finished)
                self.jobs[job.job_id] = job
                self._persist(job)
            except (OSError, ValueError, KeyError, TypeError):
                self._remove(directory)
        self.prepared = True

    def start(self) -> None:
        if not self.prepared:
            self.prepare_storage()
        self.worker.start()
        self.monitor.start()

    def close(self) -> None:
        self.stopping.set()
        with self.lock:
            for job in list(self.jobs.values()):
                if job.status == "processing":
                    self._finish(job, error="Обработка прервана остановкой сервиса. Повторите попытку.", cancel=True)
            if isinstance(self.analyzer, ProcessAnalyzer):
                self.analyzer.close()
        self.queue.put(None)
        self.monitor.join(timeout=2)
        self.worker.join(timeout=5)

    def reserve(self, suffix: str, started: float) -> Job:
        job_id = uuid.uuid4().hex
        directory = self.root / job_id
        directory.mkdir()
        temporary = tempfile.TemporaryDirectory(prefix=f"{job_id}-", dir=self.upload_root)
        job = Job(job_id, directory, Path(temporary.name) / f"video{suffix}", started,
                  temporary=temporary)
        with self.lock:
            self.jobs[job_id] = job
        return job

    def enqueue(self, job: Job) -> None:
        with self.lock:
            job.uploading = False
            self._expire(job)
            self._persist(job)
            if job.status == "processing":
                self.queue.put(job)

    def discard_upload(self, job: Job) -> None:
        with self.lock:
            self._remove_media(job)
            self.jobs.pop(job.job_id, None)
            self._remove(job.directory)

    def _remove_media(self, job: Job) -> None:
        if job.temporary is not None:
            # Windows can release a killed process's final file handle shortly
            # after its exit notification. Retry without logging the file path.
            deadline = time.monotonic() + 3
            while True:
                try:
                    job.temporary.cleanup()
                    break
                except PermissionError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.02)
            job.temporary = None

    def _finish(self, job: Job, *, result=None, error=None, cancel=False) -> None:
        """Publish a terminal status only AFTER all task media have been removed."""
        with self.lock:
            if job.status != "processing":
                return
            try:
                if cancel:
                    job.cancelled.set()
                    if job.active and isinstance(self.analyzer, ProcessAnalyzer) and (self.analyzer.busy or not self.analyzer.loaded):
                        self.analyzer.cancel()
            finally:
                self._remove_media(job)
            job.uploading = False
            job.finished_wall = time.time()
            if result is not None:
                job.api, job.raw = result
                job.status = "done"
            else:
                job.status, job.error = "failed", error
            try:
                self._persist(job)
            except OSError:
                logger.error("job_id=%s", job.job_id)

    def snapshot(self, job_id: str, *, ui: bool = False) -> dict | None:
        if not JOB_ID.fullmatch(job_id):
            return None
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            if job.finished_wall is not None and time.time() - job.finished_wall >= self.retention:
                self._remove(job.directory)
                self.jobs.pop(job_id, None)
                return None
            self._expire(job)
            data = {"job_id": job.job_id, "status": job.status}
            if job.status == "done":
                data.update(job.raw if ui else job.api)
            elif job.status == "failed":
                data["error"] = job.error
            return data

    @property
    def queue_length(self) -> int:
        with self.lock:
            return sum(j.status == "processing" and not j.active and not j.uploading for j in self.jobs.values())

    def _persist(self, job: Job) -> None:
        data = {"job_id": job.job_id, "status": job.status, "created_at": job.created_wall}
        if job.finished_wall is not None:
            data["finished_at"] = job.finished_wall
        if job.api is not None:
            data.update(job.api)
        if job.raw is not None:
            data["ui_result"] = job.raw
        if job.error:
            data["error"] = job.error
        temporary = job.directory / "state.tmp"
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        temporary.replace(job.directory / "state.json")

    def _expire(self, job: Job) -> bool:
        if job.status == "processing" and time.monotonic() - job.created >= self.timeout:
            if job.uploading:
                # The request's finally closes its open destination first (Windows).
                job.cancelled.set()
                return False
            self._finish(job, error=f"Превышен лимит {self.timeout:g} с с начала загрузки, включая ожидание в очереди. Повторите попытку.", cancel=True)
            return True
        return False

    def _work(self) -> None:
        while not self.stopping.is_set():
            job = self.queue.get()
            try:
                if job is None:
                    return
                with self.lock:
                    self._expire(job)
                    if job.status != "processing":
                        continue
                    job.active = True
                try:
                    workspace = job.path.parent
                    if isinstance(self.analyzer, ProcessAnalyzer):
                        result = self.analyzer.run(job.path, workspace, job.cancelled)
                    else:
                        # Injectable in-process doubles keep existing API/queue tests
                        # deterministic; production always uses ProcessAnalyzer.
                        result = analyze_video(self.analyzer, job.path, workspace)
                    api, ui = validate_result(result)
                    with self.lock:
                        self._expire(job)
                        if job.status == "processing":
                            self._finish(job, result=(api, ui))
                except BaseException as exc:
                    logger.error("job_id=%s", job.job_id)
                    with self.lock:
                        try:
                            self._expire(job)
                            self._finish(job, error=f"Не удалось обработать видео ({type(exc).__name__}). Проверьте файл и журнал сервера.", cancel=True)
                        except Exception:
                            logger.error("job_id=%s", job.job_id)
                finally:
                    with self.lock:
                        # A second cleanup path covers failure anywhere in the worker,
                        # including BaseException and result validation/persistence.
                        try:
                            self._remove_media(job)
                        except Exception:
                            logger.error("job_id=%s", job.job_id)
                        job.active = False
            finally:
                self.queue.task_done()

    def _monitor(self) -> None:
        next_cleanup = time.monotonic()
        while not self.stopping.wait(self.monitor_interval):
            try:
                with self.lock:
                    for job in self.jobs.values():
                        self._expire(job)
                        if job.status == "processing" and not self.worker.is_alive():
                            self._finish(job, error="Воркер обработки остановлен. Повторите попытку.", cancel=True)
                if time.monotonic() >= next_cleanup:
                    self.cleanup()
                    next_cleanup = time.monotonic() + 30
            except Exception:
                # No traceback/exception text: they can contain media paths.
                pass

    def cleanup(self) -> None:
        cutoff = time.time() - self.retention
        with self.lock:
            for directory in self.root.iterdir():
                # Never follow symlinks or remove anything outside ./jobs/<hex>.
                if not JOB_ID.fullmatch(directory.name) or directory.is_symlink() or not directory.is_dir():
                    continue
                job = self.jobs.get(directory.name)
                if job and (job.active or job.uploading):
                    continue
                created = (job.finished_wall or job.created_wall) if job else directory.stat().st_mtime
                if job is None:
                    try:
                        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
                        recorded = float(state.get("finished_at") or state["created_at"])
                        if math.isfinite(recorded):
                            created = recorded
                    except (OSError, ValueError, KeyError, TypeError):
                        # Partial uploads and files from older versions may lack
                        # metadata; use the directory timestamp in that case.
                        pass
                if created < cutoff:
                    self._remove(directory)
                    self.jobs.pop(directory.name, None)

