"""Thin integration with the unmodified training/inference pipeline."""

from __future__ import annotations

import base64
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol

# Set these BEFORE importing anything from Transformers/Hugging Face.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

if TYPE_CHECKING:
    from fc_pipeline import Video


def pipeline_module():
    """Choose lazily: pytest sets MOCK_MODEL after importing the app factory."""
    name = "fc_pipeline_mock" if os.getenv("MOCK_MODEL") == "1" else "fc_pipeline"
    return importlib.import_module(name)


def load_bundle(bundle_dir, load_4bit=None, merge=True):
    """Real loader remains lazy; importing the web service never loads weights."""
    from fc_pipeline import load_bundle as real_load_bundle
    return real_load_bundle(bundle_dir, load_4bit=load_4bit, merge=merge)


@contextmanager
def pipeline_temp_directory(directory: str):
    # The v3 pipeline creates additional TemporaryDirectory instances internally.
    # Its process is isolated and sequential, so they can share this existing
    # job workspace and remain covered by the unchanged supervisor cleanup.
    if os.getenv("MOCK_MODEL") == "1":
        yield
        return
    previous = tempfile.tempdir
    try:
        tempfile.tempdir = directory
        yield
    finally:
        tempfile.tempdir = previous


class VideoAnalyzer(Protocol):
    cfg: SimpleNamespace

    def analyze(self, v: Video) -> dict: ...


def optional_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    if value.lower() in {"1", "true", "yes"}:
        return True
    if value.lower() in {"0", "false", "no"}:
        return False
    raise ValueError("LOAD_4BIT должен быть 1/0 или true/false")


def load_analyzer() -> VideoAnalyzer:
    """Called once by the FastAPI lifespan, never per request."""
    for executable in ("ffmpeg", "ffprobe"):
        if not shutil.which(executable):
            raise RuntimeError(f"Не найден {executable}. Установите FFmpeg и добавьте его в PATH.")
    if os.getenv("MOCK_MODEL") == "1":
        return MockAnalyzer()

    bundle = Path(os.getenv("FORTE_BUNDLE_DIR") or os.getenv("BUNDLE_DIR") or "./bundle").resolve()
    if not all((bundle / name).is_file() for name in ("pipeline_config.json", "decision.json")):
        raise RuntimeError(f"FORTE_BUNDLE_DIR должен содержать pipeline_config.json и decision.json: {bundle}")
    four_bit = optional_bool(os.getenv("LOAD_4BIT"))
    base = os.getenv("BASE_MODEL_PATH")
    if base and not Path(base).is_dir():
        raise RuntimeError("BASE_MODEL_PATH должен указывать на локальную папку модели")
    # The supplied v3 loader reads BASE_MODEL_PATH itself. Its decision.json
    # branch needs neither a LoRA adapter nor peft. Hub IDs use the offline cache.
    return load_bundle(bundle, load_4bit=four_bit, merge=True)


class MockAnalyzer:
    """Synthetic findings, real uploaded-video JPEGs; no GPU/model imports."""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(fps=2.0, win_sec=8.0, long_side=576, time_budget_sec=200)

    def analyze(self, v: Video) -> dict:
        time.sleep(random.uniform(3.0, 5.0))
        last = max(0.0, v.duration - 0.15)
        times = [round(last * fraction, 3) for fraction in (0.2, 0.5, 0.8)]
        frames = []
        for timestamp in times:
            command = [
                "ffmpeg", "-nostdin", "-v", "error", "-ss", str(timestamp),
                "-i", v.path, "-map", "0:v:0", "-frames:v", "1",
                "-vf", "scale=640:-2", "-q:v", "3", "-f", "image2pipe",
                "-vcodec", "mjpeg", "pipe:1",
            ]
            jpeg = subprocess.run(command, capture_output=True, check=True, timeout=15).stdout
            if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
                raise ValueError("Не удалось получить JPEG-кадр из загруженного видео")
            frames.append({"timestamp_sec": timestamp, "image_base64": base64.b64encode(jpeg).decode("ascii")})
        findings = [
            {
                "type_key": "cash_outside_work_area",
                "type": "Наличные вне рабочей зоны",
                "timecode_sec": times[1],
                "score": 78,
                "description": "Демо-сценарий: возможное перемещение наличных за пределы контролируемой зоны. Проверьте момент вручную; mock не анализирует содержимое видео.",
                "description_en": "Demo finding: possible movement of cash outside the controlled area. Mock mode does not analyze video content.",
                "rule_ref": "Демо: команда укажет внутренний регламент и пункт правила.",
                "evidence_frames": frames,
            },
        ]
        if v.duration >= 8:
            findings.append({
                "type_key": "cash_handover_check", "type": "Проверка передачи наличных",
                "timecode_sec": times[2], "score": 61,
                "description": "Демо-сценарий: проверьте порядок пересчёта и передачи наличных. Это синтетическая отметка, а не вывод по изображению.",
                "description_en": "Synthetic demo: review the counting and cash handover procedure.",
                "rule_ref": "", "evidence_frames": frames,
            })
        return {"verdict": "violation", "violations": findings}


def analyze_video(analyzer: VideoAnalyzer, path: Path, job_dir: Path) -> dict:
    """The original model path, with all extracted frames confined to a temp dir."""
    with tempfile.TemporaryDirectory(prefix="frames-", dir=job_dir) as frame_dir:
        with pipeline_temp_directory(frame_dir):
            pipeline = pipeline_module()
            v = pipeline.Video("job", [], path, pipeline.ffprobe_duration(path), [])
            if not 0 < v.duration < float("inf"):
                raise ValueError("Видео не содержит корректной длительности")
            pipeline.extract_frames(v, frame_dir, analyzer.cfg)
            if not v.files:
                raise ValueError("В видео не найдено кадров")
            raw = analyzer.analyze(v)
            return {"api": pipeline.to_api(raw), "raw": raw}

