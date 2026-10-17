"""Thin integration with the unmodified training/inference pipeline."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time
from types import SimpleNamespace
from typing import Protocol

# Set these BEFORE importing anything from Transformers/Hugging Face.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

from fc_pipeline import Video, extract_frames, ffprobe_duration, load_bundle, to_api


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

    bundle = Path(os.getenv("BUNDLE_DIR", "./bundle")).resolve()
    if not (bundle / "pipeline_config.json").is_file() or not (bundle / "adapter").is_dir():
        raise RuntimeError(f"BUNDLE_DIR должен содержать pipeline_config.json и adapter/: {bundle}")
    four_bit = optional_bool(os.getenv("LOAD_4BIT"))
    base = os.getenv("BASE_MODEL_PATH")
    if not base:
        # HF_HUB_OFFLINE permits an already-populated local Hub cache as well.
        return load_bundle(bundle, load_4bit=four_bit, merge=True)

    local_model = Path(base).resolve()
    if not local_model.is_dir():
        raise RuntimeError("BASE_MODEL_PATH должен указывать на локальную папку модели")
    meta = json.loads((bundle / "pipeline_config.json").read_text(encoding="utf-8"))
    meta["cfg"]["model_id"] = str(local_model)
    # No edits to the user's bundle or fc_pipeline.py. Copying works on Windows
    # without administrator privileges and also with read-only Docker mounts.
    with tempfile.TemporaryDirectory(prefix="forte-bundle-") as temporary:
        patched = Path(temporary)
        (patched / "pipeline_config.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
        shutil.copytree(bundle / "adapter", patched / "adapter")
        return load_bundle(patched, load_4bit=four_bit, merge=True)


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
        v = Video("job", [], path, ffprobe_duration(path), [])
        if not 0 < v.duration < float("inf"):
            raise ValueError("Видео не содержит корректной длительности")
        extract_frames(v, frame_dir, analyzer.cfg)
        if not v.files:
            raise ValueError("В видео не найдено кадров")
        raw = analyzer.analyze(v)
        return {"api": to_api(raw), "raw": raw}

