"""FastAPI transport. API and UI share one model, queue and upload policy."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hmac
import os
from pathlib import Path
import time

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.concurrency import run_in_threadpool
from starlette.formparsers import MultiPartException, MultiPartParser

from app.jobs import JobManager
from app.model import load_analyzer
from app.worker import ProcessAnalyzer

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "app" / "static"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024


class UploadLimitExceeded(Exception):
    pass


class ClosingMultiPartParser(MultiPartParser):
    async def parse(self):
        try:
            return await super().parse()
        except BaseException:
            # Starlette closes spools on MultiPartException only. Also close on
            # receive limits, disconnects and cancellation, even before a job exists.
            for spool in self._files_to_close_on_error:
                spool.close()
            raise


@asynccontextmanager
async def upload_form(request: Request):
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "multipart/form-data":
        raise HTTPException(400, "Ожидается один видеофайл в multipart-поле file")
    parser = ClosingMultiPartParser(request.headers, request.stream(), max_files=1,
                                    max_fields=0, max_part_size=MAX_UPLOAD_BYTES)
    try:
        form = await parser.parse()
    except MultiPartException:
        raise HTTPException(400, "Некорректный multipart-запрос") from None
    try:
        yield form
    finally:
        await form.close()


class LocalRequestMiddleware:
    """Authenticate before consuming multipart data; bound even chunked bodies."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        path = scope["path"]
        if path.startswith("/api/v1"):
            expected = scope["app"].state.api_key
            supplied = headers.get(b"x-api-key", b"")
            if not hmac.compare_digest(supplied, expected):
                return await JSONResponse({"error": "Отсутствует или неверен X-API-Key"}, status_code=401)(scope, receive, send)
        scope.setdefault("state", {})["upload_started"] = time.monotonic()
        is_upload = scope["method"] == "POST" and path in {"/api/v1/analyze", "/ui/analyze"}
        total = 0

        async def limited_receive():
            nonlocal total
            message = await receive()
            if is_upload and message["type"] == "http.request":
                total += len(message.get("body", b""))
                # Allow bounded multipart headers in addition to the 200 MiB file.
                if total > MAX_UPLOAD_BYTES + 64 * 1024:
                    raise UploadLimitExceeded()
            return message

        async def secure_send(message):
            if message["type"] == "http.response.start":
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"content-security-policy", b"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"),
                    (b"cache-control", b"no-store"),
                ]
            await send(message)

        try:
            await self.app(scope, limited_receive, secure_send)
        except UploadLimitExceeded:
            await JSONResponse({"error": "Файл превышает лимит 200 МБ"}, status_code=413)(scope, receive, secure_send)


def gpu_name(mock: bool) -> str | None:
    if mock:
        return None
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def create_app(*, model_loader=None, jobs_dir: Path | None = None, job_timeout: float = 270.0,
               result_ttl: float | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        key = os.getenv("API_KEY", "")
        if not key.strip():
            raise RuntimeError("API_KEY обязателен, в том числе при MOCK_MODEL=1")
        app.state.api_key = key.encode("utf-8")
        app.state.mock = os.getenv("MOCK_MODEL") == "1"
        retention = result_ttl if result_ttl is not None else float(os.getenv("RESULT_TTL_SECONDS", "21600"))
        if not 0 < retention < float("inf"):
            raise RuntimeError("RESULT_TTL_SECONDS должен быть положительным числом")
        manager = JobManager(None, jobs_dir or ROOT / "jobs", timeout=job_timeout, retention=retention)
        app.state.jobs = manager
        # Wipe crash leftovers before loading weights or accepting any requests.
        await run_in_threadpool(manager.prepare_storage)
        if model_loader is None:
            analyzer = ProcessAnalyzer(manager.upload_root)
            await run_in_threadpool(analyzer.start)
            app.state.gpu = analyzer.gpu
        else:
            analyzer = await run_in_threadpool(model_loader)
            app.state.gpu = await run_in_threadpool(gpu_name, app.state.mock)
        manager.analyzer = analyzer
        try:
            await run_in_threadpool(manager.start)
            yield
        finally:
            await run_in_threadpool(manager.close)

    app = FastAPI(title="Forte · Контроль кассы", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(LocalRequestMiddleware)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse({"error": "Некорректный запрос"}, status_code=400)

    async def api_key(request: Request):
        # Middleware performs this before parsing; dependency documents the route
        # boundary and protects direct calls if middleware is changed later.
        supplied = request.headers.get("X-API-Key", "").encode("utf-8")
        if not hmac.compare_digest(supplied, request.app.state.api_key):
            raise HTTPException(401, "Отсутствует или неверен X-API-Key")

    async def upload(request: Request):
        manager: JobManager = request.app.state.jobs
        started = request.state.upload_started
        try:
            async with upload_form(request) as form:
                file = form.get("file")
                if not isinstance(file, UploadFile) or len(form.multi_items()) != 1:
                    raise HTTPException(400, "Ожидается один видеофайл в multipart-поле file")
                suffix = Path(file.filename or "").suffix.lower()
                if suffix not in {".mov", ".mp4"}:
                    raise HTTPException(400, "Допустимы только видео .mov и .mp4")
                if file.size is not None and file.size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Файл превышает лимит 200 МБ")
                job = await run_in_threadpool(manager.reserve, suffix, started)
                try:
                    size = 0
                    try:
                        with job.path.open("wb") as destination:
                            while chunk := await file.read(CHUNK_BYTES):
                                size += len(chunk)
                                if size > MAX_UPLOAD_BYTES:
                                    raise HTTPException(413, "Файл превышает лимит 200 МБ")
                                await run_in_threadpool(destination.write, chunk)
                    finally:
                        # The multipart spool is closed before a worker can finish.
                        await file.close()
                    if not size:
                        raise HTTPException(400, "Видео пустое")
                    await run_in_threadpool(manager.enqueue, job)
                except BaseException:
                    # Includes client disconnect/cancellation; don't leave uploads.
                    await asyncio.shield(run_in_threadpool(manager.discard_upload, job))
                    raise
                return {"job_id": job.job_id}
        except (HTTPException, UploadLimitExceeded):
            raise
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, "Некорректный multipart-запрос") from exc

    @app.post("/api/v1/analyze", dependencies=[Depends(api_key)])
    async def api_analyze(request: Request):
        return await upload(request)

    @app.post("/ui/analyze")
    async def ui_analyze(request: Request):
        return await upload(request)

    def get_job(request: Request, job_id: str, *, ui: bool):
        result = request.app.state.jobs.snapshot(job_id, ui=ui)
        if result is None:
            raise HTTPException(404, "Задание не найдено или срок хранения истёк")
        return result

    @app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(api_key)])
    def api_job(request: Request, job_id: str):
        return get_job(request, job_id, ui=False)

    @app.get("/ui/jobs/{job_id}")
    def ui_job(request: Request, job_id: str):
        return get_job(request, job_id, ui=True)

    @app.get("/health")
    def health(request: Request):
        manager = request.app.state.jobs
        loaded = not isinstance(manager.analyzer, ProcessAnalyzer) or manager.analyzer.loaded
        return {"model_loaded": manager.worker.is_alive() and loaded,
                "gpu_name": request.app.state.gpu, "queue_length": request.app.state.jobs.queue_length}

    @app.get("/ui/config")
    def ui_config(request: Request):
        return {"mock_model": request.app.state.mock}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
