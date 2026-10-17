"""Persistent inference process: cancellable without leaving open media handles."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
from pathlib import Path
import threading
import time

import psutil


class InferenceFailure(RuntimeError):
    pass


def _silence_output() -> None:
    # Native ffmpeg stderr and model prints can include paths or input data.
    # Redirect the OS descriptors too, so subprocesses inherit the same policy.
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)


def _serve(connection) -> None:
    _silence_output()
    if os.name == "posix":
        os.setsid()
    if connection.recv() != "start":
        return
    from app.model import analyze_video, load_analyzer

    try:
        analyzer = load_analyzer()
        gpu = None
        if os.getenv("MOCK_MODEL") != "1":
            import torch
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        connection.send(("ready", gpu))
    except BaseException as exc:
        connection.send(("error", type(exc).__name__))
        connection.close()
        return
    try:
        while True:
            command = connection.recv()
            if command is None:
                return
            path, workspace = command
            try:
                result = analyze_video(analyzer, Path(path), Path(workspace))
                connection.send(("result", result))
            except BaseException as exc:
                # Never send exception text, stderr, paths, prompts or tracebacks.
                connection.send(("error", type(exc).__name__))
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()


def kill_process_tree(process: psutil.Process) -> None:
    """Stop new child creation first, then close all child/native media handles."""
    try:
        process.suspend()
    except psutil.NoSuchProcess:
        return
    children = process.children(recursive=True)
    for child in reversed(children):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        process.kill()
    except psutil.NoSuchProcess:
        pass
    _, alive = psutil.wait_procs([*children, process], timeout=3)
    if any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in alive):
        raise InferenceFailure("Не удалось остановить процесс обработки")


def clear_orphan(marker: Path) -> None:
    """A killed HTTP process may have left its inference process behind."""
    if not marker.is_file() or marker.is_symlink():
        return
    try:
        saved = json.loads(marker.read_text(encoding="utf-8"))
        process = psutil.Process(saved["pid"])
        if abs(process.create_time() - saved["created"]) > 0.01:
            return  # PID has been reused; do not touch the unrelated process.
        try:
            parent = psutil.Process(saved["parent_pid"])
            if abs(parent.create_time() - saved["parent_created"]) < 0.01:
                raise RuntimeError("Папка jobs уже используется другим экземпляром сервиса")
        except psutil.NoSuchProcess:
            pass
        kill_process_tree(process)
    except (psutil.NoSuchProcess, ValueError, KeyError, TypeError):
        pass


class ProcessAnalyzer:
    """One loaded model, reused across jobs; reload only after forced termination."""

    def __init__(self, upload_root: Path) -> None:
        self.marker = upload_root.parent / ".worker.json"
        self.process = None
        self.connection = None
        self.gpu = None
        self.lock = threading.RLock()
        self.closed = False
        self.busy = False
        self.group = None

    @property
    def loaded(self) -> bool:
        process = self.process
        return process is not None and process.is_alive()

    def start(self) -> None:
        with self.lock:
            if self.closed:
                raise InferenceFailure("Обработка остановлена")
            parent, child = mp.get_context("spawn").Pipe()
            process = mp.get_context("spawn").Process(target=_serve, args=(child,), daemon=True)
            process.start()
            child.close()
            self.process, self.connection = process, parent
            self.busy = True
            try:
                self.group = ProcessGroup(process.pid)
            except BaseException:
                process.kill()
                process.join(timeout=3)
                parent.close()
                self.process, self.connection = None, None
                raise
            record = psutil.Process(process.pid)
            owner = psutil.Process()
            self.marker.write_text(json.dumps({"pid": process.pid, "created": record.create_time(),
                                              "parent_pid": owner.pid, "parent_created": owner.create_time()}),
                                   encoding="utf-8")
            parent.send("start")
        try:
            kind, payload = self._receive(process, parent, limit=600)
            if kind != "ready":
                raise InferenceFailure(f"Не удалось загрузить модель ({payload})")
            self.gpu = payload
            self.busy = False
        except BaseException:
            self.cancel()
            raise

    def _receive(self, process, connection, *, limit=None):
        started = time.monotonic()
        while True:
            try:
                if connection.poll(0.05):
                    return connection.recv()
                if not process.is_alive() or (limit is not None and time.monotonic() - started >= limit):
                    raise InferenceFailure("Процесс обработки остановлен")
            except (EOFError, BrokenPipeError, OSError) as exc:
                raise InferenceFailure("Процесс обработки остановлен") from exc

    def run(self, path: Path, workspace: Path, cancelled: threading.Event) -> dict:
        if cancelled.is_set():
            raise InferenceFailure("Задание отменено")
        if not self.loaded:
            self.start()
        with self.lock:
            if cancelled.is_set() or not self.loaded:
                raise InferenceFailure("Задание отменено")
            process, connection = self.process, self.connection
            self.busy = True
            connection.send((str(path), str(workspace)))
        try:
            kind, payload = self._receive(process, connection)
        finally:
            self.busy = False
        if kind != "result":
            raise InferenceFailure(f"Ошибка обработки ({payload})")
        return payload

    def cancel(self) -> None:
        with self.lock:
            process, connection = self.process, self.connection
            if process is not None:
                if self.group is not None:
                    self.group.terminate()
                    self.group = None
                try:
                    kill_process_tree(psutil.Process(process.pid))
                except psutil.NoSuchProcess:
                    pass
                process.join(timeout=3)
            if connection is not None:
                connection.close()
            self.process, self.connection = None, None
            self.busy = False
            self.marker.unlink(missing_ok=True)

    def close(self) -> None:
        with self.lock:
            self.closed = True
        self.cancel()


class ProcessGroup:
    """Windows Job Object / POSIX session also owns orphaned ffmpeg children."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.handle = None
        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes as w

        class Basic(ctypes.Structure):
            _fields_ = [("ProcessTime", ctypes.c_int64), ("JobTime", ctypes.c_int64),
                        ("Flags", w.DWORD), ("MinWork", ctypes.c_size_t), ("MaxWork", ctypes.c_size_t),
                        ("Active", w.DWORD), ("Affinity", ctypes.c_size_t),
                        ("Priority", w.DWORD), ("Scheduling", w.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

        class Extended(ctypes.Structure):
            _fields_ = [("Basic", Basic), ("IO", IO), ("ProcessMemory", ctypes.c_size_t),
                        ("JobMemory", ctypes.c_size_t), ("PeakProcess", ctypes.c_size_t),
                        ("PeakJob", ctypes.c_size_t)]

        class Accounting(ctypes.Structure):
            _fields_ = [(name, ctypes.c_int64) for name in ("User", "Kernel", "PeriodUser", "PeriodKernel")] + [
                (name, w.DWORD) for name in ("Faults", "Total", "Active", "Terminated")
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
            "SetInformationJobObject": ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
            "OpenProcess": ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
            "TerminateJobObject": ([w.HANDLE, w.UINT], w.BOOL),
            "QueryInformationJobObject": ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p], w.BOOL),
            "CloseHandle": ([w.HANDLE], w.BOOL),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(kernel, name)
            function.argtypes, function.restype = arguments, result
        self.kernel = kernel
        self.accounting_type = Accounting
        handle = kernel.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = Extended()
        info.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        process_handle = None
        try:
            if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            process_handle = kernel.OpenProcess(0x0101, False, pid)
            if not process_handle or not kernel.AssignProcessToJobObject(handle, process_handle):
                raise ctypes.WinError(ctypes.get_last_error())
            self.handle = handle
        finally:
            if process_handle:
                kernel.CloseHandle(process_handle)
            if self.handle is None:
                kernel.CloseHandle(handle)

    def terminate(self) -> None:
        if os.name == "nt":
            if self.handle is not None:
                self.kernel.TerminateJobObject(self.handle, 1)
                # Termination is asynchronous. Wait until *all* members have
                # released handles before deleting media, even if the leader died.
                import ctypes
                deadline = time.monotonic() + 3
                while True:
                    info = self.accounting_type()
                    if not self.kernel.QueryInformationJobObject(self.handle, 1, ctypes.byref(info), ctypes.sizeof(info), None):
                        raise ctypes.WinError(ctypes.get_last_error())
                    if info.Active == 0:
                        break
                    if time.monotonic() >= deadline:
                        raise InferenceFailure("Процессы обработки ещё не остановлены")
                    time.sleep(0.01)
                self.kernel.CloseHandle(self.handle)
                self.handle = None
        else:
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
