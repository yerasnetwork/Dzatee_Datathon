#!/bin/sh
set -eu

if [ -z "${API_KEY:-}" ]; then
    printf '%s\n' 'Ошибка: задайте API_KEY в окружении перед запуском.' >&2
    exit 1
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)
cd "$PROJECT_DIR"
PYTHON=${PYTHON:-python}
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    printf '%s\n' 'Ошибка: Python не найден (можно задать PYTHON=python3).' >&2
    exit 1
fi

CONSTRAINTS=$(mktemp "${TMPDIR:-/tmp}/forte-pip-constraints.XXXXXX")
trap 'rm -f "$CONSTRAINTS"' EXIT
trap 'exit 1' HUP INT TERM

# Read package metadata only: never import torch or initialize CUDA here.
"$PYTHON" - "$CONSTRAINTS" <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

try:
    installed = ["torch==" + version("torch")]
except PackageNotFoundError:
    raise SystemExit("Ошибка: используйте GPU-образ RunPod с уже установленным torch.")
try:
    installed.append("torchvision==" + version("torchvision"))
except PackageNotFoundError:
    pass
Path(sys.argv[1]).write_text("\n".join(installed) + "\n", encoding="utf-8")
print("Сохраняем установленные версии: " + ", ".join(installed))
PY

# No --upgrade / --force-reinstall. Conflicts fail instead of replacing Torch.
PIP_DISABLE_PIP_VERSION_CHECK=1 "$PYTHON" -m pip install --break-system-packages \
    -r "$PROJECT_DIR/requirements.txt" --constraint "$CONSTRAINTS"

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    apt-get update
    apt-get install -y ffmpeg
fi
for command_name in ffmpeg ffprobe pkill pgrep nohup; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        printf 'Ошибка: не найдена команда %s.\n' "$command_name" >&2
        exit 1
    fi
done

LOG_FILE="$PROJECT_DIR/service.log"
pkill -f 'uvicorn app.main:app' || true
# Let the old lifespan close its model process and clean its active workspace.
STOP_WAIT=0
while pgrep -f 'uvicorn app.main:app' >/dev/null 2>&1; do
    if [ "$STOP_WAIT" -ge 15 ]; then
        printf '%s\n' 'Ошибка: предыдущий uvicorn не завершился за 15 секунд.' >&2
        tail -n 30 "$LOG_FILE" 2>/dev/null || true
        exit 1
    fi
    sleep 1
    STOP_WAIT=$((STOP_WAIT + 1))
done

export API_KEY
export HF_HOME=/workspace/hf
export FORTE_BUNDLE_DIR="$PROJECT_DIR/bundle"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export PYTHONUNBUFFERED=1
unset MOCK_MODEL

nohup "$PYTHON" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 \
    > "$LOG_FILE" 2>&1 < /dev/null &
SERVICE_PID=$!
printf 'Запущен PID %s. Лог: %s\n' "$SERVICE_PID" "$LOG_FILE"

# One monotonic deadline, including HTTP request time, rather than 120 slow polls.
if "$PYTHON" - "$SERVICE_PID" <<'PY'
import json
import os
import sys
import time
import urllib.error
import urllib.request

pid = int(sys.argv[1])
deadline = time.monotonic() + 120
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        break
    try:
        timeout = max(0.01, min(2.0, deadline - time.monotonic()))
        with opener.open("http://127.0.0.1:8000/health", timeout=timeout) as response:
            health = json.load(response)
        if health.get("model_loaded") is True:
            print("Сервис готов: " + json.dumps(health, ensure_ascii=False), flush=True)
            sys.exit(0)
    except (OSError, ValueError, urllib.error.URLError):
        pass
    time.sleep(max(0.0, min(1.0, deadline - time.monotonic())))
print("Сервис не готов: процесс завершился или истекло ожидание 120 секунд.", file=sys.stderr)
sys.exit(1)
PY
then
    printf '%s\n' 'Адрес сервиса: http://127.0.0.1:8000'
else
    printf '%s\n' 'Последние 30 строк service.log:' >&2
    tail -n 30 "$LOG_FILE" >&2 || true
    exit 1
fi
