#!/usr/bin/env sh
set -eu
: "${API_KEY:?Set API_KEY before starting the service}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
# A single process owns the single loaded model and the in-memory job registry.
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
