from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import zipfile

import pytest

from deploy.build_package import build_package
from deploy.verify_bundle import APPROVED_CONFIG_SHA256, UPDATED_RULE_REF_TYPES, verify_bundle

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_config_matches_approved_attachment():
    content = (ROOT / "bundle/pipeline_config.json").read_bytes()
    assert hashlib.sha256(content).hexdigest() == APPROVED_CONFIG_SHA256
    assert json.loads(content)["cfg"]["max_windows"] == 72


@pytest.fixture
def bundle_copy(tmp_path):
    archive = ROOT / "forte.zip"
    if not archive.exists():
        pytest.skip("Исходный пакет команды не включён в этот checkout")
    (tmp_path / "bundle").mkdir()
    for name in ("fc_pipeline.py", "bundle/decision.json", "bundle/pipeline_config.json"):
        shutil.copyfile(ROOT / name, tmp_path / name)
    return tmp_path, archive


def test_bundle_matches_archive_except_approved_configuration(bundle_copy):
    _, archive = bundle_copy
    result = verify_bundle(ROOT, archive)
    assert result["pipeline_config_sha256"] == APPROVED_CONFIG_SHA256
    assert result["archive_files_identical"] == ["fc_pipeline.py", "bundle/decision.json"]
    changes = result["pipeline_config_changes"]
    assert changes["cfg.max_windows"] == {"from": 40, "to": 72}
    assert set(changes) == {"cfg.max_windows", *(f"types.{key}.rule_ref" for key in UPDATED_RULE_REF_TYPES)}
    for key in UPDATED_RULE_REF_TYPES:
        assert changes[f"types.{key}.rule_ref"]["from"] != changes[f"types.{key}.rule_ref"]["to"]


@pytest.mark.parametrize("mutation", ["time_budget", *UPDATED_RULE_REF_TYPES, "newline", "max_windows_only"])
def test_bundle_verifier_rejects_any_unapproved_configuration(bundle_copy, mutation):
    project, archive = bundle_copy
    config = project / "bundle/pipeline_config.json"
    content = config.read_bytes()
    if mutation == "time_budget":
        changed = content.replace(b'"time_budget_sec": 190.0', b'"time_budget_sec": 191.0', 1)
    elif mutation in UPDATED_RULE_REF_TYPES:
        text = json.loads(content)["types"][mutation]["rule_ref"]
        original = json.dumps(text, ensure_ascii=False).encode("utf-8")
        changed = content.replace(original, json.dumps(text + " (изменено)", ensure_ascii=False).encode("utf-8"), 1)
    elif mutation == "newline":
        changed = content + b"\n"
    else:
        with zipfile.ZipFile(archive) as outer:
            data = outer.read("forte_pack_4b_v3.zip")
        with zipfile.ZipFile(io.BytesIO(data)) as source:
            changed = source.read("forte_pack_4b_v3/bundle/pipeline_config.json")
        changed = changed.replace(b'"max_windows": 40', b'"max_windows": 72', 1)
    assert changed != content
    config.write_bytes(changed)
    with pytest.raises(ValueError, match="SHA-256"):
        verify_bundle(project, archive)


@pytest.mark.parametrize("mutation", ["one_byte", "missing", "extra"])
def test_bundle_verifier_rejects_other_bundle_changes(bundle_copy, mutation):
    project, archive = bundle_copy
    decision = project / "bundle/decision.json"
    if mutation == "one_byte":
        decision.write_bytes(decision.read_bytes() + b"\n")
    elif mutation == "missing":
        decision.unlink()
    else:
        (project / "bundle/unapproved.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="decision.json|Состав bundle"):
        verify_bundle(project, archive)


@pytest.mark.parametrize("parent_mode", ["mock", "real-parent"])
def test_parent_loads_no_model_or_cuda_and_child_uses_spawn(tmp_path, parent_mode):
    # In the real-parent case only the spawned child's loader is switched to mock.
    # The parent goes through the same branch as production, with GPU imports blocked.
    source = '''
import importlib.abc, json, multiprocessing as mp, os, sys
from pathlib import Path
sys.path.insert(0, ROOT_PLACEHOLDER)

def child_mock_serve(connection):
    os.environ["MOCK_MODEL"] = "1"
    from app.worker import _serve
    _serve(connection)

if __name__ == "__main__":
    mode = sys.argv[2]
    if mode == "mock":
        os.environ["MOCK_MODEL"] = "1"
    else:
        os.environ.pop("MOCK_MODEL", None)
    os.environ["API_KEY"] = "parent-test-key"
    class BlockGPU(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"torch", "torchvision", "transformers", "peft", "bitsandbytes", "accelerate"}:
                raise AssertionError("CUDA/model import in parent: " + fullname)
    sys.meta_path.insert(0, BlockGPU())
    from fastapi.testclient import TestClient
    from app import model
    from app.main import create_app
    import app.worker as worker
    def forbid_parent_load():
        raise AssertionError("The parent must not load an analyzer")
    model.load_analyzer = forbid_parent_load
    if mode != "mock":
        worker._serve = child_mock_serve
    if "fork" in mp.get_all_start_methods():
        mp.set_start_method("fork", force=True)
    with TestClient(create_app(jobs_dir=Path(sys.argv[1]))) as client:
        process = client.app.state.jobs.analyzer.process
        assert process.pid != os.getpid()
        assert process._start_method == "spawn"
        assert client.get("/health").json()["model_loaded"] is True
        assert "torch" not in sys.modules
        assert "fc_pipeline" not in sys.modules
        print(json.dumps({"start_method": process._start_method, "parent_cuda": False, "parent_model": False}))
'''.replace("ROOT_PLACEHOLDER", repr(str(ROOT)))
    path = tmp_path / "spawn_check.py"
    path.write_text(source, encoding="utf-8")
    completed = subprocess.run([sys.executable, "-X", "utf8", str(path), str(tmp_path / "jobs"), parent_mode],
                               capture_output=True, text=True, encoding="utf-8", check=True, timeout=25)
    assert json.loads(completed.stdout) == {"start_method": "spawn", "parent_cuda": False, "parent_model": False}


def test_deployment_zip_has_only_allowed_files_and_linux_scripts(tmp_path):
    files = {
        "app/main.py", "app/static/index.html", "bundle/pipeline_config.json", "bundle/decision.json",
        "deploy/start_on_pod.sh", "fc_pipeline.py", "fc_pipeline_mock.py", "requirements.txt",
        "requirements-service.txt", "run.sh", "README.md",
    }
    unwanted = {".git/config", ".venv/python", "jobs/private.mov", "tests/test_private.py", "other.zip",
                "app/__pycache__/main.pyc", "app/nested.zip", "deploy/__pycache__/build.pyc",
                "app/.test-deps/private.py", "bundle/.verification/private.mov"}
    for name in files | unwanted:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"#!/bin/sh\r\nexit 0\r\n" if path.suffix == ".sh" else b"test")
    artifact = build_package(tmp_path)
    with zipfile.ZipFile(artifact) as archive:
        assert set(archive.namelist()) == files
        assert archive.testzip() is None
        for name in ("run.sh", "deploy/start_on_pod.sh"):
            assert b"\r\n" not in archive.read(name)
            assert archive.getinfo(name).external_attr >> 16 & stat.S_IXUSR
    assert build_package(tmp_path) == artifact  # Rebuilding replaces only the ZIP.


def shell_path():
    if os.name == "nt":
        git_shell = Path("C:/Program Files/Git/bin/bash.exe")
        if git_shell.exists():
            return str(git_shell)
    shell = shutil.which("sh")
    if shell:
        return shell
    pytest.skip("Для проверки sh нужен POSIX shell или Git Bash")


def test_start_script_shell_syntax():
    subprocess.run([shell_path(), "-n", str(ROOT / "deploy/start_on_pod.sh")],
                   capture_output=True, check=True, timeout=10)


@pytest.fixture
def pod_commands(tmp_path):
    project = tmp_path / "pod project"
    (project / "deploy").mkdir(parents=True)
    shutil.copyfile(ROOT / "deploy/start_on_pod.sh", project / "deploy/start_on_pod.sh")
    (project / "requirements.txt").write_text("transformers==4.57.6\ntorch\n", encoding="utf-8")
    (project / "bundle").mkdir()
    binary = tmp_path / "commands"
    binary.mkdir()
    temporary = tmp_path / "temp"
    temporary.mkdir()
    trace = tmp_path / "trace.txt"

    def command(name, source):
        path = binary / name
        path.write_text("#!/bin/sh\nset -eu\n" + source + "\n", encoding="utf-8", newline="\n")
        path.chmod(0o755)

    command("fake-python", '''
if [ "$1" = '-' ]; then
    case "$2" in
        *forte-pip-constraints.*)
            printf 'torch==2.8.0+cu128\\ntorchvision==0.23.0+cu128\\n' > "$2"
            exit 0 ;;
    esac
    count=0
    while [ ! -s "$PWD/service.log" ] && [ "$count" -lt 100 ]; do
        sleep 0.01
        count=$((count + 1))
    done
    if [ "${READY_RESULT:-ok}" = fail ]; then exit 1; fi
    printf '%s\\n' '{"model_loaded":true,"gpu_name":"mock child","queue_length":0}'
    exit 0
fi
printf 'python %s\\n' "$*" >> "$TRACE"
if [ "$1" = '-m' ] && [ "$2" = pip ]; then
    previous=''
    for argument in "$@"; do
        if [ "$previous" = '--constraint' ]; then cat "$argument" >> "$TRACE"; fi
        previous=$argument
    done
fi
if [ "$1" = '-m' ] && [ "$2" = uvicorn ]; then
    printf 'HF_HOME=%s FORTE_BUNDLE_DIR=%s HF_HUB_OFFLINE=%s TRANSFORMERS_OFFLINE=%s MOCK_MODEL=%s\\n' \\
        "$HF_HOME" "$FORTE_BUNDLE_DIR" "$HF_HUB_OFFLINE" "$TRANSFORMERS_OFFLINE" "${MOCK_MODEL-unset}" >> "$TRACE"
    printf '%s\\n' 'FAKE SERVICE LOG'
fi
''')
    command("pkill", 'printf "pkill %s\\n" "$*" >> "$TRACE"\nexit 1')
    command("pgrep", "exit 1")
    command("nohup", '''
printf 'nohup %s\\n' "$*" >> "$TRACE"
printf 'HF_HOME=%s FORTE_BUNDLE_DIR=%s HF_HUB_OFFLINE=%s TRANSFORMERS_OFFLINE=%s MOCK_MODEL=%s\\n' \\
    "$HF_HOME" "$FORTE_BUNDLE_DIR" "$HF_HUB_OFFLINE" "$TRANSFORMERS_OFFLINE" "${MOCK_MODEL-unset}" >> "$TRACE"
exec "$@"
''')
    command("apt-get", '''
printf 'apt-get %s\\n' "$*" >> "$TRACE"
if [ "$1" = install ]; then
    printf '#!/bin/sh\\nexit 0\\n' > "$STUB_BIN/ffmpeg"
    printf '#!/bin/sh\\nexit 0\\n' > "$STUB_BIN/ffprobe"
    chmod +x "$STUB_BIN/ffmpeg" "$STUB_BIN/ffprobe"
fi
''')
    base_paths = [str(binary)]
    if os.name == "nt":
        base_paths += ["C:/Program Files/Git/usr/bin", "C:/Program Files/Git/bin", "C:/Windows/System32"]
    else:
        # Do not expose the host's real ffmpeg: the missing-package test must
        # exercise apt-get through the stub on Linux as well as Windows.
        for utility in ("dirname", "mktemp", "rm", "sleep", "tail", "cat", "chmod"):
            target = shutil.which(utility)
            assert target, utility
            (binary / utility).symlink_to(target)
    environment = dict(os.environ, API_KEY="private-pod-key", PYTHON=str(binary / "fake-python"),
                       PATH=os.pathsep.join(base_paths), TRACE=str(trace), STUB_BIN=str(binary),
                       TMPDIR=str(temporary), MOCK_MODEL="1")

    def run(**overrides):
        env = {**environment, **overrides}
        return subprocess.run([shell_path(), str(project / "deploy/start_on_pod.sh")],
                              cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=15)

    return run, command, trace, temporary


def test_start_script_can_be_repeated_without_reinstalling_torch(pod_commands):
    run, command, trace, temporary = pod_commands
    command("ffmpeg", "exit 0")
    command("ffprobe", "exit 0")
    for _ in range(2):
        result = run()
        assert result.returncode == 0, result.stdout + result.stderr
        assert "model_loaded" in result.stdout
    calls = trace.read_text(encoding="utf-8")
    assert calls.count("pip install --break-system-packages") == 2
    assert calls.count("pkill -f uvicorn app.main:app") == 2
    assert "torch==2.8.0+cu128" in calls
    assert "--upgrade" not in calls and "--force-reinstall" not in calls
    assert "--host 0.0.0.0 --port 8000 --workers 1" in calls
    assert "HF_HOME=/workspace/hf" in calls
    assert "pod project/bundle" in calls
    assert "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MOCK_MODEL=unset" in calls
    assert "private-pod-key" not in calls
    assert not list(temporary.iterdir())


def test_start_script_requires_key_before_any_changes(pod_commands):
    run, _, trace, _ = pod_commands
    result = run(API_KEY="")
    assert result.returncode != 0
    assert "API_KEY" in result.stderr
    assert not trace.exists()


def test_start_script_installs_missing_ffmpeg(pod_commands):
    run, _, trace, _ = pod_commands
    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "apt-get install -y ffmpeg" in trace.read_text(encoding="utf-8")


def test_start_script_reports_log_on_readiness_failure(pod_commands):
    run, command, _, _ = pod_commands
    command("ffmpeg", "exit 0")
    command("ffprobe", "exit 0")
    result = run(READY_RESULT="fail")
    assert result.returncode != 0
    assert "Последние 30 строк" in result.stderr
    assert "FAKE SERVICE LOG" in result.stderr
