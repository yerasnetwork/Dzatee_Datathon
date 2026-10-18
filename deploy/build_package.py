"""Build a clean Linux deployment ZIP from an explicit allowlist."""

from __future__ import annotations

from pathlib import Path
import stat
import tempfile
import zipfile

ROOT_FILES = {"fc_pipeline.py", "fc_pipeline_mock.py", "run.sh", "README.md"}
ROOT_DIRS = {"app", "bundle", "deploy"}
EXCLUDED = {"__pycache__", ".venv", ".git", "jobs", ".verification", ".test-deps", "tests"}


def package_files(root: Path) -> list[Path]:
    candidates = [root / name for name in ROOT_FILES]
    candidates.extend(root.glob("requirements*.txt"))
    for name in ROOT_DIRS:
        candidates.extend((root / name).rglob("*"))
    files = []
    for path in candidates:
        relative = path.relative_to(root)
        if any(part in EXCLUDED or part.startswith(".") for part in relative.parts):
            continue
        if path.is_symlink() or not path.is_file() or path.suffix.lower() in {".pyc", ".pyo", ".zip"}:
            continue
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Архивируемый файл находится вне проекта")
        files.append(path)
    return sorted(set(files))


def build_package(root: Path) -> Path:
    destination = root / "forte_deploy.zip"
    files = package_files(root)
    if not ROOT_FILES.issubset({p.relative_to(root).as_posix() for p in files}):
        raise ValueError("Отсутствуют обязательные файлы сервиса")
    descriptor = tempfile.NamedTemporaryFile(prefix=".forte-deploy-", suffix=".tmp", dir=root, delete=False)
    temporary = Path(descriptor.name)
    descriptor.close()
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                name = path.relative_to(root).as_posix()
                data = path.read_bytes()
                executable = path.suffix == ".sh"
                if executable:
                    data = data.replace(b"\r\n", b"\n")
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | (0o755 if executable else 0o644)) << 16
                archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("Ошибка контрольной суммы ZIP")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


if __name__ == "__main__":
    result = build_package(Path(__file__).resolve().parents[1])
    print(f"{result.name}: {result.stat().st_size} bytes")
