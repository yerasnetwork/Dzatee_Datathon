"""Pin the team's approved config; compare every other bundle file with v3."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile


APPROVED_CONFIG_SHA256 = "e4987d48696c050e4c6c6718cd6f7718b63b83665ee3eaa359e3209e437fc0d4"
UPDATED_RULE_REF_TYPES = ("unsafe_money", "stranger_coming", "client_handling")
CONFIG_PATH = "bundle/pipeline_config.json"


def verify_bundle(project: Path, archive: Path) -> dict:
    approved_bytes = (project / CONFIG_PATH).read_bytes()
    config_sha256 = hashlib.sha256(approved_bytes).hexdigest()
    if config_sha256 != APPROVED_CONFIG_SHA256:
        raise ValueError("SHA-256 pipeline_config.json не совпадает с приложенным конфигом команды")
    approved = json.loads(approved_bytes)
    with zipfile.ZipFile(archive) as outer:
        if "forte_pack_4b_v3.zip" in outer.namelist():
            data = outer.read("forte_pack_4b_v3.zip")
        else:
            data = archive.read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as source:
        prefix = "forte_pack_4b_v3/"
        bundle_files = {name[len(prefix):] for name in source.namelist()
                        if name.startswith(prefix + "bundle/") and not name.endswith("/")}
        actual_files = {path.relative_to(project).as_posix() for path in (project / "bundle").rglob("*")
                        if path.is_file()}
        if actual_files != bundle_files:
            raise ValueError("Состав bundle не совпадает с исходным архивом v3")
        identical = ["fc_pipeline.py", *sorted(bundle_files - {CONFIG_PATH})]
        for name in identical:
            if (project / name).read_bytes() != source.read(prefix + name):
                raise ValueError(f"Неожиданное изменение {name}")
        expected = json.loads(source.read(prefix + CONFIG_PATH))
        changes = {"cfg.max_windows": {"from": expected["cfg"]["max_windows"],
                                       "to": approved["cfg"]["max_windows"]}}
        expected["cfg"]["max_windows"] = approved["cfg"]["max_windows"]
        for key in UPDATED_RULE_REF_TYPES:
            changes[f"types.{key}.rule_ref"] = {"from": expected["types"][key]["rule_ref"],
                                               "to": approved["types"][key]["rule_ref"]}
            expected["types"][key]["rule_ref"] = approved["types"][key]["rule_ref"]
        if expected != approved:
            raise ValueError("Неожиданные отличия приложенного конфига от архива v3")
    return {"archive_files_identical": identical,
            "pipeline_config_sha256": config_sha256,
            "pipeline_config_changes": changes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(verify_bundle(root, args.archive or root / "forte.zip"), ensure_ascii=False, indent=2))
