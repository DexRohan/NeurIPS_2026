from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _data_root() -> Path:
    env_value = os.environ.get("QC_DATA_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()

    path_file = REPO_ROOT / "path_to_data.txt"
    if path_file.exists():
        raw = path_file.read_text(encoding="utf-8").strip()
        if raw and not raw.startswith("#"):
            return Path(raw).expanduser().resolve()

    return (REPO_ROOT / "data").resolve()


DATA_ROOT = _data_root()

cwd_data = str(DATA_ROOT)
cwd_scripts_preprocesing = str(REPO_ROOT / "scripts" / "preprocesing")
cwd_scripts_extraction_str = str(REPO_ROOT / "scripts" / "timestructure")
