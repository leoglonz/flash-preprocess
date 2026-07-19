"""Per-user local paths, loaded from config.yaml at the repo root.

Copy config.yaml.example to config.yaml and edit it for your own
machine/account; config.yaml is gitignored so each user's copy stays local.
Falls back to config.yaml.example if config.yaml doesn't exist yet.
"""

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / 'config.yaml'
if not _CONFIG_PATH.exists():
    _CONFIG_PATH = _REPO_ROOT / 'config.yaml.example'

with open(_CONFIG_PATH) as f:
    _config = yaml.safe_load(f)

HYDROFABRIC_GPKG = Path(_config['hydrofabric_gpkg'])
