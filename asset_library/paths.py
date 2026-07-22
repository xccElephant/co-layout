"""
Path configuration for the Imaginarium asset library.

Locates the (user-downloaded) Imaginarium dataset and the small
repo-bundled sidecar files that augment it.
"""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = PACKAGE_DIR.parent


def _detect_datasets_root() -> Path:
    """Root directory that holds (or will hold) the downloaded Imaginarium dataset.

    Override with the ``DATASETS_ROOT`` environment variable to point at an
    external location, e.g. a shared drive or a symlink target:
        export DATASETS_ROOT=/workspace/datasets
    Defaults to ``<project root>/datasets`` when unset.
    """
    env_path = os.environ.get("DATASETS_ROOT")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return PROJECT_ROOT / "datasets"


# ==================== External dataset (downloaded by the user) ====================
DATASETS_ROOT = _detect_datasets_root()

IMAGINARIUM_DIR = DATASETS_ROOT / "Imaginarium"
IMAGINARIUM_ASSETS_DIR = IMAGINARIUM_DIR / "imaginarium_assets"
IMAGINARIUM_ASSET_INFO_CSV = IMAGINARIUM_ASSETS_DIR / "imaginarium_asset_info.csv"

# ==================== Repo-bundled sidecar files (small, tracked in git) ====================
# These are *our own* derived annotations / known-issue lists, not the 3D assets
# themselves, so redistributing them does not raise the licensing concerns that
# redistributing the (mostly CC BY-NC-SA 4.0) 3D models would.
IMAGINARIUM_SIDECAR_CSV = PACKAGE_DIR / "imaginarium_assets.csv"
IMAGINARIUM_ISSUE_ASSET_BLACKLIST = PACKAGE_DIR / "issue_asset_blacklist.txt"

# ==================== Official Hugging Face source ====================
HF_REPO_ID = "HiHiAllen/Imaginarium-Dataset"
HF_REPO_TYPE = "dataset"
HF_ASSET_INFO_FILENAME = "imaginarium_asset_info.csv"
HF_ASSETS_ARCHIVE_FILENAME = "imaginarium_assets.tar.gz"

# ==================== Project-internal output ====================
# Per-session visualization outputs live under output/sessions/<session_id>/visualization/
# (see constants.get_visualization_dir), not a single global directory.
OUTPUT_DIR = PROJECT_ROOT / "output"
ASSET_INDEX_DEFAULT_PATH = OUTPUT_DIR / "asset_index_imaginarium"


def ensure_dirs() -> None:
    IMAGINARIUM_DIR.mkdir(parents=True, exist_ok=True)


def print_paths() -> None:
    from_env = os.environ.get("DATASETS_ROOT") is not None
    source = "environment variable DATASETS_ROOT" if from_env else "default <project root>/datasets"
    print("=" * 70)
    print(f"[Datasets root source] {source}")
    print("=" * 70)
    print(f"PROJECT_ROOT:                 {PROJECT_ROOT}")
    print(f"DATASETS_ROOT:                {DATASETS_ROOT}")
    print("-" * 70)
    print(f"IMAGINARIUM_DIR:               {IMAGINARIUM_DIR}")
    print(f"IMAGINARIUM_ASSETS_DIR:        {IMAGINARIUM_ASSETS_DIR}")
    print(f"IMAGINARIUM_ASSET_INFO_CSV:    {IMAGINARIUM_ASSET_INFO_CSV}")
    print(f"IMAGINARIUM_SIDECAR_CSV:       {IMAGINARIUM_SIDECAR_CSV}")
    print(f"IMAGINARIUM_ISSUE_BLACKLIST:   {IMAGINARIUM_ISSUE_ASSET_BLACKLIST}")
    print("-" * 70)
    print(f"ASSET_INDEX_DEFAULT_PATH:      {ASSET_INDEX_DEFAULT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    print_paths()
