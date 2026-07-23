"""
Path configuration for Imaginarium asset download + retrieval.

Locates the (user-downloaded) Imaginarium dataset and the small
repo-bundled sidecar files that augment it.
"""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = PACKAGE_DIR.parent


def _detect_asset_library_root() -> Path:
    """Root directory that holds (or will hold) the downloaded Imaginarium dataset.

    Override with the ``ASSET_LIBRARY_ROOT`` environment variable to point
    directly at an existing copy elsewhere on disk (a shared drive, another
    project's dataset dir, etc.) instead of downloading a second copy:
        export ASSET_LIBRARY_ROOT=/path/to/existing/Imaginarium
    Defaults to ``<project root>/asset_library`` when unset.
    """
    env_path = os.environ.get("ASSET_LIBRARY_ROOT")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return PROJECT_ROOT / "asset_library"


# ==================== External dataset (downloaded by the user) ====================
ASSET_LIBRARY_ROOT = _detect_asset_library_root()

IMAGINARIUM_ASSETS_DIR = ASSET_LIBRARY_ROOT / "imaginarium_assets"
IMAGINARIUM_ASSET_INFO_CSV = IMAGINARIUM_ASSETS_DIR / "imaginarium_asset_info.csv"

# ==================== Repo-bundled sidecar files (small, tracked in git) ====================
# These are *our own* derived annotations / known-issue lists, not the 3D assets
# themselves, so redistributing them does not raise the licensing concerns that
# redistributing the (mostly CC BY-NC-SA 4.0) 3D models would. They live under
# this package's own data/ dir (not ASSET_LIBRARY_ROOT) so they're always found
# regardless of where the (much larger, user-downloaded) dataset itself lives.
SIDECAR_DATA_DIR = PACKAGE_DIR / "data"
IMAGINARIUM_SIDECAR_CSV = SIDECAR_DATA_DIR / "imaginarium_assets.csv"
IMAGINARIUM_ISSUE_ASSET_BLACKLIST = SIDECAR_DATA_DIR / "issue_asset_blacklist.txt"

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
    ASSET_LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)


def print_paths() -> None:
    from_env = os.environ.get("ASSET_LIBRARY_ROOT") is not None
    source = "environment variable ASSET_LIBRARY_ROOT" if from_env else "default <project root>/asset_library"
    print("=" * 70)
    print(f"[Asset library root source] {source}")
    print("=" * 70)
    print(f"PROJECT_ROOT:                 {PROJECT_ROOT}")
    print(f"ASSET_LIBRARY_ROOT:           {ASSET_LIBRARY_ROOT}")
    print("-" * 70)
    print(f"IMAGINARIUM_ASSETS_DIR:        {IMAGINARIUM_ASSETS_DIR}")
    print(f"IMAGINARIUM_ASSET_INFO_CSV:    {IMAGINARIUM_ASSET_INFO_CSV}")
    print(f"IMAGINARIUM_SIDECAR_CSV:       {IMAGINARIUM_SIDECAR_CSV}")
    print(f"IMAGINARIUM_ISSUE_BLACKLIST:   {IMAGINARIUM_ISSUE_ASSET_BLACKLIST}")
    print("-" * 70)
    print(f"ASSET_INDEX_DEFAULT_PATH:      {ASSET_INDEX_DEFAULT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    print_paths()
