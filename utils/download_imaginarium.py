"""
Download the Imaginarium asset library from its official Hugging Face
dataset repo (``HiHiAllen/Imaginarium-Dataset``).

This script only *fetches* files from the upstream repo; co-layout does not
host or redistribute the 3D assets itself. If you already have the dataset
downloaded elsewhere, point ``ASSET_LIBRARY_ROOT`` at it instead of
re-downloading (see utils/paths.py).

"Already downloaded" is detected by content -- asset subfolders already
present under ``imaginarium_assets/``, or the info CSV already existing --
rather than a script-written marker file, so it also correctly recognizes a
dataset placed there by other means (e.g. ``ASSET_LIBRARY_ROOT`` pointed at an
existing external copy) and skips by default instead of re-downloading tens
of GB on top of it. Pass ``--force`` to re-download/re-extract anyway.

Usage:
    python -m utils.download_imaginarium
    python -m utils.download_imaginarium --skip-assets   # CSV only, for a quick retrieval smoke test
    python -m utils.download_imaginarium --force         # re-download even if already present
"""

import argparse
import shutil
import tarfile

from utils.paths import (
    ASSET_LIBRARY_ROOT,
    HF_ASSET_INFO_FILENAME,
    HF_ASSETS_ARCHIVE_FILENAME,
    HF_REPO_ID,
    HF_REPO_TYPE,
    IMAGINARIUM_ASSET_INFO_CSV,
    IMAGINARIUM_ASSETS_DIR,
)


def _download_from_hf(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    print(f"[download] Fetching {filename} from {HF_REPO_ID} ...")
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        filename=filename,
    )
    print(f"[download] Cached at {local_path}")
    return local_path


def _assets_present(min_count: int = 1) -> bool:
    """Content-based "already downloaded" check: at least `min_count` asset
    subfolders already exist under IMAGINARIUM_ASSETS_DIR (each expected to
    hold a <jid>/<jid>.fbx). Deliberately not a script-written marker file, so
    it also recognizes a dataset placed there by other means (e.g.
    ASSET_LIBRARY_ROOT pointed at an existing local copy).
    """
    if not IMAGINARIUM_ASSETS_DIR.is_dir():
        return False
    count = 0
    for entry in IMAGINARIUM_ASSETS_DIR.iterdir():
        if entry.is_dir():
            count += 1
            if count >= min_count:
                return True
    return False


def download_asset_info_csv(force: bool = False) -> None:
    if IMAGINARIUM_ASSET_INFO_CSV.exists() and not force:
        print(f"[download] Asset info CSV already exists, skipping: {IMAGINARIUM_ASSET_INFO_CSV}")
        return

    IMAGINARIUM_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = _download_from_hf(HF_ASSET_INFO_FILENAME)
    shutil.copyfile(local_path, IMAGINARIUM_ASSET_INFO_CSV)
    print(f"[download] Copied asset info CSV to {IMAGINARIUM_ASSET_INFO_CSV}")


def download_and_extract_assets(force: bool = False) -> None:
    if _assets_present() and not force:
        print(
            f"[download] Found existing asset subfolders under {IMAGINARIUM_ASSETS_DIR}, skipping "
            "(pass --force to re-download/re-extract anyway)."
        )
        return

    local_archive = _download_from_hf(HF_ASSETS_ARCHIVE_FILENAME)

    print(f"[download] Extracting {local_archive} -> {ASSET_LIBRARY_ROOT} (this can take a while, ~tens of GB) ...")
    ASSET_LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(local_archive, "r:gz") as tar:
        tar.extractall(path=ASSET_LIBRARY_ROOT)

    print(f"[download] Done. Assets extracted to {IMAGINARIUM_ASSETS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Imaginarium asset library from the official HF dataset repo."
    )
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="Only download imaginarium_asset_info.csv, skip the (large) 3D asset archive. "
        "Useful for testing asset_retriever.py without downloading the full dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download / re-extract even if local targets already exist.",
    )
    args = parser.parse_args()

    download_asset_info_csv(force=args.force)
    if not args.skip_assets:
        download_and_extract_assets(force=args.force)
    else:
        print("[download] --skip-assets set, not downloading imaginarium_assets.tar.gz")


if __name__ == "__main__":
    main()
