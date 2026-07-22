"""
Download the Imaginarium asset library from its official Hugging Face
dataset repo (``HiHiAllen/Imaginarium-Dataset``).

This script only *fetches* files from the upstream repo; co-layout does not
host or redistribute the 3D assets itself. If you already have the dataset
downloaded elsewhere, point ``DATASETS_ROOT`` at it instead of re-downloading
(see asset_library/paths.py).

Usage:
    python -m asset_library.download_imaginarium
    python -m asset_library.download_imaginarium --skip-assets   # CSV only, for a quick retrieval smoke test
    python -m asset_library.download_imaginarium --force         # re-download even if targets already exist
"""

import argparse
import shutil
import tarfile

from asset_library.paths import (
    HF_ASSET_INFO_FILENAME,
    HF_ASSETS_ARCHIVE_FILENAME,
    HF_REPO_ID,
    HF_REPO_TYPE,
    IMAGINARIUM_ASSET_INFO_CSV,
    IMAGINARIUM_ASSETS_DIR,
    IMAGINARIUM_DIR,
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


def download_asset_info_csv(force: bool = False) -> None:
    if IMAGINARIUM_ASSET_INFO_CSV.exists() and not force:
        print(f"[download] Asset info CSV already exists, skipping: {IMAGINARIUM_ASSET_INFO_CSV}")
        return

    IMAGINARIUM_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = _download_from_hf(HF_ASSET_INFO_FILENAME)
    shutil.copyfile(local_path, IMAGINARIUM_ASSET_INFO_CSV)
    print(f"[download] Copied asset info CSV to {IMAGINARIUM_ASSET_INFO_CSV}")


def download_and_extract_assets(force: bool = False) -> None:
    marker = IMAGINARIUM_ASSETS_DIR / ".extracted"
    if marker.exists() and not force:
        print(f"[download] Assets already extracted, skipping: {IMAGINARIUM_ASSETS_DIR}")
        return

    local_archive = _download_from_hf(HF_ASSETS_ARCHIVE_FILENAME)

    print(f"[download] Extracting {local_archive} -> {IMAGINARIUM_DIR} (this can take a while, ~tens of GB) ...")
    IMAGINARIUM_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(local_archive, "r:gz") as tar:
        tar.extractall(path=IMAGINARIUM_DIR)

    marker.touch()
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
