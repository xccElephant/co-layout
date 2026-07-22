"""
Imaginarium asset retrieval.

Builds a searchable index over the Imaginarium asset catalog (text embeddings
+ size), then retrieves the best-matching asset(s) for a piece of furniture
given a text description and an (optional) target size. Only supports
Imaginarium; co-layout only ever targets Imaginarium assets. Default embedding
model is the lightweight ``sentence-transformers/all-MiniLM-L6-v2`` (override
with ``--embedding-model``).

The official ``imaginarium_asset_info.csv`` (downloaded from HF) is missing
``short_desc``/``category``, and its ``bbx`` size column is wrong (wildly off
scale, or axis-swapped) for ~9% of assets. We patch both issues via a small
repo-bundled sidecar CSV (asset_library/imaginarium_assets.csv,
keyed by ``name_en``) carrying our own ``short_desc``/``category`` plus a
corrected, mesh-derived ``size``; entries missing from the sidecar fall back
to the CSV's own fields. A handful of known-issue FBX files (see
asset_library/issue_asset_blacklist.txt) are skipped entirely.
"""

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from asset_library.paths import (
    ASSET_INDEX_DEFAULT_PATH,
    IMAGINARIUM_ASSET_INFO_CSV,
    IMAGINARIUM_ASSETS_DIR,
    IMAGINARIUM_ISSUE_ASSET_BLACKLIST,
    IMAGINARIUM_SIDECAR_CSV,
)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---- size scoring weights ----
_SIZE_SCORE_CLOSE = 0.08
_SIZE_SCORE_MODERATE = 0.03
_SIZE_PENALTY_FAR = -0.02
_SIZE_PENALTY_VERY_FAR = -0.08


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _parse_size_field(value: Any) -> Optional[List[float]]:
    """Parse a size/bbx field that may be a JSON list or a comma-separated string.

    A dimension of exactly 0 is legitimate here (e.g. flat objects like rugs,
    carpets, or paper sheets are modeled as zero-thickness planes), so we only
    reject negative dimensions. Callers that need a strictly positive size
    (e.g. for Blender's ``object.dimensions``) should clamp with ``max(dim, eps)``.
    """
    text = _clean_text(value)
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) == 3:
            size = [float(x) for x in parsed]
            if all(dim >= 0 for dim in size):
                return size
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    try:
        size = [float(x) for x in text.split(",")]
        if len(size) == 3 and all(dim >= 0 for dim in size):
            return size
    except ValueError:
        pass
    return None


def _compute_size_score(
    asset_size: Optional[List[float]],
    target_size: Optional[List[float]],
    tolerance: float,
) -> float:
    """Soft size-similarity bonus/penalty, added on top of the semantic score."""
    if not asset_size or not target_size or len(asset_size) != 3 or len(target_size) != 3:
        return 0.0

    valid_diffs = []
    for asset_dim, target_dim in zip(asset_size, target_size):
        if target_dim <= 0 or asset_dim <= 0:
            continue
        valid_diffs.append(abs(np.log(float(asset_dim) / float(target_dim))))

    if not valid_diffs:
        return 0.0

    mean_diff = float(np.mean(valid_diffs))
    if mean_diff <= np.log(1 + tolerance):
        return _SIZE_SCORE_CLOSE
    if mean_diff <= np.log(1 + tolerance * 2):
        return _SIZE_SCORE_MODERATE
    if mean_diff <= np.log(1 + tolerance * 3):
        return _SIZE_PENALTY_FAR
    return _SIZE_PENALTY_VERY_FAR


def load_issue_asset_blacklist(path: Path = IMAGINARIUM_ISSUE_ASSET_BLACKLIST) -> set:
    """Load the known-issue jid blacklist (one id per line, '#' comments allowed)."""
    if not path.exists():
        return set()
    blacklist = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            blacklist.add(line)
    return blacklist


def load_sidecar_annotations(path: Path = IMAGINARIUM_SIDECAR_CSV) -> Dict[str, Dict[str, str]]:
    """Load the repo-bundled {name_en: {short_desc, category, size}} sidecar table.

    ``size`` is a corrected, mesh-derived bounding box (computed offline from
    each asset's point cloud) that overrides the official CSV's ``bbx``
    column, which is wrong for a meaningful fraction of assets.
    """
    if not path.exists():
        return {}
    annotations: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name_en = _clean_text(row.get("name_en"))
            if not name_en:
                continue
            annotations[name_en] = {
                "short_desc": _clean_text(row.get("short_desc")),
                "category": _clean_text(row.get("category")),
                "size": _clean_text(row.get("size")),
            }
    return annotations


class AssetIndex:
    """Stores asset metadata + embeddings for fast retrieval."""

    def __init__(self):
        self.assets: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.jid_list: List[str] = []
        self._embedding_buffer: List[np.ndarray] = []

    def add_asset(
        self,
        jid: str,
        short_desc: str,
        size: List[float],
        category: str = "",
        description: str = "",
        embedding: Optional[np.ndarray] = None,
    ):
        self.assets[jid] = {
            "jid": jid,
            "short_desc": short_desc,
            "size": size,
            "category": category,
            "description": description or short_desc,
        }
        if jid not in self.jid_list:
            self.jid_list.append(jid)
            if embedding is not None:
                self._embedding_buffer.append(embedding.reshape(1, -1))

    def finalize(self):
        if self._embedding_buffer:
            self.embeddings = np.vstack(self._embedding_buffer)
            self._embedding_buffer.clear()

    def get_asset(self, jid: str) -> Optional[Dict[str, Any]]:
        return self.assets.get(jid)

    def save(self, path: str):
        self.finalize()
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump({"assets": self.assets, "jid_list": self.jid_list}, f, ensure_ascii=False, indent=2)

        if self.embeddings is not None:
            np.save(save_path.with_suffix(".npy"), self.embeddings)

        print(f"[AssetIndex] Saved {len(self.assets)} assets to {save_path}")

    def load(self, path: str):
        load_path = Path(path)
        with open(load_path.with_suffix(".json"), "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assets = data["assets"]
            self.jid_list = data["jid_list"]

        embeddings_file = load_path.with_suffix(".npy")
        if embeddings_file.exists():
            self.embeddings = np.load(embeddings_file)

        print(f"[AssetIndex] Loaded {len(self.assets)} assets from {load_path}")

    def __len__(self) -> int:
        return len(self.assets)

    def __contains__(self, jid: str) -> bool:
        return jid in self.assets


class AssetRetriever:
    """Semantic + size-aware retrieval over an :class:`AssetIndex`."""

    def __init__(
        self,
        index_path: Optional[str] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = None,
    ):
        self.index = AssetIndex()
        self.embedding_model_name = embedding_model

        if device is None:
            try:
                import torch

                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device
        self._model = None

        if index_path:
            self.index.load(index_path)

    def _get_embedding_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            print(f"[AssetRetriever] Loading embedding model: {self.embedding_model_name}")
            self._model = SentenceTransformer(self.embedding_model_name, device=self.device)
        return self._model

    def encode_text(self, text: str) -> np.ndarray:
        return self._get_embedding_model().encode(text, convert_to_numpy=True)

    def retrieve(
        self,
        description: str,
        category: Optional[str] = None,
        size_constraint: Optional[List[float]] = None,
        size_tolerance: float = 0.5,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve matching assets.

        Args:
            description: free-text description of the furniture item (semantic query).
            category: optional room/category hint, prepended to the query text.
            size_constraint: optional target [width, depth, height] in meters.
            size_tolerance: relative size tolerance (0.5 = 50%).
            top_k: max number of results to return.
            min_score: results below this score are only used as a last-resort fallback.

        Returns:
            Matches sorted by descending score. Always returns at least 1 result
            unless the index is empty.
        """
        if len(self.index) == 0:
            print("[WARNING] Asset index is empty")
            return []
        if self.index.embeddings is None:
            raise ValueError("Asset index does not contain embeddings. Please rebuild the index.")

        query_text = f"{category} {description}".strip() if category else description
        query_embedding = self.encode_text(query_text)
        norms = np.linalg.norm(self.index.embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        cosine_scores = self.index.embeddings @ query_embedding / norms

        all_scored: List[Dict[str, Any]] = []
        for i, jid in enumerate(self.index.jid_list):
            asset = self.index.assets[jid]
            score = float(cosine_scores[i])
            score += _compute_size_score(asset.get("size"), size_constraint, size_tolerance)
            all_scored.append({**asset, "score": score})

        all_scored.sort(key=lambda x: x["score"], reverse=True)

        results = [r for r in all_scored if r["score"] >= min_score]
        if results:
            return results[:top_k]

        if all_scored:
            fallback = all_scored[0]
            print(
                f"[WARNING] No asset above min_score={min_score} for '{description}', "
                f"using best match: {fallback.get('short_desc')} (score: {fallback['score']:.3f})"
            )
            return [fallback]

        return []


def build_asset_index(
    asset_info_csv_path: Path = IMAGINARIUM_ASSET_INFO_CSV,
    asset_dir: Path = IMAGINARIUM_ASSETS_DIR,
    output_path: Path = ASSET_INDEX_DEFAULT_PATH,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    sidecar_csv_path: Path = IMAGINARIUM_SIDECAR_CSV,
    blacklist_path: Path = IMAGINARIUM_ISSUE_ASSET_BLACKLIST,
    require_assets_on_disk: bool = True,
) -> AssetIndex:
    """
    Build an :class:`AssetIndex` from the (HF-downloaded) Imaginarium asset info CSV.

    Args:
        asset_info_csv_path: path to ``imaginarium_asset_info.csv``.
        asset_dir: path to ``imaginarium_assets/`` (containing ``<jid>/<jid>.fbx``).
        output_path: where to save the built index (``.json`` + ``.npy``).
        embedding_model: sentence-transformers model name.
        sidecar_csv_path: repo-bundled ``{name_en: short_desc, category, size}`` overlay.
        blacklist_path: repo-bundled known-issue jid list, skipped entirely.
        require_assets_on_disk: if True (default), skip rows whose FBX file is not
            found under ``asset_dir``. Set to False to build an index purely from
            the CSV before downloading the (large) asset archive, e.g. for testing.
    """
    if not asset_info_csv_path.exists():
        raise FileNotFoundError(
            f"Asset info CSV not found: {asset_info_csv_path}\n"
            "Run `python -m asset_library.download_imaginarium` first."
        )

    blacklist = load_issue_asset_blacklist(blacklist_path)
    sidecar = load_sidecar_annotations(sidecar_csv_path)
    assets_dir_exists = asset_dir.exists()
    if require_assets_on_disk and not assets_dir_exists:
        print(
            f"[build_asset_index] WARNING: assets directory not found ({asset_dir}); "
            "building index from CSV only, without checking FBX availability."
        )

    index = AssetIndex()

    from sentence_transformers import SentenceTransformer

    print(f"[build_asset_index] Loading embedding model: {embedding_model}")
    model = SentenceTransformer(embedding_model)

    print(f"[build_asset_index] Reading asset info from {asset_info_csv_path}")
    n_total = 0
    n_blacklisted = 0
    n_missing_asset = 0
    n_missing_size = 0
    n_size_from_sidecar = 0

    with open(asset_info_csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            jid = _clean_text(row.get("name_en") or row.get("jid"))
            if not jid:
                continue
            if jid in blacklist:
                n_blacklisted += 1
                continue

            if require_assets_on_disk and assets_dir_exists:
                fbx_path = asset_dir / jid / f"{jid}.fbx"
                if not fbx_path.exists():
                    n_missing_asset += 1
                    continue

            overlay = sidecar.get(jid, {})

            size = _parse_size_field(overlay.get("size"))
            if size is not None:
                n_size_from_sidecar += 1
            else:
                size = _parse_size_field(row.get("bbx") or row.get("size"))
            if size is None:
                n_missing_size += 1
                continue
            size = [max(float(d), 0.01) for d in size]

            short_desc = overlay.get("short_desc") or _clean_text(row.get("short_desc"))
            category = (overlay.get("category") or _clean_text(row.get("category"))).lower().replace("_", " ")
            caption = _clean_text(row.get("caption_en"))
            class_en = _clean_text(row.get("class_en")).replace("_", " ")

            if not short_desc:
                short_desc = caption or class_en or jid
            if not category:
                category = class_en

            description = caption or short_desc

            embedding = model.encode(short_desc, convert_to_numpy=True)
            index.add_asset(
                jid=jid,
                short_desc=short_desc,
                size=size,
                category=category,
                description=description,
                embedding=embedding,
            )

    print(
        f"[build_asset_index] Indexed {len(index)}/{n_total} assets "
        f"(skipped: {n_blacklisted} blacklisted, {n_missing_asset} missing on disk, "
        f"{n_missing_size} missing size; {n_size_from_sidecar} sizes taken from the "
        f"corrected sidecar column rather than the CSV's bbx column)"
    )
    index.save(str(output_path))
    return index


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build the Imaginarium asset index.")
    parser.add_argument("--output", type=str, default=None, help="Output index path (no extension)")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"sentence-transformers model name (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--allow-missing-assets",
        action="store_true",
        help="Build the index from the CSV even if the FBX files have not been downloaded yet.",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else ASSET_INDEX_DEFAULT_PATH
    build_asset_index(
        output_path=output_path,
        embedding_model=args.embedding_model,
        require_assets_on_disk=not args.allow_missing_assets,
    )


if __name__ == "__main__":
    main()

# python -m asset_library.asset_retriever
