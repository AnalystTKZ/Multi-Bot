"""
vector_store.py — FAISS-GPU vector similarity search for trading patterns.

Three separate indices:
  - trade_patterns    : 74-dim  (GRU SEQUENCE_FEATURES per-bar snapshot)
  - market_structures : 53-dim  (REGIME_FEATURES — SMC + MTF + macro)
  - regime_embeddings : 64-dim  (GRU shared-layer encoding, richer than raw features)

Each index stores:
  - The raw float32 vector
  - A metadata dict (symbol, timeframe, timestamp, regime, outcome, etc.)

GPU acceleration:
  - Uses faiss.StandardGpuResources + faiss.index_cpu_to_gpu when CUDA available
  - Falls back to CPU IndexFlatIP (cosine via L2-normalised vectors) silently

Persistence:
  - Saved to weights/vector_store/{index_name}.faiss + {index_name}_meta.pkl
  - Auto-loads on instantiation if files exist

Usage:
    store = VectorStore()

    # Index a trade-pattern vector
    store.add("trade_patterns", vec_74d, {"symbol": "EURUSD", "ts": "2024-01-01", "outcome": "tp"})

    # Query nearest 5 similar patterns
    results = store.query("trade_patterns", query_vec_45d, k=5)
    # returns list of {"score": float, "meta": dict}

    # Save/load
    store.save()
    store.load()
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # trading-engine/
_STORE_DIR = os.path.join(_MODEL_ROOT, "weights", "vector_store")

# Index dimensionalities — must match the feature engines exactly.
# trade_patterns: SEQUENCE_FEATURES list in feature_engine.py (74 features as of current contract).
# market_structures: REGIME_FEATURES (53-dim HTF feature matrix).
# regime_embeddings: GRU hidden_size (64-dim shared-layer encoding).
INDEX_DIMS = {
    "trade_patterns":    74,   # SEQUENCE_FEATURES (per-bar snapshot, 74 features)
    "market_structures": 53,   # REGIME_FEATURES
    "regime_embeddings": 64,   # GRU shared layer output
}


def _get_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        raise ImportError(
            "faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)"
        )


def _build_index(dim: int) -> Any:
    """
    Build a flat inner-product index (cosine similarity after L2 normalisation).
    Uses GPU resource if CUDA is available, falls back to CPU.
    """
    faiss = _get_faiss()
    cpu_index = faiss.IndexFlatIP(dim)

    try:
        import torch
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            logger.info("VectorStore: GPU FAISS index built (dim=%d)", dim)
            return gpu_index
    except Exception as exc:
        logger.debug("VectorStore: GPU FAISS unavailable (%s), using CPU", exc)

    logger.info("VectorStore: CPU FAISS index built (dim=%d)", dim)
    return cpu_index


def _l2_normalise(vec: np.ndarray) -> np.ndarray:
    """Normalise a 1-D or 2-D float32 array to unit norm (row-wise)."""
    vec = np.atleast_2d(vec).astype(np.float32)
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return (vec / norms).astype(np.float32)


class VectorStore:
    """
    FAISS-backed vector store with three named indices.

    Each index maintains a parallel list of metadata dicts so that similarity
    results carry context (symbol, timestamp, regime, trade outcome, etc.).
    """

    def __init__(self, store_dir: Optional[str] = None):
        self._dir = store_dir or _STORE_DIR
        os.makedirs(self._dir, exist_ok=True)

        self._indices: Dict[str, Any] = {}
        self._meta:    Dict[str, List[dict]] = {}

        for name in INDEX_DIMS:
            self._indices[name] = _build_index(INDEX_DIMS[name])
            self._meta[name] = []

        # Auto-load if persisted data exists
        if any(
            os.path.exists(os.path.join(self._dir, f"{n}.faiss"))
            for n in INDEX_DIMS
        ):
            self.load()

    # ── Public API ──────────────────────────────────────────────────────────────

    def add(
        self,
        index_name: str,
        vector: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Add a single vector to the named index.

        vector    : 1-D float32 array of shape (dim,)
        metadata  : arbitrary dict stored alongside the vector (symbol, ts, etc.)
        """
        self._validate_index(index_name)
        dim = INDEX_DIMS[index_name]
        vec = np.asarray(vector, dtype=np.float32).flatten()
        if vec.shape[0] != dim:
            raise ValueError(
                f"VectorStore.add: '{index_name}' expects dim={dim}, got {vec.shape[0]}"
            )
        normed = _l2_normalise(vec)
        self._indices[index_name].add(normed)
        self._meta[index_name].append(metadata or {})

    def add_batch(
        self,
        index_name: str,
        vectors: np.ndarray,
        metadata_list: Optional[List[dict]] = None,
    ) -> None:
        """
        Add a batch of vectors.

        vectors       : (N, dim) float32 array
        metadata_list : list of N dicts; if None, empty dicts are stored
        """
        self._validate_index(index_name)
        dim = INDEX_DIMS[index_name]
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        if vecs.shape[1] != dim:
            raise ValueError(
                f"VectorStore.add_batch: '{index_name}' expects dim={dim}, got {vecs.shape[1]}"
            )
        normed = _l2_normalise(vecs)
        self._indices[index_name].add(normed)
        metas = metadata_list or [{} for _ in range(len(vecs))]
        self._meta[index_name].extend(metas)

    def query(
        self,
        index_name: str,
        vector: np.ndarray,
        k: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Find the k nearest neighbours for the query vector.

        Returns list of dicts: [{"score": float, "rank": int, "meta": dict}, ...]
        Sorted by descending cosine similarity.
        min_score: filter out results below this cosine similarity threshold.
        """
        self._validate_index(index_name)
        idx = self._indices[index_name]
        if idx.ntotal == 0:
            return []

        dim = INDEX_DIMS[index_name]
        vec = np.asarray(vector, dtype=np.float32).flatten()
        if vec.shape[0] != dim:
            raise ValueError(
                f"VectorStore.query: '{index_name}' expects dim={dim}, got {vec.shape[0]}"
            )
        normed = _l2_normalise(vec)
        k_capped = min(k, idx.ntotal)
        scores, indices = idx.search(normed, k_capped)

        results = []
        for rank, (score, i) in enumerate(zip(scores[0], indices[0])):
            if i < 0:
                continue
            s = float(score)
            if s < min_score:
                continue
            results.append({
                "score": s,
                "rank":  rank,
                "meta":  self._meta[index_name][i],
            })
        return results

    def query_multi(
        self,
        trade_vec:    Optional[np.ndarray] = None,
        structure_vec: Optional[np.ndarray] = None,
        embedding_vec: Optional[np.ndarray] = None,
        k: int = 5,
        min_score: float = 0.0,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convenience: query all three indices at once.

        Returns dict with keys: 'trade_patterns', 'market_structures', 'regime_embeddings'.
        Any index whose query vector is None is skipped (empty list returned).
        """
        results: Dict[str, List] = {n: [] for n in INDEX_DIMS}
        if trade_vec is not None:
            results["trade_patterns"]    = self.query("trade_patterns",    trade_vec,    k, min_score)
        if structure_vec is not None:
            results["market_structures"] = self.query("market_structures", structure_vec, k, min_score)
        if embedding_vec is not None:
            results["regime_embeddings"] = self.query("regime_embeddings", embedding_vec, k, min_score)
        return results

    def size(self, index_name: str) -> int:
        """Number of vectors currently in the index."""
        self._validate_index(index_name)
        return int(self._indices[index_name].ntotal)

    def sizes(self) -> Dict[str, int]:
        """Return sizes for all three indices."""
        return {n: self.size(n) for n in INDEX_DIMS}

    def clear(self, index_name: Optional[str] = None) -> None:
        """Reset one or all indices (removes all vectors and metadata)."""
        targets = [index_name] if index_name else list(INDEX_DIMS.keys())
        for name in targets:
            self._validate_index(name)
            self._indices[name] = _build_index(INDEX_DIMS[name])
            self._meta[name] = []
        logger.info("VectorStore: cleared %s", targets)

    # ── Persistence ─────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist all indices and metadata to disk."""
        faiss = _get_faiss()
        os.makedirs(self._dir, exist_ok=True)
        for name in INDEX_DIMS:
            idx = self._indices[name]
            # Move to CPU before saving (FAISS can only serialise CPU indices)
            try:
                cpu_idx = faiss.index_gpu_to_cpu(idx)
            except Exception:
                cpu_idx = idx
            faiss.write_index(cpu_idx, os.path.join(self._dir, f"{name}.faiss"))

            with open(os.path.join(self._dir, f"{name}_meta.pkl"), "wb") as f:
                pickle.dump(self._meta[name], f)

        total = sum(v for v in self.sizes().values())
        logger.info("VectorStore: saved %d total vectors to %s", total, self._dir)

    def load(self) -> None:
        """Load persisted indices from disk, then push to GPU if available."""
        faiss = _get_faiss()
        for name in INDEX_DIMS:
            faiss_path = os.path.join(self._dir, f"{name}.faiss")
            meta_path  = os.path.join(self._dir, f"{name}_meta.pkl")

            if not os.path.exists(faiss_path):
                continue

            try:
                cpu_idx = faiss.read_index(faiss_path)
                # Try to push to GPU
                try:
                    import torch
                    if torch.cuda.is_available():
                        res = faiss.StandardGpuResources()
                        self._indices[name] = faiss.index_cpu_to_gpu(res, 0, cpu_idx)
                        logger.info("VectorStore: loaded '%s' → GPU (%d vectors)", name, cpu_idx.ntotal)
                    else:
                        self._indices[name] = cpu_idx
                        logger.info("VectorStore: loaded '%s' → CPU (%d vectors)", name, cpu_idx.ntotal)
                except Exception:
                    self._indices[name] = cpu_idx

                if os.path.exists(meta_path):
                    with open(meta_path, "rb") as f:
                        self._meta[name] = pickle.load(f)
                else:
                    # Rebuild empty meta list if pkl missing
                    self._meta[name] = [{} for _ in range(cpu_idx.ntotal)]

            except Exception as exc:
                logger.warning("VectorStore: failed to load '%s': %s", name, exc)

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _validate_index(self, name: str) -> None:
        if name not in self._indices:
            raise KeyError(
                f"VectorStore: unknown index '{name}'. "
                f"Valid names: {list(INDEX_DIMS.keys())}"
            )

    def __repr__(self) -> str:
        sizes = self.sizes()
        return (
            f"VectorStore(trade_patterns={sizes['trade_patterns']}, "
            f"market_structures={sizes['market_structures']}, "
            f"regime_embeddings={sizes['regime_embeddings']}, "
            f"dir={self._dir!r})"
        )
