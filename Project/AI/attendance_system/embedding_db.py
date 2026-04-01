"""
embedding_db.py — FAISS-backed face embedding database.

WHY FAISS:
    With 30 students × 5 photos each = 150 embeddings, a brute-force linear
    scan would work fine. But FAISS also scales to thousands of students with
    no code changes, and adds a useful API for batch search.

    We use IndexFlatIP (Inner Product) because our ArcFace embeddings are
    L2-normalized. For unit vectors:
        cosine_similarity(a, b) = dot_product(a, b)
    So IndexFlatIP gives us exact cosine similarity without extra math.

DATABASE LAYOUT:
    data/embeddings/
        index.faiss    — the FAISS binary index (embedding vectors)
        metadata.json  — maps FAISS integer IDs → student info
                         {
                           "0": {"student_id": "S001", "name": "Alice", "photo": "..."},
                           "1": {"student_id": "S001", "name": "Alice", "photo": "..."},
                           ...
                         }
    Each student can have MULTIPLE embeddings (one per enrollment photo).
    The search returns the best match across ALL embeddings of ALL students.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# Type alias: maps FAISS vector index (int) → student metadata dict
MetadataStore = Dict[str, dict]


class EmbeddingDatabase:
    """
    Stores face embeddings in a FAISS index and retrieves nearest neighbours.

    Each enrolled student can have multiple embeddings (photos from different
    angles/lighting). A query embedding is compared against ALL stored vectors.
    We report the student whose best-matching embedding has the highest
    cosine similarity.
    """

    EMBEDDING_DIM = 512  # ArcFace output dimensionality — never changes

    def __init__(self, config: dict) -> None:
        paths = config["paths"]
        rec   = config["recognition"]

        self.index_path    = Path(paths["faiss_index"])
        self.metadata_path = Path(paths["metadata"])
        self.threshold     = rec["similarity_threshold"]
        self.top_k         = rec["top_k"]

        # Try GPU FAISS first; fall back to CPU gracefully
        self.use_gpu = config["device"]["use_gpu"]
        self.gpu_id  = config["device"]["gpu_id"]

        self._index:    Optional[faiss.Index] = None
        self._metadata: MetadataStore = {}

        # Load existing index if available
        if self.index_path.exists() and self.metadata_path.exists():
            self.load()
        else:
            logger.info("No existing database found. Starting fresh.")
            self._init_empty_index()

    # ── Index management ──────────────────────────────────────────────────────

    def _init_empty_index(self) -> None:
        """Create a new empty FAISS inner-product index."""
        cpu_index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        self._index    = self._to_gpu(cpu_index)
        self._metadata = {}
        logger.info("Created new FAISS IndexFlatIP(dim=%d)", self.EMBEDDING_DIM)

    def _to_gpu(self, cpu_index: faiss.Index) -> faiss.Index:
        """Move a CPU FAISS index to GPU if available and configured."""
        if not self.use_gpu:
            return cpu_index
        try:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, self.gpu_id, cpu_index)
            logger.info("FAISS index moved to GPU %d", self.gpu_id)
            return gpu_index
        except Exception as e:
            logger.warning("Could not move FAISS to GPU (%s). Using CPU.", e)
            return cpu_index

    def _to_cpu(self) -> faiss.Index:
        """Get a CPU copy of the index for saving to disk."""
        try:
            return faiss.index_gpu_to_cpu(self._index)
        except Exception:
            return self._index  # Already CPU

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        cpu_index = self._to_cpu()
        faiss.write_index(cpu_index, str(self.index_path))

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2, ensure_ascii=False)

        logger.info(
            "Database saved: %d vectors, %d students",
            cpu_index.ntotal,
            len(self._get_unique_students())
        )

    def load(self) -> None:
        """Load a previously saved FAISS index and metadata from disk."""
        cpu_index = faiss.read_index(str(self.index_path))
        self._index = self._to_gpu(cpu_index)

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        logger.info(
            "Database loaded: %d vectors, %d students",
            cpu_index.ntotal,
            len(self._get_unique_students())
        )

    # ── Enrollment ────────────────────────────────────────────────────────────

    def add_embedding(
        self,
        embedding:  np.ndarray,
        student_id: str,
        name:       str,
        photo_path: str = "",
    ) -> int:
        """
        Add one embedding to the database.

        Parameters
        ----------
        embedding   : 512-d L2-normalized float32 vector from ArcFace
        student_id  : unique student ID string (e.g. "S001")
        name        : display name
        photo_path  : source photo path (for traceability)

        Returns
        -------
        int : FAISS index of this new vector
        """
        assert embedding.shape == (self.EMBEDDING_DIM,), \
            f"Embedding must be {self.EMBEDDING_DIM}-d, got {embedding.shape}"

        # FAISS requires float32 row vectors, shape (1, dim)
        vec = embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)   # ensure unit norm (ArcFace should already do this)

        # The new vector's index in FAISS is the current total count
        new_id = self._to_cpu().ntotal
        self._index.add(vec)

        self._metadata[str(new_id)] = {
            "student_id": student_id,
            "name":       name,
            "photo":      photo_path,
        }

        logger.debug("Added embedding %d → student_id=%s name=%s", new_id, student_id, name)
        return new_id

    def add_student_embeddings(
        self,
        student_id:  str,
        name:        str,
        embeddings:  List[np.ndarray],
        photo_paths: Optional[List[str]] = None,
    ) -> None:
        """
        Add ALL embeddings for one student in one call.
        Preferred over multiple add_embedding() calls for clarity.
        """
        if photo_paths is None:
            photo_paths = [""] * len(embeddings)

        for emb, path in zip(embeddings, photo_paths):
            self.add_embedding(emb, student_id, name, path)

        logger.info(
            "Enrolled student: id=%s name=%s photos=%d",
            student_id, name, len(embeddings)
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
    ) -> Tuple[Optional[str], Optional[str], float]:
        """
        Find the closest enrolled student to a query embedding.

        Parameters
        ----------
        query_embedding : 512-d L2-normalized float32 from ArcFace

        Returns
        -------
        (student_id, name, similarity)
            student_id : matched ID, or None if below threshold
            name       : matched name, or None
            similarity : cosine similarity (0–1), or 0.0 if no match
        """
        n_total = self._to_cpu().ntotal
        if n_total == 0:
            logger.warning("Database is empty — enroll students first.")
            return None, None, 0.0

        vec = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)

        # Search for top-K nearest neighbours
        # D = distances (inner products = cosine sims for unit vectors)
        # I = indices in the FAISS index
        k = min(self.top_k, n_total)
        D, I = self._index.search(vec, k)

        best_sim = float(D[0][0])
        best_idx = int(I[0][0])

        if best_idx == -1 or best_sim < self.threshold:
            logger.debug(
                "No match: best_sim=%.3f < threshold=%.2f",
                best_sim, self.threshold
            )
            return None, None, best_sim

        meta = self._metadata.get(str(best_idx), {})
        student_id = meta.get("student_id")
        name       = meta.get("name", "Unknown")

        logger.debug(
            "Match: %s (%s) sim=%.3f", name, student_id, best_sim
        )
        return student_id, name, best_sim

    def search_batch(
        self,
        embeddings: np.ndarray,    # shape (N, 512)
    ) -> List[Tuple[Optional[str], Optional[str], float]]:
        """
        Search for N query embeddings at once — more efficient than N calls to search().

        Returns list of (student_id, name, similarity) tuples, one per query.
        """
        n_total = self._to_cpu().ntotal
        if n_total == 0:
            return [(None, None, 0.0)] * len(embeddings)

        vecs = embeddings.astype(np.float32)
        faiss.normalize_L2(vecs)

        k = min(self.top_k, n_total)
        D, I = self._index.search(vecs, k)

        results = []
        for sim_row, idx_row in zip(D, I):
            best_sim = float(sim_row[0])
            best_idx = int(idx_row[0])

            if best_idx == -1 or best_sim < self.threshold:
                results.append((None, None, best_sim))
            else:
                meta = self._metadata.get(str(best_idx), {})
                results.append((
                    meta.get("student_id"),
                    meta.get("name", "Unknown"),
                    best_sim,
                ))
        return results

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _get_unique_students(self) -> Dict[str, str]:
        """Return {student_id: name} for all enrolled students."""
        seen = {}
        for entry in self._metadata.values():
            sid = entry.get("student_id", "")
            if sid and sid not in seen:
                seen[sid] = entry.get("name", "")
        return seen

    @property
    def student_list(self) -> List[Dict[str, str]]:
        """Return list of {student_id, name} for all enrolled students."""
        unique = self._get_unique_students()
        return [{"student_id": sid, "name": name} for sid, name in sorted(unique.items())]

    @property
    def total_vectors(self) -> int:
        """Total number of embedding vectors stored."""
        return self._to_cpu().ntotal

    def remove_student(self, student_id: str) -> int:
        """
        Remove all embeddings for a given student ID.
        Rebuilds the entire index (FAISS doesn't support partial deletion cleanly).

        Returns number of vectors removed.
        """
        # Collect indices to keep
        keep_ids   = []
        keep_metas = {}
        new_id     = 0

        cpu_idx = self._to_cpu()
        all_vecs = faiss.rev_swig_ptr(cpu_idx.get_xb(), cpu_idx.ntotal * self.EMBEDDING_DIM)
        all_vecs = np.array(all_vecs).reshape(cpu_idx.ntotal, self.EMBEDDING_DIM)

        removed = 0
        for old_id_str, meta in self._metadata.items():
            if meta.get("student_id") == student_id:
                removed += 1
                continue
            keep_ids.append(int(old_id_str))
            keep_metas[str(new_id)] = meta
            new_id += 1

        if not keep_ids:
            self._init_empty_index()
            self._metadata = {}
            logger.info("Removed all %d embeddings for student %s", removed, student_id)
            return removed

        kept_vecs = all_vecs[keep_ids]
        new_cpu_idx = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        new_cpu_idx.add(kept_vecs)

        self._index    = self._to_gpu(new_cpu_idx)
        self._metadata = keep_metas

        logger.info("Removed %d embeddings for student %s", removed, student_id)
        return removed
