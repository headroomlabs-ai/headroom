"""HNSW vector index for Headroom Memory using hnswlib.

Provides fast approximate nearest neighbor search with cosine similarity
for semantic memory retrieval. Supports filtering by user_id, session_id,
agent_id, category, and entity references.

Note: hnswlib is an optional dependency. Install with:
    pip install hnswlib

Or via headroom extras:
    pip install "headroom-ai[memory]"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np

from ..models import Memory, ScopeLevel
from ..ports import VectorFilter, VectorSearchResult

# hnswlib is optional - may not compile on all platforms
# NOTE: We don't import hnswlib at module level because it can crash with SIGILL
# (Illegal Instruction) on CPUs without required AVX instructions. The crash
# happens at the C level before Python's try/except can catch it.
# Instead, we import lazily when HNSWVectorIndex is actually instantiated.
hnswlib: Any = None  # Will be imported lazily
HNSW_AVAILABLE: bool | None = None  # None = not yet checked, True/False = checked


def _check_hnswlib_available() -> bool:
    """Check if hnswlib is available, using subprocess to avoid SIGILL crash.

    Returns:
        True if hnswlib is available and working.

    Note:
        This function caches the result in HNSW_AVAILABLE.
        On CPUs without AVX support, importing hnswlib crashes with SIGILL
        (Illegal Instruction) at the C level before Python's try/except
        can catch it. We use subprocess to safely probe for hnswlib,
        isolating any potential crash.
    """
    global hnswlib, HNSW_AVAILABLE
    import logging

    logger = logging.getLogger(__name__)

    if HNSW_AVAILABLE is not None:
        return HNSW_AVAILABLE

    # Use subprocess to safely probe for hnswlib - if it crashes with SIGILL,
    # only the subprocess dies, not our main process
    import subprocess
    import sys

    try:
        # Probe by importing and creating a small Index to catch SIGILL
        # at both import time and during first use of AVX instructions
        probe_code = "import hnswlib; hnswlib.Index(space='cosine', dim=4)"
        result = subprocess.run(
            [sys.executable, "-c", probe_code],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Safe to import in main process now
            import hnswlib as _hnswlib

            hnswlib = _hnswlib
            HNSW_AVAILABLE = True
            logger.debug("hnswlib is available")
        else:
            HNSW_AVAILABLE = False
            stderr = result.stderr.decode() if result.stderr else "(no stderr)"
            logger.debug("hnswlib probe failed (exit %d): %s", result.returncode, stderr)
    except subprocess.TimeoutExpired:
        HNSW_AVAILABLE = False
        logger.debug("hnswlib probe timed out")
    except (FileNotFoundError, OSError) as e:
        HNSW_AVAILABLE = False
        logger.debug("hnswlib probe failed: %s", e)

    return HNSW_AVAILABLE


if TYPE_CHECKING:
    from ..tracker import ComponentStats


@dataclass
class IndexedMemoryMetadata:
    """Metadata stored alongside vectors for post-filtering.

    Stores all filterable fields from Memory to enable
    post-retrieval filtering without accessing the main store.
    """

    memory_id: str
    user_id: str
    session_id: str | None
    agent_id: str | None
    valid_until: datetime | None
    entity_refs: list[str]
    content: str  # For reconstructing Memory in search results
    created_at: datetime
    importance: float
    metadata: dict[str, Any] | None = None  # Custom metadata from Memory

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "entity_refs": self.entity_refs,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "importance": self.importance,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexedMemoryMetadata:
        """Deserialize from dictionary."""
        return cls(
            memory_id=data["memory_id"],
            user_id=data["user_id"],
            session_id=data.get("session_id"),
            agent_id=data.get("agent_id"),
            valid_until=(
                datetime.fromisoformat(data["valid_until"]) if data.get("valid_until") else None
            ),
            entity_refs=data.get("entity_refs", []),
            content=data["content"],
            created_at=datetime.fromisoformat(data["created_at"]),
            importance=data.get("importance", 0.5),
            metadata=data.get("metadata"),
        )

    @classmethod
    def from_memory(cls, memory: Memory) -> IndexedMemoryMetadata:
        """Create metadata from a Memory object."""
        return cls(
            memory_id=memory.id,
            user_id=memory.user_id,
            session_id=memory.session_id,
            agent_id=memory.agent_id,
            valid_until=memory.valid_until,
            entity_refs=memory.entity_refs.copy(),
            content=memory.content,
            created_at=memory.created_at,
            importance=memory.importance,
            metadata=memory.metadata.copy() if memory.metadata else None,
        )

    def to_memory(self, embedding: np.ndarray | None = None) -> Memory:
        """Reconstruct a basic Memory object from metadata.

        Note: This creates a partial Memory with only indexed fields.
        For full Memory objects, retrieve from the MemoryStore.
        """
        return Memory(
            id=self.memory_id,
            content=self.content,
            user_id=self.user_id,
            session_id=self.session_id,
            agent_id=self.agent_id,
            valid_until=self.valid_until,
            entity_refs=self.entity_refs.copy(),
            created_at=self.created_at,
            importance=self.importance,
            embedding=embedding,
            metadata=self.metadata.copy() if self.metadata else {},
        )


class HNSWVectorIndex:
    """HNSW-based vector index using hnswlib.

    Features:
    - Fast approximate nearest neighbor search with cosine similarity
    - Configurable HNSW parameters (ef_construction, M, ef_search)
    - Post-filtering by user_id, session_id, agent_id, category, entity_refs
    - Bidirectional ID mapping (string memory_id <-> integer hnsw_id)
    - Persistence support with save_index/load_index
    - Thread-safe operations with Lock
    - Optional auto-save on index modifications
    - **Bounded memory with LRU eviction** (when max_entries is set)

    Usage:
        index = HNSWVectorIndex(dimension=384, max_entries=10000)
        await index.index(memory_with_embedding)
        results = await index.search(VectorFilter(
            query_vector=query_embedding,
            top_k=10,
            user_id="alice"
        ))

    HNSW Parameters:
    - ef_construction: Size of dynamic candidate list during index construction.
      Higher values give better quality but slower construction. Default: 200
    - M: Number of bi-directional links per element. Higher values give
      better recall but use more memory. Default: 16
    - ef_search: Size of dynamic candidate list during search. Higher values
      give better recall but slower search. Default: 50

    Memory Bounding:
    - max_entries: Soft limit on number of entries. When reached, lowest
      importance entries are evicted to make room. Default: None (unbounded).
    - eviction_batch_size: Number of entries to evict at once when limit
      is reached. Default: 100.
    """

    def __init__(
        self,
        dimension: int = 384,
        max_elements: int = 100000,
        ef_construction: int = 200,
        m: int = 16,
        ef_search: int = 50,
        auto_save: bool = False,
        save_path: str | Path | None = None,
        max_entries: int | None = None,
        eviction_batch_size: int = 100,
    ) -> None:
        """Initialize the HNSW vector index.

        Args:
            dimension: Embedding dimension. Default 384 for MiniLM.
            max_elements: Maximum number of elements the index can hold.
                         Can be resized later with resize_index().
            ef_construction: HNSW construction parameter. Higher = better quality,
                           slower construction. Default: 200
            m: HNSW links per element. Higher = better recall, more memory.
               Default: 16
            ef_search: HNSW search parameter. Higher = better recall, slower
                      search. Default: 50
            auto_save: If True and save_path is set, automatically save
                      index after modifications.
            save_path: Path for auto-save operations. Required if auto_save=True.
            max_entries: Soft limit on number of entries. When reached,
                        lowest importance entries are evicted. None = unbounded.
            eviction_batch_size: Number of entries to evict when limit is reached.

        Raises:
            ValueError: If auto_save is True but save_path is not provided.
            ImportError: If hnswlib is not installed.
        """
        if not _check_hnswlib_available():
            raise ImportError(
                "hnswlib is required for HNSWVectorIndex. "
                "Install with: pip install hnswlib\n"
                "Note: hnswlib requires C++ compilation and may not be "
                "available on all platforms (crashes with SIGILL on CPUs "
                "without AVX support)."
            )

        if auto_save and save_path is None:
            raise ValueError("save_path must be provided when auto_save is True")

        self._dimension = dimension
        self._max_elements = max_elements
        self._ef_construction = ef_construction
        self._m = m
        self._ef_search = ef_search
        self._auto_save = auto_save
        self._save_path = Path(save_path) if save_path else None

        # Memory bounding
        self._max_entries = max_entries
        self._eviction_batch_size = eviction_batch_size
        self._eviction_count = 0  # Track total evictions for stats

        # Initialize HNSW index with cosine similarity
        # hnswlib uses 'cosine' space which internally normalizes vectors
        # Note: hnswlib is guaranteed non-None here due to _check_hnswlib_available() above
        self._index = hnswlib.Index(space="cosine", dim=dimension)  # type: ignore[union-attr]
        self._index.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=m,
        )
        self._index.set_ef(ef_search)

        # ID mappings: string memory_id <-> integer hnsw_id
        self._memory_to_hnsw: dict[str, int] = {}
        self._hnsw_to_memory: dict[int, str] = {}
        self._next_hnsw_id: int = 0

        # Metadata storage for filtering
        self._metadata: dict[str, IndexedMemoryMetadata] = {}

        # Embedding storage for retrieval
        self._embeddings: dict[str, np.ndarray] = {}

        # Thread safety
        self._lock = Lock()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension this index expects."""
        return self._dimension

    @property
    def size(self) -> int:
        """Return the number of vectors currently indexed."""
        with self._lock:
            return len(self._memory_to_hnsw)

    async def index(self, memory: Memory) -> None:
        """Index a memory's embedding for similarity search.

        The memory must have an embedding set. If max_entries is set and
        the limit is reached, low-importance entries are evicted.

        Args:
            memory: The memory to index.

        Raises:
            ValueError: If the memory has no embedding or wrong dimension.
        """
        if memory.embedding is None:
            raise ValueError(f"Memory {memory.id} has no embedding")

        embedding = np.asarray(memory.embedding, dtype=np.float32)
        if embedding.shape[0] != self._dimension:
            raise ValueError(
                f"Embedding dimension {embedding.shape[0]} does not match "
                f"index dimension {self._dimension}"
            )

        with self._lock:
            # Check if already indexed - update if so
            if memory.id in self._memory_to_hnsw:
                await self._update_embedding_internal(memory.id, embedding)
                # Update metadata
                self._metadata[memory.id] = IndexedMemoryMetadata.from_memory(memory)
            else:
                # Evict if at capacity (before adding new entry)
                if self._max_entries is not None:
                    current_size = len(self._memory_to_hnsw)
                    if current_size >= self._max_entries:
                        self._evict_entries(self._eviction_batch_size)

                # Resize HNSW index if needed (separate from entry limit)
                if self._next_hnsw_id >= self._max_elements:
                    self._resize_index(self._max_elements * 2)

                # Add to HNSW index
                hnsw_id = self._next_hnsw_id
                self._index.add_items(
                    embedding.reshape(1, -1),
                    np.array([hnsw_id]),
                )

                # Update mappings
                self._memory_to_hnsw[memory.id] = hnsw_id
                self._hnsw_to_memory[hnsw_id] = memory.id
                self._next_hnsw_id += 1

                # Store metadata and embedding
                self._metadata[memory.id] = IndexedMemoryMetadata.from_memory(memory)
                self._embeddings[memory.id] = embedding.copy()

        if self._auto_save and self._save_path:
            self.save_index(self._save_path)

    def _evict_entries(self, count: int) -> int:
        """Evict the lowest importance entries from the index.

        Must be called with lock held.

        Eviction strategy: Sort by importance (ascending), then by age
        (oldest first for ties). Evict the lowest scoring entries.

        Args:
            count: Number of entries to evict.

        Returns:
            Number of entries actually evicted.
        """
        if not self._metadata:
            return 0

        # Sort entries by importance (ascending), then by created_at (oldest first)
        sorted_entries = sorted(
            self._metadata.items(),
            key=lambda x: (x[1].importance, x[1].created_at),
        )

        # Evict the lowest importance entries
        evicted = 0
        for memory_id, _metadata in sorted_entries[:count]:
            if memory_id not in self._memory_to_hnsw:
                continue

            hnsw_id = self._memory_to_hnsw[memory_id]

            # Mark as deleted in HNSW index
            self._index.mark_deleted(hnsw_id)

            # Remove from mappings
            del self._memory_to_hnsw[memory_id]
            del self._hnsw_to_memory[hnsw_id]

            # Remove metadata and embedding
            if memory_id in self._metadata:
                del self._metadata[memory_id]
            if memory_id in self._embeddings:
                del self._embeddings[memory_id]

            evicted += 1

        self._eviction_count += evicted
        return evicted

    async def index_batch(self, memories: list[Memory]) -> int:
        """Index multiple memories' embeddings.

        Memories without embeddings are skipped.

        Args:
            memories: List of memories to index.

        Returns:
            Number of memories actually indexed.
        """
        # Filter memories with valid embeddings
        valid_memories: list[tuple[Memory, np.ndarray]] = []
        for memory in memories:
            if memory.embedding is not None:
                embedding = np.asarray(memory.embedding, dtype=np.float32)
                if embedding.shape[0] == self._dimension:
                    valid_memories.append((memory, embedding))

        if not valid_memories:
            return 0

        with self._lock:
            # Separate new memories from updates
            new_memories: list[tuple[Memory, np.ndarray, int]] = []
            update_memories: list[tuple[Memory, np.ndarray]] = []

            for memory, embedding in valid_memories:
                if memory.id in self._memory_to_hnsw:
                    update_memories.append((memory, embedding))
                else:
                    hnsw_id = self._next_hnsw_id
                    new_memories.append((memory, embedding, hnsw_id))
                    self._next_hnsw_id += 1

            # Resize if needed
            required_capacity = len(self._memory_to_hnsw) + len(new_memories)
            if required_capacity > self._max_elements:
                new_max = max(self._max_elements * 2, required_capacity + 1000)
                self._resize_index(new_max)

            # Batch add new memories
            if new_memories:
                embeddings_array = np.vstack([emb for _, emb, _ in new_memories]).astype(np.float32)
                ids_array = np.array([hid for _, _, hid in new_memories])

                self._index.add_items(embeddings_array, ids_array)

                # Update mappings and metadata
                for memory, embedding, hnsw_id in new_memories:
                    self._memory_to_hnsw[memory.id] = hnsw_id
                    self._hnsw_to_memory[hnsw_id] = memory.id
                    self._metadata[memory.id] = IndexedMemoryMetadata.from_memory(memory)
                    self._embeddings[memory.id] = embedding.copy()

            # Handle updates (hnswlib doesn't support true updates, so we track separately)
            for memory, embedding in update_memories:
                self._metadata[memory.id] = IndexedMemoryMetadata.from_memory(memory)
                self._embeddings[memory.id] = embedding.copy()
                # Note: HNSW embedding stays unchanged unless we remove and re-add

        if self._auto_save and self._save_path:
            self.save_index(self._save_path)

        return len(valid_memories)

    async def remove(self, memory_id: str) -> bool:
        """Remove a memory from the vector index.

        Note: hnswlib doesn't support true deletion. We mark the item as deleted
        and exclude it from results. The space is reclaimed on next save/load.

        Args:
            memory_id: The unique identifier of the memory.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if memory_id not in self._memory_to_hnsw:
                return False

            hnsw_id = self._memory_to_hnsw[memory_id]

            # Mark as deleted in HNSW index
            self._index.mark_deleted(hnsw_id)

            # Remove from our mappings
            del self._memory_to_hnsw[memory_id]
            del self._hnsw_to_memory[hnsw_id]

            # Remove metadata and embedding
            if memory_id in self._metadata:
                del self._metadata[memory_id]
            if memory_id in self._embeddings:
                del self._embeddings[memory_id]

        if self._auto_save and self._save_path:
            self.save_index(self._save_path)

        return True

    async def remove_batch(self, memory_ids: list[str]) -> int:
        """Remove multiple memories from the vector index.

        Args:
            memory_ids: List of memory IDs to remove.

        Returns:
            Number of memories actually removed.
        """
        removed_count = 0

        with self._lock:
            for memory_id in memory_ids:
                if memory_id not in self._memory_to_hnsw:
                    continue

                hnsw_id = self._memory_to_hnsw[memory_id]

                # Mark as deleted in HNSW index
                self._index.mark_deleted(hnsw_id)

                # Remove from our mappings
                del self._memory_to_hnsw[memory_id]
                del self._hnsw_to_memory[hnsw_id]

                # Remove metadata and embedding
                if memory_id in self._metadata:
                    del self._metadata[memory_id]
                if memory_id in self._embeddings:
                    del self._embeddings[memory_id]

                removed_count += 1

        if removed_count > 0 and self._auto_save and self._save_path:
            self.save_index(self._save_path)

        return removed_count

    async def search(self, filter: VectorFilter) -> list[VectorSearchResult]:
        """Search for similar memories using vector similarity.

        Args:
            filter: Vector search filter with query and constraints.

        Returns:
            List of search results sorted by similarity (descending).

        Raises:
            ValueError: If neither query_vector nor query_text is provided,
                       or if query_text is provided (embedding must be done externally).
        """
        if filter.query_vector is None:
            if filter.query_text is not None:
                raise ValueError(
                    "query_text provided but HNSWVectorIndex does not embed text. "
                    "Provide query_vector directly or use an Embedder first."
                )
            raise ValueError("Either query_vector or query_text must be provided")

        query_vector = np.asarray(filter.query_vector, dtype=np.float32)
        if query_vector.shape[0] != self._dimension:
            raise ValueError(
                f"Query vector dimension {query_vector.shape[0]} does not match "
                f"index dimension {self._dimension}"
            )

        with self._lock:
            # NOTE: Use len() directly, not self.size - Lock is not reentrant!
            current_size = len(self._memory_to_hnsw)
            if current_size == 0:
                return []

            # Search with more results than needed to account for filtering
            # Retrieve extra candidates to improve recall after filtering
            k_with_buffer = min(
                filter.top_k * 10,  # Get 10x candidates for filtering
                current_size,  # But not more than we have
            )

            # Query HNSW index
            # Returns (labels, distances) where labels are hnsw_ids
            labels, distances = self._index.knn_query(
                query_vector.reshape(1, -1),
                k=k_with_buffer,
            )

            # Convert distances to similarities
            # hnswlib with 'cosine' space returns 1 - cosine_similarity
            # So similarity = 1 - distance
            similarities = 1.0 - distances[0]

            # Build results with post-filtering
            results: list[VectorSearchResult] = []

            for hnsw_id, similarity in zip(labels[0], similarities):
                # Skip if not in our mapping (deleted)
                if hnsw_id not in self._hnsw_to_memory:
                    continue

                memory_id = self._hnsw_to_memory[hnsw_id]

                # Skip if below minimum similarity
                if similarity < filter.min_similarity:
                    continue

                # Get metadata for filtering
                metadata = self._metadata.get(memory_id)
                if metadata is None:
                    continue

                # Apply filters
                if not self._passes_filter(metadata, filter):
                    continue

                # Get stored embedding
                embedding = self._embeddings.get(memory_id)

                # Create Memory from metadata
                memory = metadata.to_memory(embedding=embedding)

                results.append(
                    VectorSearchResult(
                        memory=memory,
                        similarity=float(similarity),
                        rank=0,  # Will be set after sorting
                    )
                )

                # Stop if we have enough results
                if len(results) >= filter.top_k:
                    break

            # Sort by similarity (descending) and assign ranks
            results.sort(key=lambda r: r.similarity, reverse=True)
            for i, result in enumerate(results):
                result.rank = i + 1

            return results[: filter.top_k]

    def _passes_filter(
        self,
        metadata: IndexedMemoryMetadata,
        filter: VectorFilter,
    ) -> bool:
        """Check if metadata passes all filter constraints.

        Args:
            metadata: The indexed memory metadata.
            filter: The vector filter with constraints.

        Returns:
            True if all filter constraints pass, False otherwise.
        """
        # User ID filter
        if filter.user_id is not None and metadata.user_id != filter.user_id:
            return False

        # Session ID filter
        if filter.session_id is not None and metadata.session_id != filter.session_id:
            return False

        # Agent ID filter
        if filter.agent_id is not None and metadata.agent_id != filter.agent_id:
            return False

        # Scope level filter
        if filter.scope_levels is not None:
            # Determine the memory's scope level
            if metadata.agent_id is not None:
                memory_scope = ScopeLevel.AGENT
            elif metadata.session_id is not None:
                memory_scope = ScopeLevel.SESSION
            else:
                memory_scope = ScopeLevel.USER

            if memory_scope not in filter.scope_levels:
                return False

        # Temporal filter - valid_at
        if filter.valid_at is not None:
            # Memory must be valid at the specified time
            # If valid_until is None, memory is current (always valid after creation)
            # If valid_until is set, memory must have been valid at that time
            if metadata.valid_until is not None:
                if filter.valid_at > metadata.valid_until:
                    return False

        # Superseded filter
        if not filter.include_superseded:
            # Exclude superseded memories (those with valid_until set)
            if metadata.valid_until is not None:
                return False

        # Entity refs filter - any match
        if filter.entity_refs is not None and len(filter.entity_refs) > 0:
            if not any(ref in metadata.entity_refs for ref in filter.entity_refs):
                return False

        # Metadata filters (custom key-value pairs)
        # Note: We don't store custom metadata in IndexedMemoryMetadata
        # This would require extending the metadata storage

        return True

    async def update_embedding(self, memory_id: str, embedding: np.ndarray) -> bool:
        """Update the embedding for an indexed memory.

        Note: hnswlib doesn't support in-place updates. This stores the new
        embedding but the HNSW graph uses the original. For full update,
        remove and re-index the memory.

        Args:
            memory_id: The unique identifier of the memory.
            embedding: The new embedding vector.

        Returns:
            True if updated, False if memory not found in index.
        """
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.shape[0] != self._dimension:
            raise ValueError(
                f"Embedding dimension {embedding.shape[0]} does not match "
                f"index dimension {self._dimension}"
            )

        with self._lock:
            result = await self._update_embedding_internal(memory_id, embedding)

        if result and self._auto_save and self._save_path:
            self.save_index(self._save_path)

        return result

    async def _update_embedding_internal(self, memory_id: str, embedding: np.ndarray) -> bool:
        """Internal embedding update without lock (caller must hold lock).

        hnswlib doesn't support true in-place updates, so we:
        1. Store the new embedding in our local cache
        2. The HNSW index continues to use the old embedding for search

        For a true update, the caller should remove and re-index.
        """
        if memory_id not in self._memory_to_hnsw:
            return False

        self._embeddings[memory_id] = embedding.copy()
        return True

    def _resize_index(self, new_max_elements: int) -> None:
        """Resize the HNSW index to accommodate more elements.

        Must be called with lock held.

        Args:
            new_max_elements: New maximum capacity.
        """
        self._index.resize_index(new_max_elements)
        self._max_elements = new_max_elements

    def save_index(self, path: str | Path) -> None:
        """Save the index to disk.

        Saves both the HNSW index and all metadata/mappings.

        Args:
            path: Base path for the saved files. Will create:
                  - {path}.hnsw - The HNSW index
                  - {path}.meta - Metadata and mappings (pickled)
        """
        path = Path(path)

        with self._lock:
            # Save HNSW index
            hnsw_path = path.with_suffix(".hnsw")
            self._index.save_index(str(hnsw_path))

            # Save metadata, mappings, and embeddings
            meta_path = path.with_suffix(".meta")
            meta_data = {
                "dimension": self._dimension,
                "max_elements": self._max_elements,
                "ef_construction": self._ef_construction,
                "m": self._m,
                "ef_search": self._ef_search,
                "max_entries": self._max_entries,
                "eviction_batch_size": self._eviction_batch_size,
                "eviction_count": self._eviction_count,
                "memory_to_hnsw": self._memory_to_hnsw,
                "hnsw_to_memory": self._hnsw_to_memory,
                "next_hnsw_id": self._next_hnsw_id,
                "metadata": {mid: meta.to_dict() for mid, meta in self._metadata.items()},
                "embeddings": {mid: emb.tolist() for mid, emb in self._embeddings.items()},
            }

            with open(meta_path, "w") as f:
                json.dump(meta_data, f)

    def load_index(self, path: str | Path) -> None:
        """Load the index from disk.

        Loads both the HNSW index and all metadata/mappings.

        Args:
            path: Base path for the saved files.

        Raises:
            FileNotFoundError: If the index files don't exist.
            ValueError: If the saved dimension doesn't match.
        """
        path = Path(path)
        hnsw_path = path.with_suffix(".hnsw")
        meta_path = path.with_suffix(".meta")

        if not hnsw_path.exists():
            raise FileNotFoundError(f"HNSW index not found: {hnsw_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        # Load metadata first to get parameters
        with open(meta_path) as f:
            meta_data = json.load(f)

        # Verify dimension matches
        saved_dimension = meta_data["dimension"]
        if saved_dimension != self._dimension:
            raise ValueError(
                f"Saved index dimension {saved_dimension} does not match "
                f"current dimension {self._dimension}"
            )

        with self._lock:
            # Update parameters
            self._max_elements = meta_data["max_elements"]
            self._ef_construction = meta_data["ef_construction"]
            self._m = meta_data["m"]
            self._ef_search = meta_data["ef_search"]

            # Restore bounding parameters (with defaults for backward compatibility)
            self._max_entries = meta_data.get("max_entries")
            self._eviction_batch_size = meta_data.get("eviction_batch_size", 100)
            self._eviction_count = meta_data.get("eviction_count", 0)

            # Create new index and load from file
            self._index = hnswlib.Index(space="cosine", dim=self._dimension)  # type: ignore[union-attr]
            self._index.load_index(
                str(hnsw_path),
                max_elements=self._max_elements,
            )
            self._index.set_ef(self._ef_search)

            # Restore mappings
            # JSON converts int keys to strings, so we need to convert back
            self._memory_to_hnsw = meta_data["memory_to_hnsw"]
            self._hnsw_to_memory = {int(k): v for k, v in meta_data["hnsw_to_memory"].items()}
            self._next_hnsw_id = meta_data["next_hnsw_id"]

            # Restore metadata
            self._metadata = {
                mid: IndexedMemoryMetadata.from_dict(meta_dict)
                for mid, meta_dict in meta_data["metadata"].items()
            }

            # Restore embeddings
            self._embeddings = {
                mid: np.array(emb, dtype=np.float32) for mid, emb in meta_data["embeddings"].items()
            }

    def clear(self) -> None:
        """Clear all entries from the index."""
        with self._lock:
            # Reinitialize the index
            self._index = hnswlib.Index(space="cosine", dim=self._dimension)  # type: ignore[union-attr]
            self._index.init_index(
                max_elements=self._max_elements,
                ef_construction=self._ef_construction,
                M=self._m,
            )
            self._index.set_ef(self._ef_search)

            # Clear all mappings and metadata
            self._memory_to_hnsw.clear()
            self._hnsw_to_memory.clear()
            self._next_hnsw_id = 0
            self._metadata.clear()
            self._embeddings.clear()
            self._eviction_count = 0

        if self._auto_save and self._save_path:
            self.save_index(self._save_path)

    def stats(self) -> dict[str, Any]:
        """Get index statistics.

        Returns:
            Dictionary with index metrics.
        """
        with self._lock:
            current_size = len(self._memory_to_hnsw)
            return {
                "size": current_size,
                "dimension": self._dimension,
                "max_elements": self._max_elements,
                "max_entries": self._max_entries,
                "ef_construction": self._ef_construction,
                "m": self._m,
                "ef_search": self._ef_search,
                "eviction_count": self._eviction_count,
                "utilization": (
                    (current_size / self._max_elements) * 100 if self._max_elements > 0 else 0.0
                ),
                "entry_utilization": (
                    (current_size / self._max_entries) * 100 if self._max_entries else None
                ),
            }

    def get_memory_stats(self) -> ComponentStats:
        """Get memory statistics for the MemoryTracker.

        Returns:
            ComponentStats with current memory usage.
        """
        import sys

        from ..tracker import ComponentStats

        with self._lock:
            size_bytes = 0

            # ID mappings
            size_bytes += sys.getsizeof(self._memory_to_hnsw)
            for mem_id, hnsw_id in self._memory_to_hnsw.items():
                size_bytes += len(mem_id) + sys.getsizeof(hnsw_id)

            size_bytes += sys.getsizeof(self._hnsw_to_memory)
            for hnsw_id, mem_id in self._hnsw_to_memory.items():
                size_bytes += sys.getsizeof(hnsw_id) + len(mem_id)

            # Metadata storage
            size_bytes += sys.getsizeof(self._metadata)
            for mem_id, meta in self._metadata.items():
                size_bytes += len(mem_id)
                size_bytes += sys.getsizeof(meta)
                # Estimate metadata fields
                if meta.content:
                    size_bytes += len(meta.content)
                if meta.entity_refs:
                    size_bytes += sys.getsizeof(meta.entity_refs)
                if meta.metadata:
                    size_bytes += sys.getsizeof(meta.metadata)

            # Embeddings storage (numpy arrays)
            size_bytes += sys.getsizeof(self._embeddings)
            for mem_id, embedding in self._embeddings.items():
                size_bytes += len(mem_id)
                # numpy array size: dtype size * number of elements
                size_bytes += embedding.nbytes

            # HNSW index size estimate
            # The actual index is in hnswlib C++ memory, so we estimate:
            # Each element uses approximately: dimension * 4 bytes (float32) + M * 8 bytes (neighbors)
            index_size_estimate = len(self._memory_to_hnsw) * (self._dimension * 4 + self._m * 8)
            size_bytes += index_size_estimate

            # Calculate budget based on max_entries if set
            # Budget = estimated size at max capacity
            budget_bytes = None
            if self._max_entries is not None:
                # Estimate: each entry ~= embedding bytes + metadata overhead (~500 bytes)
                per_entry_estimate = self._dimension * 4 + self._m * 8 + 500
                budget_bytes = self._max_entries * per_entry_estimate

            return ComponentStats(
                name="vector_index",
                entry_count=len(self._memory_to_hnsw),
                size_bytes=size_bytes,
                budget_bytes=budget_bytes,
                hits=0,
                misses=0,
                evictions=self._eviction_count,
            )

    def set_ef_search(self, ef_search: int) -> None:
        """Update the ef_search parameter for query time.

        Higher values give better recall but slower search.

        Args:
            ef_search: New ef_search value.
        """
        with self._lock:
            self._ef_search = ef_search
            self._index.set_ef(ef_search)
