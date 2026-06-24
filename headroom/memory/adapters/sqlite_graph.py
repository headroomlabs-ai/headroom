"""SQLite graph store for Headroom's knowledge graph memory system.

Provides persistent storage for entities and relationships with efficient
lookup via database indexes and BFS-based traversal. Memory usage is bounded
by SQLite's page cache.

This is a drop-in replacement for InMemoryGraphStore that:
- Persists all data to disk
- Keeps memory bounded (configurable page cache)
- Maintains the same async interface
- Supports all query patterns (by ID, by name, BFS traversal)
"""

from __future__ import annotations

import json
import sqlite3
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any

from .graph_models import Entity, Relationship, RelationshipDirection, Subgraph

if TYPE_CHECKING:
    from ..tracker import ComponentStats


class SQLiteGraphStore:
    """SQLite-based graph store implementing the GraphStore protocol.

    Provides persistent storage for entities and relationships with efficient
    lookup via database indexes. All operations are thread-safe.

    Features:
    - O(log n) entity and relationship lookup by ID (indexed)
    - O(log n) entity lookup by name (per user, case-insensitive)
    - O(log n) lookup of relationships by source or target entity
    - BFS-based subgraph traversal with configurable hop limit
    - BFS-based shortest path finding between entities
    - Configurable page cache for memory bounding

    Schema:
        - entities: id, user_id, name, name_lower, entity_type, description,
                   properties, created_at, updated_at, metadata
        - relationships: id, user_id, source_id, target_id, relation_type,
                        weight, properties, created_at, metadata

    Usage:
        store = SQLiteGraphStore("./graph.db")
        await store.add_entity(Entity(user_id="alice", name="Project X", entity_type="project"))
        entity = await store.get_entity_by_name("alice", "project x")  # Case-insensitive
        subgraph = await store.query_subgraph(["entity-id"], max_hops=2)
    """

    def __init__(
        self,
        db_path: str | Path = "headroom_graph.db",
        page_cache_size_kb: int = 8192,  # 8MB default cache
    ) -> None:
        """Initialize the SQLite graph store.

        Args:
            db_path: Path to SQLite database file. Created if it doesn't exist.
            page_cache_size_kb: SQLite page cache size in KB. Higher = more memory,
                              faster queries. Set to -1 for default SQLite behavior.
        """
        self.db_path = Path(db_path)
        self._page_cache_size_kb = page_cache_size_kb
        self._lock = RLock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a new database connection (thread-safe pattern).

        Returns:
            A new SQLite connection with row factory configured.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        # Configure page cache size (negative = KB, positive = pages)
        if self._page_cache_size_kb > 0:
            conn.execute(f"PRAGMA cache_size = -{self._page_cache_size_kb}")

        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")

        return conn

    def _init_db(self) -> None:
        """Initialize the database schema with indexes."""
        with self._get_conn() as conn:
            # Create entities table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    name_lower TEXT NOT NULL,
                    entity_type TEXT NOT NULL DEFAULT 'unknown',
                    description TEXT,
                    properties TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)

            # Create relationships table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL DEFAULT 'related_to',
                    weight REAL NOT NULL DEFAULT 1.0,
                    properties TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (source_id) REFERENCES entities(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES entities(id) ON DELETE CASCADE
                )
            """)

            # Create indexes for efficient querying
            # Entity indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_user_id ON entities(user_id)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_name_lookup "
                "ON entities(user_id, name_lower)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)")

            # Relationship indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relation_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relationships_user ON relationships(user_id)"
            )

            conn.commit()

    def _entity_to_row(self, entity: Entity) -> dict[str, Any]:
        """Convert Entity object to row dict for insertion."""
        return {
            "id": entity.id,
            "user_id": entity.user_id,
            "name": entity.name,
            "name_lower": entity.name.lower(),
            "entity_type": entity.entity_type,
            "description": entity.description,
            "properties": json.dumps(entity.properties),
            "created_at": entity.created_at.isoformat(),
            "updated_at": entity.updated_at.isoformat(),
            "metadata": json.dumps(entity.metadata),
        }

    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        """Convert database row to Entity object."""
        return Entity(
            id=row["id"],
            user_id=row["user_id"],
            name=row["name"],
            entity_type=row["entity_type"],
            description=row["description"],
            properties=json.loads(row["properties"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=json.loads(row["metadata"]),
        )

    def _relationship_to_row(self, relationship: Relationship) -> dict[str, Any]:
        """Convert Relationship object to row dict for insertion."""
        return {
            "id": relationship.id,
            "user_id": relationship.user_id,
            "source_id": relationship.source_id,
            "target_id": relationship.target_id,
            "relation_type": relationship.relation_type,
            "weight": relationship.weight,
            "properties": json.dumps(relationship.properties),
            "created_at": relationship.created_at.isoformat(),
            "metadata": json.dumps(relationship.metadata),
        }

    def _row_to_relationship(self, row: sqlite3.Row) -> Relationship:
        """Convert database row to Relationship object."""
        return Relationship(
            id=row["id"],
            user_id=row["user_id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation_type=row["relation_type"],
            weight=row["weight"],
            properties=json.loads(row["properties"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=json.loads(row["metadata"]),
        )

    # =========================================================================
    # Entity Operations
    # =========================================================================

    async def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph store.

        If an entity with the same ID already exists, it will be replaced.

        Args:
            entity: The entity to add.
        """
        row = self._entity_to_row(entity)

        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO entities (
                        id, user_id, name, name_lower, entity_type, description,
                        properties, created_at, updated_at, metadata
                    ) VALUES (
                        :id, :user_id, :name, :name_lower, :entity_type, :description,
                        :properties, :created_at, :updated_at, :metadata
                    )
                    """,
                    row,
                )
                conn.commit()

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Retrieve an entity by ID.

        Args:
            entity_id: The unique identifier of the entity.

        Returns:
            The entity if found, None otherwise.
        """
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "SELECT * FROM entities WHERE id = ?",
                    (entity_id,),
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                return self._row_to_entity(row)

    async def get_entity_by_name(self, user_id: str, name: str) -> Entity | None:
        """Retrieve an entity by name (case-insensitive).

        Args:
            user_id: The user who owns the entity.
            name: The name of the entity (case-insensitive).

        Returns:
            The entity if found, None otherwise.
        """
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "SELECT * FROM entities WHERE user_id = ? AND name_lower = ?",
                    (user_id, name.lower()),
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                return self._row_to_entity(row)

    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and all its relationships.

        Relationships are automatically deleted via ON DELETE CASCADE.

        Args:
            entity_id: The unique identifier of the entity.

        Returns:
            True if the entity was deleted, False if not found.
        """
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "DELETE FROM entities WHERE id = ?",
                    (entity_id,),
                )
                conn.commit()
                return cursor.rowcount > 0

    # =========================================================================
    # Relationship Operations
    # =========================================================================

    async def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the graph store.

        If a relationship with the same ID already exists, it will be replaced.

        Args:
            relationship: The relationship to add.
        """
        row = self._relationship_to_row(relationship)

        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO relationships (
                        id, user_id, source_id, target_id, relation_type,
                        weight, properties, created_at, metadata
                    ) VALUES (
                        :id, :user_id, :source_id, :target_id, :relation_type,
                        :weight, :properties, :created_at, :metadata
                    )
                    """,
                    row,
                )
                conn.commit()

    async def get_relationships(
        self,
        entity_id: str,
        direction: RelationshipDirection = RelationshipDirection.BOTH,
        relation_type: str | None = None,
    ) -> list[Relationship]:
        """Get relationships for an entity.

        Args:
            entity_id: The entity ID to get relationships for.
            direction: Whether to get outgoing, incoming, or both relationships.
            relation_type: Optional filter for relationship type.

        Returns:
            List of matching relationships.
        """
        with self._lock:
            with self._get_conn() as conn:
                conditions = []
                params: list[Any] = []

                if direction == RelationshipDirection.OUTGOING:
                    conditions.append("source_id = ?")
                    params.append(entity_id)
                elif direction == RelationshipDirection.INCOMING:
                    conditions.append("target_id = ?")
                    params.append(entity_id)
                else:  # BOTH
                    conditions.append("(source_id = ? OR target_id = ?)")
                    params.extend([entity_id, entity_id])

                if relation_type is not None:
                    conditions.append("relation_type = ?")
                    params.append(relation_type)

                where_clause = " AND ".join(conditions)
                cursor = conn.execute(
                    f"SELECT * FROM relationships WHERE {where_clause}",  # nosec B608
                    params,
                )

                return [self._row_to_relationship(row) for row in cursor]

    async def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a single relationship.

        Args:
            relationship_id: The unique identifier of the relationship.

        Returns:
            True if the relationship was deleted, False if not found.
        """
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "DELETE FROM relationships WHERE id = ?",
                    (relationship_id,),
                )
                conn.commit()
                return cursor.rowcount > 0

    # =========================================================================
    # Graph Traversal Operations
    # =========================================================================

    async def query_subgraph(
        self,
        entity_ids: list[str],
        max_hops: int = 2,
        direction: RelationshipDirection = RelationshipDirection.BOTH,
        relation_types: list[str] | None = None,
    ) -> Subgraph:
        """Query a subgraph starting from given entities using BFS traversal.

        Performs a breadth-first traversal from the starting entities,
        collecting all entities and relationships within the specified
        number of hops.

        Args:
            entity_ids: Starting entity IDs for the traversal.
            max_hops: Maximum number of hops from starting entities (default 2).
            direction: Direction of relationship traversal.
            relation_types: Optional filter for relationship types.

        Returns:
            Subgraph containing all reachable entities and relationships.
        """
        with self._lock:
            with self._get_conn() as conn:
                collected_entities: dict[str, Entity] = {}
                collected_relationships: dict[str, Relationship] = {}

                # BFS queue: (entity_id, current_depth)
                queue: deque[tuple[str, int]] = deque()
                visited: set[str] = set()

                # Initialize queue with starting entities
                for entity_id in entity_ids:
                    cursor = conn.execute(
                        "SELECT * FROM entities WHERE id = ?",
                        (entity_id,),
                    )
                    row = cursor.fetchone()
                    if row is not None:
                        queue.append((entity_id, 0))
                        visited.add(entity_id)
                        collected_entities[entity_id] = self._row_to_entity(row)

                # BFS traversal
                while queue:
                    current_id, depth = queue.popleft()

                    if depth >= max_hops:
                        continue

                    # Build relationship query based on direction
                    if direction == RelationshipDirection.OUTGOING:
                        rel_query = "SELECT * FROM relationships WHERE source_id = ?"
                        rel_params: list[Any] = [current_id]
                    elif direction == RelationshipDirection.INCOMING:
                        rel_query = "SELECT * FROM relationships WHERE target_id = ?"
                        rel_params = [current_id]
                    else:  # BOTH
                        rel_query = (
                            "SELECT * FROM relationships WHERE source_id = ? OR target_id = ?"
                        )
                        rel_params = [current_id, current_id]

                    # Filter by relation types if specified
                    if relation_types is not None and len(relation_types) > 0:
                        placeholders = ", ".join("?" * len(relation_types))
                        rel_query += f" AND relation_type IN ({placeholders})"
                        rel_params.extend(relation_types)

                    cursor = conn.execute(rel_query, rel_params)

                    for rel_row in cursor:
                        rel = self._row_to_relationship(rel_row)

                        # Add relationship
                        collected_relationships[rel.id] = rel

                        # Determine neighbor based on direction
                        neighbor_id = None
                        if direction == RelationshipDirection.OUTGOING:
                            if rel.source_id == current_id:
                                neighbor_id = rel.target_id
                        elif direction == RelationshipDirection.INCOMING:
                            if rel.target_id == current_id:
                                neighbor_id = rel.source_id
                        else:  # BOTH
                            if rel.source_id == current_id:
                                neighbor_id = rel.target_id
                            elif rel.target_id == current_id:
                                neighbor_id = rel.source_id

                        if neighbor_id is not None and neighbor_id not in visited:
                            # Fetch neighbor entity
                            neighbor_cursor = conn.execute(
                                "SELECT * FROM entities WHERE id = ?",
                                (neighbor_id,),
                            )
                            neighbor_row = neighbor_cursor.fetchone()
                            if neighbor_row is not None:
                                visited.add(neighbor_id)
                                collected_entities[neighbor_id] = self._row_to_entity(neighbor_row)
                                queue.append((neighbor_id, depth + 1))

                return Subgraph(
                    entities=list(collected_entities.values()),
                    relationships=list(collected_relationships.values()),
                    root_entity_ids=entity_ids,
                )

    async def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 10,
        direction: RelationshipDirection = RelationshipDirection.BOTH,
    ) -> list[str] | None:
        """Find the shortest path between two entities using BFS.

        Args:
            source_id: Starting entity ID.
            target_id: Target entity ID.
            max_depth: Maximum path length to search (default 10).
            direction: Direction of relationship traversal.

        Returns:
            List of entity IDs representing the path (including source and target),
            or None if no path exists within max_depth.
        """
        with self._lock:
            with self._get_conn() as conn:
                # Edge cases
                if source_id == target_id:
                    cursor = conn.execute(
                        "SELECT id FROM entities WHERE id = ?",
                        (source_id,),
                    )
                    return [source_id] if cursor.fetchone() else None

                # Check both exist
                cursor = conn.execute(
                    "SELECT id FROM entities WHERE id IN (?, ?)",
                    (source_id, target_id),
                )
                found_ids = {row["id"] for row in cursor}
                if source_id not in found_ids or target_id not in found_ids:
                    return None

                # BFS with path tracking
                queue: deque[tuple[str, list[str]]] = deque()
                visited: set[str] = set()

                queue.append((source_id, [source_id]))
                visited.add(source_id)

                while queue:
                    current_id, path = queue.popleft()

                    if len(path) > max_depth:
                        continue

                    # Build relationship query
                    if direction == RelationshipDirection.OUTGOING:
                        rel_query = "SELECT * FROM relationships WHERE source_id = ?"
                        rel_params = [current_id]
                    elif direction == RelationshipDirection.INCOMING:
                        rel_query = "SELECT * FROM relationships WHERE target_id = ?"
                        rel_params = [current_id]
                    else:
                        rel_query = (
                            "SELECT * FROM relationships WHERE source_id = ? OR target_id = ?"
                        )
                        rel_params = [current_id, current_id]

                    cursor = conn.execute(rel_query, rel_params)

                    for rel_row in cursor:
                        # Determine neighbor
                        neighbor_id = None
                        if direction == RelationshipDirection.OUTGOING:
                            if rel_row["source_id"] == current_id:
                                neighbor_id = rel_row["target_id"]
                        elif direction == RelationshipDirection.INCOMING:
                            if rel_row["target_id"] == current_id:
                                neighbor_id = rel_row["source_id"]
                        else:
                            if rel_row["source_id"] == current_id:
                                neighbor_id = rel_row["target_id"]
                            elif rel_row["target_id"] == current_id:
                                neighbor_id = rel_row["source_id"]

                        if neighbor_id is None or neighbor_id in visited:
                            continue

                        new_path = path + [neighbor_id]
                        if neighbor_id == target_id:
                            return new_path

                        if len(new_path) <= max_depth:
                            visited.add(neighbor_id)
                            queue.append((neighbor_id, new_path))

                return None

    # =========================================================================
    # User Management Operations
    # =========================================================================

    async def clear_user(self, user_id: str) -> tuple[int, int]:
        """Clear all entities and relationships for a user.

        Args:
            user_id: The user ID to clear data for.

        Returns:
            Tuple of (entities_deleted, relationships_deleted).
        """
        with self._lock:
            with self._get_conn() as conn:
                # Delete relationships for user first (also cascade-deleted)
                cursor = conn.execute(
                    "DELETE FROM relationships WHERE user_id = ?",
                    (user_id,),
                )
                relationships_deleted = cursor.rowcount

                # Delete entities (cascades remaining relationships)
                cursor = conn.execute(
                    "DELETE FROM entities WHERE user_id = ?",
                    (user_id,),
                )
                entities_deleted = cursor.rowcount

                conn.commit()
                return entities_deleted, relationships_deleted

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def get_entities_for_user(self, user_id: str) -> list[Entity]:
        """Get all entities for a user.

        Args:
            user_id: The user ID to get entities for.

        Returns:
            List of all entities belonging to the user.
        """
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "SELECT * FROM entities WHERE user_id = ?",
                    (user_id,),
                )
                return [self._row_to_entity(row) for row in cursor]

    async def clear(self) -> None:
        """Clear all data from the store."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM relationships")
                conn.execute("DELETE FROM entities")
                conn.commit()

    @property
    def entity_count(self) -> int:
        """Get the total number of entities."""
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM entities")
                result = cursor.fetchone()[0]
                return int(result)

    @property
    def relationship_count(self) -> int:
        """Get the total number of relationships."""
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM relationships")
                result = cursor.fetchone()[0]
                return int(result)

    def stats(self) -> dict:
        """Get store statistics.

        Returns:
            Dict with counts and database info.
        """
        with self._lock:
            with self._get_conn() as conn:
                entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
                rel_count = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
                users_count = conn.execute(
                    "SELECT COUNT(DISTINCT user_id) FROM entities"
                ).fetchone()[0]

                # Get database file size
                db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

                return {
                    "entity_count": entity_count,
                    "relationship_count": rel_count,
                    "users_count": users_count,
                    "db_path": str(self.db_path),
                    "db_size_bytes": db_size,
                    "page_cache_size_kb": self._page_cache_size_kb,
                }

    def get_memory_stats(self) -> ComponentStats:
        """Get memory statistics for the MemoryTracker.

        Note: SQLite manages its own memory via page cache. We report
        the configured cache size plus estimated Python overhead.

        Returns:
            ComponentStats with current memory usage.
        """
        import sys

        from ..tracker import ComponentStats

        with self._lock:
            with self._get_conn() as conn:
                entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
                rel_count = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]

            # Estimate Python overhead (connection objects, etc.)
            # The actual data is on disk, managed by SQLite's page cache
            python_overhead = sys.getsizeof(self) + sys.getsizeof(self._lock)

            # Page cache memory (this is the bounded amount)
            page_cache_bytes = self._page_cache_size_kb * 1024

            return ComponentStats(
                name="sqlite_graph_store",
                entry_count=entity_count + rel_count,
                size_bytes=python_overhead + page_cache_bytes,
                budget_bytes=page_cache_bytes,  # Cache size is the budget
                hits=0,
                misses=0,
                evictions=0,
            )

    def vacuum(self) -> None:
        """Reclaim unused space in the database file.

        Call this periodically after many deletes to reduce file size.
        """
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("VACUUM")

    def close(self) -> None:
        """Close any open connections (cleanup).

        Note: This store uses connection-per-request pattern,
        so there's typically nothing to close.
        """
        pass
