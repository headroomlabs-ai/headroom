"""In-memory graph store for Headroom's knowledge graph memory system.

Provides thread-safe in-memory storage for entities and relationships
with efficient lookup via multiple indexes and BFS-based traversal.
"""

from __future__ import annotations

from collections import deque
from threading import RLock
from typing import TYPE_CHECKING

from .graph_models import Entity, Relationship, RelationshipDirection, Subgraph

if TYPE_CHECKING:
    from ..tracker import ComponentStats


class InMemoryGraphStore:
    """Thread-safe in-memory graph store implementing the GraphStore protocol.

    Provides storage for entities and relationships with efficient lookup
    via multiple indexes. All operations are thread-safe using RLock.

    Features:
    - O(1) entity and relationship lookup by ID
    - O(1) entity lookup by name (per user, case-insensitive)
    - O(1) lookup of relationships by source or target entity
    - BFS-based subgraph traversal with configurable hop limit
    - BFS-based shortest path finding between entities

    Indexes:
    - _entities: dict[str, Entity] - entity_id -> Entity
    - _relationships: dict[str, Relationship] - relationship_id -> Relationship
    - _entities_by_user: dict[str, set[str]] - user_id -> entity_ids
    - _entities_by_name: dict[str, dict[str, str]] - user_id -> {name_lower: entity_id}
    - _relationships_by_source: dict[str, set[str]] - entity_id -> relationship_ids (outgoing)
    - _relationships_by_target: dict[str, set[str]] - entity_id -> relationship_ids (incoming)

    Usage:
        store = InMemoryGraphStore()
        await store.add_entity(Entity(user_id="alice", name="Project X", entity_type="project"))
        entity = await store.get_entity_by_name("alice", "project x")  # Case-insensitive
        subgraph = await store.query_subgraph(["entity-id"], max_hops=2)
    """

    def __init__(self) -> None:
        """Initialize the in-memory graph store with empty indexes."""
        # Primary storage
        self._entities: dict[str, Entity] = {}
        self._relationships: dict[str, Relationship] = {}

        # Secondary indexes for efficient lookup
        self._entities_by_user: dict[str, set[str]] = {}
        self._entities_by_name: dict[str, dict[str, str]] = {}  # user_id -> {name_lower: entity_id}
        self._relationships_by_source: dict[str, set[str]] = {}  # entity_id -> relationship_ids
        self._relationships_by_target: dict[str, set[str]] = {}  # entity_id -> relationship_ids

        # Thread safety - use RLock to allow re-entrant locking
        self._lock = RLock()

    # =========================================================================
    # Entity Operations
    # =========================================================================

    async def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph store.

        If an entity with the same ID already exists, it will be replaced
        and indexes will be updated accordingly.

        Args:
            entity: The entity to add.
        """
        with self._lock:
            # If entity already exists, remove from old indexes first
            if entity.id in self._entities:
                old_entity = self._entities[entity.id]
                self._remove_entity_from_indexes(old_entity)

            # Store the entity
            self._entities[entity.id] = entity

            # Update user index
            if entity.user_id not in self._entities_by_user:
                self._entities_by_user[entity.user_id] = set()
            self._entities_by_user[entity.user_id].add(entity.id)

            # Update name index (case-insensitive)
            if entity.user_id not in self._entities_by_name:
                self._entities_by_name[entity.user_id] = {}
            name_lower = entity.name.lower()
            self._entities_by_name[entity.user_id][name_lower] = entity.id

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Retrieve an entity by ID.

        Args:
            entity_id: The unique identifier of the entity.

        Returns:
            The entity if found, None otherwise.
        """
        with self._lock:
            return self._entities.get(entity_id)

    async def get_entity_by_name(self, user_id: str, name: str) -> Entity | None:
        """Retrieve an entity by name (case-insensitive).

        Args:
            user_id: The user who owns the entity.
            name: The name of the entity (case-insensitive).

        Returns:
            The entity if found, None otherwise.
        """
        with self._lock:
            user_names = self._entities_by_name.get(user_id)
            if user_names is None:
                return None

            entity_id = user_names.get(name.lower())
            if entity_id is None:
                return None

            return self._entities.get(entity_id)

    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and all its relationships.

        Removes the entity from storage and all indexes, and deletes
        all relationships where this entity is source or target.

        Args:
            entity_id: The unique identifier of the entity.

        Returns:
            True if the entity was deleted, False if not found.
        """
        with self._lock:
            entity = self._entities.get(entity_id)
            if entity is None:
                return False

            # Collect all relationships to delete
            relationships_to_delete: set[str] = set()

            # Outgoing relationships
            outgoing = self._relationships_by_source.get(entity_id, set())
            relationships_to_delete.update(outgoing)

            # Incoming relationships
            incoming = self._relationships_by_target.get(entity_id, set())
            relationships_to_delete.update(incoming)

            # Delete relationships
            for rel_id in relationships_to_delete:
                self._remove_relationship_internal(rel_id)

            # Remove entity from indexes
            self._remove_entity_from_indexes(entity)

            # Remove from primary storage
            del self._entities[entity_id]

            return True

    def _remove_entity_from_indexes(self, entity: Entity) -> None:
        """Remove an entity from secondary indexes (internal, must hold lock)."""
        # Remove from user index
        user_entities = self._entities_by_user.get(entity.user_id)
        if user_entities is not None:
            user_entities.discard(entity.id)
            if not user_entities:
                del self._entities_by_user[entity.user_id]

        # Remove from name index
        user_names = self._entities_by_name.get(entity.user_id)
        if user_names is not None:
            name_lower = entity.name.lower()
            if user_names.get(name_lower) == entity.id:
                del user_names[name_lower]
            if not user_names:
                del self._entities_by_name[entity.user_id]

    # =========================================================================
    # Relationship Operations
    # =========================================================================

    async def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the graph store.

        If a relationship with the same ID already exists, it will be replaced
        and indexes will be updated accordingly.

        Args:
            relationship: The relationship to add.
        """
        with self._lock:
            # If relationship already exists, remove from old indexes first
            if relationship.id in self._relationships:
                old_rel = self._relationships[relationship.id]
                self._remove_relationship_from_indexes(old_rel)

            # Store the relationship
            self._relationships[relationship.id] = relationship

            # Update source index
            if relationship.source_id not in self._relationships_by_source:
                self._relationships_by_source[relationship.source_id] = set()
            self._relationships_by_source[relationship.source_id].add(relationship.id)

            # Update target index
            if relationship.target_id not in self._relationships_by_target:
                self._relationships_by_target[relationship.target_id] = set()
            self._relationships_by_target[relationship.target_id].add(relationship.id)

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
            rel_ids: set[str] = set()

            if direction in (RelationshipDirection.OUTGOING, RelationshipDirection.BOTH):
                rel_ids.update(self._relationships_by_source.get(entity_id, set()))

            if direction in (RelationshipDirection.INCOMING, RelationshipDirection.BOTH):
                rel_ids.update(self._relationships_by_target.get(entity_id, set()))

            relationships = []
            for rel_id in rel_ids:
                rel = self._relationships.get(rel_id)
                if rel is not None:
                    if relation_type is None or rel.relation_type == relation_type:
                        relationships.append(rel)

            return relationships

    async def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a single relationship.

        Args:
            relationship_id: The unique identifier of the relationship.

        Returns:
            True if the relationship was deleted, False if not found.
        """
        with self._lock:
            return self._remove_relationship_internal(relationship_id)

    def _remove_relationship_internal(self, relationship_id: str) -> bool:
        """Remove a relationship (internal, must hold lock).

        Returns:
            True if removed, False if not found.
        """
        rel = self._relationships.get(relationship_id)
        if rel is None:
            return False

        # Remove from indexes
        self._remove_relationship_from_indexes(rel)

        # Remove from primary storage
        del self._relationships[relationship_id]
        return True

    def _remove_relationship_from_indexes(self, relationship: Relationship) -> None:
        """Remove a relationship from secondary indexes (internal, must hold lock)."""
        # Remove from source index
        source_rels = self._relationships_by_source.get(relationship.source_id)
        if source_rels is not None:
            source_rels.discard(relationship.id)
            if not source_rels:
                del self._relationships_by_source[relationship.source_id]

        # Remove from target index
        target_rels = self._relationships_by_target.get(relationship.target_id)
        if target_rels is not None:
            target_rels.discard(relationship.id)
            if not target_rels:
                del self._relationships_by_target[relationship.target_id]

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
            collected_entities: dict[str, Entity] = {}
            collected_relationships: dict[str, Relationship] = {}

            # BFS queue: (entity_id, current_depth)
            queue: deque[tuple[str, int]] = deque()
            visited: set[str] = set()

            # Initialize queue with starting entities
            for entity_id in entity_ids:
                if entity_id in self._entities:
                    queue.append((entity_id, 0))
                    visited.add(entity_id)
                    collected_entities[entity_id] = self._entities[entity_id]

            # BFS traversal
            while queue:
                current_id, depth = queue.popleft()

                if depth >= max_hops:
                    continue

                # Get relationships based on direction
                rel_ids: set[str] = set()

                if direction in (RelationshipDirection.OUTGOING, RelationshipDirection.BOTH):
                    rel_ids.update(self._relationships_by_source.get(current_id, set()))

                if direction in (RelationshipDirection.INCOMING, RelationshipDirection.BOTH):
                    rel_ids.update(self._relationships_by_target.get(current_id, set()))

                # Process relationships
                for rel_id in rel_ids:
                    rel = self._relationships.get(rel_id)
                    if rel is None:
                        continue

                    # Filter by relation type if specified
                    if relation_types is not None and rel.relation_type not in relation_types:
                        continue

                    # Add relationship
                    collected_relationships[rel_id] = rel

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
                        neighbor = self._entities.get(neighbor_id)
                        if neighbor is not None:
                            visited.add(neighbor_id)
                            collected_entities[neighbor_id] = neighbor
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
            # Edge cases
            if source_id == target_id:
                return [source_id] if source_id in self._entities else None

            if source_id not in self._entities or target_id not in self._entities:
                return None

            # BFS with path tracking
            # Queue: (current_entity_id, path_so_far)
            queue: deque[tuple[str, list[str]]] = deque()
            visited: set[str] = set()

            queue.append((source_id, [source_id]))
            visited.add(source_id)

            while queue:
                current_id, path = queue.popleft()

                if len(path) > max_depth:
                    continue

                # Get neighbors
                rel_ids: set[str] = set()

                if direction in (RelationshipDirection.OUTGOING, RelationshipDirection.BOTH):
                    rel_ids.update(self._relationships_by_source.get(current_id, set()))

                if direction in (RelationshipDirection.INCOMING, RelationshipDirection.BOTH):
                    rel_ids.update(self._relationships_by_target.get(current_id, set()))

                for rel_id in rel_ids:
                    rel = self._relationships.get(rel_id)
                    if rel is None:
                        continue

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

                    if neighbor_id is None or neighbor_id in visited:
                        continue

                    # Check if we found the target
                    new_path = path + [neighbor_id]
                    if neighbor_id == target_id:
                        return new_path

                    # Continue BFS if within depth limit
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
            entities_deleted = 0
            relationships_deleted = 0

            # Get all entity IDs for this user
            entity_ids = list(self._entities_by_user.get(user_id, set()))

            # Delete each entity (which also deletes its relationships)
            for entity_id in entity_ids:
                entity = self._entities.get(entity_id)
                if entity is None:
                    continue

                # Count and delete relationships
                outgoing = self._relationships_by_source.get(entity_id, set())
                incoming = self._relationships_by_target.get(entity_id, set())
                rel_ids_to_delete = outgoing | incoming

                for rel_id in list(rel_ids_to_delete):
                    if self._remove_relationship_internal(rel_id):
                        relationships_deleted += 1

                # Remove entity from indexes
                self._remove_entity_from_indexes(entity)

                # Remove from primary storage
                del self._entities[entity_id]
                entities_deleted += 1

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
            entity_ids = self._entities_by_user.get(user_id, set())
            return [self._entities[eid] for eid in entity_ids if eid in self._entities]

    async def clear(self) -> None:
        """Clear all data from the store."""
        with self._lock:
            self._entities.clear()
            self._relationships.clear()
            self._entities_by_user.clear()
            self._entities_by_name.clear()
            self._relationships_by_source.clear()
            self._relationships_by_target.clear()

    @property
    def entity_count(self) -> int:
        """Get the total number of entities."""
        with self._lock:
            return len(self._entities)

    @property
    def relationship_count(self) -> int:
        """Get the total number of relationships."""
        with self._lock:
            return len(self._relationships)

    def stats(self) -> dict:
        """Get store statistics.

        Returns:
            Dict with counts and index sizes.
        """
        with self._lock:
            return {
                "entity_count": len(self._entities),
                "relationship_count": len(self._relationships),
                "users_count": len(self._entities_by_user),
                "source_index_size": len(self._relationships_by_source),
                "target_index_size": len(self._relationships_by_target),
            }

    def get_memory_stats(self) -> ComponentStats:
        """Get memory statistics for the MemoryTracker.

        Returns:
            ComponentStats with current memory usage.
        """
        import sys

        from ..tracker import ComponentStats

        with self._lock:
            # Calculate size of all data structures
            size_bytes = 0

            # Entities
            size_bytes += sys.getsizeof(self._entities)
            for entity_id, entity in self._entities.items():
                size_bytes += len(entity_id)
                size_bytes += sys.getsizeof(entity)
                size_bytes += len(entity.id) + len(entity.user_id) + len(entity.name)
                size_bytes += len(entity.entity_type)
                if entity.properties:
                    size_bytes += sys.getsizeof(entity.properties)

            # Relationships
            size_bytes += sys.getsizeof(self._relationships)
            for rel_id, rel in self._relationships.items():
                size_bytes += len(rel_id)
                size_bytes += sys.getsizeof(rel)
                size_bytes += len(rel.id) + len(rel.source_id) + len(rel.target_id)
                size_bytes += len(rel.relation_type)
                if rel.properties:
                    size_bytes += sys.getsizeof(rel.properties)

            # Indexes
            size_bytes += sys.getsizeof(self._entities_by_user)
            for user_id, entity_ids in self._entities_by_user.items():
                size_bytes += len(user_id)
                size_bytes += sys.getsizeof(entity_ids)

            size_bytes += sys.getsizeof(self._entities_by_name)
            for user_id, name_map in self._entities_by_name.items():
                size_bytes += len(user_id)
                size_bytes += sys.getsizeof(name_map)

            size_bytes += sys.getsizeof(self._relationships_by_source)
            size_bytes += sys.getsizeof(self._relationships_by_target)

            entry_count = len(self._entities) + len(self._relationships)

            return ComponentStats(
                name="graph_store",
                entry_count=entry_count,
                size_bytes=size_bytes,
                budget_bytes=None,
                hits=0,
                misses=0,
                evictions=0,
            )
