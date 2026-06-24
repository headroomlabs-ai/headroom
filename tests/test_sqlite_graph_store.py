"""Comprehensive integration tests for SQLiteGraphStore.

Tests verify:
- Entity CRUD operations with persistence
- Relationship CRUD operations with CASCADE delete
- Case-insensitive name lookups
- BFS subgraph traversal
- Shortest path finding
- User data isolation
- Memory bounding via page cache
- Database file persistence across instances
"""

from __future__ import annotations

import os
import tempfile

import pytest

from headroom.memory.adapters.graph_models import (
    Entity,
    Relationship,
    RelationshipDirection,
)
from headroom.memory.adapters.sqlite_graph import SQLiteGraphStore


class TestSQLiteGraphStoreEntityOperations:
    """Tests for entity CRUD operations."""

    @pytest.fixture
    def store(self):
        """Create a temporary SQLite graph store."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = SQLiteGraphStore(db_path=db_path)
        yield store
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_add_and_get_entity(self, store):
        """Test adding and retrieving an entity."""
        entity = Entity(
            user_id="user1",
            name="Project Alpha",
            entity_type="project",
            description="A test project",
            properties={"priority": "high"},
        )

        await store.add_entity(entity)
        retrieved = await store.get_entity(entity.id)

        assert retrieved is not None
        assert retrieved.id == entity.id
        assert retrieved.user_id == "user1"
        assert retrieved.name == "Project Alpha"
        assert retrieved.entity_type == "project"
        assert retrieved.description == "A test project"
        assert retrieved.properties == {"priority": "high"}

    @pytest.mark.asyncio
    async def test_get_entity_not_found(self, store):
        """Test retrieving a non-existent entity."""
        result = await store.get_entity("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_entity_by_name_case_insensitive(self, store):
        """Test case-insensitive entity lookup by name."""
        entity = Entity(
            user_id="user1",
            name="MyEntity",
            entity_type="test",
        )

        await store.add_entity(entity)

        # All these should find the same entity
        assert (await store.get_entity_by_name("user1", "MyEntity")) is not None
        assert (await store.get_entity_by_name("user1", "myentity")) is not None
        assert (await store.get_entity_by_name("user1", "MYENTITY")) is not None
        assert (await store.get_entity_by_name("user1", "mYeNtItY")) is not None

    @pytest.mark.asyncio
    async def test_get_entity_by_name_user_isolation(self, store):
        """Test that entity lookup is scoped to user."""
        entity1 = Entity(user_id="user1", name="SharedName", entity_type="test")
        entity2 = Entity(user_id="user2", name="SharedName", entity_type="test")

        await store.add_entity(entity1)
        await store.add_entity(entity2)

        result1 = await store.get_entity_by_name("user1", "SharedName")
        result2 = await store.get_entity_by_name("user2", "SharedName")

        assert result1 is not None
        assert result2 is not None
        assert result1.id != result2.id
        assert result1.user_id == "user1"
        assert result2.user_id == "user2"

    @pytest.mark.asyncio
    async def test_update_entity(self, store):
        """Test updating an existing entity."""
        entity = Entity(
            user_id="user1",
            name="Original Name",
            entity_type="test",
        )

        await store.add_entity(entity)

        # Update the entity
        entity.name = "Updated Name"
        entity.entity_type = "updated_type"
        entity.properties = {"new_key": "new_value"}
        await store.add_entity(entity)

        retrieved = await store.get_entity(entity.id)
        assert retrieved is not None
        assert retrieved.name == "Updated Name"
        assert retrieved.entity_type == "updated_type"
        assert retrieved.properties == {"new_key": "new_value"}

    @pytest.mark.asyncio
    async def test_delete_entity(self, store):
        """Test deleting an entity."""
        entity = Entity(user_id="user1", name="ToDelete", entity_type="test")

        await store.add_entity(entity)
        assert await store.get_entity(entity.id) is not None

        result = await store.delete_entity(entity.id)
        assert result is True

        assert await store.get_entity(entity.id) is None

    @pytest.mark.asyncio
    async def test_delete_entity_not_found(self, store):
        """Test deleting a non-existent entity."""
        result = await store.delete_entity("nonexistent-id")
        assert result is False


class TestSQLiteGraphStoreRelationshipOperations:
    """Tests for relationship CRUD operations."""

    @pytest.fixture
    async def store_with_entities(self):
        """Create a store with some entities."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        store = SQLiteGraphStore(db_path=db_path)

        # Create entities
        alice = Entity(user_id="user1", name="Alice", entity_type="person")
        bob = Entity(user_id="user1", name="Bob", entity_type="person")
        charlie = Entity(user_id="user1", name="Charlie", entity_type="person")

        await store.add_entity(alice)
        await store.add_entity(bob)
        await store.add_entity(charlie)

        yield store, alice, bob, charlie

        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_add_and_get_relationship(self, store_with_entities):
        """Test adding and retrieving a relationship."""
        store, alice, bob, _ = store_with_entities

        rel = Relationship(
            user_id="user1",
            source_id=alice.id,
            target_id=bob.id,
            relation_type="knows",
            weight=0.9,
            properties={"since": "2020"},
        )

        await store.add_relationship(rel)

        # Get outgoing relationships from Alice
        rels = await store.get_relationships(alice.id, RelationshipDirection.OUTGOING)
        assert len(rels) == 1
        assert rels[0].id == rel.id
        assert rels[0].source_id == alice.id
        assert rels[0].target_id == bob.id
        assert rels[0].relation_type == "knows"
        assert rels[0].weight == 0.9
        assert rels[0].properties == {"since": "2020"}

    @pytest.mark.asyncio
    async def test_get_relationships_by_direction(self, store_with_entities):
        """Test getting relationships by direction."""
        store, alice, bob, charlie = store_with_entities

        # Alice -> Bob
        rel1 = Relationship(
            user_id="user1",
            source_id=alice.id,
            target_id=bob.id,
            relation_type="knows",
        )
        # Charlie -> Alice
        rel2 = Relationship(
            user_id="user1",
            source_id=charlie.id,
            target_id=alice.id,
            relation_type="follows",
        )

        await store.add_relationship(rel1)
        await store.add_relationship(rel2)

        # Outgoing from Alice
        outgoing = await store.get_relationships(alice.id, RelationshipDirection.OUTGOING)
        assert len(outgoing) == 1
        assert outgoing[0].target_id == bob.id

        # Incoming to Alice
        incoming = await store.get_relationships(alice.id, RelationshipDirection.INCOMING)
        assert len(incoming) == 1
        assert incoming[0].source_id == charlie.id

        # Both directions
        both = await store.get_relationships(alice.id, RelationshipDirection.BOTH)
        assert len(both) == 2

    @pytest.mark.asyncio
    async def test_get_relationships_filter_by_type(self, store_with_entities):
        """Test filtering relationships by type."""
        store, alice, bob, charlie = store_with_entities

        rel1 = Relationship(
            user_id="user1",
            source_id=alice.id,
            target_id=bob.id,
            relation_type="knows",
        )
        rel2 = Relationship(
            user_id="user1",
            source_id=alice.id,
            target_id=charlie.id,
            relation_type="manages",
        )

        await store.add_relationship(rel1)
        await store.add_relationship(rel2)

        # Filter by type
        knows_rels = await store.get_relationships(
            alice.id, RelationshipDirection.OUTGOING, relation_type="knows"
        )
        assert len(knows_rels) == 1
        assert knows_rels[0].target_id == bob.id

        manages_rels = await store.get_relationships(
            alice.id, RelationshipDirection.OUTGOING, relation_type="manages"
        )
        assert len(manages_rels) == 1
        assert manages_rels[0].target_id == charlie.id

    @pytest.mark.asyncio
    async def test_delete_relationship(self, store_with_entities):
        """Test deleting a relationship."""
        store, alice, bob, _ = store_with_entities

        rel = Relationship(
            user_id="user1",
            source_id=alice.id,
            target_id=bob.id,
            relation_type="knows",
        )

        await store.add_relationship(rel)
        rels = await store.get_relationships(alice.id, RelationshipDirection.OUTGOING)
        assert len(rels) == 1

        result = await store.delete_relationship(rel.id)
        assert result is True

        rels = await store.get_relationships(alice.id, RelationshipDirection.OUTGOING)
        assert len(rels) == 0

    @pytest.mark.asyncio
    async def test_cascade_delete_relationships(self, store_with_entities):
        """Test that deleting an entity cascades to its relationships."""
        store, alice, bob, charlie = store_with_entities

        # Create relationships
        rel1 = Relationship(
            user_id="user1",
            source_id=alice.id,
            target_id=bob.id,
            relation_type="knows",
        )
        rel2 = Relationship(
            user_id="user1",
            source_id=charlie.id,
            target_id=alice.id,
            relation_type="follows",
        )

        await store.add_relationship(rel1)
        await store.add_relationship(rel2)

        # Delete Alice - should cascade delete both relationships
        await store.delete_entity(alice.id)

        # Both relationships should be gone
        bob_rels = await store.get_relationships(bob.id, RelationshipDirection.BOTH)
        assert len(bob_rels) == 0

        charlie_rels = await store.get_relationships(charlie.id, RelationshipDirection.BOTH)
        assert len(charlie_rels) == 0


class TestSQLiteGraphStoreTraversal:
    """Tests for graph traversal operations."""

    @pytest.fixture
    async def store_with_graph(self):
        """Create a store with a connected graph.

        Graph structure:
            A -> B -> D
            |    |
            v    v
            C -> E
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        store = SQLiteGraphStore(db_path=db_path)

        # Create entities
        a = Entity(user_id="user1", name="A", entity_type="node")
        b = Entity(user_id="user1", name="B", entity_type="node")
        c = Entity(user_id="user1", name="C", entity_type="node")
        d = Entity(user_id="user1", name="D", entity_type="node")
        e = Entity(user_id="user1", name="E", entity_type="node")

        for entity in [a, b, c, d, e]:
            await store.add_entity(entity)

        # Create relationships: A->B, A->C, B->D, B->E, C->E
        rels = [
            Relationship(user_id="user1", source_id=a.id, target_id=b.id, relation_type="edge"),
            Relationship(user_id="user1", source_id=a.id, target_id=c.id, relation_type="edge"),
            Relationship(user_id="user1", source_id=b.id, target_id=d.id, relation_type="edge"),
            Relationship(user_id="user1", source_id=b.id, target_id=e.id, relation_type="edge"),
            Relationship(user_id="user1", source_id=c.id, target_id=e.id, relation_type="edge"),
        ]

        for rel in rels:
            await store.add_relationship(rel)

        yield store, {"A": a, "B": b, "C": c, "D": d, "E": e}

        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_query_subgraph_single_hop(self, store_with_graph):
        """Test querying subgraph with single hop."""
        store, nodes = store_with_graph

        subgraph = await store.query_subgraph(
            [nodes["A"].id], max_hops=1, direction=RelationshipDirection.OUTGOING
        )

        entity_names = {e.name for e in subgraph.entities}
        assert entity_names == {"A", "B", "C"}
        assert len(subgraph.relationships) == 2

    @pytest.mark.asyncio
    async def test_query_subgraph_two_hops(self, store_with_graph):
        """Test querying subgraph with two hops."""
        store, nodes = store_with_graph

        subgraph = await store.query_subgraph(
            [nodes["A"].id], max_hops=2, direction=RelationshipDirection.OUTGOING
        )

        entity_names = {e.name for e in subgraph.entities}
        assert entity_names == {"A", "B", "C", "D", "E"}
        assert len(subgraph.relationships) == 5

    @pytest.mark.asyncio
    async def test_query_subgraph_incoming(self, store_with_graph):
        """Test querying subgraph with incoming direction."""
        store, nodes = store_with_graph

        subgraph = await store.query_subgraph(
            [nodes["E"].id], max_hops=2, direction=RelationshipDirection.INCOMING
        )

        entity_names = {e.name for e in subgraph.entities}
        # E <- B <- A, E <- C <- A
        assert "E" in entity_names
        assert "B" in entity_names
        assert "C" in entity_names
        assert "A" in entity_names

    @pytest.mark.asyncio
    async def test_find_path_direct(self, store_with_graph):
        """Test finding a direct path."""
        store, nodes = store_with_graph

        path = await store.find_path(nodes["A"].id, nodes["B"].id)

        assert path is not None
        assert len(path) == 2
        assert path[0] == nodes["A"].id
        assert path[1] == nodes["B"].id

    @pytest.mark.asyncio
    async def test_find_path_multi_hop(self, store_with_graph):
        """Test finding a multi-hop path."""
        store, nodes = store_with_graph

        path = await store.find_path(nodes["A"].id, nodes["D"].id)

        assert path is not None
        assert len(path) == 3  # A -> B -> D
        assert path[0] == nodes["A"].id
        assert path[-1] == nodes["D"].id

    @pytest.mark.asyncio
    async def test_find_path_not_found(self, store_with_graph):
        """Test that None is returned when no path exists."""
        store, nodes = store_with_graph

        # D has no outgoing edges, so no path from D to A
        path = await store.find_path(
            nodes["D"].id, nodes["A"].id, direction=RelationshipDirection.OUTGOING
        )

        assert path is None

    @pytest.mark.asyncio
    async def test_find_path_self(self, store_with_graph):
        """Test finding a path to self."""
        store, nodes = store_with_graph

        path = await store.find_path(nodes["A"].id, nodes["A"].id)

        assert path is not None
        assert path == [nodes["A"].id]


class TestSQLiteGraphStoreUserManagement:
    """Tests for user data management."""

    @pytest.fixture
    def store(self):
        """Create a temporary SQLite graph store."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = SQLiteGraphStore(db_path=db_path)
        yield store
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_get_entities_for_user(self, store):
        """Test getting all entities for a specific user."""
        # Create entities for two users
        for i in range(3):
            await store.add_entity(
                Entity(user_id="user1", name=f"User1Entity{i}", entity_type="test")
            )
        for i in range(2):
            await store.add_entity(
                Entity(user_id="user2", name=f"User2Entity{i}", entity_type="test")
            )

        user1_entities = await store.get_entities_for_user("user1")
        user2_entities = await store.get_entities_for_user("user2")

        assert len(user1_entities) == 3
        assert len(user2_entities) == 2
        assert all(e.user_id == "user1" for e in user1_entities)
        assert all(e.user_id == "user2" for e in user2_entities)

    @pytest.mark.asyncio
    async def test_clear_user(self, store):
        """Test clearing all data for a user."""
        # Create entities and relationships for two users
        alice = Entity(user_id="user1", name="Alice", entity_type="person")
        bob = Entity(user_id="user1", name="Bob", entity_type="person")
        carol = Entity(user_id="user2", name="Carol", entity_type="person")

        await store.add_entity(alice)
        await store.add_entity(bob)
        await store.add_entity(carol)

        rel = Relationship(
            user_id="user1",
            source_id=alice.id,
            target_id=bob.id,
            relation_type="knows",
        )
        await store.add_relationship(rel)

        # Clear user1
        entities_deleted, rels_deleted = await store.clear_user("user1")

        assert entities_deleted == 2
        assert rels_deleted == 1

        # User2 data should still exist
        assert await store.get_entity(carol.id) is not None
        assert store.entity_count == 1

    @pytest.mark.asyncio
    async def test_clear_all(self, store):
        """Test clearing all data."""
        # Add some data
        for i in range(5):
            await store.add_entity(
                Entity(user_id=f"user{i}", name=f"Entity{i}", entity_type="test")
            )

        assert store.entity_count == 5

        await store.clear()

        assert store.entity_count == 0
        assert store.relationship_count == 0


class TestSQLiteGraphStorePersistence:
    """Tests for data persistence across store instances."""

    @pytest.mark.asyncio
    async def test_data_persists_across_instances(self):
        """Test that data survives store restart."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Create store and add data
            store1 = SQLiteGraphStore(db_path=db_path)
            entity = Entity(user_id="user1", name="Persistent", entity_type="test")
            await store1.add_entity(entity)
            entity_id = entity.id

            # Create new store instance pointing to same database
            store2 = SQLiteGraphStore(db_path=db_path)

            # Data should be there
            retrieved = await store2.get_entity(entity_id)
            assert retrieved is not None
            assert retrieved.name == "Persistent"
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_relationships_persist(self):
        """Test that relationships persist across instances."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store1 = SQLiteGraphStore(db_path=db_path)

            alice = Entity(user_id="user1", name="Alice", entity_type="person")
            bob = Entity(user_id="user1", name="Bob", entity_type="person")
            await store1.add_entity(alice)
            await store1.add_entity(bob)

            rel = Relationship(
                user_id="user1",
                source_id=alice.id,
                target_id=bob.id,
                relation_type="knows",
            )
            await store1.add_relationship(rel)

            # New instance
            store2 = SQLiteGraphStore(db_path=db_path)

            rels = await store2.get_relationships(alice.id, RelationshipDirection.OUTGOING)
            assert len(rels) == 1
            assert rels[0].target_id == bob.id
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestSQLiteGraphStoreMemoryStats:
    """Tests for memory statistics and bounding."""

    @pytest.fixture
    def store(self):
        """Create a temporary SQLite graph store."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = SQLiteGraphStore(db_path=db_path, page_cache_size_kb=4096)
        yield store
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_stats(self, store):
        """Test getting store statistics."""
        # Add some data
        for i in range(10):
            await store.add_entity(Entity(user_id="user1", name=f"Entity{i}", entity_type="test"))

        stats = store.stats()

        assert stats["entity_count"] == 10
        assert stats["relationship_count"] == 0
        assert stats["users_count"] == 1
        assert stats["page_cache_size_kb"] == 4096
        assert "db_path" in stats
        assert stats["db_size_bytes"] > 0

    @pytest.mark.asyncio
    async def test_memory_stats(self, store):
        """Test memory statistics for MemoryTracker."""
        # Add some data
        for i in range(5):
            await store.add_entity(Entity(user_id="user1", name=f"Entity{i}", entity_type="test"))

        stats = store.get_memory_stats()

        assert stats.name == "sqlite_graph_store"
        assert stats.entry_count == 5
        assert stats.size_bytes > 0
        # Budget should be the page cache size
        assert stats.budget_bytes == 4096 * 1024

    @pytest.mark.asyncio
    async def test_vacuum(self, store):
        """Test vacuuming the database."""
        # Add and delete data
        entities = []
        for i in range(100):
            e = Entity(user_id="user1", name=f"Entity{i}", entity_type="test")
            await store.add_entity(e)
            entities.append(e)

        # Delete all
        for e in entities:
            await store.delete_entity(e.id)

        # Get size before vacuum
        stats_before = store.stats()

        # Vacuum
        store.vacuum()

        # Size should be smaller or same after vacuum
        stats_after = store.stats()
        assert stats_after["db_size_bytes"] <= stats_before["db_size_bytes"]


class TestSQLiteGraphStoreEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def store(self):
        """Create a temporary SQLite graph store."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = SQLiteGraphStore(db_path=db_path)
        yield store
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_empty_subgraph_query(self, store):
        """Test querying subgraph with no entities."""
        subgraph = await store.query_subgraph(["nonexistent-id"])
        assert len(subgraph.entities) == 0
        assert len(subgraph.relationships) == 0

    @pytest.mark.asyncio
    async def test_entity_with_special_characters(self, store):
        """Test entity names with special characters."""
        entity = Entity(
            user_id="user1",
            name='Test\'s "Entity" (Special & <More>)',
            entity_type="test",
            description="Description with 'quotes' and \"more\"",
            properties={"key": "value with 'quotes'"},
        )

        await store.add_entity(entity)
        retrieved = await store.get_entity(entity.id)

        assert retrieved is not None
        assert retrieved.name == entity.name
        assert retrieved.description == entity.description
        assert retrieved.properties == entity.properties

    @pytest.mark.asyncio
    async def test_entity_with_unicode(self, store):
        """Test entity names with unicode characters."""
        entity = Entity(
            user_id="user1",
            name="Test 你好 🚀 Ñoño",
            entity_type="test",
        )

        await store.add_entity(entity)
        retrieved = await store.get_entity(entity.id)

        assert retrieved is not None
        assert retrieved.name == "Test 你好 🚀 Ñoño"

    @pytest.mark.asyncio
    async def test_large_properties(self, store):
        """Test entities with large properties."""
        large_props = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}

        entity = Entity(
            user_id="user1",
            name="LargeEntity",
            entity_type="test",
            properties=large_props,
        )

        await store.add_entity(entity)
        retrieved = await store.get_entity(entity.id)

        assert retrieved is not None
        assert retrieved.properties == large_props

    @pytest.mark.asyncio
    async def test_concurrent_access(self, store):
        """Test basic concurrent access (thread-safe pattern)."""
        import asyncio

        async def add_entity(i: int):
            e = Entity(user_id="user1", name=f"Concurrent{i}", entity_type="test")
            await store.add_entity(e)
            return e.id

        # Add entities concurrently
        ids = await asyncio.gather(*[add_entity(i) for i in range(20)])

        # All should exist
        assert len(ids) == 20
        assert store.entity_count == 20


class TestSQLiteGraphStoreMemoryTrackerIntegration:
    """Tests for MemoryTracker integration."""

    @pytest.fixture
    def store(self):
        """Create a temporary SQLite graph store."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = SQLiteGraphStore(db_path=db_path, page_cache_size_kb=4096)
        yield store
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_memory_tracker_registration(self, store):
        """Test registering SQLiteGraphStore with MemoryTracker."""
        from headroom.memory.tracker import MemoryTracker

        tracker = MemoryTracker.get()

        # Unregister if already registered from previous test
        tracker.unregister("sqlite_graph_test")

        # Register the store
        tracker.register("sqlite_graph_test", store.get_memory_stats)

        try:
            # Add some data
            for i in range(10):
                await store.add_entity(
                    Entity(user_id="user1", name=f"Entity{i}", entity_type="test")
                )

            # Get report
            report = tracker.get_report()

            # Find our component in the report's components dict
            assert "sqlite_graph_test" in report.components
            graph_stats = report.components["sqlite_graph_test"]

            assert graph_stats is not None
            assert graph_stats.entry_count == 10
            assert graph_stats.budget_bytes == 4096 * 1024  # 4MB cache
        finally:
            tracker.unregister("sqlite_graph_test")

    @pytest.mark.asyncio
    async def test_memory_stats_tracks_growth(self, store):
        """Test that memory stats track entity/relationship growth."""
        stats_before = store.get_memory_stats()
        assert stats_before.entry_count == 0

        # Add entities
        entities = []
        for i in range(5):
            e = Entity(user_id="user1", name=f"Entity{i}", entity_type="test")
            await store.add_entity(e)
            entities.append(e)

        stats_after_entities = store.get_memory_stats()
        assert stats_after_entities.entry_count == 5

        # Add relationships
        for i in range(4):
            rel = Relationship(
                user_id="user1",
                source_id=entities[i].id,
                target_id=entities[i + 1].id,
                relation_type="connected",
            )
            await store.add_relationship(rel)

        stats_after_rels = store.get_memory_stats()
        assert stats_after_rels.entry_count == 9  # 5 entities + 4 relationships

    @pytest.mark.asyncio
    async def test_memory_stats_bounded_by_cache(self, store):
        """Test that reported size is bounded by cache size."""
        # Add many entities
        for i in range(100):
            await store.add_entity(Entity(user_id="user1", name=f"Entity{i}", entity_type="test"))

        stats = store.get_memory_stats()

        # Size should be bounded by cache + overhead
        # Cache is 4MB, overhead should be small
        assert stats.size_bytes < 5 * 1024 * 1024  # Less than 5MB
        assert stats.budget_bytes == 4096 * 1024  # Exactly 4MB

    @pytest.mark.asyncio
    async def test_memory_report_includes_sqlite_graph(self, store):
        """Test that MemoryTracker report includes SQLiteGraphStore stats."""
        from headroom.memory.tracker import MemoryTracker

        tracker = MemoryTracker.get()

        # Unregister if already registered from previous test
        tracker.unregister("graph_report_test")
        tracker.register("graph_report_test", store.get_memory_stats)

        try:
            # Add data
            await store.add_entity(Entity(user_id="user1", name="Test", entity_type="test"))

            # Get full report dict
            report = tracker.get_report()
            report_dict = report.to_dict()

            # Verify structure
            assert "components" in report_dict
            assert len(report_dict["components"]) > 0

            # Find our component (components is a dict in to_dict output)
            assert "graph_report_test" in report_dict["components"]
            comp = report_dict["components"]["graph_report_test"]

            assert comp["name"] == "sqlite_graph_store"
            assert comp["entry_count"] == 1
            assert "size_bytes" in comp
            assert "budget_bytes" in comp
        finally:
            tracker.unregister("graph_report_test")

    @pytest.mark.asyncio
    async def test_memory_stats_after_delete(self, store):
        """Test that memory stats decrease after deletion."""
        # Add entities
        entities = []
        for i in range(10):
            e = Entity(user_id="user1", name=f"Entity{i}", entity_type="test")
            await store.add_entity(e)
            entities.append(e)

        stats_before_delete = store.get_memory_stats()
        assert stats_before_delete.entry_count == 10

        # Delete half
        for e in entities[:5]:
            await store.delete_entity(e.id)

        stats_after_delete = store.get_memory_stats()
        assert stats_after_delete.entry_count == 5
