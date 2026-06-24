"""Comprehensive tests for Headroom's memory system.

Tests cover:
- Entity and Relationship graph models
- InMemoryGraphStore operations
- MemorySystem orchestrator
- Memory tool definitions
- LocalBackend integration

Following the patterns established in test_memory_eval.py and test_memory/test_hierarchical.py.
"""

# CRITICAL: Must set TOKENIZERS_PARALLELISM before any imports that might
# trigger sentence_transformers/transformers loading.
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

# Check if hnswlib is available for LocalBackend tests
try:
    from headroom.memory.adapters.hnsw import _check_hnswlib_available

    HNSW_AVAILABLE = _check_hnswlib_available()
except ImportError:
    HNSW_AVAILABLE = False

from headroom.memory.adapters.graph import InMemoryGraphStore
from headroom.memory.adapters.graph_models import (
    Entity,
    Relationship,
    RelationshipDirection,
    Subgraph,
)
from headroom.memory.models import Memory
from headroom.memory.ports import MemorySearchResult as PortsMemorySearchResult
from headroom.memory.system import MemorySearchResult, MemorySystem
from headroom.memory.tools import MEMORY_TOOLS, get_memory_tools, get_tool_names

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def graph_store():
    """Create a fresh InMemoryGraphStore for testing."""
    return InMemoryGraphStore()


@pytest.fixture
def sample_entity():
    """Create a sample entity for testing."""
    return Entity(
        id="entity-1",
        user_id="alice",
        name="Project Alpha",
        entity_type="project",
        description="Main development project",
        properties={"status": "active", "priority": 1},
        metadata={"source": "test"},
    )


@pytest.fixture
def sample_relationship():
    """Create a sample relationship for testing."""
    return Relationship(
        id="rel-1",
        user_id="alice",
        source_id="entity-1",
        target_id="entity-2",
        relation_type="works_on",
        weight=0.9,
        properties={"role": "lead"},
        metadata={"source": "test"},
    )


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)


# =============================================================================
# TestEntity and TestRelationship - Unit tests for graph models
# =============================================================================


class TestEntity:
    """Tests for Entity dataclass."""

    def test_entity_creation_with_defaults(self):
        """Test entity creation with default values."""
        entity = Entity()

        assert entity.id is not None  # Auto-generated UUID
        assert entity.user_id == ""
        assert entity.name == ""
        assert entity.entity_type == "unknown"
        assert entity.description is None
        assert entity.properties == {}
        assert entity.metadata == {}
        assert entity.created_at is not None
        assert entity.updated_at is not None

    def test_entity_creation_with_values(self, sample_entity):
        """Test entity creation with provided values."""
        assert sample_entity.id == "entity-1"
        assert sample_entity.user_id == "alice"
        assert sample_entity.name == "Project Alpha"
        assert sample_entity.entity_type == "project"
        assert sample_entity.description == "Main development project"
        assert sample_entity.properties["status"] == "active"
        assert sample_entity.metadata["source"] == "test"

    def test_entity_to_dict(self, sample_entity):
        """Test entity serialization to dictionary."""
        data = sample_entity.to_dict()

        assert data["id"] == "entity-1"
        assert data["user_id"] == "alice"
        assert data["name"] == "Project Alpha"
        assert data["entity_type"] == "project"
        assert data["description"] == "Main development project"
        assert data["properties"]["status"] == "active"
        assert "created_at" in data
        assert "updated_at" in data

    def test_entity_from_dict(self, sample_entity):
        """Test entity deserialization from dictionary."""
        data = sample_entity.to_dict()
        restored = Entity.from_dict(data)

        assert restored.id == sample_entity.id
        assert restored.user_id == sample_entity.user_id
        assert restored.name == sample_entity.name
        assert restored.entity_type == sample_entity.entity_type
        assert restored.description == sample_entity.description
        assert restored.properties == sample_entity.properties

    def test_entity_roundtrip(self, sample_entity):
        """Test entity serialization roundtrip."""
        data = sample_entity.to_dict()
        restored = Entity.from_dict(data)
        data2 = restored.to_dict()

        assert data == data2


class TestRelationship:
    """Tests for Relationship dataclass."""

    def test_relationship_creation_with_defaults(self):
        """Test relationship creation with default values."""
        rel = Relationship()

        assert rel.id is not None  # Auto-generated UUID
        assert rel.user_id == ""
        assert rel.source_id == ""
        assert rel.target_id == ""
        assert rel.relation_type == "related_to"
        assert rel.weight == 1.0
        assert rel.properties == {}
        assert rel.metadata == {}

    def test_relationship_creation_with_values(self, sample_relationship):
        """Test relationship creation with provided values."""
        assert sample_relationship.id == "rel-1"
        assert sample_relationship.user_id == "alice"
        assert sample_relationship.source_id == "entity-1"
        assert sample_relationship.target_id == "entity-2"
        assert sample_relationship.relation_type == "works_on"
        assert sample_relationship.weight == 0.9
        assert sample_relationship.properties["role"] == "lead"

    def test_relationship_to_dict(self, sample_relationship):
        """Test relationship serialization to dictionary."""
        data = sample_relationship.to_dict()

        assert data["id"] == "rel-1"
        assert data["source_id"] == "entity-1"
        assert data["target_id"] == "entity-2"
        assert data["relation_type"] == "works_on"
        assert data["weight"] == 0.9
        assert "created_at" in data

    def test_relationship_from_dict(self, sample_relationship):
        """Test relationship deserialization from dictionary."""
        data = sample_relationship.to_dict()
        restored = Relationship.from_dict(data)

        assert restored.id == sample_relationship.id
        assert restored.source_id == sample_relationship.source_id
        assert restored.target_id == sample_relationship.target_id
        assert restored.relation_type == sample_relationship.relation_type
        assert restored.weight == sample_relationship.weight

    def test_relationship_roundtrip(self, sample_relationship):
        """Test relationship serialization roundtrip."""
        data = sample_relationship.to_dict()
        restored = Relationship.from_dict(data)
        data2 = restored.to_dict()

        assert data == data2


class TestSubgraph:
    """Tests for Subgraph dataclass."""

    def test_subgraph_creation_empty(self):
        """Test empty subgraph creation."""
        subgraph = Subgraph()

        assert subgraph.entities == []
        assert subgraph.relationships == []
        assert subgraph.root_entity_ids == []

    def test_subgraph_entity_ids(self):
        """Test entity_ids property."""
        entities = [
            Entity(id="e1", name="Entity 1"),
            Entity(id="e2", name="Entity 2"),
            Entity(id="e3", name="Entity 3"),
        ]
        subgraph = Subgraph(entities=entities)

        assert subgraph.entity_ids == {"e1", "e2", "e3"}

    def test_subgraph_relationship_ids(self):
        """Test relationship_ids property."""
        relationships = [
            Relationship(id="r1", source_id="e1", target_id="e2"),
            Relationship(id="r2", source_id="e2", target_id="e3"),
        ]
        subgraph = Subgraph(relationships=relationships)

        assert subgraph.relationship_ids == {"r1", "r2"}

    def test_subgraph_get_entity(self):
        """Test get_entity method."""
        e1 = Entity(id="e1", name="Entity 1")
        e2 = Entity(id="e2", name="Entity 2")
        subgraph = Subgraph(entities=[e1, e2])

        assert subgraph.get_entity("e1") == e1
        assert subgraph.get_entity("e2") == e2
        assert subgraph.get_entity("e3") is None

    def test_subgraph_get_neighbors(self):
        """Test get_neighbors method."""
        e1 = Entity(id="e1", name="Entity 1")
        e2 = Entity(id="e2", name="Entity 2")
        e3 = Entity(id="e3", name="Entity 3")
        r1 = Relationship(id="r1", source_id="e1", target_id="e2")
        r2 = Relationship(id="r2", source_id="e1", target_id="e3")

        subgraph = Subgraph(entities=[e1, e2, e3], relationships=[r1, r2])

        neighbors = subgraph.get_neighbors("e1")
        neighbor_ids = {n.id for n in neighbors}
        assert neighbor_ids == {"e2", "e3"}

        # e2 only connected to e1
        neighbors_e2 = subgraph.get_neighbors("e2")
        assert len(neighbors_e2) == 1
        assert neighbors_e2[0].id == "e1"

    def test_subgraph_to_dict(self):
        """Test subgraph serialization."""
        e1 = Entity(id="e1", name="Entity 1", user_id="alice", entity_type="person")
        r1 = Relationship(id="r1", user_id="alice", source_id="e1", target_id="e2")

        subgraph = Subgraph(
            entities=[e1],
            relationships=[r1],
            root_entity_ids=["e1"],
        )

        data = subgraph.to_dict()
        assert len(data["entities"]) == 1
        assert len(data["relationships"]) == 1
        assert data["root_entity_ids"] == ["e1"]

    def test_subgraph_from_dict(self):
        """Test subgraph deserialization."""
        e1 = Entity(id="e1", name="Entity 1", user_id="alice", entity_type="person")
        r1 = Relationship(id="r1", user_id="alice", source_id="e1", target_id="e2")

        subgraph = Subgraph(
            entities=[e1],
            relationships=[r1],
            root_entity_ids=["e1"],
        )

        data = subgraph.to_dict()
        restored = Subgraph.from_dict(data)

        assert len(restored.entities) == 1
        assert restored.entities[0].id == "e1"
        assert len(restored.relationships) == 1
        assert restored.root_entity_ids == ["e1"]


class TestSubgraphToContext:
    """Tests for Subgraph.to_context() method (from ports.py)."""

    def test_to_context_empty_subgraph(self):
        """Test to_context with empty subgraph."""
        from headroom.memory.ports import Subgraph as PortsSubgraph

        subgraph = PortsSubgraph()
        context = subgraph.to_context()
        assert context == ""

    def test_to_context_entities_only(self):
        """Test to_context with only entities."""
        from headroom.memory.ports import Entity as PortsEntity
        from headroom.memory.ports import Subgraph as PortsSubgraph

        entities = [
            PortsEntity(id="e1", name="Alice", entity_type="person", user_id="user1"),
            PortsEntity(
                id="e2",
                name="Project X",
                entity_type="project",
                user_id="user1",
                metadata={"status": "active"},
            ),
        ]

        subgraph = PortsSubgraph(entities=entities)
        context = subgraph.to_context()

        assert "Entities:" in context
        assert "Alice (person)" in context
        assert "Project X (project)" in context
        assert "[status=active]" in context

    def test_to_context_with_relationships(self):
        """Test to_context with entities and relationships."""
        from headroom.memory.ports import Entity as PortsEntity
        from headroom.memory.ports import Relationship as PortsRel
        from headroom.memory.ports import Subgraph as PortsSubgraph

        entities = [
            PortsEntity(id="e1", name="Alice", entity_type="person", user_id="user1"),
            PortsEntity(id="e2", name="Project X", entity_type="project", user_id="user1"),
        ]

        relationships = [
            PortsRel(
                source_entity_id="e1",
                target_entity_id="e2",
                relation_type="works_on",
                user_id="user1",
            )
        ]

        subgraph = PortsSubgraph(entities=entities, relationships=relationships)
        context = subgraph.to_context()

        assert "Entities:" in context
        assert "Relationships:" in context
        assert "Alice --[works_on]--> Project X" in context

    def test_to_context_with_weighted_relationships(self):
        """Test to_context with weighted relationships."""
        from headroom.memory.ports import Entity as PortsEntity
        from headroom.memory.ports import Relationship as PortsRel
        from headroom.memory.ports import Subgraph as PortsSubgraph

        entities = [
            PortsEntity(id="e1", name="Alice", entity_type="person", user_id="user1"),
            PortsEntity(id="e2", name="Bob", entity_type="person", user_id="user1"),
        ]

        relationships = [
            PortsRel(
                source_entity_id="e1",
                target_entity_id="e2",
                relation_type="knows",
                weight=0.8,
                user_id="user1",
            )
        ]

        subgraph = PortsSubgraph(entities=entities, relationships=relationships)
        context = subgraph.to_context()

        assert "Alice --[knows]--> Bob" in context
        assert "(weight=0.8)" in context


# =============================================================================
# TestInMemoryGraphStore - Integration tests for graph store
# =============================================================================


class TestInMemoryGraphStore:
    """Tests for InMemoryGraphStore."""

    @pytest.mark.asyncio
    async def test_add_entity(self, graph_store, sample_entity):
        """Test adding an entity."""
        await graph_store.add_entity(sample_entity)

        retrieved = await graph_store.get_entity(sample_entity.id)
        assert retrieved is not None
        assert retrieved.id == sample_entity.id
        assert retrieved.name == sample_entity.name

    @pytest.mark.asyncio
    async def test_add_entity_replaces_existing(self, graph_store):
        """Test that adding an entity with same ID replaces it."""
        entity1 = Entity(id="e1", user_id="alice", name="Original")
        entity2 = Entity(id="e1", user_id="alice", name="Updated")

        await graph_store.add_entity(entity1)
        await graph_store.add_entity(entity2)

        retrieved = await graph_store.get_entity("e1")
        assert retrieved.name == "Updated"
        assert graph_store.entity_count == 1

    @pytest.mark.asyncio
    async def test_get_entity_not_found(self, graph_store):
        """Test getting a non-existent entity."""
        result = await graph_store.get_entity("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_entity(self, graph_store, sample_entity):
        """Test deleting an entity."""
        await graph_store.add_entity(sample_entity)

        deleted = await graph_store.delete_entity(sample_entity.id)
        assert deleted is True

        retrieved = await graph_store.get_entity(sample_entity.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_entity_not_found(self, graph_store):
        """Test deleting a non-existent entity."""
        deleted = await graph_store.delete_entity("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_delete_entity_cascades_relationships(self, graph_store):
        """Test that deleting an entity also deletes its relationships."""
        e1 = Entity(id="e1", user_id="alice", name="Entity 1")
        e2 = Entity(id="e2", user_id="alice", name="Entity 2")
        rel = Relationship(id="r1", user_id="alice", source_id="e1", target_id="e2")

        await graph_store.add_entity(e1)
        await graph_store.add_entity(e2)
        await graph_store.add_relationship(rel)

        assert graph_store.relationship_count == 1

        await graph_store.delete_entity("e1")

        assert graph_store.relationship_count == 0

    @pytest.mark.asyncio
    async def test_add_relationship(self, graph_store, sample_relationship):
        """Test adding a relationship."""
        # First add the entities
        e1 = Entity(id="entity-1", user_id="alice", name="Entity 1")
        e2 = Entity(id="entity-2", user_id="alice", name="Entity 2")
        await graph_store.add_entity(e1)
        await graph_store.add_entity(e2)

        await graph_store.add_relationship(sample_relationship)

        assert graph_store.relationship_count == 1

    @pytest.mark.asyncio
    async def test_get_relationships_outgoing(self, graph_store):
        """Test getting outgoing relationships."""
        e1 = Entity(id="e1", user_id="alice", name="Entity 1")
        e2 = Entity(id="e2", user_id="alice", name="Entity 2")
        e3 = Entity(id="e3", user_id="alice", name="Entity 3")
        r1 = Relationship(
            id="r1", user_id="alice", source_id="e1", target_id="e2", relation_type="a"
        )
        r2 = Relationship(
            id="r2", user_id="alice", source_id="e1", target_id="e3", relation_type="b"
        )
        r3 = Relationship(
            id="r3", user_id="alice", source_id="e2", target_id="e1", relation_type="c"
        )

        await graph_store.add_entity(e1)
        await graph_store.add_entity(e2)
        await graph_store.add_entity(e3)
        await graph_store.add_relationship(r1)
        await graph_store.add_relationship(r2)
        await graph_store.add_relationship(r3)

        # Get outgoing from e1
        outgoing = await graph_store.get_relationships(
            "e1", direction=RelationshipDirection.OUTGOING
        )
        assert len(outgoing) == 2

    @pytest.mark.asyncio
    async def test_get_relationships_incoming(self, graph_store):
        """Test getting incoming relationships."""
        e1 = Entity(id="e1", user_id="alice", name="Entity 1")
        e2 = Entity(id="e2", user_id="alice", name="Entity 2")
        r1 = Relationship(id="r1", user_id="alice", source_id="e2", target_id="e1")

        await graph_store.add_entity(e1)
        await graph_store.add_entity(e2)
        await graph_store.add_relationship(r1)

        incoming = await graph_store.get_relationships(
            "e1", direction=RelationshipDirection.INCOMING
        )
        assert len(incoming) == 1
        assert incoming[0].source_id == "e2"

    @pytest.mark.asyncio
    async def test_get_relationships_both(self, graph_store):
        """Test getting relationships in both directions."""
        e1 = Entity(id="e1", user_id="alice", name="Entity 1")
        e2 = Entity(id="e2", user_id="alice", name="Entity 2")
        e3 = Entity(id="e3", user_id="alice", name="Entity 3")
        r1 = Relationship(id="r1", user_id="alice", source_id="e1", target_id="e2")
        r2 = Relationship(id="r2", user_id="alice", source_id="e3", target_id="e1")

        await graph_store.add_entity(e1)
        await graph_store.add_entity(e2)
        await graph_store.add_entity(e3)
        await graph_store.add_relationship(r1)
        await graph_store.add_relationship(r2)

        both = await graph_store.get_relationships("e1", direction=RelationshipDirection.BOTH)
        assert len(both) == 2

    @pytest.mark.asyncio
    async def test_get_relationships_with_type_filter(self, graph_store):
        """Test filtering relationships by type."""
        e1 = Entity(id="e1", user_id="alice", name="Entity 1")
        e2 = Entity(id="e2", user_id="alice", name="Entity 2")
        e3 = Entity(id="e3", user_id="alice", name="Entity 3")
        r1 = Relationship(
            id="r1",
            user_id="alice",
            source_id="e1",
            target_id="e2",
            relation_type="works_with",
        )
        r2 = Relationship(
            id="r2",
            user_id="alice",
            source_id="e1",
            target_id="e3",
            relation_type="manages",
        )

        await graph_store.add_entity(e1)
        await graph_store.add_entity(e2)
        await graph_store.add_entity(e3)
        await graph_store.add_relationship(r1)
        await graph_store.add_relationship(r2)

        # Filter by type
        works_with = await graph_store.get_relationships("e1", relation_type="works_with")
        assert len(works_with) == 1
        assert works_with[0].target_id == "e2"

    @pytest.mark.asyncio
    async def test_delete_relationship(self, graph_store):
        """Test deleting a relationship."""
        e1 = Entity(id="e1", user_id="alice", name="Entity 1")
        e2 = Entity(id="e2", user_id="alice", name="Entity 2")
        r1 = Relationship(id="r1", user_id="alice", source_id="e1", target_id="e2")

        await graph_store.add_entity(e1)
        await graph_store.add_entity(e2)
        await graph_store.add_relationship(r1)

        deleted = await graph_store.delete_relationship("r1")
        assert deleted is True
        assert graph_store.relationship_count == 0

    @pytest.mark.asyncio
    async def test_delete_relationship_not_found(self, graph_store):
        """Test deleting a non-existent relationship."""
        deleted = await graph_store.delete_relationship("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_get_entity_by_name_case_insensitive(self, graph_store):
        """Test case-insensitive entity lookup by name."""
        entity = Entity(id="e1", user_id="alice", name="Project Alpha", entity_type="project")
        await graph_store.add_entity(entity)

        # Exact case
        result = await graph_store.get_entity_by_name("alice", "Project Alpha")
        assert result is not None
        assert result.id == "e1"

        # Lowercase
        result = await graph_store.get_entity_by_name("alice", "project alpha")
        assert result is not None
        assert result.id == "e1"

        # Uppercase
        result = await graph_store.get_entity_by_name("alice", "PROJECT ALPHA")
        assert result is not None
        assert result.id == "e1"

        # Mixed case
        result = await graph_store.get_entity_by_name("alice", "pRoJeCt AlPhA")
        assert result is not None
        assert result.id == "e1"

    @pytest.mark.asyncio
    async def test_get_entity_by_name_wrong_user(self, graph_store):
        """Test that entity lookup respects user scope."""
        entity = Entity(id="e1", user_id="alice", name="Project Alpha")
        await graph_store.add_entity(entity)

        result = await graph_store.get_entity_by_name("bob", "Project Alpha")
        assert result is None

    @pytest.mark.asyncio
    async def test_query_subgraph_single_hop(self, graph_store):
        """Test querying subgraph with single hop."""
        # Create a simple graph: e1 -> e2 -> e3
        e1 = Entity(id="e1", user_id="alice", name="Entity 1")
        e2 = Entity(id="e2", user_id="alice", name="Entity 2")
        e3 = Entity(id="e3", user_id="alice", name="Entity 3")
        r1 = Relationship(id="r1", user_id="alice", source_id="e1", target_id="e2")
        r2 = Relationship(id="r2", user_id="alice", source_id="e2", target_id="e3")

        await graph_store.add_entity(e1)
        await graph_store.add_entity(e2)
        await graph_store.add_entity(e3)
        await graph_store.add_relationship(r1)
        await graph_store.add_relationship(r2)

        # Query 1 hop from e1
        subgraph = await graph_store.query_subgraph(["e1"], max_hops=1)

        assert len(subgraph.entities) == 2  # e1 and e2
        assert len(subgraph.relationships) == 1  # r1 only
        assert subgraph.root_entity_ids == ["e1"]

    @pytest.mark.asyncio
    async def test_query_subgraph_multiple_hops(self, graph_store):
        """Test querying subgraph with multiple hops."""
        # Create a chain: e1 -> e2 -> e3 -> e4
        entities = [Entity(id=f"e{i}", user_id="alice", name=f"Entity {i}") for i in range(1, 5)]
        relationships = [
            Relationship(id=f"r{i}", user_id="alice", source_id=f"e{i}", target_id=f"e{i + 1}")
            for i in range(1, 4)
        ]

        for e in entities:
            await graph_store.add_entity(e)
        for r in relationships:
            await graph_store.add_relationship(r)

        # Query 2 hops from e1
        subgraph = await graph_store.query_subgraph(["e1"], max_hops=2)

        # Should include e1, e2, e3 (not e4)
        assert len(subgraph.entities) == 3
        entity_ids = {e.id for e in subgraph.entities}
        assert entity_ids == {"e1", "e2", "e3"}

    @pytest.mark.asyncio
    async def test_query_subgraph_with_relation_type_filter(self, graph_store):
        """Test querying subgraph with relation type filter."""
        e1 = Entity(id="e1", user_id="alice", name="Entity 1")
        e2 = Entity(id="e2", user_id="alice", name="Entity 2")
        e3 = Entity(id="e3", user_id="alice", name="Entity 3")
        r1 = Relationship(
            id="r1",
            user_id="alice",
            source_id="e1",
            target_id="e2",
            relation_type="works_with",
        )
        r2 = Relationship(
            id="r2",
            user_id="alice",
            source_id="e1",
            target_id="e3",
            relation_type="manages",
        )

        await graph_store.add_entity(e1)
        await graph_store.add_entity(e2)
        await graph_store.add_entity(e3)
        await graph_store.add_relationship(r1)
        await graph_store.add_relationship(r2)

        # Query only "works_with" relationships
        subgraph = await graph_store.query_subgraph(
            ["e1"], max_hops=1, relation_types=["works_with"]
        )

        assert len(subgraph.entities) == 2  # e1 and e2
        assert len(subgraph.relationships) == 1
        assert subgraph.relationships[0].relation_type == "works_with"

    @pytest.mark.asyncio
    async def test_query_subgraph_multiple_start_entities(self, graph_store):
        """Test querying subgraph from multiple starting entities."""
        e1 = Entity(id="e1", user_id="alice", name="Entity 1")
        e2 = Entity(id="e2", user_id="alice", name="Entity 2")
        e3 = Entity(id="e3", user_id="alice", name="Entity 3")
        e4 = Entity(id="e4", user_id="alice", name="Entity 4")
        r1 = Relationship(id="r1", user_id="alice", source_id="e1", target_id="e3")
        r2 = Relationship(id="r2", user_id="alice", source_id="e2", target_id="e4")

        await graph_store.add_entity(e1)
        await graph_store.add_entity(e2)
        await graph_store.add_entity(e3)
        await graph_store.add_entity(e4)
        await graph_store.add_relationship(r1)
        await graph_store.add_relationship(r2)

        # Query from both e1 and e2
        subgraph = await graph_store.query_subgraph(["e1", "e2"], max_hops=1)

        assert len(subgraph.entities) == 4
        assert len(subgraph.relationships) == 2

    @pytest.mark.asyncio
    async def test_find_path_direct_connection(self, graph_store):
        """Test finding a path between directly connected entities."""
        e1 = Entity(id="e1", user_id="alice", name="Entity 1")
        e2 = Entity(id="e2", user_id="alice", name="Entity 2")
        r1 = Relationship(id="r1", user_id="alice", source_id="e1", target_id="e2")

        await graph_store.add_entity(e1)
        await graph_store.add_entity(e2)
        await graph_store.add_relationship(r1)

        path = await graph_store.find_path("e1", "e2")
        assert path is not None
        assert path == ["e1", "e2"]

    @pytest.mark.asyncio
    async def test_find_path_multi_hop(self, graph_store):
        """Test finding a path through multiple hops."""
        # e1 -> e2 -> e3 -> e4
        entities = [Entity(id=f"e{i}", user_id="alice", name=f"Entity {i}") for i in range(1, 5)]
        relationships = [
            Relationship(id=f"r{i}", user_id="alice", source_id=f"e{i}", target_id=f"e{i + 1}")
            for i in range(1, 4)
        ]

        for e in entities:
            await graph_store.add_entity(e)
        for r in relationships:
            await graph_store.add_relationship(r)

        path = await graph_store.find_path("e1", "e4")
        assert path is not None
        assert path == ["e1", "e2", "e3", "e4"]

    @pytest.mark.asyncio
    async def test_find_path_no_path_exists(self, graph_store):
        """Test finding a path when no path exists."""
        e1 = Entity(id="e1", user_id="alice", name="Entity 1")
        e2 = Entity(id="e2", user_id="alice", name="Entity 2")

        await graph_store.add_entity(e1)
        await graph_store.add_entity(e2)

        path = await graph_store.find_path("e1", "e2")
        assert path is None

    @pytest.mark.asyncio
    async def test_find_path_same_entity(self, graph_store):
        """Test finding a path from an entity to itself."""
        e1 = Entity(id="e1", user_id="alice", name="Entity 1")
        await graph_store.add_entity(e1)

        path = await graph_store.find_path("e1", "e1")
        assert path == ["e1"]

    @pytest.mark.asyncio
    async def test_find_path_max_depth_exceeded(self, graph_store):
        """Test that find_path respects max_depth."""
        # Create a chain longer than max_depth
        entities = [Entity(id=f"e{i}", user_id="alice", name=f"Entity {i}") for i in range(1, 6)]
        relationships = [
            Relationship(id=f"r{i}", user_id="alice", source_id=f"e{i}", target_id=f"e{i + 1}")
            for i in range(1, 5)
        ]

        for e in entities:
            await graph_store.add_entity(e)
        for r in relationships:
            await graph_store.add_relationship(r)

        # Path e1 -> e5 requires 4 hops, but we limit to 2
        path = await graph_store.find_path("e1", "e5", max_depth=2)
        assert path is None

    @pytest.mark.asyncio
    async def test_clear_user(self, graph_store):
        """Test clearing all data for a user."""
        # Create data for alice
        e1 = Entity(id="e1", user_id="alice", name="Alice Entity 1")
        e2 = Entity(id="e2", user_id="alice", name="Alice Entity 2")
        r1 = Relationship(id="r1", user_id="alice", source_id="e1", target_id="e2")

        # Create data for bob
        e3 = Entity(id="e3", user_id="bob", name="Bob Entity")

        await graph_store.add_entity(e1)
        await graph_store.add_entity(e2)
        await graph_store.add_entity(e3)
        await graph_store.add_relationship(r1)

        # Clear alice's data
        entities_deleted, relationships_deleted = await graph_store.clear_user("alice")

        assert entities_deleted == 2
        assert relationships_deleted == 1

        # Bob's data should remain
        assert await graph_store.get_entity("e3") is not None
        assert await graph_store.get_entity("e1") is None

    @pytest.mark.asyncio
    async def test_get_entities_for_user(self, graph_store):
        """Test getting all entities for a user."""
        e1 = Entity(id="e1", user_id="alice", name="Entity 1")
        e2 = Entity(id="e2", user_id="alice", name="Entity 2")
        e3 = Entity(id="e3", user_id="bob", name="Entity 3")

        await graph_store.add_entity(e1)
        await graph_store.add_entity(e2)
        await graph_store.add_entity(e3)

        alice_entities = await graph_store.get_entities_for_user("alice")
        assert len(alice_entities) == 2

        bob_entities = await graph_store.get_entities_for_user("bob")
        assert len(bob_entities) == 1

    @pytest.mark.asyncio
    async def test_clear_all(self, graph_store):
        """Test clearing all data from the store."""
        e1 = Entity(id="e1", user_id="alice", name="Entity 1")
        e2 = Entity(id="e2", user_id="bob", name="Entity 2")
        r1 = Relationship(id="r1", user_id="alice", source_id="e1", target_id="e2")

        await graph_store.add_entity(e1)
        await graph_store.add_entity(e2)
        await graph_store.add_relationship(r1)

        await graph_store.clear()

        assert graph_store.entity_count == 0
        assert graph_store.relationship_count == 0

    def test_stats(self, graph_store):
        """Test stats method."""
        stats = graph_store.stats()

        assert "entity_count" in stats
        assert "relationship_count" in stats
        assert "users_count" in stats
        assert stats["entity_count"] == 0


# =============================================================================
# TestMemorySystem - Tests for the orchestrator
# =============================================================================


class MockBackend:
    """Mock backend for testing MemorySystem without real storage."""

    def __init__(self):
        self._memories: dict[str, Memory] = {}

    async def save_memory(
        self,
        content: str,
        user_id: str,
        importance: float,
        entities: list[str] | None = None,
        relationships: list[dict[str, str]] | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        # Pre-extraction fields for optimized storage (optional)
        facts: list[str] | None = None,
        extracted_entities: list[dict[str, str]] | None = None,
        extracted_relationships: list[dict[str, str]] | None = None,
        background: bool | None = None,
    ) -> Memory:
        memory = Memory(
            content=content,
            user_id=user_id,
            importance=importance,
            entity_refs=entities or [],
            session_id=session_id,
            metadata=metadata or {},
        )
        self._memories[memory.id] = memory
        return memory

    async def search_memories(
        self,
        query: str,
        user_id: str,
        entities: list[str] | None = None,
        include_related: bool = False,
        top_k: int = 10,
        session_id: str | None = None,
    ) -> list[MemorySearchResult]:
        results = []
        for memory in self._memories.values():
            if memory.user_id == user_id:
                results.append(
                    MemorySearchResult(
                        memory=memory,
                        score=0.9,
                    )
                )
        return results[:top_k]

    async def update_memory(
        self,
        memory_id: str,
        new_content: str,
        reason: str | None = None,
        user_id: str | None = None,
    ) -> Memory:
        if memory_id not in self._memories:
            raise ValueError(f"Memory not found: {memory_id}")

        old_memory = self._memories[memory_id]
        new_memory = Memory(
            content=new_content,
            user_id=old_memory.user_id,
            importance=old_memory.importance,
            supersedes=memory_id,
        )
        self._memories[new_memory.id] = new_memory
        old_memory.superseded_by = new_memory.id
        old_memory.valid_until = datetime.now(timezone.utc).replace(tzinfo=None)
        return new_memory

    async def delete_memory(
        self,
        memory_id: str,
        reason: str | None = None,
        user_id: str | None = None,
    ) -> bool:
        if memory_id in self._memories:
            del self._memories[memory_id]
            return True
        return False

    async def get_memory(self, memory_id: str) -> Memory | None:
        return self._memories.get(memory_id)

    @property
    def supports_graph(self) -> bool:
        return False

    @property
    def supports_vector_search(self) -> bool:
        return True


@pytest.fixture
def mock_backend():
    """Create a mock backend for testing."""
    return MockBackend()


@pytest.fixture
def memory_system(mock_backend):
    """Create a MemorySystem with mock backend."""
    return MemorySystem(mock_backend, user_id="alice", session_id="session-1")


class TestMemorySystem:
    """Tests for MemorySystem orchestrator."""

    @pytest.mark.asyncio
    async def test_handle_memory_save_valid(self, memory_system):
        """Test saving a memory with valid inputs."""
        result = await memory_system.handle_memory_save(
            content="User prefers Python",
            importance=0.8,
            entities=["Python"],
        )

        assert result["success"] is True
        assert "memory_id" in result
        assert result["content"] == "User prefers Python"
        assert result["importance"] == 0.8

    @pytest.mark.asyncio
    async def test_handle_memory_save_with_metadata(self, memory_system):
        """Test saving a memory with optional metadata."""
        result = await memory_system.handle_memory_save(
            content="Test content",
            importance=0.5,
            metadata={"source": "test", "context": "unit-test"},
        )

        assert result["success"] is True
        assert "memory_id" in result

    @pytest.mark.asyncio
    async def test_handle_memory_save_invalid_importance_high(self, memory_system):
        """Test saving a memory with importance > 1.0."""
        result = await memory_system.handle_memory_save(
            content="Test content",
            importance=1.5,
        )

        assert result["success"] is False
        assert "Invalid importance" in result["error"]

    @pytest.mark.asyncio
    async def test_handle_memory_save_invalid_importance_low(self, memory_system):
        """Test saving a memory with importance < 0.0."""
        result = await memory_system.handle_memory_save(
            content="Test content",
            importance=-0.1,
        )

        assert result["success"] is False
        assert "Invalid importance" in result["error"]

    @pytest.mark.asyncio
    async def test_handle_memory_search(self, memory_system, mock_backend):
        """Test searching memories."""
        # First save a memory
        await mock_backend.save_memory(
            content="User prefers Python",
            user_id="alice",
            importance=0.8,
        )

        result = await memory_system.handle_memory_search(
            query="programming preferences",
            top_k=5,
        )

        assert result["success"] is True
        assert result["count"] >= 1
        assert len(result["memories"]) >= 1

    @pytest.mark.asyncio
    async def test_handle_memory_search_empty_results(self, memory_system):
        """Test search with no matching memories."""
        result = await memory_system.handle_memory_search(
            query="nonexistent topic",
        )

        # Should succeed but with no results
        assert result["success"] is True
        assert result["count"] == 0
        assert result["memories"] == []

    @pytest.mark.asyncio
    async def test_handle_memory_update(self, memory_system, mock_backend):
        """Test updating a memory."""
        # First save a memory
        memory = await mock_backend.save_memory(
            content="User prefers Python",
            user_id="alice",
            importance=0.8,
        )

        result = await memory_system.handle_memory_update(
            memory_id=memory.id,
            new_content="User now prefers Rust",
            reason="Changed preference",
        )

        assert result["success"] is True
        assert result["old_content"] == "User prefers Python"
        assert result["new_content"] == "User now prefers Rust"

    @pytest.mark.asyncio
    async def test_handle_memory_update_not_found(self, memory_system):
        """Test updating a non-existent memory."""
        result = await memory_system.handle_memory_update(
            memory_id="nonexistent-id",
            new_content="New content",
            reason="Test update",
        )

        assert result["success"] is False
        assert "Memory not found" in result["error"]

    @pytest.mark.asyncio
    async def test_handle_memory_update_wrong_user(self, memory_system, mock_backend):
        """Test updating a memory belonging to another user."""
        # Save memory for bob
        memory = await mock_backend.save_memory(
            content="Bob's memory",
            user_id="bob",  # Different user
            importance=0.5,
        )

        # Try to update as alice
        result = await memory_system.handle_memory_update(
            memory_id=memory.id,
            new_content="Trying to update",
            reason="Test",
        )

        assert result["success"] is False
        assert "Permission denied" in result["error"]

    @pytest.mark.asyncio
    async def test_handle_memory_delete(self, memory_system, mock_backend):
        """Test deleting a memory."""
        memory = await mock_backend.save_memory(
            content="Memory to delete",
            user_id="alice",
            importance=0.5,
        )

        result = await memory_system.handle_memory_delete(
            memory_id=memory.id,
            reason="No longer needed",
        )

        assert result["success"] is True
        assert result["deleted_content"] == "Memory to delete"

    @pytest.mark.asyncio
    async def test_handle_memory_delete_not_found(self, memory_system):
        """Test deleting a non-existent memory."""
        result = await memory_system.handle_memory_delete(
            memory_id="nonexistent-id",
            reason="Test delete",
        )

        assert result["success"] is False
        assert "Memory not found" in result["error"]

    @pytest.mark.asyncio
    async def test_handle_memory_delete_wrong_user(self, memory_system, mock_backend):
        """Test deleting a memory belonging to another user."""
        memory = await mock_backend.save_memory(
            content="Bob's memory",
            user_id="bob",
            importance=0.5,
        )

        result = await memory_system.handle_memory_delete(
            memory_id=memory.id,
            reason="Test",
        )

        assert result["success"] is False
        assert "Permission denied" in result["error"]

    @pytest.mark.asyncio
    async def test_process_tool_call_memory_save(self, memory_system):
        """Test process_tool_call dispatcher for memory_save."""
        result = await memory_system.process_tool_call(
            "memory_save",
            {
                "content": "Test content",
                "importance": 0.5,
            },
        )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_process_tool_call_memory_search(self, memory_system):
        """Test process_tool_call dispatcher for memory_search."""
        result = await memory_system.process_tool_call(
            "memory_search",
            {"query": "test query"},
        )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_process_tool_call_unknown_tool(self, memory_system):
        """Test process_tool_call with unknown tool name."""
        result = await memory_system.process_tool_call(
            "unknown_tool",
            {},
        )

        assert result["success"] is False
        assert "Unknown tool" in result["error"]

    @pytest.mark.asyncio
    async def test_process_tool_call_handles_exception(self, memory_system):
        """Test that process_tool_call handles exceptions gracefully."""
        # Missing required argument should cause an exception
        result = await memory_system.process_tool_call(
            "memory_save",
            {
                # Missing required "content" field
                "importance": 0.5,
            },
        )

        assert result["success"] is False
        assert "error" in result

    def test_get_tools(self, memory_system):
        """Test that get_tools returns the tool definitions."""
        tools = memory_system.get_tools()

        assert isinstance(tools, list)
        assert len(tools) >= 4  # At least 4 memory tools

    def test_properties(self, memory_system):
        """Test MemorySystem properties."""
        assert memory_system.user_id == "alice"
        assert memory_system.session_id == "session-1"
        assert memory_system.supports_vector_search is True
        assert memory_system.supports_graph is False


# =============================================================================
# TestMemoryTools - Tests for tool definitions
# =============================================================================


class TestMemoryTools:
    """Tests for memory tool definitions."""

    def test_get_memory_tools_returns_list(self):
        """Test that get_memory_tools returns a list."""
        tools = get_memory_tools()
        assert isinstance(tools, list)
        assert len(tools) >= 4

    def test_get_memory_tools_returns_copy(self):
        """Test that get_memory_tools returns a copy, not the original."""
        tools1 = get_memory_tools()
        tools2 = get_memory_tools()
        assert tools1 is not tools2

    def test_get_tool_names(self):
        """Test that get_tool_names returns correct tool names."""
        names = get_tool_names()

        assert "memory_save" in names
        assert "memory_search" in names
        assert "memory_update" in names
        assert "memory_delete" in names

    def test_tools_have_openai_format(self):
        """Test that tools are in OpenAI function calling format."""
        for tool in MEMORY_TOOLS:
            assert "type" in tool
            assert tool["type"] == "function"
            assert "function" in tool

    def test_tools_have_required_fields(self):
        """Test that all tools have required fields."""
        for tool in MEMORY_TOOLS:
            func = tool["function"]

            assert "name" in func
            assert "description" in func
            assert "parameters" in func

            params = func["parameters"]
            assert "type" in params
            assert params["type"] == "object"
            assert "properties" in params

    def test_memory_save_tool_structure(self):
        """Test memory_save tool has correct structure."""
        save_tool = next(t for t in MEMORY_TOOLS if t["function"]["name"] == "memory_save")

        params = save_tool["function"]["parameters"]
        properties = params["properties"]

        # Check required fields exist (category is no longer part of the API)
        assert "content" in properties
        assert "importance" in properties
        assert "entities" in properties
        assert "relationships" in properties

        # Check required array
        assert "required" in params
        assert "content" in params["required"]
        assert "importance" in params["required"]

    def test_memory_search_tool_structure(self):
        """Test memory_search tool has correct structure."""
        search_tool = next(t for t in MEMORY_TOOLS if t["function"]["name"] == "memory_search")

        params = search_tool["function"]["parameters"]
        properties = params["properties"]

        assert "query" in properties
        assert "entities" in properties
        assert "top_k" in properties

        # Query is required
        assert "query" in params["required"]

    def test_memory_update_tool_structure(self):
        """Test memory_update tool has correct structure."""
        update_tool = next(t for t in MEMORY_TOOLS if t["function"]["name"] == "memory_update")

        params = update_tool["function"]["parameters"]
        properties = params["properties"]

        assert "memory_id" in properties
        assert "new_content" in properties
        assert "reason" in properties

        # Required fields for update (reason is optional)
        assert "memory_id" in params["required"]
        assert "new_content" in params["required"]

    def test_memory_delete_tool_structure(self):
        """Test memory_delete tool has correct structure."""
        delete_tool = next(t for t in MEMORY_TOOLS if t["function"]["name"] == "memory_delete")

        params = delete_tool["function"]["parameters"]
        properties = params["properties"]

        assert "memory_id" in properties
        assert "reason" in properties

        # Required fields for delete (reason is optional)
        assert "memory_id" in params["required"]

    def test_tool_descriptions_are_helpful(self):
        """Test that tool descriptions contain useful information."""
        for tool in MEMORY_TOOLS:
            description = tool["function"]["description"]

            # Description should be substantive
            assert len(description) > 50

            # Should contain guidance on when to use
            assert any(
                keyword in description.lower() for keyword in ["use", "when", "for", "example"]
            )

    def test_importance_parameter(self):
        """Test that importance parameter is defined correctly."""
        save_tool = next(t for t in MEMORY_TOOLS if t["function"]["name"] == "memory_save")

        importance_prop = save_tool["function"]["parameters"]["properties"]["importance"]
        # Importance should be a number type
        assert importance_prop["type"] == "number"


# =============================================================================
# TestLocalBackend - Integration tests
# =============================================================================


@pytest.mark.skipif(not HNSW_AVAILABLE, reason="hnswlib not available")
class TestLocalBackend:
    """Integration tests for LocalBackend."""

    @pytest.fixture
    def backend_config(self, temp_db_path):
        """Create backend config with temp database."""
        from headroom.memory.backends.local import LocalBackendConfig

        return LocalBackendConfig(
            db_path=str(temp_db_path),
            embedder_model="all-MiniLM-L6-v2",
            vector_dimension=384,
        )

    @pytest.fixture
    def backend(self, backend_config):
        """Create LocalBackend instance."""
        from headroom.memory.backends.local import LocalBackend

        return LocalBackend(backend_config)

    @pytest.mark.asyncio
    async def test_save_memory_basic(self, backend):
        """Test saving a basic memory."""
        memory = await backend.save_memory(
            content="User prefers Python",
            user_id="alice",
            importance=0.8,
        )

        assert memory is not None
        assert memory.content == "User prefers Python"
        assert memory.user_id == "alice"
        assert memory.importance == 0.8

    @pytest.mark.asyncio
    async def test_save_memory_with_entities(self, backend):
        """Test saving a memory with entities."""
        memory = await backend.save_memory(
            content="Alice works at Acme Corp",
            user_id="alice",
            importance=0.7,
            entities=["Alice", "Acme Corp"],
        )

        assert "Alice" in memory.entity_refs
        assert "Acme Corp" in memory.entity_refs

        # Verify entities were added to graph
        graph = await backend.get_graph()
        alice_entity = await graph.get_entity_by_name("alice", "Alice")
        acme_entity = await graph.get_entity_by_name("alice", "Acme Corp")

        assert alice_entity is not None
        assert acme_entity is not None

    @pytest.mark.asyncio
    async def test_save_memory_with_relationships(self, backend):
        """Test saving a memory with relationships."""
        await backend.save_memory(
            content="Alice works at Acme Corp",
            user_id="alice",
            importance=0.7,
            entities=["Alice", "Acme Corp"],
            relationships=[{"source": "Alice", "target": "Acme Corp", "type": "works_at"}],
        )

        # Verify relationship was created
        graph = await backend.get_graph()
        alice_entity = await graph.get_entity_by_name("alice", "Alice")

        rels = await graph.get_relationships(alice_entity.id)
        assert len(rels) >= 1

        works_at_rels = [r for r in rels if r.relation_type == "works_at"]
        assert len(works_at_rels) == 1

    @pytest.mark.asyncio
    async def test_search_memories_basic(self, backend):
        """Test basic memory search."""
        # Save some memories
        await backend.save_memory(
            content="User prefers Python for data science",
            user_id="alice",
            importance=0.8,
        )
        await backend.save_memory(
            content="User likes JavaScript for web development",
            user_id="alice",
            importance=0.7,
        )

        # Search
        results = await backend.search_memories(
            query="programming language preferences",
            user_id="alice",
            top_k=5,
        )

        assert len(results) >= 1
        assert all(isinstance(r, PortsMemorySearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_memories_with_entity_filter(self, backend):
        """Test memory search with entity filter."""
        await backend.save_memory(
            content="Alice manages the Python project",
            user_id="user1",
            importance=0.7,
            entities=["Alice", "Python project"],
        )
        await backend.save_memory(
            content="Bob works on the Java project",
            user_id="user1",
            importance=0.7,
            entities=["Bob", "Java project"],
        )

        # Search with entity filter
        results = await backend.search_memories(
            query="project work",
            user_id="user1",
            entities=["Alice"],
            top_k=10,
        )

        # Results should be related to Alice
        assert len(results) >= 1
        for r in results:
            assert any("alice" in entity.lower() for entity in r.memory.entity_refs) or any(
                "alice" in entity.lower() for entity in r.related_entities
            )

    @pytest.mark.asyncio
    async def test_search_memories_graph_expansion(self, backend):
        """Test that search includes graph-expanded results."""
        # Create memories with connected entities
        await backend.save_memory(
            content="Alice manages Project X",
            user_id="alice",
            importance=0.8,
            entities=["Alice", "Project X"],
            relationships=[{"source": "Alice", "target": "Project X", "type": "manages"}],
        )
        await backend.save_memory(
            content="Project X uses Python",
            user_id="alice",
            importance=0.7,
            entities=["Project X", "Python"],
            relationships=[{"source": "Project X", "target": "Python", "type": "uses"}],
        )

        # Search should potentially return related memories via graph
        results = await backend.search_memories(
            query="Alice's work",
            user_id="alice",
            include_related=True,
            top_k=10,
        )

        # Should have results (graph expansion may include connected memories)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_update_memory_creates_version_chain(self, backend):
        """Test that updating a memory creates a supersession chain."""
        # Save original memory
        original = await backend.save_memory(
            content="User prefers Python",
            user_id="alice",
            importance=0.8,
        )

        # Update it
        updated = await backend.update_memory(
            memory_id=original.id,
            new_content="User now prefers Rust",
        )

        # Check the chain
        assert updated.supersedes == original.id

        # Original should be marked as superseded
        retrieved_original = await backend.get_memory(original.id)
        assert retrieved_original.superseded_by == updated.id
        assert retrieved_original.is_current is False

    @pytest.mark.asyncio
    async def test_update_memory_not_found(self, backend):
        """Test updating a non-existent memory raises error."""
        with pytest.raises(ValueError, match="not found"):
            await backend.update_memory(
                memory_id="nonexistent-id",
                new_content="New content",
            )

    @pytest.mark.asyncio
    async def test_delete_memory(self, backend):
        """Test deleting a memory."""
        memory = await backend.save_memory(
            content="Memory to delete",
            user_id="alice",
            importance=0.5,
        )

        deleted = await backend.delete_memory(memory.id)
        assert deleted is True

        # Should be gone
        retrieved = await backend.get_memory(memory.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_memory_cleans_up_graph(self, backend):
        """Test that deleting a memory cleans up graph entities."""
        memory = await backend.save_memory(
            content="Alice works at Acme",
            user_id="alice",
            importance=0.7,
            entities=["Alice", "Acme"],
        )

        # Verify entities exist
        graph = await backend.get_graph()
        alice_before = await graph.get_entity_by_name("alice", "Alice")
        assert alice_before is not None

        # Delete memory
        await backend.delete_memory(memory.id)

        # Entities created from this memory should be deleted
        alice_after = await graph.get_entity_by_name("alice", "Alice")
        assert alice_after is None

    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self, backend):
        """Test deleting a non-existent memory returns False."""
        deleted = await backend.delete_memory("nonexistent-id")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_get_memory(self, backend):
        """Test getting a memory by ID."""
        memory = await backend.save_memory(
            content="Test memory",
            user_id="alice",
            importance=0.5,
        )

        retrieved = await backend.get_memory(memory.id)
        assert retrieved is not None
        assert retrieved.id == memory.id
        assert retrieved.content == memory.content

    @pytest.mark.asyncio
    async def test_get_memory_not_found(self, backend):
        """Test getting a non-existent memory returns None."""
        retrieved = await backend.get_memory("nonexistent-id")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_user_memories(self, backend):
        """Test getting all memories for a user."""
        # Save memories for alice
        await backend.save_memory(content="Alice memory 1", user_id="alice", importance=0.5)
        await backend.save_memory(content="Alice memory 2", user_id="alice", importance=0.5)

        # Save memory for bob
        await backend.save_memory(content="Bob memory", user_id="bob", importance=0.5)

        alice_memories = await backend.get_user_memories("alice")
        assert len(alice_memories) == 2

    @pytest.mark.asyncio
    async def test_clear_user(self, backend):
        """Test clearing all data for a user."""
        # Save memories with entities
        await backend.save_memory(
            content="Alice memory",
            user_id="alice",
            importance=0.5,
            entities=["Entity1"],
        )
        await backend.save_memory(
            content="Alice memory 2",
            user_id="alice",
            importance=0.5,
        )

        # Save for bob
        await backend.save_memory(content="Bob memory", user_id="bob", importance=0.5)

        count = await backend.clear_user("alice")
        assert count == 2

        # Alice's memories should be gone
        alice_memories = await backend.get_user_memories("alice")
        assert len(alice_memories) == 0

        # Bob's should remain
        bob_memories = await backend.get_user_memories("bob")
        assert len(bob_memories) == 1

    @pytest.mark.asyncio
    async def test_query_subgraph(self, backend):
        """Test querying subgraph from the backend."""
        # Create a graph structure
        await backend.save_memory(
            content="Alice manages Project X",
            user_id="alice",
            importance=0.8,
            entities=["Alice", "Project X"],
            relationships=[{"source": "Alice", "target": "Project X", "type": "manages"}],
        )

        subgraph = await backend.query_subgraph(
            entity_names=["Alice"],
            user_id="alice",
            max_hops=1,
        )

        assert len(subgraph.entities) >= 1
        entity_names = {e.name.lower() for e in subgraph.entities}
        assert "alice" in entity_names

    @pytest.mark.asyncio
    async def test_properties(self, backend):
        """Test backend property accessors."""
        # Force initialization
        await backend._ensure_initialized()

        assert backend.supports_graph is True
        assert backend.supports_vector_search is True
        assert backend.supports_text_search is True

    @pytest.mark.asyncio
    async def test_close(self, backend):
        """Test closing the backend."""
        await backend._ensure_initialized()
        assert backend._initialized is True

        await backend.close()

        assert backend._initialized is False
        assert backend._hierarchical_memory is None
        assert backend._graph is None


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_content_save(self):
        """Test saving memory with empty content."""
        backend = MockBackend()
        system = MemorySystem(backend, user_id="alice")

        # Empty content should still work (validation is content-level)
        result = await system.handle_memory_save(
            content="",
            importance=0.5,
        )

        # Should succeed (empty content is allowed)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_importance_boundary_values(self):
        """Test importance at exact boundary values."""
        backend = MockBackend()
        system = MemorySystem(backend, user_id="alice")

        # Exactly 0.0
        result = await system.handle_memory_save(content="Test", importance=0.0)
        assert result["success"] is True

        # Exactly 1.0
        result = await system.handle_memory_save(content="Test", importance=1.0)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_search_top_k_clamping(self):
        """Test that top_k is clamped to valid range."""
        backend = MockBackend()
        system = MemorySystem(backend, user_id="alice")

        # Very large top_k should be clamped
        result = await system.handle_memory_search(
            query="test",
            top_k=1000,  # Should be clamped to 50
        )
        assert result["success"] is True

        # Zero top_k should be clamped to 1
        result = await system.handle_memory_search(
            query="test",
            top_k=0,
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_graph_store_thread_safety(self, graph_store):
        """Test that graph store handles concurrent operations."""
        import asyncio

        # Create entities concurrently
        async def add_entity(i: int):
            entity = Entity(id=f"e{i}", user_id="alice", name=f"Entity {i}")
            await graph_store.add_entity(entity)

        await asyncio.gather(*[add_entity(i) for i in range(100)])

        assert graph_store.entity_count == 100

    @pytest.mark.asyncio
    async def test_subgraph_with_nonexistent_entities(self, graph_store):
        """Test querying subgraph with non-existent entity IDs."""
        subgraph = await graph_store.query_subgraph(
            ["nonexistent-1", "nonexistent-2"],
            max_hops=2,
        )

        # Should return empty subgraph
        assert len(subgraph.entities) == 0
        assert len(subgraph.relationships) == 0

    @pytest.mark.asyncio
    async def test_find_path_nonexistent_entities(self, graph_store):
        """Test finding path with non-existent entities."""
        path = await graph_store.find_path("nonexistent-1", "nonexistent-2")
        assert path is None

    def test_entity_properties_mutable(self):
        """Test that entity properties dict is mutable."""
        entity = Entity(id="e1", properties={"key": "value"})
        entity.properties["new_key"] = "new_value"

        assert "new_key" in entity.properties

    def test_relationship_metadata_mutable(self):
        """Test that relationship metadata dict is mutable."""
        rel = Relationship(id="r1", metadata={"key": "value"})
        rel.metadata["new_key"] = "new_value"

        assert "new_key" in rel.metadata
