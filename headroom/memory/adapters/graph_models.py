"""Graph data models for Headroom's knowledge graph memory system.

Provides Entity, Relationship, and Subgraph dataclasses for representing
structured knowledge as a graph.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class RelationshipDirection(Enum):
    """Direction for relationship queries."""

    OUTGOING = "outgoing"  # Relationships where entity is the source
    INCOMING = "incoming"  # Relationships where entity is the target
    BOTH = "both"  # Relationships in either direction


@dataclass
class Entity:
    """A node in the knowledge graph representing a named entity.

    Entities represent people, places, things, concepts, or any other
    discrete units of knowledge that can have relationships.

    Attributes:
        id: Unique identifier for the entity.
        user_id: User who owns this entity.
        name: Human-readable name of the entity.
        entity_type: Type/category of entity (e.g., "person", "project", "concept").
        description: Optional description of the entity.
        properties: Additional key-value properties.
        created_at: When the entity was created.
        updated_at: When the entity was last modified.
        metadata: Arbitrary metadata.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    name: str = ""
    entity_type: str = "unknown"
    description: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entity:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            name=data["name"],
            entity_type=data.get("entity_type", "unknown"),
            description=data.get("description"),
            properties=data.get("properties", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Relationship:
    """An edge in the knowledge graph connecting two entities.

    Relationships represent typed connections between entities, forming
    the structure of the knowledge graph.

    Attributes:
        id: Unique identifier for the relationship.
        user_id: User who owns this relationship.
        source_id: ID of the source entity.
        target_id: ID of the target entity.
        relation_type: Type of relationship (e.g., "works_with", "manages", "knows").
        weight: Optional weight/strength of the relationship (0.0 - 1.0).
        properties: Additional key-value properties.
        created_at: When the relationship was created.
        metadata: Arbitrary metadata.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    source_id: str = ""
    target_id: str = ""
    relation_type: str = "related_to"
    weight: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "weight": self.weight,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Relationship:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data.get("relation_type", "related_to"),
            weight=data.get("weight", 1.0),
            properties=data.get("properties", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Subgraph:
    """A subset of the knowledge graph containing entities and their relationships.

    Used to return query results that include connected entities and
    the relationships between them.

    Attributes:
        entities: List of entities in the subgraph.
        relationships: List of relationships connecting the entities.
        root_entity_ids: IDs of the entities that were the starting point of the query.
    """

    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    root_entity_ids: list[str] = field(default_factory=list)

    @property
    def entity_ids(self) -> set[str]:
        """Get all entity IDs in the subgraph."""
        return {e.id for e in self.entities}

    @property
    def relationship_ids(self) -> set[str]:
        """Get all relationship IDs in the subgraph."""
        return {r.id for r in self.relationships}

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID from this subgraph."""
        for entity in self.entities:
            if entity.id == entity_id:
                return entity
        return None

    def get_neighbors(self, entity_id: str) -> list[Entity]:
        """Get all entities directly connected to the given entity."""
        neighbor_ids: set[str] = set()
        for rel in self.relationships:
            if rel.source_id == entity_id:
                neighbor_ids.add(rel.target_id)
            elif rel.target_id == entity_id:
                neighbor_ids.add(rel.source_id)

        return [e for e in self.entities if e.id in neighbor_ids]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "root_entity_ids": self.root_entity_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Subgraph:
        """Create from dictionary."""
        return cls(
            entities=[Entity.from_dict(e) for e in data.get("entities", [])],
            relationships=[Relationship.from_dict(r) for r in data.get("relationships", [])],
            root_entity_ids=data.get("root_entity_ids", []),
        )
