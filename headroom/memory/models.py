"""Hierarchical memory data models for Headroom."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


class ScopeLevel(Enum):
    """Memory scope hierarchy levels."""

    USER = "user"  # Persistent across all sessions
    SESSION = "session"  # Persistent within a task/conversation
    AGENT = "agent"  # Persistent within an agent's lifetime
    TURN = "turn"  # Ephemeral, single LLM call


@dataclass
class Memory:
    """A hierarchically-scoped memory with temporal awareness."""

    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""

    # Hierarchical Scoping (required: user_id, optional: narrower scopes)
    user_id: str = ""
    session_id: str | None = None
    agent_id: str | None = None
    turn_id: str | None = None

    # Temporal
    created_at: datetime = field(default_factory=datetime.utcnow)
    valid_from: datetime = field(default_factory=datetime.utcnow)
    valid_until: datetime | None = None  # None = current/active

    # Classification
    importance: float = 0.5  # 0.0 - 1.0

    # Lineage (for supersession and bubbling)
    supersedes: str | None = None  # ID of memory this replaced
    superseded_by: str | None = None  # ID of memory that replaced this
    promoted_from: str | None = None  # ID of child memory (if bubbled up)
    promotion_chain: list[str] = field(default_factory=list)

    # Access tracking
    access_count: int = 0
    last_accessed: datetime | None = None

    # Entity references
    entity_refs: list[str] = field(default_factory=list)

    # Embedding (for vector search)
    embedding: Any = None  # np.ndarray when numpy is available

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def scope_level(self) -> ScopeLevel:
        """Compute the scope level from hierarchy fields."""
        if self.turn_id is not None:
            return ScopeLevel.TURN
        if self.agent_id is not None:
            return ScopeLevel.AGENT
        if self.session_id is not None:
            return ScopeLevel.SESSION
        return ScopeLevel.USER

    @property
    def is_current(self) -> bool:
        """Check if this memory is current (not superseded)."""
        return self.valid_until is None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "turn_id": self.turn_id,
            "created_at": self.created_at.isoformat(),
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "importance": self.importance,
            "supersedes": self.supersedes,
            "superseded_by": self.superseded_by,
            "promoted_from": self.promoted_from,
            "promotion_chain": self.promotion_chain,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "entity_refs": self.entity_refs,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Memory:
        """Create from dictionary."""
        embedding = None
        if data.get("embedding") and np is not None:
            embedding = np.array(data["embedding"], dtype=np.float32)

        return cls(
            id=data["id"],
            content=data["content"],
            user_id=data["user_id"],
            session_id=data.get("session_id"),
            agent_id=data.get("agent_id"),
            turn_id=data.get("turn_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            valid_from=datetime.fromisoformat(data["valid_from"]),
            valid_until=datetime.fromisoformat(data["valid_until"])
            if data.get("valid_until")
            else None,
            importance=data["importance"],
            supersedes=data.get("supersedes"),
            superseded_by=data.get("superseded_by"),
            promoted_from=data.get("promoted_from"),
            promotion_chain=data.get("promotion_chain", []),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"])
            if data.get("last_accessed")
            else None,
            entity_refs=data.get("entity_refs", []),
            embedding=embedding,
            metadata=data.get("metadata", {}),
        )
