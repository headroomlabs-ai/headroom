"""Real-world evaluation tests for UniversalCompressor.

These tests simulate agent tool outputs with realistic content:
- JSON API responses (search results, user data)
- Code file reads (Python, JavaScript)
- Log outputs (application logs)
- Mixed content scenarios (agent processing multiple tools)

Evaluation metrics:
1. Structure Preservation: Keys, signatures, templates remain visible
2. Compression Ratio: Actual token reduction achieved
3. Retrievability: Can LLM identify what data exists for CCR retrieval?
"""

import json
from dataclasses import dataclass

import pytest

from headroom.compression.detector import ContentType
from headroom.compression.universal import (
    CompressionResult,
    UniversalCompressor,
    UniversalCompressorConfig,
)

# =============================================================================
# Realistic Test Fixtures - Simulated Tool Outputs
# =============================================================================

GITHUB_SEARCH_RESPONSE = json.dumps(
    {
        "total_count": 3,
        "incomplete_results": False,
        "items": [
            {
                "id": 12345678,
                "node_id": "MDEwOlJlcG9zaXRvcnkxMjM0NTY3OA==",
                "name": "headroom",
                "full_name": "anthropic/headroom",
                "private": False,
                "owner": {
                    "login": "anthropic",
                    "id": 98765,
                    "avatar_url": "https://avatars.githubusercontent.com/u/98765?v=4",
                    "type": "Organization",
                },
                "description": "Context optimization layer for LLM applications with intelligent compression and caching",
                "fork": False,
                "url": "https://api.github.com/repos/anthropic/headroom",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-06-20T15:45:00Z",
                "pushed_at": "2024-06-20T14:30:00Z",
                "stargazers_count": 1250,
                "watchers_count": 1250,
                "forks_count": 89,
                "language": "Python",
                "topics": ["llm", "compression", "caching", "context-management"],
                "default_branch": "main",
            },
            {
                "id": 23456789,
                "node_id": "MDEwOlJlcG9zaXRvcnkyMzQ1Njc4OQ==",
                "name": "llm-cache",
                "full_name": "openai/llm-cache",
                "private": False,
                "owner": {
                    "login": "openai",
                    "id": 87654,
                    "avatar_url": "https://avatars.githubusercontent.com/u/87654?v=4",
                    "type": "Organization",
                },
                "description": "High-performance caching layer for large language model responses with semantic similarity matching",
                "fork": False,
                "url": "https://api.github.com/repos/openai/llm-cache",
                "created_at": "2023-09-01T08:00:00Z",
                "updated_at": "2024-05-15T12:00:00Z",
                "pushed_at": "2024-05-15T11:30:00Z",
                "stargazers_count": 890,
                "watchers_count": 890,
                "forks_count": 67,
                "language": "Python",
                "topics": ["caching", "llm", "semantic-search"],
                "default_branch": "main",
            },
            {
                "id": 34567890,
                "node_id": "MDEwOlJlcG9zaXRvcnkzNDU2Nzg5MA==",
                "name": "prompt-optimizer",
                "full_name": "google/prompt-optimizer",
                "private": False,
                "owner": {
                    "login": "google",
                    "id": 76543,
                    "avatar_url": "https://avatars.githubusercontent.com/u/76543?v=4",
                    "type": "Organization",
                },
                "description": "Automatic prompt optimization using evolutionary algorithms and reinforcement learning from human feedback",
                "fork": False,
                "url": "https://api.github.com/repos/google/prompt-optimizer",
                "created_at": "2024-02-20T09:15:00Z",
                "updated_at": "2024-06-18T16:20:00Z",
                "pushed_at": "2024-06-18T16:00:00Z",
                "stargazers_count": 2100,
                "watchers_count": 2100,
                "forks_count": 156,
                "language": "Python",
                "topics": ["prompt-engineering", "optimization", "rlhf"],
                "default_branch": "main",
            },
        ],
    },
    indent=2,
)

PYTHON_FILE_CONTENT = '''"""Authentication middleware for FastAPI applications.

This module provides JWT-based authentication with role-based access control.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Any

import jwt
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Configuration constants
JWT_SECRET = "your-secret-key-here-replace-in-production"
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 60


class TokenPayload(BaseModel):
    """JWT token payload structure."""

    sub: str = Field(..., description="Subject (user ID)")
    exp: datetime = Field(..., description="Expiration time")
    roles: List[str] = Field(default_factory=list, description="User roles")
    permissions: List[str] = Field(default_factory=list, description="Specific permissions")


class UserContext(BaseModel):
    """Authenticated user context passed to endpoints."""

    user_id: str
    email: Optional[str] = None
    roles: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    is_admin: bool = False


security = HTTPBearer()


def create_access_token(
    user_id: str,
    roles: List[str] = None,
    permissions: List[str] = None,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a new JWT access token.

    Args:
        user_id: The user's unique identifier.
        roles: List of role names assigned to the user.
        permissions: List of specific permissions.
        expires_delta: Custom expiration time. Defaults to TOKEN_EXPIRE_MINUTES.

    Returns:
        Encoded JWT token string.

    Example:
        >>> token = create_access_token("user_123", roles=["admin", "editor"])
        >>> print(token)  # eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
    """
    if expires_delta:
        expire = datetime.now(timezone.utc).replace(tzinfo=None) + expires_delta
    else:
        expire = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub": user_id,
        "exp": expire,
        "roles": roles or [],
        "permissions": permissions or [],
        "iat": datetime.now(timezone.utc).replace(tzinfo=None),
    }

    encoded_jwt = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    logger.info(f"Created access token for user {user_id}, expires at {expire}")
    return encoded_jwt


def decode_token(token: str) -> TokenPayload:
    """Decode and validate a JWT token.

    Args:
        token: The JWT token string to decode.

    Returns:
        TokenPayload with decoded claims.

    Raises:
        HTTPException: If token is invalid, expired, or malformed.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return TokenPayload(**payload)
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError as e:
        logger.error(f"Token validation failed: {e}")
        raise HTTPException(status_code=401, detail="Could not validate credentials")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> UserContext:
    """FastAPI dependency to get the current authenticated user.

    Args:
        credentials: HTTP Bearer token from request header.

    Returns:
        UserContext with user information.

    Raises:
        HTTPException: If authentication fails.
    """
    token_data = decode_token(credentials.credentials)

    return UserContext(
        user_id=token_data.sub,
        roles=token_data.roles,
        permissions=token_data.permissions,
        is_admin="admin" in token_data.roles,
    )


def require_roles(*required_roles: str):
    """Decorator factory for role-based access control.

    Args:
        *required_roles: One or more role names required to access the endpoint.

    Returns:
        FastAPI dependency that validates user has required roles.

    Example:
        >>> @app.get("/admin/users")
        >>> async def list_users(user: UserContext = Depends(require_roles("admin"))):
        >>>     return {"users": [...]}
    """
    async def role_checker(user: UserContext = Depends(get_current_user)) -> UserContext:
        if not any(role in user.roles for role in required_roles):
            logger.warning(f"User {user.user_id} lacks required roles: {required_roles}")
            raise HTTPException(
                status_code=403,
                detail=f"Requires one of roles: {', '.join(required_roles)}"
            )
        return user
    return role_checker


def require_permissions(*required_permissions: str):
    """Decorator factory for permission-based access control.

    Args:
        *required_permissions: Permissions required to access the endpoint.

    Returns:
        FastAPI dependency that validates user has required permissions.
    """
    async def permission_checker(user: UserContext = Depends(get_current_user)) -> UserContext:
        missing = set(required_permissions) - set(user.permissions)
        if missing and not user.is_admin:  # Admins bypass permission checks
            logger.warning(f"User {user.user_id} missing permissions: {missing}")
            raise HTTPException(
                status_code=403,
                detail=f"Missing permissions: {', '.join(missing)}"
            )
        return user
    return permission_checker


class RateLimiter:
    """Simple in-memory rate limiter for API endpoints.

    Attributes:
        requests_per_minute: Maximum requests allowed per minute per user.
        window_size: Time window in seconds for rate limiting.
    """

    def __init__(self, requests_per_minute: int = 60, window_size: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_size = window_size
        self._requests: dict[str, list[datetime]] = {}

    def is_allowed(self, user_id: str) -> bool:
        """Check if a request from this user is allowed.

        Args:
            user_id: The user making the request.

        Returns:
            True if request is allowed, False if rate limited.
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cutoff = now - timedelta(seconds=self.window_size)

        # Clean old entries
        if user_id in self._requests:
            self._requests[user_id] = [
                t for t in self._requests[user_id] if t > cutoff
            ]
        else:
            self._requests[user_id] = []

        if len(self._requests[user_id]) >= self.requests_per_minute:
            return False

        self._requests[user_id].append(now)
        return True

    async def __call__(self, user: UserContext = Depends(get_current_user)) -> UserContext:
        """FastAPI dependency for rate limiting."""
        if not self.is_allowed(user.user_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        return user
'''

APPLICATION_LOGS = """2024-06-20 10:15:23.456 INFO  [main] com.example.app.Application - Starting Application v2.3.1 on server-prod-01 with PID 12345
2024-06-20 10:15:24.789 INFO  [main] com.example.app.config.DatabaseConfig - Initializing connection pool: host=db-primary.internal, port=5432, maxConnections=100
2024-06-20 10:15:25.123 DEBUG [main] com.example.app.config.DatabaseConfig - Connection pool parameters: minIdle=10, maxWait=30000ms, validationQuery=SELECT 1
2024-06-20 10:15:26.456 INFO  [main] com.example.app.config.CacheConfig - Redis cache configured: host=redis-cluster.internal:6379, cluster=true
2024-06-20 10:15:27.789 INFO  [main] com.example.app.Application - Application started successfully in 4.333 seconds
2024-06-20 10:15:30.123 INFO  [http-nio-8080-exec-1] com.example.app.controller.UserController - GET /api/v2/users?page=1&size=20 - 200 OK (45ms)
2024-06-20 10:15:31.456 INFO  [http-nio-8080-exec-2] com.example.app.controller.UserController - GET /api/v2/users/12345 - 200 OK (12ms)
2024-06-20 10:15:32.789 WARN  [http-nio-8080-exec-3] com.example.app.service.AuthService - Failed login attempt for user: john.doe@example.com from IP: 192.168.1.100
2024-06-20 10:15:33.012 WARN  [http-nio-8080-exec-3] com.example.app.service.AuthService - Account locked after 3 failed attempts: john.doe@example.com
2024-06-20 10:15:35.456 INFO  [http-nio-8080-exec-4] com.example.app.controller.OrderController - POST /api/v2/orders - 201 Created (156ms) - orderId=ORD-2024-00012345
2024-06-20 10:15:36.789 DEBUG [http-nio-8080-exec-4] com.example.app.service.PaymentService - Processing payment for order ORD-2024-00012345: amount=$149.99, method=CREDIT_CARD
2024-06-20 10:15:37.123 INFO  [http-nio-8080-exec-4] com.example.app.service.PaymentService - Payment successful: transactionId=TXN-abc123def456, status=COMPLETED
2024-06-20 10:15:40.456 ERROR [http-nio-8080-exec-5] com.example.app.service.InventoryService - Failed to update inventory for SKU-789012: Connection timeout to inventory-service.internal
2024-06-20 10:15:40.789 ERROR [http-nio-8080-exec-5] com.example.app.service.InventoryService - Stack trace: java.net.SocketTimeoutException: connect timed out
	at java.base/sun.nio.ch.NioSocketImpl.timedFinishConnect(NioSocketImpl.java:546)
	at java.base/sun.nio.ch.NioSocketImpl.connect(NioSocketImpl.java:597)
	at java.base/java.net.Socket.connect(Socket.java:633)
	at com.example.app.client.InventoryClient.updateStock(InventoryClient.java:87)
	at com.example.app.service.InventoryService.decrementStock(InventoryService.java:156)
2024-06-20 10:15:41.123 INFO  [scheduler-1] com.example.app.job.CleanupJob - Starting daily cleanup job: removing sessions older than 24 hours
2024-06-20 10:15:42.456 INFO  [scheduler-1] com.example.app.job.CleanupJob - Cleanup completed: removed 1,234 expired sessions, freed 45.6 MB
2024-06-20 10:15:45.789 INFO  [http-nio-8080-exec-6] com.example.app.controller.SearchController - GET /api/v2/search?q=wireless+headphones&category=electronics - 200 OK (234ms) - results=47
2024-06-20 10:15:50.123 INFO  [metrics-reporter] com.example.app.metrics.MetricsReporter - System metrics: cpu=45.2%, memory=67.8% (5.4GB/8GB), activeThreads=23, queuedRequests=0
"""

JAVASCRIPT_FILE_CONTENT = """/**
 * Real-time collaboration module using WebSocket connections.
 * Handles presence, cursors, and document synchronization.
 *
 * @module collaboration
 * @requires socket.io-client
 */

import { io, Socket } from 'socket.io-client';
import { EventEmitter } from 'events';

/** User presence information */
interface UserPresence {
  userId: string;
  displayName: string;
  color: string;
  cursor?: CursorPosition;
  lastSeen: Date;
  status: 'active' | 'idle' | 'away';
}

/** Cursor position in the document */
interface CursorPosition {
  line: number;
  column: number;
  selection?: {
    startLine: number;
    startColumn: number;
    endLine: number;
    endColumn: number;
  };
}

/** Document operation for conflict-free replication */
interface DocumentOperation {
  type: 'insert' | 'delete' | 'replace';
  position: number;
  content?: string;
  length?: number;
  timestamp: number;
  userId: string;
  vectorClock: Record<string, number>;
}

/** Configuration options for collaboration client */
interface CollaborationConfig {
  serverUrl: string;
  documentId: string;
  userId: string;
  displayName: string;
  reconnectAttempts?: number;
  heartbeatInterval?: number;
}

/**
 * Collaboration client for real-time document editing.
 *
 * @example
 * ```typescript
 * const collab = new CollaborationClient({
 *   serverUrl: 'wss://collab.example.com',
 *   documentId: 'doc-123',
 *   userId: 'user-456',
 *   displayName: 'Alice'
 * });
 *
 * collab.on('presence', (users) => console.log('Active users:', users));
 * collab.on('operation', (op) => applyOperation(op));
 *
 * await collab.connect();
 * ```
 */
export class CollaborationClient extends EventEmitter {
  private socket: Socket | null = null;
  private config: CollaborationConfig;
  private presence: Map<string, UserPresence> = new Map();
  private vectorClock: Record<string, number> = {};
  private pendingOperations: DocumentOperation[] = [];
  private reconnectCount = 0;
  private heartbeatTimer: NodeJS.Timer | null = null;

  constructor(config: CollaborationConfig) {
    super();
    this.config = {
      reconnectAttempts: 5,
      heartbeatInterval: 30000,
      ...config
    };
    this.vectorClock[config.userId] = 0;
  }

  /**
   * Connect to the collaboration server.
   *
   * @returns Promise that resolves when connected
   * @throws Error if connection fails after all retry attempts
   */
  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.socket = io(this.config.serverUrl, {
        auth: {
          documentId: this.config.documentId,
          userId: this.config.userId,
          displayName: this.config.displayName
        },
        reconnection: true,
        reconnectionAttempts: this.config.reconnectAttempts
      });

      this.socket.on('connect', () => {
        console.log(`[Collab] Connected to ${this.config.serverUrl}`);
        this.reconnectCount = 0;
        this.startHeartbeat();
        this.syncPendingOperations();
        resolve();
      });

      this.socket.on('disconnect', (reason) => {
        console.warn(`[Collab] Disconnected: ${reason}`);
        this.stopHeartbeat();
        this.emit('disconnected', reason);
      });

      this.socket.on('connect_error', (error) => {
        console.error(`[Collab] Connection error:`, error.message);
        this.reconnectCount++;
        if (this.reconnectCount >= (this.config.reconnectAttempts ?? 5)) {
          reject(new Error(`Failed to connect after ${this.reconnectCount} attempts`));
        }
      });

      this.setupEventHandlers();
    });
  }

  /**
   * Disconnect from the collaboration server.
   */
  disconnect(): void {
    this.stopHeartbeat();
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.presence.clear();
    console.log('[Collab] Disconnected');
  }

  /**
   * Send a document operation to other collaborators.
   *
   * @param type - Type of operation
   * @param position - Position in document
   * @param content - Content for insert/replace operations
   * @param length - Length for delete operations
   */
  sendOperation(
    type: 'insert' | 'delete' | 'replace',
    position: number,
    content?: string,
    length?: number
  ): void {
    this.vectorClock[this.config.userId]++;

    const operation: DocumentOperation = {
      type,
      position,
      content,
      length,
      timestamp: Date.now(),
      userId: this.config.userId,
      vectorClock: { ...this.vectorClock }
    };

    if (this.socket?.connected) {
      this.socket.emit('operation', operation);
    } else {
      this.pendingOperations.push(operation);
      console.warn('[Collab] Operation queued (offline)');
    }

    this.emit('localOperation', operation);
  }

  /**
   * Update cursor position for other collaborators.
   *
   * @param cursor - Current cursor position
   */
  updateCursor(cursor: CursorPosition): void {
    if (this.socket?.connected) {
      this.socket.emit('cursor', {
        userId: this.config.userId,
        cursor
      });
    }
  }

  /**
   * Get all active users in the document.
   *
   * @returns Array of user presence information
   */
  getActiveUsers(): UserPresence[] {
    return Array.from(this.presence.values()).filter(
      user => user.status !== 'away'
    );
  }

  private setupEventHandlers(): void {
    if (!this.socket) return;

    this.socket.on('presence:join', (user: UserPresence) => {
      console.log(`[Collab] ${user.displayName} joined`);
      this.presence.set(user.userId, user);
      this.emit('presence', this.getActiveUsers());
    });

    this.socket.on('presence:leave', (userId: string) => {
      const user = this.presence.get(userId);
      if (user) {
        console.log(`[Collab] ${user.displayName} left`);
        this.presence.delete(userId);
        this.emit('presence', this.getActiveUsers());
      }
    });

    this.socket.on('presence:update', (update: Partial<UserPresence> & { userId: string }) => {
      const existing = this.presence.get(update.userId);
      if (existing) {
        this.presence.set(update.userId, { ...existing, ...update });
        this.emit('presence', this.getActiveUsers());
      }
    });

    this.socket.on('cursor', (data: { userId: string; cursor: CursorPosition }) => {
      const user = this.presence.get(data.userId);
      if (user) {
        user.cursor = data.cursor;
        this.emit('cursor', data);
      }
    });

    this.socket.on('operation', (operation: DocumentOperation) => {
      // Update vector clock
      for (const [userId, clock] of Object.entries(operation.vectorClock)) {
        this.vectorClock[userId] = Math.max(
          this.vectorClock[userId] || 0,
          clock
        );
      }
      this.emit('operation', operation);
    });

    this.socket.on('sync', (state: { operations: DocumentOperation[]; presence: UserPresence[] }) => {
      console.log(`[Collab] Syncing ${state.operations.length} operations`);
      state.presence.forEach(user => this.presence.set(user.userId, user));
      state.operations.forEach(op => this.emit('operation', op));
      this.emit('synced');
    });
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.socket?.connected) {
        this.socket.emit('heartbeat', { userId: this.config.userId });
      }
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private syncPendingOperations(): void {
    if (this.pendingOperations.length > 0 && this.socket?.connected) {
      console.log(`[Collab] Syncing ${this.pendingOperations.length} pending operations`);
      this.pendingOperations.forEach(op => this.socket!.emit('operation', op));
      this.pendingOperations = [];
    }
  }
}

export default CollaborationClient;
"""


# =============================================================================
# Evaluation Metrics
# =============================================================================


@dataclass
class EvalMetrics:
    """Evaluation metrics for compression quality."""

    # Basic compression metrics
    original_length: int
    compressed_length: int
    compression_ratio: float
    tokens_before: int
    tokens_after: int

    # Structure preservation metrics
    keys_preserved: list[str]  # JSON keys found in compressed
    keys_missing: list[str]  # JSON keys not found in compressed
    key_preservation_ratio: float

    signatures_preserved: list[str]  # Function/class signatures found
    signatures_missing: list[str]  # Signatures not found
    signature_preservation_ratio: float

    # Content type detection
    detected_type: ContentType
    expected_type: ContentType
    detection_correct: bool

    def __str__(self) -> str:
        lines = [
            f"Compression: {self.compression_ratio:.1%} ({self.original_length} → {self.compressed_length} chars)",
            f"Tokens: {self.tokens_before} → {self.tokens_after} ({self.tokens_before - self.tokens_after} saved)",
            f"Keys: {self.key_preservation_ratio:.0%} ({len(self.keys_preserved)}/{len(self.keys_preserved) + len(self.keys_missing)})",
            f"Signatures: {self.signature_preservation_ratio:.0%} ({len(self.signatures_preserved)}/{len(self.signatures_preserved) + len(self.signatures_missing)})",
            f"Detection: {'✓' if self.detection_correct else '✗'} ({self.detected_type.name})",
        ]
        if self.keys_missing:
            lines.append(f"Missing keys: {self.keys_missing[:5]}...")
        if self.signatures_missing:
            lines.append(f"Missing signatures: {self.signatures_missing[:3]}...")
        return "\n".join(lines)


def evaluate_compression(
    result: CompressionResult,
    expected_keys: list[str] | None = None,
    expected_signatures: list[str] | None = None,
    expected_type: ContentType | None = None,
) -> EvalMetrics:
    """Evaluate compression quality.

    Args:
        result: Compression result to evaluate.
        expected_keys: JSON keys that should be preserved.
        expected_signatures: Function/class signatures that should be preserved.
        expected_type: Expected content type.

    Returns:
        EvalMetrics with detailed evaluation.
    """
    expected_keys = expected_keys or []
    expected_signatures = expected_signatures or []

    # Check key preservation
    keys_preserved = [k for k in expected_keys if k in result.compressed]
    keys_missing = [k for k in expected_keys if k not in result.compressed]

    # Check signature preservation
    signatures_preserved = [s for s in expected_signatures if s in result.compressed]
    signatures_missing = [s for s in expected_signatures if s not in result.compressed]

    return EvalMetrics(
        original_length=len(result.original),
        compressed_length=len(result.compressed),
        compression_ratio=result.compression_ratio,
        tokens_before=result.tokens_before,
        tokens_after=result.tokens_after,
        keys_preserved=keys_preserved,
        keys_missing=keys_missing,
        key_preservation_ratio=len(keys_preserved) / len(expected_keys) if expected_keys else 1.0,
        signatures_preserved=signatures_preserved,
        signatures_missing=signatures_missing,
        signature_preservation_ratio=len(signatures_preserved) / len(expected_signatures)
        if expected_signatures
        else 1.0,
        detected_type=result.content_type,
        expected_type=expected_type or result.content_type,
        detection_correct=expected_type is None or result.content_type == expected_type,
    )


# =============================================================================
# Evaluation Tests
# =============================================================================


class TestJSONAPIResponseEval:
    """Evaluate compression of JSON API responses."""

    @pytest.fixture
    def compressor(self):
        """Create compressor with simple compression (no Kompress for tests)."""
        config = UniversalCompressorConfig(
            use_magika=False,  # Use fallback for consistent tests
            use_kompress=False,
            ccr_enabled=False,
        )
        return UniversalCompressor(config=config)

    def test_github_search_structure_preservation(self, compressor):
        """Test that GitHub search response keys are preserved."""
        result = compressor.compress(GITHUB_SEARCH_RESPONSE)

        # Expected keys that must remain visible for LLM to know what to ask for
        expected_keys = [
            "total_count",
            "items",
            "id",
            "name",
            "full_name",
            "owner",
            "login",
            "description",
            "stargazers_count",
            "forks_count",
            "language",
            "topics",
            "default_branch",
        ]

        metrics = evaluate_compression(
            result,
            expected_keys=expected_keys,
            expected_type=ContentType.JSON,
        )

        print(f"\n{metrics}")

        # Critical assertion: all keys must be preserved
        assert metrics.key_preservation_ratio == 1.0, f"Missing keys: {metrics.keys_missing}"

        # Should achieve some compression on the long descriptions
        assert metrics.compression_ratio < 1.0, "Should compress the response"

    def test_nested_user_data_preservation(self, compressor):
        """Test preservation of nested user data structure."""
        user_data = json.dumps(
            {
                "user": {
                    "id": "usr_8f14e45f-ceea-4123-8f14-e45fceea4123",
                    "profile": {
                        "displayName": "Alice Smith",
                        "email": "alice@example.com",
                        "bio": "Software engineer passionate about distributed systems and machine learning. "
                        * 10,
                        "avatar": "https://avatars.example.com/u/12345?v=4",
                    },
                    "preferences": {
                        "theme": "dark",
                        "language": "en-US",
                        "notifications": {
                            "email": True,
                            "push": False,
                            "sms": False,
                        },
                    },
                    "stats": {
                        "repositories": 42,
                        "contributions": 1250,
                        "followers": 89,
                        "following": 23,
                    },
                },
                "metadata": {
                    "requestId": "req_abc123def456",
                    "timestamp": "2024-06-20T15:30:00Z",
                    "version": "v2",
                },
            },
            indent=2,
        )

        result = compressor.compress(user_data)

        expected_keys = [
            "user",
            "id",
            "profile",
            "displayName",
            "email",
            "bio",
            "avatar",
            "preferences",
            "theme",
            "language",
            "notifications",
            "stats",
            "repositories",
            "contributions",
            "followers",
            "metadata",
            "requestId",
            "timestamp",
            "version",
        ]

        metrics = evaluate_compression(
            result,
            expected_keys=expected_keys,
            expected_type=ContentType.JSON,
        )

        print(f"\n{metrics}")

        assert metrics.key_preservation_ratio == 1.0, f"Missing keys: {metrics.keys_missing}"

        # UUID should be preserved (high entropy)
        assert (
            "8f14e45f-ceea-4123-8f14-e45fceea4123" in result.compressed
            or "usr_8f14e45f" in result.compressed
        ), "UUID should be preserved due to high entropy"


class TestCodeFileEval:
    """Evaluate compression of code files."""

    @pytest.fixture
    def compressor(self):
        """Create compressor."""
        config = UniversalCompressorConfig(
            use_magika=False,
            use_kompress=False,
            ccr_enabled=False,
        )
        return UniversalCompressor(config=config)

    def test_python_signatures_preserved(self, compressor):
        """Test that Python function/class signatures are preserved."""
        result = compressor.compress(PYTHON_FILE_CONTENT)

        expected_signatures = [
            "class TokenPayload",
            "class UserContext",
            "class RateLimiter",
            "def create_access_token",
            "def decode_token",
            "async def get_current_user",
            "def require_roles",
            "def require_permissions",
            "def is_allowed",
        ]

        metrics = evaluate_compression(
            result,
            expected_signatures=expected_signatures,
            expected_type=ContentType.CODE,
        )

        print(f"\n{metrics}")

        # At least 80% of signatures should be preserved
        assert metrics.signature_preservation_ratio >= 0.8, (
            f"Missing signatures: {metrics.signatures_missing}"
        )

    def test_javascript_signatures_preserved(self, compressor):
        """Test that JavaScript/TypeScript signatures are preserved."""
        result = compressor.compress(JAVASCRIPT_FILE_CONTENT)

        expected_signatures = [
            "interface UserPresence",
            "interface CursorPosition",
            "interface DocumentOperation",
            "interface CollaborationConfig",
            "class CollaborationClient",
            "async connect()",
            "disconnect()",
            "sendOperation(",
            "updateCursor(",
            "getActiveUsers()",
        ]

        metrics = evaluate_compression(
            result,
            expected_signatures=expected_signatures,
            expected_type=ContentType.CODE,
        )

        print(f"\n{metrics}")

        # At least 60% of signatures should be preserved
        # (methods inside class bodies may be compressed, which is expected)
        assert metrics.signature_preservation_ratio >= 0.6, (
            f"Missing signatures: {metrics.signatures_missing}"
        )


class TestLogOutputEval:
    """Evaluate compression of log outputs.

    NOTE: Without a specialized LogHandler, logs are handled by NoOpHandler
    which marks everything compressible. This results in aggressive compression.
    A future LogHandler should preserve:
    - Log levels (INFO, WARN, ERROR)
    - Timestamps
    - Component names
    - Key identifiers (order IDs, transaction IDs)
    - Error types and stack traces
    """

    @pytest.fixture
    def compressor(self):
        """Create compressor."""
        config = UniversalCompressorConfig(
            use_magika=False,
            use_kompress=False,
            ccr_enabled=False,
        )
        return UniversalCompressor(config=config)

    def test_log_compression_report(self, compressor):
        """Report log compression behavior (no specialized handler yet)."""
        result = compressor.compress(APPLICATION_LOGS)

        # Key information we'd like to preserve
        desired_content = [
            "INFO",
            "WARN",
            "ERROR",
            "DEBUG",  # Log levels
            "Application",
            "DatabaseConfig",
            "CacheConfig",  # Components
            "Starting",
            "Initializing",
            "Failed",  # Key actions
            "ORD-2024-00012345",  # Order ID
            "TXN-abc123def456",  # Transaction ID
            "SocketTimeoutException",  # Error type
        ]

        preserved = [c for c in desired_content if c in result.compressed]
        missing = [c for c in desired_content if c not in result.compressed]

        print(f"\nLog compression: {result.compression_ratio:.1%}")
        print(f"Preserved: {len(preserved)}/{len(desired_content)}")
        print(f"  Found: {preserved}")
        if missing:
            print(f"  Missing (needs LogHandler): {missing}")

        # Basic assertion: some compression happened
        assert result.compression_ratio < 1.0, "Should compress logs"

        # Note: with a LogHandler, we'd assert 70%+ preservation
        # For now, just verify content type detection
        assert result.content_type in [ContentType.TEXT, ContentType.LOG], (
            f"Should detect as TEXT or LOG, got {result.content_type}"
        )


class TestMultiToolAgentScenario:
    """Test realistic multi-tool agent scenario.

    Simulates an agent that:
    1. Searches GitHub for repositories
    2. Reads a code file
    3. Checks application logs

    All tool outputs should be compressible while preserving
    key structure that lets the LLM know what's available.
    """

    @pytest.fixture
    def compressor(self):
        """Create compressor."""
        config = UniversalCompressorConfig(
            use_magika=False,
            use_kompress=False,
            ccr_enabled=False,
        )
        return UniversalCompressor(config=config)

    def test_batch_compress_mixed_content(self, compressor):
        """Test batch compression of mixed tool outputs."""
        tool_outputs = [
            GITHUB_SEARCH_RESPONSE,
            PYTHON_FILE_CONTENT,
            APPLICATION_LOGS,
        ]

        results = compressor.compress_batch(tool_outputs)

        assert len(results) == 3

        # Check content type detection
        assert results[0].content_type == ContentType.JSON
        assert results[1].content_type == ContentType.CODE
        # Logs may be detected as TEXT or LOG depending on detector
        assert results[2].content_type in [ContentType.TEXT, ContentType.LOG, ContentType.CODE]

        # All should achieve some compression
        total_original = sum(r.tokens_before for r in results)
        total_compressed = sum(r.tokens_after for r in results)

        print("\nBatch compression results:")
        print(f"  Total tokens: {total_original} → {total_compressed}")
        print(
            f"  Savings: {total_original - total_compressed} tokens ({(1 - total_compressed / total_original):.1%})"
        )

        for result in results:
            print(f"  [{result.content_type.name}] {result.compression_ratio:.1%} compression")

    def test_agent_context_window_simulation(self, compressor):
        """Simulate agent compressing tool outputs to fit context window.

        Scenario: Agent has 8K token limit, tools returned ~4K tokens.
        After compression, should fit comfortably with room for reasoning.
        """
        # Combine all tool outputs (simulating agent context)
        combined_context = f"""
## Tool Output 1: GitHub Search Results
```json
{GITHUB_SEARCH_RESPONSE}
```

## Tool Output 2: File Content (auth.py)
```python
{PYTHON_FILE_CONTENT}
```

## Tool Output 3: Application Logs
```
{APPLICATION_LOGS}
```
"""

        # Estimate original tokens (~4 chars per token)
        original_tokens = len(combined_context) // 4
        print(f"\nOriginal context: ~{original_tokens} tokens")

        # Compress each section
        results = compressor.compress_batch(
            [
                GITHUB_SEARCH_RESPONSE,
                PYTHON_FILE_CONTENT,
                APPLICATION_LOGS,
            ]
        )

        total_compressed_tokens = sum(r.tokens_after for r in results)

        print(f"After compression: ~{total_compressed_tokens} tokens")
        print(f"Savings: ~{original_tokens - total_compressed_tokens} tokens")

        # Verify key information is still accessible
        # JSON: Can see repository names and keys
        assert "headroom" in results[0].compressed
        assert "stargazers_count" in results[0].compressed

        # Code: Can see function signatures
        assert "create_access_token" in results[1].compressed

        # Logs: Can see error indicators
        # (may vary based on compression)


class TestCompressionQualityMetrics:
    """Test overall compression quality metrics."""

    @pytest.fixture
    def compressor(self):
        """Create compressor."""
        config = UniversalCompressorConfig(
            use_magika=False,
            use_kompress=False,
            ccr_enabled=False,
            compression_ratio_target=0.3,  # Target 70% reduction
        )
        return UniversalCompressor(config=config)

    def test_compression_preserves_retrievability(self, compressor):
        """Test that compressed content still allows CCR retrieval.

        Key insight: The compressed content should contain enough
        structure that an LLM can identify what data exists and
        request specific items via CCR.
        """
        # Large JSON with many items
        data = {
            "results": [
                {
                    "id": f"item_{i}",
                    "title": f"Product {i}",
                    "description": f"This is a detailed description for product {i}. " * 5,
                    "price": 19.99 + i,
                    "category": ["electronics", "gadgets", "accessories"][i % 3],
                    "tags": [f"tag_{j}" for j in range(5)],
                }
                for i in range(20)
            ],
            "pagination": {
                "page": 1,
                "total_pages": 10,
                "total_items": 200,
            },
        }
        content = json.dumps(data, indent=2)

        result = compressor.compress(content)

        # Check that schema is discoverable
        # LLM should be able to see: results array with items having id, title, etc.
        discoverable_keys = ["results", "id", "title", "price", "category", "pagination"]

        discovered = [k for k in discoverable_keys if k in result.compressed]

        print("\nRetrievability test:")
        print(f"  Original: {len(content)} chars, ~{len(content) // 4} tokens")
        print(
            f"  Compressed: {len(result.compressed)} chars, ~{len(result.compressed) // 4} tokens"
        )
        print(f"  Discoverable keys: {len(discovered)}/{len(discoverable_keys)}")

        # All structural keys must be discoverable
        assert len(discovered) == len(discoverable_keys), (
            f"Missing keys for CCR discovery: {set(discoverable_keys) - set(discovered)}"
        )

    def test_summary_report(self, compressor):
        """Generate summary report of compression quality across all test fixtures."""
        test_cases = [
            ("GitHub API Response", GITHUB_SEARCH_RESPONSE, ContentType.JSON),
            ("Python Auth Module", PYTHON_FILE_CONTENT, ContentType.CODE),
            ("JavaScript Collab Module", JAVASCRIPT_FILE_CONTENT, ContentType.CODE),
            ("Application Logs", APPLICATION_LOGS, ContentType.TEXT),
        ]

        print("\n" + "=" * 60)
        print("COMPRESSION QUALITY SUMMARY REPORT")
        print("=" * 60)

        total_original = 0
        total_compressed = 0

        for name, content, expected_type in test_cases:
            result = compressor.compress(content)
            total_original += result.tokens_before
            total_compressed += result.tokens_after

            print(f"\n{name}:")
            print(f"  Type: {result.content_type.name} (expected: {expected_type.name})")
            print(f"  Compression: {result.compression_ratio:.1%}")
            print(f"  Tokens: {result.tokens_before} → {result.tokens_after}")
            print(f"  Handler: {result.handler_used}")

        print("\n" + "-" * 60)
        print(f"TOTAL: {total_original} → {total_compressed} tokens")
        print(
            f"OVERALL SAVINGS: {total_original - total_compressed} tokens ({(1 - total_compressed / total_original):.1%})"
        )
        print("=" * 60)
