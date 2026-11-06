# Embedding Server Sidecar (Option E)

## Prerequisites

`--embedding-server` requires the `memory` extra (provides `hnswlib` and
`sentence-transformers`):

```bash
# uv (recommended)
uv sync --extra memory

# pip
pip install headroom-ai[memory]
```

The sidecar detects a missing `hnswlib` at startup, logs
`event=embedding_server_missing_dep`, and exits with code 3. The watchdog
treats exit code 3 as a permanent error and does **not** retry — check the
log for the exact install command.

## Problem

The headroom proxy spawns 8 uvicorn worker processes (multiprocessing spawn mode).
Each worker currently loads:

- `OnnxLocalEmbedder`: ~86 MB ONNX model (all-MiniLM-L6-v2)
- `HNSWVectorIndex`: in-memory HNSW graph

This means:
- **~602 MB wasted** (7 extra copies of the 86 MB model)
- **Cross-worker HNSW fragmentation**: memories stored by worker A are never
  searched by worker B. The index diverges per-worker, so recall degrades.

## Solution

Run a single sidecar process that owns the embedder and HNSW index. All 8
workers connect to it over a Unix domain socket and delegate embed/search
operations.

## Architecture

```
 ┌─────────────────────────────────────────────────────┐
 │  headroom proxy (main process)                      │
 │  EmbeddingServerWatchdog                            │
 │    └── spawns + monitors EmbeddingServer process   │
 └───────────────┬─────────────────────────────────────┘
                 │  Unix socket  /tmp/headroom-embed-{port}.sock
     ┌───────────┴───────────────────────────────────┐
     │           EmbeddingServer process              │
     │  ┌─────────────────┐  ┌──────────────────┐   │
     │  │ OnnxLocalEmbedder│  │  HNSWVectorIndex  │   │
     │  │   (86 MB, 1x)   │  │  (shared truth)   │   │
     │  └─────────────────┘  └──────────────────┘   │
     └───────────────────────────────────────────────┘
              ^  ^  ^  ^  ^  ^  ^  ^
              |  |  |  |  |  |  |  |
    ┌─────────┴──┴──┴──┴──┴──┴──┴──┴────────────────┐
    │  8 uvicorn worker processes                     │
    │  RemoteEmbedder  RemoteVectorIndex              │
    │  (thin clients, no model loaded)                │
    └─────────────────────────────────────────────────┘
```

## How to Enable

```bash
headroom proxy --embedding-server --port 8787
```

Optional: specify a custom socket path:

```bash
headroom proxy --embedding-server \
  --embedding-server-socket /tmp/my-embed.sock
```

Default socket path: `/tmp/headroom-embed-{port}.sock`

Disable explicitly (default):

```bash
headroom proxy --no-embedding-server
```

## Socket Protocol

Length-prefixed JSON frames over a Unix domain socket:

```
[4 bytes: uint32 LE length] [length bytes: UTF-8 JSON body]
```

### Request format

```json
{"op": "embed", "id": "req-uuid", "text": "hello world"}
```

### Response format

```json
{"id": "req-uuid", "embedding": [0.01, -0.02, ...]}
```

### Operations

| op | Request fields | Response fields |
|----|---------------|----------------|
| `ping` | - | `status: "ok"` |
| `embed` | `text: str` | `embedding: list[float]` |
| `embed_batch` | `texts: list[str]` | `embeddings: list[list[float]]` |
| `search` | `query_embedding?: list[float]`, `query_text?: str`, `top_k: int`, `min_similarity: float`, `user_id?: str` | `results: list[{memory_id, similarity, rank, content, user_id}]` |
| `store` | `memory: {id, content, user_id, ...}`, `embedding?: list[float]` | `status: "stored"`, `memory_id: str` |
| `delete` | `memory_id: str` | `status: "deleted"/"not_found"` |
| `stats` | - | `{index_size, total_requests, total_embed_calls, avg_latency_ms, uptime_seconds}` |

Socket permissions are set to `0600` (owner read/write only).

## Watchdog Behavior

`EmbeddingServerWatchdog` monitors the server process:

- Polls liveness every 1s
- On unexpected crash: waits `restart_delay * 2^(n-1)` seconds (capped at 30s)
- After `max_restarts` consecutive crashes within 60s: logs
  `event=embedding_server_giving_up` and stops restarting
- Workers that cannot connect to the server raise `EmbeddingServerUnavailable`
  — memory features degrade gracefully (search/store disabled, proxy continues)

Default: `restart_delay=1s`, `max_restarts=10`

## Monitoring

### Stats endpoint

```bash
# Via the socket directly (using Python):
python3 -c "
import asyncio
from headroom.memory.adapters.remote import RemoteEmbedder
async def main():
    e = RemoteEmbedder('/tmp/headroom-embed-8787.sock')
    idx = await e._conn.send_request('stats')
    print(idx)
asyncio.run(main())
"
```

### Proxy /stats endpoint

The proxy's existing `/stats` endpoint shows memory backend health.
When `--embedding-server` is active, worker logs include:

```
event=using_remote_embedding_server socket=/tmp/headroom-embed-8787.sock
```

## Performance Characteristics

- Single embed latency: < 5ms p99 (local Unix socket + ONNX)
- Batch of 16: < 15ms p99
- Throughput: >= 500 embeds/sec on a modern laptop
- Micro-batching: embed requests are coalesced into batches over a 5ms window
  (batch inference is 3-4x faster per item than individual inference)

## Memory Savings

| Config | RSS |
|--------|-----|
| 8 workers, no sidecar | ~8 x 86 MB = 688 MB (model only) |
| 8 workers + sidecar | ~1 x 86 MB = 86 MB (model only) |
| **Savings** | **~602 MB** |

(Plus savings from consolidating 8 HNSW indexes into 1.)
