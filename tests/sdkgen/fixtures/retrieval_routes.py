"""Source fixture: current CCR handler bodies, with proposed metadata added.

Bodies transcribed from server.py at c46e74d06b5ffa9643f420f28d2da6686fb7e487.
This is not a full proxy checkout; integration tests isolate these handler ASTs.
The full-source gate selects the real server.py when applied to Headroom.
"""

# ruff: noqa: F821 -- deliberately non-importable AST-only source fixture
from headroom.proxy.sdk_contracts import (
    RetrievalError,
    RetrieveRequest,
    RetrieveResponse,
    sdk_operation,
)

# Deliberately unsafe to import: the source compiler must NOT execute this module.
raise RuntimeError("Source fixtures are parsed, never imported")


def install_routes(app):
    @app.post(
        "/v1/retrieve",
        dependencies=[Depends(_require_loopback), Depends(_require_same_origin)],
    )
    @sdk_operation(
        operation_id="retrieve",
        request=RetrieveRequest,
        response=RetrieveResponse,
        access="loopback-same-origin",
        errors={400: RetrievalError, 404: RetrievalError},
    )
    async def ccr_retrieve(request: Request):
        data = await request.json()
        hash_key = data.get("hash")
        if not hash_key:
            raise HTTPException(status_code=400, detail="hash required")
        store = get_compression_store()
        entry_status = store.get_entry_status(hash_key, clean_expired=True)
        if entry_status["status"] != "available":
            raise HTTPException(
                status_code=404,
                detail=format_retrieval_miss_detail(entry_status),
            )
        entry = store.retrieve(hash_key)
        if entry:
            return {
                "hash": hash_key,
                "original_content": entry.original_content,
                "original_tokens": entry.original_tokens,
                "original_item_count": entry.original_item_count,
                "compressed_item_count": entry.compressed_item_count,
                "tool_name": entry.tool_name,
                "retrieval_count": entry.retrieval_count,
            }
        raise HTTPException(
            status_code=404,
            detail=format_retrieval_miss_detail(
                store.get_entry_status(hash_key, clean_expired=True)
            ),
        )

    @app.get("/v1/retrieve/{hash_key}", dependencies=[Depends(_require_loopback)])
    @sdk_operation(
        operation_id="retrieve_get",
        response=RetrieveResponse,
        access="loopback",
        errors={404: RetrievalError},
    )
    async def ccr_retrieve_get(hash_key: str):
        store = get_compression_store()
        entry_status = store.get_entry_status(hash_key, clean_expired=True)
        if entry_status["status"] != "available":
            raise HTTPException(
                status_code=404,
                detail=format_retrieval_miss_detail(entry_status),
            )
        entry = store.retrieve(hash_key)
        if entry:
            return {
                "hash": hash_key,
                "original_content": entry.original_content,
                "original_tokens": entry.original_tokens,
                "original_item_count": entry.original_item_count,
                "compressed_item_count": entry.compressed_item_count,
                "tool_name": entry.tool_name,
                "retrieval_count": entry.retrieval_count,
            }
        raise HTTPException(
            status_code=404,
            detail=format_retrieval_miss_detail(
                store.get_entry_status(hash_key, clean_expired=True)
            ),
        )
