# CCR Retrieve Literacy

Trust kept rows unless you have a concrete gap.

## Use `headroom_retrieve` when

- The user explicitly asks for raw, original, full, exact, or omitted content.
- You have a targeted follow-up that the kept summary cannot answer.
- You need to inspect or quote a specific row, record, line, or file that was compressed away.

## Do not use `headroom_retrieve` when

- The kept summary already answers the question.
- The only reason to retrieve is to be thorough, careful, or to double-check.
- You can answer from the kept rows without looking at the full payload.

## Retrieval style

- Prefer `headroom_retrieve(hash, query=...)` for a focused gap.
- Omit `query` only when you truly need the full original payload.
