# CCR Retrieve Literacy

Trust kept rows unless you have a concrete gap.

## Use `headroom_retrieve` when

- The user explicitly asks for raw, original, full, exact, or omitted content.
- You need to inspect the original payload for a specific follow-up the kept summary cannot answer.
- You need to inspect or quote a specific row, record, line, or file that was compressed away.

## Do not use `headroom_retrieve` when

- The kept summary already answers the question.
- The only reason to retrieve is to be thorough, careful, or to double-check.
- You can answer from the kept rows without looking at the full payload.

## Retrieval style

- Use `query` only as a note about the concrete gap you are checking.
- Current retrieval still returns the full original payload, even when `query` is present.
