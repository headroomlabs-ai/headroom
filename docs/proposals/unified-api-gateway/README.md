# Unified API Gateway Contract

This directory retains the reviewed version-1 configuration contract and the
stable R01–R19 / T001–T100 acceptance ledgers used by the implementation.

The contract is intentionally fail-closed. Unknown configuration keys are
rejected, secrets are referenced rather than embedded, gateway v1 is local and
Python-authoritative, and unqualified native subscription identities remain
unavailable.

The source proposal is
`headroom-unified-proxy-proposal-2026-09-21.zip`, reviewed against Headroom
`94206e265203acfd72a3b939e9a964e29175ad50`. Passing these schema checks proves
contract consistency only; runtime and artifact qualification are separate gates.
