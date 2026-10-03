#!/usr/bin/env python3
"""Record standard parity fixtures for the CodeAwareCompressor only.

Installs the individual-grammar parser patch, then drives the Python
`CodeAwareCompressor` (enable_ccr=False, fallback_to_kompress=False) over
`_varied_code_inputs()` while only that compressor is patched, so unrelated
transform fixtures are not rewritten.

The grammar wheels must be installed at the versions the Rust crates pin
(same version number on PyPI + crates.io selects the intended grammar source;
run the cross-runtime canary separately when Cargo is available):

    pip install tree-sitter==0.25.2 \\
        tree-sitter-python==0.25.0 tree-sitter-javascript==0.25.0 \\
        tree-sitter-typescript==0.23.2 tree-sitter-go==0.25.0 \\
        tree-sitter-rust==0.24.2 tree-sitter-java==0.23.5 \\
        tree-sitter-c==0.24.2 tree-sitter-cpp==0.23.4 tree-sitter-ruby==0.23.1
    python scripts/record_code_compressor_fixtures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    from tests.parity.recorder import (
        _varied_code_inputs,
        install_individual_grammar_parsers,
        record_code_aware,
    )

    status = record_code_aware()
    if not status.startswith("patched"):
        print(
            f"code_aware_compressor not patched: {status}",
            file=sys.stderr,
        )
        return 1

    install_individual_grammar_parsers()

    from headroom.transforms.code_compressor import (
        CodeAwareCompressor,
        CodeCompressorConfig,
    )

    inputs = [
        source
        for source in _varied_code_inputs()
        if any(marker in source for marker in ("require 'json'", "class Compact", "class Query"))
    ]
    cac = CodeAwareCompressor(CodeCompressorConfig(enable_ccr=False, fallback_to_kompress=False))
    for s in inputs:
        cac.compress(s)

    out_dir = REPO / "tests" / "parity" / "fixtures" / "code_aware_compressor"
    n = len(list(out_dir.glob("*.json")))
    print(
        f"recorded {n} code_aware_compressor fixtures from {len(inputs)} Ruby inputs -> {out_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
