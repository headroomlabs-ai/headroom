#!/usr/bin/env python3
"""Record standard parity fixtures for the CodeAwareCompressor only.

Installs the individual-grammar parser patch, then drives the Python
`CodeAwareCompressor` (enable_ccr=False, fallback_to_kompress=False) over
`_varied_code_inputs()` while `record_all()` has the `compress` method
patched, so only `tests/parity/fixtures/code_aware_compressor/` is
(re)written — no churn to other transforms' fixtures.

The grammar wheels must be installed at the versions the Rust crates pin
(same version number on PyPI + crates.io = same grammar source = identical
ASTs; verified by the grammar-parity canary):

    pip install tree-sitter==0.25.2 \\
        tree-sitter-python==0.25.0 tree-sitter-javascript==0.25.0 \\
        tree-sitter-typescript==0.23.2 tree-sitter-go==0.25.0 \\
        tree-sitter-rust==0.24.2 tree-sitter-java==0.23.5 \\
        tree-sitter-c==0.24.2 tree-sitter-cpp==0.23.4
    python scripts/record_code_compressor_fixtures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    from tests.parity.recorder import (
        _docstring_mode_inputs,
        _varied_code_inputs,
        install_individual_grammar_parsers,
        record_all,
    )

    statuses = record_all()
    if not statuses.get("code_aware_compressor", "").startswith("patched"):
        print(
            f"code_aware_compressor not patched: {statuses.get('code_aware_compressor')}",
            file=sys.stderr,
        )
        return 1

    install_individual_grammar_parsers()

    from headroom.transforms.code_compressor import (
        CodeAwareCompressor,
        CodeCompressorConfig,
        DocstringMode,
    )

    inputs = _varied_code_inputs()
    cac = CodeAwareCompressor(
        CodeCompressorConfig(enable_ccr=False, fallback_to_kompress=False)
    )
    for s in inputs:
        cac.compress(s)

    # Non-default docstring modes (FULL / REMOVE) over docstring-bearing
    # samples — distinct config hash → distinct fixtures.
    ds_inputs = _docstring_mode_inputs()
    extra = 0
    for mode in (DocstringMode.FULL, DocstringMode.REMOVE):
        c = CodeAwareCompressor(
            CodeCompressorConfig(
                enable_ccr=False, fallback_to_kompress=False, docstring_mode=mode
            )
        )
        for s in ds_inputs:
            c.compress(s)
            extra += 1

    out_dir = REPO / "tests" / "parity" / "fixtures" / "code_aware_compressor"
    n = len(list(out_dir.glob("*.json")))
    print(
        f"recorded {n} code_aware_compressor fixtures "
        f"from {len(inputs)} default + {extra} docstring-mode inputs -> {out_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
