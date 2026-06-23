#!/usr/bin/env python3
"""Demo / test harness for tabular + spreadsheet compression.

Generates representative sample data and runs it through Headroom's tabular
compressor so you can see where it helps (verbose / redundant tables, and
query-driven selection) and where it correctly does nothing (compact, all-unique
data with no signal to compress against).

Usage:
    python examples/tabular_compression_demo.py            # run all scenarios
    python examples/tabular_compression_demo.py --write DIR # also save sample files

The .xlsx scenario requires the spreadsheet extra:
    pip install headroom-ai[spreadsheet]
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import headroom
from headroom.transforms.content_router import ContentRouter

_HAS_OPENPYXL = importlib.util.find_spec("openpyxl") is not None


# ─── Sample data generators ─────────────────────────────────────────────────


def compact_unique_csv(rows: int = 60) -> str:
    """Minimal CSV, every row unique — nothing safely removable (~0 savings)."""
    lines = ["id,name,age,city"]
    lines += [f"{i},user_{i},{20 + i % 50},city_{i}" for i in range(rows)]
    return "\n".join(lines)


def redundant_csv(rows: int = 120) -> str:
    """Highly repetitive rows — SmartCrusher can dedupe (big savings)."""
    lines = ["region,product,status"]
    lines += ["EMEA,widget-A,shipped" for _ in range(rows)]
    return "\n".join(lines)


def verbose_markdown(rows: int = 40) -> str:
    """A padded markdown table — verbose source, lossless compaction wins."""
    header = "| name | age | city | status | dept |\n| --- | --- | --- | --- | --- |"
    body = "\n".join(
        f"| user_{i} | {20 + i} | city_{i % 5} | active | engineering |" for i in range(rows)
    )
    return f"{header}\n{body}"


# ─── Runners ────────────────────────────────────────────────────────────────


def _run_router(label: str, content: str) -> None:
    """Compress raw tabular text through the ContentRouter."""
    result = ContentRouter().compress(content)
    before = len(content)
    after = len(result.compressed)
    pct = 100 * (before - after) / before if before else 0.0
    print(
        f"{label:24s} strat={result.strategy_used.value:9s} "
        f"chars {before:6d} -> {after:6d}  ({pct:5.1f}% saved)"
    )


def _run_messages(label: str, content: str) -> None:
    """Compress via the full pipeline (real tokenizer accounting)."""
    res = headroom.compress(
        [{"role": "user", "content": content}],
        compress_user_messages=True,
    )
    pct = 100 * res.tokens_saved / res.tokens_before if res.tokens_before else 0.0
    print(
        f"{label:24s} tokens    {res.tokens_before:6d} -> "
        f"{res.tokens_after:6d}  ({pct:5.1f}% saved)"
    )


def _run_xlsx(label: str, path: Path) -> None:
    res = headroom.compress_spreadsheet(str(path))
    pct = 100 * res.tokens_saved / res.tokens_before if res.tokens_before else 0.0
    print(
        f"{label:24s} tokens    {res.tokens_before:6d} -> "
        f"{res.tokens_after:6d}  ({pct:5.1f}% saved)"
    )


def _build_xlsx(path: Path) -> None:
    import openpyxl

    wb = openpyxl.Workbook()
    unique = wb.active
    unique.title = "Unique"
    unique.append(["id", "name", "dept"])
    for i in range(60):
        unique.append([i, f"user_{i}", ["eng", "sales", "ops"][i % 3]])

    redundant = wb.create_sheet("Redundant")
    redundant.append(["region", "product", "status"])
    for _ in range(120):
        redundant.append(["EMEA", "widget-A", "shipped"])

    wb.save(path)


# ─── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        metavar="DIR",
        help="Also write the generated sample files (.csv/.md/.xlsx) to DIR",
    )
    args = parser.parse_args()

    samples = {
        "compact_unique.csv": compact_unique_csv(),
        "redundant.csv": redundant_csv(),
        "verbose_table.md": verbose_markdown(),
    }

    print("=== Raw tabular text (ContentRouter, char-level) ===")
    _run_router("compact unique CSV", samples["compact_unique.csv"])
    _run_router("redundant CSV", samples["redundant.csv"])
    _run_router("verbose markdown", samples["verbose_table.md"])

    print("\n=== Full pipeline (real tokenizer) ===")
    _run_messages("redundant CSV", samples["redundant.csv"])

    print("\n=== Binary spreadsheet (.xlsx) ===")
    if not _HAS_OPENPYXL:
        print("  skipped — install: pip install headroom-ai[spreadsheet]")
    else:
        out_dir = Path(args.write) if args.write else Path("/tmp")
        out_dir.mkdir(parents=True, exist_ok=True)
        xlsx_path = out_dir / "demo.xlsx"
        _build_xlsx(xlsx_path)
        _run_xlsx("2-sheet workbook", xlsx_path)

    if args.write:
        out = Path(args.write)
        out.mkdir(parents=True, exist_ok=True)
        for name, content in samples.items():
            (out / name).write_text(content)
        print(f"\nSample files written to {out.resolve()}")

    print(
        "\nTakeaway: redundant/verbose tables compress; compact all-unique data "
        "correctly passes through (lossless-only — nothing safely removable)."
    )


if __name__ == "__main__":
    main()
