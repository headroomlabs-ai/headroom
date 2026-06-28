#!/usr/bin/env python
"""Push the Headroom pilot dataset + runnable harness to the HuggingFace Hub.

Uploads the data, the dataset card, and the benchmark/eval/judge scripts so the
repo is a self-contained pilot kit:  load_dataset(repo) and run eval_accuracy.py.

Auth: reads HF_TOKEN (or HUGGINGFACE_TOKEN) from the environment / .env.

Usage:
  python benchmarks/hf_pilot/upload.py --repo chopratejas/headroom-datasets          # public
  python benchmarks/hf_pilot/upload.py --repo chopratejas/headroom-datasets --private
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_REPO = HERE.parents[1]


def _load_env() -> None:
    env = _REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))


# Files uploaded to the dataset repo. (path-on-disk, path-in-repo)
FILES = [
    (HERE / "data" / "headroom_pilot.jsonl", "headroom_pilot.jsonl"),
    (HERE / "README.md", "README.md"),
    (HERE / "partner_quickstart.md", "partner_quickstart.md"),
    (HERE / "benchmark.py", "benchmark.py"),
    (HERE / "eval_accuracy.py", "eval_accuracy.py"),
    (HERE / "judge.py", "judge.py"),
    (HERE / "build_dataset.py", "build_dataset.py"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="chopratejas/headroom-datasets")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="list what would upload, do nothing")
    args = p.parse_args()

    _load_env()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set (checked env and .env).")

    missing = [str(src) for src, _ in FILES if not src.exists()]
    if missing:
        sys.exit("missing files (build the dataset first?):\n  " + "\n  ".join(missing))

    print(f"repo: {args.repo}  private={args.private}")
    for src, dst in FILES:
        print(f"  {dst:<28} <- {src.relative_to(_REPO)}  ({src.stat().st_size:,} B)")
    if args.dry_run:
        print("dry-run: nothing uploaded.")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    for src, dst in FILES:
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dst,
            repo_id=args.repo,
            repo_type="dataset",
        )
        print(f"  uploaded {dst}")
    print(f"\nDone: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
