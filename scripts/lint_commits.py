"""Lint ordinary commits by ID, leaving real merge commits out of the range."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", required=True)
    parser.add_argument("--to", dest="end", default="HEAD")
    parser.add_argument("--commitlint", default="commitlint")
    parser.add_argument(
        "--config", default=str(Path(__file__).resolve().parents[1] / ".commitlintrc.json")
    )
    args = parser.parse_args()
    executable = shutil.which(args.commitlint)
    if executable is None:
        parser.error(
            "commitlint is not on PATH; run this script inside the npm execution environment"
        )

    # npm exec exposes the CLI without exposing its sibling config package
    # from the repository. Support npm exec and CI's separate prefix install.
    env = os.environ.copy()
    package_root = next(
        (parent for parent in Path(executable).absolute().parents if parent.name == "node_modules"),
        None,
    )
    if package_root is not None:
        env["NODE_PATH"] = os.pathsep.join(
            filter(None, [str(package_root), env.get("NODE_PATH", "")])
        )

    # Resolve references before constructing the range; selection is based on
    # each commit's actual parents, never on a message another commit can copy.
    revisions = [
        subprocess.check_output(
            ["git", "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
            text=True,
        ).strip()
        for ref in (args.start, args.end)
    ]
    commits = subprocess.check_output(
        ["git", "rev-list", "--reverse", "--no-merges", f"{revisions[0]}..{revisions[1]}"],
        text=True,
    ).splitlines()
    failed = False
    for commit in commits:
        message = subprocess.check_output(
            ["git", "show", "--no-patch", "--format=%B", commit], encoding="utf-8"
        )
        result = subprocess.run(
            [executable, "--config", args.config],
            input=message,
            text=True,
            encoding="utf-8",
            env=env,
        )
        if result.returncode:
            print(f"commitlint failed for {commit}", flush=True)
            failed = True
    print(f"commitlint checked {len(commits)} non-merge commits", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
