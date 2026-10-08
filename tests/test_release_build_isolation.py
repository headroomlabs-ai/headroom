from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_source_execution_has_no_repository_permissions_or_authenticated_checkout():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/release-build.yml").read_text(encoding="utf-8")
    )
    for name in ("detect-version", "build", "build-wheels", "smoke-import-wheels"):
        job = workflow["jobs"][name]
        assert job["permissions"] == {}, name
        assert all("checkout" not in step.get("uses", "") for step in job["steps"]), name
    prepare = workflow["jobs"]["prepare-source"]
    checkout = next(step for step in prepare["steps"] if "checkout" in step.get("uses", ""))
    assert checkout["with"]["persist-credentials"] is False
    assert prepare["permissions"] == {"contents": "read"}
