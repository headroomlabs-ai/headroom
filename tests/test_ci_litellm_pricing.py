"""CI must load LiteLLM's version-matched, bundled pricing map."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
PRICING_JOBS = {
    "python-314-wheels",
    "test",
    "test-extras",
    "test-agno",
    "test-dashboard-ui",
}


def test_ci_jobs_load_bundled_litellm_pricing() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    assert PRICING_JOBS <= jobs.keys()

    for name in PRICING_JOBS:
        job = jobs[name]
        job_env = workflow.get("env", {}) | job.get("env", {})
        for step in job["steps"]:
            effective_env = job_env | step.get("env", {})
            assert str(effective_env.get("LITELLM_LOCAL_MODEL_COST_MAP", "")).lower() == "true", (
                name,
                step.get("name", step.get("uses")),
            )

    # Exercise LiteLLM itself with the environment inherited by a CI test job.
    # A remote fetch can succeed while silently removing models required by tests.
    test_env = os.environ | workflow.get("env", {}) | jobs["test"].get("env", {})
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import litellm; "
            "from litellm.litellm_core_utils.get_model_cost_map import "
            "get_model_cost_map_source_info; "
            "assert get_model_cost_map_source_info()['is_env_forced']; "
            "assert 'claude-sonnet-4-20250514' in litellm.model_cost; "
            "assert 'groq/llama-3.3-70b-versatile' in litellm.model_cost",
        ],
        env=test_env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
