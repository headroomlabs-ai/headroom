"""Tests for the documented environment-variable consistency check."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script = Path(__file__).parent.parent / "ci" / "verify_documented_env_vars.py"
    spec = importlib.util.spec_from_file_location("verify_documented_env_vars", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_minimal_project(root: Path, *, docs: str, source: str) -> None:
    docs_path = root / "docs" / "content" / "docs" / "configuration.mdx"
    docs_path.parent.mkdir(parents=True)
    docs_path.write_text(docs, encoding="utf-8")
    source_path = root / "headroom" / "config.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(source, encoding="utf-8")


def test_exact_documented_variable_must_exist_in_source(tmp_path: Path) -> None:
    module = _load_module()
    _write_minimal_project(
        tmp_path,
        docs="Use `HEADROOM_REAL_SETTING` and `HEADROOM_MISSPELLED_SETTING`.\n",
        source='REAL_SETTING = "HEADROOM_REAL_SETTING"\n',
    )

    missing = module.missing_variables(tmp_path)

    assert list(missing) == ["HEADROOM_MISSPELLED_SETTING"]
    assert missing["HEADROOM_MISSPELLED_SETTING"][0].line == 1


def test_documented_wildcard_matches_a_concrete_source_variable(tmp_path: Path) -> None:
    module = _load_module()
    _write_minimal_project(
        tmp_path,
        docs="Leave `HEADROOM_OTEL_*` unset to use the ambient provider.\n",
        source='ENABLED = "HEADROOM_OTEL_METRICS_ENABLED"\n',
    )

    assert module.missing_variables(tmp_path) == {}


def test_placeholder_wildcard_is_normalized_and_matches_source(tmp_path: Path) -> None:
    module = _load_module()
    _write_minimal_project(
        tmp_path,
        docs="Set `DISABLE_PROMPT_CACHING_<FAMILY>` for one model family.\n",
        source='SETTING = "DISABLE_PROMPT_CACHING_OPUS"\n',
    )

    documented = module.documented_variables(tmp_path)

    assert set(documented) == {"DISABLE_PROMPT_CACHING_*"}
    assert module.missing_variables(tmp_path) == {}


def test_single_word_variable_is_discovered_from_shell_context(tmp_path: Path) -> None:
    module = _load_module()
    _write_minimal_project(
        tmp_path,
        docs="The wrapper reads `$ACME` and a typo such as `$ACMEE` must fail.\n",
        source='value = os.environ.get("ACME")\n',
    )

    assert set(module.documented_variables(tmp_path)) == {"ACME", "ACMEE"}
    assert set(module.missing_variables(tmp_path)) == {"ACMEE"}


def test_host_side_variable_can_be_implemented_by_install_source(tmp_path: Path) -> None:
    module = _load_module()
    _write_minimal_project(
        tmp_path,
        docs="Set `HEADROOM_DOCKER_IMAGE` before installation.\n",
        source="",
    )
    install_script = tmp_path / "scripts" / "install.sh"
    install_script.parent.mkdir(parents=True)
    install_script.write_text('IMAGE="${HEADROOM_DOCKER_IMAGE:-latest}"\n', encoding="utf-8")

    assert module.missing_variables(tmp_path) == {}


def test_root_docs_and_generic_environment_variable_names_are_scanned(
    tmp_path: Path,
) -> None:
    module = _load_module()
    (tmp_path / "README.md").write_text(
        "Use `ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL`, or `ACME_SERVICE_TOKEN`.\n",
        encoding="utf-8",
    )
    (tmp_path / "SECURITY.md").write_text("Set `DO_NOT_TRACK=1`.\n", encoding="utf-8")
    source_path = tmp_path / "headroom" / "proxy.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        'VARIABLES = ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", '
        '"ACME_SERVICE_TOKEN", "DO_NOT_TRACK")\n',
        encoding="utf-8",
    )

    documented = module.documented_variables(tmp_path)

    assert set(documented) == {
        "ACME_SERVICE_TOKEN",
        "ANTHROPIC_BASE_URL",
        "DO_NOT_TRACK",
        "OPENAI_BASE_URL",
    }
    assert module.missing_variables(tmp_path) == {}


def test_code_constants_and_lowercase_name_fragments_are_not_variables(
    tmp_path: Path,
) -> None:
    module = _load_module()
    _write_minimal_project(
        tmp_path,
        docs=(
            "Use `HEADROOM_REAL_SETTING`.\n"
            "Compare PipelineStage.PRE_SEND with `FIRST_LINE`.\n"
            "An install may contain .DS_Store or report CERTIFICATE_VERIFY_FAILED.\n"
        ),
        source='SETTING = "HEADROOM_REAL_SETTING"\n',
    )

    assert set(module.documented_variables(tmp_path)) == {"HEADROOM_REAL_SETTING"}


def test_external_consumer_policy_is_explicit_and_does_not_hide_typos(
    tmp_path: Path,
) -> None:
    module = _load_module()
    _write_minimal_project(
        tmp_path,
        docs="Set `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_ENDPONT`.\n",
        source="",
    )

    missing = module.missing_variables(tmp_path)

    assert set(missing) == {"OTEL_EXPORTER_OTLP_ENDPONT"}


def test_cli_success_reports_source_and_external_policy_scope(tmp_path: Path, capsys) -> None:
    module = _load_module()
    _write_minimal_project(
        tmp_path,
        docs="Set `HEADROOM_REAL_SETTING` and `OTEL_EXPORTER_OTLP_ENDPOINT`.\n",
        source='SETTING = "HEADROOM_REAL_SETTING"\n',
    )

    exit_code = module.main(["--root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert (
        "Verified 2 documented environment variables "
        "(1 against repository source; 1 against the approved external-consumer policy)."
        in captured.out
    )


def test_verifier_policy_does_not_count_as_an_implementation_reference(
    tmp_path: Path,
) -> None:
    module = _load_module()
    _write_minimal_project(
        tmp_path,
        docs="Set `ACME_EXTERNAL_SETTING`.\n",
        source="",
    )
    verifier = tmp_path / "scripts" / "ci" / "verify_documented_env_vars.py"
    verifier.parent.mkdir(parents=True)
    verifier.write_text('POLICY = {"ACME_EXTERNAL_SETTING"}\n', encoding="utf-8")

    assert set(module.missing_variables(tmp_path)) == {"ACME_EXTERNAL_SETTING"}


def test_verifier_tests_do_not_count_as_implementation_references(tmp_path: Path) -> None:
    module = _load_module()
    _write_minimal_project(
        tmp_path,
        docs="Set `ACME_TEST_ONLY_SETTING`.\n",
        source="",
    )
    verifier_test = tmp_path / "scripts" / "tests" / "test_verify_documented_env_vars.py"
    verifier_test.parent.mkdir(parents=True)
    verifier_test.write_text(
        'REPRESENTATIVE_VARIABLE = "ACME_TEST_ONLY_SETTING"\n',
        encoding="utf-8",
    )

    assert "ACME_TEST_ONLY_SETTING" not in module.source_variables(tmp_path)
    assert set(module.missing_variables(tmp_path)) == {"ACME_TEST_ONLY_SETTING"}


def test_repository_docs_discover_representative_environment_variable_families() -> None:
    module = _load_module()
    repository_root = Path(__file__).resolve().parents[2]

    documented = set(module.documented_variables(repository_root))

    assert {
        # AWS and Bedrock.
        "AWS_ACCESS_KEY_ID",
        "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
        "BEDROCK_TARGET_API_URL",
        # Claude's alternate cloud runtimes.
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDE_CODE_USE_VERTEX",
        # Google and Vertex.
        "GOOGLE_APPLICATION_CREDENTIALS",
        "VERTEXAI_PROJECT",
        # GitHub Copilot.
        "GITHUB_COPILOT_ENTERPRISE_URL",
        # Process-wide proxy and TLS controls.
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        # OpenTelemetry.
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        # Provider keys outside the original prefix policy.
        "OPENROUTER_API_KEY",
        "XAI_API_KEY",
    } <= documented


def test_cli_fails_with_a_github_annotation_for_missing_variable(tmp_path: Path, capsys) -> None:
    module = _load_module()
    _write_minimal_project(
        tmp_path,
        docs="Configuration: `HEADROOM_REMOVED_SETTING`\n",
        source="",
    )

    exit_code = module.main(["--root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "::error file=docs/content/docs/configuration.mdx,line=1::" in captured.err
    assert "HEADROOM_REMOVED_SETTING" in captured.err
