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


def test_root_docs_and_supported_non_headroom_prefixes_are_scanned(tmp_path: Path) -> None:
    module = _load_module()
    (tmp_path / "README.md").write_text(
        "Use `ANTHROPIC_BASE_URL` or `OPENAI_BASE_URL`.\n", encoding="utf-8"
    )
    (tmp_path / "SECURITY.md").write_text("Set `DO_NOT_TRACK=1`.\n", encoding="utf-8")
    source_path = tmp_path / "headroom" / "proxy.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        'VARIABLES = ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "DO_NOT_TRACK")\n',
        encoding="utf-8",
    )

    documented = module.documented_variables(tmp_path)

    assert set(documented) == {"ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "DO_NOT_TRACK"}
    assert module.missing_variables(tmp_path) == {}


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
