"""Tests for code-graph backend selection and CodeGraph integration."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from headroom.cli import wrap as wrap_cli
from headroom.graph import backend, codegraph_installer
from headroom.graph.backend import CodeGraphBackend
from headroom.graph.watcher import CodeGraphWatcher


def test_resolve_backend_precedence(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "headroom.toml"
    config.write_text('code_graph_backend = "codegraph"\n', encoding="utf-8")

    monkeypatch.delenv("HEADROOM_CODE_GRAPH_BACKEND", raising=False)
    assert backend.resolve_code_graph_backend(project_dir=tmp_path) == CodeGraphBackend.CODEGRAPH

    monkeypatch.setenv("HEADROOM_CODE_GRAPH_BACKEND", "codebase-memory-mcp")
    assert backend.resolve_code_graph_backend(project_dir=tmp_path) == (
        CodeGraphBackend.CODEBASE_MEMORY
    )
    assert backend.resolve_code_graph_backend("codegraph", project_dir=tmp_path) == (
        CodeGraphBackend.CODEGRAPH
    )


def test_resolve_backend_defaults_to_codebase_memory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HEADROOM_CODE_GRAPH_BACKEND", raising=False)
    assert backend.resolve_code_graph_backend(project_dir=tmp_path) == (
        CodeGraphBackend.CODEBASE_MEMORY
    )


def test_codegraph_backend_replaces_legacy_graph_backend(monkeypatch) -> None:
    calls: list[str] = []

    def setup_codegraph(*, verbose: bool = False) -> bool:
        calls.append("codegraph")
        return True

    def legacy_backend_must_not_run(*, verbose: bool = False) -> bool:
        raise AssertionError("legacy codebase-memory-mcp backend must not run")

    monkeypatch.setattr(wrap_cli, "_setup_codegraph_backend", setup_codegraph)
    monkeypatch.setattr(wrap_cli, "_setup_codebase_memory_graph", legacy_backend_must_not_run)

    assert wrap_cli._setup_code_graph(CodeGraphBackend.CODEGRAPH) is True
    assert calls == ["codegraph"]


def test_codegraph_installer_uses_noninteractive_commands(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[list[str], str | None]] = []

    def fake_run_codegraph(binary, args, *, project_dir=None, timeout=60):
        calls.append(([str(binary), *args], str(project_dir) if project_dir else None))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(codegraph_installer, "run_codegraph", fake_run_codegraph)

    assert codegraph_installer.install_codegraph("codegraph") is True
    assert codegraph_installer.initialize_codegraph("codegraph", project_dir=tmp_path) is True
    assert codegraph_installer.uninstall_codegraph("codegraph") is True
    assert calls == [
        (["codegraph", "install", "--yes"], None),
        (["codegraph", "init"], str(tmp_path)),
        (["codegraph", "uninstall", "--keep-cli", "--yes"], None),
    ]


def test_codebase_memory_watcher_uses_index_command(monkeypatch, tmp_path: Path) -> None:
    graph_watcher = CodeGraphWatcher(
        tmp_path,
        cbm_binary="codebase-memory-mcp",
        backend=CodeGraphBackend.CODEBASE_MEMORY,
    )
    graph_watcher._running = True
    run_calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        run_calls.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("headroom.graph.watcher.run", fake_run)
    monkeypatch.setattr("headroom.graph.watcher.time.monotonic", lambda: 1.0)
    monkeypatch.setattr("headroom.graph.watcher.time.time", lambda: 2.0)

    graph_watcher._do_reindex()

    assert run_calls == [
        [
            "codebase-memory-mcp",
            "cli",
            "index_repository",
            f'{{"repo_path": "{tmp_path}", "mode": "fast"}}',
        ]
    ]
    assert graph_watcher.stats["reindex_count"] == 1


def test_codegraph_installer_handles_timeout(monkeypatch) -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs.get("timeout", 60))

    monkeypatch.setattr(codegraph_installer, "run_codegraph", timeout)
    assert codegraph_installer.initialize_codegraph("codegraph") is False
