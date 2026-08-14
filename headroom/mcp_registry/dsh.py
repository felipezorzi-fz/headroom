"""DeepSeek Harness (dsh) MCP registrar."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .base import MCPRegistrar, RegisterResult, RegisterStatus, ServerSpec

logger = logging.getLogger(__name__)

_MARKER_START = "# --- Headroom MCP server ---"
_MARKER_END = "# --- end Headroom MCP server ---"
_MCP_CLIENT_PLUGIN = "@deepseek-ai/dsh-mcp-client"
_HEADROOM_ENTRY_ID = "mcp-headroom"


def _dsh_home() -> Path:
    """Return the dsh Harness home ($DSH_HOME, else ~/.dsh)."""
    return Path(os.environ.get("DSH_HOME") or Path.home() / ".dsh")


def _spec_to_block(spec: ServerSpec) -> str:
    """Render ``spec`` as a marker-fenced YAML ``insert`` block."""
    config: dict[str, Any] = {
        "serverName": spec.name,
        "transport": "stdio",
        "command": spec.command,
        "args": spec.args,
    }
    if spec.env:
        config["env"] = spec.env
    entry = {
        "id": _HEADROOM_ENTRY_ID,
        "name": _MCP_CLIENT_PLUGIN,
        "config": config,
    }
    body = yaml.safe_dump([{"insert": [entry]}], default_flow_style=False, sort_keys=False)
    return f"{_MARKER_START}\n{body}{_MARKER_END}\n"


def _block_to_spec(block: str) -> tuple[ServerSpec | None, str | None]:
    """Parse a fenced block into ``(spec, corrupt_reason)``.

    Returns ``(None, reason)`` when the block is present but unparsable or does
    not contain the headroom entry — corrupt managed state that must not be
    silently treated as "not registered".
    """
    body = block.replace(_MARKER_START, "").replace(_MARKER_END, "").strip()
    try:
        doc = yaml.safe_load(body)
    except yaml.YAMLError as exc:
        return None, f"malformed YAML in managed block: {exc}"
    if not isinstance(doc, list):
        return None, "managed block is not a YAML list"
    for op in doc:
        if not isinstance(op, dict):
            continue
        for row in op.get("insert", []):
            if isinstance(row, dict) and row.get("id") == _HEADROOM_ENTRY_ID:
                config = row.get("config", {})
                return (
                    ServerSpec(
                        name=config.get("serverName", "headroom"),
                        command=config.get("command", ""),
                        args=tuple(config.get("args", [])),
                        env=config.get("env", {}),
                    ),
                    None,
                )
    return None, "managed block does not contain the headroom entry"


@dataclass
class _ManagedBlock:
    """Read-back of the marker-fenced block in ``cordis.patch.yml``.

    ``spec`` set → block present and clean; ``corrupt`` set → block present but
    broken; both ``None`` → no block.
    """

    spec: ServerSpec | None
    corrupt: str | None


def _read_managed_block(config_file: Path) -> _ManagedBlock:
    """Read the managed block, distinguishing absent / clean / corrupt."""
    if not config_file.exists():
        return _ManagedBlock(None, None)
    text = config_file.read_text(encoding="utf-8")
    if _MARKER_START not in text:
        return _ManagedBlock(None, None)
    start = text.index(_MARKER_START)
    end_marker = text.find(_MARKER_END, start)
    if end_marker == -1:
        return _ManagedBlock(None, "unterminated start marker (missing end marker)")
    block = text[start : end_marker + len(_MARKER_END)]
    spec, corrupt = _block_to_spec(block)
    return _ManagedBlock(spec, corrupt)


def _remove_managed_block(text: str) -> str:
    """Return ``text`` with the marker-fenced block removed.

    Handles a truncated block (start marker with no end marker) by dropping
    everything from the start marker to end-of-file. Unrelated bytes before
    and after the block are preserved verbatim.
    """
    if _MARKER_START not in text:
        return text
    start = text.index(_MARKER_START)
    end_marker = text.find(_MARKER_END, start)
    if end_marker == -1:
        return text[:start]
    tail = text[end_marker + len(_MARKER_END) :]
    if tail.startswith("\n"):
        tail = tail[1:]
    return text[:start] + tail


def _atomic_write(config_file: Path, text: str) -> None:
    """Write ``text`` atomically via a temp file + ``os.replace``."""
    tmp = config_file.with_name(config_file.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, config_file)


def _specs_equivalent(a: ServerSpec, b: ServerSpec) -> bool:
    return (
        a.name == b.name
        and a.command == b.command
        and tuple(a.args) == tuple(b.args)
        and dict(a.env) == dict(b.env)
    )


def _diff_specs(existing: ServerSpec, requested: ServerSpec) -> str:
    return f"existing {existing.command} {existing.args}, requested {requested.command} {requested.args}"


class DshRegistrar(MCPRegistrar):
    """Register MCP servers with DeepSeek Harness (dsh)."""

    name = "dsh"
    display_name = "DeepSeek Harness"

    @property
    def _config_file(self) -> Path:
        return _dsh_home() / "cordis.patch.yml"

    def detect(self) -> bool:
        return _dsh_home().exists()

    def get_server(self, server_name: str) -> ServerSpec | None:
        if server_name != "headroom":
            return None
        return _read_managed_block(self._config_file).spec

    def register_server(self, spec: ServerSpec, *, force: bool = False) -> RegisterResult:
        if not self.detect():
            return RegisterResult(
                RegisterStatus.NOT_DETECTED, f"dsh home not found at {_dsh_home()}"
            )
        managed = _read_managed_block(self._config_file)
        if managed.corrupt is not None:
            if not force:
                return RegisterResult(
                    RegisterStatus.FAILED,
                    f"corrupt managed block in {self._config_file}: {managed.corrupt}; "
                    "re-run with --force to overwrite",
                )
        elif managed.spec is not None:
            if _specs_equivalent(managed.spec, spec):
                return RegisterResult(
                    RegisterStatus.ALREADY, f"already registered in {self._config_file}"
                )
            if not force:
                return RegisterResult(RegisterStatus.MISMATCH, _diff_specs(managed.spec, spec))
        # Absent, or (corrupt | mismatch) with force: rewrite atomically.
        existing = (
            self._config_file.read_text(encoding="utf-8") if self._config_file.exists() else ""
        )
        new_text = _remove_managed_block(existing) + _spec_to_block(spec)
        try:
            self._config_file.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(self._config_file, new_text)
        except OSError as exc:
            return RegisterResult(
                RegisterStatus.FAILED, f"failed to write {self._config_file}: {exc}"
            )
        return RegisterResult(RegisterStatus.REGISTERED, f"wrote to {self._config_file}")

    def unregister_server(self, server_name: str) -> bool:
        if server_name != "headroom":
            return False
        if not self._config_file.exists():
            return True
        text = self._config_file.read_text(encoding="utf-8")
        if _MARKER_START not in text:
            return True
        _atomic_write(self._config_file, _remove_managed_block(text))
        return True
