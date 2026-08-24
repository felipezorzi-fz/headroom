"""Code-graph backend selection and project configuration."""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib  # type: ignore[no-redef]


class CodeGraphBackend(str, Enum):
    """Supported code-graph implementations."""

    CODEBASE_MEMORY = "codebase-memory-mcp"
    CODEGRAPH = "codegraph"


DEFAULT_CODE_GRAPH_BACKEND = CodeGraphBackend.CODEBASE_MEMORY
CODE_GRAPH_BACKEND_CHOICES = tuple(backend.value for backend in CodeGraphBackend)
_CONFIG_FILENAMES = ("headroom.toml", "headroom.yaml", "headroom.yml")


def normalize_code_graph_backend(value: str | CodeGraphBackend) -> CodeGraphBackend:
    """Convert a user-facing backend name to the canonical enum value."""

    if isinstance(value, CodeGraphBackend):
        return value
    normalized = value.strip().lower()
    try:
        return CodeGraphBackend(normalized)
    except ValueError as exc:
        choices = ", ".join(CODE_GRAPH_BACKEND_CHOICES)
        raise ValueError(f"unknown code-graph backend {value!r}; choose one of: {choices}") from exc


def _load_config(path: Path) -> dict[str, Any] | None:
    try:
        if path.suffix == ".toml":
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        else:
            import yaml  # type: ignore[import-untyped]

            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ImportError:
        logger.warning("Could not read %s: PyYAML is not installed", path)
        return None
    except Exception as exc:  # config errors must not block proxy startup
        logger.warning("Could not read %s: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def _config_backend(payload: dict[str, Any]) -> Any:
    value = payload.get("code_graph_backend")
    if value is not None:
        return value
    code_graph = payload.get("code_graph")
    if isinstance(code_graph, dict):
        return code_graph.get("backend")
    return None


def _configured_backend(project_dir: Path) -> CodeGraphBackend | None:
    for filename in _CONFIG_FILENAMES:
        path = project_dir / filename
        if not path.is_file():
            continue
        payload = _load_config(path)
        if payload is None:
            continue
        value = _config_backend(payload)
        if value is None:
            continue
        if not isinstance(value, str):
            logger.warning("Ignoring non-string code-graph backend in %s", path)
            continue
        try:
            return normalize_code_graph_backend(value)
        except ValueError as exc:
            logger.warning("Ignoring invalid code-graph backend in %s: %s", path, exc)
    return None


def resolve_code_graph_backend(
    value: str | CodeGraphBackend | None = None,
    *,
    project_dir: str | Path | None = None,
) -> CodeGraphBackend:
    """Resolve backend with CLI, environment, file, and default precedence."""

    if value is not None and (isinstance(value, CodeGraphBackend) or str(value).strip()):
        return normalize_code_graph_backend(value)

    env_value = os.environ.get("HEADROOM_CODE_GRAPH_BACKEND", "").strip()
    if env_value:
        try:
            return normalize_code_graph_backend(env_value)
        except ValueError as exc:
            logger.warning("Ignoring HEADROOM_CODE_GRAPH_BACKEND: %s", exc)

    configured = _configured_backend(Path(project_dir or Path.cwd()))
    return configured or DEFAULT_CODE_GRAPH_BACKEND
