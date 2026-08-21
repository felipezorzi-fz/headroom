"""Per-request project attribution for the proxy.

``headroom wrap`` launches agents with an ``X-Headroom-Project`` header
(via ``ANTHROPIC_CUSTOM_HEADERS`` for Claude Code and ``env_http_headers``
for Codex) naming the project directory the agent is working in. The proxy
captures that header once per request — in the HTTP middleware for regular
requests and at the WebSocket accept for Codex responses-WS sessions —
into a :mod:`contextvars` variable, so the outcome funnel can attribute
savings to a project without threading a parameter through every handler.

The value is sanitized (printable characters only, length-capped) before it
is stored; an absent or unusable header simply leaves attribution off for
that request, matching pre-feature behavior.

The HTTP middleware also binds the raw ``x-headroom-cwd`` header into a
second, unsanitized contextvar for consumers that need the literal
filesystem path (e.g. verifying a Read tool_result against disk). Not
(yet) bound at the WebSocket accept paths — an absent cwd there is already
treated as "can't resolve, don't guess."
"""

from __future__ import annotations

from collections.abc import MutableMapping
from contextvars import ContextVar
from typing import Any

from headroom.proxy.project_policy import (
    PROJECT_HEADER,
    PROJECT_PATH_PREFIX,
    classify_project,
    split_project_path,
    with_project_prefix,
)
from headroom.proxy.request_scope import normalize_scope_path
from headroom.proxy.savings_tracker import sanitize_project_name

_current_project: ContextVar[str | None] = ContextVar("headroom_current_project", default=None)

# Unsanitized, unlike _current_project — consumers join this against a
# tool's file_path and read from disk, so it must stay the literal path.
_current_cwd: ContextVar[str | None] = ContextVar("headroom_current_cwd", default=None)


def set_current_project(project: str | None) -> None:
    """Bind the active request's project for downstream outcome recording."""
    _current_project.set(sanitize_project_name(project))


def get_current_project() -> str | None:
    """Project bound to the current request context, or ``None``."""
    return _current_project.get()


def set_current_cwd(cwd: str | None) -> None:
    """Bind the active request's ``x-headroom-cwd`` header value, unmodified."""
    _current_cwd.set(cwd.strip() if isinstance(cwd, str) and cwd.strip() else None)


def get_current_cwd() -> str | None:
    """Raw cwd header bound to the current request context, or ``None``."""
    return _current_cwd.get()


def strip_project_path_prefix(scope: MutableMapping[str, Any]) -> str | None:
    """Strip a ``/p/<name>`` prefix from an ASGI scope, returning the name.

    Mutates ``scope["path"]`` (and ``raw_path``) so routing sees the
    canonical path. Must run before anything caches the request URL.
    """
    project, stripped = split_project_path(scope.get("path", ""))
    if project is not None:
        normalize_scope_path(scope, stripped)
    return project


__all__ = [
    "PROJECT_HEADER",
    "PROJECT_PATH_PREFIX",
    "classify_project",
    "get_current_cwd",
    "get_current_project",
    "set_current_cwd",
    "set_current_project",
    "split_project_path",
    "strip_project_path_prefix",
    "with_project_prefix",
]
