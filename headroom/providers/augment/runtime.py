"""Runtime helpers for AugmentCode Auggie integration.

``headroom wrap auggie`` routes Auggie through Headroom by rewriting the tenant
URL in Auggie's OAuth session so it points at the local proxy, then passing the
rewritten session to Auggie via ``AUGMENT_SESSION_AUTH``. Auggie loads its
session on startup and sets its own ``AUGMENT_API_URL`` from the session's
``tenantURL``, so rewriting that field (and only that field) is what actually
redirects Auggie's traffic; the access token is preserved byte-for-byte. The
proxy forwards every Auggie tenant path verbatim to the real tenant resolved
here and tags the ``/chat-stream`` inference call telemetry as ``augment``. The
Augment wire is proprietary, so request bodies are forwarded unchanged and no
API keys are fabricated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_AUGMENT_SESSION_PATH = Path.home() / ".augment" / "session.json"


def proxy_base_url(port: int) -> str:
    """Return the local Headroom base URL Auggie targets as its tenant URL."""
    return f"http://127.0.0.1:{port}"


def load_session(path: Path | None = None) -> dict[str, Any]:
    """Load Auggie's stored OAuth session JSON.

    Args:
        path: Session file path. Defaults to ``~/.augment/session.json``.

    Returns:
        The parsed session dict (contains ``accessToken`` and ``tenantURL``).

    Raises:
        FileNotFoundError: The session file does not exist (Auggie not logged in).
        ValueError: The file is not valid JSON, is not a JSON object, or is
            missing the ``accessToken`` / ``tenantURL`` fields the redirect needs.
    """
    session_path = path or DEFAULT_AUGMENT_SESSION_PATH
    if not session_path.exists():
        raise FileNotFoundError(str(session_path))
    try:
        data = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Auggie session at {session_path} is not readable JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Auggie session at {session_path} is not a JSON object.")
    for field in ("accessToken", "tenantURL"):
        if not data.get(field):
            raise ValueError(f"Auggie session at {session_path} is missing '{field}'.")
    return data


def resolve_augment_upstream(session: dict[str, Any], explicit: str | None = None) -> str:
    """Resolve the real Auggie tenant upstream the proxy forwards to.

    Precedence: an explicit ``--augment-api-url``, then the session's
    ``tenantURL`` (the per-user tenant gateway, e.g. an EU host). Trailing
    slashes are trimmed.
    """
    candidate = explicit or session.get("tenantURL") or ""
    return candidate.rstrip("/")


def build_redirected_session(session: dict[str, Any], port: int) -> str:
    """Return ``session`` serialized with ``tenantURL`` pointed at the local proxy.

    Only ``tenantURL`` is changed; every other field (including ``accessToken``)
    is preserved verbatim, so Auggie authenticates to the real tenant through the
    proxy with the user's own credential and Headroom never fabricates a key.
    """
    redirected = dict(session)
    redirected["tenantURL"] = proxy_base_url(port)
    return json.dumps(redirected)
