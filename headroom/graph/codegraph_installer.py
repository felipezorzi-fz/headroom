"""Install and initialize the external CodeGraph CLI."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from headroom._subprocess import run

logger = logging.getLogger(__name__)

CODEGRAPH_NPM_PACKAGE = "@colbymchenry/codegraph"
CODEGRAPH_BIN_NAME = "codegraph"
_COMMAND_TIMEOUT_SECONDS = 180


def get_codegraph_path() -> Path | None:
    """Return an installed CodeGraph executable, if one is discoverable."""

    for name in (CODEGRAPH_BIN_NAME, f"{CODEGRAPH_BIN_NAME}.exe"):
        found = shutil.which(name)
        if found:
            return Path(found)

    for directory in (Path.home() / ".local" / "bin", Path.home() / "bin"):
        for name in (CODEGRAPH_BIN_NAME, f"{CODEGRAPH_BIN_NAME}.exe"):
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def _npm_path() -> str | None:
    return shutil.which("npm") or shutil.which("npm.cmd")


def ensure_codegraph() -> Path | None:
    """Ensure CodeGraph is available, installing it through npm when needed."""

    existing = get_codegraph_path()
    if existing:
        return existing
    if os.environ.get("HEADROOM_BINARIES_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        logger.info("CodeGraph is not installed; HEADROOM_BINARIES_OFFLINE is set")
        return None

    npm = _npm_path()
    if not npm:
        logger.warning("CodeGraph is not installed and npm was not found on PATH")
        return None

    try:
        result = run(
            [npm, "install", "--global", CODEGRAPH_NPM_PACKAGE],
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Could not install CodeGraph with npm: %s", exc)
        return None
    if result.returncode != 0:
        logger.warning("Could not install CodeGraph with npm: %s", result.stderr.strip())
        return None
    return get_codegraph_path()


def run_codegraph(
    binary: str | Path,
    args: list[str],
    *,
    project_dir: str | Path | None = None,
    timeout: float = 60,
) -> subprocess.CompletedProcess[str]:
    """Run a CodeGraph command without invoking a shell."""

    return run(
        [str(binary), *args],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=str(project_dir) if project_dir is not None else None,
        timeout=timeout,
    )


def initialize_codegraph(
    binary: str | Path | None = None,
    *,
    project_dir: str | Path | None = None,
) -> bool:
    """Create or refresh the current project's CodeGraph index."""

    path = Path(binary) if binary is not None else ensure_codegraph()
    if path is None:
        return False
    try:
        result = run_codegraph(path, ["init"], project_dir=project_dir)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("CodeGraph init failed: %s", exc)
        return False
    if result.returncode != 0:
        logger.warning("CodeGraph init failed: %s", result.stderr.strip())
        return False
    return True


def install_codegraph(binary: str | Path | None = None) -> bool:
    """Register CodeGraph with detected coding agents."""

    path = Path(binary) if binary is not None else ensure_codegraph()
    if path is None:
        return False
    try:
        result = run_codegraph(path, ["install", "--yes"], timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("CodeGraph agent installation failed: %s", exc)
        return False
    if result.returncode != 0:
        logger.warning("CodeGraph agent installation failed: %s", result.stderr.strip())
        return False
    return True


def uninstall_codegraph(binary: str | Path | None = None) -> bool:
    """Remove CodeGraph agent configuration while keeping its CLI installed."""

    path = Path(binary) if binary is not None else get_codegraph_path()
    if path is None:
        return False
    try:
        result = run_codegraph(path, ["uninstall", "--keep-cli", "--yes"], timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("CodeGraph agent uninstall failed: %s", exc)
        return False
    if result.returncode != 0:
        logger.warning("CodeGraph agent uninstall failed: %s", result.stderr.strip())
        return False
    return True
