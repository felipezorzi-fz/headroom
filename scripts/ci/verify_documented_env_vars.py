#!/usr/bin/env python3
"""Verify that documented environment variables exist in implementation source.

The check is deliberately text-based and dependency-free so it can run before
either the Python package or the documentation application is installed.  It
guards against documentation for a misspelled or removed setting silently
surviving after the implementation changes.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ENV_VAR_PATTERN = re.compile(
    r"(?<![A-Z0-9_])(?:(?:HEADROOM|ANTHROPIC|OPENAI)_[A-Z0-9_]+(?:\*)?|DO_NOT_TRACK)"
    r"(?![A-Z0-9_])"
)

DOCUMENTATION_FILES = (Path("README.md"), Path("SECURITY.md"))
DOCUMENTATION_GLOB = "docs/content/docs/**/*.mdx"

# These are implementation surfaces, not tests or examples.  Docker and install
# sources are included because a few documented host-side variables are consumed
# before the Python or Rust process starts.
SOURCE_PATHS = (
    Path("headroom"),
    Path("crates"),
    Path("sdk"),
    Path("plugins"),
    Path("deploy"),
    Path("docker"),
    Path("scripts"),
)
SOURCE_SUFFIXES = {
    ".cjs",
    ".js",
    ".json",
    ".mjs",
    ".ps1",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
EXCLUDED_SOURCE_PARTS = {
    "__pycache__",
    "fixtures",
    "node_modules",
    "target",
    "tests",
}


@dataclass(frozen=True)
class Occurrence:
    """One documented environment-variable reference."""

    path: Path
    line: int


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _documentation_paths(root: Path) -> list[Path]:
    paths = [root / relative_path for relative_path in DOCUMENTATION_FILES]
    paths.extend(root.glob(DOCUMENTATION_GLOB))
    return sorted(path for path in paths if path.is_file())


def _source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for relative_path in SOURCE_PATHS:
        path = root / relative_path
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = path.rglob("*")
        else:
            continue

        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix not in SOURCE_SUFFIXES:
                continue
            relative_parts = candidate.relative_to(root).parts
            if any(part in EXCLUDED_SOURCE_PARTS for part in relative_parts):
                continue
            files.append(candidate)
    return sorted(files)


def documented_variables(root: Path) -> dict[str, list[Occurrence]]:
    """Return documented variables and the locations that mention them."""

    variables: dict[str, list[Occurrence]] = defaultdict(list)
    for path in _documentation_paths(root):
        for line_number, line in enumerate(_read_text(path).splitlines(), start=1):
            for match in ENV_VAR_PATTERN.finditer(line):
                variables[match.group(0)].append(
                    Occurrence(path=path.relative_to(root), line=line_number)
                )
    return dict(variables)


def source_variables(root: Path) -> set[str]:
    """Return environment-variable-shaped identifiers found in implementation source."""

    variables: set[str] = set()
    for path in _source_files(root):
        variables.update(ENV_VAR_PATTERN.findall(_read_text(path)))
    return variables


def missing_variables(root: Path) -> dict[str, list[Occurrence]]:
    """Return documented variables that have no implementation reference."""

    documented = documented_variables(root)
    implemented = source_variables(root)
    missing: dict[str, list[Occurrence]] = {}

    for name, occurrences in documented.items():
        if name.endswith("*"):
            prefix = name[:-1]
            found = any(
                candidate.startswith(prefix) and not candidate.endswith("*")
                for candidate in implemented
            )
        else:
            found = name in implemented
        if not found:
            missing[name] = occurrences

    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root (defaults to the checkout containing this script)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()

    documented = documented_variables(root)
    missing = missing_variables(root)
    if missing:
        for name, occurrences in sorted(missing.items()):
            first = occurrences[0]
            print(
                f"::error file={first.path},line={first.line}::"
                f"Documented environment variable {name} has no implementation reference",
                file=sys.stderr,
            )
        print(
            f"Found {len(missing)} documented environment variable(s) with no source reference.",
            file=sys.stderr,
        )
        return 1

    print(f"Verified {len(documented)} documented environment variables against source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
