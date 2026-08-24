#!/usr/bin/env python3
"""Helpers for PR health maintenance labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

FAILING_STATES = {"FAILURE", "TIMED_OUT", "ACTION_REQUIRED", "CANCELLED", "ERROR"}


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    normalized = value.removesuffix("Z") + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _check_key(check: dict[str, Any]) -> tuple[str, str]:
    workflow = str(check.get("workflowName") or check.get("workflow") or "")
    name = str(check.get("name") or check.get("context") or "")
    return workflow, name


def _check_time(check: dict[str, Any]) -> datetime:
    return max(
        _parse_timestamp(check.get("startedAt")),
        _parse_timestamp(check.get("completedAt")),
    )


def _state(check: dict[str, Any]) -> str:
    return str(check.get("conclusion") or check.get("state") or "").upper()


def current_checks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    latest_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for check in payload.get("statusCheckRollup") or []:
        if not isinstance(check, dict):
            continue
        key = _check_key(check)
        if not any(key):
            continue
        previous = latest_by_key.get(key)
        if previous is None or _check_time(check) >= _check_time(previous):
            latest_by_key[key] = check
    return list(latest_by_key.values())


def check_state(payload: dict[str, Any]) -> str:
    for check in current_checks(payload):
        if _state(check) in FAILING_STATES:
            return "failing"
    return "passing"


def parse_behind_by(value: str) -> int | None:
    """Read the commit count from a compare call that may have failed."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def drift_state(
    payload: dict[str, Any],
    behind_by: int | None,
    base_files: Iterable[Any],
) -> str:
    """Classify a pull request branch against the current tip of its base branch.

    GitHub only reports `mergeStateStatus == BEHIND` when the base branch requires
    branches to be up to date, so drift is measured from an explicit ref comparison
    instead. Being behind alone is normal on a busy base branch; the risky case is
    being behind on files the pull request also modifies, because that is where a
    semantic merge conflict hides from per-pull-request checks.

    Returns "stale", "current", or "unknown" when the comparison is unavailable.
    """
    if _merge_state(payload) == "BEHIND":
        return "stale"
    if behind_by is None:
        return "unknown"
    if behind_by <= 0:
        return "current"

    moved = {str(path) for path in base_files or []}
    touched = {
        str(entry.get("path")) for entry in payload.get("files") or [] if isinstance(entry, dict)
    }
    return "stale" if moved & touched else "current"


def _merge_state(payload: dict[str, Any]) -> str:
    return str(payload.get("mergeStateStatus") or "").upper()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-json", required=True, help="JSON from gh pr view")
    parser.add_argument(
        "--field",
        choices=("checks", "drift"),
        default="checks",
        help="Which label signal to print",
    )
    parser.add_argument(
        "--behind-by",
        default="",
        help="Commits the base branch is ahead of the pull request head, empty when unknown",
    )
    parser.add_argument(
        "--base-files",
        default="[]",
        help="JSON array of files the base branch changed since the merge base",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = json.loads(args.state_json)
    if args.field == "drift":
        base_files = json.loads(args.base_files or "[]")
        print(drift_state(payload, parse_behind_by(args.behind_by), base_files))
    else:
        print(check_state(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
