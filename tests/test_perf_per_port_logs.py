"""Per-port proxy log files.

Every live proxy writes its own ``proxy-<port>.log`` so that concurrent
instances (wrapped copilot, droid, and claude on different ports) do not share
a single ``RotatingFileHandler`` target and rotate each other's history away.
The perf reader aggregates all per-port files and still honors the legacy
shared ``proxy.log`` as a fallback.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from headroom import paths as _paths
from headroom.cli import wrap as wrap_cli
from headroom.perf import analyzer
from headroom.proxy.helpers import _setup_file_logging


def _perf_line(ts: str, rid: str, model: str) -> str:
    return (
        f"{ts} - headroom.proxy - INFO - [{rid}] PERF "
        f"model={model} msgs=3 tok_before=1000 tok_after=400 "
        f"tok_saved=600 cache_read=0 cache_write=0 cache_hit_pct=0 "
        f"opt_ms=1 transforms=agent90_smoke client=test"
    )


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_proxy_log_path_is_per_port() -> None:
    a = _paths.proxy_log_path(8888)
    b = _paths.proxy_log_path(8889)
    assert a.name == "proxy-8888.log"
    assert b.name == "proxy-8889.log"
    assert a != b  # the core anti-collision property
    # Legacy shared name preserved for the readers' fallback.
    assert _paths.proxy_log_path().name == "proxy.log"


def test_stdio_log_path_is_per_port() -> None:
    assert _paths.proxy_stdio_log_path(8888).name == "proxy-stdio-8888.log"
    assert _paths.proxy_stdio_log_path().name == "proxy-stdio.log"
    # wrap helper agrees and keeps stdio beside the runtime log.
    stdio = wrap_cli._get_proxy_stdio_log_path(8888)
    assert stdio.name == "proxy-stdio-8888.log"
    assert stdio.parent == wrap_cli._get_log_path(8888).parent


def test_setup_file_logging_targets_per_port_file(workspace: Path) -> None:
    logger = logging.getLogger("headroom")
    original = list(logger.handlers)
    for h in original:
        if isinstance(h, RotatingFileHandler):
            logger.removeHandler(h)
    try:
        _setup_file_logging(8888)
        rotating = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
        assert rotating, "expected a RotatingFileHandler to be attached"
        assert Path(rotating[-1].baseFilename).name == "proxy-8888.log"
    finally:
        for h in [x for x in logger.handlers if isinstance(x, RotatingFileHandler)]:
            h.close()
            logger.removeHandler(h)
        for h in original:
            logger.addHandler(h)


def test_setup_file_logging_reroutes_on_port_change(workspace: Path) -> None:
    """A second proxy/app in one process keeps per-port keying via the real path.

    Regression: when the ``RotatingFileHandler`` was constructed before the
    dedup guard, the second ``_setup_file_logging(port)`` left an empty stray
    ``proxy-<port>.log``, leaked the handler fd, and routed the new port's
    records into the FIRST port's file. This drives the production helper twice
    (as two ``create_app`` calls in one interpreter do) and asserts each port's
    records land in — and only in — its own file.
    """
    log_dir = workspace / "logs"
    logger = logging.getLogger("headroom")
    original = list(logger.handlers)
    for h in original:
        if isinstance(h, RotatingFileHandler):
            logger.removeHandler(h)
    try:
        _setup_file_logging(8888)
        logger.info("record-for-8888")
        _setup_file_logging(9999)  # a second proxy/app in the same process
        logger.info("record-for-9999")

        # Exactly one file handler is ever attached, now pointing at 9999.
        rotating = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
        assert len(rotating) == 1
        assert Path(rotating[0].baseFilename).name == "proxy-9999.log"
        rotating[0].flush()

        text_8888 = (log_dir / "proxy-8888.log").read_text()
        text_9999 = (log_dir / "proxy-9999.log").read_text()
        # Each record is in its own file; the 9999 record did NOT leak into the
        # 8888 file, and the 8888 file is not the empty stray of the old bug.
        assert "record-for-8888" in text_8888 and "record-for-9999" not in text_8888
        assert "record-for-9999" in text_9999 and "record-for-8888" not in text_9999
    finally:
        for h in [x for x in logger.handlers if isinstance(x, RotatingFileHandler)]:
            h.close()
            logger.removeHandler(h)
        for h in original:
            logger.addHandler(h)


def test_parse_log_files_aggregates_per_port_and_legacy(workspace: Path) -> None:
    log_dir = workspace / "logs"
    ts = "2026-08-22 10:00:00,000"
    (log_dir / "proxy-8888.log").write_text(_perf_line(ts, "hr_a", "model-A") + "\n")
    (log_dir / "proxy-8889.log").write_text(_perf_line(ts, "hr_b", "model-B") + "\n")
    (log_dir / "proxy.log").write_text(_perf_line(ts, "hr_c", "model-C") + "\n")
    # stdio captures and any other non-PERF proxy-*.log are excluded by the
    # positive filename filter, not just a proxy-stdio blacklist.
    (log_dir / "proxy-stdio-8888.log").write_text(_perf_line(ts, "hr_stdio", "model-STDIO") + "\n")
    (log_dir / "proxy-stdio.log").write_text(_perf_line(ts, "hr_stdio2", "model-STDIO2") + "\n")
    (log_dir / "proxy-errors.log").write_text(_perf_line(ts, "hr_err", "model-ERR") + "\n")

    report = analyzer.parse_log_files(last_n_hours=0.0)  # 0 => no cutoff, all data
    models = {r.model for r in report.perf_records}
    rids = {r.request_id for r in report.perf_records}

    assert {"model-A", "model-B", "model-C"} <= models  # per-port + legacy aggregated
    # Non-PERF files never ingested.
    assert models.isdisjoint({"model-STDIO", "model-STDIO2", "model-ERR"})
    assert rids.isdisjoint({"hr_stdio", "hr_stdio2", "hr_err"})
