"""Tests for the wrap-spawned proxy orphan watchdog."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from headroom.proxy import orphan_watchdog as ow


def _write_marker(clients_dir: Path, pid: int, **extra: object) -> Path:
    clients_dir.mkdir(parents=True, exist_ok=True)
    marker = clients_dir / f"{pid}.json"
    payload = {"pid": pid, "started_at": 1.0}
    payload.update(extra)
    marker.write_text(json.dumps(payload), encoding="utf-8")
    return marker


class TestEnvConfig:
    def test_enabled_requires_wrap_owned_flag(self, monkeypatch) -> None:
        monkeypatch.delenv(ow.WRAP_OWNED_ENV, raising=False)
        assert ow.orphan_watchdog_enabled() is False
        monkeypatch.setenv(ow.WRAP_OWNED_ENV, "1")
        assert ow.orphan_watchdog_enabled() is True
        monkeypatch.setenv(ow.WRAP_OWNED_ENV, "0")
        assert ow.orphan_watchdog_enabled() is False
        # Explicit mapping wins over os.environ.
        assert ow.orphan_watchdog_enabled({ow.WRAP_OWNED_ENV: "true"}) is True

    def test_grace_seconds_default_override_and_floor(self, monkeypatch) -> None:
        monkeypatch.delenv(ow.GRACE_SECONDS_ENV, raising=False)
        assert ow.orphan_grace_seconds() == ow.DEFAULT_GRACE_SECONDS
        monkeypatch.setenv(ow.GRACE_SECONDS_ENV, "120")
        assert ow.orphan_grace_seconds() == 120.0
        # Below the floor: clamped, so a typo cannot exit a proxy under a
        # client that has not registered yet.
        monkeypatch.setenv(ow.GRACE_SECONDS_ENV, "5")
        assert ow.orphan_grace_seconds() == ow.MIN_GRACE_SECONDS
        monkeypatch.setenv(ow.GRACE_SECONDS_ENV, "not-a-number")
        assert ow.orphan_grace_seconds() == ow.DEFAULT_GRACE_SECONDS


class TestLiveClientPids:
    def test_live_marker_is_kept(self, tmp_path) -> None:
        _write_marker(tmp_path, os.getpid())
        assert ow.live_client_pids(tmp_path) == [os.getpid()]

    def test_dead_pid_marker_is_pruned(self, tmp_path) -> None:
        # PID 2**31 - 1 is never alive.
        marker = _write_marker(tmp_path, 2**31 - 1)
        assert ow.live_client_pids(tmp_path) == []
        assert not marker.exists()

    def test_recycled_pid_marker_is_pruned(self, tmp_path, monkeypatch) -> None:
        # Live PID (ours) but a recorded identity that provably differs.
        monkeypatch.setattr(ow, "identity_mismatch", lambda src, recorded, pid: True)
        marker = _write_marker(tmp_path, os.getpid(), start_src="psutil", start_time=1.0)
        assert ow.live_client_pids(tmp_path) == []
        assert not marker.exists()

    def test_garbage_and_non_numeric_markers_are_ignored(self, tmp_path) -> None:
        (tmp_path / "not-a-pid.json").write_text("{}", encoding="utf-8")
        (tmp_path / f"{os.getpid()}.json").write_text("not json", encoding="utf-8")
        # Unparseable marker JSON is not proof of recycling: PID is alive, kept.
        assert ow.live_client_pids(tmp_path) == [os.getpid()]

    def test_missing_dir_is_empty(self, tmp_path) -> None:
        assert ow.live_client_pids(tmp_path / "nope") == []


class TestWatchdogLoop:
    @staticmethod
    def _proxy(port: int = 8787, active_sessions: int = 0) -> SimpleNamespace:
        ws = SimpleNamespace(active_count=lambda: active_sessions)
        return SimpleNamespace(config=SimpleNamespace(port=port), ws_sessions=ws)

    def test_stops_after_grace_with_no_clients(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
        stopped: list[bool] = []

        asyncio.run(
            ow.orphan_watchdog_loop(
                self._proxy(),
                grace_seconds=0.05,
                interval_seconds=0.01,
                stop=lambda: stopped.append(True),
            )
        )

        assert stopped == [True]

    def test_no_stop_while_client_alive(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
        _write_marker(tmp_path, os.getpid())
        stopped: list[bool] = []

        async def run() -> None:
            task = asyncio.create_task(
                ow.orphan_watchdog_loop(
                    self._proxy(),
                    grace_seconds=0.3,
                    interval_seconds=0.01,
                    stop=lambda: stopped.append(True),
                )
            )
            await asyncio.sleep(0.4)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())

        assert stopped == []

    def test_no_stop_while_ws_session_active(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
        stopped: list[bool] = []

        async def run() -> None:
            task = asyncio.create_task(
                ow.orphan_watchdog_loop(
                    self._proxy(active_sessions=1),
                    grace_seconds=0.3,
                    interval_seconds=0.01,
                    stop=lambda: stopped.append(True),
                )
            )
            await asyncio.sleep(0.4)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())

        assert stopped == []

    def test_no_stop_while_requests_are_served(self, tmp_path, monkeypatch) -> None:
        """Direct-HTTP / SSH-forwarded clients leave no marker and may hold no
        WS session; served-request counter movement must still hold the exit."""
        monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
        counter = {"n": 0}
        proxy = SimpleNamespace(
            config=SimpleNamespace(port=8787),
            ws_sessions=None,
            _request_counter=0,
        )
        stopped: list[bool] = []

        async def serve_traffic() -> None:
            while True:
                await asyncio.sleep(0.05)
                counter["n"] += 1
                proxy._request_counter = counter["n"]

        async def run() -> None:
            traffic = asyncio.create_task(serve_traffic())
            task = asyncio.create_task(
                ow.orphan_watchdog_loop(
                    proxy,
                    grace_seconds=0.15,
                    interval_seconds=0.01,
                    stop=lambda: stopped.append(True),
                )
            )
            await asyncio.sleep(0.4)
            task.cancel()
            traffic.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())

        assert stopped == []

    def test_grace_restarts_when_client_returns(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
        stopped: list[bool] = []

        async def run() -> None:
            task = asyncio.create_task(
                ow.orphan_watchdog_loop(
                    self._proxy(),
                    grace_seconds=0.5,
                    interval_seconds=0.01,
                    stop=lambda: stopped.append(True),
                )
            )
            # Let the idle clock accumulate, then register a live client: the
            # clock must reset, not fire late.
            await asyncio.sleep(0.2)
            _write_marker(tmp_path, os.getpid())
            await asyncio.sleep(0.7)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())

        assert stopped == []

    def test_missing_ws_registry_counts_as_idle(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
        proxy = SimpleNamespace(config=SimpleNamespace(port=8787), ws_sessions=None)
        stopped: list[bool] = []

        asyncio.run(
            ow.orphan_watchdog_loop(
                proxy,
                grace_seconds=0.05,
                interval_seconds=0.01,
                stop=lambda: stopped.append(True),
            )
        )

        assert stopped == [True]
