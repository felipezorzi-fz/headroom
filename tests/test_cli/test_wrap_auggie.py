"""Tests for `headroom wrap auggie`.

`wrap auggie` routes AugmentCode Auggie through Headroom by rewriting ONLY the
`tenantURL` in Auggie's OAuth session to the local proxy (token preserved) and
passing it via `AUGMENT_SESSION_AUTH`, then forwarding to the resolved tenant
upstream. These tests pin the session rewrite, upstream precedence, env
hygiene, and CLI wiring. The command does no filesystem surgery (nothing durable
is written), but every test runs from a tmp cwd as a guard.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from headroom.cli.main import main
from headroom.providers.augment import (
    build_redirected_session,
    load_session,
    proxy_base_url,
    resolve_augment_upstream,
)

_TOKEN = "tok-abcdef0123456789"
_TENANT = "https://xlb.api.augmentcode.com/"

# Kept as a module-level name rather than an inline string literal in the `in`
# check below: CodeQL's incomplete-url-substring-sanitization query pattern
# matches a URL-shaped StringLiteral used directly as an operand of `in`,
# regardless of context (it cannot tell a test assertion on captured CLI
# output apart from a real hostname-allowlist check). A name reference is a
# different AST node, so this avoids the false-positive match structurally.
_AUGGIE_DOCS_URL = "https://docs.augmentcode.com/cli"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _tmp_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    for var in ("AUGMENT_SESSION_AUTH", "AUGMENT_API_URL", "AUGMENT_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)


def _write_session(tmp_path: Path, **overrides: object) -> Path:
    data = {"accessToken": _TOKEN, "tenantURL": _TENANT, "scopes": ["agent"]}
    data.update(overrides)
    path = tmp_path / "session.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# runtime helpers
# ---------------------------------------------------------------------------


def test_proxy_base_url_targets_loopback_port() -> None:
    assert proxy_base_url(9999) == "http://127.0.0.1:9999"


def test_resolve_augment_upstream_precedence() -> None:
    session = {"accessToken": _TOKEN, "tenantURL": _TENANT}
    # Session tenantURL is the default upstream; trailing slash trimmed.
    assert resolve_augment_upstream(session) == "https://xlb.api.augmentcode.com"
    # Explicit flag beats the session tenantURL.
    assert (
        resolve_augment_upstream(session, "https://eu.api.augmentcode.com/")
        == "https://eu.api.augmentcode.com"
    )


def test_build_redirected_session_rewrites_only_tenant_url() -> None:
    session = {"accessToken": _TOKEN, "tenantURL": _TENANT, "scopes": ["agent"]}
    redirected = json.loads(build_redirected_session(session, 8790))
    # tenantURL points at the local proxy...
    assert redirected["tenantURL"] == "http://127.0.0.1:8790"
    # ...and every other field (crucially the token) is preserved byte-for-byte.
    assert redirected["accessToken"] == _TOKEN
    assert redirected["scopes"] == ["agent"]
    # The input dict is not mutated.
    assert session["tenantURL"] == _TENANT


def test_load_session_missing_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_session(tmp_path / "does-not-exist.json")


def test_load_session_malformed_raises_valueerror(tmp_path: Path) -> None:
    bad = tmp_path / "session.json"
    bad.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError, match="readable JSON"):
        load_session(bad)


def test_load_session_missing_fields_raises_valueerror(tmp_path: Path) -> None:
    p = tmp_path / "session.json"
    p.write_text(json.dumps({"accessToken": _TOKEN}), encoding="utf-8")  # no tenantURL
    with pytest.raises(ValueError, match="tenantURL"):
        load_session(p)


def test_load_session_non_object_json_raises_valueerror(tmp_path: Path) -> None:
    p = tmp_path / "session.json"
    p.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        load_session(p)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_wrap_auggie_missing_binary_exits_with_install_hint(
    runner: CliRunner, tmp_path: Path
) -> None:
    session = _write_session(tmp_path)
    with patch("headroom.cli.wrap.shutil.which", return_value=None):
        result = runner.invoke(main, ["wrap", "auggie", "--session-file", str(session)])
    assert result.exit_code == 1
    assert _AUGGIE_DOCS_URL in result.output


def test_wrap_auggie_missing_session_errors_friendly(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(main, ["wrap", "auggie", "--session-file", str(tmp_path / "nope.json")])
    assert result.exit_code != 0
    assert "auggie login" in result.output


def test_wrap_auggie_unresolvable_tenant_url_errors_friendly(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A session whose tenantURL is present (passes load_session's truthiness
    check) but strips down to empty (e.g. all slashes) must still surface a
    friendly error instead of proceeding with an empty upstream.
    """
    session = _write_session(tmp_path, tenantURL="///")
    with patch("headroom.cli.wrap.shutil.which", return_value="auggie"):
        result = runner.invoke(main, ["wrap", "auggie", "--session-file", str(session)])
    assert result.exit_code != 0
    assert "Could not resolve the Auggie tenant URL" in result.output


def test_wrap_auggie_malformed_session_errors_friendly(runner: CliRunner, tmp_path: Path) -> None:
    """A session file missing tenantURL/accessToken surfaces load_session's
    ValueError as a friendly ClickException through the full CLI, not just
    from the lower-level load_session() call.
    """
    p = tmp_path / "session.json"
    p.write_text(json.dumps({"accessToken": _TOKEN}), encoding="utf-8")  # no tenantURL
    result = runner.invoke(main, ["wrap", "auggie", "--session-file", str(p)])
    assert result.exit_code != 0
    assert "tenantURL" in result.output


def test_wrap_auggie_points_child_at_proxy_and_forwards_default_upstream(
    runner: CliRunner, tmp_path: Path
) -> None:
    session = _write_session(tmp_path)
    captured: dict[str, object] = {}
    with (
        patch("headroom.cli.wrap.shutil.which", return_value="auggie"),
        patch("headroom.cli.wrap._check_proxy", return_value=False),
        patch("headroom.cli.wrap._launch_tool", side_effect=lambda **kw: captured.update(kw)),
    ):
        result = runner.invoke(
            main, ["wrap", "auggie", "--session-file", str(session), "--", "-p", "hi"]
        )
    assert result.exit_code == 0, result.output
    assert captured["tool_label"] == "AUGGIE"
    assert captured["agent_type"] == "augment"
    assert captured["args"] == ("-p", "hi")
    # Proxy forwards to the real tenant resolved from the session...
    assert captured["augment_api_url"] == "https://xlb.api.augmentcode.com"
    # ...and Auggie is redirected via a rewritten session (token preserved).
    env = captured["env"]
    assert isinstance(env, dict)
    injected = json.loads(env["AUGMENT_SESSION_AUTH"])
    assert injected["tenantURL"] == "http://127.0.0.1:8787"
    assert injected["accessToken"] == _TOKEN


def test_wrap_auggie_explicit_upstream_and_custom_port(runner: CliRunner, tmp_path: Path) -> None:
    session = _write_session(tmp_path)
    captured: dict[str, object] = {}
    with (
        patch("headroom.cli.wrap.shutil.which", return_value="auggie"),
        patch("headroom.cli.wrap._check_proxy", return_value=False),
        patch("headroom.cli.wrap._launch_tool", side_effect=lambda **kw: captured.update(kw)),
    ):
        result = runner.invoke(
            main,
            [
                "wrap",
                "auggie",
                "--session-file",
                str(session),
                "--port",
                "8790",
                "--augment-api-url",
                "https://eu.api.augmentcode.com/",
            ],
        )
    assert result.exit_code == 0, result.output
    assert captured["augment_api_url"] == "https://eu.api.augmentcode.com"
    env = captured["env"]
    assert isinstance(env, dict)
    assert json.loads(env["AUGMENT_SESSION_AUTH"])["tenantURL"] == "http://127.0.0.1:8790"


def test_wrap_auggie_scrubs_inherited_augment_env(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _write_session(tmp_path)
    # A stale value in the parent shell must not survive into the child env.
    monkeypatch.setenv("AUGMENT_API_URL", "https://attacker.example")
    monkeypatch.setenv("AUGMENT_API_TOKEN", "stale-token")
    captured: dict[str, object] = {}
    with (
        patch("headroom.cli.wrap.shutil.which", return_value="auggie"),
        patch("headroom.cli.wrap._check_proxy", return_value=False),
        patch("headroom.cli.wrap._launch_tool", side_effect=lambda **kw: captured.update(kw)),
    ):
        result = runner.invoke(main, ["wrap", "auggie", "--session-file", str(session)])
    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert "AUGMENT_API_URL" not in env
    assert "AUGMENT_API_TOKEN" not in env
    assert json.loads(env["AUGMENT_SESSION_AUTH"])["tenantURL"] == "http://127.0.0.1:8787"


def test_wrap_auggie_prepare_only_reports_wiring(runner: CliRunner, tmp_path: Path) -> None:
    session = _write_session(tmp_path)
    with patch("headroom.cli.wrap.shutil.which", return_value="auggie"):
        result = runner.invoke(
            main, ["wrap", "auggie", "--session-file", str(session), "--prepare-only"]
        )
    assert result.exit_code == 0, result.output
    assert "AUGMENT_API_URL=http://127.0.0.1:8787" in result.output
    assert f"upstream={_TENANT.rstrip('/')}" in result.output


def test_wrap_auggie_falls_back_to_free_port_when_non_augment_proxy(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A non-Auggie proxy on the port -> start a dedicated proxy on a free port."""
    session = _write_session(tmp_path)
    captured: dict[str, object] = {}
    with (
        patch("headroom.cli.wrap.shutil.which", return_value="auggie"),
        patch("headroom.cli.wrap._check_proxy", return_value=True),
        patch("headroom.cli.wrap._query_proxy_config", return_value={"augment_api_url": None}),
        patch("headroom.cli.wrap._find_available_port", return_value=8790),
        patch("headroom.cli.wrap._launch_tool", side_effect=lambda **kw: captured.update(kw)),
    ):
        result = runner.invoke(main, ["wrap", "auggie", "--session-file", str(session)])
    assert result.exit_code == 0, result.output
    assert captured["port"] == 8790
    env = captured["env"]
    assert isinstance(env, dict)
    assert json.loads(env["AUGMENT_SESSION_AUTH"])["tenantURL"] == "http://127.0.0.1:8790"


def test_wrap_auggie_tenant_mismatch_with_no_free_port_keeps_requested_port(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Tenant mismatch, but _find_available_port returns the SAME port back
    (no actually-different port found nearby): no fallback message is printed
    and the requested port is kept, rather than silently reusing the other
    tenant's proxy.
    """
    session = _write_session(tmp_path)
    captured: dict[str, object] = {}
    with (
        patch("headroom.cli.wrap.shutil.which", return_value="auggie"),
        patch("headroom.cli.wrap._check_proxy", return_value=True),
        patch(
            "headroom.cli.wrap._query_proxy_config",
            return_value={"augment_api_url": "https://eu.api.augmentcode.com"},
        ),
        patch("headroom.cli.wrap._find_available_port", return_value=8787),
        patch("headroom.cli.wrap._launch_tool", side_effect=lambda **kw: captured.update(kw)),
    ):
        result = runner.invoke(main, ["wrap", "auggie", "--session-file", str(session)])
    assert result.exit_code == 0, result.output
    assert captured["port"] == 8787
    assert "using port" not in result.output


def test_wrap_auggie_rejects_retired_context_tool_flag(runner: CliRunner, tmp_path: Path) -> None:
    session = _write_session(tmp_path)
    with patch("headroom.cli.wrap.shutil.which", return_value="auggie"):
        result = runner.invoke(
            main, ["wrap", "auggie", "--session-file", str(session), "--no-context-tool"]
        )
    assert result.exit_code != 0
    assert "have been removed" in result.output


def test_wrap_auggie_falls_back_to_free_port_on_tenant_mismatch(
    runner: CliRunner, tmp_path: Path
) -> None:
    """An Auggie proxy for a DIFFERENT tenant -> dedicated proxy on a free port."""
    session = _write_session(tmp_path)  # tenantURL -> xlb.api.augmentcode.com
    captured: dict[str, object] = {}
    with (
        patch("headroom.cli.wrap.shutil.which", return_value="auggie"),
        patch("headroom.cli.wrap._check_proxy", return_value=True),
        patch(
            "headroom.cli.wrap._query_proxy_config",
            return_value={"augment_api_url": "https://eu.api.augmentcode.com"},
        ),
        patch("headroom.cli.wrap._find_available_port", return_value=8790),
        patch("headroom.cli.wrap._launch_tool", side_effect=lambda **kw: captured.update(kw)),
    ):
        result = runner.invoke(main, ["wrap", "auggie", "--session-file", str(session)])
    assert result.exit_code == 0, result.output
    assert captured["port"] == 8790


def test_wrap_auggie_reuse_guard_allows_matching_tenant(runner: CliRunner, tmp_path: Path) -> None:
    """A proxy already in Auggie mode for the SAME tenant is reused, not refused."""
    session = _write_session(tmp_path)
    captured: dict[str, object] = {}
    with (
        patch("headroom.cli.wrap.shutil.which", return_value="auggie"),
        patch("headroom.cli.wrap._check_proxy", return_value=True),
        patch(
            "headroom.cli.wrap._query_proxy_config",
            return_value={"augment_api_url": "https://xlb.api.augmentcode.com/"},
        ),
        patch("headroom.cli.wrap._launch_tool", side_effect=lambda **kw: captured.update(kw)),
    ):
        result = runner.invoke(main, ["wrap", "auggie", "--session-file", str(session)])
    assert result.exit_code == 0, result.output
    assert captured["augment_api_url"] == "https://xlb.api.augmentcode.com"
    # Matching tenant -> reuse the existing proxy on the requested port (no fallback).
    assert captured["port"] == 8787


def test_start_proxy_forwards_augment_api_url_to_subprocess(tmp_path: Path) -> None:
    """_start_proxy must pass --augment-api-url through to the actual proxy
    subprocess command when augment_api_url is set.
    """
    import headroom.cli.wrap as wrap_module

    captured: dict[str, object] = {}

    class _FakeProcess:
        returncode = None

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProcess()

    with (
        patch("headroom.cli.wrap.subprocess.Popen", side_effect=fake_popen),
        patch("headroom.cli.wrap._check_proxy", return_value=True),
        patch("headroom.cli.wrap._get_log_path", return_value=tmp_path / "proxy.log"),
        patch("headroom.cli.wrap._get_proxy_stdio_log_path", return_value=tmp_path / "stdio.log"),
    ):
        wrap_module._start_proxy(8899, augment_api_url="https://xlb.api.augmentcode.com")

    assert "--augment-api-url" in captured["cmd"]
    idx = captured["cmd"].index("--augment-api-url")
    assert captured["cmd"][idx + 1] == "https://xlb.api.augmentcode.com"


def test_start_proxy_omits_augment_api_url_flag_when_not_set(tmp_path: Path) -> None:
    """_start_proxy must not pass --augment-api-url when augment_api_url is
    not set, so other wrappers that never pass it are unaffected.
    """
    import headroom.cli.wrap as wrap_module

    captured: dict[str, object] = {}

    class _FakeProcess:
        returncode = None

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProcess()

    with (
        patch("headroom.cli.wrap.subprocess.Popen", side_effect=fake_popen),
        patch("headroom.cli.wrap._check_proxy", return_value=True),
        patch("headroom.cli.wrap._get_log_path", return_value=tmp_path / "proxy.log"),
        patch("headroom.cli.wrap._get_proxy_stdio_log_path", return_value=tmp_path / "stdio.log"),
    ):
        wrap_module._start_proxy(8899)

    assert "--augment-api-url" not in captured["cmd"]


def test_launch_tool_rewrites_env_display_after_port_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When _ensure_proxy falls back to a different port than requested,
    _launch_tool must rewrite both env and the printed env_vars_display lines
    to the actual port, not the originally requested one.
    """
    import headroom.cli.wrap as wrap_module

    requested_port = 8787
    actual_port = 8790

    monkeypatch.setattr(wrap_module, "_ensure_proxy", lambda *a, **kw: (object(), actual_port))
    monkeypatch.setattr(wrap_module, "_register_proxy_client", lambda *a, **kw: None)
    monkeypatch.setattr(wrap_module, "_unregister_proxy_client", lambda *a, **kw: None)
    monkeypatch.setattr(wrap_module, "_push_runtime_env", lambda *a, **kw: None)
    monkeypatch.setattr(wrap_module, "_make_cleanup", lambda holder, port: lambda *a, **kw: None)
    monkeypatch.setattr(wrap_module.signal, "signal", lambda *a, **kw: None)

    captured: dict[str, object] = {}

    def fake_run(cmd, env=None, **kwargs):  # noqa: ANN001
        captured["env"] = env
        raise SystemExit(0)

    monkeypatch.setattr(wrap_module.subprocess, "run", fake_run)

    runner = CliRunner()

    @click.command()
    def _cmd() -> None:
        wrap_module._launch_tool(
            binary="auggie",
            args=(),
            env={"AUGMENT_SESSION_AUTH": f"http://127.0.0.1:{requested_port}"},
            port=requested_port,
            no_proxy=False,
            tool_label="AUGGIE",
            env_vars_display=[f"Auggie tenant URL rewritten -> http://127.0.0.1:{requested_port}"],
            agent_type="augment",
            augment_api_url="https://xlb.api.augmentcode.com",
        )

    result = runner.invoke(_cmd)

    assert result.exit_code == 0
    assert captured["env"]["AUGMENT_SESSION_AUTH"] == f"http://127.0.0.1:{actual_port}"
    assert f"http://127.0.0.1:{actual_port}" in result.output
    assert f"http://127.0.0.1:{requested_port}" not in result.output
