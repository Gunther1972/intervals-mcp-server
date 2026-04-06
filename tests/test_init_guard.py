"""
Unit tests for InitGuardMiddleware (src/intervals_mcp_server/init_guard.py).

The middleware must:
1. Pass ``initialize`` requests through immediately.
2. Set the per-session event and pass ``notifications/initialized`` through.
3. Hold any other request until the event is set.
4. Forward requests after the timeout even if the event was never set.
5. Leave non-POST and non-/messages paths untouched.
"""

import asyncio
import json
import pathlib
import sys
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from intervals_mcp_server.init_guard import (  # pylint: disable=wrong-import-position
    INIT_TIMEOUT,
    InitGuardMiddleware,
    _session_events,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scope(method: str = "POST", path: str = "/messages/", session_id: str = "abc123") -> dict:
    return {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": f"session_id={session_id}".encode(),
    }


def _make_receive(body: bytes):
    """Return an ASGI receive callable that yields a single HTTP request body."""
    called = False

    async def receive():
        nonlocal called
        if not called:
            called = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    return receive


async def _collect_send(send_calls: list):
    """Return an ASGI send callable that records what was sent."""
    async def send(message):
        send_calls.append(message)
    return send


def _rpc_body(method: str, params: dict | None = None) -> bytes:
    msg: dict = {"jsonrpc": "2.0", "method": method, "id": 1}
    if params is not None:
        msg["params"] = params
    return json.dumps(msg).encode()


# ---------------------------------------------------------------------------
# Fixture: fresh event dict per test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_session_events():
    _session_events.clear()
    yield
    _session_events.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_initialize_passes_through_immediately():
    """initialize request must reach the inner app without delay."""
    received_bodies: list[bytes] = []

    async def inner_app(scope, receive, send):
        message = await receive()
        received_bodies.append(message["body"])

    middleware = InitGuardMiddleware(inner_app)
    body = _rpc_body("initialize")
    send_calls: list = []

    await middleware(
        _scope(session_id="s1"),
        _make_receive(body),
        (await _collect_send(send_calls)),
    )

    assert received_bodies == [body]
    # Event should exist but not yet set
    assert "s1" in _session_events
    assert not _session_events["s1"].is_set()


@pytest.mark.asyncio
async def test_initialized_notification_sets_event():
    """notifications/initialized must set the per-session event."""
    async def inner_app(scope, receive, send):
        await receive()

    middleware = InitGuardMiddleware(inner_app)
    body = _rpc_body("notifications/initialized")
    send_calls: list = []

    await middleware(
        _scope(session_id="s2"),
        _make_receive(body),
        (await _collect_send(send_calls)),
    )

    assert "s2" in _session_events
    assert _session_events["s2"].is_set()


@pytest.mark.asyncio
async def test_tool_call_waits_until_initialized():
    """A tool call arriving before initialized must be held until the event is set."""
    # Pre-create an un-set event for the session
    event = asyncio.Event()
    _session_events["s3"] = event

    reached_inner_at: list[float] = []

    async def inner_app(scope, receive, send):
        reached_inner_at.append(asyncio.get_event_loop().time())
        await receive()

    middleware = InitGuardMiddleware(inner_app)
    body = _rpc_body("tools/call", {"name": "get_activities"})
    send_calls: list = []

    # Release the event after a short delay
    async def release_after_delay():
        await asyncio.sleep(0.05)
        event.set()

    start = asyncio.get_event_loop().time()
    await asyncio.gather(
        middleware(
            _scope(session_id="s3"),
            _make_receive(body),
            (await _collect_send(send_calls)),
        ),
        release_after_delay(),
    )

    elapsed = reached_inner_at[0] - start
    assert elapsed >= 0.04, f"Request was not held long enough (elapsed={elapsed:.3f}s)"


@pytest.mark.asyncio
async def test_tool_call_passes_through_when_already_initialized():
    """A tool call arriving after initialized must pass through without delay."""
    event = asyncio.Event()
    event.set()  # already initialized
    _session_events["s4"] = event

    reached_inner: list[bool] = []

    async def inner_app(scope, receive, send):
        reached_inner.append(True)
        await receive()

    middleware = InitGuardMiddleware(inner_app)
    body = _rpc_body("tools/list")
    send_calls: list = []

    await middleware(
        _scope(session_id="s4"),
        _make_receive(body),
        (await _collect_send(send_calls)),
    )

    assert reached_inner == [True]


@pytest.mark.asyncio
async def test_timeout_forwards_request_after_expiry(monkeypatch):
    """After INIT_TIMEOUT the request must be forwarded even if event was never set."""
    monkeypatch.setattr("intervals_mcp_server.init_guard.INIT_TIMEOUT", 0.05)

    async def inner_app(scope, receive, send):
        await receive()

    middleware = InitGuardMiddleware(inner_app)
    body = _rpc_body("tools/call", {"name": "get_wellness_data"})
    send_calls: list = []

    start = time.monotonic()
    await middleware(
        _scope(session_id="s5"),
        _make_receive(body),
        (await _collect_send(send_calls)),
    )
    elapsed = time.monotonic() - start

    # Should have waited roughly INIT_TIMEOUT (0.05s) and then forwarded
    assert elapsed >= 0.04, f"Did not wait for timeout (elapsed={elapsed:.3f}s)"


@pytest.mark.asyncio
async def test_non_post_passes_through_unchanged():
    """GET requests (SSE endpoint) must bypass the middleware entirely."""
    called: list[bool] = []

    async def inner_app(scope, receive, send):
        called.append(True)

    middleware = InitGuardMiddleware(inner_app)
    await middleware(
        _scope(method="GET", path="/sse", session_id="s6"),
        _make_receive(b""),
        (await _collect_send([])),
    )

    assert called == [True]
    assert "s6" not in _session_events


@pytest.mark.asyncio
async def test_non_messages_path_passes_through_unchanged():
    """POSTs to paths other than /messages must bypass the guard."""
    called: list[bool] = []

    async def inner_app(scope, receive, send):
        called.append(True)

    middleware = InitGuardMiddleware(inner_app)
    await middleware(
        _scope(method="POST", path="/health", session_id="s7"),
        _make_receive(b"{}"),
        (await _collect_send([])),
    )

    assert called == [True]
    assert "s7" not in _session_events


@pytest.mark.asyncio
async def test_malformed_json_passes_through():
    """A POST with invalid JSON must still reach the inner app (no crash)."""
    called: list[bool] = []

    async def inner_app(scope, receive, send):
        called.append(True)
        await receive()

    middleware = InitGuardMiddleware(inner_app)
    await middleware(
        _scope(session_id="s8"),
        _make_receive(b"not-json"),
        (await _collect_send([])),
    )

    assert called == [True]


@pytest.mark.asyncio
async def test_body_is_replayed_to_inner_app():
    """The inner app must receive the exact same body bytes that arrived."""
    replayed: list[bytes] = []

    async def inner_app(scope, receive, send):
        msg = await receive()
        replayed.append(msg["body"])

    middleware = InitGuardMiddleware(inner_app)
    body = _rpc_body("notifications/initialized")
    send_calls: list = []

    await middleware(
        _scope(session_id="s9"),
        _make_receive(body),
        (await _collect_send(send_calls)),
    )

    assert replayed == [body]
