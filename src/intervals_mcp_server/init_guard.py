"""
ASGI middleware that buffers MCP messages until the initialization handshake
is complete.

When Claude Desktop reconnects to a running SSE server it sometimes fires
tool calls before finishing the MCP initialize → notifications/initialized
handshake.  The MCP SDK rejects those early requests with:

    "Failed to validate request: Received request before initialization
    was complete"

This middleware sits in front of the Starlette/SSE app and holds every
non-handshake POST to /messages back until the per-session
``notifications/initialized`` notification arrives (or until
``INIT_TIMEOUT`` seconds have elapsed, after which the request is
forwarded anyway to avoid an infinite stall).

Additionally, every new GET /sse connection (which always starts a fresh
session) triggers a cleanup of zombie sessions — sessions whose event was
never set because the previous SSE connection dropped before the handshake
completed.

Only the SSE transport is affected; stdio is not routed through this
middleware at all.
"""

import asyncio
import json
import logging
from urllib.parse import parse_qs
from typing import Any

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("intervals_icu_mcp_server")

# Maximum time (seconds) to wait for initialization before forwarding anyway.
INIT_TIMEOUT: float = 30.0

# Per-session asyncio.Event objects.  Keyed by session_id (UUID hex string).
# An event becomes *set* once the client sends notifications/initialized.
_session_events: dict[str, asyncio.Event] = {}


def _get_or_create_event(session_id: str) -> asyncio.Event:
    if session_id not in _session_events:
        _session_events[session_id] = asyncio.Event()
    return _session_events[session_id]


def _cleanup_zombie_sessions() -> None:
    """Remove sessions whose handshake never completed (event was never set).

    Called on every new GET /sse connection, which always creates a brand-new
    session.  Any previously tracked session whose event is still unset is a
    zombie left behind by a dropped SSE connection.
    """
    zombie_ids = [sid for sid, ev in _session_events.items() if not ev.is_set()]
    for sid in zombie_ids:
        del _session_events[sid]
    if zombie_ids:
        logger.info(
            "Cleaned up %d zombie session(s) on new SSE connect: %s",
            len(zombie_ids),
            zombie_ids,
        )


class InitGuardMiddleware:
    """
    Pure-ASGI middleware that delays non-handshake MCP messages until the
    per-session initialization sequence completes.

    Behaviour per incoming request:

    * GET  /sse                     → clean up zombie sessions, then pass
                                      through unchanged.
    * POST /messages initialize     → pass through immediately; create the
                                      per-session event.
    * POST /messages notifications/initialized
                                    → set the per-session event (unblocking
                                      waiting requests), then pass through.
    * POST /messages <anything else>→ wait up to ``INIT_TIMEOUT`` seconds for
                                      the event to be set, then pass through
                                      (with or without a timeout warning).
    * everything else               → pass through unchanged.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method: str = scope.get("method", "")
        path: str = scope.get("path", "")

        # New SSE connection → evict zombie sessions from previous connections.
        if method == "GET" and "/sse" in path:
            _cleanup_zombie_sessions()
            await self.app(scope, receive, send)
            return

        # Only intercept POST requests to the messages endpoint.
        if method != "POST" or "/messages" not in path:
            await self.app(scope, receive, send)
            return

        # Parse the session_id query parameter that the SSE transport adds.
        query_string: str = scope.get("query_string", b"").decode("utf-8", errors="replace")
        params = parse_qs(query_string)
        session_id: str = params.get("session_id", ["unknown"])[0]

        # Buffer the full request body so we can inspect it and replay it
        # for the inner app (the ASGI receive stream is consumed once only).
        body_chunks: list[bytes] = []
        more_body = True
        while more_body:
            message: dict[str, Any] = await receive()
            body_chunks.append(message.get("body", b""))
            more_body = message.get("more_body", False)
        body = b"".join(body_chunks)

        # Determine the JSON-RPC method without raising on malformed input.
        rpc_method = ""
        try:
            rpc_msg = json.loads(body)
            if isinstance(rpc_msg, dict):
                rpc_method = rpc_msg.get("method", "")
        except (json.JSONDecodeError, ValueError):
            pass

        if rpc_method == "initialize":
            # First handshake step — ensure the per-session event exists so
            # that concurrent requests can already start waiting on it.
            _get_or_create_event(session_id)
            logger.info(
                "MCP initialize received (session=%s) — waiting for notifications/initialized",
                session_id,
            )

        elif rpc_method == "notifications/initialized":
            # Handshake complete — unblock every request that is waiting.
            event = _get_or_create_event(session_id)
            event.set()
            logger.info("MCP initialization complete (session=%s)", session_id)

        else:
            # Any other request (tool calls, list_tools, …) — wait until the
            # handshake finishes.
            event = _get_or_create_event(session_id)
            if not event.is_set():
                logger.info(
                    "Holding '%s' until MCP init completes (session=%s, timeout=%.1fs)",
                    rpc_method,
                    session_id,
                    INIT_TIMEOUT,
                )
                try:
                    await asyncio.wait_for(event.wait(), timeout=INIT_TIMEOUT)
                    logger.debug(
                        "Init complete, forwarding '%s' (session=%s)", rpc_method, session_id
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Init wait timed out after %.1fs for '%s' (session=%s) — forwarding anyway",
                        INIT_TIMEOUT,
                        rpc_method,
                        session_id,
                    )

        # Replay the buffered body for the inner ASGI app.
        # On the first receive() call the full body is returned.
        # On any subsequent call (e.g. http.disconnect check) the original
        # receive callable is forwarded so the request lifecycle completes
        # normally.
        body_sent = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            # Body already sent; delegate to the upstream receive for
            # lifecycle events (http.disconnect, etc.).
            return await receive()

        await self.app(scope, replay_receive, send)
