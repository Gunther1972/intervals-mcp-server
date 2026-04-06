"""
Server setup and initialization for Intervals.icu MCP Server.

This module handles transport configuration and server startup logic.
"""

import os
import logging

import anyio
from mcp.server.fastmcp import FastMCP  # pylint: disable=import-error

from intervals_mcp_server.init_guard import InitGuardMiddleware
from intervals_mcp_server.utils.types import TransportAliases

logger = logging.getLogger("intervals_icu_mcp_server")


def setup_transport() -> TransportAliases:
    """
    Setup and validate the MCP transport configuration.

    Reads MCP_TRANSPORT environment variable and validates it against
    supported transport types.

    Returns:
        TransportAliases: The selected transport type.

    Raises:
        ValueError: If the transport type is not supported.
    """
    transport_env = os.getenv("MCP_TRANSPORT", TransportAliases.STDIO.value).lower()
    try:
        transport_alias = TransportAliases(transport_env)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in TransportAliases)
        raise ValueError(f"Unsupported MCP_TRANSPORT value. Use one of: {allowed}.") from exc

    # Map HTTP to STREAMABLE_HTTP
    selected_transport = (
        TransportAliases.STREAMABLE_HTTP
        if transport_alias == TransportAliases.HTTP
        else transport_alias
    )

    return selected_transport


def start_server(mcp_instance: FastMCP, transport: TransportAliases) -> None:
    # Forceer host en poort voor Render
    mcp_instance.settings.host = os.getenv("FASTMCP_HOST", "0.0.0.0")
    port = os.getenv("PORT")
    if port:
        mcp_instance.settings.port = int(port)

    host = mcp_instance.settings.host
    port = mcp_instance.settings.port

    if transport == TransportAliases.STDIO:
        logger.info("Starting MCP server with stdio transport.")
        mcp_instance.run()
    elif transport == TransportAliases.SSE:
        mount_path = os.getenv("MCP_SSE_MOUNT_PATH")
        logger.info(
            "Starting MCP server with SSE transport at http://%s:%s%s (messages: %s).",
            host,
            port,
            mcp_instance.settings.sse_path,
            mcp_instance.settings.message_path,
        )
        _run_sse_with_init_guard(mcp_instance, mount_path)
    else:  # STREAMABLE_HTTP
        logger.info(
            "Starting MCP server with Streamable HTTP transport at http://%s:%s%s.",
            host,
            port,
            mcp_instance.settings.streamable_http_path,
        )
        mcp_instance.run(transport="streamable-http")


def _run_sse_with_init_guard(mcp_instance: FastMCP, mount_path: str | None) -> None:
    """
    Start the SSE server with the InitGuardMiddleware applied.

    This replaces the plain ``mcp_instance.run(transport="sse")`` call so
    that the InitGuardMiddleware can be inserted between uvicorn and the
    FastMCP Starlette app.  The middleware delays early tool-call requests
    until the per-session MCP initialization handshake is complete,
    preventing the race condition that causes Claude Desktop to receive
    "Received request before initialization was complete" on first connect.

    The function is synchronous (blocking) just like ``FastMCP.run()``.
    """
    import uvicorn  # bundled with mcp[cli]

    async def _serve() -> None:
        starlette_app = mcp_instance.sse_app(mount_path)
        guarded_app = InitGuardMiddleware(starlette_app)
        config = uvicorn.Config(
            guarded_app,
            host=mcp_instance.settings.host,
            port=mcp_instance.settings.port,
            log_level=mcp_instance.settings.log_level.lower(),
        )
        server = uvicorn.Server(config)
        await server.serve()

    anyio.run(_serve)
