"""The MCP tool surface: mom as an MCP *server*, a sibling of the routers in ``mom.api``.

Six tools — five read-only views plus ``consult``, which runs a panel. ``consult`` folds the same
typed ``StreamEvent`` stream the wire encoders fold (``api/encoders/``); there is no second
orchestration path here, only a fourth reader of the one that exists.

Not to be confused with ``ChatRequestIR.mcp_tools``, which points the other way: those are MCP
tools a *client* manages, relayed to an upstream synthesizer that speaks the Responses API. This
package is about mom being the server on the other end of someone else's tool call.

Two transports, one server definition (``build_mcp_server``): streamable HTTP mounted at ``/mcp``
on the gateway (``mount.py``), and stdio for local clients (``stdio.py``, behind ``mom mcp``).
"""

from mom.api.mcp.mount import McpGate, mount_mcp, serve_mcp
from mom.api.mcp.server import build_mcp_server


__all__ = ["McpGate", "build_mcp_server", "mount_mcp", "serve_mcp"]
