from __future__ import annotations

from typing import Any, Dict

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: install the `mcp` package to run this server."
    ) from exc

from mcp_contracts import (
    EstimateDoeParametersRequestPayload,
    ProposeDoeExperimentsRequestPayload,
)
from mcp_tools import DoeMcpService

service = DoeMcpService()

server = FastMCP(
    name="doe-mcp",
    instructions=(
        "MCP server for Fisher-information-based design of experiments. "
        "User can call fit_parameters tool to get maximum likelyhood estimation of parameters "
        "from existin experimental data "
        "Or call propose_doe_experiments to get new experiments which are informative "
        "Requests must be strict JSON objects with finite numeric values. "
        "Responses use a structured envelope: {ok: true, data: ...} on success or "
        "{ok: false, error: {code, message, details}} on failure."
    ),
)


@server.tool()
def fit_parameters(
    request: EstimateDoeParametersRequestPayload,
) -> Dict[str, Any]:
    """Estimate model parameters from historical experiments."""
    return service.fit_parameters(request)


@server.tool()
def propose_doe_experiments(
    request: ProposeDoeExperimentsRequestPayload,
) -> Dict[str, Any]:
    """Propose new informative experiments via Fisher optimization."""
    return service.propose_doe_experiments(request)


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
