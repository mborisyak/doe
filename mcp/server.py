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
    SimulateEnzymeDynamicsRequestPayload,
)
from mcp_tools import DoeMcpService, EnzymeMcpService

doe_service = DoeMcpService()
enzyme_service = EnzymeMcpService()

server = FastMCP(
    name="doe-mcp",
    instructions=(
        "MCP server for Fisher-information-based design of experiments and enzyme reaction simulations. "
        "Call fit_parameters to get maximum likelihood estimation of parameters from existing experimental data. "
        "Call propose_doe_experiments to get new informative experiments via Fisher optimization. "
        "Call simulate_enzyme_dynamics with conditions (A, B, E, temperature) and time settings "
        "(t_start, t_end, measurements) to compute trajectories per experiment. "
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
    return doe_service.fit_parameters(request)


@server.tool()
def propose_doe_experiments(
    request: ProposeDoeExperimentsRequestPayload,
) -> Dict[str, Any]:
    """Propose new informative experiments via Fisher optimization."""
    return doe_service.propose_doe_experiments(request)


@server.tool()
def simulate_enzyme_dynamics(
    request: SimulateEnzymeDynamicsRequestPayload,
) -> Dict[str, Any]:
    """Run enzyme dynamics simulation by invoking scripts/experiment.py."""
    return enzyme_service.simulate_enzyme_dynamics(request)


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
