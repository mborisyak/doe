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
        "Call simulate_enzyme_dynamics with conditions (A, B, E, temperature) to compute trajectories per experiment; temperature is in Celsius. "
        "In model_spec expressions, use CustomODESystem condition symbols A0, B0, E0, and T; for example, "
        "\"initial_state\": {\"A\": \"A0\", \"B\": \"B0\", \"E\": \"E0\"}. "
        "Requests must be strict JSON objects with finite numeric values. "
        "Responses use a structured envelope: {ok: true, data: ...} on success or "
        "{ok: false, error: {code, message, details}} on failure."
    ),
)


@server.tool()
def fit_parameters(
    request: EstimateDoeParametersRequestPayload,
) -> Dict[str, Any]:
    """Fit ODE model parameters to historical experiment data.

    Use this when you already have measured time-series data and want
    maximum-likelihood estimates for a user-supplied model.

    Request:
    - model_spec: model JSON accepted by CustomODESystem; must include a
      non-empty "parameters" object mapping each parameter name to
      [low, high] bounds. Example initial-state snippet:
      {"A": "A0", "B": "B0", "E": "E0"}.
    - conditions: experiment label -> {A, B, E, temperature}.
    - measurements: same experiment labels -> {timestamps, measurements}.
    - initial_parameters: optional starting values with exactly the same
      parameter names as model_spec.parameters.
    - optimizer: optional {iterations, rtol, dtype}; dtype must be
      "float32" or "float64".

    Constraints:
    - conditions and measurements must contain the same experiment labels.
    - timestamps must be strictly increasing.
    - timestamps and measurements must have equal lengths.
    - all numeric values must be finite; A, B, and E must be non-negative.

    Success data:
    - parameters: fitted parameter values.
    - loss_trace: optimization loss history.
    - predictions: predicted measurement series for each experiment label.
    """
    return doe_service.fit_parameters(request)


@server.tool()
def propose_doe_experiments(
    request: ProposeDoeExperimentsRequestPayload,
) -> Dict[str, Any]:
    """Propose informative follow-up experiments using Fisher optimization.

    Use this when you have a model and want new experimental conditions
    that should maximize information gain about the parameters.

    Request:
    - model_spec: model JSON accepted by CustomODESystem; must include a
      non-empty "parameters" object mapping each parameter name to
      [low, high] bounds. Example initial-state snippet:
      {"A": "A0", "B": "B0", "E": "E0"}.
    - proposal_config: required {n_proposals, iterations, criterion, seed};
      criterion must be "A" or "D". regularization is optional.
    - parameters: optional current parameter estimates keyed exactly like
      model_spec.parameters.
    - history: optional prior experiments with
      {conditions, timestamps, measurements}; all three maps must use the
      same experiment labels and aligned series lengths.

    Constraints:
    - all numeric values must be finite.
    - A, B, and E must be non-negative.
    - history timestamps must be strictly increasing.

    Success data:
    - proposed_conditions: list of suggested {A, B, E, temperature} settings.
    - proposal_timestamps: measurement times to use for the proposals.
    - expected: expected trajectories aligned with proposed_conditions.
    """
    return doe_service.propose_doe_experiments(request)


@server.tool()
def simulate_enzyme_dynamics(
    request: SimulateEnzymeDynamicsRequestPayload,
) -> Dict[str, Any]:
    """Simulate enzyme reaction trajectories for one or more conditions.

    Use this when you want forward simulation only, without parameter
    fitting or experiment design.

    Request:
    - conditions: experiment label -> {A, B, E, temperature}, where A, B,
      and E are non-negative numeric inputs and temperature is in Celsius.

    Constraints:
    - at least one experiment label is required.
    - all numeric values must be finite.

    Success data:
    - experiments: per-label trajectories containing timestamps and A values.
    - metadata: model identifier/version, solver settings, units, warnings,
      diagnostics, deterministic flag, seed, and contract version.
    """
    return enzyme_service.simulate_enzyme_dynamics(request)


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
