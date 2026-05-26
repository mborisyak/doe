from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: install the `mcp` package to run this server."
    ) from exc

from compute_tools import ComputeResolver
from mcp_tools import DoeMcpService, EnzymeMcpService
from store_tools import StoreMcpService

from doe.store import Store

store = Store(os.environ.get("DOE_STORE_ROOT", "store"))
doe_service = DoeMcpService()
enzyme_service = EnzymeMcpService()
store_service = StoreMcpService(store)
compute = ComputeResolver(store, doe_service, enzyme_service)

server = FastMCP(
    name="doe-mcp",
    instructions=(
        "MCP server for Fisher-information-based design of experiments and enzyme reaction simulations, "
        "backed by a named, append-only store so you refer to entities by name instead of copying numbers. "
        "Entities: a model (structure / parameter ranges), a fitted_model (model + fitted values, e.g. "
        "model-x/v3), a design (intended conditions + expected outcomes), a batch (measured data), and a "
        "tool_result (call log). Pass name='-' to mint a fresh name; every tool returns a `references` block "
        "naming what it produced. "
        "fit_parameters(model, data=[batches]) -> a new fitted_model. "
        "propose_doe_experiments(fitted_model, proposal_config) -> a new design. "
        "simulate_enzyme_dynamics(design | conditions) is the experiment surrogate -> a new data batch (the "
        "only source of data). store_create registers a model/fitted_model/design (never data); store_list "
        "browses; store_get inspects (the bulky 'auxiliary' block is returned as its keys with null values; "
        "pass select='auxiliary.<key>' to fetch one in full). Responses use {ok: true, references, data} on "
        "success or {ok: false, error} on failure."
    ),
)


# --------------------------------------------------------------- compute (by name)
@server.tool()
def fit_parameters(
    model: str,
    data: List[str],
    name: str = "-",
    initial: Optional[str] = None,
    call: str = "-",
    optimizer: Optional[Dict[str, Any]] = None,
    description: str = "",
) -> Dict[str, Any]:
    """Maximum-likelihood fit of `model` (a structural model name) to one or more
    `data` batches (a list of batch names; fit consumes several at once). `initial`
    is an optional fitted_model to warm-start from. Produces a new fitted_model
    (named `name`, or minted as model/vN when '-'); its name is in `references`."""
    return compute.fit(
        model=model, data=data, name=name, initial=initial, call=call,
        optimizer=optimizer, description=description,
    )


@server.tool()
def propose_doe_experiments(
    fitted_model: str,
    proposal_config: Dict[str, Any],
    name: str = "-",
    history: Optional[List[str]] = None,
    call: str = "-",
    description: str = "",
) -> Dict[str, Any]:
    """Propose informative experiments via Fisher optimisation around `fitted_model`
    (supplies both structure and parameter values). `history` is an optional list of
    data batches already collected. `proposal_config` = {n_proposals, criterion(A|D),
    seed, ...}. Produces a new design; its name is in `references`."""
    return compute.propose(
        fitted_model=fitted_model, proposal_config=proposal_config, name=name,
        history=history, call=call, description=description,
    )


@server.tool()
def simulate_enzyme_dynamics(
    name: str = "-",
    design: Optional[str] = None,
    conditions: Optional[Dict[str, Any]] = None,
    call: str = "-",
    time: Optional[Dict[str, Any]] = None,
    solutions: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    description: str = "",
) -> Dict[str, Any]:
    """The experiment surrogate: run conditions and record the measurements as a new
    data batch (the only way data enters the store). Give either `design` (a design
    name, whose conditions are run) or `conditions` (an ad-hoc map label->{A,B,E,
    temperature}). `time`={t_start,t_end,measurements}. The batch name is in
    `references`."""
    return compute.simulate(
        name=name, design=design, conditions=conditions, call=call,
        time=time, solutions=solutions, device=device, description=description,
    )


# --------------------------------------------------------------- storage
@server.tool()
def store_create(
    type: str,
    name: str = "-",
    description: str = "",
    record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a stored entry: model, fitted_model, design, or tool_result.
    NOT data -- batches are produced only by simulate_enzyme_dynamics (the
    experiment surrogate). The store is append-only (no deletes/overwrites).
    `name` may be '-' to mint a free name; the returned `references` reports it."""
    return store_service.create(type, name, record, description=description)


@server.tool()
def store_get(
    ref: str,
    select: Optional[str] = None,
) -> Dict[str, Any]:
    """Inspect a stored record. Returns the record with the bulky `auxiliary` block
    reduced to its top-level keys (each value null) so you can see what is available;
    pass `select` (a dotted path, e.g. 'auxiliary.loss_trace') to fetch one value in
    full."""
    return store_service.get(ref, select=select)


@server.tool()
def store_list(type: Optional[str] = None, brief: bool = False) -> Dict[str, Any]:
    """List stored records (name, type, created_at, description), optionally filtered
    by type: model | fitted_model | design | batch | tool_result. Pass brief=true to
    omit descriptions."""
    return store_service.list_records(type, descriptions=not brief)


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
