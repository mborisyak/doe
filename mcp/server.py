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
from config_tools import ConfigMcpService
from mcp_tools import DoeMcpService, EnzymeMcpService
from simulate_tools import SimulateResolver
from store_tools import StoreMcpService

from doe.store import Store

store = Store(os.environ.get("DOE_STORE_ROOT", "store"))
doe_service = DoeMcpService()
enzyme_service = EnzymeMcpService()
store_service = StoreMcpService(store)
config_service = ConfigMcpService()
compute = ComputeResolver(store, doe_service)          # analysis / design (fit, propose, GP)
simulator = SimulateResolver(store, enzyme_service)    # the experiment surrogate (data source)

server = FastMCP(
    name="doe-mcp",
    instructions=(
        "MCP server for Fisher-information-based design of experiments and enzyme reaction simulations, "
        "backed by a named, append-only store so you refer to entities by name instead of copying numbers. "
        "Entities: a model (structure / parameter ranges), a fitted_model (model + fitted values, e.g. "
        "model-x/v3), a design (intended conditions; optional per-model expected outcomes in auxiliary), "
        "data (measured batches, named batch-N), and a "
        "log (call log; the logs collection). Pass name='-' to mint a fresh name; every tool returns a `references` block "
        "naming what it produced. "
        "fit_ode(model, data=[batches]) -> a new fitted_model. "
        "doe_ode(fitted_model, proposal_config) -> a new design. "
        "simulate_enzyme_dynamics(design | conditions) is the experiment surrogate -> a new data batch (the "
        "only source of data). store_create registers a model/fitted_model/design (never data); store_list "
        "browses; store_get inspects (the bulky 'auxiliary' block is returned as its keys with null values; "
        "pass select='auxiliary.<key>' to fetch one in full). get_config() returns the experiment config "
        "(condition bounds, sampling duration, concentrations, noise) -- read it before proposing/simulating. "
        "Tools return {ok: true, references} -- references name what was produced; inspect the numbers with "
        "store_get (only predict_gp/store_get/get_config return a data block). On failure: {ok: false, error}."
    ),
)


# --------------------------------------------------------------- compute (by name)
@server.tool()
def fit_ode(
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
def doe_ode(
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
    description: str = "",
) -> Dict[str, Any]:
    """The experiment surrogate: run conditions and record the measurements as a new
    data batch (the only way data enters the store). Give either `design` (a design
    name, whose conditions are run) or `conditions` (an ad-hoc map label->{A,B,E,
    temperature}). The sampling window, concentrations, and noise are config-owned (see
    get_config); they are not parameters here. The batch name is in `references`."""
    return simulator.simulate(
        name=name, design=design, conditions=conditions, call=call, description=description,
    )


# --------------------------------------------------------------- GP surrogate
@server.tool()
def create_gp(
    name: str = "-",
    kernel: str = "rbf",
    length_scales: Optional[List[float]] = None,
    length_scale: Optional[float] = None,
    variance: float = 1.0,
    noise: float = 1.0e-6,
    observables: Optional[List[str]] = None,
    description: str = "",
) -> Dict[str, Any]:
    """Register a Gaussian-process model (kind 'gp') with the given kernel hyper-parameters.
    The GP surrogates one observable as a function of (conditions, time); `length_scales`
    is per-dimension (A, B, E, temperature, t), or `length_scale` for an isotropic scale.
    `observables` must be a single name for now. Produces a model; name is in `references`."""
    return compute.create_gp(
        name=name, kernel=kernel, length_scales=length_scales, length_scale=length_scale,
        variance=variance, noise=noise, observables=observables, description=description,
    )


@server.tool()
def hyper_fit_gp(
    data: List[str],
    name: str = "-",
    kernel: str = "rbf",
    noise: float = 1.0e-6,
    observables: Optional[List[str]] = None,
    description: str = "",
) -> Dict[str, Any]:
    """Register a GP model whose kernel hyper-parameters (per-dimension length-scales +
    variance) are fit to `data` batches by maximising the log-marginal-likelihood.
    Produces a model; its name is in `references`."""
    return compute.hyper_fit_gp(
        data=data, name=name, kernel=kernel, noise=noise, observables=observables, description=description,
    )


@server.tool()
def fit_gp(
    model: str,
    data: List[str],
    name: str = "-",
    call: str = "-",
    folds: int = 5,
    description: str = "",
) -> Dict[str, Any]:
    """Fit a GP `model` to `data` batches: k-fold cross-validation (reports `cv_rmse`),
    then refit on all data; the serialised GP state + predictions are stored on a new
    fitted_model (its name is in `references`)."""
    return compute.fit_gp(model=model, data=data, name=name, call=call, folds=folds, description=description)


@server.tool()
def predict_gp(fitted_model: str, points: List[List[float]]) -> Dict[str, Any]:
    """Evaluate a fitted GP at explicit input points (each `[A, B, E, temperature, t]`).
    Returns `{<observable>: {mean, std}}`. A plain query -- writes nothing."""
    return compute.predict_gp(fitted_model=fitted_model, points=points)


@server.tool()
def doe_gp(
    fitted_model: str,
    bounds: Dict[str, List[float]],
    batch_size: int,
    name: str = "-",
    call: str = "-",
    seed: int = 0,
    description: str = "",
) -> Dict[str, Any]:
    """Batch-BALD DoE for a fitted GP: choose `batch_size` points that maximise the
    information gained about f, `I(y_B; f) = 1/2 logdet(I + noise^-1 Sigma_BB)`, within
    `bounds` ({var: [lo, hi]} for A, B, E, temperature, t). Produces a design (its expected
    GP mean/std per proposed point in `auxiliary`); name in `references`."""
    return compute.doe_gp(
        fitted_model=fitted_model, bounds=bounds, batch_size=batch_size,
        name=name, call=call, seed=seed, description=description,
    )


@server.tool()
def discriminatory_doe_gp(
    fitted_model: str,
    grid: List[List[float]],
    threshold: float,
    bounds: Dict[str, List[float]],
    batch_size: int,
    name: str = "-",
    call: str = "-",
    seed: int = 0,
    description: str = "",
) -> Dict[str, Any]:
    """Discriminative DoE for a fitted GP: choose `batch_size` points that best resolve the
    sign pattern `[f(x) > threshold]` over `grid` (list of `[A, B, E, temperature, t]`
    points), optimised within `bounds`. Produces a design; name in `references`."""
    return compute.discriminatory_doe_gp(
        fitted_model=fitted_model, grid=grid, threshold=threshold, bounds=bounds,
        batch_size=batch_size, name=name, call=call, seed=seed, description=description,
    )


# --------------------------------------------------------------- storage
@server.tool()
def store_create(
    type: str,
    name: str = "-",
    description: str = "",
    record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a stored entry: model, fitted_model, design, or logs.
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
    by type: model | fitted_model | design | data | logs. Pass brief=true to
    omit descriptions."""
    return store_service.list_records(type, descriptions=not brief)


@server.tool()
def get_config() -> Dict[str, Any]:
    """Return the agent-facing experiment config: solution concentrations, condition
    bounds, sampling duration / measurement count, noise, etc. One config per deployment,
    selected by the DOE_CONFIG env var (default 'enzyme'). Read this to learn the valid
    ranges and timing before proposing or simulating experiments."""
    return config_service.get()


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
