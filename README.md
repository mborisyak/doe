# doe — Design of Experiments for ODE-based models

A JAX library for Fisher-information-based optimal experimental design with ODE kinetic models. Given a parametric ODE model and existing measurements, `doe` estimates model parameters and proposes new experimental conditions that maximally reduce parameter uncertainty.

## Installation

```bash
pip install .
```

**Dependencies:** `jax`, `diffrax`, `optax`, `numpy`

## Usage

**1. Define your kinetic model** by subclassing `ODEModel`:

```python
from collections import namedtuple
import jax.numpy as jnp
import doe

Conditions = doe.common.Conditions          # namedtuple: A, B, E, temperature
Parameters = namedtuple('Parameters', ['q', 'K_A', 'K_B'])

class SimpleEnzyme(doe.common.ODEModel[Parameters]):
    def initial_state(self, conditions):
        Ac, Bc, Ec = doe.common.get_initial_concentrations(conditions)
        return jnp.stack([Ac, Bc, Ec], axis=-1)

    def rhs(self, state, conditions, parameters):
        A, B, E = state[..., 0], state[..., 1], state[..., 2]
        rate = E * parameters.q * A / (parameters.K_A + A) * B / (B + parameters.K_B)
        return jnp.stack([-rate, -rate, jnp.zeros_like(rate)], axis=-1)

    def observe(self, state):
        return state[..., 0]   # observe substrate A
```

**2. Fit parameters** from existing experimental data:

```python
# conditions_data: dict label -> {A, B, E, temperature}
# measurements_data: dict label -> {timestamps, measurements}
parameter_ranges = Parameters(q=[1e2, 2e3], K_A=[1e-2, 2.0], K_B=[1e-2, 2.0])

_, parameters, _ = doe.inference.maximum_likelihood_estimate(
    model, conditions_data, measurements_data, parameter_ranges, iterations=512
)
```

**3. Propose new informative experiments** using Fisher D-optimality:

```python
import jax
import jax.numpy as jnp

condition_ranges = Conditions(A=[0.1, 5.0], B=[0.1, 5.0], E=[0.1, 5.0], temperature=[0.0, 100.0])

fisher = doe.doe.Fisher(
    model, condition_ranges, parameter_ranges,
    timestamps=jnp.linspace(1.0, 29.0, 9),
    iterations=64,
)

# past_conditions: list of Conditions namedtuples
# past_timestamps: list of timestamp arrays
key = jax.random.PRNGKey(0)
loss_trace, proposed = fisher.propose(
    key, n=3,
    controls=past_conditions,    # list of Conditions — encoding handled internally
    timestamps=past_timestamps,  # list of timestamp arrays
    parameters=parameters,
)

# proposed is a Conditions namedtuple with arrays of length n
for i in range(3):
    cond = jax.tree.map(lambda x: x[i], proposed)
    print(cond)
```

## CLI tools

From the repository root, you can run helper scripts directly.

**Fit parameters** from existing data:

```bash
PYTHONPATH=. python scripts/mle.py \
  --model data/secret/simple.json \
  --conditions data/experiments/example.json \
  --data data/experiments/measurements.json \
  --iterations 512 \
  --output fitted.json
```

**Propose new informative experiments** using Fisher optimality:

```bash
PYTHONPATH=. python scripts/new_exp.py \
  --model data/secret/simple.json \
  --n 3 \
  --iterations 64 \
  --criterion D \
  --seed 0 \
  --output proposed.json
```

`--conditions`/`--data` and `--parameters` are all optional:
- if `--parameters` is omitted and history is provided, parameters are fitted from history automatically
- if neither is provided, parameters default to the centre of the ranges from `config/config.yaml`
- condition ranges and proposal timestamps are always read from `config/config.yaml`

With history:

```bash
PYTHONPATH=. python scripts/new_exp.py \
  --model data/secret/simple.json \
  --conditions data/experiments/example.json \
  --data data/experiments/measurements.json \
  --parameters fitted.json \
  --n 3 --iterations 64 --seed 0 \
  --output proposed.json
```

The `proposed.json` output includes `proposals` (decoded conditions), `expected` (predicted trajectories for each proposal), and `proposal_timestamps`.

## API overview

| Symbol | Description |
|---|---|
| `ODEModel` | Abstract base class for ODE kinetic models (`initial_state`, `rhs`, `observe`) |
| `inference.maximum_likelihood_estimate` | Gradient-based MLE (LBFGS) for parameter estimation |
| `doe.Fisher` | Proposes new conditions maximising Fisher information (D- or A-optimality) |
| `doe.ArmijoLineSearch` | Backtracking line search used internally by `Fisher` |

# mcp_doe

`mcp_doe` is a FastMCP server that exposes **Design of Experiments (DoE)** workflows from the local `doe` package over MCP.

It currently supports two tools:

- `fit_parameters`: fit kinetic parameters from historical experiment data.
- `propose_doe_experiments`: propose new conditions using Fisher-information optimization.

The implementation is contract-first, with strict pydantic validation and stable response envelopes.

The DOE library code and DOE reference tests are vendored in this repo under `doe/doe`, `tests/doe`, and `data/`.

## Table of Contents

- [Current Scope](#current-scope)
- [Architecture](#architecture)
- [Directory Layout](#directory-layout)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Running the MCP Server](#running-the-mcp-server)
- [Tool Contracts](#tool-contracts)
- [Dynamic model_spec format](#dynamic-model_spec-format)
- [Response Envelope](#response-envelope)
- [Error Handling](#error-handling)
- [Determinism](#determinism)
- [Validation Rules](#validation-rules)
- [Testing and Quality](#testing-and-quality)
- [Troubleshooting](#troubleshooting)
- [Extending the Server](#extending-the-server)
- [Related Docs](#related-docs)

## Current Scope


- model input: required `model_spec` JSON object (CustomODESystem schema)
- parameter names: dynamic; derived from `model_spec.parameters` keys
- condition fields: `A`, `B`, `E`, `temperature`

The engine validates `model_spec` with DOE `CustomODESystem`, then delegates heavy computation to CLI scripts:

- `scripts/mle.py` for parameter fitting
- `scripts/new_exp.py` for Fisher-based proposal generation

## Architecture

`mcp_doe` follows this file pattern:

1. `server.py`
   - FastMCP server creation
   - tool registration
2. `mcp_tools.py`
   - request validation
   - service-level orchestration
   - mapping exceptions to MCP error envelopes
3. `mcp_contracts.py`
   - request/response schema models
   - validation rules
4. `mcp_engine.py`
   - orchestration + subprocess execution of `scripts/mle.py` and `scripts/new_exp.py`
   - output parsing and response shaping
   - numeric sanity checks
5. `mcp_errors.py`
   - envelope helpers (`success_response`, `error_response`)
   - domain exception type (`ToolExecutionError`)

Execution flow:

`MCP tool call -> pydantic parse/validate -> DoeEngine -> run script -> parse output -> typed response -> MCP envelope`

## Directory Layout

```text
.
├── data/
│   ├── experiments/
│   └── models/
├── doe/
│   ├── common/
│   ├── doe/
│   └── inference/
├── mcp/
│   ├── __main__.py
│   ├── server.py
│   ├── mcp_contracts.py
│   ├── mcp_tools.py
│   ├── mcp_engine.py
│   └── mcp_errors.py
├── mcp/tests/
│   ├── conftest.py
│   ├── test_contracts.py
│   ├── test_tools.py
│   └── test_engine.py
├── mcp/tests/fixtures/
│   ├── conditions_example.json
│   ├── measurements_example.json
│   ├── estimate_request_v1.json
│   ├── propose_request_v1.json
│   └── model_spec_example.json
├── scripts/
│   ├── mle.py
│   └── new_exp.py
├── tests/
│   ├── test_custom_ode.py
│   ├── test_fisher.py
│   └── test_model_base.py
└── README.md
```

## Prerequisites

- Python 3.11+
- [Pixi](https://pixi.sh)
- local checkout of this repository
- Check pixi.toml file and change verison of cuda which is compatible for your system

## Quick Start

```bash
pixi run setup
pixi run test
```

The `setup` task validates that required imports are available in the environment.

## Running the MCP Server

Set `PYTHONPATH=.` so the local `doe` package is importable.

Preferred:

```bash
pixi run run
```

Or just:

```bash
python3 mcp/server.py
```

The server uses stdio transport via FastMCP.

## Tool Contracts

### 1) `fit_parameters`

Fits model parameters from historical data using `doe.inference.maximum_likelihood_estimate`.

#### Request fields

- `model_spec` (required; must follow `CustomODESystem` schema)
- `conditions` (required): map `label -> {A, B, E, temperature}`
- `measurements` (required): map `label -> {timestamps[], measurements[]}`
- parameter bounds come from `model_spec.parameters`
- `initial_parameters` (optional): map with exactly the same keys as `model_spec.parameters`
- `optimizer` (optional):
  - `iterations` (default `512`)
  - `rtol` (default `1e-6`)
  - `dtype` (`"float32"` or `"float64"`, default `"float32"`)

#### Example request

```json
{
  "model_spec": {
    "states": ["A", "B", "E"],
    "parameters": {"q": [100.0, 2000.0], "K_A": [0.01, 2.0], "K_B": [0.01, 2.0]},
    "initial_state": {"A": "A0", "B": "B0", "E": "E0"},
    "algebraics": {"rate": "E * q * A / (K_A + A) * B / (K_B + B)"},
    "rhs": {"A": "-rate", "B": "-rate", "E": "0"},
    "observables": {"A": "A"}
  },
  "conditions": {
    "experiment 1": {"A": 1.0, "B": 2.0, "E": 1.0, "temperature": 10.0},
    "experiment 2": {"A": 1.0, "B": 1.2, "E": 1.0, "temperature": 50.0}
  },
  "measurements": {
    "experiment 1": {
      "timestamps": [3.0, 6.0, 9.0],
      "measurements": [0.5368, 0.4134, 0.3104]
    },
    "experiment 2": {
      "timestamps": [3.0, 6.0, 9.0],
      "measurements": [0.5427, 0.3294, 0.2296]
    }
  },
  "optimizer": {"iterations": 64, "rtol": 1e-6, "dtype": "float32"}
}
```

#### Success data payload

```json
{
  "parameters": {"q": 818.4, "K_A": 0.42, "K_B": 0.67},
  "loss_trace": [0.19, 0.08, 0.04],
  "predictions": {
    "experiment 1": [0.54, 0.40, 0.32],
    "experiment 2": [0.55, 0.34, 0.24]
  }
}
```

### 2) `propose_doe_experiments`

Proposes new experiment conditions via `doe.doe.Fisher.propose`. All inputs except `model_spec` and `proposal_config` are optional — the tool falls back gracefully depending on what is provided.

#### Request fields

- `model_spec` (required; must follow `CustomODESystem` schema)
- parameter bounds come from `model_spec.parameters`
- condition ranges and proposal timestamps come from `config/config.yaml` server-side
- `parameters` (optional): current parameter estimate; if omitted, fitted from history or defaulted to range centres
- `history` (optional):
  - `conditions`: map `label -> {A, B, E, temperature}`
  - `timestamps`: map `label -> [t1, t2, ...]` (strictly increasing, >= 2 values)
  - `measurements`: map `label -> [y1, y2, ...]` (same length as timestamps)
- `proposal_config` (required):
  - `n_proposals` (>= 1)
  - `iterations` (default `64`)
  - `criterion` (`"D"` or `"A"`, default `"D"`)
  - `regularization` (optional float)
  - `seed` (required)

#### Fallback behaviour

| `parameters` | `history` | Parameter source |
|---|---|---|
| provided | provided | use provided parameters |
| omitted | provided | fit from history |
| omitted | omitted | centre of `model_spec.parameters` ranges |
| provided | omitted | use provided parameters, no history prior |

#### Example request

```json
{
  "model_spec": {
    "states": ["A", "B", "E"],
    "parameters": {"q": [100.0, 2000.0], "K_A": [0.01, 2.0], "K_B": [0.01, 2.0]},
    "initial_state": {"A": "A0", "B": "B0", "E": "E0"},
    "algebraics": {"rate": "E * q * A / (K_A + A) * B / (K_B + B)"},
    "rhs": {"A": "-rate", "B": "-rate", "E": "0"},
    "observables": {"A": "A"}
  },
  "parameters": {"q": 818.4, "K_A": 0.42, "K_B": 0.67},
  "history": {
    "conditions": {
      "experiment 1": {"A": 1.0, "B": 2.0, "E": 1.0, "temperature": 10.0},
      "experiment 2": {"A": 1.0, "B": 1.2, "E": 1.0, "temperature": 50.0}
    },
    "timestamps": {
      "experiment 1": [3.0, 6.0, 9.0],
      "experiment 2": [3.0, 6.0, 9.0]
    },
    "measurements": {
      "experiment 1": [0.54, 0.40, 0.32],
      "experiment 2": [0.55, 0.34, 0.24]
    }
  },
  "proposal_config": {
    "n_proposals": 3,
    "iterations": 64,
    "criterion": "D",
    "seed": 12345
  }
}
```

#### Success data payload

```json
{
  "proposed_conditions": [
    {"A": 2.4, "B": 4.7, "E": 0.8, "temperature": 63.2},
    {"A": 0.6, "B": 3.5, "E": 1.7, "temperature": 22.4}
  ],
  "proposal_timestamps": [3.33, 6.67, 10.0, 13.33, 16.67, 20.0, 23.33, 26.67, 30.0],
  "expected": [
    [0.48, 0.37, 0.29, 0.23, 0.19, 0.16, 0.14, 0.12, 0.11],
    [0.51, 0.41, 0.33, 0.27, 0.22, 0.19, 0.16, 0.14, 0.13]
  ]
}
```

### Dynamic model_spec format

Dynamic model specs are passed directly to DOE `CustomODESystem`.
Params for `model_spec` are extracted from `data/models/simple.json` in (https://github.com/mborisyak/doe) (`tests/simple.json` in older revisions).
The keys under `model_spec.parameters` define the required keys for `initial_parameters`, request `parameters`, and response `parameters`.

Example:

```json
{
  "states": ["A", "B", "E"],
  "parameters": {
    "q": [100.0, 2000.0],
    "K_A": [0.01, 2.0],
    "K_B": [0.01, 2.0]
  },
  "initial_state": {
    "A": "A0",
    "B": "B0",
    "E": "E0"
  },
  "algebraics": {
    "rate": "E * q * A / (K_A + A) * B / (K_B + B)"
  },
  "rhs": {
    "A": "-rate",
    "B": "-rate",
    "E": "0"
  },
  "observables": {
    "A": "A"
  }
}
```



## Response Envelope

All tools return a top-level envelope.

Success:

```json
{
  "ok": true,
  "data": {}
}
```

Error:

```json
{
  "ok": false,
  "error": {
    "code": "validation_error",
    "message": "Request payload failed schema validation.",
    "details": {}
  }
}
```

## Error Handling

### Common error codes

- `validation_error`
  - schema or value-level validation failure (pydantic)
- `execution_failed`
  - known DoE execution failure wrapped by engine
- `numeric_instability`
  - non-finite values or out-of-range decoded proposals
- `internal_error`
  - unexpected service-layer failure (sanitized details only)

### Exception mapping behavior

- `ValidationError` -> `validation_error`
- `ToolExecutionError` -> mapped code/message/details as-is
- unhandled exceptions -> `internal_error` with `{"type": "<ExceptionName>"}` details

## Determinism

- `fit_parameters` is deterministic for fixed inputs.
- `propose_doe_experiments` is deterministic for fixed request + `proposal_config.seed`.
- seed is required for proposal flow.

## Validation Rules

Core validations enforced in `mcp_contracts.py`:

- finite numeric values only (`NaN`/`inf` rejected)
- no unknown fields (strict models, `extra="forbid"`)
- non-empty conditions/measurements where required
- label-set alignment:
  - `fit_parameters`: `conditions` == `measurements`
  - `propose_doe_experiments` history: `conditions` == `timestamps` == `measurements`
- timestamp constraints (history and measurements):
  - min length: 2
  - strictly increasing
- measurement/timestamp length match per label
- range constraints:
  - `model_spec.parameters` must be a non-empty map with `[low, high]` bounds per key
  - `high > low` for every range
  - parameter maps (`initial_parameters`, request `parameters`) must exactly match `model_spec.parameters` keys when provided
- `proposal_config.criterion` must be `"A"` or `"D"`
- `proposal_config.n_proposals >= 1`
- `proposal_config.seed` is required

## Testing and Quality

Run project tests:

```bash
pixi run test
pixi run lint
```

Run only MCP or DOE tests:

```bash
pixi run test-mcp
pixi run test-doe
```

Focused MCP tests:

```bash
pixi run python -m pytest mcp/tests -q
```

Test suite covers:

- request contract validation and defaults
- response envelope shape stability
- structured error mapping
- engine-level finite outputs and range checks
- deterministic seeded proposal behavior

## Troubleshooting

### `ModuleNotFoundError: No module named 'jax'`

Use the Pixi environment:

```bash
pixi run test
```

### `ModuleNotFoundError: No module named 'sympy'`

`CustomODESystem` requires `sympy`. Ensure dependencies are installed in the Pixi environment:

```bash
pixi run setup
```

### `doe` import-path differences

`mcp_engine.py` includes fallback imports to support both package layouts:

- `doe.common` / `doe.inference` / `doe.doe`
- `doe.doe.common` / `doe.doe.inference` / `doe.doe.doe`

This avoids environment-specific import failures during test collection.

### GPU warning from JAX

A warning may appear when a GPU exists but CUDA-enabled `jaxlib` is not installed. The server runs on CPU by default and this warning is non-fatal.

## Extending the Server

When adding new models or tools:

1. Add/extend request and response contracts in `mcp_contracts.py`.
2. Keep transport logic in `server.py` minimal.
3. Implement orchestration logic in `mcp_engine.py` and computation logic in scripts where appropriate.
4. Map new domain errors with `ToolExecutionError`.
5. Add tests for:
   - schema validation failures
   - shape stability
   - deterministic behavior
   - error mapping
6. Keep examples in this README and fixtures in `mcp/tests/fixtures` synchronized with schema changes.

## License

MIT
