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

# encode past experiments
controls  = jnp.stack([fisher.encode_conditions(c) for c in past_conditions])
timestamps = jnp.array([exp['timestamps'] for exp in past_measurements])

# propose 3 new experiments
key = jax.random.PRNGKey(0)
loss_trace, proposed = fisher.propose(key, n=3, controls=controls, timestamps=timestamps, parameters=parameters)

# decode back to physical units
for enc in proposed:
    print(fisher.decode_conditions(enc))
```

## CLI tools

From the repository root, you can run helper scripts directly.

**Fit parameters** from existing data:

```bash
PYTHONPATH=. python scripts/mle.py \
  --model data/models/simple.json \
  --conditions data/experiments/example.json \
  --data data/experiments/measurements.json \
  --iterations 512 \
  --output fitted.json
```

**Propose new informative experiments** using Fisher optimality:

```bash
PYTHONPATH=. python scripts/new_exp.py \
  --model data/models/simple.json \
  --conditions data/experiments/example.json \
  --data data/experiments/measurements.json \
  --parameters fitted.json \
  --condition-ranges condition_ranges.json \
  --n 3 \
  --iterations 64 \
  --criterion D \
  --seed 0 \
  --output proposed.json
```

`condition_ranges.json` must define bounds for `A`, `B`, `E`, and `temperature`, for example:

```json
{
  "A": [0.1, 5.0],
  "B": [0.1, 5.0],
  "E": [0.1, 5.0],
  "temperature": [0.0, 100.0]
}
```

The `proposed.json` output includes `loss_trace`, `encoded_proposals`, and decoded `proposals` in physical units.

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
- parameter names: `q`, `K_A`, `K_B`
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
   - metadata construction
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
- `initial_parameters` (optional): `{q, K_A, K_B}`
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
  },
  "metadata": {
    "model_identifier": "enzyme_model_from_dict_v1",
    "model_version": "1.0.0",
    "solver": {
      "id": "optax.lbfgs",
      "configuration": {"iterations": 64, "rtol": 1e-6, "dtype": "float32"}
    },
    "units_map": {
      "time": "s",
      "temperature": "C",
      "solution_volume": "mL",
      "concentration": "mM"
    },
    "warnings": [],
    "diagnostics": {"num_experiments": 2, "num_timestamps": {"experiment 1": 3, "experiment 2": 3}},
    "deterministic": true,
    "seed": null
  }
}
```

### 2) `propose_doe_experiments`

Proposes new experiment conditions via `doe.doe.Fisher.propose`.

#### Request fields

- `model_spec` (required; must follow `CustomODESystem` schema)
- `condition_ranges` (required): bounds for `A`, `B`, `E`, `temperature`
- parameter bounds come from `model_spec.parameters`
- `parameters` (required): current parameter estimate `{q, K_A, K_B}`
- `history` (required):
  - `conditions`: map `label -> condition`
  - `timestamps`: map `label -> [t1, t2, ...]`
- `proposal_config` (required):
  - `n_proposals` (>= 1)
  - `timestamps` (proposal timestamps)
  - `iterations` (default `64`)
  - `criterion` (`"D"` or `"A"`, default `"D"`)
  - `regularization` (optional float)
  - `seed` (required)

#### Example request

Params for `model_spec` are extracted from `data/models/simple.json` in (https://github.com/mborisyak/doe) (`tests/simple.json` in older revisions).

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
  "condition_ranges": {
    "A": [0.1, 5.0],
    "B": [0.1, 5.0],
    "E": [0.1, 5.0],
    "temperature": [0.0, 100.0]
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
    }
  },
  "proposal_config": {
    "n_proposals": 3,
    "timestamps": [1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 25.0, 29.0],
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
  "encoded_proposals": [
    [0.22, 1.4, -0.5, 0.71],
    [-1.31, 0.88, 0.09, -0.96]
  ],
  "loss_trace": [12.8, 10.1, 8.3],
  "metadata": {
    "model_identifier": "enzyme_model_from_dict_v1",
    "model_version": "1.0.0",
    "solver": {
      "id": "fisher.armijo",
      "configuration": {"criterion": "D", "iterations": 64, "regularization": null}
    },
    "units_map": {
      "time": "s",
      "temperature": "C",
      "solution_volume": "mL",
      "concentration": "mM"
    },
    "warnings": [],
    "diagnostics": {"history_experiments": 2, "proposal_count": 2},
    "deterministic": true,
    "seed": 12345
  }
}
```

### Dynamic model_spec format

Dynamic model specs are passed directly to DOE `CustomODESystem`.
Params for `model_spec` are extracted from `data/models/simple.json` in (https://github.com/mborisyak/doe) (`tests/simple.json` in older revisions).

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
- seed is required for proposal flow and echoed in response metadata.

## Validation Rules

Core validations enforced in `mcp_contracts.py`:

- finite numeric values only (`NaN`/`inf` rejected)
- no unknown fields (strict models, `extra="forbid"`)
- non-empty conditions/measurements/history
- label-set alignment:
  - `conditions` == `measurements`
  - `history.conditions` == `history.timestamps`
- timestamp constraints:
  - min length: 2
  - strictly increasing
- measurement/timestamp length match
- range constraints:
  - exact required keys
  - exactly two values per range
  - `high > low`
- `proposal_config.criterion` must be `A` or `D`
- `proposal_config.n_proposals >= 1`

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
