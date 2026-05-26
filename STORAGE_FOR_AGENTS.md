# Using the store (for the AI scientist)

**Refer to things by name. Never copy numbers between tools.** Register a model
once, get back a name, pass that name to the next tool. The server resolves names
to data for you. (Design/internals live in [STORAGE.md](STORAGE.md).)

## What's in the store

| you say | it is |
|---|---|
| `model-x` | a **model** — structure only (`model_spec` with parameter *ranges*, or a GP spec). Not runnable on its own. |
| `model-x/v3` | a **fitted_model** — a model + fitted parameter *values*. This is what you hand to other tools. |
| `design-7` | a **design** — conditions you intend to probe + each model's *expected* outcome. |
| `batch-2` | **data** — a measured batch: conditions actually run + *real* outcomes. |
| `fit/model-x/v3`, `result-5` | a **tool_result** — a call log, or any free-form note you write. |

Experiments inside a record are sub-names: `batch-2/experiment-1`.

## The three storage tools

- **`store_list(type?, brief?)`** — browse. Returns `name, type, created_at,
  description` per record. Filter by `type`
  (`model|fitted_model|design|batch|tool_result`). `brief=true` drops descriptions.
  *Start here to see what already exists before computing anything.*
- **`store_get(ref, select?)`** — inspect one record. You get the record with its
  bulky `auxiliary` block reduced to **just its keys, values `null`** — so you see
  *what* arrays exist (e.g. `loss_trace`, `predictions`) plus all the small essential
  fields (parameters, `fit.final_loss`, conditions…). To pull one array in full, pass
  `select="auxiliary.loss_trace"`.
- **`store_create(type, name, description, record)`** — register a `model`,
  `fitted_model`, `design`, or `tool_result`. **You cannot create data** — batches
  come only from `simulate_enzyme_dynamics` (the experiment surrogate). Always give a
  short `description`; it shows up in `store_list`.
  - **`tool_result` is your free-form space.** Write *whatever you want* into the
    `record` — any analysis, comparison, or note — under **any `tool` name you
    choose** (the only required field is `tool`, and its value is arbitrary). Use
    `name="-"` to mint `result-N`, or give your own. Example:
    `store_create("tool_result", "-", "AIC comparison",
    {"tool": "model_comparison", "winner": "model-x/v3", "aic": {...}})`.

## Naming: use `-` to get a free name

Pass `name="-"` and the store mints one; the returned `references` block tells you
what it chose. Patterns: `model-<n>`, `model-x/v<n>` (a fit of `model-x`),
`design-<n>`, `batch-<n>`. Every tool returns a `references` block — read the
minted names from there to chain the next call.

## Two rules to remember

- **Append-only.** Nothing is ever deleted or overwritten. Re-fitting a model makes
  `model-x/v4`, not a new `model-x/v3`. Writing an existing name is an error.
- **Keys first, arrays on demand.** `store_get` shows the essentials but lists
  `auxiliary` as keys with `null` values (`loss_trace`, `predictions`, measured /
  expected series…). Pull one with `select="auxiliary.<key>"` only when you need the
  numbers — don't drag whole arrays around.

## Computing on the data yourself (Python)

When you need to crunch the bulky arrays (residuals, custom metrics), use the pure
Python API instead of pulling megabytes through a tool:

It looks like a dict (backed by files); reads return the **full** record, arrays
included — no nulling here.

```python
from doe.store import Store
store = Store("store")                              # same root the server uses

store["model"]["model-x"]                           # -> full record dict
store["fitted_model"]["model-x"].keys()             # -> ["v1", "v2", ...]  (the fits)
rec = store["fitted_model"]["model-x"]["v3"]        # full record, arrays included
import numpy as np; np.diff(rec["auxiliary"]["loss_trace"])   # compute on it

# Create (append-only). Name it, or mint one with lock("-"):
store["model"]["model-z"] = {"kind": "ode", "spec": {...}, "description": "two-step"}
with store["model"].lock("-") as name:              # reserve a minted name, lock held for the body
    store["model"][name] = {"kind": "ode", "spec": {...}}   # marker cleared on exit
```

## A typical loop

1. `store_list("model")` — what models exist? `store_create` a new `model` if needed.
2. `propose_doe_experiments(fitted_model=..., name="-")` → a `design` (or pick
   conditions yourself / random).
3. `simulate_enzyme_dynamics(design=..., name="-")` → a `batch` of real data.
4. `fit_parameters(model="model-x", data=["batch-1","batch-2"], name="-")` →
   `model-x/v4` (you can fit several batches at once — just list them).
5. `store_get("model-x/v4")` — check `fit.final_loss`; compare a design's `expected`
   (`select="auxiliary.expected"`) against the batch's measured values. Good enough?
   stop. Otherwise propose more or a new model and repeat.

Everything above is referenced by name end to end — no numbers are ever copied by
hand.

> **Status.** `fit_parameters`, `propose_doe_experiments`, `simulate_enzyme_dynamics`,
> the storage tools (`store_*`), and the `doe.store` Python API are all live and take
> names as shown. Not yet available: a `discriminate` tool for discriminative DoE
> (the GP-surrogate boundary search) — for that, use `scripts/discriminate_doe.py`
> directly for now.
