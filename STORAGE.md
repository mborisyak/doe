# Named storage for the DoE tools

## Why

The AI scientist drives stateless tools (MLE fit, Fisher DoE, enzyme simulation,
discriminative DoE). Today it copies numbers into every call and writes outputs to
ad-hoc files (see `data/run_2026_03_18_120000/` and
`data/session_20260318_112412/`): a model spec is re-typed into each tool call, a
simulation output is written, re-read, and copied into the next fit request.
Sub-agents pass files around by path.

This store lets the agent **refer to everything by name**. It registers a model
once, calls a tool with `model: "model-x"`, and gets back
`fitted_model: "model-x/v7"` to feed the next call. No numbers are copied; the
server resolves names to payloads and persists every result.

## Tools being wrapped

The MCP server (`mcp/server.py`) exposes ODE/experiment compute, GP-surrogate
compute, storage, and config tools (full human catalogue in `MCP.md`). The compute
tools are **pure functions of their request** — each spins a fresh subprocess in a
`TemporaryDirectory` that is destroyed on exit and remembers nothing between calls.

Each script now returns the **full record body** it owns (`parameters`/`experiments` +
`fit` + `auxiliary`); the resolver dumps that body and only stamps linkage refs.

| tool | core script | request | response (record body) |
|---|---|---|---|
| `fit_ode` (MLE) | `scripts/mle.py` (`--model --batches`) | `model_spec`, `batches{name→record}`, `initial_parameters?`, `optimizer?{iterations,rtol,dtype}` | `parameters{name→float}`, `fit{final_loss,iterations}`, `auxiliary{loss_trace[], predictions{batch→{exp→{timestamps,<obs>}}}, …}` |
| `doe_ode` (Fisher OED) | `scripts/new_exp.py` (`--model --history`) | `model_spec`, `parameters?`, `history?{name→record}`, `proposal_config{n_proposals,iterations,criterion(A\|D),regularization?,seed}` | `experiments{label→{conditions}}`, `auxiliary{expected{label→{timestamps,<obs>}}}` |
| `simulate_enzyme_dynamics` | `scripts/experiment.py` (fixed ground-truth) | `conditions{label→{A,B,E,temperature}}` (window/concentrations/noise are config-owned) | `experiments{label→{time_points[],state_trajectories{A_measured:[]}}}`, `metadata` |
| `fit_gp` | `scripts/gp_fit.py` | GP `model_spec`, `batches{name→record}`, `folds?` | `parameters{cv_rmse}`, `fit{cv_rmse,n_train,observable}`, `auxiliary{gp_state, predictions{<obs>:{mean,std}}}` |
| `hyper_fit_gp` | `scripts/gp_hyperfit.py` | base GP `model_spec`, `batches{name→record}` | `spec` (optimised length_scales+variance) |
| `predict_gp` | `scripts/gp_predict.py` | GP `model_spec`, `state`, `points[[…]]` | `{<obs>:{mean,std}}` |
| `doe_gp` | `scripts/gp_doe.py` | GP `model_spec`, `state`, `bounds[[lo,hi]]`, `batch_size`, `seed` | `experiments{label→{conditions}}`, `auxiliary{expected{…mean+sigma}, eig}` |
| `discriminatory_doe_gp` | `scripts/gp_discriminate.py` (`doe/doe/discriminative.py`) | GP `model_spec`, `state`, `grid[[…]]`, `threshold`, `bounds`, `batch_size`, `seed` | `experiments{label→{conditions}}`, `auxiliary{expected{…mean+sigma}, eig}` |

`create_gp` (register a GP model from hyper-parameters) and `get_config` (return
`config/<DOE_CONFIG>.yaml`) do no compute — no subprocess. All compute responses use
the envelope `{ok:true, data}` / `{ok:false, error:{code,message,details}}`.

## Principle: stateless compute, stateful store

Because the tools are pure, **persistence must be external** — a stateless tool
returns `parameters` and forgets them. The store is the missing persistence half,
not an add-on. We keep the roles explicit:

- **experiment surrogate** (`simulate_enzyme_dynamics`) — stands in for running a
  real experiment; it is the **only** source of data (batches), produced with the
  hidden ground-truth parameters the agent does not control.
- **compute tools** (ODE `fit_ode` / `doe_ode`, and the GP `fit_gp` / `doe_gp` /
  `discriminatory_doe_gp` …) — pure analysis/design on models + data, run as
  subprocess scripts; the cores are untouched.
- **storage** (`store_*` + the `doe.store` Python API) — inherently stateful; it
  *is* the store.
- **resolver** — a server-side layer that bridges them: resolve name refs → inline
  payload → call the pure core → catch outputs → write records → return names.

The agent passes names; the **server resolves**. This gives sub-agents one shared
namespace on disk instead of passing files by path.

## Collections

Five record types. Reference grammar: a `/` in a name is a path separator on disk;
experiment names are sub-keys *inside* a record, not separate files.

| collection | ref example | holds | produced by | lifecycle |
|---|---|---|---|---|
| `model` | `model-x` | structural `model_spec` (parameter *ranges*) or gp spec | proposed by agent | frozen / append-only |
| `fitted_model` | `model-x/v7` | model ref + parameter *point values* + provenance + fit predictions | `fit_ode` / `fit_gp` (or manual / ground-truth) | many per model |
| `design` | `design-1`, `design-1/exp-1` | **intended** `conditions` (no timestamps); *optional* **expected** model output (per fitted model, in `auxiliary`) | `doe_ode` / GP DoE / deliberate plan | — |
| `data` | `batch-1`, `batch-1/exp-1` | **executed** `conditions` + **real** `measurements` (observable-keyed + `timestamps`); optional `design` link — records are named `batch-N` | `simulate` / lab | self-contained |
| `logs` | `fit_ode/model-x-7` | tool, arguments, inputs, outputs (ids produced); one sub-dir per tool name | every call | append-only log |

Key distinctions:

- **model vs fitted model.** A `model_spec` only carries parameter ranges
  `{name:[lo,hi]}` — not runnable. `doe_ode` / `simulate` / predict need point
  values. The runnable artifact is *spec + values* = a **fitted model**, the thing
  other tools reference. One structure has many fits (refits as batches
  accumulate, multi-seed, rival candidates), so they are separate collections with
  compound names `model/vN`.
- **design vs batch.** What is executed need not equal what was designed (rounding,
  grid snap, instrument limits). So conditions live in **both**, independently:
  the design holds intended `x` (+ *optional* expected `y`); the batch holds executed
  `x` + real `y`. They are not redundant — they can hold different numbers. Each
  record is self-contained; the batch keeps an *optional* `design` link for provenance
  and does **not** inherit conditions from it. Comparison = `design.expected` vs
  `batch.measured`, by experiment label.
- **`expected` is optional, and is *not* the goal of a design.** A design only needs
  its intended `conditions`; a hand-written design can omit `expected` entirely. When
  a tool does attach it, `expected` is just the model's prediction at the chosen
  points — and DoE deliberately chooses the points where the model is *least* sure, so
  the expected values there are the least informative by construction (that uncertainty
  is exactly why those points are worth measuring). Treat `expected` as a diagnostic,
  not the design's purpose.
- **log vs artifact.** A fit *call* is an event (logged in `logs` —
  tool, arguments, inputs, outputs); the fitted model is the reusable artifact. The same
  call writes both.

### Record shapes

Each collection is a singular directory under the store root (`model/`, `fitted_model/`,
`design/`, `data/`, `logs/`). `Store._write` stamps `type`/`name`/`created_at`/
`description` onto every record; the producer writes the rest.

```jsonc
// model/model-x.json — structural, frozen
{ "name":"model-x", "type":"model", "kind":"ode",
  "spec":{ "states":[…], "parameters":{"k1_on":[1,1000], …},
           "initial_state":{…}, "algebraics":{…}, "rhs":{…}, "observables":{…} } }
// kind:"gp" → spec:{ "kernel":"rbf", "length_scales":[…], "variance":…, "noise":… }
```
```jsonc
// fitted_model/model-x/v7.json — runnable: spec ref + point values
{ "name":"model-x/v7", "type":"fitted_model", "model":"model-x",
  "parameters":{"k1_on":13.7, "kcat":8.9, …},                  // essential
  "fit":{ "data":["batch-1","batch-2"], "tool_result":"fit_ode/model-x-7",
          "final_loss":0.27, "iterations":60 },                // or "source":"ground_truth"|"manual"
  "auxiliary":{ "loss_trace":[…], "predictions":{"batch-1":{"exp-1":{"timestamps":[…],"A":[…]}}} } }
// an ODE fit's predictions carry no sigma; a GP fit's auxiliary holds gp_state + predictions{<obs>:{mean,std}}.
```
```jsonc
// design/design-1.json — intended conditions only (no timestamps); expected = model output
{ "name":"design-1", "type":"design",
  "experiments":{ "exp-1":{"conditions":{"A":1.5,"B":2.0,"E":1.2,"temperature":37}} },   // essential
  "auxiliary":{ "expected":{"model-x/v7":{"exp-1":{"timestamps":[…],"A":[…]}}} },  // GP DoE adds sigma_<obs> + top-level eig
  "source":{"tool_result":"doe_ode/model-x/v7-1", "model":"model-x/v7"} }
```
```jsonc
// data/batch-1.json — executed conditions + real measurements, self-contained
{ "name":"batch-1", "type":"data",
  "design":"design-1",                                 // optional provenance, may drift; nullable
  "experiments":{ "exp-1":{ "conditions":{"A":1.5,"B":2.0,"E":1.2,"temperature":37},
                            "measurements":{"timestamps":[…], "A":[…]} } },   // essential; observable-keyed, no sigma
  "source":{"tool_result":"simulate_enzyme_dynamics/design-1-1"} }   // simulate(truth) / lab; "…/conditions-N" for ad-hoc
```
```jsonc
// logs/fit_ode/model-x-7.json — the call log (thin); call name = <tool>/<target>-<N>
{ "name":"fit_ode/model-x-7", "type":"logs", "tool":"fit_ode", "status":"ok",
  "arguments":{"optimizer":{"iterations":60}},          // non-ref params
  "inputs":{"model":"model-x", "data":["batch-1","batch-2"]},   // consumed refs
  "outputs":{"fitted_model":"model-x/v7"} }             // produced refs
```

## Essential vs auxiliary

The **producing tool** splits its own output into two parts (decided in code):

- **essential** — identity, references, small data and scalars: `parameters`,
  `conditions`, `fit.final_loss`, `source`, … plus a batch's `measurements`
  (observable-keyed, with `timestamps`) since a batch is self-contained data.
- **`auxiliary`** — the bulky model-output arrays on fitted_model / design records: loss
  trajectories, `predictions` and `expected` (observable-keyed, `sigma_<obs>`
  predictive std), a fitted GP's serialised `gp_state`. (`logs` records are thin —
  no auxiliary.)

This split is a **wire concern only**. The MCP `store_get(ref)` returns the record
with **`auxiliary` reduced to its top-level keys, each value `null`** — so the agent
sees *what* bulky data exists without pulling it. There is no separate summary: the
essentials shown in this view are the summary. Fetch a bulky value explicitly with
`store_get(ref, select="auxiliary.loss_trace")`.

The **Python API returns full records** (arrays included): `store["fitted_model"]
["model-x"]["v7"]["auxiliary"]["loss_trace"]` is just there. `store_list` returns
identity only (`name`, `type`, `created_at`, `description`) — never `auxiliary`.

## Naming: the `-` sentinel

Every tool receives a **`call`** name (the `logs` entry, or `-`) and — when it
produces a record — an **output `name`** (or `-`). `-` means "mint a free name":

| collection | `-` mints |
|---|---|
| `model` | `model-<n>` |
| `fitted_model` | `<input model>/v<n>` — fit of `model-x` → `model-x/v7` |
| `design` | `design-<n>` |
| `data` | `batch-<n>` |
| `logs` (`call`) | `<tool>/<target>-<n>` — keyed by the input acted on (e.g. `fit_ode/model-x-7`); a free-form `store_create` mints `result-<n>` |

`<n>` = **highest existing in that namespace + 1** (monotonic: deleting `v3` never
lets a later fit reuse `v3`, so provenance refs never silently retarget).

## References output

Every tool always returns a `references` block naming everything it touched, so
the agent can read back `-`-minted names and chain calls:

```jsonc
{ "ok": true,
  "references": {"call":"fit_ode/model-x-7", "fitted_model":"model-x/v7",
                 "model":"model-x", "data":["batch-1","batch-2"]} }
```

The compute tools return **refs only** — no record dump. Read the numbers back with
`store_get` (only `predict_gp` / `store_get` / `get_config` carry a `data` block).

## Tool signatures

**compute** (ref-accepting; the pure core is unchanged behind the resolver)

| tool | inputs | `references` out |
|---|---|---|
| `fit_ode` | `call`, `name`→fitted_model, `model`, `data:[…]`, `initial?`, `optimizer?` | `{call, fitted_model, model, data, initial?}` |
| `doe_ode` | `call`, `name`→design, `fitted_model`, `history?:[…]`, `proposal_config` | `{call, design, fitted_model, history?}` |
| `simulate_enzyme_dynamics` | `call`, `name`→data (batch-N), `design` *or* `conditions` (window/concentrations/noise are config-owned) | `{call, data, design?}` |
| `create_gp` | `name`→model, `kernel`, `length_scale(s)`, `variance`, `noise`, `observables` | `{model}` |
| `hyper_fit_gp` | `name`→model, `data:[…]`, `kernel`, `noise`, `observables` | `{model, data}` |
| `fit_gp` | `call`, `name`→fitted_model, `model`, `data:[…]`, `folds?` | `{call, fitted_model, model, data}` |
| `predict_gp` | `fitted_model`, `points:[[…]]` | `{fitted_model}` (data = `{<obs>:{mean,std}}`, no record) |
| `doe_gp` | `call`, `name`→design, `fitted_model`, `bounds`, `batch_size`, `seed?` | `{call, design, fitted_model}` |
| `discriminatory_doe_gp` | `call`, `name`→design, `fitted_model`, `grid`, `threshold`, `bounds`, `batch_size`, `seed?` | `{call, design, fitted_model}` |

`get_config()` (no refs) returns the experiment config. GP DoE designs store the
proposed `(conditions,t)` points as experiments and the GP's expected mean/std (+ `eig`)
in `auxiliary`.

**storage** (MCP tools over the `doe.store` API; the store is append-only — no
delete, no overwrite — and cannot create data)

| tool | inputs | returns |
|---|---|---|
| `store_create` | `type`, `name`(or `-`), `description`, `record` | `{references:<record refs>}` — `type` ∈ model \| fitted_model \| design \| logs (**not** data) |
| `store_get` | `ref`, `select?` | `{references:<record refs>, data:<record with auxiliary values nulled, or the selected path>}` |
| `store_list` | `type?`, `brief?=false` | `{data:[{name, type, created_at, description}]}` — `brief` omits descriptions |

The **pure Python API** (`doe.store`) backs these — a filesystem-backed dict, full
records on read:

```python
from doe.store import Store
store = Store("store")

store["model"]["model-x"]                            # full record dict
store["fitted_model"]["model-x"].keys()              # ["v1", "v2", ...]  (the fits)
store["fitted_model"]["model-x"]["v7"]["auxiliary"]["loss_trace"]   # full array

# create (append-only): name it directly, or mint with lock("-")
store["model"]["model-z"] = {"kind": "ode", "spec": {...}}
with store["model"].lock("-") as name:               # reserve a minted name; body is lock-free
    params = expensive_fit(...)                        # name held by a <name>.lock marker
    store["model"][name] = {"kind": "ode", "spec": {...}}   # write; marker cleared on exit
```

`__setitem__` validates required keys and is append-only (existing name → error). A
`<name>.lock` marker (from `lock`) reserves a name during a long body so concurrent
minting skips it. The MCP `store_create` additionally refuses `data`; in
Python, batches are written by the simulate / measurement path the same way.

## Resolver rules

**Name resolution and writing live in MCP; the scripts stay store-agnostic.** The
compute scripts (`mle.py`, `new_exp.py`, `experiment.py`, `gp_*.py`) take plain file
paths (`--model`, `--batches`/`--history`, `--output <path>`) and read/write JSON
files only. The resolver resolves names → records, materialises them as files in a
temp dir, runs the script as a subprocess, then **stamps + commits** the result back
into the store. (`create_gp`/`get_config` do no compute and skip the subprocess.)

Per call the resolver:

1. **Reserves the output name** under the store lock: `with group.lock(name) as rec:`
   mints (or verifies) the name and drops its `<name>.lock` marker before the script
   runs — a concurrent call can't take it. An existing or already-locked name errors.
2. **Resolves input refs to records and writes them to the temp dir:**
   - `model:"model-x"` → its `spec`; `fitted_model:"model-x/v7"` → spec + `parameters`
     (or the GP `gp_state`).
   - `data:["batch-1","batch-2"]` → each batch **record** written to `<name>.json`; the
     script merges them via `doe.dataset` (each (experiment, timestamp) → one row,
     labelled `<batch>/<exp>`). This is how a fit consumes **several batches** at once.
   - `design:"design-1"` → its intended conditions for `simulate`.
3. **Runs the script** on those paths (no lock held during the run, only the marker).
4. **Commits**: builds the record from the script's `result.json`, writes it (via
   `Store._write`, which stamps `type/name/created_at`) and clears the marker; a thin
   `logs` entry is written too, and all names are returned in `references`.

## File layout

```
store/
  .store.lock                                   # our fcntl.flock lock: held only across mint + reserve/commit
  model/model-x.json
  fitted_model/model-x/v7.json                  # compound name -> nested path
  fitted_model/model-x/v8.lock                  # reservation marker: v8 is being computed
  design/design-1.json                          # experiments are sub-keys inside the file
  data/batch-1.json                             # data records are named batch-N
  logs/fit_ode/model-x-7.json                   # call name <tool>/<target>-<N> -> nested path
```

A `<name>.lock` marker (holding pid + timestamp) reserves a name while a tool runs;
it is removed when the record is written (or on error). Minting treats a marker
exactly like a committed record.

Root is a constructor argument, default `store/`. A per-run root (`store-1/`,
`store-2/`, …) is a one-line change later.

## No index — the filesystem is the source of truth

There is no `index.json` to keep in sync. Resolving a bare ref stats the five
collection dirs; `list` globs them and reads each record (cheap); minting scans a
namespace's existing names for the max integer suffix. Records are never deleted
or overwritten — the store is append-only.

## Concurrency

Sub-agents (separate processes) and server request threads run in parallel against
one shared store. Two levels of locking, both built on **our own ~10-line
`fcntl.flock` lock** (`_flock`) — no dependency, no `threading` primitives. A fresh
fd per acquisition makes flock serialise *threads too* (it treats distinct open
file descriptions independently, even within one process).

- **Global lock** (`store/.store.lock`) — held *briefly* around any filesystem read
  that mints/reserves a name *and* around every record write: resolve/mint →
  collision-check (`name_exists` / `name_locked`) → write (one-shot `put`) or drop
  the `<name>.lock` marker (`reserve`). A `reserve` re-takes it to **commit** (write
  record + remove marker) so that transition is atomic against a concurrent mint
  scan. Nothing mints, reserves, or writes a name without holding it.
- **Reservation markers** (`<name>.lock`) — hold a name across a long-running tool
  body *without* keeping the global lock. Minting skips marked names. The marker is
  dropped before the body runs and removed at commit (or on error, freeing the
  name). Both the marker drop and the commit happen under the global lock, so a
  minter never sees a name as momentarily belonging to neither a marker nor a record.
- **Atomic writes.** Every record is a temp file + `os.replace`, so readers never
  see a half-written file.
- **Reads** (`get` / `list` / `read`) take no lock.

Validated by `tests/mcp/test_store.py`: many threads minting `-` names, racing one
explicit name (single winner), nested `model-x/vN` versions — plus a cross-process
multiprocessing check — all yield unique names with no collisions.