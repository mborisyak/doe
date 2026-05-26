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

Three MCP tools (`mcp/server.py`) plus one capability with no tool yet. All are
**pure functions of their request** — each spins a fresh subprocess in a
`TemporaryDirectory` that is destroyed on exit and remembers nothing between
calls.

| tool | core | request | response |
|---|---|---|---|
| `fit_parameters` (MLE) | `scripts/mle.py` | `model_spec`, `conditions{label→{A,B,E,temperature}}`, `measurements{label→{timestamps[],measurements[]}}`, `initial_parameters?`, `optimizer?{iterations,rtol,dtype}` | `parameters{name→float}`, `loss_trace[]`, `predictions{label→[]}` |
| `propose_doe_experiments` (Fisher OED) | `scripts/new_exp.py` | `model_spec`, `parameters?`, `history?{conditions,timestamps,measurements}`, `proposal_config{n_proposals,iterations,criterion(A\|D),regularization?,seed}` | `proposed_conditions[{A,B,E,temperature}]`, `proposal_timestamps[]`, `expected[[…]]` |
| `simulate_enzyme_dynamics` | `scripts/experiment.py` (fixed ground-truth params) | `conditions{label→{A,B,E,temperature}}`, `time?{t_start,t_end,measurements}`, `solutions?{A,B,E}`, `device?`, `units?` | `experiments{label→{time_points[],state_trajectories{A_measured:[]}}}`, `metadata{…}` |
| discriminative DoE *(no MCP tool yet)* | `doe/doe/discriminative.py`, driver `scripts/discriminate_doe.py` | GP spec (ARD-RBF `length_scales,variance` + `noise`), grid, `threshold`, `bounds`, `batch_size`, accumulated `(X,y)`; target field `f = Σ_t|A_full − A_ref|`, reference ∈ {`mm`,`no-inhib`} | proposed batch `B`, `eig` (bits), `sign_probability` on grid, `accuracy` |

All compute responses use the envelope `{ok:true, data}` / `{ok:false, error:{code,message,details}}`.

## Principle: stateless compute, stateful store

Because the tools are pure, **persistence must be external** — a stateless tool
returns `parameters` and forgets them. The store is the missing persistence half,
not an add-on. We keep the roles explicit:

- **experiment surrogate** (`simulate_enzyme_dynamics`) — stands in for running a
  real experiment; it is the **only** source of data (batches), produced with the
  hidden ground-truth parameters the agent does not control.
- **compute tools** (`fit` / `propose` / `discriminate`) — pure analysis/design on
  models + data; the `DoeEngine` core is untouched.
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
| `models` | `model-x` | structural `model_spec` (parameter *ranges*) or gp spec | proposed by agent | frozen / append-only |
| `fitted_models` | `model-x/v7` | model ref + parameter *point values* + provenance + fit predictions | `fit` (or manual / ground-truth) | many per model |
| `designs` | `design-1`, `design-1/experiment-1` | **intended** conditions `x` + `t` + **expected** `y` (per fitted model) | `propose` / `discriminate` / deliberate plan | — |
| `batches` (data) | `batch-1`, `batch-1/experiment-1` | **executed** conditions `x` + `t` + **real** `y`; optional `design` link | `simulate` / lab | self-contained |
| `tool_results` | `fit/model-x/v7` | tool, input refs, raw envelope, ids produced | every call | append-only log |

Key distinctions:

- **model vs fitted model.** A `model_spec` only carries parameter ranges
  `{name:[lo,hi]}` — not runnable. `propose` / `simulate` / predict need point
  values. The runnable artifact is *spec + values* = a **fitted model**, the thing
  other tools reference. One structure has many fits (refits as batches
  accumulate, multi-seed, rival candidates), so they are separate collections with
  compound names `model/vN`.
- **design vs batch.** What is executed need not equal what was designed (rounding,
  grid snap, instrument limits). So conditions live in **both**, independently:
  the design holds intended `x` + expected `y`; the batch holds executed `x` + real
  `y`. They are not redundant — they can hold different numbers. Each record is
  self-contained; the batch keeps an *optional* `design` link for provenance and
  does **not** inherit conditions from it. Comparison = `design.expected` vs
  `batch.measured`, by experiment label.
- **tool_result vs artifact.** A fit *call* is an event (raw envelope, logged in
  `tool_results`); the fitted model is the reusable artifact. The same call writes
  both.

### Record shapes

```jsonc
// models/model-x.json — structural, frozen
{ "name":"model-x", "type":"model", "kind":"ode",
  "spec":{ "states":[…], "parameters":{"k1_on":[1,1000], …},
           "initial_state":{…}, "algebraics":{…}, "rhs":{…}, "observables":{…} } }
// kind:"gp" → spec:{ "kernel":"rbf_ard", "length_scales":[…], "variance":…, "noise":… }
```
```jsonc
// fitted_models/model-x/v7.json — runnable: spec ref + point values
{ "name":"model-x/v7", "type":"fitted_model", "model":"model-x",
  "parameters":{"k1_on":13.7, "kcat":8.9, …},                  // essential
  "fit":{ "data":["batch-1","batch-2"], "tool_result":"fit/model-x/v7", "final_loss":0.27 },  // or "source":"ground_truth"|"manual"
  "auxiliary":{ "loss_trace":[…], "predictions":{"batch-1/experiment-1":[…]} } }
```
```jsonc
// designs/design-1.json — intended conditions + expected y
{ "name":"design-1", "type":"design",
  "space":["A","B","E","temperature"], "observable":"A_measured",
  "experiments":{ "experiment-1":{"x":{"A":1.5,"B":2.0,"E":1.2,"temperature":37}} },   // essential
  "auxiliary":{ "experiment-1":{"t":[…]}, "expected":{"model-x/v7":{"experiment-1":[…]}} },
  "source":{"tool_result":"propose/design-1", "model":"model-x/v7"} }
```
```jsonc
// batches/batch-1.json — executed conditions + real y, self-contained
{ "name":"batch-1", "type":"batch",
  "space":["A","B","E","temperature"], "observable":"A_measured",
  "design":"design-1",                                 // optional provenance, may drift; nullable
  "experiments":{ "experiment-1":{"x":{"A":1.5,"B":2.0,"E":1.2,"temperature":37}} },   // essential
  "auxiliary":{ "experiment-1":{"t":[…], "y":[…]} },
  "source":{"tool_result":"simulate/batch-1"} }        // simulate(truth) / lab
```
```jsonc
// tool_results/fit/model-x/v7.json — the call log
{ "name":"fit/model-x/v7", "type":"tool_result", "tool":"fit_parameters",
  "status":"ok", "created_at":"…",
  "inputs":{"model":"model-x", "data":["batch-1","batch-2"], "config":{…}},
  "references":{"call":"fit/model-x/v7", "fitted_model":"model-x/v7", …},
  "auxiliary":{ "envelope":{ /* raw {ok,data} from the pure core */ } } }
```

## Essential vs auxiliary

The **producing tool** splits its own output into two parts (decided in code):

- **essential** — identity, references, small data and scalars: `parameters`,
  conditions `x`, `fit.final_loss`, `observable`, `source`, …
- **`auxiliary`** — the bulky arrays: loss trajectories, predictions, measured /
  expected `y` series, the raw tool envelope.

This split is a **wire concern only**. The MCP `store_get(ref)` returns the record
with **`auxiliary` reduced to its top-level keys, each value `null`** — so the agent
sees *what* bulky data exists without pulling it. There is no separate summary: the
essentials shown in this view are the summary. Fetch a bulky value explicitly with
`store_get(ref, select="auxiliary.loss_trace")`.

The **Python API returns full records** (arrays included): `store["fitted_model"]
["model-x"]["v7"]["auxiliary"]["loss_trace"]` is just there. `store_list` returns
identity only (`name`, `type`, `created_at`, `description`) — never `auxiliary`.

## Naming: the `-` sentinel

Every tool receives a **`call`** name (the `tool_result`, or `-`) and — when it
produces a record — an **output `name`** (or `-`). `-` means "mint a free name":

| collection | `-` mints |
|---|---|
| `models` | `model-<n>` |
| `fitted_models` | `<input model>/v<n>` — fit of `model-x` → `model-x/v7` |
| `designs` | `design-<n>` |
| `batches` | `batch-<n>` |
| `tool_results` (`call`) | `<tool>/<output-name>` (e.g. `fit/model-x/v7`); else `<tool>/<n>` |

`<n>` = **highest existing in that namespace + 1** (monotonic: deleting `v3` never
lets a later fit reuse `v3`, so provenance refs never silently retarget).

## References output

Every tool always returns a `references` block naming everything it touched, so
the agent can read back `-`-minted names and chain calls:

```jsonc
{ "ok": true,
  "references": {"call":"fit/model-x/v7", "fitted_model":"model-x/v7",
                 "model":"model-x", "data":["batch-1","batch-2"]},
  "data": { /* the created record, auxiliary values nulled */ } }
```

## Tool signatures

**compute** (ref-accepting; the pure core is unchanged behind the resolver)

| tool | inputs | `references` out |
|---|---|---|
| `fit_parameters` | `call`, `name`→fitted_model, `model`, `data:[…]`, `initial?`, `config` | `{call, fitted_model, model, data, initial?}` |
| `propose_doe_experiments` | `call`, `name`→design, `fitted_model`, `history?:[…]`, `config` | `{call, design, fitted_model, history?}` |
| `simulate_enzyme_dynamics` | `call`, `name`→batch, `design` *or* `conditions`, `time?/solutions?/device?` | `{call, batch, design?}` |
| `discriminate` *(planned)* | `call`, `name`→design, `gp` (fitted) *or* `model`+`data`, `full`, `reference`, grid/threshold/bounds/batch_size | `{call, design, gp, full, reference}` |

**storage** (MCP tools over the `doe.store` API; the store is append-only — no
delete, no overwrite — and cannot create data)

| tool | inputs | returns |
|---|---|---|
| `store_create` | `type`, `name`(or `-`), `description`, `record` | `{references:<record refs>}` — `type` ∈ model \| fitted_model \| design \| tool_result (**not** batch) |
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
minting skips it. The MCP `store_create` additionally refuses `batch` data; in
Python, batches are written by the simulate / measurement path the same way.

## Resolver rules

**Name resolution and verification live in MCP, not in the scripts.** The compute
scripts (`mle.py`, `new_exp.py`, `experiment.py`) take arbitrary file paths
(`--model/--data/--output <path>`) and stay store-agnostic. The MCP resolver
turns names into paths, verifies them, materialises the inputs as files, runs the
script, and writes the outputs back into the store.

Per call the resolver:

1. **Reserves the output name first.** `with store.lock_<kind>(name) as rec:` mints
   (or verifies) the name and drops its `<name>.lock` marker before the script
   runs — so a concurrent call can't mint or take it. An existing or already-locked
   name errors out immediately.
2. **Expands input refs** to the script's inline payload / file paths:
   - `model:"model-x"` → spec; `fitted_model:"model-x/v7"` → spec + values → the
     core's `model_spec` + `parameters`.
   - `data:["batch-1","batch-2"]` → each experiment's `x,t,y` → the core's
     `conditions{}` / `measurements{}` maps, labels namespaced
     `batch-1/experiment-1`. This is how a fit consumes **several batches** at once.
   - `design:"design-1"` → intended conditions for `simulate`.
3. **Runs the script** on those paths (no lock held during the run).
4. **Commits**: fills `rec`, which is written on clean exit and the marker cleared;
   a `tool_result` is written too, and all names are returned in `references`.

## File layout

```
store/
  .store.lock                          # our fcntl.flock lock: held only across mint + reserve/commit
  models/model-x.json
  fitted_models/model-x/v7.json        # compound name -> nested path
  fitted_models/model-x/v8.lock        # reservation marker: v8 is being computed
  designs/design-1.json                # experiments are sub-keys inside the file
  batches/batch-1.json
  tool_results/fit/model-x/v7.json
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