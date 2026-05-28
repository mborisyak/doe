"""
Discriminative DoE: find the (alpha, T) boundary where the full enzyme model and
a reference ("simple") model disagree by more than a threshold.

Design space x = (alpha, T):
    A0_vol = B0_vol = alpha/2,  E_vol = 1 - alpha   (stock volumes, total V = 1 mL)
    -> A0 = B0 = (alpha/2) * Ac,  E = (1 - alpha) * Ec   (Ac, Ec from config)
    T in [0, 100] C.

Both models share one parameter set. Products C, D enter the rate ONLY through
the apparent Michaelis constants Kapp, so the simple model is the full kinetics
evaluated with C = D = 0 (the enzyme-no-inhib variant).

Target field (the thing the GP learns):
    f(x) = sum_i | A_full(t_i) - A_simple(t_i) |     over the measurement time grid.
Boundary of interest: { f > threshold }. f -> 0 at both alpha extremes (no
substrate / no enzyme), so the positive region is a closed blob -- a clean
sign-discrimination test.

MULTI-RUN: instead of one fixed parameter set we sample several from the ranges in
config/config.yaml (`parameters: name -> [lo, hi]`), run the whole discriminative-
DoE loop (acquire batch -> measure -> refit GP) on each from an UNTRAINED GP prior,
and summarise how fast the boundary is resolved with a median accuracy curve and a
0.1-0.9 quantile band across the sets.

Batch acquisition: the whole `batch_size`-point batch is optimised continuously
over the expected posterior marginal sign-entropy on the inference grid G,
on-device by optax AdamW (multi-start, jitted). The GP posterior covariance on G
is value-independent (only the mean moves with the unknown outcome), so the proxy
marginalises the outcome by MC over the mean draws. It needs only the per-point
grid variances (the covariance diagonal), so no n_g x n_g covariance or
eigendecomposition is ever formed.
"""
import os
import json
import time
import argparse

import numpy as np
from scipy.integrate import solve_ivp

import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from doe.secret import enzyme, mm, no_inhib
from doe.gp import kernels
from doe.gp.gp import GP, State
from doe.doe.discriminative import DiscriminativeDoE

# normalised (alpha, T) box the batch optimiser searches over.
BOUNDS = [(0.0, 1.0), (0.0, 1.0)]

# ---------------------------------------------------------------- config / params

def load_config(root, name="config.yaml"):
  import yaml
  with open(os.path.join(root, "config", name)) as f:
    return yaml.safe_load(f)


def sample_parameter_sets(ranges, n_sets, rng):
  """Sample `n_sets` parameter sets uniformly from the `name -> [lo, hi]` ranges
  in config/config.yaml. Each set is a dict of (1,)-shaped float32 jax arrays,
  matching what enzyme.kinetics expects."""
  names = list(ranges)
  sets = []
  for _ in range(n_sets):
    p = {}
    for name in names:
      lo, hi = ranges[name]
      v = rng.uniform(float(lo), float(hi))
      p[name] = jnp.atleast_1d(jnp.asarray(v, dtype=jnp.float32))
    sets.append(p)
  return sets


# ---------------------------------------------------------------- enzyme model

# Rate laws to compare the full enzyme model against. Each takes
# (A, B, C, D, E, temp, params); the reactor RHS always supplies C = D = delta.
def _full_rate(A, B, C, D, E, T, p):
  return enzyme.kinetics(A, B, C, D, E, T, p)            # inhibition + denaturation

REFERENCES = {
  # mm: textbook Michaelis-Menten, NO inhibition and NO denaturation.
  "mm": lambda A, B, C, D, E, T, p: mm.kinetics(A, B, C, D, E, T, p),
  # no-inhib: enzyme kinetics with denaturation but no product inhibition;
  # isolates the inhibition contribution.
  "no-inhib": lambda A, B, C, D, E, T, p: no_inhib.kinetics(A, B, C, D, E, T, p),
}


def build_rhs(parameters, rate_fn):
  """RHS for dA/dt of the batch reactor, using `rate_fn` as the rate law."""
  def rhs(_, A, A0, B0, E, temp):
    delta = A0 - A
    B = B0 - delta
    C = D = delta
    return -rate_fn(A, B, C, D, E, temp, parameters)
  rhs = jax.jit(rhs)
  jac = jax.jit(jax.jacobian(rhs, argnums=1))
  return rhs, jac


def make_f_field(config, parameters, reference="mm"):
  """Returns (f(alpha, T) -> sum_i |A_full - A_ref|, n_meas) for one parameter
  set, where A_ref is the trajectory of the `reference` model (see REFERENCES)."""
  dur = config["experiment"]["duration"]
  n_meas = config["experiment"]["measurements"]
  ts_eval = np.linspace(0.0, dur, num=n_meas + 2, dtype=np.float32)[1:-1]
  Ac, Bc, Ec = config["solutions"]["A"], config["solutions"]["B"], config["solutions"]["E"]

  rhs_full, jac_full = build_rhs(parameters, _full_rate)
  rhs_simple, jac_simple = build_rhs(parameters, REFERENCES[reference])

  def trajectory(rhs, jac, A0, B0, E, temp):
    sol = solve_ivp(rhs, t_span=(0.0, dur), t_eval=ts_eval, y0=A0,
                    args=(A0, B0, E, temp), jac=jac, method="LSODA")
    return sol.y[0]

  def f(alpha, temp):
    cv = lambda x: np.atleast_1d(np.asarray(x, dtype=np.float32))
    # alpha/2 + alpha/2 + (1-alpha) = 1, so V = 1 and concentration = volume * stock.
    A0 = cv((alpha / 2.0) * Ac)
    B0 = cv((alpha / 2.0) * Bc)
    E = cv((1.0 - alpha) * Ec)
    temp = cv(temp)
    y_full = trajectory(rhs_full, jac_full, A0, B0, E, temp)
    y_simple = trajectory(rhs_simple, jac_simple, A0, B0, E, temp)
    return float(np.sum(np.abs(y_full - y_simple)))

  return f, n_meas


def empty_state(dim):
  return State(X_flat=jnp.zeros((0, dim)), L=jnp.zeros((0, 0)), alpha=jnp.zeros((0,)))


# ---------------------------------------------------------------- one DoE run

def run_doe_for_set(gp, Xg, ng, thr, f_field, a_lo, a_hi, pts, noise, args, set_idx):
  """Run the full DoE loop on one parameter set from an untrained GP prior.
  Returns a dict of per-step diagnostics (accuracy, sign-prob maps, batches)."""
  f_true = np.array([f_field(a, T) for a, T in pts])
  doe = DiscriminativeDoE(gp, Xg, thr, proxy="marginal", n_outer=args.n_outer,
                          n_multi_start=args.n_multi_start, n_steps=args.n_steps,
                          lr=args.lr)
  rng = np.random.default_rng(args.seed + set_idx)
  key = jax.random.PRNGKey(args.seed + set_idx)

  state = empty_state(2)
  prior_acc = doe.accuracy(state, f_true)
  prior_q = np.asarray(doe.sign_probability(state)).reshape(ng, ng)

  def measure(B_norm):
    raw = np.column_stack([a_lo + B_norm[:, 0] * (a_hi - a_lo), B_norm[:, 1] * 100.0])
    fv = np.array([f_field(a, T) for a, T in raw])
    return raw, fv + np.sqrt(noise) * rng.standard_normal(fv.shape)

  out = {"acc": [], "q": [], "pts": [], "batch": [], "eig": [],
         "prior_acc": prior_acc, "prior_q": prior_q, "f_true": f_true,
         "pos_frac": float(np.mean(f_true > thr))}
  X_data = np.empty((0, 2)); y_data = np.empty((0,)); raw_all = np.empty((0, 2))
  for step in range(args.steps):
    ts = time.time()
    key, ksel = jax.random.split(key)
    B_norm, eig = doe.suggest(state, args.batch_size, BOUNDS, ksel)
    B_norm = np.asarray(B_norm)
    raw, y_new = measure(B_norm)
    X_data = np.vstack([X_data, B_norm]); y_data = np.concatenate([y_data, y_new])
    raw_all = np.vstack([raw_all, raw])
    state = gp.fit(jnp.asarray(X_data, jnp.float32), jnp.asarray(y_data, jnp.float32))

    acc = doe.accuracy(state, f_true)
    out["acc"].append(acc)
    out["q"].append(np.asarray(doe.sign_probability(state)).reshape(ng, ng))
    out["pts"].append(raw_all.copy())                       # cumulative (raw alpha,T)
    out["batch"].append(raw.copy())                         # this step's batch (raw)
    out["eig"].append(eig)
    print(f"    step {step + 1}: acc={acc:.3f}  eig={eig:.3f}  ({time.time() - ts:.1f}s)",
          flush=True)
  return out


# ---------------------------------------------------------------- main

def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--output", default="discriminate_doe.png")
  ap.add_argument("--config", default="config.yaml", help="config file under config/")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--n-sets", type=int, default=8, help="parameter sets sampled from config ranges")
  ap.add_argument("--grid", type=int, default=32, help="grid points per axis (inference grid G)")
  ap.add_argument("--batch-size", type=int, default=8)
  ap.add_argument("--steps", type=int, default=4)
  ap.add_argument("--n-multi-start", type=int, default=8, help="parallel AdamW restarts for batch optimisation")
  ap.add_argument("--n-steps", type=int, default=200, help="AdamW steps per batch optimisation")
  ap.add_argument("--lr", type=float, default=0.05, help="AdamW learning rate")
  ap.add_argument("--n-outer", type=int, default=24, help="MC draws for outcome marginalisation")
  ap.add_argument("--threshold", type=float, default=0.1, help="boundary level on summed f")
  ap.add_argument("--reference", choices=list(REFERENCES), default="mm",
                  help="reference model to discriminate the full model against")
  ap.add_argument("--hypers", default="kernel_hypers.json",
                  help="JSON from fit_kernel_scale.py; falls back to defaults if missing")
  ap.add_argument("--length-scales", type=float, nargs=2, default=[0.22, 0.22],
                  help="RBF length scales (alpha, T) if no --hypers")
  ap.add_argument("--variance", type=float, default=1.0, help="RBF variance if no --hypers")
  args = ap.parse_args()

  root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  config = load_config(root, args.config)
  thr = args.threshold                                      # boundary level on summed f
  n_t = config["experiment"]["measurements"]
  print(f"reference model: {args.reference}   parameter sets: {args.n_sets}")

  # ---- design grid (shared across all parameter sets) ----
  ng = args.grid
  alpha = np.linspace(0.02, 0.98, ng)
  Tdeg = np.linspace(0.0, 100.0, ng)
  AA, TT = np.meshgrid(alpha, Tdeg)                         # (ng, ng): row=T, col=alpha
  pts = np.column_stack([AA.ravel(), TT.ravel()])           # (N, 2) raw (alpha, T)
  a_lo, a_hi = alpha[0], alpha[-1]
  norm_pts = np.column_stack([(pts[:, 0] - a_lo) / (a_hi - a_lo), pts[:, 1] / 100.0])
  Xg = jnp.asarray(norm_pts, dtype=jnp.float32)             # normalised grid for the GP

  # ---- GP (RBF kernel, fixed hypers throughout) ----
  noise = (np.sqrt(n_t) * config["noise"]) ** 2             # ~ propagated measurement noise
  length_scales, variance = list(args.length_scales), args.variance
  if os.path.exists(args.hypers):
    with open(args.hypers) as fh:
      hp = json.load(fh)
    length_scales = hp["length_scales"] if "length_scales" in hp else [hp["length_scale"]] * 2
    variance = hp["variance"]
    noise = hp.get("noise", noise)
    print(f"loaded hypers from {args.hypers}: length_scales(alpha,T)="
          f"[{length_scales[0]:.4f}, {length_scales[1]:.4f}], variance={variance:.4f}, "
          f"noise={noise:.5f}")
  else:
    print(f"--hypers '{args.hypers}' not found; using defaults "
          f"length_scales={length_scales}, variance={variance}")
  kernel = kernels.rbf(jnp.asarray(length_scales, jnp.float32), variance=variance)
  gp = GP(kernel, noise=noise)

  # ---- sample parameter sets and run a DoE loop on each ----
  param_sets = sample_parameter_sets(config["parameters"], args.n_sets,
                                      np.random.default_rng(args.seed))
  results = []
  t_all = time.time()
  for i, params in enumerate(param_sets):
    print(f"\n=== parameter set {i + 1}/{args.n_sets} ===", flush=True)
    t0 = time.time()
    f_field, _ = make_f_field(config, params, reference=args.reference)
    res = run_doe_for_set(gp, Xg, ng, thr, f_field, a_lo, a_hi, pts, noise, args, i)
    print(f"  pos_frac={res['pos_frac']:.2f}  final acc={res['acc'][-1]:.3f}  "
          f"({time.time() - t0:.1f}s)", flush=True)
    results.append(res)
  print(f"\nall sets done in {time.time() - t_all:.1f}s")

  plot(results, args, thr, AA, TT, a_lo, a_hi)


# ---------------------------------------------------------------- plotting

def plot(results, args, thr, AA, TT, a_lo, a_hi):
  """Heatmap matrix (cols = parameter sets, rows = prior + steps) above a
  full-width accuracy convergence plot with a median line and 0.1-0.9 quantile
  band across the sets."""
  nc = len(results)
  ng = AA.shape[0]
  n_heat = args.steps + 1                                   # row 0 = prior, then steps
  N = args.steps
  extent = (a_lo, a_hi, 0.0, 100.0)
  empty = np.empty((0, 2))

  fig = plt.figure(figsize=(3.0 * nc, 2.7 * n_heat + 3.2))
  gs = fig.add_gridspec(n_heat + 1, nc, height_ratios=[1] * n_heat + [1.4])
  heat_axes = []

  def draw_cell(ax, q, Ftrue, earlier, batch, title):
    im = ax.imshow(q, origin="lower", extent=extent, aspect="auto",
                   cmap="RdBu_r", vmin=0.0, vmax=1.0)
    ax.contour(AA, TT, Ftrue, levels=[thr], colors="k", linewidths=1.5)
    ax.scatter(earlier[:, 0], earlier[:, 1], s=9, c="0.6", edgecolors="k", linewidths=0.2)
    ax.scatter(batch[:, 0], batch[:, 1], s=55, marker="*", c="yellow",
               edgecolors="k", linewidths=0.6)
    ax.set_title(title, fontsize=8)
    return im

  # Each row r<N shows the GP fitted on data through step r (the *previous* GP the
  # acquisition saw) with batch r+1 it then selected overlaid -- so the batch is
  # judged against the field that produced it. The final row (r==N) is the GP
  # fitted on all data, no batch: the result.
  for j, res in enumerate(results):
    Ftrue = res["f_true"].reshape(ng, ng)
    for r in range(n_heat):
      ax = fig.add_subplot(gs[r, j])
      heat_axes.append(ax)
      field = res["prior_q"] if r == 0 else res["q"][r - 1]
      acc_r = res["prior_acc"] if r == 0 else res["acc"][r - 1]
      earlier = empty if r == 0 else res["pts"][r - 1]      # data the GP was fit on
      if r < N:                                             # selection context for batch r+1
        im = draw_cell(ax, field, Ftrue, earlier, res["batch"][r],
                       f"acc={acc_r:.3f} · batch {r + 1}")
        row_label = f"GP after {r}\n+ batch {r + 1}"
      else:                                                 # final fit on all data, no batch
        im = draw_cell(ax, field, Ftrue, earlier, empty, f"final  acc={acc_r:.3f}")
        row_label = "final GP\n(all data)"
      if r == 0:
        ax.annotate(f"set {j + 1}  (pos {res['pos_frac']:.2f})", xy=(0.5, 1.28),
                    xycoords="axes fraction", ha="center", fontsize=9, weight="bold")
      if j == 0:
        ax.set_ylabel(f"{row_label}\nT (C)", fontsize=8)
      if r == n_heat - 1:
        ax.set_xlabel("alpha", fontsize=8)

  # ---- full-width convergence plot: median + 0.1-0.9 quantile band across sets ----
  curves = np.array([[res["prior_acc"]] + res["acc"] for res in results])  # (n_sets, steps+1)
  xs = np.arange(0, args.steps + 1)
  med = np.median(curves, axis=0)
  q1, q9 = np.quantile(curves, 0.1, axis=0), np.quantile(curves, 0.9, axis=0)

  ax_sum = fig.add_subplot(gs[n_heat, :])
  for c in curves:
    ax_sum.plot(xs, c, color="0.7", lw=0.8, alpha=0.6, zorder=1)
  ax_sum.fill_between(xs, q1, q9, color="C0", alpha=0.25, zorder=2,
                      label="0.1-0.9 quantile band")
  ax_sum.plot(xs, med, color="C0", marker="o", lw=2.0, zorder=3, label="median")
  ax_sum.set_xlabel("DoE step (0 = untrained prior)")
  ax_sum.set_ylabel("point-wise accuracy\n(GP-uncertainty aware)")
  ax_sum.set_xticks(xs)
  ax_sum.set_title(f"Accuracy vs step across {nc} parameter sets")
  ax_sum.grid(alpha=0.3)
  ax_sum.legend(fontsize=9, loc="lower right")

  # two-line suptitle
  fig.suptitle(
    f"Discriminative DoE (full vs {args.reference}) over {nc} sampled parameter sets\n"
    f"rows = GP P(f > {thr}) before each batch (yellow* = selected); "
    "black = true boundary, gray = used data",
    fontsize=11)

  fig.tight_layout(rect=(0.0, 0.0, 0.93, 0.97))             # leave room for colorbar + suptitle
  cax = fig.add_axes((0.945, 0.40, 0.012, 0.42))
  fig.colorbar(im, cax=cax, label="P(f > thr)")
  fig.savefig(args.output, dpi=130)
  print(f"\nsaved {args.output}")

  print("\nfinal accuracy by set:")
  for j, res in enumerate(results):
    print(f"  set {j + 1:2d}: {res['acc'][-1]:.3f}   "
          f"(steps: {', '.join(f'{a:.3f}' for a in res['acc'])})")
  print(f"median final acc = {med[-1]:.3f}  "
        f"[q0.1={q1[-1]:.3f}, q0.9={q9[-1]:.3f}]")


if __name__ == "__main__":
  main()
