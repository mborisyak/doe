"""
Discriminative DoE: find the (alpha, T) boundary where the full enzyme model and
the no-inhibition ("simple") model disagree by more than a threshold.

Design space x = (alpha, T):
    A0_vol = B0_vol = alpha/2,  E_vol = 1 - alpha   (stock volumes, total V = 1 mL)
    -> A0 = B0 = (alpha/2) * Ac,  E = (1 - alpha) * Ec   (Ac, Ec from config)
    T in [0, 100] C.

Both models share one parameter file. Products C, D enter the rate ONLY through
the apparent Michaelis constants Kapp, so the simple model is the full kinetics
evaluated with C = D = 0 (exactly the enzyme-no-inhib.json variant).

Target field (the thing the GP learns):
    f(x) = sum_i | A_full(t_i) - A_simple(t_i) |     over the measurement time grid.
Boundary of interest: { f > threshold }, threshold = 0.1 per time-point.
f -> 0 at both alpha extremes (no substrate / no enzyme), so the positive region
is a closed blob -- a clean sign-discrimination test.

We run the discriminative-DoE loop (acquire batch -> measure -> refit GP) for each
of the 5 sign-pattern entropy heuristics, starting from an UNTRAINED GP prior
(stress test), and compare how fast each resolves the boundary.

Batch acquisition: greedy selection of `batch_size` points from a candidate
sub-grid that minimise the expected posterior value of the heuristic over the
inference grid G (32x32). The GP posterior covariance on G is value-independent
(only the mean moves with the unknown outcome), so the entropy heuristics
marginalise the outcome by MC over the mean draws, while the rank heuristics
(driven by the covariance spectrum) are evaluated at the expected posterior mean.
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

LN2 = np.log(2.0)

# ---------------------------------------------------------------- enzyme model

def load_config(root):
  import yaml
  with open(os.path.join(root, "config", "enzyme.yaml")) as f:
    return yaml.safe_load(f)


def load_parameters(root):
  with open(os.path.join(root, "data", "models", "enzyme.parameters.json")) as f:
    raw = json.load(f)["parameters"]
  return {k: jnp.atleast_1d(jnp.asarray(v, dtype=jnp.float32)) for k, v in raw.items()}


# Rate laws to compare the full enzyme model against. Each takes
# (A, B, C, D, E, temp, params); the reactor RHS always supplies C = D = delta.
def _full_rate(A, B, C, D, E, T, p):
  return enzyme.kinetics(A, B, C, D, E, T, p)            # inhibition + denaturation

REFERENCES = {
  # mm: textbook Michaelis-Menten, NO inhibition and NO denaturation (the
  # validity-region reference, doe/secret/mm.py).
  "mm": lambda A, B, C, D, E, T, p: mm.kinetics(A, B, C, D, E, T, p),
  # no-inhib: enzyme kinetics with denaturation but no product inhibition
  # (doe/secret/no_inhib.py); isolates the inhibition contribution.
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
  """Returns f(alpha, T) -> sum_i |A_full - A_ref| over the measurement grid,
  where A_ref is the trajectory of the `reference` model (see REFERENCES)."""
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


# ---------------------------------------------------------------- acquisition
# The discriminative-DoE engine lives in the package: doe.doe.discriminative
# (DiscriminativeDoE / optimize_batch / PROXIES). Here we only map the plot's
# display names to the package proxy keys. (marginal - total-correlation was
# dropped: its -1/2 logdet R term is anti-informative -- globally minimised by
# collapsing the batch to the box corners.)
HEURISTICS = {
  "marginal Sum H_b": "marginal",
  "marginal eigen": "marginal_eigen",
  "soft-rank M_g": "soft_rank_Mg",
  "modulated soft-rank": "modulated",
  "effective rank": "effective_rank",
}


def empty_state(dim):
  return State(X_flat=jnp.zeros((0, dim)), L=jnp.zeros((0, 0)), alpha=jnp.zeros((0,)))


# ---------------------------------------------------------------- main

def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--output", default="discriminate_doe.png")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--grid", type=int, default=32, help="grid points per axis (inference grid G)")
  ap.add_argument("--batch-size", type=int, default=8)
  ap.add_argument("--steps", type=int, default=4)
  ap.add_argument("--n-restarts", type=int, default=4, help="L-BFGS-B restarts for batch optimisation")
  ap.add_argument("--n-outer", type=int, default=24, help="MC draws for outcome marginalisation")
  ap.add_argument("--threshold", type=float, default=0.1, help="boundary level on summed f")
  ap.add_argument("--reference", choices=list(REFERENCES), default="mm",
                  help="reference model to discriminate the full model against")
  ap.add_argument("--hypers", default="kernel_hypers.json",
                  help="JSON from fit_kernel_scale.py; falls back to defaults if missing")
  ap.add_argument("--length-scales", type=float, nargs=2, default=[0.22, 0.22],
                  help="ARD RBF length scales (alpha, T) if no --hypers")
  ap.add_argument("--variance", type=float, default=1.0, help="RBF variance if no --hypers")
  args = ap.parse_args()

  root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  config = load_config(root)
  parameters = load_parameters(root)
  f_field, n_t = make_f_field(config, parameters, reference=args.reference)
  thr = args.threshold                                     # boundary level on summed f
  print(f"reference model: {args.reference}")

  # ---- design grid ----
  ng = args.grid
  alpha = np.linspace(0.02, 0.98, ng)
  Tdeg = np.linspace(0.0, 100.0, ng)
  AA, TT = np.meshgrid(alpha, Tdeg)                        # (ng, ng): row=T, col=alpha
  pts = np.column_stack([AA.ravel(), TT.ravel()])          # (N, 2) raw (alpha, T)

  # normalise to [0,1]^2 for the GP
  a_lo, a_hi = alpha[0], alpha[-1]
  norm_pts = np.column_stack([(pts[:, 0] - a_lo) / (a_hi - a_lo), pts[:, 1] / 100.0])
  Xg = jnp.asarray(norm_pts, dtype=jnp.float32)

  # ---- ground-truth field (secret: only for threshold check + scoring) ----
  print(f"evaluating true f on {pts.shape[0]} grid points ...", flush=True)
  t0 = time.time()
  f_true = np.array([f_field(a, T) for a, T in pts])
  print(f"  done in {time.time() - t0:.1f}s; f in [{f_true.min():.3f}, {f_true.max():.3f}], "
        f"thr={thr}, positive fraction={np.mean(f_true > thr):.2f}", flush=True)

  # ---- GP (ARD RBF kernel, fixed hypers throughout) ----
  noise = (np.sqrt(n_t) * config["noise"]) ** 2           # ~ propagated measurement noise
  length_scales, variance = list(args.length_scales), args.variance
  if os.path.exists(args.hypers):
    with open(args.hypers) as fh:
      hp = json.load(fh)
    length_scales = hp["length_scales"] if "length_scales" in hp else [hp["length_scale"]] * 2
    variance = hp["variance"]
    noise = hp.get("noise", noise)
    print(f"loaded ARD hypers from {args.hypers}: length_scales(alpha,T)="
          f"[{length_scales[0]:.4f}, {length_scales[1]:.4f}], variance={variance:.4f}, "
          f"noise={noise:.5f}")
  else:
    print(f"--hypers '{args.hypers}' not found; using defaults "
          f"length_scales={length_scales}, variance={variance}")
  kernel = kernels.rbf_ard(jnp.asarray(length_scales, jnp.float32), variance=variance)
  gp = GP(kernel, noise=noise)

  def measure(B_norm, rng):
    """Run the reactor at the continuous normalised batch; return raw (alpha,T)
    coordinates and the noisy summed-discrepancy observations f."""
    raw = np.column_stack([a_lo + B_norm[:, 0] * (a_hi - a_lo), B_norm[:, 1] * 100.0])
    f = np.array([f_field(a, T) for a, T in raw])
    return raw, f + np.sqrt(noise) * rng.standard_normal(f.shape)

  # ---- untrained-GP prior (shared cold start, before any acquisition) ----
  bounds = [(0.0, 1.0), (0.0, 1.0)]                          # normalised (alpha, T) box
  prior_doe = DiscriminativeDoE(gp, Xg, thr)
  prior_acc = prior_doe.accuracy(empty_state(2), f_true)
  prior_q = np.asarray(prior_doe.sign_probability(empty_state(2))).reshape(ng, ng)
  print(f"prior (untrained GP): acc={prior_acc:.3f}")

  # ---- run DoE per heuristic ----
  results = {name: {"acc": [], "q": [], "pts": [], "batch": [], "eig": []} for name in HEURISTICS}
  for name in HEURISTICS:
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    doe = DiscriminativeDoE(gp, Xg, thr, proxy=HEURISTICS[name],
                            n_outer=args.n_outer, n_restarts=args.n_restarts)
    state = empty_state(2)
    X_data = np.empty((0, 2)); y_data = np.empty((0,)); raw_all = np.empty((0, 2))
    print(f"\n=== acquisition: {name} ===", flush=True)
    for step in range(args.steps):
      ts = time.time()
      key, ksel = jax.random.split(key)
      B_norm, eig = doe.suggest(state, args.batch_size, bounds, ksel)
      B_norm = np.asarray(B_norm)
      raw, y_new = measure(B_norm, rng)
      X_data = np.vstack([X_data, B_norm]); y_data = np.concatenate([y_data, y_new])
      raw_all = np.vstack([raw_all, raw])
      state = gp.fit(jnp.asarray(X_data, jnp.float32), jnp.asarray(y_data, jnp.float32))

      acc = doe.accuracy(state, f_true)
      q = np.asarray(doe.sign_probability(state))
      results[name]["acc"].append(acc)
      results[name]["q"].append(q.reshape(ng, ng))
      results[name]["pts"].append(raw_all.copy())           # cumulative (raw alpha,T)
      results[name]["batch"].append(raw.copy())             # this step's batch (raw)
      results[name]["eig"].append(eig)
      print(f"  step {step + 1}: acc={acc:.3f}  eig={eig:.3f}  ({time.time() - ts:.1f}s)", flush=True)

  # ---- plot: acquisitions (cols) x [prior + steps] (rows), + accuracy summary ----
  names = list(HEURISTICS)
  nc = len(names)
  extent = (a_lo, a_hi, 0.0, 100.0)
  Ftrue = f_true.reshape(ng, ng)
  empty = np.empty((0, 2))
  n_heat = args.steps + 1                                  # row 0 = prior, then steps
  fig = plt.figure(figsize=(3.1 * nc, 3.0 * n_heat + 3.0))
  gs = fig.add_gridspec(n_heat + 1, nc, height_ratios=[1] * n_heat + [1.1], hspace=0.42)
  heat_axes = []

  def draw_cell(ax, q, earlier, batch, title):
    im = ax.imshow(q, origin="lower", extent=extent, aspect="auto",
                   cmap="RdBu_r", vmin=0.0, vmax=1.0)
    ax.contour(AA, TT, Ftrue, levels=[thr], colors="k", linewidths=1.5)
    ax.scatter(earlier[:, 0], earlier[:, 1], s=9, c="0.6", edgecolors="k", linewidths=0.2)
    ax.scatter(batch[:, 0], batch[:, 1], s=55, marker="*", c="yellow",
               edgecolors="k", linewidths=0.6)
    ax.set_title(title, fontsize=9)
    return im

  # Each row r<N shows the GP fitted on data through step r (the *previous* GP
  # that the acquisition saw) with batch r+1 it then selected overlaid -- so the
  # batch is judged against the field that produced it. The final row (r==N) is
  # the GP fitted on all data, no batch: the result.
  N = args.steps
  for j, name in enumerate(names):
    q_list, b_list, p_list, a_list = (results[name]["q"], results[name]["batch"],
                                      results[name]["pts"], results[name]["acc"])
    for r in range(n_heat):
      ax = fig.add_subplot(gs[r, j])
      heat_axes.append(ax)
      field = prior_q if r == 0 else q_list[r - 1]          # GP through step r
      acc_r = prior_acc if r == 0 else a_list[r - 1]
      earlier = empty if r == 0 else p_list[r - 1]          # data the GP was fit on
      if r < N:                                             # selection context for batch r+1
        im = draw_cell(ax, field, earlier, b_list[r], f"acc={acc_r:.3f}  ·  batch {r + 1}")
        row_label = f"GP after {r}\n+ batch {r + 1}"
      else:                                                 # final fit on all data, no batch
        im = draw_cell(ax, field, earlier, empty, f"final  acc={acc_r:.3f}")
        row_label = f"final GP\n(all data)"
      if r == 0:
        ax.annotate(name, xy=(0.5, 1.25), xycoords="axes fraction", ha="center",
                    fontsize=10, weight="bold")
      if j == 0:
        ax.set_ylabel(f"{row_label}\nT (C)", fontsize=9)
      if r == n_heat - 1:
        ax.set_xlabel("alpha", fontsize=9)

  # full-width accuracy-vs-step summary (step 0 = shared untrained prior)
  ax_sum = fig.add_subplot(gs[n_heat, :])
  xs = np.arange(0, args.steps + 1)
  for name in names:
    ax_sum.plot(xs, [prior_acc] + results[name]["acc"], marker="o", lw=1.8, label=name)
  ax_sum.set_xlabel("DoE step (0 = untrained prior)")
  ax_sum.set_ylabel("point-wise accuracy\n(GP-uncertainty aware)")
  ax_sum.set_xticks(xs)
  ax_sum.set_title("Accuracy vs step, all acquisitions")
  ax_sum.grid(alpha=0.3)
  ax_sum.legend(fontsize=8, ncol=min(nc, 3), loc="lower right")

  fig.suptitle(f"Discriminative DoE (full vs {args.reference}): rows = GP P(f > {thr}) BEFORE each "
               "batch with the batch it selected (yellow*) overlaid; last row = final GP on all data. "
               "black = true boundary, gray = data already used",
               y=0.995, fontsize=11)
  fig.colorbar(im, ax=heat_axes, shrink=0.6, label="P(f > thr)")
  fig.savefig(args.output, dpi=130, bbox_inches="tight")
  print(f"\nsaved {args.output}")

  print("\nfinal accuracy:")
  for name in names:
    print(f"  {name:22s} {results[name]['acc'][-1]:.3f}   "
          f"(steps: {', '.join(f'{a:.3f}' for a in results[name]['acc'])})")


if __name__ == "__main__":
  main()
