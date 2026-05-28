"""
Fit the RBF kernel scale for the discriminative-DoE GP, BEFORE running the DoE.

Workflow: run a small space-filling experiment over the (alpha, T) design space,
measure the model-discrepancy field f (with measurement noise), then choose the
RBF length scale (and signal variance) by maximising the GP log marginal
likelihood. The fitted hyper-parameters are written to JSON for discriminate_doe.py.

Inputs are normalised to [0, 1]^2 (alpha -> [0,1], T -> T/100), so the length
scale is reported in that normalised box.

Note: GP.fit_kernel minimises the log marginal likelihood (sign bug); we maximise
it directly here via negative-LML + L-BFGS-B over log-parameters.
"""
import os
import json
import argparse

import numpy as np
import jax
import jax.numpy as jnp
import scipy.optimize as spopt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from doe.gp import kernels
from doe.gp.gp import GP
from discriminate_doe import load_config, load_parameters, make_f_field, REFERENCES


def space_filling(n_per_axis):
  """Coarse grid over (alpha, T); also returns the [0,1]^2-normalised inputs."""
  alpha = np.linspace(0.05, 0.95, n_per_axis)
  Tdeg = np.linspace(0.0, 100.0, n_per_axis)
  AA, TT = np.meshgrid(alpha, Tdeg)
  pts = np.column_stack([AA.ravel(), TT.ravel()])
  norm = np.column_stack([(pts[:, 0] - alpha[0]) / (alpha[-1] - alpha[0]), pts[:, 1] / 100.0])
  return pts, norm


def fit_hypers(Xn, y, noise, ls_bounds=(0.02, 3.0), var_bounds=(1e-3, 1e3)):
  """Maximise GP log marginal likelihood over per-dimension length scales
  and the signal variance. Returns (length_scales (D,), variance, LML)."""
  X = jnp.asarray(Xn, jnp.float32)
  yj = jnp.asarray(y, jnp.float32)
  dim = Xn.shape[1]

  def neg_lml(theta):                                   # theta = [log_ls_0..D-1, log_var]
    ls, var = jnp.exp(theta[:dim]), jnp.exp(theta[dim])
    gp = GP(kernels.rbf(ls, variance=var), noise=noise)
    return -gp.log_marginal_likelihood(X, yj)

  val_grad = jax.jit(jax.value_and_grad(neg_lml))
  def fun(t):
    v, g = val_grad(jnp.asarray(t))
    return float(v), np.asarray(g, np.float64)

  bounds = [(np.log(ls_bounds[0]), np.log(ls_bounds[1]))] * dim + \
           [(np.log(var_bounds[0]), np.log(var_bounds[1]))]
  best = None
  for ls0 in (0.1, 0.25, 0.5, 1.0):                     # shared multi-start over both dims
    t0 = np.array([np.log(ls0)] * dim + [np.log(np.var(y) + 1e-6)])
    res = spopt.minimize(fun, t0, jac=True, method="L-BFGS-B", bounds=bounds)
    if best is None or res.fun < best.fun:
      best = res
  ls = np.exp(best.x[:dim]).astype(float)
  return ls, float(np.exp(best.x[dim])), -float(best.fun)


def lml_curve_dim(Xn, y, noise, ls_opt, var, d, ls_grid):
  """LML as one length scale (dim d) varies, the others held at the optimum."""
  X, yj = jnp.asarray(Xn, jnp.float32), jnp.asarray(y, jnp.float32)
  out = []
  for v in ls_grid:
    ls = ls_opt.copy(); ls[d] = float(v)
    gp = GP(kernels.rbf(ls, variance=var), noise=noise)
    out.append(float(gp.log_marginal_likelihood(X, yj)))
  return np.array(out)


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--output", default="kernel_hypers.json")
  ap.add_argument("--plot", default="kernel_scale_fit.png")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--n-per-axis", type=int, default=8, help="space-filling design = n x n")
  ap.add_argument("--reference", choices=list(REFERENCES), default="mm",
                  help="reference model the full model is compared against")
  args = ap.parse_args()

  root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  config = load_config(root)
  parameters = load_parameters(root)
  f_field, n_t = make_f_field(config, parameters, reference=args.reference)
  noise = (np.sqrt(n_t) * config["noise"]) ** 2           # propagated measurement noise

  pts, Xn = space_filling(args.n_per_axis)
  print(f"space-filling design: {pts.shape[0]} points; evaluating f ...", flush=True)
  f = np.array([f_field(a, T) for a, T in pts])
  rng = np.random.default_rng(args.seed)
  y = f + np.sqrt(noise) * rng.standard_normal(f.shape)

  ls, var, lml = fit_hypers(Xn, y, noise)
  dim_names = ["alpha", "T"]
  print("fitted length scales (normalised box): " +
        ", ".join(f"{dim_names[d]}={ls[d]:.4f}" for d in range(len(ls))) +
        f"  variance={var:.4f}  (noise={noise:.5f}, LML={lml:.2f})")

  with open(args.output, "w") as fh:
    json.dump({"length_scales": [float(v) for v in ls], "variance": var, "noise": noise,
               "n_design": int(pts.shape[0]), "seed": args.seed,
               "reference": args.reference}, fh, indent=2)
  print(f"wrote {args.output}")

  ls_grid = np.geomspace(0.03, 2.0, 60)
  fig, ax = plt.subplots(figsize=(6, 4))
  for d in range(len(ls)):
    ax.plot(ls_grid, lml_curve_dim(Xn, y, noise, ls, var, d, ls_grid),
            lw=2, label=f"{dim_names[d]} (opt={ls[d]:.3f})")
    ax.axvline(ls[d], color=f"C{d}", ls="--", alpha=0.6)
  ax.set_xscale("log")
  ax.set_xlabel("length scale of one dim (others at optimum)")
  ax.set_ylabel("log marginal likelihood")
  ax.set_title(f"RBF kernel scale selection (full vs {args.reference})")
  ax.legend()
  fig.tight_layout()
  fig.savefig(args.plot, dpi=130)
  print(f"wrote {args.plot}")


if __name__ == "__main__":
  main()
