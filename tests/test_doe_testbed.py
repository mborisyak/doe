"""
Validation testbed for the discriminative-DoE entropy heuristics.

Grid G is 3x3 (n_g = 9), so the sign-pattern vector y' = [f(x_i) > 0] lives in a
fully enumerable space of 2^9 = 512 patterns. That lets us get *exact* ground
truth: sample functions from the GP, threshold, tabulate the 512-cell frequency
table, regularise with a small Dirichlet prior (avoid p=0), read off the entropy.

For each measurement batch B we estimate the expected information gain about y' —
the mutual information I(y'; y_B) — by marginalising over the values y_B the batch
could return (the GP posterior covariance on G is value-independent; only the mean
moves). Each heuristic gets the same treatment, scored by Spearman against the
exact IG.

We sweep the kernel length-scale so the *prior* sign-pattern rank ranges from
near-independent (effrank ~ 7) through the middle down to strongly-shared
(effrank ~ 1.5) — the regime where ignoring inter-bit dependence should start to
cost the marginal heuristic, and a dependency term could earn its place.

Heuristics (entropy of y' from posterior mean mu_g, cov Sigma_g):
  - marginal sign-entropy        Sum_i H_b(p_i),  p_i = Phi(mu_i/sigma_i)
  - marginal minus total-corr    Sum_i H_b(p_i) + 1/2 logdet R   (R = corr(Sigma_g))
  - effective rank of Sigma_g    soft rank, covariance-only (mean-blind)
"""
import os
import numpy as np
from numpy.linalg import slogdet, cholesky, solve
from scipy.special import psi          # digamma, for the Bayesian entropy estimate
from scipy.stats import norm, spearmanr

import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from doe.gp import kernels

LN2 = np.log(2.0)
POW2 = (2 ** np.arange(9)).astype(np.int64)   # pattern -> integer id in [0, 512)

LENGTH_SCALES = [0.4, 0.6, 0.8, 1.0, 1.3, 1.7]


# ---------- exact (sampled + smoothed) entropy of y' ----------

def sign_ids(f):
  # f: (..., 9) -> integer pattern id per row
  return (f > 0.0).astype(np.int64) @ POW2


def entropy_bits_bayes(counts, alpha=1.0 / 512):
  # E[H] under a Dirichlet(alpha) posterior on the 512 cell-probabilities.
  # Closed form via digamma; alpha is a tiny pseudocount so no cell is ever 0.
  a = counts.astype(np.float64) + alpha
  A = a.sum()
  H_nats = psi(A + 1.0) - (a * psi(a + 1.0)).sum() / A
  return H_nats / LN2


def sampled_entropy(mean, cov, rng, n_samples):
  L = cholesky(cov + 1e-9 * np.eye(cov.shape[0]))
  z = rng.standard_normal((n_samples, cov.shape[0]))
  ids = sign_ids(mean[None, :] + z @ L.T)
  return entropy_bits_bayes(np.bincount(ids, minlength=512))


def expected_post_entropy(means, cov_post, rng, n_samples):
  # E_{y_B}[ H(y' | y_B) ]: cov_post is fixed, only the mean varies across draws.
  # Common random numbers (shared base draws) across the m means reduce variance.
  L = cholesky(cov_post + 1e-9 * np.eye(cov_post.shape[0]))
  base = rng.standard_normal((n_samples, cov_post.shape[0])) @ L.T
  Hs = [entropy_bits_bayes(np.bincount(sign_ids(means[m][None, :] + base), minlength=512))
        for m in range(means.shape[0])]
  return float(np.mean(Hs))


# ---------- heuristics: entropy of y' from (mean, cov) ----------

def _binary_entropy_bits(p):
  p = np.clip(p, 1e-12, 1.0 - 1e-12)
  return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))

def h_marginal(mean, cov):
  sigma = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
  return _binary_entropy_bits(norm.cdf(mean / sigma)).sum()

def _logdet_corr(cov):
  d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
  _, ld = slogdet(cov / np.outer(d, d))
  return ld                               # <= 0

def h_marginal_minus_tc(mean, cov):
  # H ~ Sum H_b(p_i) - TC, with Gaussian-copula TC = -1/2 logdet R (>= 0).
  return h_marginal(mean, cov) + 0.5 * _logdet_corr(cov) / LN2

def _effrank(A):
  # exp(entropy of normalised eigenvalues): soft, differentiable count of rank.
  eig = np.linalg.eigvalsh(0.5 * (A + A.T))
  eig = eig[eig > 0.0]
  p = eig / eig.sum()
  return float(np.exp(-(p * np.log(p)).sum()))

def h_effective_rank(mean, cov):
  return _effrank(cov)                       # mean-blind: soft rank of Sigma_g

def h_softrank_Mg(mean, cov):
  return _effrank(cov + np.outer(mean, mean))  # soft rank of M_g = Sigma_g + mu mu^T

def h_modulated_softrank(mean, cov):
  # eig-decompose Sigma = Q diag(lam) Q^T; in eigen-coords the mean is mu' = Q^T mean.
  # Replace each eigenvalue by w_k = sigma_k^2 / (sigma_k^2 + mu'_k^2) in (0,1]: ~1 for
  # uncertain directions (mu'~0), ->0 for confident ones (|mu'|>>sigma). Soft-rank of w
  # then counts genuinely-uncertain directions -- mean-aware, and (unlike M_g) it does
  # not saturate when the mean is large.
  lam, Q = np.linalg.eigh(0.5 * (cov + cov.T))
  var = np.clip(lam, 0.0, None)
  mu_rot = Q.T @ mean
  w = var / (var + mu_rot ** 2 + 1e-12)
  s = w.sum()
  if s <= 0:
    return 0.0
  p = w / s
  p = p[p > 0]
  return float(np.exp(-(p * np.log(p)).sum()))

HEURISTICS = {
  "marginal Sum H_b(p_i)": h_marginal,
  "marginal - total-corr": h_marginal_minus_tc,
  "soft-rank of M_g": h_softrank_Mg,
  "modulated soft-rank": h_modulated_softrank,
  "effective rank (cov-only)": h_effective_rank,
}


def compare_at_scale(K_gg, prior_mean, batches, rng, m_outer, s_inner, s_prior):
  """Exact vs heuristic info gain for every batch, given prior (mean, K_gg).

  K_BB in `batches` already carries the observation noise. We marginalise over the
  values the batch returns: residual r = y_B - mean_B ~ N(0, K_BB), and the
  posterior mean on G is prior_mean + r @ (K_BB^-1 K_Bg)."""
  H_prior = sampled_entropy(prior_mean, K_gg, rng, s_prior)
  prior_heur = {name: fn(prior_mean, K_gg) for name, fn in HEURISTICS.items()}

  ig_true = []
  ig_heur = {name: [] for name in HEURISTICS}
  for B, K_BB, K_gB in batches:
    sol = solve(K_BB, K_gB.T)                          # (s, n_g)
    cov_post = 0.5 * ((K_gg - K_gB @ sol) + (K_gg - K_gB @ sol).T)
    r = rng.standard_normal((m_outer, B.shape[0])) @ cholesky(K_BB).T
    means = prior_mean[None, :] + r @ sol              # (m_outer, n_g)

    ig_true.append(H_prior - expected_post_entropy(means, cov_post, rng, s_inner))
    for name, fn in HEURISTICS.items():
      h_post = np.mean([fn(means[m], cov_post) for m in range(m_outer)])
      ig_heur[name].append(prior_heur[name] - h_post)

  return H_prior, h_effective_rank(prior_mean, K_gg), np.array(ig_true), \
         {k: np.array(v) for k, v in ig_heur.items()}


def test_entropy_heuristics_vs_exact(seed, plot_root):
  rng = np.random.default_rng(int(seed))
  n, n_g, noise = 3, 9, 1e-4
  n_batches, m_outer, s_inner, s_prior = 40, 32, 2500, 40000

  g = jnp.linspace(0.0, 1.0, n)
  gx, gy = jnp.meshgrid(g, g, indexing="ij")
  X_grid = jnp.stack([gx, gy], axis=-1).reshape(n_g, 2)

  # fixed batch geometry, reused at every length-scale (kernel grams recomputed)
  batch_specs = [(int(rng.integers(1, 5)), rng.uniform(0.0, 1.0, (0, 2))) for _ in range(n_batches)]
  batch_specs = [jnp.asarray(rng.uniform(0.0, 1.0, (s, 2))) for s, _ in batch_specs]
  batch_sizes = np.array([int(B.shape[0]) for B in batch_specs])

  results = {}   # ls -> (H_prior, effrank_prior, ig_true, ig_heur)
  for ls in LENGTH_SCALES:
    kernel = kernels.rbf(length_scale=ls, variance=1.0)
    K_gg = np.asarray(kernels.gram(kernel, X_grid, X_grid))
    batches = []
    for B in batch_specs:
      K_BB = np.asarray(kernels.gram(kernel, B, B)) + noise * np.eye(B.shape[0])
      K_gB = np.asarray(kernels.gram(kernel, X_grid, B))
      batches.append((np.asarray(B), K_BB, K_gB))
    results[ls] = compare_at_scale(K_gg, np.zeros(n_g), batches, rng, m_outer, s_inner, s_prior)

  H_prior = np.array([results[ls][0] for ls in LENGTH_SCALES])
  effr_prior = np.array([results[ls][1] for ls in LENGTH_SCALES])
  rho = {name: np.array([spearmanr(results[ls][2], results[ls][3][name])[0] for ls in LENGTH_SCALES])
         for name in HEURISTICS}

  # ---- assertions ----
  assert np.all(np.diff(H_prior) < 0)                       # longer ls -> lower prior entropy
  assert np.all(np.diff(effr_prior) < 0)                    # ... and lower prior rank
  assert effr_prior[0] > 5.0 and effr_prior[-1] < 2.5       # we actually span the middle
  assert rho["marginal Sum H_b(p_i)"][0] > 0.9              # near-independent: marginal nails it
  assert rho["marginal Sum H_b(p_i)"].min() > 0.8           # marginal stays strong across the sweep
  # the mean term is what saves the rank proxy: M_g (Sigma + mu mu^T) tracks IG, but
  # the mean-blind effrank(Sigma_g) flips to anti-correlated once bits are coupled.
  assert rho["soft-rank of M_g"].min() > 0.0                # M_g stays positively correlated
  assert rho["effective rank (cov-only)"][-1] < rho["soft-rank of M_g"][-1]

  # ---- summary plot ----
  fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
  axL.plot(LENGTH_SCALES, H_prior, "o-", color="C0", label="prior H(y')  [bits]")
  axL.set_xlabel("length scale"); axL.set_ylabel("prior H(y')  [bits]", color="C0")
  axLt = axL.twinx()
  axLt.plot(LENGTH_SCALES, effr_prior, "s--", color="C3", label="prior effrank")
  axLt.set_ylabel("prior effective rank of Sigma_g", color="C3")
  axL.set_title("starting rank vs length scale (sweeping into the middle)")

  for name, marker in zip(HEURISTICS, ["o-", "s-", "^-"]):
    axR.plot(LENGTH_SCALES, rho[name], marker, label=name)
  axR.set_xlabel("length scale"); axR.set_ylabel("Spearman( heuristic IG , exact IG )")
  axR.set_ylim(0.0, 1.02); axR.axhline(1.0, color="gray", lw=0.5)
  axR.set_title("heuristic fidelity vs length scale"); axR.legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(os.path.join(plot_root, "fidelity_vs_scale.png"), dpi=120)
  plt.close(fig)

  # ---- per-scale scatter grid ----
  fig, axes = plt.subplots(len(LENGTH_SCALES), len(HEURISTICS),
                           figsize=(4.4 * len(HEURISTICS), 3.2 * len(LENGTH_SCALES)))
  for r, ls in enumerate(LENGTH_SCALES):
    _, effr, ig_true, ig_heur = results[ls]
    for c, (name, vals) in enumerate(ig_heur.items()):
      ax = axes[r, c]
      sc = ax.scatter(ig_true, vals, c=batch_sizes, cmap="viridis", s=28,
                      edgecolor="k", linewidth=0.3)
      ax.set_title(f"ls={ls}  effrank={effr:.1f}  rho={spearmanr(ig_true, vals)[0]:.2f}", fontsize=8)
      if r == len(LENGTH_SCALES) - 1:
        ax.set_xlabel("exact IG [bits]", fontsize=8)
      if c == 0:
        ax.set_ylabel(f"ls={ls}\nheuristic IG", fontsize=8)
  for c, name in enumerate(HEURISTICS):
    axes[0, c].annotate(name, xy=(0.5, 1.25), xycoords="axes fraction",
                        ha="center", fontsize=10, fontweight="bold")
  fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.01).set_label("batch size")
  fig.savefig(os.path.join(plot_root, "ig_scatter_grid.png"), dpi=110, bbox_inches="tight")
  plt.close(fig)

  print(f"\n  {'ls':>5} {'priorH':>7} {'effrank':>8} | " +
        " ".join(f"{n[:14]:>14}" for n in HEURISTICS))
  for i, ls in enumerate(LENGTH_SCALES):
    print(f"  {ls:5.1f} {H_prior[i]:7.2f} {effr_prior[i]:8.2f} | " +
          " ".join(f"{rho[n][i]:14.3f}" for n in HEURISTICS))


def test_entropy_heuristics_large_mean_range(seed, plot_root):
  """Stress test: true regime alpha*(||x||^2 - C) with a large range, so the prior
  mean spans ~ +-10 sigma, plus large observation noise. This is the regime where a
  mean spanning many sigma makes mu mu^T a dominant rank-1 spike: M_g's soft rank
  saturates near 1 and loses resolution, while the mean-aware marginal entropy holds."""
  rng = np.random.default_rng(int(seed))
  n, n_g = 3, 9
  alpha = 10.0
  noise = 1.0          # large observation-noise variance (kernel variance is 1)
  ls = 0.8
  n_batches, m_outer, s_inner, s_prior = 40, 32, 2500, 60000

  # grid on [-1,1]^2 so ||x||^2 in [0, 2]; with sigma=1 the mean spans ~ +-10 sigma.
  g = jnp.linspace(-1.0, 1.0, n)
  gx, gy = jnp.meshgrid(g, g, indexing="ij")
  X_grid = jnp.stack([gx, gy], axis=-1).reshape(n_g, 2)

  kernel = kernels.rbf(length_scale=ls, variance=1.0)
  K_gg = np.asarray(kernels.gram(kernel, X_grid, X_grid))

  C = float(rng.uniform(0.6, 1.5))                     # boundary ||x||^2 = C crosses the grid
  d2 = np.sum(np.asarray(X_grid) ** 2, axis=1)
  prior_mean = alpha * (d2 - C)                        # mean function, in sigma units (sigma=1)

  batch_specs = [jnp.asarray(rng.uniform(-1.0, 1.0, (int(rng.integers(1, 5)), 2)))
                 for _ in range(n_batches)]
  batch_sizes = np.array([int(B.shape[0]) for B in batch_specs])
  batches = []
  for B in batch_specs:
    K_BB = np.asarray(kernels.gram(kernel, B, B)) + noise * np.eye(B.shape[0])
    K_gB = np.asarray(kernels.gram(kernel, X_grid, B))
    batches.append((np.asarray(B), K_BB, K_gB))

  H_prior, _, ig_true, ig_heur = compare_at_scale(
    K_gg, prior_mean, batches, rng, m_outer, s_inner, s_prior)

  rho = {name: spearmanr(ig_true, ig_heur[name])[0] for name in HEURISTICS}

  # ---- assertions ----
  assert prior_mean.max() > 5.0 and prior_mean.min() < -5.0   # mean really spans many sigma
  assert H_prior < 5.0                                        # most bits are confident -> low entropy
  assert rho["marginal Sum H_b(p_i)"] > 0.8                   # mean-aware heuristic still tracks IG
  # the +-10 sigma mean spike saturates M_g's soft rank -> it loses to marginal here:
  assert rho["soft-rank of M_g"] < rho["marginal Sum H_b(p_i)"]

  # ---- plot ----
  fig, axes = plt.subplots(1, len(HEURISTICS), figsize=(4.6 * len(HEURISTICS), 4.4))
  for ax, name in zip(axes, HEURISTICS):
    sc = ax.scatter(ig_true, ig_heur[name], c=batch_sizes, cmap="viridis",
                    s=40, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("exact IG [bits]")
    ax.set_ylabel(f"heuristic IG: {name}")
    ax.set_title(f"{name}\nSpearman={rho[name]:.2f}", fontsize=9)
  fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.01).set_label("batch size")
  fig.suptitle(f"large mean range (alpha={alpha:.0f}, mean in "
               f"[{prior_mean.min():.0f},{prior_mean.max():.0f}] sigma), "
               f"noise var={noise}, prior H={H_prior:.2f} bits")
  fig.savefig(os.path.join(plot_root, "large_mean_range.png"), dpi=120, bbox_inches="tight")
  plt.close(fig)

  print(f"\n  mean span = [{prior_mean.min():.1f}, {prior_mean.max():.1f}] sigma, "
        f"prior H = {H_prior:.2f} bits, noise var = {noise}")
  for name in HEURISTICS:
    print(f"  {name:28s}  Spearman={rho[name]:+.3f}")


def confident_core_kernel(length_scale, s_min, s_max, r2_max):
  """Amplitude-modulated RBF: Var[f(x)] = a(x)^2 grows from s_min^2 at the centre
  to s_max^2 at the rim. k(x,x') = a(x) a(x') rho(x,x') stays PSD (D rho D)."""
  base = kernels.rbf(length_scale=length_scale, variance=1.0)
  def amp(x):
    t = jnp.clip(jnp.sum(x ** 2) / r2_max, 0.0, 1.0)
    return s_min + (s_max - s_min) * t
  def k(x1, x2):
    return amp(x1) * amp(x2) * base(x1, x2)
  return k


def test_marginalized_acq_confident_core(seed, plot_root):
  """Stress the marginalized (expected-IG) acquisition on a structured prior:
  a quadratic bump alpha*(C - ||x||^2) for the mean (positive confident core,
  negative rim, zero-crossing inside the grid) and a non-stationary kernel with
  tiny variance at the centre, large variance at the rim. The test asserts that
  geometry holds, then scores the heuristics' IG against the exact IG."""
  rng = np.random.default_rng(int(seed))
  n, n_g = 3, 9
  alpha, C = 3.0, 0.7
  s_min, s_max, ls = 0.2, 2.2, 0.8         # rim std 11x the core std
  r2_max, noise = 2.0, 0.25
  n_batches, m_outer, s_inner, s_prior = 40, 32, 2500, 60000

  g = jnp.linspace(-1.0, 1.0, n)           # grid on [-1,1]^2 -> ||x||^2 in [0, 2]
  gx, gy = jnp.meshgrid(g, g, indexing="ij")
  X_grid = jnp.stack([gx, gy], axis=-1).reshape(n_g, 2)

  kernel = confident_core_kernel(ls, s_min, s_max, r2_max)
  K_gg = np.asarray(kernels.gram(kernel, X_grid, X_grid)).astype(np.float64)

  d2 = np.sum(np.asarray(X_grid) ** 2, axis=1)
  prior_mean = alpha * (C - d2)            # bump: +alpha*C at centre, negative on the rim
  var = np.diag(K_gg)
  center = int(np.argmin(d2))
  outside = np.array([i for i in range(n_g) if i != center])

  # ---- check the test: the prior actually has the intended geometry ----
  assert prior_mean.min() < 0.0 < prior_mean.max()          # mean crosses 0 within the grid
  assert prior_mean[center] > 0.0                           # positive confident core
  assert var[center] == var.min()                           # lowest variance at the centre
  assert var[center] < 0.1 * prior_mean[center]             # centre: variance << mean
  assert np.all(var[outside] > np.abs(prior_mean[outside]))  # rim: variance > |mean|

  batch_specs = [jnp.asarray(rng.uniform(-1.0, 1.0, (int(rng.integers(1, 5)), 2)))
                 for _ in range(n_batches)]
  batch_sizes = np.array([int(B.shape[0]) for B in batch_specs])
  batches = []
  for B in batch_specs:
    K_BB = np.asarray(kernels.gram(kernel, B, B)).astype(np.float64) + noise * np.eye(B.shape[0])
    K_gB = np.asarray(kernels.gram(kernel, X_grid, B)).astype(np.float64)
    batches.append((np.asarray(B), K_BB, K_gB))

  H_prior, _, ig_true, ig_heur = compare_at_scale(
    K_gg, prior_mean, batches, rng, m_outer, s_inner, s_prior)
  rho = {name: spearmanr(ig_true, ig_heur[name])[0] for name in HEURISTICS}

  assert rho["marginal Sum H_b(p_i)"] > 0.7                 # marginalized acq still ordered by marginal

  # ---- prior geometry plot ----
  fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
  for ax, (field, title, cmap) in zip(axes, [
      (prior_mean, "prior mean (bump)", "coolwarm"),
      (var, "prior variance", "viridis"),
      (prior_mean / np.sqrt(var), "mean / std", "coolwarm")]):
    im = ax.imshow(field.reshape(n, n).T, origin="lower", extent=(-1, 1, -1, 1), cmap=cmap)
    ax.set_title(title); plt.colorbar(im, ax=ax)
  fig.suptitle(f"confident core, uncertain rim (prior H={H_prior:.2f} bits)")
  fig.tight_layout()
  fig.savefig(os.path.join(plot_root, "prior_geometry.png"), dpi=120)
  plt.close(fig)

  # ---- acquisition fidelity plot ----
  fig, axes = plt.subplots(1, len(HEURISTICS), figsize=(4.6 * len(HEURISTICS), 4.4))
  for ax, name in zip(axes, HEURISTICS):
    sc = ax.scatter(ig_true, ig_heur[name], c=batch_sizes, cmap="viridis",
                    s=40, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("exact IG [bits]"); ax.set_ylabel(f"heuristic IG: {name}")
    ax.set_title(f"{name}\nSpearman={rho[name]:.2f}", fontsize=9)
  fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.01).set_label("batch size")
  fig.savefig(os.path.join(plot_root, "acq_scatter.png"), dpi=120, bbox_inches="tight")
  plt.close(fig)

  print(f"\n  centre: mean={prior_mean[center]:.2f} var={var[center]:.3f} | "
        f"rim var range=[{var[outside].min():.2f},{var[outside].max():.2f}] "
        f"|mean| range=[{np.abs(prior_mean[outside]).min():.2f},{np.abs(prior_mean[outside]).max():.2f}]")
  print(f"  prior H = {H_prior:.2f} bits")
  for name in HEURISTICS:
    print(f"  {name:28s}  Spearman={rho[name]:+.3f}")


def _border_batch(rng, C, n_pts, jitter=0.08):
  """n_pts points hugging the circle ||x||^2 = C, spread over a random arc."""
  theta0 = rng.uniform(0.0, 2.0 * np.pi)
  half = rng.uniform(0.3, np.pi)                 # arc half-width drives how much border is covered
  theta = theta0 + rng.uniform(-half, half, n_pts)
  r = np.sqrt(C) + rng.normal(0.0, jitter, n_pts)
  pts = np.clip(np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1), -1.0, 1.0)
  return pts, half


def test_marginalized_acq_border_batch16(seed, plot_root):
  """Large border-hugging batches (size 16) on the confident-core prior. 16 points
  packed near the decision boundary ||x||^2 = C are strongly mutually correlated and
  strongly correlated with the rim grid points -- the redundant-cluster regime where
  ignoring grid-point correlations should hurt the marginal heuristic most, if ever."""
  rng = np.random.default_rng(int(seed))
  n, n_g = 3, 9
  alpha, C = 3.0, 0.7
  s_min, s_max, ls = 0.2, 2.2, 0.8
  r2_max, noise, batch_size = 2.0, 0.25, 16
  n_batches, m_outer, s_inner, s_prior = 40, 32, 2500, 60000

  g = jnp.linspace(-1.0, 1.0, n)
  gx, gy = jnp.meshgrid(g, g, indexing="ij")
  X_grid = jnp.stack([gx, gy], axis=-1).reshape(n_g, 2)

  kernel = confident_core_kernel(ls, s_min, s_max, r2_max)
  K_gg = np.asarray(kernels.gram(kernel, X_grid, X_grid)).astype(np.float64)
  d2 = np.sum(np.asarray(X_grid) ** 2, axis=1)
  prior_mean = alpha * (C - d2)

  batches, arc, all_pts = [], [], []
  for _ in range(n_batches):
    pts, half = _border_batch(rng, C, batch_size)
    arc.append(half); all_pts.append(pts)
    B = jnp.asarray(pts)
    K_BB = np.asarray(kernels.gram(kernel, B, B)).astype(np.float64) + noise * np.eye(batch_size)
    K_gB = np.asarray(kernels.gram(kernel, X_grid, B)).astype(np.float64)
    batches.append((pts, K_BB, K_gB))
  arc = np.array(arc)

  # ---- check the test: every batch is size 16 and hugs the border ----
  assert all(B.shape[0] == batch_size for B, _, _ in batches)
  dist_to_border = np.abs(np.linalg.norm(np.concatenate(all_pts), axis=1) - np.sqrt(C))
  assert np.median(dist_to_border) < 0.2

  H_prior, _, ig_true, ig_heur = compare_at_scale(
    K_gg, prior_mean, batches, rng, m_outer, s_inner, s_prior)
  rho = {name: spearmanr(ig_true, ig_heur[name])[0] for name in HEURISTICS}

  assert rho["marginal Sum H_b(p_i)"] > 0.7      # marginal survives the redundant cluster

  # ---- plot: example batches over the border + fidelity scatters ----
  fig, axes = plt.subplots(1, len(HEURISTICS) + 1, figsize=(4.6 * (len(HEURISTICS) + 1), 4.4))
  ax = axes[0]
  ax.add_patch(plt.Circle((0, 0), np.sqrt(C), fill=False, color="r", lw=1.5))
  for k in range(0, n_batches, max(1, n_batches // 4)):
    ax.scatter(all_pts[k][:, 0], all_pts[k][:, 1], s=18, label=f"arc={arc[k]:.1f}")
  ax.scatter(np.asarray(X_grid)[:, 0], np.asarray(X_grid)[:, 1], c="k", marker="s", s=40)
  ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_aspect("equal")
  ax.set_title("border (red) + example 16-pt batches"); ax.legend(fontsize=7)
  for ax, name in zip(axes[1:], HEURISTICS):
    sc = ax.scatter(ig_true, ig_heur[name], c=arc, cmap="plasma", s=40, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("exact IG [bits]"); ax.set_ylabel(f"heuristic IG: {name}")
    ax.set_title(f"{name}\nSpearman={rho[name]:.2f}", fontsize=9)
  fig.colorbar(sc, ax=axes.tolist(), fraction=0.02, pad=0.01).set_label("arc half-width")
  fig.savefig(os.path.join(plot_root, "border_batch16.png"), dpi=120, bbox_inches="tight")
  plt.close(fig)

  print(f"\n  batch size = {batch_size}, near border (median dist={np.median(dist_to_border):.3f}), "
        f"prior H = {H_prior:.2f} bits")
  print(f"  exact IG range = [{ig_true.min():.2f}, {ig_true.max():.2f}] bits")
  for name in HEURISTICS:
    print(f"  {name:28s}  Spearman={rho[name]:+.3f}")


def test_marginalized_acq_two_level_mean(seed, plot_root):
  """Intermediate mean regime: center mean +1, outside mean -1, stationary sigma=0.5
  everywhere -> every bit at ~ +-2 sigma. Between the O(sigma) sweep (M_g fine) and
  the +-10 sigma case (M_g collapsed): here ||mu||^2 ~ 9 already dominates
  trace(Sigma) ~ 2.25, so M_g's spike should start saturating effrank."""
  rng = np.random.default_rng(int(seed))
  n, n_g = 3, 9
  sigma, ls, noise = 0.5, 0.8, 0.1
  n_batches, m_outer, s_inner, s_prior = 40, 32, 2500, 60000

  g = jnp.linspace(-1.0, 1.0, n)
  gx, gy = jnp.meshgrid(g, g, indexing="ij")
  X_grid = jnp.stack([gx, gy], axis=-1).reshape(n_g, 2)

  kernel = kernels.rbf(length_scale=ls, variance=sigma ** 2)   # stationary -> std = sigma everywhere
  K_gg = np.asarray(kernels.gram(kernel, X_grid, X_grid)).astype(np.float64)

  d2 = np.sum(np.asarray(X_grid) ** 2, axis=1)
  center = int(np.argmin(d2))
  prior_mean = np.full(n_g, -1.0)
  prior_mean[center] = 1.0                                     # +1 at centre, -1 outside
  var = np.diag(K_gg)

  # ---- check the test ----
  assert prior_mean[center] == 1.0 and np.all(prior_mean[np.arange(n_g) != center] == -1.0)
  assert np.allclose(np.sqrt(var), sigma, atol=1e-5)           # sigma ~ 0.5 everywhere
  assert np.allclose(np.abs(prior_mean) / np.sqrt(var), 2.0, atol=1e-5)  # every bit at +-2 sigma

  batch_specs = [jnp.asarray(rng.uniform(-1.0, 1.0, (int(rng.integers(1, 5)), 2)))
                 for _ in range(n_batches)]
  batch_sizes = np.array([int(B.shape[0]) for B in batch_specs])
  batches = []
  for B in batch_specs:
    K_BB = np.asarray(kernels.gram(kernel, B, B)).astype(np.float64) + noise * np.eye(B.shape[0])
    K_gB = np.asarray(kernels.gram(kernel, X_grid, B)).astype(np.float64)
    batches.append((np.asarray(B), K_BB, K_gB))

  H_prior, _, ig_true, ig_heur = compare_at_scale(
    K_gg, prior_mean, batches, rng, m_outer, s_inner, s_prior)
  rho = {name: spearmanr(ig_true, ig_heur[name])[0] for name in HEURISTICS}

  assert rho["marginal Sum H_b(p_i)"] > 0.7

  fig, axes = plt.subplots(1, len(HEURISTICS), figsize=(4.6 * len(HEURISTICS), 4.4))
  for ax, name in zip(axes, HEURISTICS):
    sc = ax.scatter(ig_true, ig_heur[name], c=batch_sizes, cmap="viridis",
                    s=40, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("exact IG [bits]"); ax.set_ylabel(f"heuristic IG: {name}")
    ax.set_title(f"{name}\nSpearman={rho[name]:.2f}", fontsize=9)
  fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.01).set_label("batch size")
  fig.suptitle(f"two-level mean (+1 / -1), sigma={sigma}, every bit +-2 sigma, prior H={H_prior:.2f} bits")
  fig.savefig(os.path.join(plot_root, "two_level_mean.png"), dpi=120, bbox_inches="tight")
  plt.close(fig)

  print(f"\n  every bit at +-2 sigma, prior H = {H_prior:.2f} bits, "
        f"exact IG range = [{ig_true.min():.2f}, {ig_true.max():.2f}] bits")
  for name in HEURISTICS:
    print(f"  {name:28s}  Spearman={rho[name]:+.3f}")


def test_doe_optimize_continuous_batch(seed, plot_root):
  """Realistic DoE. Ground truth f(x) = ||x||^2 - 1 over [-2,2]^2 (boundary = unit
  circle), observation noise 0.25. Fit a GP to 8 noisy points near the centre, then
  optimise a continuous batch over the GP-*posterior* marginal-sign-entropy EIG.
  Compare the optimised batch against random designs and against a 'boundary' design
  of 4 points on the GP posterior-mean zero contour, and score every heuristic's EIG."""
  from doe.gp import acquisition as acq, GP

  rng = np.random.default_rng(int(seed))
  key = jax.random.PRNGKey(int(seed))
  noise, batch_size = 0.25, 4
  n_random, m_outer = 120, 96
  bounds = [(-2.0, 2.0), (-2.0, 2.0)]
  mk = "marginal Sum H_b(p_i)"

  def f_true(X):
    return np.sum(np.asarray(X) ** 2, axis=-1) - 1.0

  # 8 training points near the centre (disk r<=1.3, straddling the unit-circle boundary)
  rr = 1.3 * np.sqrt(rng.uniform(0.0, 1.0, 8))
  th = rng.uniform(0.0, 2 * np.pi, 8)
  X_train = np.stack([rr * np.cos(th), rr * np.sin(th)], axis=1)
  y_train = f_true(X_train) + np.sqrt(noise) * rng.normal(size=8)

  kernel = kernels.rbf(length_scale=1.0, variance=4.0)
  gp = GP(kernel, noise=noise)
  state = gp.fit(jnp.asarray(X_train), jnp.asarray(y_train))

  ng1, n_g = 5, 25
  g = jnp.linspace(-2.0, 2.0, ng1)
  gx, gy = jnp.meshgrid(g, g, indexing="ij")
  X_grid = jnp.stack([gx, gy], axis=-1).reshape(n_g, 2)
  X_grid_np = np.asarray(X_grid)

  # --- optimise the batch over the GP-posterior marginal EIG ---
  B_opt, eig_opt = acq.optimize_marginal_eig_state(gp, state, X_grid, batch_size, bounds, key, n_restarts=16)

  # --- GP posterior-mean zero contour -> 'boundary' design of 4 points ---
  gl = np.linspace(-2, 2, 200)
  MX, MY = np.meshgrid(gl, gl)
  mean_field = np.asarray(gp.predict(
    state, jnp.asarray(np.stack([MX, MY], -1).reshape(-1, 2)))[0]).reshape(MX.shape)
  try:
    cs = plt.contour(MX, MY, mean_field, levels=[0.0])
    segs = [s for s in cs.allsegs[0] if len(s)]
    plt.close()
  except Exception:
    segs = []
  if segs:
    verts = np.concatenate(segs, axis=0)
    ang = np.arctan2(verts[:, 1], verts[:, 0])
    B_zero = np.array([verts[np.argmin(np.abs(((ang - a + np.pi) % (2 * np.pi)) - np.pi))]
                       for a in [0.0, np.pi / 2, np.pi, -np.pi / 2]])
  else:                                       # mean never crosses 0: take the 4 closest-to-0 points
    flat = np.stack([MX, MY], -1).reshape(-1, 2)
    B_zero = flat[np.argsort(np.abs(mean_field).ravel())[:batch_size]]

  rand_Bs = [rng.uniform(-2.0, 2.0, (batch_size, 2)) for _ in range(n_random)]

  # --- EIG of every heuristic for a batch, given the fitted GP (MC over outcomes) ---
  def eig_all(B):
    B = np.asarray(B)
    m, cov = gp.predict(state, jnp.asarray(np.concatenate([X_grid_np, B], axis=0)), full_cov=True)
    m, cov = np.asarray(m).astype(float), np.asarray(cov).astype(float)
    m_g, S_gg = m[:n_g], cov[:n_g, :n_g]
    S_gB, S_BB = cov[:n_g, n_g:], cov[n_g:, n_g:]
    M = S_BB + noise * np.eye(B.shape[0])
    A = np.linalg.solve(M, S_gB.T)                          # (nb, n_g)
    S_post = 0.5 * ((S_gg - S_gB @ A) + (S_gg - S_gB @ A).T)
    r = rng.standard_normal((m_outer, B.shape[0])) @ np.linalg.cholesky(M).T
    mean_draws = m_g[None, :] + r @ A                       # (m_outer, n_g)
    return {name: fn(m_g, S_gg) - np.mean([fn(mean_draws[i], S_post) for i in range(m_outer)])
            for name, fn in HEURISTICS.items()}

  eig_opt_all = eig_all(B_opt)
  eig_zero_all = eig_all(B_zero)
  eig_rand_list = [eig_all(B) for B in rand_Bs]
  eig_rand = {name: np.array([e[name] for e in eig_rand_list]) for name in HEURISTICS}
  rho_vs_marg = {name: spearmanr(eig_rand[mk], eig_rand[name])[0] for name in HEURISTICS}

  var_opt = float(np.mean(np.asarray(gp.predict(state, jnp.asarray(np.asarray(B_opt)))[1])))
  var_zero = float(np.mean(np.asarray(gp.predict(state, jnp.asarray(B_zero))[1])))

  print(f"\n  optimised marginal EIG = {eig_opt:.3f} bits;  mean post-var at points: "
        f"optimised={var_opt:.2f}  boundary={var_zero:.2f}")
  print(f"  {'heuristic':28s} {'EIG(opt)':>9} {'EIG(bndry)':>11} {'EIG rand max':>13} {'rho vs marg':>12}")
  for nm in HEURISTICS:
    print(f"  {nm:28s} {eig_opt_all[nm]:9.3f} {eig_zero_all[nm]:11.3f} "
          f"{eig_rand[nm].max():13.3f} {rho_vs_marg[nm]:12.3f}")

  # ---- assertions ----
  assert eig_opt_all[mk] > eig_rand[mk].max()                # optimiser beats every random design
  assert eig_opt_all[mk] > eig_zero_all[mk]                  # ... and the GP_mean=0 (boundary) design
  assert var_opt > var_zero                                  # it targets higher-variance (more uncertain) bits
  # modest means here -> M_g tracks marginal well; the eigen-basis modulation does NOT help
  assert rho_vs_marg["soft-rank of M_g"] > 0.8
  assert rho_vs_marg["modulated soft-rank"] < rho_vs_marg["soft-rank of M_g"]

  # ---- plot ----
  fig, (axf, axb) = plt.subplots(1, 2, figsize=(13, 5.2))
  im = axf.imshow(mean_field, origin="lower", extent=(-2, 2, -2, 2), cmap="coolwarm",
                  vmin=-np.abs(mean_field).max(), vmax=np.abs(mean_field).max())
  axf.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="k", lw=1.5, ls="--", label="true boundary"))
  axf.contour(MX, MY, mean_field, levels=[0.0], colors="cyan", linewidths=1.2)
  axf.scatter(X_train[:, 0], X_train[:, 1], c="white", marker="x", s=50, label="train (n=8)")
  axf.scatter(B_zero[:, 0], B_zero[:, 1], c="cyan", s=70, marker="o", edgecolor="k", label="GP_mean=0 design")
  axf.scatter(np.asarray(B_opt)[:, 0], np.asarray(B_opt)[:, 1], c="lime", s=150, marker="*",
              edgecolor="k", label="optimised batch")
  axf.set_xlim(-2, 2); axf.set_ylim(-2, 2); axf.set_aspect("equal")
  axf.set_title("GP posterior mean + designs"); axf.legend(fontsize=8, loc="upper right")
  plt.colorbar(im, ax=axf, label="GP posterior mean")

  names = list(HEURISTICS)
  axb.barh(range(len(names)), [rho_vs_marg[nm] for nm in names], color="steelblue")
  axb.set_yticks(range(len(names))); axb.set_yticklabels(names, fontsize=8)
  axb.set_xlabel("Spearman(heuristic EIG, marginal EIG) over random designs")
  axb.axvline(0, color="k", lw=0.5); axb.set_xlim(-1, 1.05)
  axb.set_title("do the heuristics rank designs like marginal?")
  fig.tight_layout()
  fig.savefig(os.path.join(plot_root, "doe_realistic.png"), dpi=120)
  plt.close(fig)
