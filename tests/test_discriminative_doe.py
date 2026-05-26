"""Sanity tests for the package discriminative-DoE tool (doe.doe.discriminative)."""
import numpy as np
import jax
import jax.numpy as jnp

from doe.gp import kernels
from doe.gp.gp import GP, State
from doe.doe.discriminative import DiscriminativeDoE, optimize_batch, PROXIES


def _toy(n_train=12, grid=8):
  # f(x) = x0 + x1 on [0,1]^2; sign boundary {f > 1} is the anti-diagonal.
  rng = np.random.default_rng(0)
  X = rng.uniform(0.0, 1.0, (n_train, 2))
  y = X[:, 0] + X[:, 1]
  gp = GP(kernels.rbf_ard(jnp.array([0.3, 0.3], jnp.float32), variance=1.0), noise=1e-3)
  state = gp.fit(jnp.asarray(X, jnp.float32), jnp.asarray(y, jnp.float32))
  ax = np.linspace(0.0, 1.0, grid)
  Xg = jnp.asarray(np.stack(np.meshgrid(ax, ax), -1).reshape(-1, 2), jnp.float32)
  f_true = np.asarray(Xg)[:, 0] + np.asarray(Xg)[:, 1]
  return gp, state, Xg, f_true


def test_every_proxy_returns_a_valid_batch():
  gp, state, Xg, _ = _toy()
  for proxy in PROXIES:
    doe = DiscriminativeDoE(gp, Xg, threshold=1.0, proxy=proxy, n_outer=6,
                            n_multi_start=2, n_steps=40)
    B, eig = doe.suggest(state, batch_size=3, bounds=[(0.0, 1.0), (0.0, 1.0)],
                         key=jax.random.PRNGKey(0))
    B = np.asarray(B)
    assert B.shape == (3, 2)
    assert np.all(B >= -1e-5) and np.all(B <= 1.0 + 1e-5)   # inside the box
    assert np.isfinite(eig)


def test_sign_probability_and_accuracy_bounds():
  gp, state, Xg, f_true = _toy()
  doe = DiscriminativeDoE(gp, Xg, threshold=1.0)
  q = np.asarray(doe.sign_probability(state))
  assert q.shape == (Xg.shape[0],)
  assert np.all(q >= 0.0) and np.all(q <= 1.0)
  acc = doe.accuracy(state, f_true)
  assert 0.0 <= acc <= 1.0
  assert acc > 0.8                                          # GP recovers the linear sign well


def test_marginal_eig_nonnegative_from_prior():
  gp, _, Xg, _ = _toy()
  empty = State(X_flat=jnp.zeros((0, 2)), L=jnp.zeros((0, 0)), alpha=jnp.zeros((0,)))
  B, eig = optimize_batch(gp, empty, Xg, threshold=1.0, batch_size=3,
                          bounds=[(0.0, 1.0), (0.0, 1.0)], key=jax.random.PRNGKey(1),
                          proxy="marginal", n_outer=8, n_multi_start=2, n_steps=40)
  assert np.asarray(B).shape == (3, 2)
  assert eig >= -1e-4                                       # measuring can't increase entropy


def test_unknown_proxy_raises():
  gp, _, Xg, _ = _toy()
  try:
    DiscriminativeDoE(gp, Xg, threshold=1.0, proxy="nope")
  except ValueError:
    return
  raise AssertionError("expected ValueError for unknown proxy")
