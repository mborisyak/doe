import jax
import jax.numpy as jnp
import pytest

from doe.gp import GP, kernels


@pytest.fixture
def data():
  X = jnp.linspace(0.0, 4.0, 8).reshape(-1, 1)
  y = jnp.sin(X).flatten()
  X_test = jnp.linspace(0.0, 4.0, 20).reshape(-1, 1)
  return X, y, X_test


def test_rbf_self_kernel_is_one():
  k = kernels.rbf(length_scale=1.0, variance=1.0)
  assert jnp.allclose(k(jnp.zeros(1), jnp.zeros(1)), 1.0)


def test_fit_predict_shapes(data):
  X, y, X_test = data
  gp = GP(kernels.rbf(length_scale=1.0), noise=1.0e-4)
  state = gp.fit(X, y)
  mean, var = gp.predict(state, X_test)
  assert mean.shape == (X_test.shape[0],)
  assert var.shape == (X_test.shape[0],)
  assert jnp.all(var >= -1.0e-6)


def test_predict_full_cov(data):
  X, y, X_test = data
  gp = GP(kernels.rbf(), noise=1.0e-4)
  state = gp.fit(X, y)
  mean, cov = gp.predict(state, X_test, full_cov=True)
  assert cov.shape == (X_test.shape[0], X_test.shape[0])
  _, var = gp.predict(state, X_test)
  assert jnp.allclose(jnp.diag(cov), var, atol=1.0e-6)


def test_interpolates_training_points(data):
  X, y, _ = data
  gp = GP(kernels.rbf(length_scale=1.0), noise=1.0e-8)
  state = gp.fit(X, y)
  mean, var = gp.predict(state, X)
  assert jnp.allclose(mean, y, atol=1.0e-3)
  assert jnp.all(var < 1.0e-3)


def test_log_marginal_likelihood_finite(data):
  X, y, _ = data
  gp = GP(kernels.rbf(), noise=1.0e-3)
  state = gp.fit(X, y)
  lml = gp.log_marginal_likelihood(state, y)
  assert jnp.isfinite(lml)


def test_arbitrary_kernel_callable(data):
  X, y, X_test = data

  def my_kernel(x1, x2):
    return jnp.exp(-jnp.linalg.norm(x1 - x2))

  gp = GP(my_kernel, noise=1.0e-3)
  state = gp.fit(X, y)
  mean, var = gp.predict(state, X_test)
  assert mean.shape == (X_test.shape[0],)


def test_kernel_composition(data):
  X, y, X_test = data
  k = kernels.sum_of(kernels.rbf(length_scale=0.5, variance=0.5),
                     kernels.matern52(length_scale=2.0, variance=0.5))
  gp = GP(k, noise=1.0e-4)
  state = gp.fit(X, y)
  mean, _ = gp.predict(state, X_test)
  assert mean.shape == (X_test.shape[0],)


def test_sample_shape(data):
  X, y, X_test = data
  gp = GP(kernels.rbf(), noise=1.0e-3)
  state = gp.fit(X, y)
  s = gp.sample(state, X_test, jax.random.PRNGKey(0), num_samples=4)
  assert s.shape == (4, X_test.shape[0])


def test_prob_below_above_sum_to_one(data):
  X, y, X_test = data
  gp = GP(kernels.rbf(), noise=1.0e-3)
  state = gp.fit(X, y)
  mean, var = gp.predict(state, X_test)
  std = jnp.sqrt(var + gp.noise)
  threshold = 0.5
  p_lo = gp.prob_below(mean, std, threshold)
  p_hi = gp.prob_above(mean, std, threshold)
  assert jnp.allclose(p_lo + p_hi, 1.0, atol=1.0e-6)
  assert jnp.all((p_lo >= 0.0) & (p_lo <= 1.0))


def test_prob_below_monotonic_in_threshold(data):
  X, y, X_test = data
  gp = GP(kernels.rbf(), noise=1.0e-3)
  state = gp.fit(X, y)
  mean, var = gp.predict(state, X_test)
  std = jnp.sqrt(var + gp.noise)
  assert jnp.all(gp.prob_below(mean, std, -10.0) < gp.prob_below(mean, std, 10.0))


def test_cross_entropy_self_is_binary_entropy():
  # BCE(p, p) = H(p) = -p log p - (1-p) log(1-p)
  m, v = 0.3, 0.5
  h = GP.cross_entropy(m, v, m, v, threshold=0.0)
  p = jax.scipy.stats.norm.sf(0.0, loc=m, scale=jnp.sqrt(v))
  expected = -p * jnp.log(p) - (1.0 - p) * jnp.log(1.0 - p)
  assert jnp.allclose(h, expected, atol=1.0e-6)


def test_cross_entropy_asymmetric():
  a = GP.cross_entropy(0.0, 1.0, 1.0, 4.0, threshold=0.0)
  b = GP.cross_entropy(1.0, 4.0, 0.0, 1.0, threshold=0.0)
  assert not jnp.allclose(a, b)


def test_bald_nonneg_and_zero_at_certainty():
  gp = GP(kernels.rbf(), noise=1.0e-3)
  # When sigma is tiny, p_now ≈ 0 or 1 for any tube (certain), BALD ≈ 0
  mean = jnp.array([0.0, 5.0])
  var = jnp.array([1.0e-8, 1.0e-8])
  target = jnp.array([0.0, 0.0])
  a = gp.bald(mean, var, target, eps=0.5)
  assert jnp.all(a >= -1.0e-6)
  assert jnp.all(a < 1.0e-3)


def test_bald_peaks_near_tube_edge():
  gp = GP(kernels.rbf(), noise=1.0e-2)
  # Sweep the predictive mean across a fixed target with moderate sigma.
  # BALD should peak somewhere with substantial p_now uncertainty (near tube edge).
  target = 0.0
  eps = 1.0
  var = jnp.full((201,), 0.5)
  mean = jnp.linspace(-3.0, 3.0, 201)
  a = gp.bald(mean, var, target, eps)
  # peak is in the bulk, not at the extreme ends
  assert jnp.argmax(a) > 10 and jnp.argmax(a) < 190
  assert jnp.all(a >= -1.0e-6)


def test_jit_compatible(data):
  X, y, X_test = data
  gp = GP(kernels.rbf(length_scale=1.0), noise=1.0e-4)

  @jax.jit
  def predict_jit(X, y, X_test):
    state = gp.fit(X, y)
    return gp.predict(state, X_test)

  mean, var = predict_jit(X, y, X_test)
  assert mean.shape == (X_test.shape[0],)