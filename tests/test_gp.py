import time
import jax
import jax.numpy as jnp
import pytest

from doe.gp import GP, kernels


@pytest.fixture
def data_2d():
  # X: (n_1, n_2, 2) grid on [0, 1]², y: (n_1, n_2) = sin(x0 * x1)
  n_1, n_2 = 6, 7
  g1 = jnp.linspace(0.0, 1.0, n_1)
  g2 = jnp.linspace(0.0, 1.0, n_2)
  gx, gy = jnp.meshgrid(g1, g2, indexing="ij")
  X = jnp.stack([gx, gy], axis=-1)  # (n_1, n_2, 2)
  y = jnp.sin(X[..., 0] * X[..., 1])  # (n_1, n_2)
  # test grid offset from training points
  tg1 = jnp.linspace(0.05, 0.95, 4)
  tg2 = jnp.linspace(0.05, 0.95, 5)
  tx, ty = jnp.meshgrid(tg1, tg2, indexing="ij")
  X_test = jnp.stack([tx, ty], axis=-1)  # (4, 5, 2)
  return X, y, X_test


def test_rbf_self_kernel_is_one():
  k = kernels.rbf(length_scale=1.0, variance=1.0)
  assert jnp.allclose(k(jnp.zeros(2), jnp.zeros(2)), 1.0)


def test_fit_predict_shapes(data_2d):
  X, y, X_test = data_2d  # X: (6, 7, 2), y: (6, 7), X_test: (4, 5, 2)
  gp = GP(kernels.rbf(length_scale=0.3), noise=1.0e-4)
  state = gp.fit(X, y)
  mean, var = gp.predict(state, X_test)
  assert state.X_flat.shape == (42, 2)
  assert state.L.shape == (42, 42)
  assert mean.shape == (4, 5)
  assert var.shape == (4, 5)
  assert jnp.all(var >= -1.0e-6)


def test_predict_full_cov(data_2d):
  X, y, X_test = data_2d
  gp = GP(kernels.rbf(length_scale=0.3), noise=1.0e-4)
  state = gp.fit(X, y)
  mean, cov = gp.predict(state, X_test, full_cov=True)
  assert mean.shape == (4, 5)
  assert cov.shape == (4, 5, 4, 5)
  cov_2d = cov.reshape(20, 20)
  assert jnp.allclose(cov_2d, cov_2d.T, atol=1.0e-5)
  _, var = gp.predict(state, X_test)
  # diag of cov should match var (flattened)
  diag = jnp.einsum("ijij->ij", cov)
  assert jnp.allclose(diag, var, atol=1.0e-6)


def test_predict_full_cov_psd(data_2d):
  X, y, X_test = data_2d
  gp = GP(kernels.rbf(length_scale=0.3), noise=1.0e-4)
  state = gp.fit(X, y)
  _, cov = gp.predict(state, X_test, full_cov=True)
  cov_2d = cov.reshape(20, 20)
  eigs = jnp.linalg.eigvalsh(0.5 * (cov_2d + cov_2d.T))
  assert eigs.min() > -1.0e-5


def test_batch_predict_shapes(data_2d):
  X, y, _ = data_2d
  gp = GP(kernels.rbf(length_scale=0.3), noise=1.0e-4)
  state = gp.fit(X, y)
  # batch dims (3, 2), n_test=5, n_x=2
  X_test = jax.random.uniform(jax.random.PRNGKey(0), shape=(3, 2, 5, 2))
  mean, cov = gp.batch_predict(state, X_test)
  assert mean.shape == (3, 2, 5)
  assert cov.shape == (3, 2, 5, 5)


def test_batch_predict_matches_predict(data_2d):
  X, y, _ = data_2d
  gp = GP(kernels.rbf(length_scale=0.3), noise=1.0e-4)
  state = gp.fit(X, y)
  X_test = jax.random.uniform(jax.random.PRNGKey(0), shape=(1, 12, 2))
  mean_b, cov_b = gp.batch_predict(state, X_test)
  mean_p, cov_p = gp.predict(state, X_test[0], full_cov=True)
  assert jnp.allclose(mean_b[0], mean_p, atol=1.0e-6)
  assert jnp.allclose(cov_b[0], cov_p, atol=1.0e-6)


def test_batch_predict_singleton_matches_predict_var(data_2d):
  X, y, _ = data_2d
  gp = GP(kernels.rbf(length_scale=0.3), noise=1.0e-4)
  state = gp.fit(X, y)
  # (B, n_x) for predict vs (B, 1, n_x) for batch_predict
  X_test = jax.random.uniform(jax.random.PRNGKey(0), shape=(6, 2))
  mean_p, var_p = gp.predict(state, X_test)
  mean_b, cov_b = gp.batch_predict(state, X_test[:, None, :])
  assert mean_b.shape == (6, 1)
  assert cov_b.shape == (6, 1, 1)
  assert jnp.allclose(mean_b[:, 0], mean_p, atol=1.0e-6)
  assert jnp.allclose(cov_b[:, 0, 0], var_p, atol=1.0e-6)


def test_batch_predict_pair_matches_predict_diag_blocks(data_2d):
  X, y, _ = data_2d
  gp = GP(kernels.rbf(length_scale=0.3), noise=1.0e-4)
  state = gp.fit(X, y)
  B = 6
  X_test = jax.random.uniform(jax.random.PRNGKey(0), shape=(B, 2, 2))
  mean_p, cov_p = gp.predict(state, X_test, full_cov=True)  # (B, 2), (B, 2, B, 2)
  mean_b, cov_b = gp.batch_predict(state, X_test)            # (B, 2), (B, 2, 2)
  assert jnp.allclose(mean_b, mean_p, atol=1.0e-6)
  idx = jnp.arange(B)
  diag_blocks = cov_p[idx, :, idx, :]
  assert jnp.allclose(cov_b, diag_blocks, atol=1.0e-6)


def test_batch_predict_independent_batches(seed, data_2d):
  rng = jax.random.PRNGKey(seed)

  X, y, _ = data_2d
  gp = GP(kernels.rbf(length_scale=0.3), noise=1.0e-4)
  state = gp.fit(X, y)
  X_test = jax.random.uniform(rng, shape=(3, 4, 2))  # (B=3, n_test=4, n_x=2)

  mean, cov = gp.batch_predict(state, X_test)

  for b in range(X_test.shape[0]):
    mean_p, cov_p = gp.predict(state, X_test[b], full_cov=True)
    assert jnp.allclose(mean[b], mean_p, atol=1.0e-6)
    assert jnp.allclose(cov[b], cov_p, atol=1.0e-6)


def test_interpolates_training_points(data_2d):
  X, y, _ = data_2d
  gp = GP(kernels.rbf(length_scale=0.3), noise=1.0e-8)
  state = gp.fit(X, y)
  mean, var = gp.predict(state, X)
  assert mean.shape == y.shape
  assert jnp.allclose(mean, y, atol=1.0e-3)
  assert jnp.all(var < 1.0e-3)


def test_log_marginal_likelihood_finite(data_2d):
  X, y, _ = data_2d
  gp = GP(kernels.rbf(length_scale=0.3), noise=1.0e-3)
  lml = gp.log_marginal_likelihood(X, y)
  assert jnp.isfinite(lml)


def test_arbitrary_kernel_callable(data_2d):
  X, y, X_test = data_2d

  def my_kernel(x1, x2):
    return jnp.exp(-jnp.linalg.norm(x1 - x2))

  gp = GP(my_kernel, noise=1.0e-3)
  state = gp.fit(X, y)
  mean, var = gp.predict(state, X_test)
  assert mean.shape == (4, 5)


def test_kernel_composition(data_2d):
  X, y, X_test = data_2d
  k = kernels.sum_of(kernels.rbf(length_scale=0.3, variance=0.5),
                     kernels.matern52(length_scale=0.5, variance=0.5))
  gp = GP(k, noise=1.0e-4)
  state = gp.fit(X, y)
  mean, _ = gp.predict(state, X_test)
  assert mean.shape == (4, 5)


def test_sample_shape(data_2d):
  X, y, X_test = data_2d
  gp = GP(kernels.rbf(length_scale=0.3), noise=1.0e-3)
  state = gp.fit(X, y)
  # X_test (4, 5, 2): treat as (n_test=4*5, n_x=2)? No — sample needs (*batch, n_test, n_x).
  # Use a flat (n_test, n_x) wrapped to (1, n_test, n_x) for one batch.
  X_t = X_test.reshape(20, 2)[None]  # (1, 20, 2)
  s = gp.sample(state, X_t, jax.random.PRNGKey(0), num_samples=4)
  assert s.shape == (4, 1, 20)


def test_sample_batch_shape(data_2d):
  X, y, _ = data_2d
  gp = GP(kernels.rbf(length_scale=0.3), noise=1.0e-3)
  state = gp.fit(X, y)
  # batch dims (3, 2), n_test=5, n_x=2
  X_test = jax.random.uniform(jax.random.PRNGKey(0), shape=(3, 2, 5, 2))
  s = gp.sample(state, X_test, jax.random.PRNGKey(1), num_samples=4)
  assert s.shape == (4, 3, 2, 5)

@pytest.mark.parametrize("N,B,n_test", [(256, 64, 32), (512, 32, 64)])
def test_bench_batched_triangular_solve(N, B, n_test, capsys):
  """Benchmark broadcasting L to (B, N, N) vs vmap over the per-batch solve."""
  key = jax.random.PRNGKey(0)
  k_L, k_K = jax.random.split(key)
  A = jax.random.normal(k_L, (N, N))
  L = jnp.linalg.cholesky(A @ A.T + N * jnp.eye(N))
  K_sx = jax.random.normal(k_K, (B, N, n_test))

  def via_broadcast(L, K_sx):
    L_b = jnp.broadcast_to(L, (K_sx.shape[0], *L.shape))
    return jax.scipy.linalg.solve_triangular(L_b, K_sx, lower=True)

  def via_vmap(L, K_sx):
    return jax.vmap(lambda k: jax.scipy.linalg.solve_triangular(L, k, lower=True))(K_sx)

  jb = jax.jit(via_broadcast)
  jv = jax.jit(via_vmap)

  out_b = jb(L, K_sx).block_until_ready()
  out_v = jv(L, K_sx).block_until_ready()
  assert jnp.allclose(out_b, out_v, atol=1.0e-5)

  def time_it(f, n_repeat=100):
    # warmup
    f(L, K_sx).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
      out = f(L, K_sx)
    out.block_until_ready()
    return (time.perf_counter() - t0) / n_repeat * 1e3  # ms

  t_b = time_it(jb)
  t_v = time_it(jv)
  with capsys.disabled():
    print(f"\n  N={N} B={B} n_test={n_test}: "
          f"broadcast {t_b:6.3f} ms  |  vmap {t_v:6.3f} ms  |  ratio {t_v/t_b:.2f}x")


def test_jit_compatible(data_2d):
  X, y, X_test = data_2d
  gp = GP(kernels.rbf(length_scale=0.3), noise=1.0e-4)

  @jax.jit
  def predict_jit(X, y, X_test):
    state = gp.fit(X, y)
    return gp.predict(state, X_test)

  mean, var = predict_jit(X, y, X_test)
  assert mean.shape == (4, 5)
