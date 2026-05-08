"""Benchmarks: compile time, scaling, float32 Cholesky stability."""
import time
import jax
import jax.numpy as jnp
from doe.gp import GP, kernels


def bench_compile_and_run(N, M, dtype=jnp.float64, repeats=5):
  key = jax.random.PRNGKey(0)
  X = jax.random.uniform(key, (N, 1), dtype=dtype) * 4.0
  y = jnp.sin(X).flatten()
  X_test = jnp.linspace(0.0, 4.0, M, dtype=dtype).reshape(-1, 1)

  gp = GP(kernels.rbf(length_scale=1.0), noise=1.0e-4)

  @jax.jit
  def go(X, y, X_test):
    state = gp.fit(X, y)
    mean, var = gp.predict(state, X_test)
    return mean, var

  t0 = time.perf_counter()
  m, v = go(X, y, X_test)
  m.block_until_ready()
  t_compile = time.perf_counter() - t0

  ts = []
  for _ in range(repeats):
    t0 = time.perf_counter()
    m, v = go(X, y, X_test)
    m.block_until_ready()
    ts.append(time.perf_counter() - t0)
  t_run = min(ts)
  return t_compile, t_run


def bench_float32_cholesky(N, dtype):
  key = jax.random.PRNGKey(0)
  X = jax.random.uniform(key, (N, 1), dtype=dtype) * 4.0
  y = jnp.sin(X).flatten()
  gp = GP(kernels.rbf(length_scale=1.0), noise=1.0e-6)
  try:
    state = gp.fit(X, y)
    state.L.block_until_ready()
    finite = bool(jnp.all(jnp.isfinite(state.L)))
    diag_min = float(jnp.min(jnp.diag(state.L)))
    return finite, diag_min
  except Exception as e:
    return False, str(e)


if __name__ == "__main__":
  # Force float64 by default — JAX defaults to float32
  jax.config.update("jax_enable_x64", True)

  print("== compile/run time, predict diag (float64) ==")
  print(f"{'N':>6} {'M':>6} {'compile_s':>11} {'run_s':>11} {'mem_K_MB':>10}")
  for N in [64, 256, 1024, 2048, 4096]:
    M = 200
    t_c, t_r = bench_compile_and_run(N, M, dtype=jnp.float64)
    mem_mb = (N * N * 8) / 1e6  # K matrix, float64
    print(f"{N:>6} {M:>6} {t_c:>11.4f} {t_r:>11.4f} {mem_mb:>10.2f}")

  print("\n== float32 Cholesky stability, RBF, length_scale=1, noise=1e-6 ==")
  print(f"{'N':>6} {'finite':>8} {'min_diag(L) or err':>40}")
  for N in [64, 128, 256, 512, 1024, 2048]:
    finite, info = bench_float32_cholesky(N, dtype=jnp.float32)
    print(f"{N:>6} {str(finite):>8} {str(info)[:40]:>40}")

  print("\n== float64 Cholesky stability, same ==")
  for N in [64, 256, 1024, 2048, 4096]:
    finite, info = bench_float32_cholesky(N, dtype=jnp.float64)
    print(f"{N:>6} {str(finite):>8} {str(info)[:40]:>40}")