import os

import numpy as np
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt


def euler_mm(q, K, S0, dt, n_steps):
  def step(S, _):
    dS = -q * S / (K + S)
    S_next = S + dt * dS
    return S_next, S_next

  _, traj = jax.lax.scan(step, S0, None, length=n_steps)
  return jnp.concatenate([jnp.array([S0]), traj])


def test_mm_posterior(seed, plot_root):
  rng = np.random.default_rng(seed)

  q_true = 1.2
  K_true = 2.0
  S0 = 1.0
  T = 5.0
  n_steps = 2000
  dt = T / n_steps

  ts = jnp.linspace(0.0, T, n_steps + 1)

  S_true = euler_mm(q_true, K_true, S0, dt, n_steps)

  n_obs = 7
  obs_idx = np.linspace(0, n_steps, n_obs).astype(int)
  ts_obs = np.asarray(ts)[obs_idx]
  sigma = 0.1
  S_obs = np.asarray(S_true)[obs_idx] + rng.normal(scale=sigma, size=n_obs).astype(np.float32)

  n_grid = 80
  qs = jnp.linspace(0.0, 2 * q_true, n_grid)
  Ks = jnp.linspace(0.01, 2 * K_true, n_grid)

  def log_lik(q, K):
    S = euler_mm(q, K, S0, dt, n_steps)
    resid = S[obs_idx] - jnp.asarray(S_obs)
    return -0.5 * jnp.sum(resid ** 2) / sigma ** 2

  log_lik_vec = jax.jit(jax.vmap(jax.vmap(log_lik, in_axes=(None, 0)), in_axes=(0, None)))

  log_post = log_lik_vec(qs, Ks)
  log_post = log_post - jnp.max(log_post)
  post = jnp.exp(log_post)
  post = post / jnp.sum(post)

  fig = plt.figure(figsize=(12, 5))
  ax_traj, ax_post = fig.subplots(1, 2)

  ax_traj.plot(np.asarray(ts), np.asarray(S_true), color='C0', label='ground truth')
  ax_traj.errorbar(ts_obs, S_obs, yerr=sigma, fmt='o', color='k', label='observations', ms=4)
  ax_traj.set_xlabel('t')
  ax_traj.set_ylabel('S(t)')
  ax_traj.set_title(f'q*={q_true}, K*={K_true}')
  ax_traj.legend()

  im = ax_post.imshow(
    np.asarray(post).T,
    extent=(float(qs[0]), float(qs[-1]), float(Ks[0]), float(Ks[-1])),
    origin='lower', aspect='auto', cmap='viridis'
  )
  ax_post.scatter([q_true], [K_true], marker='x', color='red', s=80, label='truth')
  ax_post.set_xlabel('q')
  ax_post.set_ylabel('K')
  ax_post.set_title('posterior (grid)')
  ax_post.legend()
  plt.colorbar(im, ax=ax_post)

  fig.tight_layout()
  fig.savefig(os.path.join(plot_root, 'posterior.png'))
  plt.close(fig)

  q_grid, K_grid = jnp.meshgrid(qs, Ks, indexing='ij')
  q_mean = float(jnp.sum(post * q_grid))
  K_mean = float(jnp.sum(post * K_grid))
  assert abs(q_mean - q_true) < 0.2
  assert abs(K_mean - K_true) < 0.2