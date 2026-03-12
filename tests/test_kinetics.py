import os

import numpy as np
import jax
import jax.numpy as jnp

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import yaml

from doe.secret import enzyme

# <root> / tests / test_kinetics.py
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_kinetics(seed, plot_root):
  with open(os.path.join(root, 'config', 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

  bounds = config['parameters']
  rng = np.random.default_rng(seed)

  parameters = {
    name: rng.uniform(low=low, high=high, size=()).astype(np.float32)
    for name, (low, high) in bounds.items()
  }

  def rhs(_, A, A0, B0, E, temp):
    delta = A0 - A
    B = B0 - delta
    C = delta
    D = delta

    rate = enzyme.kinetics(A, B, C, D, E, temp, parameters)

    return -rate

  rhs = jax.jit(rhs)
  jac = jax.jit(jax.jacobian(rhs, argnums=1))

  T = 10.0

  ts_eval = np.linspace(0, T, num=128)

  A0_ = np.array([1.0], dtype=np.float32)
  B0_ = np.array([1.2], dtype=np.float32)
  E_ = np.array([1.0e-3], dtype=np.float32)

  temperatures = np.linspace(0.0, 100.0, num=51)

  ts = list()
  As = list()

  for i, temperature in enumerate(temperatures):
    trajectory = solve_ivp(
      rhs, t_span=(0.0, T), t_eval=ts_eval, y0=A0_,
      args=(A0_, B0_, E_, temperature), jac=jac
    )

    ts.append(trajectory.t)
    As.append(trajectory.y[0])

  fig = plt.figure(figsize=(9, 12))
  axes = fig.subplots(2, 1)

  As = np.stack(As, axis=0)
  im = axes[0].imshow(As, extent=(0.0, T, temperatures[0], temperatures[-1]), aspect=T / (temperatures[-1] - temperatures[0]))
  #for i, (t, A, temperature) in enumerate(zip(ts, As, temperatures)):
    #axes[0].plot(t, A, color=plt.cm.viridis(i / temperatures.shape[0]), label=f'T = {temperature:.2f} C')

  axes[0].set_xlabel('time')
  axes[0].set_ylabel('temperature, C')
  axes[0].set_title('substrate (time, temperature)')
  plt.colorbar(im, ax=axes[0])

  log_ratio = np.log(A0_[0]) - np.log(As)
  speeds = np.array([np.polyfit(ts_eval, lr, 1)[0] for lr in log_ratio])

  axes[1].plot(temperatures, speeds)
  axes[1].set_xlabel('temperature')
  axes[1].set_ylabel('reaction speed (approx)')

  fig.tight_layout()
  fig.savefig(os.path.join(plot_root, 'dynamics.png'))
  plt.close(fig)


