from typing import TypeVar, Generic
from collections import namedtuple

import jax
import jax.numpy as jnp

import diffrax

SOLUTIONS = {
  'A': 3.0,
  'B': 3.0,
  'E': 3.0e-3,
}

NOISE_SIGMA = 0.025

TIME_HORIZON = 30
MEASUREMENT_TIMESTAMPS = jnp.linspace(0, 30, num=11)[1:-1]

Conditions = namedtuple('Conditions', ['A', 'B', 'E', 'temperature'])

### should be a named tuple
Parameters = TypeVar('Parameters')

INITIAL_DT = 1.0e-3

def get_initial_concentrations(condition):
  A, B, E, _ = condition
  V_total = A + B + E

  Ac = A * SOLUTIONS['A'] / V_total
  Bc = B * SOLUTIONS['B'] / V_total
  Ec = E * SOLUTIONS['E'] / V_total

  return Ac, Bc, Ec

def euler(rhs, initial, n, dt):
  def step(state, _):
    ds_dt = rhs(state)
    updated = state + dt * ds_dt
    updated = jnp.clip(updated, 0)
    return updated, updated

  _, trajectory = jax.lax.scan(
    step, init=initial, length=n
  )

  return trajectory

class ODEModel(Generic[Parameters]):
  def __init__(self, n=8192, tol=1.0e-3, device=None):
    # def rhs(_, state, args):
    #   condition, parameters = args
    #   return self.rhs(state, condition, parameters)

    # self.term = diffrax.ODETerm(rhs)
    # self.solver = diffrax.Tsit5()

    self.n = n
    self.tol = tol

    self.device = device

    self.solve_euler = jax.jit(self._solve_euler, device=device)
    self.observable_jac_euler = jax.jit(self._observable_jac_euler, device=device)
    self.accumulated_value_gradient_euler = jax.jit(
      self._accumulated_value_gradient_euler,
      static_argnames=('l', ), device=device
    )

  def initial_state(self, conditions: Conditions) -> jax.Array:
    """
    Returns initial state of the system given conditions.
    Conditions typically contain initial states of the

    :param conditions: initial conditions of the experiment
    :return: initial state of the system.
    """
    raise NotImplementedError()

  def rhs(self, state: jax.Array, conditions: Conditions, parameters: Parameters) -> jax.Array:
    raise NotImplementedError()

  def observables(self, state: jax.Array, parameters: Parameters) -> jax.Array:
    raise NotImplementedError()

  def parameter_ranges(self) -> Parameters:
    raise NotImplementedError()

  def trajectory_euler(self, conditions, timestamps, parameters):
    T = jnp.max(timestamps)
    dt = T / self.n
    initial = self.initial_state(conditions)
    rhs = lambda state: self.rhs(state, conditions, parameters)
    trajectory = euler(rhs, initial=initial, n=self.n, dt=dt)
    control = euler(rhs, initial=initial, n=2 * self.n, dt=0.5 * dt)
    control = control[1::2]

    eps = jnp.finfo(control.dtype).eps
    norm = jnp.minimum(jnp.max(jnp.abs(trajectory)), jnp.max(jnp.abs(control)))
    error = jnp.max(jnp.abs(trajectory - control)) / (norm + eps)

    index = jnp.clip(jnp.ceil(timestamps / dt).astype(int), 0, self.n - 1)

    return trajectory[index], error

  def _solve_euler(self, conditions, timestamps, parameters):
    states, error = self.trajectory_euler(conditions, timestamps, parameters)
    return self.observables(states, parameters), error

  def solve_scipy(self, conditions, timestamps, parameters):
    from scipy.integrate import solve_ivp
    initial = self.initial_state(conditions)

    _rhs = jax.jit(lambda _, state: self.rhs(state, conditions, parameters), device=self.device)
    _jac = jax.jit(jax.jacobian(_rhs, argnums=1), device=self.device)

    T = jnp.max(timestamps)

    trajectory = solve_ivp(
      _rhs, t_span=(0.0, T), t_eval=timestamps, y0=initial,
      jac=_jac, method='LSODA'
    )

    return self.observables(trajectory.y.T, parameters)

  def solve(self, conditions, timestamps, parameters):
    trajectory, error = self.solve_euler(conditions, timestamps, parameters)

    if error > self.tol:
      return self.solve_scipy(conditions, timestamps, parameters)
    else:
      return trajectory

  def solve_jac_scipy(self, conditions, timestamps, parameters, device=None, dense_params=None):
    from scipy.integrate import solve_ivp
    initial = self.initial_state(conditions)

    _rhs = jax.jit(lambda _, state: self.rhs(state, conditions, parameters), device=device)
    def _dense_rhs(t, state, dense_parameters):
      params = type(parameters)(*(
        dense_parameters[..., i] for i, _ in enumerate(parameters)
      ))
      return self.rhs(state, conditions, params)

    _jac_wrt_state = jax.jit(jax.jacobian(_rhs, argnums=1), device=device)
    _jac_wrt_params = jax.jit(jax.jacobian(_dense_rhs, argnums=2), device=device)

    n = initial.shape[0]
    m = len(parameters)

    if dense_params is None:
      dense_params = jnp.stack(parameters, axis=-1)

    def _augmented(t, extended, dense_p):
      state = extended[:n]
      S = extended[n:].reshape(n, m)

      # Original RHS
      ds_dt = _rhs(t, state)

      # (n, n)
      Jy = _jac_wrt_state(t, state)
      # (n, m)
      Jp = _jac_wrt_params(t, state, dense_p)

      # Sensitivity equation
      dSdt = Jy @ S + Jp

      # Flatten and concatenate
      return jnp.concatenate([ds_dt, dSdt.reshape(-1)])

    _augmented = jax.jit(_augmented, device=device)
    _jac_augmented = jax.jit(jax.jacobian(_augmented, argnums=1), device=device)

    T = jnp.max(timestamps)

    initial_extended = jnp.concatenate([initial, jnp.zeros(n * m, dtype=initial.dtype)], axis=-1)
    solution = solve_ivp(
      _augmented, t_span=(0.0, T), t_eval=timestamps, y0=initial_extended,
      args=(dense_params, ),
      _jac=_jac_augmented, method='LSODA'
    )
    trajectory_extended = solution.y.T
    trajectory = trajectory_extended[:, :n]
    sensitivity = jnp.reshape(trajectory_extended[:, n:], shape=(len(timestamps), n, m))

    return trajectory, sensitivity

  def observable_jac_scipy(self, conditions, timestamps, parameters):
    dense_parameters = jnp.stack(parameters, axis=-1)
    trajectory, S = self.solve_jac_scipy(conditions, timestamps, parameters, device=self.device, dense_params=dense_parameters)
    # S: (T, n, m) — d(state) / d(params)

    def get_jac(traj, dense_ps):
      def obs_fn(y, dp):
        params = type(parameters)(*(
          dp[..., i] for i, _ in enumerate(parameters)
        ))
        value = self.observables(y, params)
        return value, value

      (_Jy, _Jp), _obs = jax.vmap(
        jax.jacobian(obs_fn, argnums=(0, 1), has_aux=True), in_axes=(0, None)
      )(
        traj, dense_ps
      )
      return _obs, _Jy, _Jp

    get_jac = jax.jit(get_jac, device=self.device)

    # (T, k, n)
    obs, Jy, Jp  = get_jac(trajectory, dense_parameters)

    # Jy is (T, k, n) for vector obs or (T, n) for scalar obs; S is (T, n, m).
    # Plain `Jy @ S` mis-batches when Jy is 2-D, so vmap the per-timestep matmul.
    return obs, jax.vmap(jnp.matmul)(Jy, S) + Jp

  def _observable_jac_euler(self, conditions, timestamps, parameters):
    def _dense_observe(dense_parameters):
      params = type(parameters)(*(
        dense_parameters[..., i] for i, _ in enumerate(parameters)
      ))
      trajectory, error = self._solve_euler(conditions, timestamps, params)
      return trajectory, error

    dense_parameters = jnp.stack(parameters, axis=-1)
    return jax.jacobian(_dense_observe, has_aux=True)(dense_parameters)

  def observable_jac(self, conditions, timestamps, parameters):
    J, error = self.observable_jac_euler(conditions, timestamps, parameters)

    if error > self.tol:
      return self.observable_jac_scipy(conditions, timestamps, parameters)
    else:
      return J

  def value_gradient_scipy(self, l, conditions, timestamps, parameters, arguments=None):
    trajectory, S = self.observable_jac_scipy(conditions, timestamps, parameters)

    def _value_gradient(S, ys, args):
      loss, cotangent = jax.value_and_grad(l)(ys, args)
      d_grad = jnp.tensordot(cotangent, S, axes=cotangent.ndim)
      return loss, d_grad

    _value_gradient = jax.jit(_value_gradient, device=self.device)

    value, dense_grad = _value_gradient(S, trajectory, arguments)
    return value, type(parameters)(*(dense_grad[..., i] for i, _ in enumerate(parameters)))

  def value_gradient_euler(self, l, conditions, timestamps, parameters, arguments=None):
    dense_params = jnp.stack(parameters, axis=-1)

    def _loss(dense_p):
      params = type(parameters)(*(dense_p[..., i] for i, _ in enumerate(parameters)))
      obs, error = self._solve_euler(conditions, timestamps, params)
      return l(obs, arguments), error

    (value, error), dense_grad = jax.jit(
      jax.value_and_grad(_loss, has_aux=True), device=self.device
    )(dense_params)
    return value, type(parameters)(*(dense_grad[..., i] for i, _ in enumerate(parameters))), error

  def _index_arguments(self, arguments, i):
    if hasattr(arguments, '_fields'):
      return type(arguments)(*(field[i] for field in arguments))
    return arguments[i]

  def accumulated_value_gradient_scipy(self, l, conditions, timestamps, parameters, arguments=None):
    n = timestamps.shape[0]

    values = jnp.zeros(shape=(n, ))
    dense_grad = jnp.zeros(shape=(n, len(parameters)))

    for i in range(n):
      cond_i = type(conditions)(*(field[i] for field in conditions))
      if arguments is None:
        args_i = None
      else:
        args_i = jax.tree.map(lambda x: x[i], arguments)

      value, grad = self.value_gradient_scipy(l, cond_i, timestamps[i], parameters, arguments=args_i)
      values = values.at[i].set(value)
      dense_grad = dense_grad.at[i].set(jnp.stack(grad, axis=-1))

    return values, dense_grad

  def _accumulated_value_gradient_euler(self, l, conditions, timestamps, parameters, arguments=None):
    def loss_fn(params, cs, ts, args):
      observables, err = self.solve_euler(cs, ts, params)
      return l(observables, args), err

    grad = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)

    if arguments is None:
      (values, errors), grads = jax.vmap(grad, in_axes=(None, 0, 0, None))(parameters, conditions, timestamps, None)
    else:
      (values, errors), grads = jax.vmap(grad, in_axes=(None, 0, 0, 0))(parameters, conditions, timestamps, arguments)

    return values, grads, jnp.max(errors)

  def accumulated_value_gradient(self, l, conditions, timestamps, parameters, arguments=None, ):
    value, grad, error = self.accumulated_value_gradient_euler(l, conditions, timestamps, parameters, arguments=arguments)
    if error > self.tol:
      return self.accumulated_value_gradient_scipy(l, conditions, timestamps, parameters, arguments=arguments)
    return value, grad

