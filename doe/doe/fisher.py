from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

__all__ = [
  'Fisher'
]

class FisherBase(object):
  EIGENVALUE_THRESHOLD = 1.0
  ITERATIONS = 2048

  def __init__(
      self, sampler, estimate, estimate_raw, regressor_parameters, regressor_state, optimizer,
      horizon: int, iterations: int, regularization: float | None=None, criterion: str='D'
  ):
    super().__init__(horizon=horizon)

    self.regressor_parameters = regressor_parameters
    self.regressor_state = regressor_state
    self.optimizer = optimizer

    @jax.jit
    def jacobian(controls, ts, latent, reg_parameters, reg_state):
      return jax.vmap(
        jax.vmap(
          jax.jacobian(estimate_raw, argnums=2),
          in_axes=(0, 0, None, None, None),
          out_axes=0
        ),
        in_axes=(0, 0, 0, None, None),
        out_axes=0
      )(controls, ts, latent, reg_parameters, reg_state)

    @jax.jit
    def information(controls, ts, latent, reg_parameters, reg_state, mask=None):
      J = jacobian(controls, ts, latent, reg_parameters, reg_state)
      I = jax.lax.dot_general(J, J, (([2, ], [2, ]), ([0, 1], [0, 1])))
      if mask is None:
        return jnp.sum(I, axis=1)
      else:
        return jnp.sum(I * mask[:, :, None, None], axis=(1, ))

    if criterion == 'D':
      @jax.jit
      def criterion(I):
        svdvals = jnp.linalg.svdvals(I)
        return -2 * jnp.sum(jnp.log(svdvals))
    elif criterion == 'A':
      @jax.jit
      def criterion(I):
        svdvals = jnp.linalg.svdvals(I)
        eignvals_inv = jnp.square(1 / svdvals)
        return jnp.sum(eignvals_inv)
    else:
      raise ValueError('criterion must be either A or D.')

    @jax.jit
    def loss(controls, ts, latent, I0, reg_parameters, reg_state):
      n_b, n_e, *_ = controls.shape
      I = information(controls, ts, latent, reg_parameters, reg_state)
      l = criterion(I + I0)

      if regularization is None:
        return l
      else:
        reg = jnp.sum(jnp.square(controls))
        return l + regularization * reg

    @jax.jit
    def grad_loss(controls, ts, latent, I0, reg_parameters, reg_state):
      return jax.value_and_grad(loss, argnums=0)(controls, ts, latent, I0, reg_parameters, reg_state)

    def loss2(controls, ts, latent, I0, reg_parameters, reg_state):
      *_, n_e, n_t = ts.shape
      controls = controls.reshape((1, n_e, -1))
      l = loss(controls, ts, latent, I0, reg_parameters, reg_state)
      return np.array(l)

    def grad_loss2(controls, ts, latent, I0, reg_parameters, reg_states):
      *_, n_e, n_t = ts.shape
      controls = controls.reshape((1, n_e, -1))
      l, grad = grad_loss(controls, ts, latent, I0, reg_parameters, reg_states)
      return np.array(l, dtype=np.float64), np.array(grad, dtype=np.float64).ravel()

    @jax.jit
    def step(controls, ts, latent, opt_state, I0, reg_parameters, reg_state):
      l, grad = grad_loss(controls, ts, latent, I0, reg_parameters, reg_state)
      updates, opt_state_updated = self.optimizer.update(
        grad, opt_state, controls,
        value=l, grad=grad, value_fn=loss, ts=ts, latent=latent,
        reg_parameters=reg_parameters, reg_state=reg_state
      )
      controls_updated = optax.apply_updates(controls, updates)

      return l, controls_updated, opt_state_updated

    @partial(jax.jit, static_argnums=(1, 2))
    def sample_time(key, n_b, n_e):
      ts = sampler.hyper.timestamps.sample(key, sampler.hyper_condition.timestamps, shape=(n_b, n_e))
      return sampler.hyper.timestamps.encode(ts, sampler.hyper_condition.timestamps)

    @jax.jit
    def propose(key, initial, latent, I0, reg_parameters, reg_state):
      n_b, n_e, _ = initial.shape

      def step(carry, _):
        k, proposal, opt_state = carry

        k, k_ts = jax.random.split(k, num=2)
        ts = self.sample_time(k_ts, n_b, n_e)
        l, proposal_updated, opt_state_updated = self.step(
          proposal, ts, latent, opt_state, I0, reg_parameters, reg_state
        )
        proposal_updated = jnp.clip(proposal_updated, -3, 3)

        return (k, proposal_updated, opt_state_updated), l

      (_, proposed, _), losses = jax.lax.scan(
        step, init=(key, initial, self.optimizer.init(initial)), length=iterations
      )
      return losses, proposed

    def propose_iter(key, initial, latent, reg_parameters, reg_states):
      n_b, n_e, _ = initial.shape

      key, k_ts = jax.random.split(key, num=2)
      ts = self.sample_time(k_ts, n_b, n_e)

      proposal = initial
      opt_state = self.optimizer.init(initial)
      losses = np.zeros(shape=(iterations, ))
      for i in range(iterations):
        losses[i], proposal, opt_state = step(
          proposal, ts, latent, opt_state, reg_parameters, reg_states
        )
        assert np.all(np.isfinite(ts))
        assert np.all(np.isfinite(proposal)), f'iter {i}'

      return losses, proposal

    def propose_scipy(key, initial, latent, I0, reg_parameters, reg_state):
      from scipy import optimize as opt
      n_b, n_e, n_c = initial.shape

      proposed = np.ndarray(shape=initial.shape)
      losses = np.ndarray(shape=(n_b, ))
      for i in range(n_b):
        key, key_ts = jax.random.split(key, num=2)
        ts = self.sample_time(key_ts, 1, n_e)
        sol = opt.minimize(
          grad_loss2, np.clip(np.array(initial[i:(i + 1)]).ravel(), -3, 3),
          args=(ts, latent[i:(i + 1)], I0, reg_parameters, reg_state), method='L-BFGS-B',
          bounds=[(-3, 3) for _ in range(n_e * n_c)], jac=True, options={'max_iter': iterations}
        )
        proposed[i] = sol.x.reshape(initial.shape[1:])
        if not sol.success:
          print(sol)
        losses[i] = sol.fun

      return losses, proposed

    self._estimate = lambda controls, ts, latent: estimate(
      controls, ts, latent, reg_parameters=self.regressor_parameters, reg_state=self.regressor_state
    )

    self.jacobian = lambda controls, ts, latent: jacobian(
      controls, ts, latent, reg_parameters=self.regressor_parameters, reg_state=self.regressor_state
    )

    self.information = lambda controls, ts, latent, mask=None: information(
      controls, ts, latent, reg_parameters=self.regressor_parameters, reg_state=self.regressor_state, mask=mask
    )
    self.loss = lambda controls, ts, latent: loss(
      controls, ts, latent, reg_parameters=self.regressor_parameters, reg_state=self.regressor_state
    )
    self.step = step
    self.sample_time = sample_time
    # self._propose = lambda key, initial, latent: propose(
    #   key, initial, latent, self.inference_parameters, self.inference_state
    # )
    self._propose = lambda key, initial, latent, I0: propose(
      key, initial, latent, I0, reg_parameters=self.regressor_parameters, reg_state=self.regressor_state
    )

    self.loss2 = lambda controls, ts, latent, I0: loss2(
      controls, ts, latent, I0, self.regressor_parameters, self.regressor_state
    )

  def infer(self, controls, timestamps, observations, mask=None):
    raise NotImplementedError()

  def propose(self, key, n, controls, timestamps, observations, mask: jax.Array | None=None):
    latent = self.infer(controls, timestamps, observations, mask=mask)
    I0 = self.information(controls, timestamps, latent, mask=mask)
    n_b, m1, m2 = I0.shape
    assert m1 == m2, f'{I0.shape}'
    I0 = I0 + self.EIGENVALUE_THRESHOLD * jnp.eye(m1)

    assert jnp.all(jnp.isfinite(latent))
    n_b, _, n_c = controls.shape
    key, key_initial = jax.random.split(key, num=2)

    initial = jax.random.normal(key_initial, shape=(n_b, n, n_c))
    return self._propose(key, initial=initial, latent=latent, I0=I0)

  def estimate(
      self, key: jax.Array, n: int,
      controls_history: jax.Array, timestamps_history: jax.Array, observations_history: jax.Array,
      controls: jax.Array, timestamps: jax.Array, mask: jax.Array | None=None
  ) -> jax.Array:
    latent = self.infer(
      controls_history, timestamps_history, observations_history,
      mask=mask
    )
    estimations = self._estimate(controls, timestamps, latent)
    n_b, n_e, n_t, n_obs = estimations.shape

    return jnp.broadcast_to(estimations[:, None, :, :, :], shape=(n_b, n, n_e, n_t, n_obs))

class Fisher(FisherBase):
  def __init__(
      self, inference, regressor, horizon: int, iterations: int, optimizer,
      regularization: float | None=None, criterion='D'
  ):
    _, self.inference_def, self.inference_parameters, self.inference_state = io.restore_model(
      inference, checkpoint=inference['checkpoint']
    )

    sampler, self.regressor_def, self.regressor_parameters, self.regressor_state = io.restore_model(
      regressor, checkpoint=regressor['checkpoint']
    )

    @jax.jit
    def infer(controls, ts, obs, mask, inf_parameters, inf_state):
      model = nnx.merge(self.inference_def, inf_parameters, inf_state)
      return model.infer(controls, ts, obs, mask, deterministic=True)

    @jax.jit
    def estimate(controls, ts, latent, reg_parameters, reg_state):
      model = nnx.merge(self.regressor_def, reg_parameters, reg_state)
      return model(controls, latent, ts, deterministic=True)

    @jax.jit
    def estimate_raw(controls, ts, latent, reg_parameters, reg_state):
      model = nnx.merge(self.regressor_def, reg_parameters, reg_state)
      X = jnp.concatenate([controls, latent, ts], axis=-1)
      return model.regressor(X, deterministic=True)

    self._infer = infer

    super().__init__(
      sampler, estimate, estimate_raw, self.regressor_parameters, self.regressor_state,
      optimizer=optimizer_from_config(optimizer),
      horizon=horizon, iterations=iterations, regularization=regularization,
      criterion=criterion
    )

  def infer(self, controls, timestamps, observations, mask=None):
    return self._infer(controls, timestamps, observations, mask, self.inference_parameters, self.inference_state)