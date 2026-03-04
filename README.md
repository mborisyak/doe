# doe — Design of Experiments for ODE-based models

A JAX library for Fisher-information-based optimal experimental design with ODE kinetic models. Given a parametric ODE model and existing measurements, `doe` estimates model parameters and proposes new experimental conditions that maximally reduce parameter uncertainty.

## Installation

```bash
pip install .
```

**Dependencies:** `jax`, `diffrax`, `optax`, `numpy`

## Usage

**1. Define your kinetic model** by subclassing `ODEModel`:

```python
from collections import namedtuple
import jax.numpy as jnp
import doe

Conditions = doe.common.Conditions          # namedtuple: A, B, E, temperature
Parameters = namedtuple('Parameters', ['q', 'K_A', 'K_B'])

class SimpleEnzyme(doe.common.ODEModel[Parameters]):
    def initial_state(self, conditions):
        Ac, Bc, Ec = doe.common.get_initial_concentrations(conditions)
        return jnp.stack([Ac, Bc, Ec], axis=-1)

    def rhs(self, state, conditions, parameters):
        A, B, E = state[..., 0], state[..., 1], state[..., 2]
        rate = E * parameters.q * A / (parameters.K_A + A) * B / (B + parameters.K_B)
        return jnp.stack([-rate, -rate, jnp.zeros_like(rate)], axis=-1)

    def observe(self, state):
        return state[..., 0]   # observe substrate A
```

**2. Fit parameters** from existing experimental data:

```python
# conditions_data: dict label -> {A, B, E, temperature}
# measurements_data: dict label -> {timestamps, measurements}
parameter_ranges = Parameters(q=[1e2, 2e3], K_A=[1e-2, 2.0], K_B=[1e-2, 2.0])

_, parameters, _ = doe.inference.maximum_likelihood_estimate(
    model, conditions_data, measurements_data, parameter_ranges, iterations=512
)
```

**3. Propose new informative experiments** using Fisher D-optimality:

```python
import jax
import jax.numpy as jnp

condition_ranges = Conditions(A=[0.1, 5.0], B=[0.1, 5.0], E=[0.1, 5.0], temperature=[0.0, 100.0])

fisher = doe.doe.Fisher(
    model, condition_ranges, parameter_ranges,
    timestamps=jnp.linspace(1.0, 29.0, 9),
    iterations=64,
)

# encode past experiments
controls  = jnp.stack([fisher.encode_conditions(c) for c in past_conditions])
timestamps = jnp.array([exp['timestamps'] for exp in past_measurements])

# propose 3 new experiments
key = jax.random.PRNGKey(0)
loss_trace, proposed = fisher.propose(key, n=3, controls=controls, timestamps=timestamps, parameters=parameters)

# decode back to physical units
for enc in proposed:
    print(fisher.decode_conditions(enc))
```

## API overview

| Symbol | Description |
|---|---|
| `ODEModel` | Abstract base class for ODE kinetic models (`initial_state`, `rhs`, `observe`) |
| `inference.maximum_likelihood_estimate` | Gradient-based MLE (LBFGS) for parameter estimation |
| `doe.Fisher` | Proposes new conditions maximising Fisher information (D- or A-optimality) |
| `doe.ArmijoLineSearch` | Backtracking line search used internally by `Fisher` |

## License

MIT