### Reference model for validity region detection.
### Michaelis-Menten without denaturation and inhibition.

import jax
import jax.numpy as jnp

### A + B -> C + D via E

ZERO_CELSIUS =  273.15
REFERENCE_TEMPERATURE = 10.0 + ZERO_CELSIUS
INV_TEMPERATURE_SPAN = 1 / ZERO_CELSIUS - 1 / REFERENCE_TEMPERATURE

__all__ = [
  'kinetics'
]

def vant_hoff(T, log_K_0, Q10):
  """
  K_0 --- rate at 0C;
  Q100 - K_ref / K_0, K_ref - rate at the reference temperature, 100C;
  """
  delta = 1 / ZERO_CELSIUS - 1 / (T + ZERO_CELSIUS)
  return jnp.exp(
    log_K_0 + jnp.log(Q10) * delta / INV_TEMPERATURE_SPAN
  )

def kinetics(A, B, C, D, E, temperature, parameters):
  K_A = vant_hoff(temperature, parameters['log_K0_A'], parameters['Q10_A'])
  K_B = vant_hoff(temperature, parameters['log_K0_B'], parameters['Q10_B'])

  ### actually Arrhenius, but the expression is the same
  k_cat = vant_hoff(temperature, parameters['log_k0_cat'], parameters['Q10_cat'])

  rate = k_cat * E * A * B / (A + K_A) / (B + K_B)

  return rate
