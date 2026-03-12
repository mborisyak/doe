import jax
import jax.numpy as jnp
import pytest
import os

from jax import config; config.update("jax_enable_x64", False); config.update('jax_platform_name', 'cpu')

@pytest.fixture(scope='function')
def plot_root(request):
  import os
  f = request.function
  here, _ = os.path.split(__file__)
  root = os.path.join(here, 'plots', f.__name__)
  os.makedirs(root, exist_ok=True)

  return root

def get_hashed_seed(name):
  import hashlib

  h = hashlib.sha256()
  h.update(bytes(name, encoding='utf-8'))
  digest = h.hexdigest()

  return int(digest[:8], 16)

@pytest.fixture(scope='function')
def seed(request):
  return get_hashed_seed(request.function.__name__)

@pytest.fixture(scope='function')
def rng(request):
  seed = get_hashed_seed(request.function.__name__)
  return jax.random.PRNGKey(seed)

@pytest.fixture(scope='function')
def no_jit():
  import jax
  with jax.disable_jit() as context:
    yield context

  pass