"""DoE package.

Subpackages are imported **lazily** (PEP 562): ``import doe`` no longer eagerly pulls in
``common`` (sympy), ``doe`` (optax) and ``inference``. Attribute access like
``doe.common.CustomODESystem`` still works -- the submodule is imported on first use --
but importing a light submodule (``doe.gp``, ``doe.dataset``) stays light, which matters
for the store-free scripts launched as subprocesses (no need to pay for sympy/optax there).
"""
import importlib

_SUBPACKAGES = ("common", "inference", "doe", "gp", "utils")


def __getattr__(name):
    if name in _SUBPACKAGES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_SUBPACKAGES))
