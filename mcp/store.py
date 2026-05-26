"""Re-export shim. The store is the pure Python API in the importable
``doe.store`` package (so agent code can load ``auxiliary`` arrays and compute on
them). This module keeps ``import store`` working from the flat ``mcp/`` path.
See STORAGE.md."""
from __future__ import annotations

from doe.store import (  # noqa: F401
    CREATABLE,
    DEFAULT_PREFIX,
    LINK_FIELDS,
    TYPE_DIR,
    Store,
    StoreError,
    view,
)
