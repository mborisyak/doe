"""Shared base for the by-name resolvers.

A resolver turns store names into the inline payload a tool needs, runs the tool, and
commits the result back into the store. The compute resolver (analysis/design) and the
simulate resolver (the experiment surrogate / data source) are separate -- they don't
share a tool surface -- but they share these small store helpers.
"""
from __future__ import annotations

from typing import Any, Dict

from doe.store import Store, StoreError


class _Abort(Exception):
    """Raised inside a reservation to roll it back (free the name) and surface a
    compute-error envelope unchanged."""

    def __init__(self, envelope: Dict[str, Any]) -> None:
        self.envelope = envelope


def _error(exc: StoreError) -> Dict[str, Any]:
    return {"ok": False, "error": {"code": exc.code, "message": exc.message, "details": exc.details}}


class StoreResolver:
    """Holds the store and the helpers every resolver needs: type-checked ref reads,
    call-name minting, and writing the thin log entry (the ``logs`` collection)."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def _typed(self, ref: str, expected: str) -> Dict[str, Any]:
        record = self.store.read(ref)  # raises StoreError(not_found)
        if record.get("type") != expected:
            raise StoreError(
                "wrong_type",
                f"{ref!r} is a {record.get('type')!r}, expected {expected!r}",
                {"ref": ref, "expected": expected},
            )
        return record

    @staticmethod
    def _tool_result(tool: str, arguments: Dict[str, Any], inputs: Dict[str, Any],
                     outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Build the log record, splitting a call into ``arguments`` (non-ref params),
        ``inputs`` (the refs it consumed), and ``outputs`` (the refs it produced); ``None``
        values are dropped from arguments/inputs. The caller reserves the call name *up front*
        (with the output, before the compute) and commits this under the held reservation."""
        drop = lambda d: {k: v for k, v in d.items() if v is not None}  # noqa: E731
        return {"tool": tool, "status": "ok",
                "arguments": drop(arguments), "inputs": drop(inputs), "outputs": outputs}
