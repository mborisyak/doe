"""MCP service over the filesystem store (``doe.store``). Returns the envelope
``{ok, references?, data?}`` / ``{ok:false, error}``; the store is append-only and
``store_create`` refuses to create data (batches). Over the wire, ``store_get``
returns the record with ``auxiliary`` values nulled (keys listed) unless ``select``
asks for one path -- the Python API returns full records."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from doe.store import CREATABLE, Store, StoreError, view


class StoreMcpService:
    def __init__(self, store: Optional[Store] = None) -> None:
        self.store = store or Store()

    @staticmethod
    def _ok(references: Optional[Dict[str, Any]] = None, data: Any = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {"ok": True}
        if references is not None:
            out["references"] = references
        if data is not None:
            out["data"] = data
        return out

    @staticmethod
    def _err(exc: StoreError) -> Dict[str, Any]:
        return {"ok": False, "error": {"code": exc.code, "message": exc.message, "details": exc.details}}

    def create(
        self,
        type_: str,
        name: str,
        record: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        if type_ == "batch":
            return self._err(StoreError(
                "forbidden",
                "data (batches) cannot be created directly; they come from the simulate / "
                "measurement path",
                {"type": type_},
            ))
        if type_ not in CREATABLE:
            return self._err(StoreError(
                "unknown_type", f"type {type_!r} is not creatable; choose from {list(CREATABLE)}", {"type": type_}
            ))
        body = dict(record or {})
        if description is not None:
            body["description"] = description
        try:
            group = self.store[type_]
            with group.lock(name or "-") as resolved:   # top-level: resolved == full name
                group[resolved] = body
            full = self.store.read(resolved)
        except StoreError as exc:
            return self._err(exc)
        return self._ok(self.store.references_of(full))

    def get(self, ref: str, select: Optional[str] = None) -> Dict[str, Any]:
        try:
            full = self.store.read(ref)
            data = view(full, select=select)
        except StoreError as exc:
            return self._err(exc)
        return self._ok(self.store.references_of(full), data)

    def list_records(self, type_: Optional[str] = None, descriptions: bool = True) -> Dict[str, Any]:
        types = [type_] if type_ is not None else list(self.store)
        rows: List[Dict[str, Any]] = []
        try:
            for one_type in types:
                for name in self.store.names(one_type):
                    record = self.store.read(name)
                    row = {"name": name, "type": one_type, "created_at": record.get("created_at", "")}
                    if descriptions:
                        row["description"] = record.get("description", "")
                    rows.append(row)
        except StoreError as exc:
            return self._err(exc)
        rows.sort(key=lambda entry: entry["created_at"])
        return self._ok(data=rows)
