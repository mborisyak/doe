"""Named JSON store for the DoE tools (see STORAGE.md).

Looks like a dict, backed by the filesystem -- nothing is held in memory. Records
live one-per-file under a directory per collection; the filesystem is the source of
truth. The store is **append-only**: records are never deleted or overwritten.

    store["model"]["model-x"]                  # -> full record dict (reads the file)
    store["fitted_model"]["model-x"].keys()    # -> ["v1", "v2", ...]  (lists the fits)
    store["model"]["model-z"] = {...}          # create (also: store["model"].create(...))
    with store["model"].lock("-") as name:     # reserve a (minted) name; marker cleared on exit
        ...                                     # long work, lock-free
        store["model"][name] = {...}           # write the result
    with store["fitted_model"]["model-x"].lock("v5"): ...   # reserve a specific name

A ``/`` in a name is a path separator on disk; ``store["fitted_model"]["model-x"]``
is the sub-namespace ``fitted_models/model-x/`` whose keys are the versions. Reads
return the **full** record (auxiliary arrays included) -- the wire-only summarised
view lives in the MCP layer.

Concurrency: a single ``fcntl.flock`` (our own ``_flock``, fresh fd per acquire ->
serialises threads and processes) is held briefly around every name mint, reserve,
and write. A ``<name>.lock`` marker holds a name across a long ``lock`` body without
keeping the global lock; minting skips marked names.
"""
from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

TYPE_DIR: Dict[str, str] = {
    "model": "models",
    "fitted_model": "fitted_models",
    "design": "designs",
    "batch": "batches",
    "tool_result": "tool_results",
}

# default ``-`` mint prefix for a top-level collection group (None -> minting there
# needs a name; e.g. fitted_models mint per-parent under the version namespace).
DEFAULT_PREFIX: Dict[str, Optional[str]] = {
    "model": "model-",
    "design": "design-",
    "batch": "batch-",
    "fitted_model": None,
    "tool_result": "result-",   # agents may log free-form tool_results; `-` mints result-N
}

# collections whose top level is a *namespace* of sub-groups, with the mint prefix
# used inside each sub-group.
CHILD_PREFIX: Dict[str, str] = {"fitted_model": "v"}

# types the MCP ``store_create`` may author: anything except data (batches).
CREATABLE = ("model", "fitted_model", "design", "tool_result")

# minimal required keys validated on write, per type. A tool_result only needs a
# `tool` name (any string) -- the rest of its content is whatever the agent wants.
REQUIRED: Dict[str, tuple] = {
    "model": ("spec",),
    "fitted_model": ("model", "parameters"),
    "design": ("experiments",),
    "batch": ("experiments",),
    "tool_result": ("tool",),
}

# generic structural links surfaced in a record's ``references`` block.
LINK_FIELDS = ("model", "fitted_model", "design")


class StoreError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = details or {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


@contextmanager
def _flock(path: str) -> Iterator[None]:
    """Our own store-wide lock: an exclusive whole-file advisory lock via
    ``fcntl.flock``, fresh fd per call so it serialises threads and processes alike.
    POSIX only."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _select_path(record: Dict[str, Any], dotted: str) -> Any:
    node: Any = record
    for seg in dotted.split("."):
        if not isinstance(node, dict) or seg not in node:
            raise StoreError(
                "not_found",
                f"path {dotted!r} does not exist in record",
                {"select": dotted, "missing_segment": seg},
            )
        node = node[seg]
    return node


def view(record: Dict[str, Any], *, select: Optional[str] = None) -> Any:
    """MCP/wire view: with ``select`` return that value in full; otherwise the record
    with ``auxiliary`` reduced to its top-level keys (values nulled). The Python dict
    API returns the full record and never calls this."""
    if select is not None:
        return _select_path(record, select)
    out = dict(record)
    aux = out.get("auxiliary")
    if isinstance(aux, dict):
        out["auxiliary"] = {key: None for key in aux}
    return out


class _Group(Mapping):
    """Filesystem-backed dict view of one collection directory (or sub-namespace)."""

    def __init__(
        self,
        store: "Store",
        type_: str,
        directory: Path,
        *,
        prefix: Optional[str],
        is_namespace: bool = False,
        child_prefix: Optional[str] = None,
    ) -> None:
        self._store = store
        self._type = type_
        self._dir = directory
        self._prefix = prefix
        self._is_namespace = is_namespace
        self._child_prefix = child_prefix

    def __getitem__(self, name: str):
        leaf = self._dir / f"{name}.json"
        if leaf.is_file():
            return _read_json(leaf)
        sub = self._dir / name
        if self._is_namespace or sub.is_dir():
            return _Group(self._store, self._type, sub, prefix=self._child_prefix)
        raise KeyError(name)

    def __setitem__(self, name: str, record: Dict[str, Any]) -> None:
        self._store._write(self._type, self._dir, name, record)

    def create(self, name: str, record: Dict[str, Any]) -> str:
        self._store._write(self._type, self._dir, name, record)
        return name

    def lock(self, name: str = "-"):
        """Reserve ``name`` (mint a free one if ``-``); yields the name, clears the
        marker on exit. Write inside via ``group[name] = record``."""
        return self._store._lock(self._type, self._dir, name, self._prefix)

    def __contains__(self, name: object) -> bool:
        return (self._dir / f"{name}.json").is_file() or (self._dir / str(name)).is_dir()

    def __iter__(self) -> Iterator[str]:
        if not self._dir.is_dir():
            return iter(())
        names = set()
        for path in self._dir.iterdir():
            if path.name.startswith("."):
                continue
            if path.is_dir():
                names.add(path.name)
            elif path.suffix == ".json":
                names.add(path.stem)
        return iter(sorted(names))

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __repr__(self) -> str:
        return f"<store[{self._type!r}] @ {self._dir} {list(self)}>"


class Store:
    def __init__(self, root: str | os.PathLike = "store") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lockfile = str(self.root / ".store.lock")

    # ---------------------------------------------------------------- dict view
    def __getitem__(self, type_: str) -> _Group:
        if type_ not in TYPE_DIR:
            raise KeyError(type_)
        directory = self.root / TYPE_DIR[type_]
        return _Group(
            self,
            type_,
            directory,
            prefix=DEFAULT_PREFIX.get(type_),
            is_namespace=type_ in CHILD_PREFIX,
            child_prefix=CHILD_PREFIX.get(type_),
        )

    def __iter__(self) -> Iterator[str]:
        return iter(TYPE_DIR)

    def __contains__(self, type_: object) -> bool:
        return type_ in TYPE_DIR

    def read(self, ref: str) -> Dict[str, Any]:
        """Full record by bare name, searched across collections (for refs that do
        not carry their type, e.g. at the MCP boundary)."""
        for directory in TYPE_DIR.values():
            path = self.root / directory / f"{ref}.json"
            if path.is_file():
                return _read_json(path)
        raise StoreError("not_found", f"no record named {ref!r}", {"ref": ref})

    def names(self, type_: str) -> List[str]:
        """Every record name in a collection (recursive; for listings)."""
        directory = self.root / TYPE_DIR[type_]
        if not directory.is_dir():
            return []
        return sorted(
            p.relative_to(directory).with_suffix("").as_posix()
            for p in directory.rglob("*.json")
            if not p.name.startswith(".")
        )

    @staticmethod
    def references_of(record: Dict[str, Any]) -> Dict[str, Any]:
        refs: Dict[str, Any] = {}
        type_ = record.get("type")
        if isinstance(type_, str):
            refs[type_] = record.get("name")
        for field in LINK_FIELDS:
            value = record.get(field)
            if value is not None:
                refs[field] = value
        fit = record.get("fit")
        if isinstance(fit, dict):
            if fit.get("data") is not None:
                refs["data"] = fit["data"]
            if fit.get("tool_result") is not None:
                refs["call"] = fit["tool_result"]
        return refs

    # ---------------------------------------------------------------- internals
    def _locked(self):
        return _flock(self._lockfile)

    def _full_name(self, type_: str, directory: Path, name: str) -> str:
        collection_root = self.root / TYPE_DIR[type_]
        return (directory / name).relative_to(collection_root).as_posix()

    def _exists_anywhere(self, full: str) -> bool:
        return any((self.root / d / f"{full}.json").is_file() for d in TYPE_DIR.values())

    @staticmethod
    def _mint_in(directory: Path, prefix: str) -> str:
        """``<prefix><max integer suffix + 1>`` over the directory's immediate
        children (records *and* reservation markers), in one ``iterdir`` pass."""
        highest = 0
        if directory.is_dir():
            for path in directory.iterdir():
                if path.name.startswith(".") or path.suffix not in (".json", ".lock"):
                    continue
                stem = path.stem
                if stem.startswith(prefix):
                    tail = stem[len(prefix):]
                    if tail.isdigit():
                        highest = max(highest, int(tail))
        return f"{prefix}{highest + 1}"

    @staticmethod
    def _validate(type_: str, record: Dict[str, Any]) -> None:
        if not isinstance(record, dict):
            raise StoreError("invalid_record", f"{type_} record must be a dict", {"type": type_})
        missing = [key for key in REQUIRED.get(type_, ()) if key not in record]
        if missing:
            raise StoreError(
                "invalid_record",
                f"{type_} record is missing required keys: {', '.join(missing)}",
                {"type": type_, "missing": missing},
            )

    @staticmethod
    def _atomic_write(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
        os.replace(tmp, path)

    @staticmethod
    def _unlink(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _write(self, type_: str, directory: Path, name: str, record: Dict[str, Any]) -> str:
        """Append-only write under the global lock: validate, refuse an existing
        name (anywhere), inject ``type``/``name``/``created_at``/``description``."""
        with self._locked():
            full = self._full_name(type_, directory, name)
            if self._exists_anywhere(full):
                raise StoreError("name_exists", f"a record named {full!r} already exists", {"name": full})
            self._validate(type_, record)
            body = dict(record)
            body["type"] = type_
            body["name"] = full
            body.setdefault("created_at", _now())
            body.setdefault("description", "")
            self._atomic_write(directory / f"{name}.json", body)
        return full

    @contextmanager
    def _lock(self, type_: str, directory: Path, name: str, prefix: Optional[str]) -> Iterator[str]:
        with self._locked():
            if not name or name == "-":
                if prefix is None:
                    raise StoreError(
                        "naming_error",
                        "cannot mint a name at this level; give a name or descend into a sub-namespace",
                        {"type": type_},
                    )
                name = self._mint_in(directory, prefix)
            full = self._full_name(type_, directory, name)
            marker = directory / f"{name}.lock"
            if self._exists_anywhere(full):
                raise StoreError("name_exists", f"a record named {full!r} already exists", {"name": full})
            if marker.exists():
                raise StoreError("name_locked", f"the name {full!r} is currently reserved", {"name": full})
            marker.parent.mkdir(parents=True, exist_ok=True)
            try:
                fd = os.open(str(marker), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError as exc:  # pragma: no cover - guarded above
                raise StoreError("name_locked", f"the name {full!r} is currently reserved", {"name": full}) from exc
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"name": full, "type": type_, "pid": os.getpid(), "reserved_at": _now()}, stream)
        try:
            yield name
        finally:
            with self._locked():
                self._unlink(marker)
