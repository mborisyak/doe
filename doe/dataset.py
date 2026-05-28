"""Assemble batch files into a single dataset.

A *batch file* is the self-describing JSON the store writes -- or any hand-written
plain file in the same shape::

    {"experiments": {
        "<exp>": {"conditions":   {<var>: float, ...},
                  "measurements": {"timestamps": [...], "<obs>": [...], ...}},
        ...}}

The **batch name is the file's basename stem** (``batch-1.json`` -> ``batch-1``),
never the in-file ``name`` key -- so scripts chain over loose files on disk with no
store and no MCP involved. Experiments are merged across files and labelled by
``(batch, exp)``; the flat string label is ``"<batch>/<exp>"``.

Measurements are *quantity-keyed*: ``timestamps`` is the shared sample grid and every
other key is one observable's measured series (multiple observables are allowed). The
same shape is used for model output, where each observable may be paired with a
``sigma_<obs>`` predictive-uncertainty series -- see ``observables_of``.

Run standalone to merge files into one dataset JSON::

    python -m doe.dataset store/batches/batch-1.json store/batches/batch-2.json -o dataset.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

Label = Tuple[str, str]  # (batch, exp)

TIMESTAMPS = "timestamps"


def batch_name(path: str | Path) -> str:
    """Batch identity = the file's basename stem, deliberately not the ``name`` key."""
    return Path(path).stem


def observables_of(measurements: Dict[str, Any]) -> List[str]:
    """Observable names in a measurement/prediction dict: every key that is not the
    shared ``timestamps`` grid and not a ``sigma_<obs>`` uncertainty series."""
    return [
        key
        for key in measurements
        if key != TIMESTAMPS and not key.startswith("sigma_")
    ]


def load_batch(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Read one batch file, returning its ``experiments`` mapping. Accepts either a
    full record (``{"experiments": {...}, ...}``) or a bare experiments dict. Store
    metadata (``name``, ``type``, ``source``, ...) is ignored."""
    with open(path) as stream:
        record = json.load(stream)
    experiments = record.get("experiments", record) if isinstance(record, dict) else None
    if not isinstance(experiments, dict):
        raise ValueError(f"{path}: no 'experiments' mapping found")
    return experiments


class Dataset:
    """Experiments merged across batch files, keyed by ``(batch, exp)`` in insertion
    order. Each entry is ``{"conditions": {...}, "measurements": {...}}``; measurements
    may be absent (e.g. a design carries conditions only)."""

    def __init__(self) -> None:
        self.entries: Dict[Label, Dict[str, Any]] = {}

    # -- construction ---------------------------------------------------------
    def add_record(self, batch: str, record: Dict[str, Any]) -> None:
        """Add an already-loaded batch ``record`` (e.g. ``store["data"][name]``) under
        the name ``batch``. Store metadata keys are ignored; only ``experiments`` is used."""
        experiments = record.get("experiments", record) if isinstance(record, dict) else None
        if not isinstance(experiments, dict):
            raise ValueError(f"batch {batch!r}: no 'experiments' mapping")
        for exp, payload in experiments.items():
            label = (batch, exp)
            if label in self.entries:
                raise ValueError(f"duplicate experiment {label!r}")
            if "conditions" not in payload:
                raise ValueError(f"batch {batch!r}: experiment {exp!r} has no 'conditions'")
            self.entries[label] = payload

    def add_batch(self, path: str | Path) -> None:
        """Add a batch from a JSON file on disk; batch name = the file's basename stem."""
        self.add_record(batch_name(path), load_batch(path))

    # -- views ----------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[Label]:
        return iter(self.entries)

    @staticmethod
    def slash(label: Label) -> str:
        return f"{label[0]}/{label[1]}"

    def conditions(self) -> Dict[str, Dict[str, Any]]:
        """``{"<batch>/<exp>": {<var>: float, ...}}`` for every entry."""
        return {self.slash(lbl): e["conditions"] for lbl, e in self.entries.items()}

    def observables(self) -> List[str]:
        """Sorted union of observable names across all measured entries."""
        names: set[str] = set()
        for entry in self.entries.values():
            names.update(observables_of(entry.get("measurements", {})))
        return sorted(names)

    def measured_labels(self) -> List[Label]:
        return [lbl for lbl, e in self.entries.items() if e.get("measurements")]

    def series(self, observable: str) -> Dict[str, Dict[str, List[float]]]:
        """Bridge to the single-observable inference core: for one observable, return
        ``{"<batch>/<exp>": {"timestamps": [...], "measurements": [...]}}`` over the
        entries that measured it -- the generic shape the fit/likelihood expects."""
        out: Dict[str, Dict[str, List[float]]] = {}
        for lbl, entry in self.entries.items():
            meas = entry.get("measurements") or {}
            if observable in meas:
                out[self.slash(lbl)] = {
                    "timestamps": meas[TIMESTAMPS],
                    "measurements": meas[observable],
                }
        return out

    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        """Flat, self-contained, JSON-ready merged dataset keyed by ``"<batch>/<exp>"``."""
        return {self.slash(lbl): entry for lbl, entry in self.entries.items()}


def assemble(paths: Iterable[str | Path]) -> Dataset:
    """Merge the given batch files into one :class:`Dataset` (batch = filename stem)."""
    dataset = Dataset()
    for path in paths:
        dataset.add_batch(path)
    return dataset


def from_store(store: Any, names: Iterable[str]) -> Dataset:
    """Merge batches read from a store by name (``store["data"][name]``) into a
    :class:`Dataset`. This is how the scripts assemble data: by ref, through the store
    API, with the batch name being the store ref."""
    dataset = Dataset()
    for name in names:
        dataset.add_record(name, store["data"][name])
    return dataset


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Merge batch files into a single dataset JSON.")
    ap.add_argument("batches", nargs="+", metavar="FILE", help="batch JSON files (stem = batch name)")
    ap.add_argument("-o", "--output", metavar="FILE", default=None,
                    help="write the merged dataset here instead of stdout")
    args = ap.parse_args(argv)

    dataset = assemble(args.batches)
    payload = json.dumps(dataset.as_dict(), indent=2)
    if args.output is None:
        print(payload)
    else:
        Path(args.output).write_text(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
