"""MCP tool exposing the agent-facing experiment config.

One config per deployment, selected by the ``DOE_CONFIG`` env var (default ``enzyme``)
and read from ``<root>/<name>.yaml`` (root from ``DOE_CONFIG_ROOT``, default ``config``).
The agent reads it for the experiment's concentrations, condition bounds, sampling
duration, measurement count, noise, etc.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from mcp_errors import error_response, success_response


class ConfigMcpService:
    def __init__(self, root: Optional[str] = None, name: Optional[str] = None) -> None:
        self.root = Path(root if root is not None else os.environ.get("DOE_CONFIG_ROOT", "config"))
        self.name = name if name is not None else os.environ.get("DOE_CONFIG", "enzyme")

    def get(self) -> Dict[str, Any]:
        path = self.root / f"{self.name}.yaml"
        try:
            with path.open("r", encoding="utf-8") as stream:
                config = yaml.safe_load(stream)
        except FileNotFoundError:
            return error_response(
                "not_found",
                f"config {self.name!r} not found at {path}",
                {"name": self.name, "path": str(path)},
            )
        return success_response({"name": self.name, "config": config})
