from __future__ import annotations

import runpy


def test_all_tools_are_exposed() -> None:
    namespace = runpy.run_path("mcp/server.py", run_name="not_main")
    server = namespace["server"]
    assert set(server._tool_manager._tools.keys()) == {
        "fit_parameters",
        "propose_doe_experiments",
        "simulate_enzyme_dynamics",
    }

