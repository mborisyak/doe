from __future__ import annotations

import runpy


def test_all_tools_are_exposed() -> None:
    namespace = runpy.run_path("mcp/server.py", run_name="not_main")
    server = namespace["server"]
    assert set(server._tool_manager._tools.keys()) == {
        "fit_ode",
        "doe_ode",
        "simulate_enzyme_dynamics",
        "store_create",
        "store_get",
        "store_list",
        "get_config",
        "create_gp",
        "hyper_fit_gp",
        "fit_gp",
        "predict_gp",
        "doe_gp",
        "discriminatory_doe_gp",
    }

