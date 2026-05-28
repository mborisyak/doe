from __future__ import annotations

from config_tools import ConfigMcpService


def test_get_config_reads_named_yaml(tmp_path):
    (tmp_path / "enzyme.yaml").write_text("experiment:\n  duration: 30.0\nnoise: 0.025\n")
    out = ConfigMcpService(root=str(tmp_path), name="enzyme").get()
    assert out["ok"]
    assert out["data"]["name"] == "enzyme"
    assert out["data"]["config"]["experiment"]["duration"] == 30.0
    assert out["data"]["config"]["noise"] == 0.025


def test_get_config_missing_is_error(tmp_path):
    out = ConfigMcpService(root=str(tmp_path), name="ghost").get()
    assert out["ok"] is False
    assert out["error"]["code"] == "not_found"


def test_get_config_name_and_root_from_env(tmp_path, monkeypatch):
    (tmp_path / "custom.yaml").write_text("k: 1\n")
    monkeypatch.setenv("DOE_CONFIG", "custom")
    monkeypatch.setenv("DOE_CONFIG_ROOT", str(tmp_path))
    out = ConfigMcpService().get()
    assert out["ok"]
    assert out["data"]["name"] == "custom"
    assert out["data"]["config"]["k"] == 1


def test_get_config_defaults_to_enzyme():
    # the real config/enzyme.yaml shipped in the repo
    out = ConfigMcpService(root="config").get()
    assert out["ok"]
    assert out["data"]["name"] == "enzyme"
    assert "experiment" in out["data"]["config"]
