from __future__ import annotations

import threading
import time

import pytest

from doe.store import Store, StoreError, view
from store_tools import StoreMcpService


def _run_threads(worker, n):
    barrier = threading.Barrier(n)
    threads = [threading.Thread(target=worker, args=(barrier, i)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def _model(description="a candidate mechanism"):
    return {"kind": "ode", "description": description,
            "spec": {"states": ["A", "B"], "parameters": {"k": [0.0, 1.0]}}}


def _fitted():
    return {"model": "model-x", "parameters": {"k": 0.5}, "fit": {"final_loss": 0.27},
            "auxiliary": {"loss_trace": [0.5, 0.3, 0.27]}}


def _batch():
    return {"space": ["A"], "observable": "A_measured",
            "experiments": {"experiment-1": {"x": {"A": 1.0}}},
            "auxiliary": {"experiment-1": {"t": [1.0, 2.0], "y": [0.9, 0.8]}}}


# --------------------------------------------------------------- dict-like reads
def test_setitem_and_getitem(tmp_path):
    s = Store(tmp_path / "store")
    s["model"]["model-x"] = _model()
    rec = s["model"]["model-x"]                      # full record (auxiliary included)
    assert rec["name"] == "model-x" and rec["type"] == "model"
    assert rec["description"] == "a candidate mechanism"
    assert rec["spec"]["states"] == ["A", "B"]


def test_nested_fitted_namespace(tmp_path):
    s = Store(tmp_path / "store")
    s["fitted_model"]["model-x"]["v1"] = _fitted()
    s["fitted_model"]["model-x"]["v2"] = _fitted()
    assert sorted(s["fitted_model"]["model-x"].keys()) == ["v1", "v2"]
    rec = s["fitted_model"]["model-x"]["v1"]         # full content, arrays included
    assert rec["name"] == "model-x/v1"
    assert rec["auxiliary"]["loss_trace"] == [0.5, 0.3, 0.27]
    assert (tmp_path / "store" / "fitted_models" / "model-x" / "v1.json").is_file()


def test_keys_contains_len(tmp_path):
    s = Store(tmp_path / "store")
    s["model"]["model-a"] = _model()
    s["model"]["model-b"] = _model()
    assert set(s["model"].keys()) == {"model-a", "model-b"}
    assert "model-a" in s["model"] and "nope" not in s["model"]
    assert len(s["model"]) == 2


def test_missing_key_raises(tmp_path):
    s = Store(tmp_path / "store")
    with pytest.raises(KeyError):
        _ = s["model"]["nope"]


# --------------------------------------------------------------- writes / rules
def test_append_only_no_overwrite(tmp_path):
    s = Store(tmp_path / "store")
    s["model"]["model-x"] = _model()
    with pytest.raises(StoreError) as exc:
        s["model"]["model-x"] = _model("different")
    assert exc.value.code == "name_exists"


def test_validate_required_keys(tmp_path):
    s = Store(tmp_path / "store")
    with pytest.raises(StoreError) as exc:
        s["model"]["m"] = {"description": "no spec"}
    assert exc.value.code == "invalid_record" and "spec" in exc.value.details["missing"]


def test_no_delete(tmp_path):
    s = Store(tmp_path / "store")
    assert not hasattr(s, "delete")
    assert not hasattr(s["model"], "__delitem__") or "model" not in s  # append-only


# --------------------------------------------------------------- lock / minting
def test_lock_mints_and_yields_name(tmp_path):
    s = Store(tmp_path / "store")
    with s["model"].lock("-") as name:
        assert name == "model-1"
        assert (tmp_path / "store" / "models" / "model-1.lock").is_file()  # reserved
        s["model"][name] = _model()
    assert s["model"]["model-1"]["type"] == "model"
    assert not (tmp_path / "store" / "models" / "model-1.lock").exists()   # cleared


def test_lock_fitted_mints_version(tmp_path):
    s = Store(tmp_path / "store")
    with s["fitted_model"]["model-x"].lock("-") as v:
        assert v == "v1"
        s["fitted_model"]["model-x"][v] = _fitted()
    with s["fitted_model"]["model-x"].lock("-") as v:
        assert v == "v2"


def test_lock_top_level_fitted_has_no_mint(tmp_path):
    s = Store(tmp_path / "store")
    with pytest.raises(StoreError) as exc:
        with s["fitted_model"].lock("-"):
            pass
    assert exc.value.code == "naming_error"


def test_minting_skips_reserved(tmp_path):
    s = Store(tmp_path / "store")
    with s["model"].lock("-") as outer:
        assert outer == "model-1"
        with s["model"].lock("-") as inner:        # must skip the locked model-1
            assert inner == "model-2"


def test_lock_existing_or_locked_raises(tmp_path):
    s = Store(tmp_path / "store")
    s["model"]["model-x"] = _model()
    with pytest.raises(StoreError) as e1:
        with s["model"].lock("model-x"):           # already a record
            pass
    assert e1.value.code == "name_exists"
    with s["model"].lock("model-y"):
        with pytest.raises(StoreError) as e2:
            with s["model"].lock("model-y"):       # already reserved
                pass
        assert e2.value.code == "name_locked"


def test_read_and_not_found(tmp_path):
    s = Store(tmp_path / "store")
    s["fitted_model"]["model-x"]["v1"] = _fitted()
    assert s.read("model-x/v1")["parameters"] == {"k": 0.5}   # bare-name find
    with pytest.raises(StoreError) as exc:
        s.read("nope")
    assert exc.value.code == "not_found"


def test_references_of(tmp_path):
    s = Store(tmp_path / "store")
    s["fitted_model"]["model-x"]["v1"] = {
        "model": "model-x", "parameters": {"k": 0.5},
        "fit": {"data": ["batch-1"], "tool_result": "fit/model-x/v1"},
    }
    refs = s.references_of(s.read("model-x/v1"))
    assert refs == {"fitted_model": "model-x/v1", "model": "model-x",
                    "data": ["batch-1"], "call": "fit/model-x/v1"}


# --------------------------------------------------------------- view (MCP only)
def test_view_nulls_auxiliary(tmp_path):
    s = Store(tmp_path / "store")
    s["fitted_model"]["model-x"]["v1"] = _fitted()
    rec = s["fitted_model"]["model-x"]["v1"]
    assert view(rec)["auxiliary"] == {"loss_trace": None}
    assert view(rec, select="auxiliary.loss_trace") == [0.5, 0.3, 0.27]
    assert view(rec)["fit"]["final_loss"] == 0.27


# --------------------------------------------------------------- concurrency
def test_concurrent_lock_mints_unique(tmp_path):
    s = Store(tmp_path / "store")
    n = 50
    names, errors = [], []
    guard = threading.Lock()

    def worker(barrier, i):
        barrier.wait()
        try:
            with s["model"].lock("-") as name:
                time.sleep(0.001)
                s["model"][name] = _model(str(i))
            with guard:
                names.append(name)
        except Exception as exc:  # noqa: BLE001
            with guard:
                errors.append(repr(exc))

    _run_threads(worker, n)
    assert errors == []
    assert len(names) == n == len(set(names))
    assert set(s["model"].keys()) == {f"model-{i}" for i in range(1, n + 1)}


def test_concurrent_fitted_versions_unique(tmp_path):
    s = Store(tmp_path / "store")
    n = 40
    versions = []
    guard = threading.Lock()

    def worker(barrier, i):
        barrier.wait()
        with s["fitted_model"]["model-x"].lock("-") as v:
            s["fitted_model"]["model-x"][v] = {"model": "model-x", "parameters": {"k": i}}
        with guard:
            versions.append(v)

    _run_threads(worker, n)
    assert len(versions) == n == len(set(versions))
    assert set(s["fitted_model"]["model-x"].keys()) == {f"v{i}" for i in range(1, n + 1)}


def test_concurrent_same_name_single_winner(tmp_path):
    s = Store(tmp_path / "store")
    n = 30
    won, failed = [], []
    guard = threading.Lock()

    def worker(barrier, i):
        barrier.wait()
        try:
            with s["model"].lock("model-x") as name:
                s["model"][name] = _model(str(i))
            with guard:
                won.append(i)
        except StoreError as exc:
            with guard:
                failed.append(exc.code)

    _run_threads(worker, n)
    assert len(won) == 1
    assert set(failed) <= {"name_exists", "name_locked"}
    assert s["model"]["model-x"]["description"] == str(won[0])


# --------------------------------------------------------------- MCP service
def test_service_create_get_list(tmp_path):
    svc = StoreMcpService(Store(tmp_path / "store"))
    created = svc.create("model", "-", {"kind": "ode", "spec": {}}, description="first try")
    assert created["ok"] and created["references"]["model"] == "model-1"
    got = svc.get("model-1")
    assert got["ok"] and got["data"]["description"] == "first try"
    listed = svc.list_records("model")
    assert listed["ok"] and listed["data"][0]["description"] == "first try"
    forbidden = svc.create("batch", "-", _batch())
    assert forbidden["ok"] is False and forbidden["error"]["code"] == "forbidden"
    missing = svc.get("nope")
    assert missing["ok"] is False and missing["error"]["code"] == "not_found"


def test_service_get_nulls_auxiliary(tmp_path):
    store = Store(tmp_path / "store")
    store["fitted_model"]["model-x"]["v1"] = _fitted()
    svc = StoreMcpService(store)
    got = svc.get("model-x/v1")
    assert got["data"]["auxiliary"] == {"loss_trace": None}
    assert svc.get("model-x/v1", select="auxiliary.loss_trace")["data"] == [0.5, 0.3, 0.27]
