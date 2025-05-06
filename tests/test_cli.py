import os
import json
import base64
import pytest
import tempfile

import cli

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def clean_env_and_stub_decrypt(monkeypatch, tmp_path):
    """
    - Run each test in an empty temp dir
    - Ensure TEST_KEY is unset
    - Stub out cipher.decrypt_str to return (blob, key)
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TEST_KEY", raising=False)
    # stub decrypt_str for predictable lookup results
    monkeypatch.setattr(cli.cipher, "decrypt_str", lambda blob, key: (blob, key))
    yield

# -----------------------------------------------------------------------------
# lookup() tests
# -----------------------------------------------------------------------------
def test_lookup_decrypt_with_env():
    # prepare a Base64 blob for b'hi'
    b64 = base64.b64encode(b"hi").decode("ascii")
    # env var contains Base64-encoded key "mykeydata"
    os.environ["TEST_KEY"] = base64.b64encode(b"mykeydata").decode("ascii")
    values = {"foo": "${" + f"decrypt('{b64}','TEST_KEY')" + "}"}
    result = cli.lookup("foo", values)
    assert result == (b"hi", b"mykeydata")

def test_lookup_json_literal():
    json_str = '{"x":10,"y":[1,2]}'
    values = {"foo": json_str}
    result = cli.lookup("foo", values)
    assert result == {"x": 10, "y": [1, 2]}

def test_lookup_fallback_non_matching():
    values = {"foo": "plain text", "bar": 123}
    assert cli.lookup("foo", values) == "plain text"
    assert cli.lookup("bar", values) == 123

def test_lookup_unknown_path_raises():
    with pytest.raises(KeyError):
        cli.lookup("no.such", {})

def test_lookup_invalid_json_raises():
    values = {"foo": "{invalid: }"}
    with pytest.raises(ValueError) as exc:
        cli.lookup("foo", values)
    assert "Invalid JSON" in str(exc.value)

# -----------------------------------------------------------------------------
# interpolate() tests
# -----------------------------------------------------------------------------
def test_interpolate_simple_substitution():
    vals = {"a": "${b} world", "b": "hello"}
    out = cli.interpolate("${a}", vals)
    assert out == "hello world"

def test_interpolate_numeric_coercion():
    vals = {"num": 123, "s": "${num}"}
    assert cli.interpolate("${s}", vals) == "123"

def test_interpolate_nested():
    vals = {"a": "${b}", "b": "${c}", "c": "end"}
    assert cli.interpolate("${a}", vals) == "end"

def test_interpolate_cycle_detection():
    vals = {"a": "${b}", "b": "${a}"}
    with pytest.raises(ValueError) as exc:
        cli.interpolate("${a}", vals)
    assert "Cycle detected" in str(exc.value)

# -----------------------------------------------------------------------------
# Args.parse_arg tests
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("arg, expected_path, expected_value", [
    ("x.y.3=true",        ["x", "y", 3],    True),
    ("n=3.14",            ["n"],            3.14),
    ("flag",              ["flag"],         True),
    ("s=hello",           ["s"],            "hello"),
    ("i=42",              ["i"],            42),
])
def test_parse_arg_various(arg, expected_path, expected_value):
    path, value = cli.Args.parse_arg(arg)
    assert path == expected_path
    assert value == expected_value

# -----------------------------------------------------------------------------
# Args.set_arg tests
# -----------------------------------------------------------------------------
def test_set_arg_creates_nested_structures():
    a = cli.Args()
    a.set_arg(["a", "b", 2], "val")
    assert a.args == {"a": {"b": [None, None, "val"]}}

def test_set_arg_invalid_list_index_raises():
    a = cli.Args()
    with pytest.raises(TypeError):
        a.set_arg([0], "oops")

# -----------------------------------------------------------------------------
# Args.inc tests
# -----------------------------------------------------------------------------
def test_inc_weak_merge(tmp_path):
    a = cli.Args()
    a._args = {"foo": 10}
    file = tmp_path / "inc.json"
    file.write_text(json.dumps({"foo": 1, "bar": 2}))
    a.inc(str(file), strongly=False)
    assert a.args == {"foo": 10, "bar": 2}

def test_inc_strong_merge(tmp_path):
    a = cli.Args()
    a._args = {"foo": 10}
    file = tmp_path / "inc.json"
    file.write_text(json.dumps({"foo": 1, "bar": 2}))
    a.inc(str(file), strongly=True)
    assert a.args == {"foo": 1, "bar": 2}

# -----------------------------------------------------------------------------
# Args.from_cli tests
# -----------------------------------------------------------------------------
def test_from_cli_default_include(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"cfg": 1}))
    args = cli.Args.from_cli(cfg_path=str(cfg), cli_args=["x=5"])
    assert args.args["cfg"] == 1
    assert args.args["x"] == 5

def test_from_cli_no_inline(tmp_path):
    f1 = tmp_path / "f1.json"
    f1.write_text(json.dumps({"a": 1}))
    args = cli.Args.from_cli(cfg_path=str(f1), cli_args=[])
    assert args.args == {"a": 1}

def test_from_cli_inline_inc_strong_overrides_default(tmp_path):
    cfg_default = tmp_path / "default.json"
    cfg_default.write_text(json.dumps({"d": 9}))
    cfg2 = tmp_path / "second.json"
    cfg2.write_text(json.dumps({"s": 2}))
    # 'inc!(second)' should load second.json and skip default.json
    args = cli.Args.from_cli(cfg_path=str(cfg_default), cli_args=[f"inc!({cfg2.name})"])
    assert args.args == {"s": 2}

def test_from_cli_inline_inc_weak_then_set(tmp_path):
    cfg1 = tmp_path / "cfg1.json"
    cfg1.write_text(json.dumps({"a": 1}))
    cfg2 = tmp_path / "cfg2.json"
    cfg2.write_text(json.dumps({"b": 2}))
    args = cli.Args.from_cli(cfg_path=str(cfg1), cli_args=[f"inc({cfg2.name})", "c=3"])
    assert args.args == {"b": 2, "c": 3}

# -----------------------------------------------------------------------------
# Args.val tests
# -----------------------------------------------------------------------------
def test_val_missing_and_defaults():
    a = cli.Args()
    assert a.val(["no"]) is None
    assert a.val(["no"], missing=42) == 42
    assert a.val(["no"], missing=lambda: 7) == 7

def test_val_with_interpolation(monkeypatch):
    a = cli.Args()
    a._args = {"x": "${y}", "y": "hello"}
    assert a.val(["x"]) == "hello"

# -----------------------------------------------------------------------------
# top-level args() alias
# -----------------------------------------------------------------------------
def test_args_alias(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"foo": 1}))
    args = cli.args(cfg_path=str(cfg), cli_args=["bar=2"])
    assert args.args == {"foo": 1, "bar": 2}