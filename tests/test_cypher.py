import os
import json
import base64
import hashlib
import tempfile
from pathlib import Path

import pytest

import cipher

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def fixed_key():
    # 32-byte key for deterministic tests
    return b"\x01" * 32

@pytest.fixture(autouse=True)
def no_git_env(monkeypatch, tmp_path, request):
    """
    Ensure GIT_CRYPT_KEY is unset and we start in an empty temp dir,
    unless a test explicitly sets GIT_CRYPT_KEY or chdirs.
    """
    monkeypatch.delenv("GIT_CRYPT_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    yield

# -----------------------------------------------------------------------------
# 1) load_git_crypt_key()
# -----------------------------------------------------------------------------
def test_load_git_crypt_key_from_file(tmp_path, monkeypatch):
    keyfile = tmp_path / "mykey.bin"
    keyfile.write_bytes(b"supersecret")
    monkeypatch.setenv("GIT_CRYPT_KEY", str(keyfile))
    assert cipher.load_git_crypt_key() == b"supersecret"

def test_load_git_crypt_key_from_base64(monkeypatch):
    raw = b"anothersecret"
    encoded = base64.b64encode(raw).decode("ascii")
    monkeypatch.setenv("GIT_CRYPT_KEY", encoded)
    assert cipher.load_git_crypt_key() == raw

def test_load_git_crypt_key_walks_up_git_dir(tmp_path):
    # create nested directories
    project = tmp_path / "proj"
    nested = project / "a" / "b" / "c"
    nested.mkdir(parents=True)
    # simulate .git/git-crypt/keys/default
    key_dir = project / ".git" / "git-crypt" / "keys"
    key_dir.mkdir(parents=True)
    default_key = key_dir / "default"
    default_key.write_bytes(b"walkedkey")
    # chdir into the deepest
    os.chdir(nested)
    assert cipher.load_git_crypt_key() == b"walkedkey"

def test_load_git_crypt_key_not_found(tmp_path):
    # no GIT_CRYPT_KEY and no .git folder → FileNotFoundError
    with pytest.raises(FileNotFoundError):
        cipher.load_git_crypt_key()

# -----------------------------------------------------------------------------
# 2) derive_key()
# -----------------------------------------------------------------------------
def test_derive_key_matches_sha256(fixed_key):
    raw = b"input_bytes"
    # compare to direct hashlib.sha256
    assert cipher.derive_key(raw) == hashlib.sha256(raw).digest()

# -----------------------------------------------------------------------------
# 3) encrypt_bytes()/decrypt_bytes()
# -----------------------------------------------------------------------------
def test_encrypt_decrypt_bytes_roundtrip(fixed_key):
    plaintext = b"hello, world!"
    blob = cipher.encrypt_bytes(plaintext, key=fixed_key)
    assert isinstance(blob, bytes)
    recovered = cipher.decrypt_bytes(blob, key=fixed_key)
    assert recovered == plaintext

def test_decrypt_bytes_too_short(fixed_key):
    with pytest.raises(ValueError, match="too short"):
        cipher.decrypt_bytes(b"", key=fixed_key)

def test_decrypt_bytes_authentication_failure(fixed_key):
    blob = cipher.encrypt_bytes(b"data", key=fixed_key)
    # flip one bit in the tag
    tampered = bytearray(blob)
    tampered[-1] ^= 0x01
    with pytest.raises(ValueError, match="Authentication failed"):
        cipher.decrypt_bytes(bytes(tampered), key=fixed_key)

# -----------------------------------------------------------------------------
# 4) encrypt_str()/decrypt_str()
# -----------------------------------------------------------------------------
def test_encrypt_decrypt_str_roundtrip(fixed_key):
    s = "¡Hola! 🌍"
    blob = cipher.encrypt_str(s, key=fixed_key)
    assert isinstance(blob, bytes)
    out = cipher.decrypt_str(blob, key=fixed_key)
    assert out == s

# -----------------------------------------------------------------------------
# 5) encrypt_json()/decrypt_json()
# -----------------------------------------------------------------------------
def test_encrypt_decrypt_json_roundtrip(fixed_key):
    obj = {"num": 42, "text": "foo", "list": [1, 2, 3]}
    blob = cipher.encrypt_json(obj, key=fixed_key, sort_keys=True)
    assert isinstance(blob, bytes)
    result = cipher.decrypt_json(blob, key=fixed_key)
    assert result == obj

# -----------------------------------------------------------------------------
# 6) Padding edge case
# -----------------------------------------------------------------------------
def test_full_block_padding(fixed_key):
    # data length exactly multiple of NONCE_SIZE
    size = cipher.NONCE_SIZE * 2
    data = b"\x00" * size
    blob = cipher.encrypt_bytes(data, key=fixed_key)
    out = cipher.decrypt_bytes(blob, key=fixed_key)
    assert out == data

# -----------------------------------------------------------------------------
# 7) JSON wrappers raise on bad input
# -----------------------------------------------------------------------------
def test_decrypt_json_invalid_tag(fixed_key):
    blob = cipher.encrypt_json({"a": 1}, key=fixed_key)
    # corrupt tag
    tampered = bytearray(blob)
    tampered[-cipher.TAG_SIZE] ^= 0xFF
    with pytest.raises(ValueError):
        cipher.decrypt_json(bytes(tampered), key=fixed_key)