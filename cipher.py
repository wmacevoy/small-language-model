from typing import Any, Optional
import json
import os
import hashlib
import base64
import hmac
import argparse

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
NONCE_SIZE = 32  # bytes == SHA-256 output size
TAG_SIZE = 32    # bytes == SHA-256 output size

# -----------------------------------------------------------------------------
# 1) Load git-crypt key
# -----------------------------------------------------------------------------
def load_git_crypt_key(env_var: str = "GIT_CRYPT_KEY") -> bytes:
    """
    Load the raw git-crypt key.
    
    1) If the GIT_CRYPT_KEY env var is set:
       – if it names an existing file, load and return its contents;
       – otherwise, base64-decode its value.
    2) Otherwise, walk up from cwd to root looking for `.git/git-crypt/key`.
    """
    val = os.environ.get(env_var)
    if val:
        # Env var provided
        if os.path.isfile(val):
            with open(val, "rb") as f:
                return f.read()
        try:
            return base64.b64decode(val)
        except Exception:
            raise ValueError(f"{env_var} is not a file path nor valid base64")

    # Walk up directory tree to find .git/git-crypt/key
    path = os.path.abspath(os.getcwd())
    while True:
        git_dir = os.path.join(path, ".git")
        key_file = os.path.join(git_dir, "git-crypt", "keys", "default")
        if os.path.isdir(git_dir) and os.path.isfile(key_file):
            with open(key_file, "rb") as f:
                return f.read()
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent

    raise FileNotFoundError(
        "Could not locate git-crypt key; set GIT_CRYPT_KEY or run inside a git-crypt repo."
    )

# -----------------------------------------------------------------------------
# 2) Derive key
# -----------------------------------------------------------------------------
def derive_key(raw_key: bytes) -> bytes:
    """
    Derive a 32-byte AES key from the raw git-crypt key via SHA-256.
    """
    return hashlib.sha256(raw_key).digest()

# -----------------------------------------------------------------------------
# 3) Encrypt / Decrypt utilities (SHA-256-CTR + HMAC tag)
# -----------------------------------------------------------------------------
def encrypt_bytes(data: bytes, key: Optional[bytes] = None) -> bytes:
    """
    Encrypt `data` using SHA-256-CTR plus a SHA-256 authentication tag.
    Returns: nonce || ciphertext || tag
      - nonce: NONCE_SIZE bytes
      - tag:   TAG_SIZE bytes
    """
    if key is None:
        key = derive_key(load_git_crypt_key())

    # 1) Nonce
    nonce = os.urandom(NONCE_SIZE)
    # 2) Seed = SHA256(key || nonce)
    seed = hashlib.sha256(key + nonce).digest()
    # 3) PKCS#7 padding to multiple of NONCE_SIZE
    pad_len = NONCE_SIZE - (len(data) % NONCE_SIZE)
    padded = data + bytes([pad_len]) * pad_len
    # 4) CTR-mode encryption
    ciphertext = bytearray()
    for counter in range(len(padded) // NONCE_SIZE):
        ctr = counter.to_bytes(NONCE_SIZE, 'big')
        keystream = hashlib.sha256(seed + ctr).digest()
        block = padded[counter*NONCE_SIZE:(counter+1)*NONCE_SIZE]
        ciphertext.extend(b ^ k for b, k in zip(block, keystream))
    # 5) Tag = SHA256(seed || ciphertext)
    tag = hashlib.sha256(seed + ciphertext).digest()
    return nonce + bytes(ciphertext) + tag

def decrypt_bytes(blob: bytes, key: Optional[bytes] = None) -> bytes:
    """
    Decrypt data from encrypt_bytes; verify tag. Returns the plaintext.
    Raises ValueError on authentication or padding failure.
    """
    if len(blob) < NONCE_SIZE + TAG_SIZE:
        raise ValueError("Ciphertext too short")
    if key is None:
        key = derive_key(load_git_crypt_key())

    nonce = blob[:NONCE_SIZE]
    tag    = blob[-TAG_SIZE:]
    ct     = blob[NONCE_SIZE:-TAG_SIZE]
    seed = hashlib.sha256(key + nonce).digest()
    # Verify tag in constant time
    expected = hashlib.sha256(seed + ct).digest()
    if not hmac.compare_digest(expected, tag):
        raise ValueError("Authentication failed")

    # Decrypt CTR
    padded = bytearray()
    for counter in range(len(ct) // NONCE_SIZE):
        ctr = counter.to_bytes(NONCE_SIZE, 'big')
        keystream = hashlib.sha256(seed + ctr).digest()
        block = ct[counter*NONCE_SIZE:(counter+1)*NONCE_SIZE]
        padded.extend(b ^ k for b, k in zip(block, keystream))

    # Remove PKCS#7 padding
    pad_len = padded[-1]
    if not 1 <= pad_len <= NONCE_SIZE:
        raise ValueError("Invalid padding")
    return bytes(padded[:-pad_len])

# -----------------------------------------------------------------------------
# 4) String & JSON wrappers
# -----------------------------------------------------------------------------
def encrypt_str(s: str, key: Optional[bytes] = None) -> bytes:
    return encrypt_bytes(s.encode("utf-8"), key)

def decrypt_str(blob: bytes, key: Optional[bytes] = None) -> str:
    return decrypt_bytes(blob, key).decode("utf-8")

def encrypt_json(obj: Any, key: Optional[bytes] = None, **dump_kwargs) -> bytes:
    raw = json.dumps(obj, **dump_kwargs).encode("utf-8")
    return encrypt_bytes(raw, key)

def decrypt_json(blob: bytes, key: Optional[bytes] = None) -> Any:
    raw = decrypt_bytes(blob, key)
    return json.loads(raw.decode("utf-8"))

def main():
    parser = argparse.ArgumentParser(description="Encrypt/decrypt files")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_enc = sub.add_parser("encrypt", help="Encrypt a file")
    p_enc.add_argument("-k","--key", help="Key base 16 (omit for git-crypt key)")
    p_enc.add_argument("-t","--text", required=True, help="Plaintext")

    p_dec = sub.add_parser("decrypt", help="Decrypt a file")
    p_dec.add_argument("-k","--key", help="Key base 16 (omit for git-crypt key)")
    p_dec.add_argument("-c","--cipher", required=True, help="Ciphertext base 16 input")

    args = parser.parse_args()
    key = args.key

    if args.cmd == "encrypt":
        data = args.text
        blob = encrypt_str(data, key)
        print(base64.b64encode(blob).decode('utf-8'))
    else:
        blob = base64.b64decode(args.cipher)
        pt = decrypt_str(blob, key)
        print(pt)

if __name__ == "__main__":
    main()