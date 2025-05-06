import sys,os,re,json,base64
import cipher

from typing import *

# decrypt('base64blob') or decrypt('base64blob','ENV_VAR_NAME')
_DECRYPT_RE = re.compile(
    r"""^decrypt\(\s*'([A-Za-z0-9+/=]+)'\s*
         (?:,\s*'([A-Za-z_][A-Za-z0-9_]*)'\s* )?
       \)$""",
    re.VERBOSE
)

def lookup(path: str, values: Dict[str, Any]) -> Any:
    # 1) Drill down by dot-path
    v: Any = values
    for part in path.split('.'):
        if not isinstance(v, dict) or part not in v:
            raise KeyError(f"Unknown placeholder: {path}")
        v = v[part]

    # 2) Only strings can decrypt/JSON-parse
    if isinstance(v, str):
        s = v.strip()

        # ─── unwrap ${…} if present ─────────────────────────────
        if s.startswith("${") and s.endswith("}"):
            s = s[2:-1].strip()

        # ─── decrypt case? ───────────────────────────────────────
        m = _DECRYPT_RE.match(s)
        if m:
            b64_blob, env_var = m.group(1), m.group(2)
            blob = base64.b64decode(b64_blob)

            if env_var:
                if env_var not in os.environ:
                    raise KeyError(f"Environment variable {env_var!r} not set for decrypt key")
                key_b64 = os.environ[env_var]
                try:
                    key = base64.b64decode(key_b64)
                except Exception as e:
                    raise ValueError(f"Invalid Base64 in env var {env_var!r}: {e}")
            else:
                key = None

            return cipher.decrypt_str(blob, key)

        # ─── JSON object/array literal? ─────────────────────────
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in placeholder '{path}': {e}")

    # 3) fallback
    return v

# Precompile the placeholder pattern for ${key} or ${key.subkey}
_PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")

def interpolate(
    s: str,
    values: Dict[str, Any],
    seen: Optional[Set[str]] = None
) -> str:
    """
    Recursively interpolate placeholders in `s` of the form `${key}` or `${key.subkey}`
    using nested lookups in `values`. Detects cycles.

    Args:
        s: the input string containing placeholders
        values: dict of values for interpolation (may be nested)
        seen: internal set of keys currently in progress (for cycle detection)

    Returns:
        A new string with all placeholders replaced.

    Raises:
        KeyError:   if a placeholder path is not found in `values`
        ValueError: if a cycle is detected (e.g. A → B → A)
    """
    if seen is None:
        seen = set()

    def _replacer(match: re.Match) -> str:
        key = match.group(1)
        if key in seen:
            raise ValueError(f"Cycle detected on key: {key}")
        raw = lookup(key,values)
        seen.add(key)
        # Convert non-string to str and recurse
        result = interpolate(str(raw), values, seen)
        seen.remove(key)
        return result

    return _PLACEHOLDER_RE.sub(_replacer, s)


class Args:
    """
    Command-line style argument parser supporting nested dict/list paths,
    JSON-based values, file includes, and interpolation.
    """
    def __init__(self):
        # Internal storage for parsed arguments
        self._args: Dict[str, Any] = {}

    @staticmethod
    def parse_arg(arg: str) -> Tuple[List[Union[str,int]], Any]:
        """
        Parse a string like "x.y.3=true" into (path, value):
          - path: ["x","y",3]
          - value: parsed via json.loads (with fallback to raw string)
        If no "=", value defaults to True.
        """
        eq = arg.find('=')
        if eq >= 0:
            name = arg[:eq].strip()
            rhs = arg[eq+1:].strip()
            try:
                value = json.loads(rhs)
            except json.JSONDecodeError:
                value = rhs
        else:
            name = arg.strip()
            value = True

        path: List[Union[str,int]] = []
        for part in name.split('.'):
            if part.isdigit():
                path.append(int(part))
            else:
                path.append(part)
        return path, value

    def set_arg(self, path: List[Union[str,int]], value: Any) -> None:
        """
        Ensure self._args[path[0]][path[1]]... exists (creating dicts/lists as needed)
        and set the final slot to `value`.
        """
        node: Any = self._args
        for i, key in enumerate(path):
            last = (i == len(path) - 1)
            subpath = ".".join(str(p) for p in path[:i]) or 'root'

            if isinstance(key, int):  # list index
                if not isinstance(node, list):
                    raise TypeError(f"args[{subpath}] is not a list.")
                if key >= len(node):
                    node.extend([None] * (key + 1 - len(node)))
                if last:
                    node[key] = value
                else:
                    child = node[key]
                    next_key = path[i+1]
                    if not isinstance(child, (dict, list)):
                        node[key] = {} if isinstance(next_key, str) else []
                    node = node[key]
            else:  # dict key
                if not isinstance(node, dict):
                    raise TypeError(f"args[{subpath}] is not a dict.")
                if last:
                    node[key] = value
                else:
                    child = node.get(key)
                    next_key = path[i+1]
                    if not isinstance(child, (dict, list)):
                        node[key] = {} if isinstance(next_key, str) else []
                    node = node[key]

    def inc(self, filename: str, strongly: bool = False) -> None:
        """
        Merge settings from a JSON file.
        If `strongly` is True, file overrides current; else current overrides file.
        """
        with open(filename, 'r', encoding='utf-8') as f:
            file_settings = json.load(f)
        if strongly:
            self._args = {**self._args, **file_settings}
        else:
            self._args = {**file_settings, **self._args}

    @classmethod
    def from_cli(
        cls,
        cfg_path: str = "private/cfg.json",
        cli_args: Optional[List[str]] = None
    ) -> 'Args':
        """
        Create an Args instance from the command line.
        Supports tokens:
          - x.y.3=val      → nested set via parse_arg / set_arg
          - inc!(file).    → include file, strong override
          - inc(file).     → include file, weak override
        Remaining include is applied at end if no inline inc(...).
        """
        if cli_args is None:
            cli_args = sys.argv[1:]
        ans = cls()
        pending_cfg = cfg_path
        inc_pattern = re.compile(r'^\s*inc(!?)\(([^)]+)\)\s*$')
        for token in cli_args:
            m = inc_pattern.match(token)
            if m:
                strong = (m.group(1) == '!')
                filename = m.group(2)
                ans.inc(filename, strong)
                pending_cfg = None
            else:
                path, val = cls.parse_arg(token)
                ans.set_arg(path, val)
        if pending_cfg is not None:
            ans.inc(pending_cfg, False)
        return ans

    @property
    def args(self) -> Dict[str, Any]:
        """Return the internal args dictionary."""
        return self._args

    def val(
        self,
        path: List[Union[str,int]],
        missing: Union[Any, Callable[[], Any]] = None
    ) -> Any:
        """
        Traverse nested dict/list by `path`. If lookup fails, return `missing` or missing().
        If result is str, run interpolation against full args dict.
        """
        node: Any = self._args
        try:
            for part in path:
                node = node[part]
        except (KeyError, IndexError, TypeError):
            return missing() if callable(missing) else missing
        if isinstance(node, str):
            return interpolate(node, self._args)
        return node


def args(
    cfg_path: str = "private/cfg.json",
    cli_args: Optional[List[str]] = None
) -> Args:
    """
    Helper: parse CLI into a flat dict of settings.
    """
    return Args.from_cli(cfg_path, cli_args)
