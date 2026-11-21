"""The policy side of the process boundary: one policy in one restricted child process.

The judge-side runner (`cookedbooks.runner.remote`) copies this package, and
only this package, into a fresh temporary folder and starts

    python -I -S -B <copy>/cookedbooks/policies/worker.py

with an empty temporary working directory, a minimal environment and pipes
for stdin and stdout. The child never gets a path, the episode's seed,
truth, the scorer or rewards beyond what the contract gives the policy.

Restrictions, set up before the policy module is imported:
- `-I -S`: no PYTHON* variables, no user or site packages, no script folder
  or current folder on `sys.path`; only the standard library and the copy of
  `cookedbooks.policies` can be imported. The grader, the generator, the
  verifier, the loader and the environment are not on disk next to it.
- An audit hook (it cannot be removed) refuses: importing any `cookedbooks`
  module outside `cookedbooks.policies` or `judge_adapter`; opening or
  listing files outside the standard library, the package copy and the
  working directory; writing anywhere but the working directory; processes,
  signals, sockets, `ctypes` and changing directory.
- Resource limits on CPU time, file size, open files and core dumps.
- The protocol uses private duplicates of the original stdin/stdout file
  descriptors; fd 0 and fd 1 are redirected, so a policy that prints cannot
  corrupt the protocol by accident.

Protocol: JSON, one object per line, UTF-8.

    runner -> child  {"type": "start", "policy": name, "seed": int, "options": {...}, "debug": bool}
    child  -> runner {"type": "ready"}
    runner -> child  {"type": "observation", "observation": {...}, "reward": number}
    child  -> runner {"type": "action", "action": {...}}
    ...
    runner -> child  {"type": "end", "observation": {...}, "reward": number, "terminated": bool, "truncated": bool}
    child  -> runner {"type": "debug", ...}   (only when "debug" was true)

On any failure the child sends {"type": "error", "error": "<exception type>"}
and exits with status 1. The debug message (tests only) lists the modules
loaded at exit, the working directory and its entries, and every line the
child received.

Standard library only; imports nothing from `cookedbooks`.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
from typing import Any, BinaryIO

FORBIDDEN_PREFIXES = ("cookedbooks.", "judge_adapter")
ALLOWED_PREFIX = "cookedbooks.policies"
DENIED_EVENTS = (
    "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
    "os.kill", "os.killpg", "os.chdir", "os.fchdir", "os.putenv", "os.unsetenv", "pty.spawn",
    "webbrowser.open", "urllib.Request", "ftplib.connect", "smtplib.connect", "http.client.connect",
)
DENIED_EVENT_PREFIXES = ("socket.", "ctypes.", "shutil.", "winreg.", "msvcrt.", "_winapi.")
PATH_WRITE_EVENTS = (
    "os.remove", "os.rmdir", "os.rename", "os.mkdir", "os.chmod", "os.chown", "os.symlink", "os.link",
    "os.truncate", "os.utime", "os.chflags", "os.lchflags", "os.lchmod", "os.setxattr", "os.removexattr",
)
PATH_READ_EVENTS = ("os.listdir", "os.scandir", "glob.glob", "os.listxattr", "os.getxattr")
POLICY_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")
NOT_POLICIES = ("base", "worker")
MAX_LINE_BYTES = 16 * 1024 * 1024


class Refused(PermissionError):
    """Raised by the audit hook for anything outside the policy's sandbox."""


def _real(path: Any) -> str | None:
    if isinstance(path, int):
        return None
    try:
        text = os.fsdecode(os.fspath(path))
    except TypeError:
        return ""
    return os.path.realpath(os.path.abspath(text))


def _under(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == root or path.startswith(root.rstrip(os.sep) + os.sep) for root in roots)


def install_guard(read_roots: tuple[str, ...], write_root: str) -> None:
    """Install the audit hook that confines everything after this call."""
    read_roots = tuple(sorted({os.path.realpath(r) for r in read_roots if r}))
    write_roots = (os.path.realpath(write_root),)
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def hook(event: str, args: tuple) -> None:
        if event == "import":
            name = args[0] if args else ""
            if isinstance(name, str) and name.startswith(FORBIDDEN_PREFIXES) \
                    and not (name == ALLOWED_PREFIX or name.startswith(ALLOWED_PREFIX + ".")):
                raise Refused("import refused")
            if name in ("ctypes", "_ctypes"):
                raise Refused("import refused")
            return
        if event == "open":
            path = _real(args[0]) if args else ""
            if path is None:  # an already open file descriptor
                return
            mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ""
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            writing = any(c in mode for c in "wax+") or bool(flags & write_flags)
            if not _under(path, write_roots if writing else read_roots + write_roots):
                raise Refused("open refused")
            return
        if event in PATH_READ_EVENTS:
            path = _real(args[0]) if args and args[0] is not None else _real(".")
            if path is not None and not _under(path, read_roots + write_roots):
                raise Refused("listing refused")
            return
        if event in PATH_WRITE_EVENTS:
            for arg in args[:2]:
                if isinstance(arg, (str, bytes, os.PathLike)):
                    path = _real(arg)
                    if path is not None and not _under(path, write_roots):
                        raise Refused("write refused")
            return
        if event in DENIED_EVENTS or event.startswith(DENIED_EVENT_PREFIXES):
            raise Refused("refused")

    sys.addaudithook(hook)


def set_limits() -> None:
    """Best-effort resource limits (POSIX only; a limit the OS refuses is skipped)."""
    try:
        import resource
    except ImportError:  # pragma: no cover - not POSIX
        return
    limits = [("RLIMIT_CPU", 120), ("RLIMIT_FSIZE", 1 << 20), ("RLIMIT_CORE", 0), ("RLIMIT_NOFILE", 64)]
    if sys.platform.startswith("linux"):
        limits.append(("RLIMIT_AS", 2 << 30))
    for name, value in limits:
        which = getattr(resource, name, None)
        if which is None:
            continue
        try:
            soft, hard = resource.getrlimit(which)
            value = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(which, (value, value))
        except (ValueError, OSError):
            pass


class Channel:
    """The JSON-lines protocol over private file descriptors."""

    def __init__(self, reader: BinaryIO, writer: BinaryIO, *, keep_log: bool = False) -> None:
        self._reader = reader
        self._writer = writer
        self.log: list[str] | None = [] if keep_log else None

    def receive(self) -> dict | None:
        line = self._reader.readline(MAX_LINE_BYTES)
        if not line:
            return None
        text = line.decode("utf-8")
        if self.log is not None:
            self.log.append(text.rstrip("\n"))
        message = json.loads(text)
        if not isinstance(message, dict):
            raise ValueError("message is not an object")
        return message

    def send(self, message: dict) -> None:
        data = json.dumps(message, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        self._writer.write(data.encode("utf-8") + b"\n")
        self._writer.flush()


def load_policy(name: Any, seed: Any, options: Any) -> Any:
    """Import `cookedbooks.policies.<name>` and call its `make_policy(seed, **options)`."""
    if not isinstance(name, str) or not POLICY_NAME.match(name) or name in NOT_POLICIES:
        raise ValueError("bad policy name")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bad seed")
    module = importlib.import_module(ALLOWED_PREFIX + "." + name)
    return module.make_policy(seed, **(options if isinstance(options, dict) else {}))


def serve(channel: Channel) -> int:
    """Run the protocol until the episode ends or the runner closes stdin. Returns the exit status."""
    start = channel.receive()
    if start is None:
        return 0
    if start.get("type") != "start":
        raise ValueError("expected a start message")
    debug = start.get("debug") is True
    if debug and channel.log is None:
        channel.log = [json.dumps(start, ensure_ascii=False, separators=(",", ":"))]
    policy = load_policy(start.get("policy"), start.get("seed"), start.get("options"))
    channel.send({"type": "ready"})
    while True:
        message = channel.receive()
        if message is None:
            return 0
        kind = message.get("type")
        if kind == "observation":
            action = policy.act(message.get("observation"))
            channel.send({"type": "action", "action": action})
        elif kind == "end":
            if debug:
                channel.send(debug_report(channel.log or []))
            return 0
        else:
            raise ValueError("unknown message type")


def debug_report(received: list[str]) -> dict:
    """What a test needs to check the boundary: modules, working directory, everything received."""
    cwd = os.getcwd()
    return {
        "type": "debug",
        "modules": sorted(sys.modules),
        "cwd": cwd,
        "cwd_entries": sorted(os.listdir(cwd)),
        "received": received,
    }


def main() -> int:
    """Entry point of the child process (run as a script from the sandbox copy)."""
    lib = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    stdlib_roots = tuple(p for p in sys.path if p)
    work = os.getcwd()
    # Private protocol descriptors; fd 0 reads nothing and fd 1 goes to stderr.
    reader = os.fdopen(os.dup(0), "rb")
    writer = os.fdopen(os.dup(1), "wb")
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    os.dup2(2, 1)
    sys.stdin = open(os.devnull, encoding="utf-8")
    sys.stdout = sys.stderr
    sys.path[:] = [lib, *stdlib_roots]
    set_limits()
    install_guard(stdlib_roots + (lib,), work)
    channel = Channel(reader, writer)
    try:
        return serve(channel)
    except BaseException as exc:  # report the failure type only, never a traceback
        try:
            channel.send({"type": "error", "error": type(exc).__name__})
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
