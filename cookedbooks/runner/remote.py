"""Judge side of the process boundary: start one policy in a restricted child and talk to it.

`PolicyProcess` copies `cookedbooks/policies` (only that package) into a
fresh temporary folder, creates a second, empty temporary folder as the
working directory, and starts `cookedbooks/policies/worker.py` from the copy
with `python -I -S -B`, a minimal environment, stdin/stdout pipes and stderr
discarded. See `cookedbooks.policies.worker` for what the child may do.

Every read from the child has a deadline, a line-length limit and a JSON
check; a write never blocks the runner (a writer thread does it). A child
that crashes, hangs, exits early or sends garbage turns into a `PolicyFailure`
and is killed with its whole process group, so the runner never hangs.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

POLICIES_DIR = Path(__file__).resolve().parents[1] / "policies"
MAX_LINE_BYTES = 4 * 1024 * 1024   # one action; the largest legal submission is far smaller
STARTUP_TIMEOUT_S = 15.0           # import + make_policy, before the episode clock starts
SHUTDOWN_TIMEOUT_S = 2.0
NOT_POLICIES = ("__init__", "base", "worker")

_EOF = object()
_TOO_LONG = object()


class PolicyFailure(Exception):
    """The child broke the protocol; `kind` is one of crash, timeout, garbage, exited, refused."""

    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind
        self.detail = detail


def _reject_constant(token: str) -> Any:
    """NaN and Infinity are not JSON."""
    raise ValueError(f"not JSON: {token}")


def available_policies() -> list[str]:
    """Policy module names present in `cookedbooks/policies`, sorted."""
    return sorted(p.stem for p in POLICIES_DIR.glob("*.py") if p.stem not in NOT_POLICIES)


def _copy_package(target: Path, extra_sources: Mapping[str, str]) -> Path:
    """Copy the policies package (.py files only) to target/lib; return the worker script path."""
    lib = target / "lib"
    package = lib / "cookedbooks" / "policies"
    package.mkdir(parents=True)
    (lib / "cookedbooks" / "__init__.py").write_text("", encoding="utf-8")
    for source in sorted(POLICIES_DIR.rglob("*.py")):
        relative = source.relative_to(POLICIES_DIR)
        if "__pycache__" in relative.parts:
            continue
        (package / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, package / relative)
    for name, text in sorted(extra_sources.items()):
        (package / f"{name}.py").write_text(text, encoding="utf-8")
    return package / "worker.py"


def _child_env() -> dict[str, str]:
    """A minimal environment: no HOME, no paths of the repository, no PYTHON* variables."""
    return {"PATH": os.defpath, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"}


@dataclass
class _Line:
    data: Any  # bytes, _EOF or _TOO_LONG


class PolicyProcess:
    """One policy child. Use as a context manager; `close()` always kills and cleans up."""

    def __init__(self, name: str, seed: int, options: Mapping[str, Any] | None = None, *,
                 debug: bool = False, extra_sources: Mapping[str, str] | None = None) -> None:
        extra = dict(extra_sources or {})
        if name not in available_policies() and name not in extra:
            raise ValueError(f"unknown policy {name!r}")
        self.name = name
        self.seed = seed
        self.options = dict(options or {})
        self.debug = debug
        self._sandbox = Path(tempfile.mkdtemp(prefix="cookedbooks-policy-lib-"))
        self._work = Path(tempfile.mkdtemp(prefix="cookedbooks-policy-cwd-"))
        self._script = _copy_package(self._sandbox, extra)
        self._lines: queue.Queue[_Line] = queue.Queue()
        self._outbox: queue.Queue[bytes | None] = queue.Queue()
        self._proc: subprocess.Popen | None = None
        self.exit_code: int | None = None

    @property
    def work_dir(self) -> str:
        """The child's working directory (judge side knowledge, for tests)."""
        return str(self._work)

    # --- lifecycle ----------------------------------------------------------------------

    def __enter__(self) -> PolicyProcess:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def start(self) -> None:
        """Spawn the child and wait until the policy is built (raises PolicyFailure)."""
        self._proc = subprocess.Popen(
            [sys.executable, "-I", "-S", "-B", str(self._script)],
            cwd=str(self._work), env=_child_env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True,
        )
        threading.Thread(target=self._read_loop, daemon=True).start()
        threading.Thread(target=self._write_loop, daemon=True).start()
        self.send({"type": "start", "policy": self.name, "seed": self.seed, "options": self.options,
                   "debug": self.debug})
        reply = self.receive(time.monotonic() + STARTUP_TIMEOUT_S)
        if reply.get("type") != "ready":
            raise PolicyFailure("garbage", "expected ready")

    def close(self) -> None:
        """Stop the child (and anything it started) and remove both temporary folders."""
        proc = self._proc
        if proc is not None:
            self._outbox.put(None)
            try:
                proc.wait(timeout=SHUTDOWN_TIMEOUT_S if proc.poll() is None else 0)
            except subprocess.TimeoutExpired:
                pass
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()
                proc.wait()
            self.exit_code = proc.returncode
            for stream in (proc.stdout, proc.stdin):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
            self._proc = None
        shutil.rmtree(self._sandbox, ignore_errors=True)
        shutil.rmtree(self._work, ignore_errors=True)

    def kill(self) -> None:
        """Kill the child's process group now."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                os.killpg(self._proc.pid, signal.SIGKILL)
            except OSError:
                self._proc.kill()

    # --- messages -----------------------------------------------------------------------

    def send(self, message: dict) -> None:
        """Queue one message; never blocks."""
        data = json.dumps(message, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        self._outbox.put(data.encode("utf-8") + b"\n")

    def receive(self, deadline: float) -> dict:
        """The next message from the child, or PolicyFailure (the child is then killed)."""
        try:
            line = self._lines.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            self.kill()
            raise PolicyFailure("timeout") from None
        if line.data is _EOF:
            self.kill()
            raise PolicyFailure("exited")
        if line.data is _TOO_LONG:
            self.kill()
            raise PolicyFailure("garbage", "line too long")
        try:
            message = json.loads(line.data.decode("utf-8"), parse_constant=_reject_constant)
        except (UnicodeDecodeError, ValueError):
            self.kill()
            raise PolicyFailure("garbage", "not JSON") from None
        if not isinstance(message, dict):
            self.kill()
            raise PolicyFailure("garbage", "not an object")
        if message.get("type") == "error":
            self.kill()
            error = message.get("error")
            raise PolicyFailure("crash", error if isinstance(error, str) else "")
        return message

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        stream = self._proc.stdout
        try:
            while True:
                line = stream.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_LINE_BYTES or not line.endswith(b"\n"):
                    self._lines.put(_Line(_TOO_LONG if len(line) > MAX_LINE_BYTES else line))
                    if len(line) > MAX_LINE_BYTES:
                        return
                    continue
                self._lines.put(_Line(line))
        except (OSError, ValueError):
            pass
        self._lines.put(_Line(_EOF))

    def _write_loop(self) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        stream = self._proc.stdin
        while True:
            data = self._outbox.get()
            try:
                if data is None:
                    stream.close()
                    return
                stream.write(data)
                stream.flush()
            except (OSError, ValueError):
                return
