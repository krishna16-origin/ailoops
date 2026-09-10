"""
Live E2B sandbox manager for ailoops Code mode.

Gives each chat session its own cloud sandbox (E2B) that the generated
project is synced into, so it can actually be *run* (npm install, pip
install, dev servers, tests, arbitrary shell commands) instead of only
being statically previewed in an iframe. Once a server is started inside
the sandbox, its public URL is returned so the frontend can point a real
iframe at a live, running app -- the "Manus-style" live computer/preview.

Nothing here talks to FastAPI directly; app.py wires this into the
/sandbox/* endpoints and into the Code-mode agent loop's run_command /
start_server tool actions.
"""
import os
import time
import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from e2b import AsyncSandbox, CommandExitException

PROJECT_DIR = "/home/user/project"

# How long an idle sandbox is allowed to live before E2B auto-kills it. Reset
# on every sync/command/server call so an actively-used sandbox never expires
# mid-session.
SANDBOX_IDLE_TIMEOUT_SECONDS = int(os.getenv("E2B_SANDBOX_TIMEOUT", "1800"))
COMMAND_TIMEOUT_SECONDS = float(os.getenv("E2B_COMMAND_TIMEOUT", "300"))
SERVER_READY_TIMEOUT_SECONDS = float(os.getenv("E2B_SERVER_READY_TIMEOUT", "45"))

LogCallback = Callable[[str, str], Awaitable[None]]  # (stream, text) -> None


class SandboxNotConfigured(RuntimeError):
    """Raised whenever E2B_API_KEY is missing/blank."""


def sandbox_configured() -> bool:
    key = (os.getenv("E2B_API_KEY") or "").strip()
    return bool(key) and key.lower() not in ("demo", "none", "changeme")


def _require_configured() -> None:
    if not sandbox_configured():
        raise SandboxNotConfigured(
            "E2B_API_KEY is not set on the server. Add it to your .env "
            "(E2B_API_KEY=e2b_...) and restart the backend to enable the live sandbox."
        )


@dataclass
class _SandboxSession:
    sandbox: AsyncSandbox
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    file_hashes: Dict[str, str] = field(default_factory=dict)
    server_handle: Optional[Any] = None
    server_port: Optional[int] = None
    server_url: Optional[str] = None
    last_active: float = field(default_factory=time.time)


# In-memory registry, keyed by the app's own chat session_id. Matches the
# rest of this codebase (the `sessions` dict in app.py is also in-memory).
_sessions: Dict[str, _SandboxSession] = {}


async def _create_sandbox() -> AsyncSandbox:
    _require_configured()
    sbx = await AsyncSandbox.create(
        timeout=SANDBOX_IDLE_TIMEOUT_SECONDS,
        api_key=os.getenv("E2B_API_KEY"),
    )
    await sbx.commands.run(f"mkdir -p {PROJECT_DIR}", timeout=15)
    return sbx


async def ensure_session(session_id: str) -> _SandboxSession:
    """Get this chat session's sandbox, creating (or replacing a dead) one."""
    _require_configured()
    state = _sessions.get(session_id)
    if state is not None:
        try:
            if await state.sandbox.is_running():
                await state.sandbox.set_timeout(SANDBOX_IDLE_TIMEOUT_SECONDS)
                state.last_active = time.time()
                return state
        except Exception:
            pass
        _sessions.pop(session_id, None)  # stale/dead — recreate below

    sbx = await _create_sandbox()
    state = _SandboxSession(sandbox=sbx)
    _sessions[session_id] = state
    return state


def get_session(session_id: str) -> Optional[_SandboxSession]:
    return _sessions.get(session_id)


async def stop_session(session_id: str) -> bool:
    state = _sessions.pop(session_id, None)
    if state is None:
        return False
    if state.server_handle is not None:
        try:
            await state.server_handle.kill()
        except Exception:
            pass
    try:
        await state.sandbox.kill()
    except Exception:
        pass
    return True


def _hash(content: str) -> str:
    return hashlib.sha1((content or "").encode("utf-8", errors="ignore")).hexdigest()


async def sync_files(session_id: str, file_store: Dict[str, str]) -> List[str]:
    """Write only new/changed files into the sandbox project dir and remove
    ones no longer in file_store. Returns the list of paths actually written."""
    state = await ensure_session(session_id)
    async with state.lock:
        to_write: List[Tuple[str, str, str]] = []
        for rel_path, content in file_store.items():
            h = _hash(content)
            if state.file_hashes.get(rel_path) != h:
                to_write.append((rel_path, content, h))
        removed = [p for p in state.file_hashes if p not in file_store]

        if to_write:
            dirnames = sorted({
                os.path.dirname(f"{PROJECT_DIR}/{p}")
                for p, _, _ in to_write if os.path.dirname(p)
            })
            if dirnames:
                mkdir_cmd = "mkdir -p " + " ".join(f'"{d}"' for d in dirnames)
                await state.sandbox.commands.run(mkdir_cmd, timeout=20)
            payload = [{"path": f"{PROJECT_DIR}/{p}", "data": content} for p, content, _ in to_write]
            await state.sandbox.files.write_files(payload)
            for p, _, h in to_write:
                state.file_hashes[p] = h

        for p in removed:
            try:
                await state.sandbox.files.remove(f"{PROJECT_DIR}/{p}")
            except Exception:
                pass
            state.file_hashes.pop(p, None)

        await state.sandbox.set_timeout(SANDBOX_IDLE_TIMEOUT_SECONDS)
        state.last_active = time.time()
        return [p for p, _, _ in to_write]


async def run_command(
    session_id: str,
    command: str,
    on_output: Optional[LogCallback] = None,
    cwd: Optional[str] = None,
    timeout: float = COMMAND_TIMEOUT_SECONDS,
) -> Tuple[int, str]:
    """Run a command to completion inside the sandbox, streaming stdout/stderr
    live via on_output(stream, text). Returns (exit_code, combined_output)."""
    state = await ensure_session(session_id)
    combined: List[str] = []

    async def _stdout(data: str):
        combined.append(data)
        if on_output:
            await on_output("stdout", data)

    async def _stderr(data: str):
        combined.append(data)
        if on_output:
            await on_output("stderr", data)

    try:
        result = await state.sandbox.commands.run(
            command,
            cwd=cwd or PROJECT_DIR,
            timeout=timeout,
            on_stdout=_stdout,
            on_stderr=_stderr,
        )
        exit_code = result.exit_code
    except CommandExitException as exc:
        exit_code = exc.exit_code
    except Exception as exc:
        message = f"\n[sandbox error] {exc}\n"
        if on_output:
            await on_output("stderr", message)
        return 1, "".join(combined) + message

    state.last_active = time.time()
    return exit_code, "".join(combined)


async def start_server(
    session_id: str,
    command: str,
    port: int,
    on_output: Optional[LogCallback] = None,
    cwd: Optional[str] = None,
) -> str:
    """Start (or restart) a long-running server on `port` in the sandbox,
    wait briefly for it to come up, and return its public HTTPS URL."""
    state = await ensure_session(session_id)
    async with state.lock:
        if state.server_handle is not None:
            try:
                await state.server_handle.kill()
            except Exception:
                pass
            state.server_handle = None
            state.server_url = None

        async def _stdout(data: str):
            if on_output:
                await on_output("stdout", data)

        async def _stderr(data: str):
            if on_output:
                await on_output("stderr", data)

        handle = await state.sandbox.commands.run(
            command,
            cwd=cwd or PROJECT_DIR,
            background=True,
            on_stdout=_stdout,
            on_stderr=_stderr,
        )
        state.server_handle = handle
        state.server_port = port

        deadline = time.time() + SERVER_READY_TIMEOUT_SECONDS
        ready = False
        while time.time() < deadline:
            if handle.exit_code is not None:
                break  # process already exited
            try:
                check = await state.sandbox.commands.run(
                    f"curl -s -o /dev/null -w '%{{http_code}}' --max-time 2 http://localhost:{port}",
                    timeout=5,
                )
                code = (check.stdout or "").strip()
                if code and code != "000":
                    ready = True
                    break
            except Exception:
                pass
            await asyncio.sleep(1.5)

        if not ready and handle.exit_code is not None:
            raise RuntimeError(
                f"Server exited (code {handle.exit_code}) before it started listening on port {port}."
            )

        host = state.sandbox.get_host(port)
        url = f"https://{host}"
        state.server_url = url
        await state.sandbox.set_timeout(SANDBOX_IDLE_TIMEOUT_SECONDS)
        state.last_active = time.time()
        return url


def detect_default_start(file_store: Dict[str, str]) -> Tuple[str, int]:
    """Best-effort guess at how to serve the current project when the caller
    (or the agent) doesn't specify a command/port explicitly."""
    names = set(file_store.keys())

    if "package.json" in names:
        import json as _json
        try:
            pkg = _json.loads(file_store["package.json"])
        except Exception:
            pkg = {}
        scripts = (pkg or {}).get("scripts", {}) or {}
        deps = {**(pkg.get("dependencies") or {}), **(pkg.get("devDependencies") or {})}
        install = "npm install --no-audit --no-fund"
        if "dev" in scripts:
            port = 5173 if "vite" in deps or "vite" in scripts.get("dev", "") else 3000
            return (f"{install} && npm run dev -- --host 0.0.0.0 --port {port}", port)
        if "start" in scripts:
            port = 3000
            return (f"{install} && PORT={port} npm start -- --host 0.0.0.0 --port {port}", port)
        port = 3000
        return (f"{install} && npx --yes serve -l {port}", port)

    py_entry = next((n for n in ("app.py", "main.py", "server.py") if n in names), None)
    if py_entry:
        port = 8000
        install = "pip install -r requirements.txt --break-system-packages --quiet && " if "requirements.txt" in names else ""
        return (f"{install}PORT={port} python3 {py_entry}", port)

    # Static site fallback — matches this app's current default output (HTML/CSS/JS).
    port = 8080
    return (f"python3 -m http.server {port} --bind 0.0.0.0", port)
