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
import shlex
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
    # Best-effort: make sure the basics for "download something / extract an
    # archive / test something" work even on a minimal base image, without
    # slowing down sandboxes that already have them. Failures here (no apt,
    # no root, offline mirror, etc.) are swallowed — run_command will simply
    # surface a real "command not found" to the model/user if a tool truly
    # isn't available, same as any other command.
    try:
        await sbx.commands.run(
            "command -v curl >/dev/null && command -v wget >/dev/null && "
            "command -v unzip >/dev/null && command -v zip >/dev/null && "
            "command -v tar >/dev/null && command -v gzip >/dev/null && "
            "command -v git >/dev/null && command -v jq >/dev/null || "
            "(apt-get update -qq && apt-get install -y -qq "
            "curl wget unzip zip tar gzip git jq ca-certificates file >/dev/null 2>&1)",
            timeout=60,
        )
    except Exception:
        pass
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


# Extensions we never pull back into file_store as text: binary content decoded
# "successfully" as UTF-8 is usually garbage, and file_store/the preview canvas
# are text-oriented. Real command output already reports what a download/unzip
# produced, so the person isn't losing visibility — just the raw bytes.
_BINARY_SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".svgz",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".pdf",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".mp3", ".mp4", ".mov", ".avi", ".wav", ".ogg",
    ".wasm", ".so", ".pyc", ".class", ".jar", ".exe", ".bin", ".db", ".sqlite",
}
_SNAPSHOT_EXCLUDE_DIRS = ("node_modules", ".git", "__pycache__", "venv", ".venv", "dist", "build", ".next")


async def snapshot_project_files(session_id: str) -> Dict[str, str]:
    """Read back every reasonably-small text file currently under PROJECT_DIR
    inside the sandbox. Used after run_command/start_server so files a real
    command downloaded (curl/wget), extracted (unzip/tar), generated (a build
    step), or otherwise changed on disk become real project files the agent
    can read_file/edit_file and that show up in the final result — instead of
    only ever existing as a side effect inside the sandbox that then vanishes
    with it. Also refreshes this session's file_hashes so the next sync_files()
    call doesn't immediately try to overwrite what it just read back."""
    state = get_session(session_id)
    if state is None:
        return {}
    exclude_clause = " ".join(f"-not -path '*/{d}/*'" for d in _SNAPSHOT_EXCLUDE_DIRS)
    find_cmd = f"find {PROJECT_DIR} -type f {exclude_clause} -size -2M 2>/dev/null"
    try:
        result = await state.sandbox.commands.run(find_cmd, timeout=20)
        abs_paths = [p.strip() for p in (result.stdout or "").splitlines() if p.strip()]
    except Exception:
        return {}

    files: Dict[str, str] = {}
    for abs_path in abs_paths:
        if not abs_path.startswith(PROJECT_DIR + "/"):
            continue
        rel_path = abs_path[len(PROJECT_DIR) + 1:]
        ext = ("." + rel_path.rsplit(".", 1)[-1].lower()) if "." in rel_path.rsplit("/", 1)[-1] else ""
        if ext in _BINARY_SKIP_EXTENSIONS:
            continue
        try:
            raw = await state.sandbox.files.read(abs_path)
        except Exception:
            continue
        if isinstance(raw, (bytes, bytearray)):
            try:
                raw = raw.decode("utf-8")
            except Exception:
                continue  # not real text — skip rather than corrupt it
        files[rel_path] = raw
        state.file_hashes[rel_path] = _hash(raw)
    return files


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


# ---------------------------------------------------------------------------
# Higher-level "Manus-style" tools, built on top of run_command.
#
# These exist so the agent (and app.py's turn parser) get a small, reliable,
# parameterized surface for the operations that come up constantly — download
# a file, unpack an archive, run whatever test suite the project has, list
# what's on disk — instead of asking the model to hand-roll shell for each one
# every time. Under the hood every one of these still runs a single real
# command through run_command(): nothing here is simulated, and nothing the
# model merely *says* (its THOUGHT) is ever treated as one of these actions —
# only the literal argument passed in (a URL, a path, an optional command
# override) reaches the shell. run_command itself remains available for
# anything that doesn't fit one of these shapes.
# ---------------------------------------------------------------------------

DOWNLOAD_TIMEOUT_SECONDS = float(os.getenv("E2B_DOWNLOAD_TIMEOUT", "180"))
EXTRACT_TIMEOUT_SECONDS = float(os.getenv("E2B_EXTRACT_TIMEOUT", "180"))
TEST_TIMEOUT_SECONDS = float(os.getenv("E2B_TEST_TIMEOUT", "300"))

_ARCHIVE_SUFFIXES = (
    ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar", ".zip", ".gz", ".7z",
)


def _derive_download_dest(url: str) -> str:
    """Best-effort filename from a URL when the caller doesn't specify one."""
    from urllib.parse import urlparse
    name = os.path.basename(urlparse(url).path).strip()
    return name or "downloaded_file"


async def download_file(
    session_id: str,
    url: str,
    dest_path: Optional[str] = None,
    on_output: Optional[LogCallback] = None,
    timeout: float = DOWNLOAD_TIMEOUT_SECONDS,
) -> Tuple[int, str, str]:
    """Download `url` into the sandbox project dir as a real curl/wget command
    (curl first, falling back to wget so either being present is enough).
    Returns (exit_code, combined_output, dest_path)."""
    url = (url or "").strip()
    if not url:
        return 1, "download_file: no URL given.", dest_path or ""
    dest_path = (dest_path or "").strip() or _derive_download_dest(url)
    q_url = shlex.quote(url)
    q_dest = shlex.quote(dest_path)
    cmd = (
        f'mkdir -p "$(dirname {q_dest})" 2>/dev/null; '
        f'curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 -o {q_dest} {q_url} '
        f'|| wget -q -O {q_dest} {q_url}'
    )
    exit_code, output = await run_command(session_id, cmd, on_output=on_output, timeout=timeout)
    return exit_code, output, dest_path


def _strip_archive_suffix(name: str) -> str:
    low = name.lower()
    for suf in _ARCHIVE_SUFFIXES:
        if low.endswith(suf):
            return name[: -len(suf)]
    return name


def _archive_extract_command(archive_path: str, dest_dir: str) -> Optional[str]:
    low = archive_path.lower()
    src = shlex.quote(archive_path)
    dst = shlex.quote(dest_dir)
    mkdir = f"mkdir -p {dst}"
    if low.endswith((".tar.gz", ".tgz")):
        return f"{mkdir} && tar -xzf {src} -C {dst}"
    if low.endswith((".tar.bz2", ".tbz2")):
        return f"{mkdir} && tar -xjf {src} -C {dst}"
    if low.endswith((".tar.xz", ".txz")):
        return f"{mkdir} && tar -xJf {src} -C {dst}"
    if low.endswith(".tar"):
        return f"{mkdir} && tar -xf {src} -C {dst}"
    if low.endswith(".zip"):
        return f"{mkdir} && unzip -o {src} -d {dst}"
    if low.endswith(".gz"):
        base = shlex.quote(os.path.basename(archive_path))
        return f"{mkdir} && cp {src} {dst}/ && gunzip -f {dst}/{base}"
    if low.endswith(".7z"):
        return (
            f"{mkdir} && (command -v 7z >/dev/null || "
            f"(apt-get update -qq && apt-get install -y -qq p7zip-full >/dev/null 2>&1)) && "
            f"7z x -y {src} -o{dst}"
        )
    return None


async def extract_archive(
    session_id: str,
    archive_path: str,
    dest_dir: Optional[str] = None,
    on_output: Optional[LogCallback] = None,
    timeout: float = EXTRACT_TIMEOUT_SECONDS,
) -> Tuple[int, str, str]:
    """Extract a .zip/.tar/.tar.gz/.tar.bz2/.tar.xz/.gz/.7z archive already
    present in the project dir. Returns (exit_code, combined_output, dest_dir)."""
    archive_path = (archive_path or "").strip()
    if not archive_path:
        return 1, "extract_archive: no archive path given.", dest_dir or ""
    dest_dir = (dest_dir or "").strip() or _strip_archive_suffix(os.path.basename(archive_path)) or "extracted"
    cmd = _archive_extract_command(archive_path, dest_dir)
    if cmd is None:
        supported = ", ".join(_ARCHIVE_SUFFIXES)
        return 1, f"extract_archive: unsupported archive type for '{archive_path}'. Supported: {supported}", dest_dir
    exit_code, output = await run_command(session_id, cmd, on_output=on_output, timeout=timeout)
    return exit_code, output, dest_dir


# Auto-detected test runner: tries, in order, an npm "test" script, pytest
# (via common markers), Go, then Rust — same "look at what's actually in the
# project" spirit as detect_default_start below, but for running tests
# instead of serving the app. Only used when the caller doesn't supply an
# explicit test command.
_AUTO_TEST_SCRIPT = r"""
if [ -f package.json ] && grep -q '"test"' package.json; then
  echo "[detected] npm test"
  npm install --no-audit --no-fund --silent 2>/dev/null
  npm test
elif [ -f pytest.ini ] || [ -f pyproject.toml ] || [ -f setup.cfg ] || [ -d tests ] || ls test_*.py >/dev/null 2>&1 || ls *_test.py >/dev/null 2>&1; then
  echo "[detected] pytest"
  [ -f requirements.txt ] && pip install -r requirements.txt --break-system-packages --quiet 2>/dev/null
  pip install pytest --break-system-packages --quiet 2>/dev/null
  pytest -q
elif [ -f go.mod ]; then
  echo "[detected] go test"
  go test ./...
elif [ -f Cargo.toml ]; then
  echo "[detected] cargo test"
  cargo test
else
  echo "No recognized test suite found (checked package.json \"test\" script, pytest markers/tests dir, go.mod, Cargo.toml)."
  exit 3
fi
""".strip()


async def run_tests(
    session_id: str,
    command: Optional[str] = None,
    on_output: Optional[LogCallback] = None,
    timeout: float = TEST_TIMEOUT_SECONDS,
) -> Tuple[int, str]:
    """Run the project's test suite. If `command` is given, run exactly that
    (still via run_command, so it's a real command, never fabricated output).
    Otherwise auto-detect the right runner from what's actually in the
    project directory inside the sandbox."""
    cmd = (command or "").strip() or _AUTO_TEST_SCRIPT
    return await run_command(session_id, cmd, on_output=on_output, timeout=timeout)


async def list_dir(
    session_id: str,
    path: str = ".",
    max_depth: int = 3,
    on_output: Optional[LogCallback] = None,
    timeout: float = 20.0,
) -> Tuple[int, str]:
    """List files/directories under `path` (relative to the project dir),
    skipping the same noisy directories snapshot_project_files ignores."""
    path = (path or ".").strip() or "."
    try:
        depth = max(1, int(max_depth))
    except (TypeError, ValueError):
        depth = 3
    exclude_clause = " ".join(f"-not -path '*/{d}/*'" for d in _SNAPSHOT_EXCLUDE_DIRS)
    cmd = f"find {shlex.quote(path)} -maxdepth {depth} {exclude_clause} 2>/dev/null | sort"
    return await run_command(session_id, cmd, on_output=on_output, timeout=timeout)


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
