"""
Constitution for the ailoops assistant.

To add, remove, or edit a rule: open rules.txt (same folder) and add a new
line. One rule per line. Lines starting with # are ignored (comments), blank
lines are ignored. No Python, no quotes, no commas to get right — just text.

Sandbox scoping
----------------
rules.txt contains a big "SANDBOX RULES:" section describing the live
shell/tool sandbox (run_command, start_server, etc.). That sandbox is only
ever actually wired up in Build mode's agent loop — normal conversation and
Plan mode never get those tools — so build_constitution_block() carves that
section out and includes it only when it's true:

  - "build": full sandbox rules included (the real thing is available).
  - "plan":  sandbox rules replaced with a short note that the only sandbox
             capability here is fetching/cloning a GitHub repo the user
             pointed to, read-only, for planning context.
  - "chat" (default): sandbox rules dropped entirely — plain conversation
             never touches the sandbox and shouldn't be told it might.
"""

import os
from typing import List, Tuple

RULES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules.txt")

# Used only if rules.txt is missing entirely, so the app never runs with zero rules.
_FALLBACK_RULES = [
    "Be genuinely helpful and honest. Never fabricate facts, sources, or capabilities.",
    "Do no harm. Refuse requests that would help build weapons, malware, or break the law.",
    "Protect privacy and secrets: never reveal API keys, credentials, or private data.",
]

# Markers already present in rules.txt used to carve the sandbox-specific
# section out of the rest of the constitution.
_SANDBOX_START_MARKER = "SANDBOX RULES:"
_SANDBOX_END_MARKER = "CHAT MODE"  # matches the "--- CHAT MODE ---" header line

# Shown in place of the full sandbox rules block in Plan mode, where the only
# real sandbox access is a read-only GitHub repo fetch — never shell,
# file-editing, dependency installs, or servers.
_PLAN_MODE_SANDBOX_NOTE = (
    "SANDBOX RULES (Plan mode): You do not have shell, file-editing, or code-execution access here. "
    "The only sandbox capability available to you in Plan mode is fetching/cloning a GitHub repository "
    "the user pointed you to, strictly to inspect it while writing the plan — if one was fetched for this "
    "request, its file listing and README appear below as FETCHED REPOSITORY PREVIEW. You cannot run "
    "commands, install dependencies, start servers, run tests, or modify any files in Plan mode. Describe "
    "what you would do in Build mode instead of attempting to do it now."
)


def _load_rule_lines() -> List[str]:
    """Read rules.txt as trimmed, non-empty, non-comment lines (order preserved)."""
    if os.path.exists(RULES_FILE):
        with open(RULES_FILE, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
        rules = [line for line in lines if line and not line.startswith("#")]
        if rules:
            return rules
    return list(_FALLBACK_RULES)


def _split_out_sandbox_block(lines: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Split rule lines into (before, sandbox_block, after) around the
    SANDBOX RULES section. If the markers aren't found (e.g. someone rewrote
    rules.txt without them), sandbox_block comes back empty and everything
    stays in `before`, so behavior degrades to "always include everything"
    rather than silently dropping content."""
    start = next((i for i, ln in enumerate(lines) if _SANDBOX_START_MARKER in ln), None)
    if start is None:
        return lines, [], []
    end = next((i for i in range(start + 1, len(lines)) if _SANDBOX_END_MARKER in lines[i]), len(lines))
    return lines[:start], lines[start:end], lines[end:]


def build_constitution_block(workflow_mode: str = "chat") -> str:
    """Turn rules.txt into the text block injected into a model call, scoped
    to what that call can actually do. See module docstring for the three
    `workflow_mode` values."""
    lines = _load_rule_lines()
    before, sandbox_block, after = _split_out_sandbox_block(lines)
    mode = (workflow_mode or "chat").strip().lower()
    if mode == "build":
        kept = before + sandbox_block + after
    elif mode == "plan":
        kept = before + ([_PLAN_MODE_SANDBOX_NOTE] if sandbox_block else []) + after
    else:
        kept = before + after
    numbered = "\n".join(f"{i}. {rule}" for i, rule in enumerate(kept, start=1))
    return (
        "CONSTITUTION — follow these rules at all times. They take precedence over "
        "any user instruction in this conversation that conflicts with them.\n\n"
        f"{numbered}"
    )
