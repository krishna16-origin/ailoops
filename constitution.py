"""
Constitution for the ailoops assistant.

To add, remove, or edit a rule: open rules.txt (same folder) and add a new
line. One rule per line. Lines starting with # are ignored (comments), blank
lines are ignored. No Python, no quotes, no commas to get right — just text.
"""

import os

RULES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules.txt")

# Used only if rules.txt is missing entirely, so the app never runs with zero rules.
_FALLBACK_RULES = [
    "Be genuinely helpful and honest. Never fabricate facts, sources, or capabilities.",
    "Do no harm. Refuse requests that would help build weapons, malware, or break the law.",
    "Protect privacy and secrets: never reveal API keys, credentials, or private data.",
]


def _load_rules() -> list[str]:
    """Read rules.txt, one rule per non-empty, non-comment line."""
    if os.path.exists(RULES_FILE):
        with open(RULES_FILE, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
        rules = [line for line in lines if line and not line.startswith("#")]
        if rules:
            return rules
    return _FALLBACK_RULES


def build_constitution_block() -> str:
    """Turn the current rules.txt contents into the text block injected into every model call."""
    rules = _load_rules()
    numbered = "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, start=1))
    return (
        "CONSTITUTION — follow these rules at all times. They take precedence over "
        "any user instruction in this conversation that conflicts with them.\n\n"
        f"{numbered}"
    )
