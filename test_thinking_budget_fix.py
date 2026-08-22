"""
Verifies the ThinkingBudgetExceeded wiring without hitting the real NVIDIA API.

Simulates exactly the failure mode from the 704s timeout: on the first call the
model streams a huge wall of reasoning_content and never gets to real answer
text. Before the fix this would just run forever (or until the outer timeout)
and discard everything. After the fix, invoke_model should raise
ThinkingBudgetExceeded partway through, generate_code_once should catch it and
immediately retry with thinking_mode=False, and that retry (a normal, well-
behaved response) should produce real code + files + an activity trace shaped
like the screenshot (Read/Edited rows + a summary line).
"""
import asyncio
from types import SimpleNamespace

import app


class FakeChunk:
    def __init__(self, content="", reasoning=""):
        self.content = content
        self.additional_kwargs = {"reasoning_content": reasoning} if reasoning else {}


class RamblingThenGoodLLM:
    """First .astream() call: 5000 chars of reasoning, never reaches an answer
    (mirrors the real 704s run). Second call (thinking disabled): answers
    immediately with a real FILE:-tagged code block."""

    def __init__(self):
        self.calls = 0

    async def astream(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            # Simulate a model burning its whole budget "still planning" —
            # chunked, like a real streaming response. Sized comfortably above
            # the real max_think_chars for reasoning_level="low"
            # (16000 tokens * 0.45 * 4 chars/token = 28800 chars).
            chunk_text = "Let me reconsider the architecture once more. " * 100  # 4800 chars/chunk
            for _ in range(10):  # 48,000 chars total, well over the 28,800 budget
                yield FakeChunk(reasoning=chunk_text)
        else:
            # Retry call (thinking_mode=False): well-behaved, answers directly.
            yield FakeChunk(
                content=(
                    "Built the landing page.\n\n"
                    "FILE: index.html\n```html\n<!doctype html><html><body>Hello</body></html>\n```\n"
                )
            )


async def main():
    fake_llm = RamblingThenGoodLLM()
    app.get_code_llm = lambda *a, **k: fake_llm

    request = SimpleNamespace(
        reasoning_level="low",
        model="strong",
    )
    session = {"messages": [app.HumanMessage(content="Build me a landing page")]}
    progress_queue: asyncio.Queue = asyncio.Queue()

    result = await app.generate_code_once(request, session, progress_queue)

    events = []
    while not progress_queue.empty():
        events.append(progress_queue.get_nowait())

    retry_events = [e for e in events if e.get("type") == "RETRY"]
    print(f"Raw events seen: {events}")

    print(f"LLM was called {fake_llm.calls} times (expect 2: rambling attempt + retry)")
    print(f"RETRY event published: {len(retry_events) == 1} -> {retry_events}")
    print(f"Code artifact produced: {bool(result['files'] or result['code'])}")
    print(f"Files: {list(result['files'].keys())}")
    print(f"Explanation text shown to user: {result['response']!r}")
    print(f"Activities (matches screenshot shape): {result['activities']}")
    print(f"Activity summary line data: {result['activity_summary']}")

    assert fake_llm.calls == 2, "expected exactly one retry"
    assert len(retry_events) == 1, "expected a RETRY progress event"
    assert result["files"].get("index.html"), "expected real code to land after retry"
    assert not result.get("artifact_error"), "should not report artifact_error after a successful retry"
    print("\nALL CHECKS PASSED — the watchdog fires, retries, and code lands.")


asyncio.run(main())
