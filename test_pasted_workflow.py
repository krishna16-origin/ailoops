import asyncio
from types import SimpleNamespace

import code_agent
from langchain_core.messages import HumanMessage


class FakeLLM:
    pass


async def main():
    calls = {"count": 0}

    async def fake_invoke(messages, llm, progress=None, on_answer_piece=None, thinking_mode=None, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return "1. Read index.html\n2. Edit index.html\n3. Review the result"
        if calls["count"] == 2:
            return "THOUGHT: I'll inspect the existing page first.\nACTION: read_file\nPATH: index.html"
        if calls["count"] == 3:
            return "THOUGHT: I'll add the activity feed without replacing the existing page.\nACTION: edit_file\nPATH: index.html\n```html\n<html>\n  <body>\n    <section class=\"activity-feed\">Live activity</section>\n  </body>\n</html>\n```"
        return "THOUGHT: The activity feed is complete.\nACTION: final\nThe page now has a live activity feed."

    code_agent.invoke_model = fake_invoke
    code_agent.get_code_llm = lambda *args, **kwargs: FakeLLM()
    code_agent.MAX_AGENT_STEPS = 4

    session = {
        "messages": [HumanMessage(content="Add an activity feed to my application.")],
        "code_files": {"index.html": "<html>\n  <body>\n    <h1>App</h1>\n  </body>\n</html>"},
    }
    request = SimpleNamespace(message="Add an activity feed to my application.", reasoning_level="medium", model="medium")
    events = []

    async def emit(event):
        events.append(event)

    result, _ = await code_agent._run_agent(request, session, emit)
    types = [event["type"] for event in events]
    expected = ["plan_created", "agent_message", "activity_start", "file_read", "activity_complete", "agent_message", "activity_start", "file_edited", "diff_created", "activity_complete", "agent_message", "final_message", "artifact_created", "complete"]
    if types != expected:
        raise AssertionError(f"Unexpected event sequence: {types}")
    if result["files"]["index.html"].find("activity-feed") == -1:
        raise AssertionError("Edited file was not persisted")
    if not result["diffs"] or result["diffs"][0]["additions"] != 1:
        raise AssertionError(f"Unexpected diff result: {result['diffs']}")
    print("PASTED_WORKFLOW_OK")
    print("EVENTS:", ",".join(types))


asyncio.run(main())
