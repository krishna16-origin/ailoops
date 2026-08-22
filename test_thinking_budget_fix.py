"""Offline regression test for the pasted PLAN -> FILE EVENTS -> SSE workflow."""
import asyncio
from types import SimpleNamespace

import code_agent
from langchain_core.messages import HumanMessage


async def main():
    calls = {"count": 0}

    async def fake_invoke(messages, llm, progress=None, on_answer_piece=None, thinking_mode=None, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return "1. Read index.html\n2. Edit index.html\n3. Review the result"
        if calls["count"] == 2:
            return "THOUGHT: I will inspect the existing page first.\nACTION: read_file\nPATH: index.html"
        if calls["count"] == 3:
            return "THOUGHT: I will add the feed without replacing unrelated content.\nACTION: edit_file\nPATH: index.html\n```html\n<html>\n<body>\n<section class=\"activity-feed\">Live activity</section>\n</body>\n</html>\n```"
        return "THOUGHT: The activity feed is complete.\nACTION: final\nThe activity feed is complete."

    original_invoke = code_agent.invoke_model
    original_llm = code_agent.get_code_llm
    try:
        code_agent.invoke_model = fake_invoke
        code_agent.get_code_llm = lambda *args, **kwargs: object()
        session = {
            "messages": [HumanMessage(content="Add an activity feed to my application.")],
            "code_files": {"index.html": "<html>\n<body>\n<h1>App</h1>\n</body>\n</html>"},
        }
        request = SimpleNamespace(message="Add an activity feed to my application.", reasoning_level="medium", model="medium")
        events = []

        async def emit(event):
            events.append(event)

        result, _ = await code_agent._run_agent(request, session, emit)
        types = [event["type"] for event in events]
        expected = ["plan_created", "agent_message", "activity_start", "file_read", "activity_complete", "agent_message", "activity_start", "file_edited", "diff_created", "activity_complete", "agent_message", "final_message", "artifact_created", "complete"]
        assert types == expected, types
        assert "activity-feed" in result["files"]["index.html"]
        assert result["diffs"][0]["additions"] == 1
        print("RICH_WORKFLOW_REGRESSION_OK")
        print("EVENTS:", ",".join(types))
    finally:
        code_agent.invoke_model = original_invoke
        code_agent.get_code_llm = original_llm


asyncio.run(main())
