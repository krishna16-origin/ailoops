import asyncio
import json
from types import SimpleNamespace

import code_agent
from langchain_core.messages import HumanMessage


async def main():
    original = code_agent._run_agent

    async def fake_run(request, session, emit):
        await emit({"type": "plan_created", "steps": ["Read index.html", "Edit index.html"]})
        await emit({"type": "agent_message_delta", "text": "I'll inspect the existing structure first."})
        await emit({"type": "activity_start", "activity": {"id": "activity_1", "action": "read", "file": "index.html", "status": "running"}})
        await emit({"type": "file_read", "activity": {"id": "activity_1", "action": "read", "file": "index.html", "status": "completed"}, "file": "index.html", "content": "<html />"})
        await emit({"type": "activity_complete", "activity": {"id": "activity_1", "action": "read", "file": "index.html", "status": "completed"}})
        result = {"response": "The activity feed is complete.", "files": {"index.html": "<html />"}, "file_languages": {"index.html": "html"}, "activities": [], "activity_summary": {}, "diffs": [], "code": "", "language": "", "show_preview": True}
        await emit({"type": "final_message", "text": result["response"]})
        await emit({"type": "artifact_created", "files": result["files"], "file_languages": result["file_languages"], "activities": [], "activity_summary": {}, "plan": [], "diffs": []})
        await emit({"type": "complete", "status": "completed"})
        return result, []

    code_agent._run_agent = fake_run
    try:
        request = SimpleNamespace()
        session = {"messages": [HumanMessage(content="Add an activity feed.")]}
        frames = []
        async for frame in code_agent.stream_code_agent(request, session, "session-test"):
            assert frame.startswith("data: ") and frame.endswith("\n\n"), frame
            frames.append(json.loads(frame[6:].strip()))
        types = [frame["type"] for frame in frames]
        expected = ["plan_created", "agent_message_delta", "activity_start", "file_read", "activity_complete", "final_message", "artifact_created", "complete"]
        assert types == expected, types
        assert all(frame["session_id"] == "session-test" for frame in frames)
        print("SSE_CONTRACT_OK")
        print("EVENTS:", ",".join(types))
    finally:
        code_agent._run_agent = original


asyncio.run(main())
