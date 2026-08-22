import asyncio

import code_agent


async def main():
    events = []

    async def emit(event):
        events.append(event)

    plan_watcher = code_agent.make_plan_stream_watcher(emit)
    await plan_watcher("1. Read index.html\n2. Edit index.html\n")

    turn_watcher = code_agent.make_agent_stream_watcher(emit, "activity_turn_1")
    await turn_watcher("THOUGHT: I'll inspect")
    assert not any(event["type"] == "activity_start" for event in events)
    await turn_watcher(" the existing page.\nACT")
    thought_text = "".join(event["text"] for event in events if event["type"] == "agent_message_delta")
    assert "I'll inspect the existing page." in thought_text
    await turn_watcher("ION: read_file\nPAT")
    assert not any(event["type"] == "activity_start" for event in events)
    await turn_watcher("H: index.html\n")
    assert events[-1]["type"] == "activity_start"
    assert events[-1]["activity"]["file"] == "index.html"
    await turn_watcher("ACTION: final\nThe activity feed is complete.")
    await turn_watcher.flush()
    assert events[-1] == {"type": "final_message", "text": "The activity feed is complete."}

    types = [event["type"] for event in events]
    assert types[:2] == ["plan_update", "plan_update"], types
    assert "agent_message_delta" in types, types
    assert "activity_start" in types, types
    assert types[-1] == "final_message", types
    print("LIVE_INCREMENTAL_OK")
    print("EVENTS:", ",".join(types))


asyncio.run(main())
