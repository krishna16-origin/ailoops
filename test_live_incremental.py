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
    assert not events[-1].get("text", "").endswith("inspect")
    await turn_watcher(" the existing page.\nACT")
    assert events[-1] == {"type": "agent_message", "text": "I'll inspect the existing page."}
    await turn_watcher("ION: read_file\nPAT")
    assert not any(event["type"] == "activity_start" for event in events)
    await turn_watcher("H: index.html\n")
    assert events[-1]["type"] == "activity_start"
    assert events[-1]["activity"]["file"] == "index.html"
    await turn_watcher("ACTION: final\nThe activity feed is complete.\n")
    assert events[-1] == {"type": "final_message", "text": "The activity feed is complete."}

    types = [event["type"] for event in events]
    expected = ["plan_update", "plan_update", "agent_message", "activity_start", "final_message"]
    assert types == expected, types
    print("LIVE_INCREMENTAL_OK")
    print("EVENTS:", ",".join(types))


asyncio.run(main())
