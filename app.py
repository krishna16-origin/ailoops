import os
import os
import json
import re
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from tavily import TavilyClient

load_dotenv()

if not os.getenv("NVIDIA_API_KEY"):
    print("WARNING: NVIDIA_API_KEY not found in environment. The API calls will fail.")
if not os.getenv("TAVILY_API_KEY"):
    print("WARNING: TAVILY_API_KEY not found in environment. Web search will be disabled.")

_tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) if os.getenv("TAVILY_API_KEY") else None

THINKING_LEVELS = {
    "low": {"label": "Low", "max_tokens": 8000, "description": "Quick, focused thinking"},
    "medium": {"label": "Medium", "max_tokens": 16000, "description": "Balanced analysis"},
    "high": {"label": "High", "max_tokens": 24000, "description": "Deep reasoning"},
    "extra": {"label": "Extra", "max_tokens": 32000, "description": "Comprehensive analysis"},
    "max": {"label": "Max", "max_tokens": 40000, "description": "Exhaustive reasoning"},
}
DEFAULT_THINKING_LEVEL = "medium"

# How hard the model is asked to actually reason inside its own <think> block at
# each level — this is what makes "Max" genuinely think longer and deeper than
# "Low", not just a bigger token ceiling with the same shallow pass.
THINKING_DEPTH_INSTRUCTIONS = {
    "low": "Think briefly — a couple of sentences on your approach is enough before answering.",
    "medium": "Think through the key considerations in a short, organized way before answering.",
    "high": "Reason carefully and thoroughly: consider multiple angles, check your logic, and catch mistakes before answering.",
    "extra": "Reason extensively and rigorously: break the problem into parts, explore alternative approaches, weigh trade-offs, and verify your conclusion step by step before answering.",
    "max": "Reason exhaustively, like a world-class expert working through a hard problem: decompose it fully, question your own assumptions, consider edge cases and counter-arguments, verify each step, and only then commit to a final answer.",
}
CODE_THINKING_DEPTH_INSTRUCTIONS = {
    "low": "Briefly plan your approach in a couple of sentences before writing code.",
    "medium": "Plan your approach — data structures, edge cases, file layout — before writing code.",
    "high": "Plan thoroughly: consider the architecture, data flow, error handling, and edge cases before writing code.",
    "extra": "Plan extensively: weigh alternative designs, consider performance and maintainability, and enumerate edge cases and failure modes before committing to an approach and writing code.",
    "max": "Plan like a principal engineer: weigh multiple architectures and their trade-offs, edge cases, error handling, performance, and testability; decide on the strongest approach and justify it to yourself, then write the implementation.",
}


def normalize_thinking_level(level: str) -> str:
    """Fold any input onto one of the five valid thinking-level keys."""
    normalized = (level or DEFAULT_THINKING_LEVEL).strip().lower()
    return normalized if normalized in THINKING_LEVELS else DEFAULT_THINKING_LEVEL


def get_thinking_config(level: str) -> dict:
    """Return a valid thinking level and its token budget."""
    return THINKING_LEVELS[normalize_thinking_level(level)]


def get_llm(model_type: str, temperature: float, max_tokens: int) -> ChatNVIDIA:
    """Create the selected model with the requested completion-token budget."""
    model_name = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    model_type_clean = (model_type or "balanced").strip().lower()
    if model_type_clean == "fast":
        model_name = "deepseek-ai/deepseek-v4-pro"
    elif model_type_clean == "reasoning":
        model_name = "nvidia/nemotron-3-ultra-550b-a55b"
    return ChatNVIDIA(model=model_name, temperature=temperature, max_tokens=max_tokens, timeout=120)


def strip_thinking(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    return text.strip()


def get_current_datetime_str() -> str:
    return datetime.now(timezone.utc).strftime("%A, %B %d, %Y, %I:%M %p UTC")


WEB_SEARCH_TIMEOUT = 9.0
WEB_SEARCH_MAX_RESULTS = 5
WEB_IMAGE_SEARCH_MAX_RESULTS = 4


def _run_web_search_sync(query: str, max_results: int) -> list:
    if _tavily_client is None:
        return []
    response = _tavily_client.search(query, max_results=max_results, search_depth="basic")
    return [
        {"title": r.get("title", ""), "href": r.get("url", ""), "body": r.get("content", "")}
        for r in (response.get("results") or [])
    ]


async def web_search(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS) -> tuple[str, list]:
    """Perform one search attempt; failures return no search context."""
    query = (query or "").strip()
    if not query or _tavily_client is None:
        return "", []
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(_run_web_search_sync, query, max_results),
            timeout=WEB_SEARCH_TIMEOUT,
        )
    except Exception as exc:
        print(f"Web search failed: {exc}")
        return "", []
    if not results:
        return "", []
    lines = [f"Web search results for '{query}' (current date/time: {get_current_datetime_str()}):"]
    for index, item in enumerate(results, start=1):
        lines.append(
            f"{index}. {item.get('title', '')}\n"
            f"   {item.get('body', '')}\n"
            f"   Source: {item.get('href', '')}"
        )
    return "\n".join(lines), results


def _run_web_image_search_sync(query: str, max_results: int) -> list:
    if _tavily_client is None:
        return []
    response = _tavily_client.search(
        query,
        max_results=max_results,
        search_depth="basic",
        include_images=True,
        include_image_descriptions=True,
    )
    images = (response.get("images") or [])[:max_results]
    return [
        {"title": image.get("description") or "Image", "image": image.get("url", ""), "url": image.get("url", "")}
        for image in images
    ]


async def web_image_search(query: str, max_results: int = WEB_IMAGE_SEARCH_MAX_RESULTS) -> list:
    query = (query or "").strip()
    if not query or _tavily_client is None:
        return []
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_run_web_image_search_sync, query, max_results),
            timeout=WEB_SEARCH_TIMEOUT,
        )
    except Exception as exc:
        print(f"Image search failed: {exc}")
        return []


_WEB_SEARCH_KEYWORDS = (
    "latest", "current", "currently", "today", "right now", "this week",
    "this month", "this year", "recent", "recently", "up to date", "up-to-date",
    "news", "release", "released", "price", "cost", "stock", "exchange rate", "weather",
)
_EXPLICIT_SEARCH_PREFIXES = ("search:", "/search", "search for:", "websearch:", "web search:")
_BARE_SEARCH_COMMANDS = {
    "websearch", "web search", "search", "do a web search", "please websearch",
    "please web search", "search the web", "search online", "look it up",
}


def needs_web_search(message: str) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return False
    if text.startswith(_EXPLICIT_SEARCH_PREFIXES):
        return True
    if text.rstrip("?%!. ") in _BARE_SEARCH_COMMANDS:
        return True
    if any(keyword in text for keyword in _WEB_SEARCH_KEYWORDS):
        return True
    return bool(re.search(r"\b20[2-9]\d\b", text))


def extract_search_query(message: str) -> str:
    text = (message or "").strip()
    lowered = text.lower()
    for prefix in _EXPLICIT_SEARCH_PREFIXES:
        if lowered.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def resolve_search_query(messages: List[BaseMessage], latest_user_message: str) -> str:
    query = extract_search_query(latest_user_message)
    if query.strip().lower().rstrip("?!. ") in _BARE_SEARCH_COMMANDS:
        for message in reversed(messages[:-1]):
            content = (message.content or "").strip()
            if isinstance(message, HumanMessage) and content:
                return content
    return query


def build_web_sources_markdown(links: list, images: list) -> str:
    if not links and not images:
        return ""
    parts = ["\n\n---"]
    if links:
        source_lines = []
        for index, item in enumerate(links, start=1):
            title = (item.get("title") or item.get("href") or f"Source {index}").replace("[", "").replace("]", "")
            url = (item.get("href") or "").strip()
            if url:
                source_lines.append(f"{index}. [{title}]({url})")
        if source_lines:
            parts.extend(["**Sources**", "\n".join(source_lines)])
    if images:
        image_lines = []
        for item in images:
            title = (item.get("title") or "Image").replace("[", "").replace("]", "")
            image_url = (item.get("image") or "").strip()
            source_url = (item.get("url") or image_url).strip()
            if image_url:
                image_lines.append(f"[![{title}]({image_url})]({source_url})")
        if image_lines:
            parts.extend(["**Images**", " ".join(image_lines)])
    return "\n\n".join(parts)


sessions: Dict[str, Dict[str, Any]] = {}


def trim_memory(messages: List[BaseMessage], limit: int = 10) -> List[BaseMessage]:
    """Keep recent messages without an extra summarization call."""
    return messages[-limit:]


def request_excerpt(text: str, limit: int = 180) -> str:
    """Create a compact, user-facing excerpt for the visible rationale panel."""
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


async def publish_progress(progress, step: str, label: str, detail: str) -> None:
    """Publish a concrete backend event to a queue or collect it for non-streaming replies."""
    event = {"type": "status", "step": step, "label": label, "detail": detail}
    if isinstance(progress, asyncio.Queue):
        await progress.put(event)
    elif isinstance(progress, list):
        progress.append({k: event[k] for k in ("step", "label", "detail")})


async def publish_token(progress, text: str) -> None:
    if not text:
        return
    if isinstance(progress, asyncio.Queue):
        await progress.put({"type": "token", "text": text})


async def publish_thought(progress, text: str) -> None:
    """Publish a live slice of the model's OWN reasoning trace — whatever it wrote
    inside <think>/<thinking> tags — as it streams in, token by token. This is real
    chain-of-thought from the model, not synthesized narration."""
    if not text:
        return
    if isinstance(progress, asyncio.Queue):
        await progress.put({"type": "thought", "text": text})


async def publish_activity(progress, kind: str, label: str, **extra) -> None:
    """Publish one concrete, already-happened Code Mode action (a file actually
    generated, with its real line count) for the activity trace."""
    if isinstance(progress, asyncio.Queue):
        event = {"type": "activity", "kind": kind, "label": label}
        event.update(extra)
        await progress.put(event)


async def publish_turn_summary(progress, **counts) -> None:
    """Publish the final real tally (files touched, etc.) for the turn's recap line."""
    if isinstance(progress, asyncio.Queue):
        await progress.put({"type": "turn_summary", **counts})


def build_messages(history: List[BaseMessage], thinking_level: str, search_text: str = "") -> List[BaseMessage]:
    level_key = normalize_thinking_level(thinking_level)
    config = THINKING_LEVELS[level_key]
    depth = THINKING_DEPTH_INSTRUCTIONS[level_key]
    system_text = (
        "You are a sharp, genuinely helpful assistant with real step-by-step reasoning ability.\n"
        f"Before answering, think inside a single <think>...</think> block. {depth}\n"
        "Write that block as your own natural reasoning as you work through the problem — not a "
        "restatement of these instructions and not a performance for an audience.\n"
        "After the closing </think> tag, give the user a direct, clean final answer with no meta-commentary "
        "about your process.\n"
        f"Thinking level: {config['label']} — {config['description']}.\n"
        f"Maximum completion budget: {config['max_tokens']} tokens.\n"
        f"Current date and time: {get_current_datetime_str()}"
    )
    if search_text:
        system_text += "\n\nUse the following web results only when relevant:\n" + search_text
    return [SystemMessage(content=system_text), *history[-6:]]


async def invoke_model(messages: List[BaseMessage], llm: ChatNVIDIA, progress=None) -> str:
    """Invoke once. When a live queue is provided, stream BOTH the visible answer
    ('token' events) and the model's own live reasoning trace ('thought' events) in
    real time, exactly as the model produces them — mirroring Claude.ai's extended
    thinking pane instead of a canned status message."""
    if not isinstance(progress, asyncio.Queue):
        result = await llm.ainvoke(messages)
        return strip_thinking(getattr(result, "content", "") or "").strip()

    OPEN_TAGS = ("<think>", "<thinking>")
    CLOSE_TAGS = ("</think>", "</thinking>")
    max_tag_len = max(len(t) for t in OPEN_TAGS + CLOSE_TAGS)

    full = ""
    buffer = ""
    in_thought = False

    async def drain(flush_all: bool) -> None:
        nonlocal buffer, in_thought
        while True:
            if not in_thought:
                positions = [buffer.find(t) for t in OPEN_TAGS if t in buffer]
                idx = min(positions) if positions else -1
                if idx == -1:
                    hold_back = 0 if flush_all else min(len(buffer), max_tag_len - 1)
                    send_len = len(buffer) - hold_back
                    if send_len > 0:
                        await publish_token(progress, buffer[:send_len])
                        buffer = buffer[send_len:]
                    return
                if idx:
                    await publish_token(progress, buffer[:idx])
                tag = next(t for t in OPEN_TAGS if buffer[idx:].startswith(t))
                buffer = buffer[idx + len(tag):]
                in_thought = True
            else:
                positions = [buffer.find(t) for t in CLOSE_TAGS if t in buffer]
                idx = min(positions) if positions else -1
                if idx == -1:
                    hold_back = 0 if flush_all else min(len(buffer), max_tag_len - 1)
                    send_len = len(buffer) - hold_back
                    if send_len > 0:
                        await publish_thought(progress, buffer[:send_len])
                        buffer = buffer[send_len:]
                    return
                if idx:
                    await publish_thought(progress, buffer[:idx])
                tag = next(t for t in CLOSE_TAGS if buffer[idx:].startswith(t))
                buffer = buffer[idx + len(tag):]
                in_thought = False

    async for chunk in llm.astream(messages):
        piece = getattr(chunk, "content", "") or ""
        if not piece:
            continue
        full += piece
        buffer += piece
        await drain(flush_all=False)

    await drain(flush_all=True)
    return strip_thinking(full).strip()


async def chat_understand_node(request: "ChatRequest", session: dict, progress=None) -> dict:
    history = session["messages"]
    latest = history[-1].content if history else ""
    config = get_thinking_config(request.thinking_level)
    excerpt = request_excerpt(latest)
    await publish_progress(progress, "chat_understand_node", "chat_understand_node", f"Read the latest user request and isolated the topic: “{excerpt}”")
    return {"history": history, "latest": latest, "config": config}


async def chat_context_node(state: dict, progress=None) -> dict:
    latest = state["latest"]
    history = state["history"]
    search_text = ""
    links = []
    images = []
    if needs_web_search(latest):
        query = resolve_search_query(history, latest)
        await publish_progress(progress, "chat_context_node", "chat_context_node", f"Detected a current-information request and searched for: {query}")
        (search_text, links), images = await asyncio.gather(web_search(query), web_image_search(query))
        await publish_progress(progress, "chat_context_result", "chat_context_result", f"Collected {len(links)} source result(s) for the response context.")
    else:
        await publish_progress(progress, "chat_context_node", "chat_context_node", "No web lookup was required; continuing with the conversation context.")
    state.update({"search_text": search_text, "links": links, "images": images})
    return state


async def chat_compose_node(request: "ChatRequest", state: dict, progress=None) -> dict:
    config = state["config"]
    await publish_progress(progress, "chat_compose_node", "chat_compose_node", f"Invoking the model with {config['label']} thinking and a {config['max_tokens']}-token budget.")
    llm = get_llm(request.model_type, request.temperature, config["max_tokens"])
    response = await invoke_model(build_messages(state["history"], request.thinking_level, state["search_text"]), llm, progress)
    state["response"] = response or "I apologize, I encountered an issue formulating my answer."
    return state


async def chat_finalize_node(state: dict, progress=None) -> str:
    response = state["response"]
    sources = build_web_sources_markdown(state["links"], state["images"])
    if sources:
        response += sources
        await publish_token(progress, sources)
    await publish_progress(progress, "chat_finalize_node", "chat_finalize_node", "Validated the response format and prepared the final answer for display.")
    return response


async def generate_response_once(request: "ChatRequest", session: dict, progress=None) -> str:
    state = await chat_understand_node(request, session, progress)
    state = await chat_context_node(state, progress)
    state = await chat_compose_node(request, state, progress)
    return await chat_finalize_node(state, progress)


app = FastAPI(title="AI Assistant")
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")


@app.get("/")
@app.head("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str
    model_type: str = "balanced"
    stream: bool = False
    temperature: float = 0.7
    thinking_level: str = DEFAULT_THINKING_LEVEL


class ClearSessionRequest(BaseModel):
    session_id: str


class CodeChatRequest(BaseModel):
    message: str
    session_id: str
    model: str = "glm"
    reasoning_level: str = DEFAULT_THINKING_LEVEL
    stream: bool = False


def build_code_messages(history: List[BaseMessage], reasoning_level: str) -> List[BaseMessage]:
    level_key = normalize_thinking_level(reasoning_level)
    config = THINKING_LEVELS[level_key]
    depth = CODE_THINKING_DEPTH_INSTRUCTIONS[level_key]
    system_text = (
        "You are a practical coding assistant. Produce the requested implementation directly in one pass.\n"
        f"Before writing code, think inside a single <think>...</think> block. {depth}\n"
        "Write that block as your own natural engineering reasoning — not a restatement of these instructions.\n"
        "After the closing </think> tag, give a concise user-facing explanation and put the implementation in "
        "fenced code blocks. If the request needs multiple files, write a separate line `FILE: path/to/file.ext` "
        "immediately before each fenced block. Use the correct language tag for every block. If one file is "
        "enough, return one block without a FILE header.\n"
        f"Effort: {config['label']}. Maximum completion budget: {config['max_tokens']} tokens.\n"
        f"Current date and time: {get_current_datetime_str()}"
    )
    return [SystemMessage(content=system_text), *history[-6:]]


def extract_code_artifact(text: str) -> tuple[str, str, str, dict, dict]:
    """Extract one file or multiple named files from Code-mode output."""
    source = text or ""
    named_pattern = re.compile(
        r"(?:^|\n)\s*FILE\s*:\s*(?P<filename>[^\n]+)\n\s*```(?P<language>[A-Za-z0-9_+#.-]*)\s*\n(?P<code>[\s\S]*?)```",
        re.IGNORECASE,
    )
    named_matches = list(named_pattern.finditer(source))
    if named_matches:
        files = {}
        file_languages = {}
        for match in named_matches:
            filename = match.group("filename").strip().strip('`')
            if not filename:
                continue
            files[filename] = match.group("code").strip()
            file_languages[filename] = (match.group("language") or "text").lower()
        explanation = named_pattern.sub("", source).strip()
        return explanation, "", "", files, file_languages

    matches = list(re.finditer(r"```([A-Za-z0-9_+#.-]*)\s*\n([\s\S]*?)```", source))
    if not matches:
        return source.strip(), "", "", {}, {}
    if len(matches) == 1:
        language = (matches[0].group(1) or "text").lower()
        code = matches[0].group(2).strip()
        explanation = re.sub(r"```[A-Za-z0-9_+#.-]*\s*\n[\s\S]*?```", "", source).strip()
        return explanation, code, language, {}, {}

    extension_map = {"python": "py", "py": "py", "javascript": "js", "js": "js", "typescript": "ts", "html": "html", "css": "css", "json": "json"}
    files = {}
    file_languages = {}
    for index, match in enumerate(matches, start=1):
        language = (match.group(1) or "text").lower()
        extension = extension_map.get(language, "txt")
        filename = f"file_{index}.{extension}"
        files[filename] = match.group(2).strip()
        file_languages[filename] = language
    explanation = re.sub(r"```[A-Za-z0-9_+#.-]*\s*\n[\s\S]*?```", "", source).strip()
    return explanation, "", "", files, file_languages


async def code_understand_node(request: CodeChatRequest, session: dict, progress=None) -> dict:
    config = get_thinking_config(request.reasoning_level)
    excerpt = request_excerpt(session["messages"][-1].content if session["messages"] else "")
    await publish_progress(progress, "code_understand_node", "code_understand_node", f"Read the build request and identified the requested result: “{excerpt}”")
    return {"config": config, "history": session["messages"]}


async def code_compose_node(request: CodeChatRequest, state: dict, progress=None) -> dict:
    config = state["config"]
    model_type = "reasoning" if request.model in {"glm", "kimik2.6"} else "balanced"
    await publish_progress(progress, "code_compose_node", "code_compose_node", f"Invoking the code model with {config['label']} effort and a {config['max_tokens']}-token budget.")
    llm = get_llm(model_type, 0.2, config["max_tokens"])
    state["raw"] = await invoke_model(build_code_messages(state["history"], request.reasoning_level), llm, progress)
    return state


async def code_extract_artifact_node(state: dict, progress=None) -> dict:
    explanation, code, language, files, file_languages = extract_code_artifact(state["raw"])
    state.update({"explanation": explanation, "code": code, "language": language, "files": files, "file_languages": file_languages})
    if files:
        await publish_progress(progress, "code_extract_artifact_node", "code_extract_artifact_node", f"Separated {len(files)} named files for the canvas: {', '.join(files)}")
        for filename, content in files.items():
            line_count = len(content.splitlines()) if content else 0
            await publish_activity(
                progress, "create", f"Created {filename}",
                filename=filename, additions=line_count, deletions=0,
            )
    elif code:
        await publish_progress(progress, "code_extract_artifact_node", "code_extract_artifact_node", f"Extracted the {language or 'text'} implementation block for the canvas.")
        line_count = len(code.splitlines()) if code else 0
        implied_name = f"main.{language}" if language else "main"
        await publish_activity(
            progress, "create", f"Created {implied_name}",
            filename=implied_name, additions=line_count, deletions=0,
        )
    else:
        await publish_progress(progress, "code_extract_artifact_node", "code_extract_artifact_node", "No fenced artifact was returned, so the explanation remains the visible result.")
    return state


async def code_finalize_node(state: dict, progress=None) -> dict:
    config = state["config"]
    files = state.get("files", {})
    file_languages = state.get("file_languages", {})
    previewable = {"html", "htm", "css", "js", "javascript", "jsx", "tsx"}
    show_preview = state["language"] in previewable or any(language in previewable for language in file_languages.values())
    await publish_progress(progress, "code_finalize_node", "code_finalize_node", f"Prepared {len(files) if files else 1} separate artifact file(s) and preview metadata for the frontend.")
    files_touched = len(files) if files else (1 if state.get("code") else 0)
    await publish_turn_summary(progress, files_touched=files_touched, commands_run=0, files_read=0, notes=0)
    return {
        "response": state["explanation"] or "I generated the requested code.",
        "code": state["code"],
        "language": state["language"],
        "files": files,
        "file_languages": file_languages,
        "show_preview": show_preview,
        "thinking_summary": f"Generated the implementation with explicit backend nodes using {config['label']} effort.",
        "thinking_level": config["label"],
        "max_tokens": config["max_tokens"],
    }


async def generate_code_once(request: CodeChatRequest, session: dict, progress=None) -> dict:
    state = await code_understand_node(request, session, progress)
    state = await code_compose_node(request, state, progress)
    state = await code_extract_artifact_node(state, progress)
    return await code_finalize_node(state, progress)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/clear-session")
async def clear_session(request: ClearSessionRequest):
    sessions.pop(request.session_id, None)
    return {"status": "success", "message": f"Session {request.session_id} cleared."}


async def forward_live_events(task: asyncio.Task, progress_queue: asyncio.Queue, session_id: str):
    """Yield queue events while the backend task is still running."""
    while not task.done() or not progress_queue.empty():
        try:
            event = await asyncio.wait_for(progress_queue.get(), timeout=2.5)
            yield event
        except asyncio.TimeoutError:
            yield {"type": "status", "step": "processing", "label": "Working", "detail": "The backend is still processing the request."}


async def generate_stream(request: ChatRequest, session: dict, session_id: str):
    progress_queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(generate_response_once(request, session, progress_queue))
    final_response = ""
    emitted_content = False
    try:
        async for event in forward_live_events(task, progress_queue, session_id):
            if event["type"] in ("status", "thought"):
                yield f"data: {json.dumps(event)}\n\n"
            elif event["type"] == "token":
                emitted_content = True
                final_response += event["text"]
                yield f"data: {json.dumps({'type': 'message', 'assistant_message': event['text'], 'conversation_id': session_id, 'session_id': session_id})}\n\n"
        completed_response = await task
        if not emitted_content:
            final_response = completed_response
            yield f"data: {json.dumps({'type': 'message', 'assistant_message': final_response, 'conversation_id': session_id, 'session_id': session_id})}\n\n"
    except Exception as exc:
        print(f"[{session_id}] Response generation failed: {exc}")
        final_response = "I could not generate a response right now. Please try again."
        yield f"data: {json.dumps({'type': 'message', 'assistant_message': final_response, 'conversation_id': session_id, 'session_id': session_id})}\n\n"
    session["messages"].append(AIMessage(content=final_response))


async def generate_code_stream(request: CodeChatRequest, session: dict, session_id: str):
    progress_queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(generate_code_once(request, session, progress_queue))
    result = None
    streamed_text = ""
    try:
        async for event in forward_live_events(task, progress_queue, session_id):
            if event["type"] in ("status", "thought", "activity", "turn_summary"):
                yield f"data: {json.dumps(event)}\n\n"
            elif event["type"] == "token":
                streamed_text += event["text"]
                yield f"data: {json.dumps({'type': 'message', 'assistant_message': event['text'], 'conversation_id': session_id, 'session_id': session_id})}\n\n"
        result = await task
    except Exception as exc:
        print(f"[{session_id}] Code generation failed: {exc}")
        result = {
            "response": "I could not generate code right now. Please try again.",
            "code": "",
            "language": "",
            "files": {},
            "file_languages": {},
            "show_preview": False,
        }
    if result.get("code") or result.get("files"):
        yield f"data: {json.dumps({'type': 'code_result', **result, 'session_id': session_id})}\n\n"
    if not streamed_text:
        yield f"data: {json.dumps({'type': 'message', 'assistant_message': result['response'], 'conversation_id': session_id, 'session_id': session_id})}\n\n"
    session["messages"].append(AIMessage(content=result["response"]))


@app.post("/code-chat")
async def code_chat(request: CodeChatRequest):
    session = sessions.setdefault(request.session_id, {"messages": []})
    session["messages"] = trim_memory(session["messages"])
    session["messages"].append(HumanMessage(content=request.message))
    if request.stream:
        return StreamingResponse(
            generate_code_stream(request, session, request.session_id),
            media_type="text/event-stream",
        )
    thinking_steps = []
    try:
        result = await generate_code_once(request, session, thinking_steps)
    except Exception as exc:
        print(f"[{request.session_id}] Code generation failed: {exc}")
        result = {
            "response": "I could not generate code right now. Please try again.",
            "code": "",
            "language": "",
            "files": {},
            "file_languages": {},
            "show_preview": False,
        }
    session["messages"].append(AIMessage(content=result["response"]))
    result["thinking_steps"] = thinking_steps
    return {"session_id": request.session_id, **result}


@app.post("/chat")
async def chat(request: ChatRequest):
    session = sessions.setdefault(request.session_id, {"messages": []})
    session["messages"] = trim_memory(session["messages"])
    session["messages"].append(HumanMessage(content=request.message))
    if request.stream:
        return StreamingResponse(
            generate_stream(request, session, request.session_id),
            media_type="text/event-stream",
        )
    thinking_steps = []
    try:
        response = await generate_response_once(request, session, thinking_steps)
    except Exception as exc:
        print(f"[{request.session_id}] Response generation failed: {exc}")
        response = "I could not generate a response right now. Please try again."
    session["messages"].append(AIMessage(content=response))
    config = get_thinking_config(request.thinking_level)
    return {
        "response": response,
        "session_id": request.session_id,
        "thinking_summary": f"Answered directly with {config['label']} thinking; no planning or revision pass was run.",
        "thinking_steps": thinking_steps,
        "thinking_level": config["label"],
        "max_tokens": config["max_tokens"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
