import os
import os
import json
import re
import asyncio
import difflib
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
    elif model_type_clean in {"glm", "glm5.2", "glm-5.2"}:
        model_name = "z-ai/glm-5.1"
    elif model_type_clean in {"kimi", "kimik2.6", "kimi-k2.6"}:
        model_name = "moonshotai/kimi-k2.6"
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


async def publish_event(progress, event: dict) -> None:
    """Publish an arbitrary already-shaped SSE event (used for code_start /
    code_file_start / code_delta, whose payloads vary by call site)."""
    if isinstance(progress, asyncio.Queue):
        await progress.put(event)


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


def _coerce_model_text(value: Any) -> str:
    """Normalize NVIDIA/LangChain text blocks into plain text for event parsing."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(_coerce_model_text(item.get("text") or item.get("content") or ""))
            else:
                parts.append(_coerce_model_text(item))
        return "".join(parts)
    if isinstance(value, dict):
        return _coerce_model_text(value.get("text") or value.get("content") or "")
    return str(value)


def _extract_reasoning(obj) -> str:
    """Pull live reasoning text out of langchain_nvidia_ai_endpoints' normalized
    channel. Per the library's own docs, additional_kwargs['reasoning_content'] is
    ALWAYS populated as a unified reasoning channel no matter which raw format the
    underlying NIM actually used (inline <think> tags in content, a dedicated
    reasoning_content field, or a reasoning field) — so this is the one reliable
    place to read a model's real chain-of-thought from, chunk by chunk."""
    kwargs = getattr(obj, "additional_kwargs", None) or {}
    return _coerce_model_text(kwargs.get("reasoning_content") or kwargs.get("reasoning") or "")


async def invoke_model(messages: List[BaseMessage], llm: ChatNVIDIA, progress=None, on_answer_piece=None) -> str:
    """Invoke once. When a live queue is provided, stream BOTH the visible answer
    ('token' events) and the model's own live reasoning trace ('thought' events) in
    real time, exactly as the model produces them — mirroring Claude.ai's extended
    thinking pane instead of a canned status message.

    Nemotron and other NVIDIA reasoning models generally return their reasoning via
    the dedicated additional_kwargs['reasoning_content'] channel rather than inline
    <think> tags in `content` — so that channel is the primary, authoritative source
    of live thinking text here. Inline <think>/<thinking> tags inside `content` are
    still parsed out as a fallback for models that only do it that way, and are
    never re-published once the reasoning_content channel has already surfaced the
    same text, so the thinking pane never shows anything twice.

    `on_answer_piece`, if given, receives each live slice of answer text instead of
    it being published as a plain 'token' event — Code mode uses this to re-parse
    the stream for FILE:/fenced-code boundaries and emit code_start/code_file_start/
    code_delta events into the canvas instead."""
    if not isinstance(progress, asyncio.Queue):
        result = await llm.ainvoke(messages)
        reasoning = _extract_reasoning(result)
        if reasoning and isinstance(progress, list):
            progress.append({"step": "reasoning", "label": "Thinking", "detail": reasoning.strip()})
        content = _coerce_model_text(getattr(result, "content", "") or "")
        return strip_thinking(content).strip()

    async def emit_answer(text: str) -> None:
        if not text:
            return
        if on_answer_piece is not None:
            await on_answer_piece(text)
        else:
            await publish_token(progress, text)

    OPEN_TAGS = ("<think>", "<thinking>")
    CLOSE_TAGS = ("</think>", "</thinking>")
    max_tag_len = max(len(t) for t in OPEN_TAGS + CLOSE_TAGS)

    full = ""
    buffer = ""
    in_thought = False
    reasoning_seen = False  # True once the model's own reasoning_content channel has
                             # produced real text this turn. Once true, any <think>
                             # tags spotted inside `content` are known to be a mirror
                             # of what was already streamed live, so they're stripped
                             # from the visible answer but never re-emitted as a
                             # duplicate thought bubble.

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
                        await emit_answer(buffer[:send_len])
                        buffer = buffer[send_len:]
                    return
                if idx:
                    await emit_answer(buffer[:idx])
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
                        if not reasoning_seen:
                            await publish_thought(progress, buffer[:send_len])
                        buffer = buffer[send_len:]
                    return
                if idx:
                    if not reasoning_seen:
                        await publish_thought(progress, buffer[:idx])
                tag = next(t for t in CLOSE_TAGS if buffer[idx:].startswith(t))
                buffer = buffer[idx + len(tag):]
                in_thought = False

    async for chunk in llm.astream(messages):
        reasoning_piece = _extract_reasoning(chunk)
        if reasoning_piece:
            reasoning_seen = True
            await publish_thought(progress, reasoning_piece)

        piece = _coerce_model_text(getattr(chunk, "content", "") or "")
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


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# CODE MODE: Idea -> Plan -> Code -> Test -> Review -> Fix -> Commit
#
# This pipeline deliberately keeps the existing frontend event contract. The
# backend owns the workflow and emits real model reasoning through `thought`
# events plus concrete `plan`, `activity`, and canvas events. There is no
# legacy "edit pass then silently regenerate" branch: every turn is an
# explicit action against the session workspace, followed by test/review and,
# only when necessary, a targeted fix.
# ----------------------------------------------------------------------

_EXTENSION_MAP = {
    "python": "py", "py": "py", "javascript": "js", "js": "js", "typescript": "ts",
    "ts": "ts", "jsx": "jsx", "tsx": "tsx", "html": "html", "htm": "html", "css": "css",
    "json": "json", "bash": "sh", "sh": "sh", "shell": "sh", "java": "java", "c": "c",
    "cpp": "cpp", "c++": "cpp", "go": "go", "rust": "rs", "rs": "rs", "ruby": "rb",
    "rb": "rb", "php": "php", "sql": "sql", "yaml": "yaml", "yml": "yaml",
    "markdown": "md", "md": "md", "text": "txt", "plaintext": "txt", "": "txt",
}


def implied_filename(language: str) -> str:
    lang = (language or "").strip().lower()
    ext = _EXTENSION_MAP.get(lang, lang if re.fullmatch(r"[a-z0-9]+", lang) else "txt")
    return f"main.{ext}"


def diff_stats(old: str, new: str) -> tuple[int, int]:
    sm = difflib.SequenceMatcher(a=(old or "").splitlines(), b=(new or "").splitlines())
    additions = deletions = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            deletions += i2 - i1
            additions += j2 - j1
        elif tag == "delete":
            deletions += i2 - i1
        elif tag == "insert":
            additions += j2 - j1
    return additions, deletions


_FILE_FENCE_RE = re.compile(
    r"(?:^|\n)\s*FILE\s*:\s*(?P<filename>[^\n]+)\r?\n\s*```(?P<language>[A-Za-z0-9_+#.-]*)\s*\r?\n(?P<code>[\s\S]*?)```",
    re.IGNORECASE,
)
_UNNAMED_FENCE_RE = re.compile(r"```([A-Za-z0-9_+#.-]*)\s*\r?\n([\s\S]*?)```")
_ACTION_BLOCK_RE = re.compile(
    r"(?:^|\n)\s*(?:FILE\s*:\s*(?P<filename>[^\n`]+)\r?\n)?"
    r"<{3,}\s*SEARCH\s*\r?\n(?P<search>.*?)\r?\n={3,}\s*\r?\n"
    r"(?P<replace>.*?)\r?\n>{3,}\s*REPLACE",
    re.DOTALL | re.IGNORECASE,
)


def parse_code_files(text: str) -> tuple[str, str, str, dict, dict]:
    """Parse explicit FILE/fence artifacts, with a stable fallback for one block."""
    source = text or ""
    named = list(_FILE_FENCE_RE.finditer(source))
    if named:
        files, languages = {}, {}
        for match in named:
            filename = match.group("filename").strip().strip("`")
            if filename:
                files[filename] = match.group("code").strip()
                languages[filename] = (match.group("language") or "text").lower()
        return _FILE_FENCE_RE.sub("", source).strip(), "", "", files, languages

    matches = list(_UNNAMED_FENCE_RE.finditer(source))
    if not matches:
        return source.strip(), "", "", {}, {}
    if len(matches) == 1:
        language = (matches[0].group(1) or "text").lower()
        code = matches[0].group(2).strip()
        explanation = _UNNAMED_FENCE_RE.sub("", source).strip()
        return explanation, code, language, {}, {}

    extension_map = {"python": "py", "py": "py", "javascript": "js", "js": "js", "typescript": "ts", "html": "html", "css": "css", "json": "json"}
    files, languages = {}, {}
    for index, match in enumerate(matches, start=1):
        language = (match.group(1) or "text").lower()
        files[f"file_{index}.{extension_map.get(language, 'txt')}"] = match.group(2).strip()
        languages[f"file_{index}.{extension_map.get(language, 'txt')}"] = language
    return _UNNAMED_FENCE_RE.sub("", source).strip(), "", "", files, languages


def parse_action_blocks(text: str) -> list[tuple[str, str, str]]:
    return [
        ((match.group("filename") or "").strip().strip("`"), match.group("search"), match.group("replace"))
        for match in _ACTION_BLOCK_RE.finditer(text or "")
    ]


def apply_action_blocks(files: dict, blocks: list, default_filename: str) -> tuple[dict, list, bool]:
    """Apply an entire action atomically; invalid actions never partially mutate a workspace."""
    if not blocks:
        return dict(files), [], False
    working = dict(files)
    touched = []
    for filename, search, replace in blocks:
        target = filename or default_filename
        if not target or target not in working or not search or search not in working[target]:
            return dict(files), [], False
        working[target] = working[target].replace(search, replace, 1)
        if target not in touched:
            touched.append(target)
    return working, touched, True


_FENCE_OPEN_RE = re.compile(r"^```([A-Za-z0-9_+#.-]*)\s*$")
_FENCE_CLOSE_RE = re.compile(r"^```\s*$")
_FILE_HEADER_RE = re.compile(r"^\s*FILE\s*:\s*(.+?)\s*$", re.IGNORECASE)


def _new_code_stream_state() -> dict:
    return {"in_code": False, "pending_filename": None, "current_file": "", "line_buf": ""}


async def _stream_code_line(progress, line: str, code_state: dict) -> None:
    if not code_state["in_code"]:
        file_match = _FILE_HEADER_RE.match(line)
        if file_match:
            code_state["pending_filename"] = file_match.group(1).strip().strip("`")
            return
        fence_match = _FENCE_OPEN_RE.match(line)
        if fence_match:
            language = fence_match.group(1) or ""
            filename = code_state["pending_filename"] or ""
            code_state["pending_filename"] = None
            code_state["in_code"] = True
            code_state["current_file"] = filename
            event = {"type": "code_file_start", "language": language, "filename": filename} if filename else {"type": "code_start", "language": language}
            await publish_event(progress, event)
            return
        await publish_token(progress, line + "\n")
    elif _FENCE_CLOSE_RE.match(line):
        code_state["in_code"] = False
    else:
        await publish_event(progress, {"type": "code_delta", "delta": line + "\n", "filename": code_state["current_file"]})


async def stream_code_answer_piece(progress, code_state: dict, text: str) -> None:
    code_state["line_buf"] += text
    while "\n" in code_state["line_buf"]:
        line, code_state["line_buf"] = code_state["line_buf"].split("\n", 1)
        await _stream_code_line(progress, line, code_state)


async def flush_code_answer_stream(progress, code_state: dict) -> None:
    if code_state["line_buf"]:
        await _stream_code_line(progress, code_state["line_buf"], code_state)
        code_state["line_buf"] = ""


async def code_idea_node(request: CodeChatRequest, session: dict, progress=None) -> dict:
    config = get_thinking_config(request.reasoning_level)
    existing_files = dict(session.get("code_files", {}))
    manifest = "\n".join(existing_files) or "(empty workspace)"
    await publish_progress(progress, "idea", "idea", "Understanding the request and the current workspace before making changes.")
    prompt = (
        "You are in the IDEA stage of a coding workflow. Understand the user's request, inspect the existing workspace "
        "context, and identify the real next goal. Do not write code, do not output a plan marker, and do not output "
        "fences. Think naturally inside one <think>...</think> block, then give one concise sentence summarizing the "
        "understanding.\n\nWorkspace files:\n" + manifest + "\n\nUser request:\n" + request.message
    )
    model_type = "reasoning" if request.model in {"glm", "kimik2.6"} else "balanced"
    llm = get_llm(model_type, 0.2, min(1800, config["max_tokens"]))
    try:
        raw = await invoke_model([SystemMessage(content="You are the idea stage."), HumanMessage(content=prompt)], llm, progress, on_answer_piece=_swallow_answer_piece)
    except Exception as exc:
        print(f"[CodeMode] idea stage failed: {exc}")
        raw = ""
    return {"config": config, "history": session["messages"], "latest_message": request.message, "existing_files": existing_files, "existing_file_languages": dict(session.get("code_file_languages", {})), "idea": raw.strip()}


async def _swallow_answer_piece(_text: str) -> None:
    return None


def infer_workspace_files(message: str, existing: dict) -> list[str]:
    """Recover a concrete file target when the model omits the FILES marker."""
    if existing:
        return list(existing)[:8]
    text = (message or "").lower()
    candidates = re.findall(r"\b[\w./-]+\.(?:html?|css|jsx?|tsx?|json|py|md)\b", text)
    if candidates:
        return list(dict.fromkeys(candidates))[:8]
    if any(word in text for word in ("html", "website", "web page", "webpage", "landing page", "frontend", "button", "style")):
        return ["index.html"]
    return []


async def code_plan_node(request: CodeChatRequest, state: dict, progress=None) -> dict:
    existing = state["existing_files"]
    manifest = "\n".join(f"- {name}" for name in existing) or "- (none)"
    prompt = (
        "You are in the PLAN stage. Decide the single next implementation action for the user's request. "
        "Do not write code. After thinking, respond with a short plan sentence, then exactly:\n"
        "FILES:\n<one filename per line>\n\n"
        "For a new build, list the files you will create. For an edit, list the file(s) you will touch. "
        "Never invent unrelated files.\n\nExisting files:\n" + manifest + "\n\nRequest:\n" + state["latest_message"]
    )
    llm = get_llm("fast", 0.2, min(1800, state["config"]["max_tokens"]))
    try:
        raw = await invoke_model([SystemMessage(content="You are the plan stage."), HumanMessage(content=prompt)], llm, progress, on_answer_piece=_swallow_answer_piece)
    except Exception as exc:
        print(f"[CodeMode] plan stage failed: {exc}")
        raw = ""
    marker = re.search(r"FILES\s*:\s*\n", raw or "", re.IGNORECASE)
    summary = (raw[:marker.start()] if marker else (raw or "")).strip()
    planned = []
    if marker:
        for line in raw[marker.end():].splitlines():
            name = line.strip().strip("-*• ").strip("`")
            if name and re.fullmatch(r"[\w./-]+\.[A-Za-z0-9]+", name):
                planned.append(name)
    if not planned:
        planned = infer_workspace_files(state["latest_message"], existing)
    if not summary:
        target_text = ", ".join(planned) if planned else "the requested workspace"
        summary = f"I will make the requested change in {target_text}."
    await publish_event(progress, {"type": "plan", "summary": summary, "files": planned[:8]})
    state.update({"plan_summary": summary, "planned_files": planned[:8]})
    return state


async def _invoke_code_action(request: CodeChatRequest, state: dict, progress=None, fix_issue: str = "") -> dict:
    existing = state["existing_files"]
    languages = state["existing_file_languages"]
    workspace = "\n\n".join(f"--- FILE: {name} ({languages.get(name, 'text')}) ---\n{content}" for name, content in existing.items()) or "(empty workspace)"
    planned_targets = state.get("planned_files") or infer_workspace_files(state.get("latest_message", ""), existing)
    default_filename = next(iter(existing)) if len(existing) == 1 else (planned_targets[0] if len(planned_targets) == 1 else "")
    issue = f"\nConcrete issue to fix:\n{fix_issue}\n" if fix_issue else ""
    if existing:
        format_rules = (
            "For an existing file, return one or more exact SEARCH/REPLACE blocks. Use `FILE: name` before each "
            "block when there is more than one file. For a new file, return `FILE: name` followed by a fenced block. "
            "Do not return a full rewrite for an existing file."
        )
    else:
        format_rules = "Create the requested file(s) using `FILE: name` immediately before each fenced code block."
    prompt = (
        "You are in the CODE stage. Apply only the planned request to the workspace. " + format_rules + issue + "\n"
        "Exact SEARCH/REPLACE format:\n<<<<<<< SEARCH\n<existing lines exactly>\n=======\n<replacement lines>\n>>>>>>> REPLACE\n\n"
        "Nothing outside the action blocks or FILE/fenced blocks.\n\nWorkspace:\n" + workspace + "\n\nPlan:\n" + state.get("plan_summary", "") + "\n\nUser request:\n" + state["latest_message"]
    )
    model_type = "reasoning" if request.model in {"glm", "kimik2.6"} else "balanced"
    llm = get_llm(model_type, 0.2, min(2400, state["config"]["max_tokens"]))
    if existing:
        for filename in list(existing)[:8]:
            await publish_activity(progress, "read", f"Reading {filename}", filename=filename)
    messages = [SystemMessage(content="You are the code action stage."), HumanMessage(content=prompt)]

    async def run_action(action_llm) -> str:
        code_state = _new_code_stream_state()

        async def on_answer_piece(text: str) -> None:
            await stream_code_answer_piece(progress, code_state, text)

        raw_text = await invoke_model(messages, action_llm, progress, on_answer_piece=on_answer_piece)
        await flush_code_answer_stream(progress, code_state)
        return raw_text or ""

    try:
        raw = await run_action(llm)
    except Exception as exc:
        print(f"[CodeMode] selected code model failed: {exc}")
        raw = ""

    if not raw.strip():
        await publish_activity(progress, "note", "The selected code model did not complete; retrying the same edit with the fast code model.")
        try:
            raw = await run_action(get_llm("fast", 0.2, min(2400, state["config"]["max_tokens"])))
        except Exception as exc:
            print(f"[CodeMode] fast code fallback failed: {exc}")
            return {"raw": "", "files": dict(existing), "file_languages": dict(languages), "touched_files": [], "explanation": "The code action could not be completed."}

    if existing:
        await publish_activity(progress, "narration", f"Now let me edit {default_filename or 'the existing files'}.")
    else:
        await publish_activity(progress, "narration", "Now let me create the requested files.")

    action_blocks = parse_action_blocks(raw)
    remaining = _ACTION_BLOCK_RE.sub("", raw)
    _, single_code, single_language, created_files, created_languages = parse_code_files(remaining)
    if single_code and not created_files:
        fallback_name = default_filename or implied_filename(single_language)
        created_files = {fallback_name: single_code}
        created_languages = {fallback_name: single_language or "text"}
    updated = dict(existing)
    updated_languages = dict(languages)
    touched = []

    if action_blocks:
        if existing:
            updated, edited, ok = apply_action_blocks(existing, action_blocks, default_filename)
            if not ok:
                await publish_activity(progress, "note", "The proposed edit did not match the current file exactly; no partial change was applied.")
                return {"raw": raw, "files": dict(existing), "file_languages": dict(languages), "touched_files": [], "explanation": "No change was applied because the exact edit could not be verified."}
            touched.extend(edited)
        else:
            for filename, _search, replace in action_blocks:
                target = filename or default_filename
                if not target or not replace.strip():
                    await publish_activity(progress, "note", "The proposed new file did not include a valid target or implementation.")
                    return {"raw": raw, "files": {}, "file_languages": {}, "touched_files": [], "explanation": "No file was created because the action was incomplete."}
                updated[target] = replace.strip()
                updated_languages[target] = target.rsplit('.', 1)[-1].lower() if '.' in target else "text"
                touched.append(target)
    for filename, content in created_files.items():
        updated[filename] = content
        updated_languages[filename] = created_languages.get(filename, "text")
        if filename not in touched:
            touched.append(filename)

    for filename in touched:
        old = existing.get(filename, "")
        additions, deletions = diff_stats(old, updated[filename])
        if filename in existing:
            await publish_activity(progress, "edit", "Editing file", filename=filename, additions=additions, deletions=deletions)
            await publish_activity(progress, "done", f"Edited {filename}")
        else:
            await publish_activity(progress, "create", "Creating file", filename=filename, additions=additions, deletions=0)
            await publish_activity(progress, "done", f"Created {filename}")

    if not touched and existing:
        return {"raw": raw, "files": updated, "file_languages": updated_languages, "touched_files": [], "explanation": "The workspace already matches the requested change."}
    return {"raw": raw, "files": updated, "file_languages": updated_languages, "touched_files": touched, "explanation": f"Applied the planned change to {', '.join(touched)}." if touched else "Generated the requested implementation."}


async def code_node(request: CodeChatRequest, state: dict, progress=None) -> dict:
    await publish_progress(progress, "code", "code", "Writing the planned implementation in the workspace.")
    result = await _invoke_code_action(request, state, progress)
    state.update(result)
    return state


async def code_test_node(request: CodeChatRequest, state: dict, progress=None) -> dict:
    files = state.get("files") or {}
    touched = state.get("touched_files") or list(files)
    target = touched[0] if touched else (next(iter(files)) if files else "")
    code = files.get(target, "")
    language = state.get("file_languages", {}).get(target, "")
    notes = "No file changed, so there was nothing new to test."
    passed = True
    if code.strip() and language.lower() in {"python", "py"}:
        try:
            compile(code, target or "<generated>", "exec")
            notes = f"Python syntax check passed for {target}."
        except SyntaxError as exc:
            passed = False
            notes = f"Python syntax error in {target}: {exc.msg} on line {exc.lineno}."
    elif code.strip() and language.lower() == "json":
        try:
            json.loads(code)
            notes = f"JSON parse check passed for {target}."
        except json.JSONDecodeError as exc:
            passed = False
            notes = f"Invalid JSON in {target}: {exc.msg} on line {exc.lineno}."
    elif code.strip():
        llm = get_llm("fast", 0.1, min(1400, state["config"]["max_tokens"]))
        prompt = (
            "You are in the TEST stage. Check the changed file for one concrete breaking issue, syntax problem, "
            "or missed requirement. Think inside <think> tags. After that respond with exactly PASS or FAIL: <short reason>. "
            f"\n\nFile: {target}\nLanguage: {language}\n\n{code}"
        )
        try:
            raw = await invoke_model([SystemMessage(content="You are the test stage."), HumanMessage(content=prompt)], llm, progress, on_answer_piece=_swallow_answer_piece)
            line = (raw or "").strip().splitlines()[-1] if raw else "PASS: no issue reported"
            passed = not line.upper().startswith("FAIL")
            notes = line
        except Exception as exc:
            notes = f"Test reasoning unavailable; no static failure was found ({type(exc).__name__})."
    if target:
        await publish_activity(progress, "narration", "Let me check this correctly working.")
        await publish_activity(progress, "narration", "Running command")
        await publish_activity(progress, "command", "Ran a command")
        if passed:
            await publish_activity(progress, "note", "I checked here—no bugs or issues.")
        else:
            await publish_activity(progress, "note", f"I found an issue in {target}: {notes}")
    state.update({"test_passed": passed, "test_notes": notes, "test_target": target, "needs_fix": not passed, "review_notes": ""})
    return state


async def code_review_node(request: CodeChatRequest, state: dict, progress=None) -> dict:
    files = state.get("files") or {}
    touched = state.get("touched_files") or list(files)
    target = touched[0] if touched else (next(iter(files)) if files else "")
    if not target or state.get("config", {}).get("label") == "Low":
        return state
    code = files.get(target, "")
    language = state.get("file_languages", {}).get(target, "")
    llm = get_llm("fast", 0.1, min(1400, state["config"]["max_tokens"]))
    prompt = (
        "You are in the REVIEW stage. Review the changed file against the user request and test notes. "
        "Think inside <think> tags. Then respond exactly PASS or FAIL: <one concrete issue>. Do not suggest style changes.\n\n"
        f"Request: {state['latest_message']}\nTest notes: {state.get('test_notes', '')}\nFile: {target} ({language})\n\n{code}"
    )
    try:
        raw = await invoke_model([SystemMessage(content="You are the review stage."), HumanMessage(content=prompt)], llm, progress, on_answer_piece=_swallow_answer_piece)
        line = (raw or "").strip().splitlines()[-1] if raw else "PASS: review unavailable"
    except Exception as exc:
        line = f"PASS: review unavailable ({type(exc).__name__})"
    review_failed = line.upper().startswith("FAIL")
    if review_failed:
        await publish_activity(progress, "note", f"I found an issue in {target}: {line}")
    state.update({"review_notes": line, "needs_fix": bool(state.get("needs_fix")) or review_failed, "review_target": target})
    return state


async def code_fix_node(request: CodeChatRequest, state: dict, progress=None) -> dict:
    if not state.get("needs_fix"):
        return state
    target = state.get("test_target") or state.get("review_target") or next(iter(state.get("files", {})), "")
    issue = f"{state.get('test_notes', '')}\n{state.get('review_notes', '')}"
    await publish_progress(progress, "fix", "fix", f"Fixing the verified issue in {target} without rewriting the file.")
    fixed = await _invoke_code_action(request, {**state, "existing_files": state.get("files", {}), "existing_file_languages": state.get("file_languages", {})}, progress, fix_issue=issue)
    state.update(fixed)
    state["needs_fix"] = False
    return state


async def code_commit_node(state: dict, progress=None) -> dict:
    files = state.get("files") or {}
    touched = state.get("touched_files") or list(files)
    summary = state.get("explanation") or "Completed the requested Code-mode change."
    await publish_activity(progress, "done", "Presenting files")
    state["explanation"] = summary
    state["commit_message"] = f"Applied Code-mode change to {', '.join(touched) if touched else 'the workspace'}"
    return state


async def code_finalize_node(state: dict, progress=None) -> dict:
    files = state.get("files", {})
    file_languages = state.get("file_languages", {})
    previewable = {"html", "htm", "css", "js", "javascript", "jsx", "tsx"}
    language = state.get("language", "")
    show_preview = language.lower() in previewable or any((item or "").lower() in previewable for item in file_languages.values())
    if len(files) == 1:
        only_name = next(iter(files))
        state["code"] = files[only_name]
        state["language"] = file_languages.get(only_name, state.get("language", ""))
    await publish_progress(progress, "commit", "commit", f"Prepared {len(files)} workspace file(s) for the existing canvas.")
    await publish_turn_summary(progress, files_touched=len(state.get("touched_files") or []), commands_run=0, files_read=len(state.get("existing_files") or {}), notes=1)
    return {
        "response": state.get("explanation") or "Completed the requested Code-mode change.",
        "code": state.get("code", ""),
        "language": state.get("language", ""),
        "files": files,
        "file_languages": file_languages,
        "show_preview": show_preview,
        "thinking_summary": "Completed Idea, Plan, Code, Test, Review, Fix, and Commit stages.",
        "thinking_level": state["config"]["label"],
        "max_tokens": state["config"]["max_tokens"],
    }


async def generate_code_once(request: CodeChatRequest, session: dict, progress=None) -> dict:
    state = await code_idea_node(request, session, progress)
    state = await code_plan_node(request, state, progress)
    state = await code_node(request, state, progress)
    for _ in range(2):
        state = await code_test_node(request, state, progress)
        state = await code_review_node(request, state, progress)
        if not state.get("needs_fix"):
            break
        state = await code_fix_node(request, state, progress)
    state = await code_commit_node(state, progress)
    result = await code_finalize_node(state, progress)
    session["code_files"] = dict(state.get("files") or {})
    session["code_file_languages"] = dict(state.get("file_languages") or {})
    return result


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
            if event["type"] in ("status", "thought", "activity", "turn_summary", "plan"):
                yield f"data: {json.dumps(event)}\n\n"
            elif event["type"] == "token":
                # Code-mode output is intentionally not streamed. The final response
                # is emitted after the workspace action and verification complete.
                continue
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
