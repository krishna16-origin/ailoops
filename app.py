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


def _extract_reasoning(obj) -> str:
    """Pull live reasoning text out of langchain_nvidia_ai_endpoints' normalized
    channel. Per the library's own docs, additional_kwargs['reasoning_content'] is
    ALWAYS populated as a unified reasoning channel no matter which raw format the
    underlying NIM actually used (inline <think> tags in content, a dedicated
    reasoning_content field, or a reasoning field) — so this is the one reliable
    place to read a model's real chain-of-thought from, chunk by chunk."""
    kwargs = getattr(obj, "additional_kwargs", None) or {}
    return kwargs.get("reasoning_content") or kwargs.get("reasoning") or ""


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
        content = getattr(result, "content", "") or ""
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


# ----------------------------------------------------------------------
# CODE MODE: session memory + incremental (diff-based) edits
#
# Root cause of "it rewrites the whole file for every tiny change": session
# history only ever kept the one-line explanation ("I generated the requested
# code.") in the AIMessage — the actual file contents were never persisted
# anywhere, so every follow-up turn started from zero knowledge of what had
# already been built. code_understand_node below now also loads the session's
# real file map, and code_compose_node uses it to ask for a targeted
# SEARCH/REPLACE patch instead of a full regenerate whenever a prior version
# of the file already exists. This entire mechanism is Code-mode-only — plain
# /chat never touches code_files.
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
    """A stable filename for a single unnamed code block, so a follow-up turn
    has something real to target with a SEARCH/REPLACE edit instead of the
    file identity being lost the moment the response leaves the model."""
    lang = (language or "").strip().lower()
    ext = _EXTENSION_MAP.get(lang, lang if re.fullmatch(r"[a-z0-9]+", lang) else "txt")
    return f"main.{ext}"


def diff_stats(old: str, new: str) -> tuple[int, int]:
    """Real added/removed line counts between two versions of a file — the same
    classification `git diff --stat` uses — for the activity trace's +/- badges.
    Never a token-count heuristic or a guess."""
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


_REWRITE_KEYWORDS = (
    "rewrite everything", "rewrite the whole", "start over", "start from scratch",
    "from scratch", "redo the whole", "redesign completely", "throw away",
    "completely different", "scrap it", "rebuild it", "new version of the whole",
    "build a new website", "build a new site", "create a new website", "create a new site",
    "make a new website", "make a new site",
)


def wants_full_rewrite(message: str) -> bool:
    """Heuristic: does the request explicitly call for replacing the whole
    project rather than a targeted change? Deliberately biased toward False
    (the cheaper diff path) — a false negative just costs one harmless
    SEARCH/REPLACE attempt that fails closed and falls back to a full
    regenerate anyway; a false positive pays for a full regenerate when a
    small patch would have done."""
    text = (message or "").lower()
    return any(kw in text for kw in _REWRITE_KEYWORDS)


_BUILD_SCOPE_KEYWORDS = (
    "website", "web site", "webpage", "web page", "landing page", "app",
    "application", "project", "dashboard", "portfolio", "game", "platform",
    "system", "tool", "site",
)


def looks_like_a_build(message: str) -> bool:
    """Whether a brand-new request looks substantial enough to warrant a short
    plan before writing code — vs. a quick one-off snippet where a plan step
    would just add latency without helping anyone."""
    text = (message or "").lower()
    if any(kw in text for kw in _BUILD_SCOPE_KEYWORDS):
        return True
    return len(text.split()) >= 18


EDIT_BLOCK_RE = re.compile(
    r"(?:FILE\s*:\s*(?P<filename>[^\n`]+)\n)?"
    r"<{3,}\s*SEARCH\s*\n(?P<search>.*?)\n={3,}\s*\n(?P<replace>.*?)\n>{3,}\s*REPLACE",
    re.DOTALL | re.IGNORECASE,
)


def parse_edit_blocks(text: str) -> list[tuple[str, str, str]]:
    """Parses SEARCH/REPLACE blocks out of model output into
    (filename_or_'', search, replace) tuples. filename is '' when the block
    has no leading `FILE:` header — used for single-file edits/reviews where
    the target is already known some other way."""
    return [
        ((m.group("filename") or "").strip().strip("`"), m.group("search"), m.group("replace"))
        for m in EDIT_BLOCK_RE.finditer(text or "")
    ]


def apply_edit_blocks(files: dict, blocks: list, default_filename: str) -> tuple[dict, list, bool]:
    """Applies parsed SEARCH/REPLACE blocks to a {filename: content} map.
    Fails closed: if a block names an unknown file, has an empty (ambiguous)
    SEARCH, or its SEARCH text isn't found verbatim, the WHOLE apply is
    rejected (ok=False) rather than landing a partial or garbled edit — the
    caller then falls back to a full regenerate instead of guessing."""
    if not blocks:
        return files, [], False
    working = dict(files)
    touched = []
    for filename, search, replace in blocks:
        target = filename or default_filename
        if not target or target not in working:
            return files, [], False
        if search == "":
            return files, [], False
        if search not in working[target]:
            return files, [], False
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
    """Consume one complete line of the model's (post-thinking) answer text and
    turn it into the right live event. A `FILE: name` header is swallowed and
    remembered; a fenced-code open/close line flips between prose and code; prose
    lines stream to the chat bubble as plain 'token' events exactly as before,
    while code lines now stream straight into the canvas as 'code_delta' events —
    the event type the frontend's code canvas already knew how to render live,
    that the backend was simply never sending, so the canvas only ever filled in
    all at once at the very end instead of live as the model wrote it."""
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
            if filename:
                await publish_event(progress, {"type": "code_file_start", "language": language, "filename": filename})
            else:
                await publish_event(progress, {"type": "code_start", "language": language})
            return
        await publish_token(progress, line + "\n")
    else:
        if _FENCE_CLOSE_RE.match(line):
            code_state["in_code"] = False
            return
        await publish_event(progress, {"type": "code_delta", "delta": line + "\n", "filename": code_state["current_file"]})


async def stream_code_answer_piece(progress, code_state: dict, text: str) -> None:
    """Feed a live slice of raw answer text into the line-buffered code parser
    above. Model output doesn't arrive aligned to line boundaries, so this holds
    onto whatever's left of the current line between calls."""
    code_state["line_buf"] += text
    while "\n" in code_state["line_buf"]:
        line, code_state["line_buf"] = code_state["line_buf"].split("\n", 1)
        await _stream_code_line(progress, line, code_state)


async def flush_code_answer_stream(progress, code_state: dict) -> None:
    """Called once the model's turn is fully done — the last line of a response
    usually has no trailing newline, so it never gets processed inside the loop
    above and has to be flushed explicitly."""
    if code_state["line_buf"]:
        await _stream_code_line(progress, code_state["line_buf"], code_state)
        code_state["line_buf"] = ""


async def code_understand_node(request: CodeChatRequest, session: dict, progress=None) -> dict:
    config = get_thinking_config(request.reasoning_level)
    latest_message = session["messages"][-1].content if session["messages"] else request.message
    excerpt = request_excerpt(latest_message)
    # The real project state from every prior turn in this session — empty on
    # the very first Code-mode message, populated from here on by
    # generate_code_once's session persistence at the end of every turn.
    existing_files = dict(session.get("code_files", {}))
    existing_file_languages = dict(session.get("code_file_languages", {}))
    if existing_files:
        await publish_progress(progress, "code_understand_node", "code_understand_node", f"Read the request against the existing project ({', '.join(existing_files)}): “{excerpt}”")
    else:
        await publish_progress(progress, "code_understand_node", "code_understand_node", f"Read the build request and identified the requested result: “{excerpt}”")
    return {
        "config": config,
        "history": session["messages"],
        "latest_message": latest_message,
        "existing_files": existing_files,
        "existing_file_languages": existing_file_languages,
    }


async def _swallow_answer_piece(_text: str) -> None:
    """Used as invoke_model's on_answer_piece for internal sub-calls (plan,
    edit-diff, review) whose visible output must never leak into the chat
    bubble as ordinary answer tokens — only their live reasoning trace (real
    'thought' events) should reach the user while they run."""
    return None


async def code_plan_node(request: CodeChatRequest, state: dict, progress=None) -> dict:
    """Runs ONLY on a brand-new build in this session (no existing files yet)
    for a request that looks substantial — mirrors "think, then plan the
    files, then write code" instead of jumping straight to code with no
    stated intent. One short real LLM call: its own reasoning streams live
    into the Thinking pane exactly like every other call here, never a
    canned status line. Published as a 'plan' event so the frontend can show
    the manifest before any code starts streaming. Silently skipped (falls
    through to code_compose_node as before) for edits, tiny one-off asks, or
    if anything about the call goes wrong — a missing plan should never block
    the actual build."""
    if state["existing_files"] or not looks_like_a_build(state["latest_message"]):
        return state
    config = state["config"]
    llm = get_llm("fast", 0.2, min(1200, config["max_tokens"]))
    prompt = (
        "A user just asked a coding assistant to build something. Before any code is "
        "written, decide what you'll build and which file(s) it needs. Think inside a "
        "single <think>...</think> block about the approach, briefly. Then, after the "
        "closing </think> tag, write 1-2 short sentences describing what you'll build, "
        "then on a new line write exactly 'FILES:' followed by one bare filename per "
        "line (e.g. index.html) for every file the build will need — no paths, no "
        "commentary, nothing else.\n\n"
        f"Request: {state['latest_message']}"
    )
    messages = [SystemMessage(content="You plan briefly before you build."), HumanMessage(content=prompt)]
    try:
        raw = await invoke_model(messages, llm, progress, on_answer_piece=_swallow_answer_piece)
    except Exception as e:
        print(f"[CodeMode] code_plan_node failed, skipping plan: {e}")
        return state
    if not raw:
        return state
    marker = re.search(r"FILES\s*:\s*\n", raw, re.IGNORECASE)
    summary = (raw[: marker.start()] if marker else raw).strip()
    files: list[str] = []
    if marker:
        for line in raw[marker.end():].splitlines():
            name = line.strip().strip("-*•").strip().strip("`")
            if name and re.match(r"^[\w.\-]+\.[A-Za-z0-9]+$", name):
                files.append(name)
    if not summary and not files:
        return state
    await publish_event(progress, {"type": "plan", "summary": summary, "files": files[:8]})
    state["planned_files"] = files[:8]
    return state


async def run_code_edit_pass(request: CodeChatRequest, state: dict, llm, progress=None) -> Optional[dict]:
    """The heart of "edit the same file instead of rewriting all the code":
    asks the model for one or more small SEARCH/REPLACE blocks against the
    project's CURRENT file contents (loaded from session), applies them
    locally with plain string replacement, and only pays output tokens for
    the lines that actually changed. Also lets the model add a brand-new file
    via a `FILE:`+fenced block in the same response (e.g. adding a
    requirements.txt to an existing build). Returns None (never a partial or
    guessed result) if the model didn't produce anything usable — the caller
    then falls back to code_compose_node's original full-generate path."""
    existing_files = state["existing_files"]
    existing_file_languages = state["existing_file_languages"]
    default_filename = next(iter(existing_files)) if len(existing_files) == 1 else ""
    file_list = "\n\n".join(
        f"--- FILE: {name} ({existing_file_languages.get(name, 'text')}) ---\n{content}"
        for name, content in existing_files.items()
    )
    filename_note = (
        f"There is exactly one existing file, '{default_filename}' — SEARCH/REPLACE blocks for it don't need a FILE: header."
        if default_filename else
        "There are multiple existing files — put a `FILE: <name>` line immediately before every SEARCH/REPLACE block, naming which file it targets."
    )
    prompt = (
        "The project below already exists. Make ONLY the change described by the user's "
        "latest message below — do not rewrite files that don't need to change, and do "
        "not regenerate anything beyond the specific lines that actually need to change.\n\n"
        f"{filename_note}\n\n"
        "Respond with one or more blocks in exactly this shape, and nothing else — no "
        "commentary outside the blocks:\n\n"
        "<<<<<<< SEARCH\n<exact existing lines, copied character-for-character>\n=======\n"
        "<the new lines that replace them>\n>>>>>>> REPLACE\n\n"
        "Rules:\n"
        "- Every SEARCH block must match the existing file EXACTLY, including whitespace.\n"
        "- Keep each SEARCH block as short as possible while still being unique in its file.\n"
        "- Use several blocks for several separate changes, even across different files.\n"
        "- To insert code, include a short unique anchor line in SEARCH and put that anchor "
        "plus the new lines in REPLACE.\n"
        "- To delete code, put it in SEARCH and leave REPLACE empty.\n"
        "- If (and only if) the request needs a brand-new file that doesn't exist yet, "
        "instead write a line `FILE: path/to/new_file.ext` followed by a fenced code "
        "block containing that file's full contents.\n\n"
        f"Existing project:\n{file_list}\n\n"
        f"User's latest request: {state['latest_message']}"
    )
    messages = [
        SystemMessage(content="You make precise, minimal, targeted code edits to an existing project — never a full rewrite when a small patch will do."),
        HumanMessage(content=prompt),
    ]
    try:
        raw = await invoke_model(messages, llm, progress, on_answer_piece=_swallow_answer_piece)
    except Exception as e:
        print(f"[CodeMode] run_code_edit_pass failed, falling back to full generate: {e}")
        return None
    if not raw:
        return None

    edit_blocks = parse_edit_blocks(raw)
    remaining_text = EDIT_BLOCK_RE.sub("", raw)
    _, _, _, new_files, new_file_languages = extract_code_artifact(remaining_text)

    updated_files = dict(existing_files)
    updated_languages = dict(existing_file_languages)
    touched: list[str] = []

    if edit_blocks:
        merged, edited_names, ok = apply_edit_blocks(existing_files, edit_blocks, default_filename)
        if not ok:
            return None  # fail closed — an edit that doesn't apply cleanly is never guessed at
        updated_files.update(merged)
        touched.extend(edited_names)

    for filename, content in new_files.items():
        updated_files[filename] = content
        updated_languages[filename] = new_file_languages.get(filename, "text")
        if filename not in touched:
            touched.append(filename)

    if not touched:
        return None  # nothing usable came back — let the caller fall back to a full generate

    for filename in touched:
        old_content = existing_files.get(filename, "")
        new_content = updated_files[filename]
        additions, deletions = diff_stats(old_content, new_content)
        if filename in existing_files:
            await publish_activity(progress, "edit", f"Edited {filename}", filename=filename, additions=additions, deletions=deletions)
        else:
            await publish_activity(progress, "create", f"Added {filename}", filename=filename, additions=additions, deletions=0)

    explanation = (
        f"Applied a targeted edit to {touched[0]} without rewriting the rest of the code."
        if len(touched) == 1 else
        f"Applied targeted edits to {', '.join(touched)} without rewriting the rest of the code."
    )
    return {
        "files": updated_files,
        "file_languages": updated_languages,
        "code": "",
        "language": "",
        "explanation": explanation,
        "touched_files": touched,
    }


async def code_compose_node(request: CodeChatRequest, state: dict, progress=None) -> dict:
    config = state["config"]
    model_type = "reasoning" if request.model in {"glm", "kimik2.6"} else "balanced"
    llm = get_llm(model_type, 0.2, config["max_tokens"])

    existing_files = state["existing_files"]
    edit_eligible = bool(existing_files) and not wants_full_rewrite(state["latest_message"])

    if edit_eligible:
        await publish_progress(progress, "code_compose_node", "code_compose_node", f"Existing project found ({', '.join(existing_files)}); editing in place instead of rewriting.")
        edit_result = await run_code_edit_pass(request, state, llm, progress)
        if edit_result is not None:
            state.update(edit_result)
            state["edit_mode"] = True
            return state
        await publish_progress(progress, "code_compose_node", "code_compose_node", "The targeted edit didn't apply cleanly; regenerating the file(s) in full instead.")

    await publish_progress(progress, "code_compose_node", "code_compose_node", f"Invoking the code model with {config['label']} effort and a {config['max_tokens']}-token budget.")

    code_state = _new_code_stream_state()

    async def on_answer_piece(text: str) -> None:
        await stream_code_answer_piece(progress, code_state, text)

    state["raw"] = await invoke_model(
        build_code_messages(state["history"], request.reasoning_level), llm, progress,
        on_answer_piece=on_answer_piece,
    )
    await flush_code_answer_stream(progress, code_state)
    state["edit_mode"] = False
    return state


async def code_extract_artifact_node(state: dict, progress=None) -> dict:
    if state.get("edit_mode"):
        # Already parsed and applied as targeted SEARCH/REPLACE edits inside
        # run_code_edit_pass (including that pass's own activity events) —
        # nothing left to extract here.
        return state

    explanation, code, language, files, file_languages = extract_code_artifact(state["raw"])
    state.update({"explanation": explanation, "code": code, "language": language, "files": files, "file_languages": file_languages})
    existing_files = state.get("existing_files", {})

    if files:
        await publish_progress(progress, "code_extract_artifact_node", "code_extract_artifact_node", f"Separated {len(files)} named files for the canvas: {', '.join(files)}")
        for filename, content in files.items():
            if filename in existing_files:
                # A full-generate that landed on a filename that already existed
                # (e.g. the edit-pass above failed and this is the fallback) — show
                # it as a real edit with real diff stats, not a fake "create".
                additions, deletions = diff_stats(existing_files[filename], content)
                await publish_activity(progress, "edit", f"Rewrote {filename}", filename=filename, additions=additions, deletions=deletions)
            else:
                line_count = len(content.splitlines()) if content else 0
                await publish_activity(progress, "create", f"Created {filename}", filename=filename, additions=line_count, deletions=0)
    elif code:
        await publish_progress(progress, "code_extract_artifact_node", "code_extract_artifact_node", f"Extracted the {language or 'text'} implementation block for the canvas.")
        implied_name = implied_filename(language)
        if implied_name in existing_files:
            additions, deletions = diff_stats(existing_files[implied_name], code)
            await publish_activity(progress, "edit", f"Rewrote {implied_name}", filename=implied_name, additions=additions, deletions=deletions)
        else:
            line_count = len(code.splitlines()) if code else 0
            await publish_activity(progress, "create", f"Created {implied_name}", filename=implied_name, additions=line_count, deletions=0)
    else:
        await publish_progress(progress, "code_extract_artifact_node", "code_extract_artifact_node", "No fenced artifact was returned, so the explanation remains the visible result.")
    return state


async def code_merge_node(state: dict, progress=None) -> dict:
    """Reconciles this turn's file(s) against the rest of the project so a
    turn that only touches index.html never silently drops requirements.txt
    (or any other file) from the response — the canvas is always reconciled
    against the FULL current project, not just what changed this turn. Also
    normalizes a single unnamed code block into the same {filename: content}
    shape as a named build, using a stable implied filename, so every turn
    from here on (including this one, if edited again later) has a real name
    to target."""
    existing_files = state.get("existing_files", {})
    existing_langs = state.get("existing_file_languages", {})

    if state.get("edit_mode"):
        # run_code_edit_pass already merged its changes into the full existing
        # file map — state["files"]/["file_languages"] are already complete.
        merged_files = dict(state.get("files") or {})
        merged_langs = dict(state.get("file_languages") or {})
    else:
        this_turn_files = dict(state.get("files") or {})
        this_turn_langs = dict(state.get("file_languages") or {})
        if not this_turn_files and state.get("code"):
            name = implied_filename(state.get("language", ""))
            this_turn_files = {name: state["code"]}
            this_turn_langs = {name: state.get("language", "")}
        state["touched_files"] = list(this_turn_files.keys())
        merged_files = {**existing_files, **this_turn_files}
        merged_langs = {**existing_langs, **this_turn_langs}

    state["files"] = merged_files
    state["file_languages"] = merged_langs
    # Keep the top-level code/language fields populated for a single-file
    # result too, so anything that only looks at those two fields still works.
    if len(merged_files) == 1:
        only_name = next(iter(merged_files))
        state["code"] = merged_files[only_name]
        state["language"] = merged_langs.get(only_name, "")
    return state


async def code_review_node(state: dict, progress=None) -> dict:
    """One lightweight, real self-review pass over the file(s) this turn
    actually touched — a second small LLM call reads the finished code back
    and is asked to point out one concrete bug worth flagging, if there is
    one. If it finds one, the fix is applied the exact same way as any other
    Code-mode edit: a targeted SEARCH/REPLACE, never a full rewrite. Skipped
    when there's nothing to review, or at the 'Low' thinking level, to keep
    quick asks quick."""
    files = state.get("files") or {}
    touched = state.get("touched_files") or []
    if not files or not touched or state["config"]["label"] == "Low":
        return state
    target = touched[0]
    code = files.get(target, "")
    if not code.strip():
        return state

    await publish_activity(progress, "note", f"Checking {target} for errors…")
    language = state.get("file_languages", {}).get(target, "")
    prompt = (
        f"Review the following{f' {language}' if language else ''} file for ONE concrete, "
        "obvious bug — a syntax error, a broken reference, a mismatched tag, an "
        "off-by-one, something that would actually break. Don't nitpick style or "
        "suggest improvements.\n\n"
        "If you find a real bug, respond with EXACTLY one block fixing it, in this "
        "shape and nothing else:\n"
        "<<<<<<< SEARCH\n<exact existing lines>\n=======\n<the fixed lines>\n>>>>>>> REPLACE\n\n"
        "If the file looks correct, respond with exactly: NONE\n\n"
        f"File: {target}\n\n{code}"
    )
    llm = get_llm("fast", 0.1, 1200)
    try:
        raw = await invoke_model([HumanMessage(content=prompt)], llm, progress, on_answer_piece=_swallow_answer_piece)
    except Exception as e:
        print(f"[CodeMode] code_review_node failed, skipping review: {e}")
        return state
    if not raw or raw.strip().upper().startswith("NONE"):
        return state

    blocks = parse_edit_blocks(raw)
    updated, touched_by_fix, ok = apply_edit_blocks({target: code}, blocks, target)
    if not ok or updated[target] == code:
        return state

    new_code = updated[target]
    additions, deletions = diff_stats(code, new_code)
    files[target] = new_code
    state["files"] = files
    if len(files) == 1:
        state["code"] = new_code
    await publish_activity(progress, "edit", f"Found and fixed a bug in {target}", filename=target, additions=additions, deletions=deletions)
    return state


async def code_finalize_node(state: dict, progress=None) -> dict:
    config = state["config"]
    files = state.get("files", {})
    file_languages = state.get("file_languages", {})
    previewable = {"html", "htm", "css", "js", "javascript", "jsx", "tsx"}
    show_preview = state["language"] in previewable or any(language in previewable for language in file_languages.values())
    await publish_progress(progress, "code_finalize_node", "code_finalize_node", f"Prepared {len(files) if files else 1} separate artifact file(s) and preview metadata for the frontend.")
    touched = state.get("touched_files") or list(files.keys())
    files_touched = len(touched) if touched else (1 if state.get("code") else 0)
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
    state = await code_plan_node(request, state, progress)
    state = await code_compose_node(request, state, progress)
    state = await code_extract_artifact_node(state, progress)
    state = await code_merge_node(state, progress)
    state = await code_review_node(state, progress)
    result = await code_finalize_node(state, progress)
    # Persist the full, current project back onto the session so the NEXT turn
    # can edit these exact files in place instead of starting from nothing —
    # this is the piece that makes "editing the same file, like Claude.ai"
    # possible across turns at all.
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
            if event["type"] in ("status", "thought", "activity", "turn_summary", "code_start", "code_file_start", "code_delta", "plan"):
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
