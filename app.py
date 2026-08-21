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


def get_thinking_config(level: str) -> dict:
    """Return a valid thinking level and its token budget."""
    normalized = (level or DEFAULT_THINKING_LEVEL).strip().lower()
    return THINKING_LEVELS.get(normalized, THINKING_LEVELS[DEFAULT_THINKING_LEVEL])


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


def build_messages(history: List[BaseMessage], thinking_level: str, search_text: str = "") -> List[BaseMessage]:
    config = get_thinking_config(thinking_level)
    system_text = (
        "You are a sharp, honest, genuinely helpful assistant. Answer the user's latest message directly. "
        "Answer directly without exposing private internal workflow narration.\n"
        f"Thinking level: {config['label']} — {config['description']}.\n"
        f"Maximum completion budget: {config['max_tokens']} tokens.\n"
        f"Current date and time: {get_current_datetime_str()}"
    )
    if search_text:
        system_text += "\n\nUse the following web results only when relevant:\n" + search_text
    return [SystemMessage(content=system_text), *history[-6:]]


async def invoke_model(messages: List[BaseMessage], llm: ChatNVIDIA, progress=None) -> str:
    """Invoke once; stream visible answer chunks when a live queue is provided."""
    if not isinstance(progress, asyncio.Queue):
        result = await llm.ainvoke(messages)
        return strip_thinking(getattr(result, "content", "") or "").strip()

    full = ""
    pending = ""
    hidden_open = False
    async for chunk in llm.astream(messages):
        piece = getattr(chunk, "content", "") or ""
        if not piece:
            continue
        full += piece
        pending += piece

        if not hidden_open and ("<think>" in pending or "<thinking>" in pending):
            hidden_open = True
        if hidden_open:
            close_positions = [p for p in (pending.find("</think>"), pending.find("</thinking>")) if p >= 0]
            if not close_positions:
                continue
            close_at = min(close_positions)
            close_tag = "</thinking>" if pending.find("</thinking>") == close_at else "</think>"
            pending = pending[close_at + len(close_tag):]
            hidden_open = False
        await publish_token(progress, pending)
        pending = ""

    if pending and not hidden_open:
        await publish_token(progress, pending)
    return strip_thinking(full).strip()


async def generate_response_once(request: "ChatRequest", session: dict, progress=None) -> str:
    history = session["messages"]
    latest = history[-1].content if history else ""
    config = get_thinking_config(request.thinking_level)
    await publish_progress(progress, "received", "Request received", "The backend accepted the chat request.")
    search_text = ""
    links = []
    images = []
    if needs_web_search(latest):
        query = resolve_search_query(history, latest)
        await publish_progress(progress, "search_start", "Looking up information", f"Searching for current context: {query}")
        (search_text, links), images = await asyncio.gather(web_search(query), web_image_search(query))
        result_count = len(links)
        await publish_progress(progress, "search_done", "Context ready", f"The backend received {result_count} web result(s).")
    else:
        await publish_progress(progress, "search_skip", "No web lookup needed", "The request can be answered from the conversation context.")
    await publish_progress(progress, "model_start", "Composing the answer", f"Invoking the selected model with {config['label']} thinking and a {config['max_tokens']}-token budget.")
    llm = get_llm(request.model_type, request.temperature, config["max_tokens"])
    response = await invoke_model(build_messages(history, request.thinking_level, search_text), llm, progress)
    if not response:
        response = "I apologize, I encountered an issue formulating my answer."
    sources = build_web_sources_markdown(links, images)
    if sources:
        response += sources
        await publish_token(progress, sources)
    await publish_progress(progress, "complete", "Answer ready", "The backend completed the direct response.")
    return response


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
    config = get_thinking_config(reasoning_level)
    system_text = (
        "You are a practical coding assistant. Produce the requested implementation directly in one pass. "
        "Give a concise explanation and put the implementation in a fenced code block with its language. "
        "Do not reveal private chain-of-thought or hidden internal deliberation; provide only a brief, user-facing rationale.\n"
        f"Effort: {config['label']}. Maximum completion budget: {config['max_tokens']} tokens.\n"
        f"Current date and time: {get_current_datetime_str()}"
    )
    return [SystemMessage(content=system_text), *history[-6:]]


def extract_code_artifact(text: str) -> tuple[str, str, str]:
    """Extract one fenced artifact while retaining the user-facing explanation."""
    matches = re.findall(r"```([A-Za-z0-9_+#.-]*)\s*\n([\s\S]*?)```", text or "")
    if not matches:
        return (text or "").strip(), "", ""
    language, code = matches[0]
    explanation = re.sub(r"```[A-Za-z0-9_+#.-]*\s*\n[\s\S]*?```", "", text or "").strip()
    return explanation, code.strip(), (language or "text").lower()


async def generate_code_once(request: CodeChatRequest, session: dict, progress=None) -> dict:
    config = get_thinking_config(request.reasoning_level)
    await publish_progress(progress, "received", "Request received", "The backend accepted the code request.")
    model_type = "reasoning" if request.model in {"glm", "kimik2.6"} else "balanced"
    await publish_progress(progress, "model_start", "Generating code", f"Invoking the selected code model with {config['label']} effort and a {config['max_tokens']}-token budget.")
    llm = get_llm(model_type, 0.2, config["max_tokens"])
    result_text = await invoke_model(build_code_messages(session["messages"], request.reasoning_level), llm, progress)
    raw = result_text
    explanation, code, language = extract_code_artifact(raw)
    if code:
        await publish_progress(progress, "artifact", "Artifact extracted", f"The backend extracted a {language or 'text'} code artifact for the canvas.")
    else:
        await publish_progress(progress, "artifact_missing", "No artifact block found", "The backend received text without a fenced code artifact.")
    await publish_progress(progress, "complete", "Code response ready", "The backend completed the direct code response.")
    return {
        "response": explanation or "I generated the requested code.",
        "code": code,
        "language": language,
        "files": {},
        "file_languages": {},
        "show_preview": language in {"html", "htm", "css", "js", "javascript", "jsx", "tsx"},
        "thinking_summary": f"Generated the implementation directly with {config['label']} effort; no planning or revision pass was run.",
        "thinking_level": config["label"],
        "max_tokens": config["max_tokens"],
    }


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
            if event["type"] == "status":
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
            if event["type"] == "status":
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
    if result.get("code"):
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
