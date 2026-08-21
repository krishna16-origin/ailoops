# ==================================================
# CONFIGURATION
# ==================================================
import os
import json
import re
import difflib
import random
import asyncio
import contextvars
from datetime import datetime, timezone
from typing import TypedDict, List, Dict, Any, Optional, Annotated

from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field

# LangChain & LangGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Web Search — Tavily API (requires TAVILY_API_KEY)
from tavily import TavilyClient

# Load environment variables
load_dotenv()

if not os.getenv("NVIDIA_API_KEY"):
    print("WARNING: NVIDIA_API_KEY not found in environment. The API calls will fail.")

if not os.getenv("TAVILY_API_KEY"):
    print("WARNING: TAVILY_API_KEY not found in environment. Web search will be disabled.")

_tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) if os.getenv("TAVILY_API_KEY") else None

# ==================================================
# THINKING LEVELS & TOKEN BUDGETS
# ==================================================

THINKING_LEVELS = {
    "low": {"label": "Low", "max_tokens": 8000, "description": "Quick, focused thinking"},
    "medium": {"label": "Medium", "max_tokens": 16000, "description": "Balanced analysis"},
    "high": {"label": "High", "max_tokens": 24000, "description": "Deep reasoning"},
    "extra": {"label": "Extra", "max_tokens": 32000, "description": "Comprehensive analysis"},
    "max": {"label": "Max", "max_tokens": 40000, "description": "Exhaustive reasoning"},
}

DEFAULT_THINKING_LEVEL = "medium"

# ==================================================
# MODELS (Pydantic for Structured Output)
# ==================================================

class GoalExtraction(BaseModel):
    main_goal: str = Field(description="The primary objective of the user")
    hidden_intent: str = Field(description="Any implied or implicit needs")
    constraints: List[str] = Field(description="Rules or restrictions to follow")
    requested_output: str = Field(description="The format the user wants the answer in")
    missing_information: List[str] = Field(description="What we need to ask the user, if anything")

class ExecutorOutput(BaseModel):
    response: str = Field(description="The generated draft or answer")
    reasoning_summary: str = Field(description="Why this response is correct and helpful")
    confidence: float = Field(description="Confidence from 0.0 to 1.0")

class ReflectorOutput(BaseModel):
    quality: str = Field(description="Assessment of the response quality")
    correctness: str = Field(description="Is the response factually correct?")
    hallucination_risk: str = Field(description="Is there any fabricated info?")
    improvements: str = Field(description="Actionable advice to improve the response")

# ==================================================
# LANGGRAPH STATE
# ==================================================

class AgentState(TypedDict):
    messages: List[BaseMessage]
    goal: str
    hidden_intent: str
    constraints: List[str]
    thinking_level: str
    reflection: str
    completion_score: int
    executor_reasoning: str
    response: str
    conversation_summary: str
    model_type: str
    temperature: float
    web_search_query: str
    web_search_results: str
    web_search_links: List[Dict[str, str]]
    web_search_images: List[Dict[str, str]]

# ==================================================
# MEMORY (In-Memory Sessions)
# ==================================================

sessions: Dict[str, Dict[str, Any]] = {}

async def summarize_memory(messages: List[BaseMessage], llm: ChatNVIDIA) -> List[BaseMessage]:
    """Summarizes older messages if conversation history grows too long."""
    if len(messages) <= 10:
        return messages
    
    # Keep the last 4 messages, summarize the rest
    recent_messages = messages[-4:]
    older_messages = messages[:-4]
    
    history_str = "\n".join([f"{m.type}: {m.content}" for m in older_messages])
    prompt = f"Provide a concise summary of the following conversation history. Retain key facts and user preferences:\n\n{history_str}"
    
    try:
        summary_response = await llm.ainvoke(prompt)
        summary_msg = SystemMessage(content=f"Previous conversation summary: {summary_response.content}")
        return [summary_msg] + recent_messages
    except Exception as e:
        print(f"Summarization failed: {e}")
        return messages

# ==================================================
# MODEL ROUTER & HELPER
# ==================================================

def get_llm(model_type: str, temperature: float = 0.7, max_tokens: int = 16384) -> ChatNVIDIA:
    """Routes to the correct NVIDIA model based on user selection."""
    model_name = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    
    model_type_clean = model_type.strip().lower()
    if model_type_clean == "fast":
        model_name = "deepseek-ai/deepseek-v4-pro"
    elif model_type_clean == "reasoning":
        model_name = "nvidia/nemotron-3-ultra-550b-a55b"
        
    return ChatNVIDIA(model=model_name, temperature=temperature, max_tokens=max_tokens, timeout=120)

def strip_thinking(text: str) -> str:
    """Removes <think>...</think> reasoning blocks some models emit before the real answer."""
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    return text.strip()

# ==================================================
# CURRENT DATE & TIME
# ==================================================

def get_current_datetime_str() -> str:
    """Returns the current UTC date/time, human-readable."""
    return datetime.now(timezone.utc).strftime("%A, %B %d, %Y, %I:%M %p UTC")

# ==================================================
# WEB SEARCH (Tavily API)
# ==================================================

WEB_SEARCH_TIMEOUT = 9.0
WEB_SEARCH_MAX_RESULTS = 5
WEB_SEARCH_RETRIES = 2
WEB_SEARCH_DEPTH = "basic"

def _run_web_search_sync(query: str, max_results: int) -> list:
    """Blocking Tavily text search."""
    if _tavily_client is None:
        return []
    response = _tavily_client.search(query, max_results=max_results, search_depth=WEB_SEARCH_DEPTH)
    return [
        {"title": r.get("title", ""), "href": r.get("url", ""), "body": r.get("content", "")}
        for r in (response.get("results") or [])
    ]

async def web_search(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS) -> tuple[str, list]:
    """Runs a Tavily web search and returns (formatted_text, raw_results)."""
    query = (query or "").strip()
    if not query:
        return "", []
    if _tavily_client is None:
        print("Web search skipped: TAVILY_API_KEY not configured.")
        return "", []

    last_error: Optional[Exception] = None
    for attempt in range(WEB_SEARCH_RETRIES):
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(_run_web_search_sync, query, max_results),
                timeout=WEB_SEARCH_TIMEOUT,
            )
            if results:
                lines = [f"Web search results for '{query}' (fetched just now — current date/time is {get_current_datetime_str()}):"]
                for i, r in enumerate(results, start=1):
                    title = (r.get("title") or "").strip()
                    snippet = (r.get("body") or "").strip()
                    url = (r.get("href") or "").strip()
                    lines.append(f"{i}. {title}\n   {snippet}\n   Source: {url}")
                return "\n".join(lines), results
            print(f"Web search for '{query}' returned zero results (no retry).")
            return "", []
        except asyncio.TimeoutError as e:
            last_error = e
            print(f"Web search attempt {attempt + 1}/{WEB_SEARCH_RETRIES} for '{query}' timed out.")
        except Exception as e:
            last_error = e
            print(f"Web search attempt {attempt + 1}/{WEB_SEARCH_RETRIES} for '{query}' failed: {e}")

        if attempt < WEB_SEARCH_RETRIES - 1:
            await asyncio.sleep(1.0 + random.random())

    print(f"Web search for '{query}' exhausted all retries. Last error: {last_error}")
    return "", []

# ==================================================
# WEB IMAGE SEARCH
# ==================================================

WEB_IMAGE_SEARCH_MAX_RESULTS = 4

def _run_web_image_search_sync(query: str, max_results: int) -> list:
    """Blocking Tavily image search."""
    if _tavily_client is None:
        return []
    response = _tavily_client.search(
        query,
        max_results=max_results,
        search_depth=WEB_SEARCH_DEPTH,
        include_images=True,
        include_image_descriptions=True,
    )
    images = (response.get("images") or [])[:max_results]
    return [
        {"title": (img.get("description") or "Image"), "image": img.get("url", ""), "url": img.get("url", "")}
        for img in images
    ]

async def web_image_search(query: str, max_results: int = WEB_IMAGE_SEARCH_MAX_RESULTS) -> list:
    """Chat-mode-only Tavily image search."""
    query = (query or "").strip()
    if not query or _tavily_client is None:
        return []
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(_run_web_image_search_sync, query, max_results),
            timeout=WEB_SEARCH_TIMEOUT,
        )
    except asyncio.TimeoutError:
        print(f"Image search for '{query}' timed out.")
        return []
    except Exception as e:
        print(f"Image search failed for '{query}': {e}")
        return []
    return results or []

def _escape_md_brackets(text: str) -> str:
    """Markdown link/image text can't safely contain raw '[' ']'."""
    return (text or "").replace("[", "").replace("]", "").strip()

def build_web_sources_markdown(links: list, images: list) -> str:
    """Builds a Markdown block listing source links and images."""
    if not links and not images:
        return ""

    parts = ["\n\n---"]

    if links:
        parts.append("**Sources**")
        source_lines = []
        for i, item in enumerate(links, start=1):
            title = _escape_md_brackets(item.get("title") or item.get("href") or f"Source {i}")
            url = (item.get("href") or "").strip()
            if not url:
                continue
            source_lines.append(f"{i}. [{title}]({url})")
        if source_lines:
            parts.append("\n".join(source_lines))

    if images:
        parts.append("**Images**")
        image_lines = []
        for item in images:
            title = _escape_md_brackets(item.get("title") or "Image")
            image_url = (item.get("image") or "").strip()
            source_url = (item.get("url") or image_url or "").strip()
            if not image_url:
                continue
            image_lines.append(f"[![{title}]({image_url})]({source_url})")
        if image_lines:
            parts.append(" ".join(image_lines))

    return "\n\n".join(parts) if len(parts) > 1 else ""

# ==================================================
# WEB SEARCH DETECTION
# ==================================================

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
    """True if the message looks like it needs current/external information."""
    text = (message or "").strip().lower()
    if not text:
        return False
    if text.startswith(_EXPLICIT_SEARCH_PREFIXES):
        return True
    if text.rstrip("?!.") in _BARE_SEARCH_COMMANDS:
        return True
    if any(kw in text for kw in _WEB_SEARCH_KEYWORDS):
        return True
    if re.search(r"\b20[2-9]\d\b", text):
        return True
    return False

def extract_search_query(message: str) -> str:
    """Strips an explicit search prefix off the front of a message."""
    text = (message or "").strip()
    lowered = text.lower()
    for prefix in _EXPLICIT_SEARCH_PREFIXES:
        if lowered.startswith(prefix):
            return text[len(prefix):].strip()
    return text

def resolve_search_query(messages: List[BaseMessage], latest_user_message: str) -> str:
    """Figures out what to actually search for."""
    query = extract_search_query(latest_user_message)
    normalized = query.strip().lower().rstrip("?!.")
    if normalized in _BARE_SEARCH_COMMANDS:
        for m in reversed(messages[:-1]):
            content = (m.content or "").strip()
            if isinstance(m, HumanMessage) and content and content.lower().rstrip("?!.") not in _BARE_SEARCH_COMMANDS:
                return content
    return query

# ==================================================
# STREAMING SUPPORT
# ==================================================

STREAM_DELIM = "###FINAL_ANSWER###"
_current_token_queue: "contextvars.ContextVar" = contextvars.ContextVar("current_token_queue", default=None)

async def stream_plain_response(llm: ChatNVIDIA, prompt: str, token_queue: Optional[asyncio.Queue]):
    """Streams response tokens live to token_queue."""
    full = ""
    delim_seen = False
    pending_after = ""
    started = False

    async for chunk in llm.astream([HumanMessage(content=prompt)]):
        piece = getattr(chunk, "content", "") or ""
        if not piece:
            continue
        full += piece

        if not delim_seen:
            if STREAM_DELIM in full:
                delim_seen = True
                pending_after = full.split(STREAM_DELIM, 1)[1]
            else:
                continue
        else:
            pending_after += piece

        if not started:
            stripped = pending_after.lstrip("\r\n")
            if not stripped:
                continue
            started = True
            if token_queue is not None:
                await token_queue.put(("token", stripped))
            pending_after = ""
            continue

        if token_queue is not None:
            await token_queue.put(("token", piece))

    if STREAM_DELIM in full:
        reasoning, final_text = full.split(STREAM_DELIM, 1)
    else:
        reasoning, final_text = "", full
    return strip_thinking(reasoning).strip(), strip_thinking(final_text).strip()

async def execute_llm_structured(llm: ChatNVIDIA, prompt_str: str, pydantic_model, state: dict, retries: int = 2):
    """Executes an LLM call and ensures structured Pydantic output."""
    parser = PydanticOutputParser(pydantic_object=pydantic_model)
    format_instructions = parser.get_format_instructions()
    
    system_prompt = (
        f"Current Date & Time: {get_current_datetime_str()}\n\n"
        "You are GoalAI, a goal-oriented general-purpose AI assistant.\n\n"
        "Your job is to understand what the user is trying to accomplish and help them reach that outcome."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\n{format_instructions}"),
        ("user", prompt_str)
    ])
    
    chain = prompt | llm
    
    for attempt in range(retries):
        try:
            res = await chain.ainvoke({"format_instructions": format_instructions, **state})
            content = strip_thinking(res.content).strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            return parser.parse(content.strip())
        except Exception as e:
            print(f"Structured Parsing Retry {attempt + 1}/{retries} failed: {e}")
            await asyncio.sleep(0.5)
            
    return None

def format_context(state: AgentState) -> str:
    """Formats the current graph state into a readable string for the prompt."""
    msg_str = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in state.get("messages", [])])
    
    return f"""Current Date & Time: {get_current_datetime_str()}

Conversation History:
{msg_str}

Current Internal State:
- Goal: {state.get('goal', 'Not set')}
- Constraints: {state.get('constraints', [])}
- Thinking Level: {state.get('thinking_level', 'medium')}
- Prior Response Draft: {state.get('response', 'None')}
- Reflection on Draft: {state.get('reflection', 'None')}

Web Search Results (use these for anything current/time-sensitive; ignore if 'None'):
{state.get('web_search_results') or 'None — no web search was performed for this request.'}
"""

# ==================================================
# LANGGRAPH NODES
# ==================================================

async def understand_goal_node(state: AgentState) -> dict:
    """Understands the user's goal and performs web search if needed."""
    messages = state.get("messages", [])
    latest_user_message = messages[-1].content if messages else ""
    web_search_query = ""
    web_search_results = ""
    web_search_raw_links: list = []
    web_search_images: list = []
    
    if needs_web_search(latest_user_message):
        web_search_query = resolve_search_query(messages, latest_user_message)
        (web_search_results, web_search_raw_links), web_search_images = await asyncio.gather(
            web_search(web_search_query),
            web_image_search(web_search_query),
        )

    llm = get_llm(state["model_type"], state["temperature"])
    prompt = "Analyze this conversation and extract the core goal, intent, constraints, and missing info.\n\n{context}"
    
    res = await execute_llm_structured(
        llm, prompt, GoalExtraction,
        {"context": format_context({**state, "web_search_results": web_search_results})}
    )

    return {
        "goal": res.main_goal if res else "Provide a helpful response.",
        "hidden_intent": res.hidden_intent if res else "",
        "constraints": res.constraints if res else [],
        "web_search_query": web_search_query,
        "web_search_results": web_search_results,
        "web_search_links": web_search_raw_links,
        "web_search_images": web_search_images,
    }

async def executor_node(state: AgentState) -> dict:
    """Executes the thinking and generates a response."""
    llm = get_llm(state["model_type"], state["temperature"])
    token_queue = _current_token_queue.get()
    
    # Get max tokens for thinking level
    thinking_level = state.get("thinking_level", DEFAULT_THINKING_LEVEL).lower()
    thinking_config = THINKING_LEVELS.get(thinking_level, THINKING_LEVELS[DEFAULT_THINKING_LEVEL])
    max_tokens = thinking_config["max_tokens"]

    prompt = (
        "Execute the task to satisfy the user's goal with thorough thinking.\n\n"
        f"Maximum thinking budget: {max_tokens} tokens\n"
        f"Thinking level: {thinking_config['label']} - {thinking_config['description']}\n\n"
        f"{format_context(state)}\n\n"
        "First, in one short sentence, note why this response is correct and helpful. "
        f"Then, on its own line, write exactly: {STREAM_DELIM}\n"
        "Then write ONLY the final response text the user should see — no extra commentary before or after it."
    )

    if token_queue is not None:
        await token_queue.put(("executor_start", None))

    try:
        reasoning, response_text = await stream_plain_response(llm, prompt, token_queue)
    except Exception as e:
        print(f"executor_node streaming failed: {e}")
        reasoning, response_text = "", ""

    if not response_text:
        response_text = "I apologize, I encountered an issue formulating my answer."
        if token_queue is not None:
            await token_queue.put(("token", response_text))

    # Append web sources if search was performed
    if token_queue is not None:
        sources_md = build_web_sources_markdown(
            state.get("web_search_links") or [], state.get("web_search_images") or []
        )
        if sources_md:
            response_text += sources_md
            await token_queue.put(("token", sources_md))

    return {
        "response": response_text,
        "executor_reasoning": reasoning,
        "completion_score": 100,
    }

async def reflector_node(state: AgentState) -> dict:
    """Reviews the response quality."""
    llm = get_llm(state["model_type"], state["temperature"])
    prompt = "Review the response against the goal and constraints. Evaluate quality and potential issues.\n\n{context}"
    
    res = await execute_llm_structured(llm, prompt, ReflectorOutput, {"context": format_context(state)})
    
    reflection_str = "Looks solid."
    if res:
        reflection_str = f"Quality: {res.quality} | Correctness: {res.correctness} | Improvements: {res.improvements}"
        
    return {
        "reflection": reflection_str
    }

def decision_edge(state: AgentState) -> str:
    """Decides whether to continue or finish."""
    return END

# ==================================================
# LANGGRAPH WORKFLOW SETUP
# ==================================================

workflow = StateGraph(AgentState)

workflow.add_node("understand_goal", understand_goal_node)
workflow.add_node("executor", executor_node)
workflow.add_node("reflector", reflector_node)

workflow.set_entry_point("understand_goal")
workflow.add_edge("understand_goal", "executor")
workflow.add_edge("executor", "reflector")
workflow.add_edge("reflector", END)

app_graph = workflow.compile()

# ==================================================
# FASTAPI & API ENDPOINTS
# ==================================================

app = FastAPI(title="Goal-Oriented AI Assistant")

# Serve the frontend static files
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

# Root route serves the frontend
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
    thinking_level: str = "medium"

class ClearSessionRequest(BaseModel):
    session_id: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/clear-session")
async def clear_session(request: ClearSessionRequest):
    if request.session_id in sessions:
        del sessions[request.session_id]
    return {"status": "success", "message": f"Session {request.session_id} cleared."}

NODE_LABELS = {
    "understand_goal": "Understanding the goal",
    "executor": "Drafting a response",
    "reflector": "Checking the draft",
}

FALLBACK_LABELS = ["Fathoming", "Pondering", "Discovering", "Triangulating", "Sifting"]

def node_detail(node_name: str, state: dict) -> str:
    """Turns node output into readable text."""
    if node_name == "understand_goal":
        goal = state.get("goal", "")
        search_query = state.get("web_search_query", "")
        prefix = f"Searched the web for '{search_query}'. " if search_query else ""
        return f"{prefix}Understanding the goal: {goal}" if goal else f"{prefix}Understanding the goal."
    if node_name == "executor":
        reasoning = state.get("executor_reasoning", "")
        level = state.get("thinking_level", "medium")
        return reasoning if reasoning else f"Drafting a response with {level} thinking."
    if node_name == "reflector":
        reflection = state.get("reflection", "")
        return reflection if reflection else "Checking the draft for quality and accuracy."
    return NODE_LABELS.get(node_name, node_name)

async def run_graph_streaming(initial_state: dict, timeout: float, token_queue: asyncio.Queue):
    """Runs the LangGraph workflow and streams events."""
    SPINNER_HEARTBEAT_INTERVAL = 2.5

    state_acc = dict(initial_state)
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    agen = app_graph.astream(initial_state, stream_mode="updates")

    graph_next = asyncio.ensure_future(agen.__anext__())
    queue_next = asyncio.ensure_future(token_queue.get())

    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            wait_chunk = min(SPINNER_HEARTBEAT_INTERVAL, remaining)

            done, _ = await asyncio.wait(
                {graph_next, queue_next}, timeout=wait_chunk, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:
                yield ("spinner_tick", None)
                continue

            if queue_next in done:
                kind, payload = queue_next.result()
                if kind == "executor_start":
                    yield ("reset", None)
                else:
                    yield ("token", payload)
                queue_next = asyncio.ensure_future(token_queue.get())

            if graph_next in done:
                try:
                    chunk = graph_next.result()
                except StopAsyncIteration:
                    break
                node_name, node_output = next(iter(chunk.items()))
                state_acc.update(node_output)
                yield ("status", node_name, state_acc)
                graph_next = asyncio.ensure_future(agen.__anext__())
    finally:
        for t in (graph_next, queue_next):
            t.cancel()
        for t in (graph_next, queue_next):
            try:
                await t
            except (asyncio.CancelledError, StopAsyncIteration, Exception):
                pass
        await agen.aclose()

async def generate_stream(request: "ChatRequest", session: dict, session_id: str):
    """Streams SSE events to the frontend."""
    final_response = ""
    last_spinner: Optional[str] = None

    initial_state = {
        "messages": session["messages"],
        "model_type": request.model_type,
        "temperature": request.temperature,
        "thinking_level": request.thinking_level,
        "completion_score": 0,
    }
    
    timeout = 240.0
    final_state = initial_state
    token_queue: asyncio.Queue = asyncio.Queue()
    qtoken = _current_token_queue.set(token_queue)

    try:
        async for evt in run_graph_streaming(initial_state, timeout, token_queue):
            kind = evt[0]
            if kind == "status":
                _, node_name, state_so_far = evt
                last_spinner = random.choice(FALLBACK_LABELS)
                detail = node_detail(node_name, state_so_far)
                yield f"data: {json.dumps({'type': 'status', 'step': node_name, 'label': last_spinner, 'detail': detail})}\n\n"
                final_state = state_so_far
            elif kind == "spinner_tick":
                last_spinner = random.choice(FALLBACK_LABELS)
                yield f"data: {json.dumps({'type': 'status', 'step': 'thinking', 'label': last_spinner, 'detail': 'Still working on it...'})}\n\n"
            elif kind == "reset":
                final_response = ""
                yield f"data: {json.dumps({'type': 'message_reset'})}\n\n"
            elif kind == "token":
                final_response += evt[1]
                yield f"data: {json.dumps({'type': 'message', 'assistant_message': evt[1], 'conversation_id': session_id, 'session_id': session_id, 'goal_complete': False})}\n\n"

        if not final_response.strip():
            fallback_text = final_state.get("response", "Task completed but no response was formulated.")
            if fallback_text:
                final_response = fallback_text
                yield f"data: {json.dumps({'type': 'message', 'assistant_message': fallback_text, 'conversation_id': session_id, 'session_id': session_id, 'goal_complete': True})}\n\n"

    except asyncio.TimeoutError:
        print(f"[{session_id}] Streaming workflow timed out. Falling back to direct answer.")
        yield f"data: {json.dumps({'type': 'status', 'step': 'fallback', 'label': 'Recovering', 'detail': 'The reasoning was taking too long.'})}\n\n"
        final_state = {"completion_score": 100}
    except Exception as e:
        print(f"[{session_id}] Streaming workflow failed: {e}.")
        final_state = {"completion_score": 100}
    finally:
        _current_token_queue.reset(qtoken)

    session["messages"].append(AIMessage(content=final_response))

async def answer_directly(message: str, history: List[BaseMessage], model_type: str, temperature: float) -> str:
    """Always returns a real answer — quick path for simple questions."""
    system_content = (
        f"Current Date & Time: {get_current_datetime_str()}\n\n"
        "You are GoalAI, a sharp, honest, genuinely helpful assistant."
    )

    messages = [SystemMessage(content=system_content)]
    messages.extend(history[-6:])
    messages.append(HumanMessage(content=message))

    try:
        llm = get_llm(model_type, temperature)
        res = await llm.ainvoke(messages)
        answer = strip_thinking(res.content).strip()
        if answer:
            return answer
    except Exception as e:
        print(f"Primary model failed: {e}")

    return "I'm having trouble reaching the model right now. Please try again in a moment."

async def answer_directly_stream(message: str, history: List[BaseMessage], model_type: str, temperature: float):
    """Streaming version of answer_directly."""
    system_content = (
        f"Current Date & Time: {get_current_datetime_str()}\n\n"
        "You are GoalAI, a sharp, honest assistant. Be direct and concise."
    )

    messages = [SystemMessage(content=system_content)]
    messages.extend(history[-6:])
    messages.append(HumanMessage(content=message))

    try:
        llm = get_llm(model_type, temperature)
        async for chunk in llm.astream(messages):
            piece = getattr(chunk, "content", "") or ""
            if piece:
                yield piece
    except Exception as e:
        print(f"Streaming failed: {e}")
        yield "I'm having trouble reaching the model right now. Please try again in a moment."

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        return await _handle_chat(request)
    except Exception as e:
        print(f"Chat handler failed entirely: {e}")
        return {
            "response": "Something went wrong on my end — please try again.",
            "session_id": request.session_id,
            "goal_progress": 0,
            "completed": False,
        }

async def _handle_chat(request: ChatRequest):
    session_id = request.session_id
    
    if session_id not in sessions:
        sessions[session_id] = {"messages": []}
        
    session = sessions[session_id]
    llm = get_llm(request.model_type, request.temperature)

    session["messages"] = await summarize_memory(session["messages"], llm)
    session["messages"].append(HumanMessage(content=request.message))

    if request.stream:
        return StreamingResponse(
            generate_stream(request, session, session_id),
            media_type="text/event-stream"
        )

    # Non-streaming path
    initial_state = {
        "messages": session["messages"],
        "model_type": request.model_type,
        "temperature": request.temperature,
        "thinking_level": request.thinking_level,
        "completion_score": 0,
    }

    try:
        timeout = 240.0
        final_state = await asyncio.wait_for(app_graph.ainvoke(initial_state), timeout=timeout)
        final_response = final_state.get("response", "Task completed but no response was formulated.")
    except asyncio.TimeoutError:
        print(f"[{session_id}] Workflow timed out. Falling back to direct answer.")
        final_response = await answer_directly(
            request.message, session["messages"], request.model_type, request.temperature
        )
        final_state = {"completion_score": 100}
    except Exception as e:
        print(f"[{session_id}] Workflow failed: {e}. Falling back to direct answer.")
        final_response = await answer_directly(
            request.message, session["messages"], request.model_type, request.temperature
        )
        final_state = {"completion_score": 100}

    session["messages"].append(AIMessage(content=final_response))

    return {
        "response": final_response,
        "session_id": session_id,
        "goal_progress": final_state.get("completion_score", 100),
        "completed": final_state.get("completion_score", 100) >= 100,
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
