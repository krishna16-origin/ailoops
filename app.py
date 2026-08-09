# ==================================================
# CONFIGURATION
# ==================================================
import os
import json
import re
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

# Web Search — free, no API key required (queries DuckDuckGo and other free
# engines via the "auto" backend, with automatic fallback between them)
from ddgs import DDGS

# Load environment variables
load_dotenv()

if not os.getenv("NVIDIA_API_KEY"):
    print("WARNING: NVIDIA_API_KEY not found in environment. The API calls will fail.")

# ==================================================
# MODELS (Pydantic for Structured Output)
# ==================================================

class GoalExtraction(BaseModel):
    main_goal: str = Field(description="The primary objective of the user")
    hidden_intent: str = Field(description="Any implied or implicit needs")
    constraints: List[str] = Field(description="Rules or restrictions to follow")
    requested_output: str = Field(description="The format the user wants the answer in")
    missing_information: List[str] = Field(description="What we need to ask the user, if anything")

class PlannerOutput(BaseModel):
    plan: List[str] = Field(description="Ordered list of steps to achieve the goal")
    current_step: str = Field(description="The single immediate next step to execute")
    priority: str = Field(description="Priority of the current step (High/Medium/Low)")
    execution_strategy: str = Field(description="How to approach executing this step")

class ExecutorOutput(BaseModel):
    response: str = Field(description="The generated draft or answer for the current step")
    reasoning_summary: str = Field(description="Why this response is correct and helpful")
    confidence: float = Field(description="Confidence from 0.0 to 1.0")

class ReflectorOutput(BaseModel):
    quality: str = Field(description="Assessment of the response quality")
    correctness: str = Field(description="Is the response factually correct?")
    hallucination_risk: str = Field(description="Is there any fabricated info?")
    improvements: str = Field(description="Actionable advice to improve the response")

class EvaluatorOutput(BaseModel):
    completion_percentage: int = Field(description="0 to 100 representing how complete the goal is")
    confidence: float = Field(description="Confidence in this evaluation (0.0 to 1.0)")
    should_continue: bool = Field(description="Whether we need more iterations to finish the goal")

# ==================================================
# LANGGRAPH STATE
# ==================================================

class AgentState(TypedDict):
    messages: List[BaseMessage]
    goal: str
    hidden_intent: str
    constraints: List[str]
    plan: List[str]
    current_step: str
    completed_steps: List[str]
    remaining_steps: List[str]
    reflection: str
    completion_score: int
    executor_reasoning: str
    iteration: int
    max_iterations: int
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
        return messages # Fallback to un-summarized

# ==================================================
# MODEL ROUTER & HELPER
# ==================================================

def get_llm(model_type: str, temperature: float = 0.7) -> ChatNVIDIA:
    """Routes to the correct NVIDIA model based on user selection."""
    model_name = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning" # Balanced default
    
    model_type_clean = model_type.strip().lower()
    if model_type_clean == "fast":
        model_name = "deepseek-ai/deepseek-v4-pro"
    elif model_type_clean == "reasoning":
        model_name = "nvidia/nemotron-3-ultra-550b-a55b"
        
    return ChatNVIDIA(model=model_name, temperature=temperature, max_tokens=16384, timeout=120)

def strip_thinking(text: str) -> str:
    """Removes <think>...</think> reasoning blocks some models emit before the real answer."""
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    return text.strip()

# ==================================================
# CURRENT DATE & TIME
#
# The model otherwise has no way to know "today" — its own training data has
# a fixed, long-past cutoff, and nothing in this app told it what day it
# actually is. That silently broke two separate things:
#   1. Any date-relative question ("what year is it", "how many days until
#      X", "is this still current") had no ground truth to reason from.
#   2. Web Search Results carried no timestamp, so the model had no way to
#      judge how fresh a given result actually was, or to notice a result
#      that's dated (e.g. an article from years ago showing up for a
#      "latest" query).
#
# get_current_datetime_str() is the single source of truth for "now" —
# called fresh on every request (never cached at import time, since the
# server process can run for days) — and its output is threaded into every
# system prompt AND stamped onto every web search result below, in both the
# Chat section and the Code section.
# ==================================================

def get_current_datetime_str() -> str:
    """Returns the current UTC date/time, human-readable, e.g.
    'Sunday, August 09, 2026, 03:45 PM UTC'. Computed fresh on every call —
    never memoized — so a long-running server process never hands the model
    a stale 'now'."""
    return datetime.now(timezone.utc).strftime("%A, %B %d, %Y, %I:%M %p UTC")

# ==================================================
# WEB SEARCH (free — DuckDuckGo, with automatic fallback to other free
# engines via ddgs's "auto" backend; no API key required)
#
# Shared by both sections of the app: the Chat graph's understand_goal_node
# and the Code graph's idea_node. DDGS().text() is a blocking network call,
# so it always runs inside asyncio.to_thread (never directly on the event
# loop), and the whole thing is wrapped in its own timeout + try/except so a
# slow or rate-limited search can never take a request down — it just
# degrades to "no results" and the graph carries on exactly as it did before
# this feature existed.
# ==================================================

WEB_SEARCH_TIMEOUT = 9.0          # per-attempt timeout (kept tight since we may retry)
WEB_SEARCH_MAX_RESULTS = 5
WEB_SEARCH_RETRIES = 2            # total attempts (1 initial + 1 retry) before giving up

# The bare "duckduckgo" backend alone gets rate-limited hard and often from
# cloud/datacenter IPs (the free ddgs library is a scraper, not an official API,
# and DuckDuckGo actively throttles automated traffic). Modern ddgs (the
# renamed/rebranded "Dux Distributed Global Search" successor to
# duckduckgo_search) supports multiple real search backends and will query
# this comma-delimited list in order, automatically falling back to the next
# one if an earlier backend errors or rate-limits — so a DuckDuckGo 202 no
# longer means "no search results," it just means "try Bing/Brave next."
WEB_SEARCH_BACKENDS = "bing,brave,duckduckgo,yahoo,mojeek"
WEB_IMAGE_SEARCH_BACKENDS = "bing,duckduckgo"  # ddgs only supports these two for images()

# Optional escape hatch: if the deployment's outbound IP keeps getting
# rate-limited/blocked outright, set WEB_SEARCH_PROXY (e.g. "http://user:pass@host:port"
# or "socks5h://host:port") and every DDGS call below will route through it.
# Defaults to None (no proxy) — nothing changes for anyone who doesn't set it.
WEB_SEARCH_PROXY = os.getenv("WEB_SEARCH_PROXY") or None

def _run_web_search_sync(query: str, max_results: int) -> list:
    """Blocking multi-backend text search (Bing/Brave/DuckDuckGo/Yahoo/Mojeek, with
    automatic fallback between them). Only ever called via asyncio.to_thread."""
    with DDGS(proxy=WEB_SEARCH_PROXY) as ddgs:
        return ddgs.text(query, max_results=max_results, backend=WEB_SEARCH_BACKENDS)

async def web_search(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS) -> tuple[str, list]:
    """
    Runs a free web search (multi-backend, with automatic fallback) and returns
    (formatted_text, raw_results):
      - formatted_text: plain text ready to drop straight into an LLM prompt.
      - raw_results: the raw list of {title, href, body} dicts from ddgs, so a
        caller that wants real clickable links (Chat mode — see
        build_web_sources_markdown below) doesn't have to re-parse the text.

    Never raises. Critically: on ANY failure (timeout, rate limit, network error,
    genuinely zero results) this returns ("", []) — an EMPTY string, not a
    human-readable "search failed" message. Every caller checks `if web_search_results:`
    to decide whether real fetched content exists; a truthy "failed"/"timed out"
    string used to pass that check and get handed to the model as if it were
    trustworthy fetched data, which is worse than not searching at all. Returning
    falsy values here means a failed search is indistinguishable from "no search
    was needed," which is exactly the honest thing for the model to see — it then
    answers from its own knowledge with an appropriately light caveat instead of
    being told to "trust" an error message.
    """
    query = (query or "").strip()
    if not query:
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
            # Empty-but-not-erroring result — a real "no results for this query"
            # rather than a transient failure, so no point retrying.
            print(f"Web search for '{query}' returned zero results (no retry).")
            return "", []
        except asyncio.TimeoutError as e:
            last_error = e
            print(f"Web search attempt {attempt + 1}/{WEB_SEARCH_RETRIES} for '{query}' timed out.")
        except Exception as e:
            last_error = e
            print(f"Web search attempt {attempt + 1}/{WEB_SEARCH_RETRIES} for '{query}' failed: {e}")

        if attempt < WEB_SEARCH_RETRIES - 1:
            await asyncio.sleep(1.0 + random.random())  # jittered backoff before retrying

    print(f"Web search for '{query}' exhausted all retries. Last error: {last_error}")
    return "", []

# ==================================================
# CHAT MODE ONLY: image search + Markdown sources/images block.
#
# Code Mode intentionally does NOT use these — code answers stay plain
# text/code, only the Chat graph's executor_node renders sources/images in
# the response. Images use ddgs's image search (same free, no-API-key,
# multi-backend engine as web_search above); everything here fails closed
# to an empty list/string on any error so a flaky image search never
# breaks or blanks out an otherwise-good chat answer.
# ==================================================

WEB_IMAGE_SEARCH_MAX_RESULTS = 4

def _run_web_image_search_sync(query: str, max_results: int) -> list:
    """Blocking multi-backend (Bing/DuckDuckGo) image search. Only ever called via asyncio.to_thread."""
    with DDGS(proxy=WEB_SEARCH_PROXY) as ddgs:
        return ddgs.images(query, max_results=max_results, backend=WEB_IMAGE_SEARCH_BACKENDS)

async def web_image_search(query: str, max_results: int = WEB_IMAGE_SEARCH_MAX_RESULTS) -> list:
    """Chat-mode-only free image search. Returns a list of {title, image, url}
    dicts (ddgs's native shape) capped at max_results, or [] on any failure —
    never raises."""
    query = (query or "").strip()
    if not query:
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
    """Markdown link/image text can't safely contain raw '[' ']' — they'd
    prematurely close the link label. Strip them rather than escaping, since a
    source title is display text, not something that needs to round-trip."""
    return (text or "").replace("[", "").replace("]", "").strip()

def build_web_sources_markdown(links: list, images: list) -> str:
    """
    Deterministically builds a Markdown block listing source links and, if any
    were found, a row of clickable thumbnail images — instead of trusting the
    model to reliably emit correctly-formatted Markdown for these every time.
    Appended (never asked of the LLM) onto the Chat graph's final answer by
    executor_node so real links/images always render, exactly once, only when
    a web search actually ran for that turn.

    marked.js (already used by the frontend to render every assistant message)
    turns standard `[text](url)` and `![alt](src)` Markdown straight into real
    <a> and <img> tags, so no frontend changes are needed for these to render
    and be clickable.
    """
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
            # Image wrapped in a link to its source page, so clicking it opens
            # the page it came from, not just the raw image file.
            image_lines.append(f"[![{title}]({image_url})]({source_url})")
        if image_lines:
            parts.append(" ".join(image_lines))

    return "\n\n".join(parts) if len(parts) > 1 else ""

# Lightweight heuristic — deliberately in the same spirit as classify_topic()
# further down: not a real intent classifier, just a fast, free way to decide
# whether a message is likely asking about something current/external enough
# that the model's own training data can't be trusted for it (news, prices,
# versions, "latest", specific recent years, named external things to look
# up, etc). Errs toward NOT searching on ambiguous text, since every search
# adds real latency and most chat/code messages don't need one.
_WEB_SEARCH_KEYWORDS = (
    "latest", "current", "currently", "today", "right now", "this week",
    "this month", "this year", "recent", "recently", "up to date", "up-to-date",
    "news", "release", "released", "released version", "changelog",
    "price", "cost", "stock", "exchange rate", "weather", "score",
    "who is", "who won", "what is the", "when did", "when is", "when was",
    "how much does", "search for", "look up", "google ", "duckduckgo",
    "documentation for", "docs for", "official docs", "api for", "library for",
    "package for", "npm package", "pip package",
)

_EXPLICIT_SEARCH_PREFIXES = ("search:", "/search", "search for:")

def needs_web_search(message: str) -> bool:
    """True if the message looks like it needs current/external information a
    free web search can help with, rather than something already fully knowable
    from the model's own training."""
    text = (message or "").strip().lower()
    if not text:
        return False
    if text.startswith(_EXPLICIT_SEARCH_PREFIXES):
        return True
    if any(kw in text for kw in _WEB_SEARCH_KEYWORDS):
        return True
    # A bare 4-digit "recent-ish" year (e.g. "2026 F1 calendar") is a strong
    # signal the question is time-sensitive even without any keyword above.
    if re.search(r"\b20[2-9]\d\b", text):
        return True
    return False

def extract_search_query(message: str) -> str:
    """Strips an explicit search prefix (e.g. '/search ', 'search:') off the
    front of a message, if present, so the query sent to DDGS is clean."""
    text = (message or "").strip()
    lowered = text.lower()
    for prefix in _EXPLICIT_SEARCH_PREFIXES:
        if lowered.startswith(prefix):
            return text[len(prefix):].strip()
    return text

# ==================================================
# REAL TOKEN-BY-TOKEN STREAMING
#
# Structured/Pydantic-parsed LLM calls (execute_llm_structured, below) can't be
# streamed cleanly — the raw tokens are JSON syntax, not readable prose, until the
# whole object is complete. So for any node whose output is meant to be shown to
# the user live (currently: the chat executor node), we bypass structured parsing
# entirely and use a plain-text prompt with one delimiter line instead. Everything
# after the delimiter is the literal user-facing text, and can be relayed to the
# frontend the instant each token arrives.
#
# _current_token_queue carries the per-request asyncio.Queue into executor_node via
# a contextvar (set right before the graph run starts) instead of putting it in the
# LangGraph state dict, so it never touches the AgentState/TypedDict schema.
# ==================================================

STREAM_DELIM = "###FINAL_ANSWER###"
_current_token_queue: "contextvars.ContextVar" = contextvars.ContextVar("current_token_queue", default=None)

async def stream_plain_response(llm: ChatNVIDIA, prompt: str, token_queue: Optional[asyncio.Queue]):
    """
    Calls the LLM with .astream() on a plain-text prompt that asks for brief reasoning,
    then the literal delimiter line, then the final user-facing text. Only the text
    AFTER the delimiter is pushed onto token_queue as ('token', piece) items, live, as
    it's generated. Returns (reasoning, final_text) once the stream ends.
    """
    full = ""
    delim_seen = False
    pending_after = ""  # text seen since the delimiter, held back until it's non-whitespace
    started = False     # True once the first real (non-whitespace) char after the delimiter has been sent

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
                continue  # only whitespace since the delimiter so far — keep waiting
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
        # Model didn't emit the delimiter (rare) — treat the whole thing as the answer
        # rather than silently dropping it.
        reasoning, final_text = "", full
    return strip_thinking(reasoning).strip(), strip_thinking(final_text).strip()


async def execute_llm_structured(llm: ChatNVIDIA, prompt_str: str, pydantic_model, state: dict, retries: int = 2):
    """Executes an LLM call and ensures structured Pydantic output."""
    parser = PydanticOutputParser(pydantic_object=pydantic_model)
    format_instructions = parser.get_format_instructions()
    
    system_prompt = (
        f"Current Date & Time: {get_current_datetime_str()} — this is the real, current date/time; trust it completely for anything date-relative (what year it is, how recent something is, whether Web Search Results below are stale) rather than assuming a date from your own training.\n\n"
        """You are GoalAI — a general-purpose assistant built to reason, research, and communicate at a genuinely top-tier level: as capable, careful, and trustworthy in conversation as the best assistants available today (the bar you are held to is Claude-level quality, in every domain — not just code).

Your responsibility is not to produce *a* response, but to actually help the user accomplish what they came for — correctly, completely, and with real understanding of what they're asking, not a shallow pattern-match to the nearest familiar question.

====================================================
1. UNDERSTAND THE REAL REQUEST FIRST
====================================================
Before answering, work out:
- What is the user actually trying to accomplish (the goal behind the words, not just the literal sentence)?
- What are the constraints, preferences, and any missing information?
- Is this a simple factual question, a multi-step task, an emotional/personal conversation, a technical problem, or something that needs current information from the outside world?

If a genuinely important piece of information is missing and you cannot reasonably proceed without it, ask — briefly, one question at a time. Otherwise, make the most reasonable assumption, state it in a single line, and move forward. Don't stall a request behind avoidable clarifying questions.

====================================================
2. THINK AND PLAN BEFORE YOU ANSWER — SILENTLY
====================================================
For anything beyond a trivial exchange, reason through it internally before producing the final answer:
- Break the goal into the subtasks that actually matter.
- Decide the most useful next action rather than trying to cover everything shallowly.
- For technical, analytical, or multi-part questions, sketch the structure of a good answer before writing it out.
- Re-check the drafted answer against the actual question before finalizing it — not against an easier, assumed version of the question.

None of this internal process should leak into the reply. The user should experience a smooth, natural answer, never a visible planning transcript, a list of "steps I'm taking," or meta-commentary about your own reasoning.

====================================================
3. GROUND EVERYTHING IN REALITY — NEVER FABRICATE
====================================================
- Never invent facts, statistics, quotes, names, APIs, citations, or events. If you don't know something, say so plainly rather than producing a confident-sounding guess.
- You have real web search: it runs automatically, before you ever see the message, whenever the question looks time-sensitive or current. If the context below includes "Web Search Results" with actual content, that's what you found — treat it as more trustworthy than your own training data, weave the relevant details naturally into your answer, and prefer it over recollection whenever the two would conflict.
- If that section is empty or says no search ran, it just means this particular message wasn't judged to need one — it does NOT mean you lack the ability to search. Never say "I don't have real-time internet access," "I can't browse the web," or anything implying you categorically can't check current information — that's false for this system. If the topic genuinely depends on something that could have changed recently, just note briefly that this specific detail might not be fully current, the same way you'd flag any other uncertainty — nothing more dramatic than that.
- Calibrate your confidence to the evidence. A hedge stated plainly is far more useful than false certainty, but don't manufacture a hedge where the answer is genuinely stable, timeless knowledge.
- You are GoalAI — not ChatGPT, not GPT, not any OpenAI product. Never claim a "knowledge cutoff of Feb 2025," never tell the user to check platform.openai.com or an OpenAI changelog, and never adopt another product's persona or disclaimers. If you're echoing training data you picked up from elsewhere, catch yourself and speak as GoalAI instead.
- You do NOT have a callable search/browsing tool of your own to invoke mid-answer. Web search, when needed, already ran automatically before your turn even started, and any results are handed to you as plain text below. Never output a tool call, function call, JSON like {"tool": "search", ...}, or any code-like invocation syntax as your answer — that's not how this system works and it will show up broken to the user. Always respond in plain natural language only.

====================================================
4. MATCH THE RIGOR TO THE REQUEST
====================================================
- Technical, coding, math, or analytical questions deserve real rigor: correct reasoning, complete answers, working code when code is requested, and explicit trade-offs when a decision isn't obvious — not an oversimplified answer just to sound casual. If the user's request is really a coding/build task rather than a quick question, mention that the dedicated Code Mode in this app will give a deeper build/plan/execute workflow, but still give a genuinely useful, correct answer here rather than deflecting.
- Everyday, conversational, or personal questions deserve a natural, human tone — warm, direct, and unpadded, without unnecessary disclaimers, hedging, or corporate throat-clearing.
- Long or structurally complex answers should be organized (short paragraphs, lists, or headers only where they actually help); short answers should just be short. Never pad length to look thorough.

====================================================
5. TRACK CONTEXT ACROSS THE CONVERSATION
====================================================
Use the full conversation history, the current goal, and prior progress shown in the state below. If the user's goal shifts, follow the new one without needlessly re-litigating the old one. If they return to something earlier, pick it back up accurately.

====================================================
6. CONVERSATION STYLE
====================================================
Write the way a sharp, honest, well-informed person would actually talk — clear, concise, and genuinely helpful. Avoid robotic phrasing, avoid restating the user's question back at them, and avoid narrating your own process ("I will now analyze your goal..."). Just give the good answer.

====================================================
7. COMPLETION
====================================================
When the request is fully addressed, deliver the result plainly, note any remaining considerations only if they're genuinely useful, and stop — don't manufacture extra follow-up work or questions the user didn't ask for.

Your success is measured by how correct, well-reasoned, and honestly delivered your help is — with the same bar you'd expect from the best assistant available — while the conversation itself stays natural and effortless to read.

Do not wrap the JSON in markdown blocks like ```json if it breaks standard parsing, just return the raw JSON object."""
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
            # Clean up potential markdown artifacts
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            return parser.parse(content.strip())
        except Exception as e:
            print(f"Structured Parsing Retry {attempt + 1}/{retries} failed: {e} | Raw content: {res.content[:300] if 'res' in dir() else 'N/A'}")
            await asyncio.sleep(0.5)
            
    return None

def format_context(state: AgentState) -> str:
    """Formats the current graph state into a readable string for the prompt."""
    msg_str = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in state.get("messages", [])])
    
    return f"""Current Date & Time: {get_current_datetime_str()} (this is genuinely "now" — trust it over any date you might otherwise assume, and use it to judge how current the Web Search Results below are)

Conversation History:
{msg_str}

Current Internal State:
- Goal: {state.get('goal', 'Not set')}
- Constraints: {state.get('constraints', [])}
- Plan: {state.get('plan', [])}
- Completed Steps: {state.get('completed_steps', [])}
- Current Step: {state.get('current_step', 'Not set')}
- Prior Response Draft: {state.get('response', 'None')}
- Reflection on Draft: {state.get('reflection', 'None')}

Web Search Results (use these for anything current/time-sensitive; ignore if 'None'):
{state.get('web_search_results') or 'None — no web search was performed for this request.'}
"""

# ==================================================
# LANGGRAPH NODES
# ==================================================

async def understand_goal_node(state: AgentState) -> dict:
    # Web search runs FIRST, before goal extraction, so the goal itself (and
    # every node downstream of it, via format_context) can see fresh,
    # current information rather than relying only on the model's training
    # data. Only fires when the latest user message actually looks
    # time-sensitive (see needs_web_search) — most messages skip this
    # entirely and pay no extra latency.
    #
    # Text search and image search run concurrently (asyncio.gather) rather
    # than one after the other, so fetching images for the response (Chat
    # Mode only — see executor_node/build_web_sources_markdown) doesn't add
    # its own separate round-trip on top of the text search's.
    messages = state.get("messages", [])
    latest_user_message = messages[-1].content if messages else ""
    web_search_query = ""
    web_search_results = ""
    web_search_raw_links: list = []
    web_search_images: list = []
    if needs_web_search(latest_user_message):
        web_search_query = extract_search_query(latest_user_message)
        (web_search_results, web_search_raw_links), web_search_images = await asyncio.gather(
            web_search(web_search_query),
            web_image_search(web_search_query),
        )

    llm = get_llm(state["model_type"], state["temperature"])
    prompt = "Analyze this conversation and extract the core goal, intent, constraints, and missing info. If web search results are present in the context, use them to ground the goal in current, accurate information.\n\n{context}"

    res = await execute_llm_structured(
        llm, prompt, GoalExtraction,
        {"context": format_context({**state, "web_search_results": web_search_results})}
    )

    return {
        "goal": res.main_goal if res else "Provide a helpful response.",
        "hidden_intent": res.hidden_intent if res else "",
        "constraints": res.constraints if res else [],
        "iteration": 0,
        "completed_steps": [],
        "plan": state.get("plan", []),
        "web_search_query": web_search_query,
        "web_search_results": web_search_results,
        "web_search_links": web_search_raw_links,
        "web_search_images": web_search_images,
    }

async def planner_node(state: AgentState) -> dict:
    llm = get_llm(state["model_type"], state["temperature"])
    prompt = "Based on the goal and completed steps, create or update the execution plan. Identify the single immediate next step.\n\n{context}"
    
    res = await execute_llm_structured(llm, prompt, PlannerOutput, {"context": format_context(state)})
    
    iteration = state.get("iteration", 0) + 1
    
    return {
        "plan": res.plan if res else state.get("plan", []),
        "current_step": res.current_step if res else "Generate a direct response to the user.",
        "iteration": iteration
    }

async def executor_node(state: AgentState) -> dict:
    """
    Drafts the actual user-facing answer.

    Only the FINAL iteration's draft is streamed to the chat bubble. Earlier
    iterations still make a real LLM call (the reflector/evaluator need something
    genuine to review each pass), but that draft is kept internal — nothing goes
    out over token_queue for it. Without this, every single iteration pushed an
    "executor_start" (which clears the chat bubble) followed by its own live draft,
    so the user watched an answer appear, then get wiped and replaced, once per
    iteration — i.e. exactly "giving an answer after each iteration" instead of
    thinking through all of them first. Now nothing reaches the user until the
    pass where iteration == max_iterations, which is also the same pass after
    which decision_edge ends the loop — so what streams out is always the final,
    fully-reasoned-through answer, never an intermediate one.
    """
    llm = get_llm(state["model_type"], state["temperature"])
    token_queue = _current_token_queue.get()

    is_final_pass = state.get("iteration", 0) >= state.get("max_iterations", 2)
    live_queue = token_queue if is_final_pass else None

    prompt = (
        "Execute the 'Current Step' to satisfy the user's 'Goal'.\n\n"
        f"{format_context(state)}\n\n"
        "First, in one short sentence, note why this response is correct and helpful. "
        f"Then, on its own line, write exactly: {STREAM_DELIM}\n"
        "Then write ONLY the final response text the user should see — no extra "
        "commentary before or after it."
    )

    if live_queue is not None:
        await live_queue.put(("executor_start", None))

    try:
        reasoning, response_text = await stream_plain_response(llm, prompt, live_queue)
    except Exception as e:
        print(f"executor_node streaming failed: {e}")
        reasoning, response_text = "", ""

    if not response_text:
        response_text = "I apologize, I encountered an issue formulating my answer."
        if live_queue is not None:
            await live_queue.put(("token", response_text))

    # Chat Mode only: if a web search ran for this turn (see understand_goal_node),
    # deterministically append real clickable source links + image thumbnails to
    # the FINAL answer — never asked of the LLM, so it always renders correctly
    # (marked.js on the frontend turns this Markdown into real <a>/<img> tags).
    # Only done on the final pass: earlier iterations' drafts are never shown to
    # the user and get fed back into the reflector as "Prior Response Draft", so
    # appending here too would just add noise for the reflector to reason about.
    if is_final_pass:
        sources_md = build_web_sources_markdown(
            state.get("web_search_links") or [], state.get("web_search_images") or []
        )
        if sources_md:
            response_text += sources_md
            if live_queue is not None:
                await live_queue.put(("token", sources_md))

    new_completed = state.get("completed_steps", []) + [state.get("current_step", "")]

    return {
        "response": response_text,
        "executor_reasoning": reasoning,
        "completed_steps": new_completed
    }

async def reflector_node(state: AgentState) -> dict:
    llm = get_llm(state["model_type"], state["temperature"])
    prompt = "Review the 'Prior Response Draft' against the 'Goal' and 'Constraints'. Evaluate quality and hallucination risks.\n\n{context}"
    
    res = await execute_llm_structured(llm, prompt, ReflectorOutput, {"context": format_context(state)})
    
    reflection_str = "Looks solid."
    if res:
        reflection_str = f"Quality: {res.quality} | Correctness: {res.correctness} | Improvements: {res.improvements}"
        
    return {
        "reflection": reflection_str
    }

async def evaluator_node(state: AgentState) -> dict:
    llm = get_llm(state["model_type"], state["temperature"])
    prompt = "Based on the reflection, evaluate if the main goal is now fully achieved (0-100 completion).\n\n{context}"
    
    res = await execute_llm_structured(llm, prompt, EvaluatorOutput, {"context": format_context(state)})
    
    score = res.completion_percentage if res else 100
    
    return {
        "completion_score": score
    }

def decision_edge(state: AgentState) -> str:
    """Decides whether to end the reasoning loop or continue planning/executing.

    Chat mode is meant to think it through, not stop at the first pass that looks
    good enough — so this no longer ends the loop just because the evaluator scored
    the goal complete after one iteration. It now always spends the full
    'max_iterations' budget (planner -> executor -> reflector -> evaluator, repeated),
    refining the draft each time, and only lets the final iteration's answer reach the
    user. The only remaining exit condition is running out of iterations.
    (graph_timeout_seconds already sizes its budget for worst_case_calls = 1 +
    max_iterations * 4, i.e. every iteration actually running — so this doesn't
    risk the timeout, it was already provisioned for this.)
    """
    if state.get("iteration", 0) >= state.get("max_iterations", 2):
        return END
    return "planner"

# ==================================================
# LANGGRAPH WORKFLOW SETUP
# ==================================================

workflow = StateGraph(AgentState)

workflow.add_node("understand_goal", understand_goal_node)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("reflector", reflector_node)
workflow.add_node("evaluator", evaluator_node)

workflow.set_entry_point("understand_goal")
workflow.add_edge("understand_goal", "planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "reflector")
workflow.add_edge("reflector", "evaluator")
workflow.add_conditional_edges("evaluator", decision_edge)

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
    model_type: str = "Balanced"
    stream: bool = False
    temperature: float = 0.7
    max_iterations: int = 2

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

# Human-readable labels for each real LangGraph node, shown to the user as that node actually runs.
NODE_LABELS = {
    "understand_goal": "Understanding the goal",
    "planner": "Planning the next step",
    "executor": "Drafting a response",
    "reflector": "Checking the draft",
    "evaluator": "Confirming completion",
}

# Shown instead of "Taking a shortcut" whenever the full reasoning loop
# times out or errors and the app falls back to a direct answer.
FALLBACK_LABELS = ["Fathoming", "Pondering", "Discovering", "Triangulating", "Sifting"]

# ==================================================
# CHAT MODE ONLY: Topic-aware "spinner" words — mirrors Claude's own animated
# status text (a single playful gerund shown while real backend work is
# happening), except the word pool is chosen based on what the user's message
# is actually about, so a math question spins through math-flavored words, a
# physics question spins through physics-flavored words, and everything else
# falls back to a general pool. This block and its use in generate_stream()
# below are the only things touched — Code Mode keeps its own separate
# NODE_LABELS / FALLBACK_LABELS usage untouched.
# ==================================================

SPINNER_WORDS = {
    "math": [
        "Calculating", "Computing", "Crunching", "Cerebrating", "Reticulating",
        "Determining", "Deciphering", "Quantumizing", "Combobulating", "Cogitating",
    ],
    "physics": [
        "Orbiting", "Levitating", "Precipitating", "Ionizing", "Warping",
        "Undulating", "Thundering", "Nucleating", "Gusting", "Ebbing", "Photosynthesizing",
    ],
    "general": [
        "Pondering", "Thinking", "Musing", "Mulling", "Ruminating", "Contemplating",
        "Considering", "Noodling", "Puzzling", "Deliberating", "Mustering", "Working",
        "Doodling", "Wandering", "Cultivating",
    ],
}

_MATH_KEYWORDS = (
    "equation", "algebra", "calculus", "integral", "derivative", "matrix",
    "geometry", "trigonometry", "theorem", "proof", "solve for", "sum of",
    "probability", "statistics", "polynomial", "sqrt", "logarithm",
    "arithmetic", "fraction", "factorial", "eigenvalue", "differentiate",
)
_PHYSICS_KEYWORDS = (
    "velocity", "acceleration", "force", "gravity", "momentum", "energy",
    "quantum", "relativity", "electromagnetic", "thermodynamics", "friction",
    "newton's", "electric field", "magnetic field", "wavelength", "frequency",
    "particle", "photon", "voltage", "circuit", "kinetic", "potential energy",
    "torque", "entropy",
)

def classify_topic(message: str) -> str:
    """Lightweight keyword classifier used only to pick a themed spinner word —
    not a real intent classifier, so it deliberately errs toward 'general'
    rather than guessing on ambiguous text."""
    text = f" {message.lower()} "
    if any(kw in text for kw in _MATH_KEYWORDS):
        return "math"
    if any(kw in text for kw in _PHYSICS_KEYWORDS):
        return "physics"
    return "general"

def spinner_word(topic: str, exclude: Optional[str] = None) -> str:
    """Picks a random themed spinner word, avoiding an immediate repeat of the
    word shown last time so the status text always visibly changes, the same
    way Claude's own indicator never sits on one word for two updates in a row."""
    words = SPINNER_WORDS.get(topic, SPINNER_WORDS["general"])
    choices = [w for w in words if w != exclude] or words
    return random.choice(choices)

def node_detail(node_name: str, state: dict) -> str:
    """
    Turns whatever a node actually produced into a real sentence of reasoning text,
    so the thinking UI shows genuine content instead of a decorative status word.
    """
    if node_name == "understand_goal":
        goal = state.get("goal", "")
        search_query = state.get("web_search_query", "")
        prefix = f"Searched the web for '{search_query}'. " if search_query else ""
        return f"{prefix}Understanding the goal: {goal}" if goal else f"{prefix}Understanding the goal."
    if node_name == "planner":
        step = state.get("current_step", "")
        return f"Planning the next step: {step}" if step else "Planning the next step."
    if node_name == "executor":
        reasoning = state.get("executor_reasoning", "")
        return reasoning if reasoning else "Drafting a response for the current step."
    if node_name == "reflector":
        reflection = state.get("reflection", "")
        return reflection if reflection else "Checking the draft for quality and accuracy."
    if node_name == "evaluator":
        score = state.get("completion_score", None)
        return f"Completion check: {score}% of the goal is done." if score is not None else "Checking whether the goal is complete."
    return NODE_LABELS.get(node_name, node_name)

async def run_graph_streaming(initial_state: dict, timeout: float, token_queue: asyncio.Queue):
    """
    Runs the LangGraph workflow node-by-node AND, concurrently, relays whatever
    executor_node pushes onto token_queue as it generates the answer — interleaving both
    into one chronological stream of events:
      ("status", node_name, state_so_far)  — a node just finished (real backend progress)
      ("reset", None)                      — a new executor pass is starting (e.g. the loop
                                              went around for another iteration); the caller
                                              should clear whatever draft it's shown so far
      ("token", text)                      — a real, live delta of the answer being written,
                                              straight from the model, the instant it arrives
      ("spinner_tick", None)               — nothing new finished yet, just a signal to cycle
                                              the spinner word so the status text keeps visibly
                                              changing during a single long-running node call,
                                              the same way Claude's own indicator does
    Enforces the same overall timeout budget as before so a slow run still falls back cleanly.
    """
    SPINNER_HEARTBEAT_INTERVAL = 2.5  # how often to emit a fresh spinner word while a single
                                       # node call is still in flight, instead of only getting
                                       # a new status word once per finished node.

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
    """
    Streams SSE events to the frontend:
      - {"type": "status", "step": <node name>, "label": <text>}   real backend progress —
                                                                     "label" is a themed spinner
                                                                     word (e.g. "Calculating",
                                                                     "Orbiting", "Pondering")
                                                                     drawn from a pool matched to
                                                                     the topic of the user's
                                                                     message, the same way
                                                                     Claude's own status
                                                                     indicator cycles through
                                                                     playful words while it works
      - {"type": "message_reset"}                                  a redo iteration started;
                                                                     clear the chat bubble
      - {"type": "message", "assistant_message": <delta>, ...}     a REAL token, live from
                                                                     the model — not a replay
                                                                     of an already-finished string
    """
    final_response = ""
    topic = classify_topic(request.message)  # "math" | "physics" | "general" — picks the
                                               # spinner word pool for this whole turn
    last_spinner: Optional[str] = None

    if is_simple_message(request.message):
        last_spinner = spinner_word(topic)
        yield f"data: {json.dumps({'type': 'status', 'step': 'direct', 'label': last_spinner, 'detail': 'This is a short message, so answering directly without the full planning loop.'})}\n\n"
        async for piece in answer_directly_stream(
            request.message, session["messages"], request.model_type, request.temperature
        ):
            final_response += piece
            yield f"data: {json.dumps({'type': 'message', 'assistant_message': piece, 'conversation_id': session_id, 'session_id': session_id, 'goal_complete': True})}\n\n"
        final_state = {"completion_score": 100, "iteration": 1}

    else:
        initial_state = {
            "messages": session["messages"],
            "model_type": request.model_type,
            "temperature": request.temperature,
            "max_iterations": request.max_iterations,
            "iteration": 0,
            "completion_score": 0,
            "completed_steps": [],
            "plan": []
        }
        timeout = graph_timeout_seconds(request.model_type, request.max_iterations)
        final_state = initial_state
        token_queue: asyncio.Queue = asyncio.Queue()
        qtoken = _current_token_queue.set(token_queue)

        try:
            async for evt in run_graph_streaming(initial_state, timeout, token_queue):
                kind = evt[0]
                if kind == "status":
                    _, node_name, state_so_far = evt
                    last_spinner = spinner_word(topic, exclude=last_spinner)
                    detail = node_detail(node_name, state_so_far)
                    yield f"data: {json.dumps({'type': 'status', 'step': node_name, 'label': last_spinner, 'detail': detail})}\n\n"
                    final_state = state_so_far
                elif kind == "spinner_tick":
                    # A single node call is still in flight — cycle the spinner word so the
                    # status text keeps visibly changing instead of freezing until that node
                    # finishes, matching how Claude's own indicator behaves mid-thought.
                    last_spinner = spinner_word(topic, exclude=last_spinner)
                    yield f"data: {json.dumps({'type': 'status', 'step': 'thinking', 'label': last_spinner, 'detail': 'Still working on it...'})}\n\n"
                elif kind == "reset":
                    # Another planning iteration started — the draft streamed so far gets
                    # superseded by a fresh executor pass, so clear the bubble instead of
                    # appending the new draft onto the stale one.
                    final_response = ""
                    yield f"data: {json.dumps({'type': 'message_reset'})}\n\n"
                elif kind == "token":
                    final_response += evt[1]
                    yield f"data: {json.dumps({'type': 'message', 'assistant_message': evt[1], 'conversation_id': session_id, 'session_id': session_id, 'goal_complete': False})}\n\n"

            if not final_response.strip():
                # Safety net: state has a response but somehow no tokens made it onto the
                # queue (shouldn't normally happen) — still get an answer to the user.
                fallback_text = final_state.get("response", "Task completed but no response was formulated.")
                if fallback_text:
                    final_response = fallback_text
                    yield f"data: {json.dumps({'type': 'message', 'assistant_message': fallback_text, 'conversation_id': session_id, 'session_id': session_id, 'goal_complete': True})}\n\n"

        except asyncio.TimeoutError:
            print(f"[{session_id}] Streaming workflow timed out after {timeout:.0f}s. Falling back to direct answer.")
            last_spinner = spinner_word(topic, exclude=last_spinner)
            yield f"data: {json.dumps({'type': 'status', 'step': 'fallback', 'label': last_spinner, 'detail': 'The full reasoning loop was taking too long, so falling back to a direct answer.'})}\n\n"
            if final_response:
                yield f"data: {json.dumps({'type': 'message_reset'})}\n\n"
            final_response = ""
            async for piece in answer_directly_stream(
                request.message, session["messages"], request.model_type, request.temperature
            ):
                final_response += piece
                yield f"data: {json.dumps({'type': 'message', 'assistant_message': piece, 'conversation_id': session_id, 'session_id': session_id, 'goal_complete': True})}\n\n"
            final_state = {"completion_score": 100, "iteration": 1}
        except Exception as e:
            print(f"[{session_id}] Streaming workflow failed: {e}. Falling back to direct answer.")
            last_spinner = spinner_word(topic, exclude=last_spinner)
            yield f"data: {json.dumps({'type': 'status', 'step': 'fallback', 'label': last_spinner, 'detail': 'Something went wrong in the reasoning loop, so falling back to a direct answer.'})}\n\n"
            if final_response:
                yield f"data: {json.dumps({'type': 'message_reset'})}\n\n"
            final_response = ""
            async for piece in answer_directly_stream(
                request.message, session["messages"], request.model_type, request.temperature
            ):
                final_response += piece
                yield f"data: {json.dumps({'type': 'message', 'assistant_message': piece, 'conversation_id': session_id, 'session_id': session_id, 'goal_complete': True})}\n\n"
            final_state = {"completion_score": 100, "iteration": 1}
        finally:
            _current_token_queue.reset(qtoken)

    # Now that we actually have the final answer, save it to memory
    session["messages"].append(AIMessage(content=final_response))

def is_simple_message(message: str) -> bool:
    """Quick heuristic: short greetings/small talk don't need the full plan/execute/reflect loop."""
    text = message.strip().lower()
    if len(text) <= 20:
        return True
    greetings = ("hi", "hello", "hey", "yo", "sup", "thanks", "thank you", "ok", "okay", "bye")
    return any(text == g or text.startswith(g + " ") or text.startswith(g + ",") for g in greetings)

def graph_timeout_seconds(model_type: str, max_iterations: int) -> float:
    """
    Sizes the overall graph timeout to the worst-case number of sequential LLM calls
    (1 understand_goal call + up to max_iterations * 4 planner/executor/reflector/evaluator calls),
    with extra headroom per call for the slower 'reasoning' model.
    """
    # Each node call can retry up to twice against a 60s per-call LLM timeout (see get_llm),
    # so the per-call budget here must comfortably exceed 60s or the outer graph deadline
    # will cut off a call that hadn't even hit its own timeout yet.
    per_call_seconds = 130.0 if model_type.strip().lower() == "reasoning" else 100.0
    worst_case_calls = 1 + (max_iterations * 4)
    return min(worst_case_calls * per_call_seconds, 240.0)  # hard ceiling so a request can never hang indefinitely

async def answer_directly(message: str, history: List[BaseMessage], model_type: str, temperature: float) -> str:
    """Always returns a real answer — falls back to the fast model if the primary one times out."""
    # Short/greeting messages can still be genuinely time-sensitive (e.g. "bitcoin
    # price today" is <=20 chars and counts as "simple" per is_simple_message), so
    # this fast path gets the same free web search + Markdown sources/images
    # treatment as the full graph's understand_goal_node/executor_node, just
    # condensed into one function since there's no multi-node pipeline here.
    web_search_results = ""
    sources_md = ""
    if needs_web_search(message):
        query = extract_search_query(message)
        (web_search_results, links), images = await asyncio.gather(
            web_search(query), web_image_search(query)
        )
        sources_md = build_web_sources_markdown(links, images)

    system_content = (
        f"Current Date & Time: {get_current_datetime_str()} — this is the real, current date/time; "
        "trust it completely for anything date-relative rather than assuming a date from your own "
        "training data.\n\n"
        "You are GoalAI, a sharp, honest, genuinely helpful assistant — same quality bar as a "
        "top-tier assistant like Claude, just answering fast for a short message. Be direct and "
        "concise, never robotic or padded. Never invent facts, APIs, or events; if you're not "
        "sure, say so plainly instead of guessing confidently. You are GoalAI specifically — "
        "never call yourself ChatGPT/GPT/an OpenAI model, never mention a 'knowledge cutoff of "
        "Feb 2025', and never tell the user to check platform.openai.com or an OpenAI changelog; "
        "that is not this product. You also do NOT have a callable search/browsing tool of your "
        "own — web search, if needed, already ran automatically before this message reached you, "
        "and any results are given to you below as plain text. Never output a tool call, a "
        "function call, JSON like {\"tool\": ...}, or any code-like invocation syntax as your "
        "answer — always respond in plain natural language only."
    )
    if web_search_results:
        system_content += (
            f"\n\nWeb Search Results (fetched just now because this looks time-sensitive — "
            f"treat this as more current than your own training data and use it):\n{web_search_results}"
        )
    else:
        system_content += (
            "\n\nNo web search ran for this specific message (the app only searches when a "
            "message looks time-sensitive) — this does NOT mean you lack web access in general. "
            "Never claim you can't browse the internet or don't have real-time access; that's "
            "false for this system. Only if this particular topic genuinely depends on something "
            "recent, add a brief, low-key note that this detail specifically might not be fully "
            "current — nothing more than that."
        )

    messages = [SystemMessage(content=system_content)]
    messages.extend(history[-6:])
    messages.append(HumanMessage(content=message))

    # Try the requested model first
    try:
        llm = get_llm(model_type, temperature)
        res = await llm.ainvoke(messages)
        answer = strip_thinking(res.content).strip()
        if answer:
            return answer + sources_md
    except Exception as e:
        print(f"Primary model failed: {e}")

    # Fallback: always try the fast model before giving up
    try:
        fallback_llm = get_llm("fast", temperature)
        res = await fallback_llm.ainvoke(messages)
        answer = strip_thinking(res.content).strip()
        if answer:
            return answer + sources_md
    except Exception as e:
        print(f"Fallback model also failed: {e}")

    # Absolute last resort — user still gets a real response, never a crash
    return "I'm having trouble reaching the model right now. Please try again in a moment."

async def answer_directly_stream(message: str, history: List[BaseMessage], model_type: str, temperature: float):
    """
    Real streaming counterpart to answer_directly(): yields token deltas the instant they
    arrive from the model, instead of returning a single completed string. Used by
    generate_stream() for the short/direct-message path and as its fallback on error/timeout.
    """
    # Same web search + sources/images treatment as answer_directly() above —
    # see its comment for why this fast path needs it too.
    web_search_results = ""
    sources_md = ""
    if needs_web_search(message):
        query = extract_search_query(message)
        (web_search_results, links), images = await asyncio.gather(
            web_search(query), web_image_search(query)
        )
        sources_md = build_web_sources_markdown(links, images)

    system_content = (
        f"Current Date & Time: {get_current_datetime_str()} — this is the real, current date/time; "
        "trust it completely for anything date-relative rather than assuming a date from your own "
        "training data.\n\n"
        "You are GoalAI, a sharp, honest, genuinely helpful assistant — same quality bar as a "
        "top-tier assistant like Claude, just answering fast for a short message. Be direct and "
        "concise, never robotic or padded. Never invent facts, APIs, or events; if you're not "
        "sure, say so plainly instead of guessing confidently. You are GoalAI specifically — "
        "never call yourself ChatGPT/GPT/an OpenAI model, never mention a 'knowledge cutoff of "
        "Feb 2025', and never tell the user to check platform.openai.com or an OpenAI changelog; "
        "that is not this product. You also do NOT have a callable search/browsing tool of your "
        "own — web search, if needed, already ran automatically before this message reached you, "
        "and any results are given to you below as plain text. Never output a tool call, a "
        "function call, JSON like {\"tool\": ...}, or any code-like invocation syntax as your "
        "answer — always respond in plain natural language only."
    )
    if web_search_results:
        system_content += (
            f"\n\nWeb Search Results (fetched just now because this looks time-sensitive — "
            f"treat this as more current than your own training data and use it):\n{web_search_results}"
        )
    else:
        system_content += (
            "\n\nNo web search ran for this specific message (the app only searches when a "
            "message looks time-sensitive) — this does NOT mean you lack web access in general. "
            "Never claim you can't browse the internet or don't have real-time access; that's "
            "false for this system. Only if this particular topic genuinely depends on something "
            "recent, add a brief, low-key note that this detail specifically might not be fully "
            "current — nothing more than that."
        )

    messages = [SystemMessage(content=system_content)]
    messages.extend(history[-6:])
    messages.append(HumanMessage(content=message))

    THINK_TAG = "<think>"

    async def _stream_from(llm):
        # Some models emit a leading <think>...</think> block before the real answer.
        # strip_thinking() worked on a complete string; on a live stream we buffer just
        # until we're past </think>, so those tags never reach the chat bubble.
        #
        # Small-chunk models (many stream token-by-token, sometimes near char-by-char)
        # mean the opening tag itself can arrive split across several deltas — e.g. "<",
        # "th", "ink>". Deciding "not a think tag" the moment the buffer merely fails a
        # full startswith("<think>") check would leak those leading characters before
        # we've actually seen enough to know. Instead: keep waiting as long as what
        # we've buffered so far is still a valid PREFIX of "<think>"; only resolve once
        # it either matches in full or definitively diverges.
        buf = ""
        in_think = False
        resolved_think = False
        async for chunk in llm.astream(messages):
            piece = getattr(chunk, "content", "") or ""
            if not piece:
                continue
            if not resolved_think:
                buf += piece
                stripped = buf.lstrip()
                if not in_think:
                    if stripped.startswith(THINK_TAG):
                        in_think = True
                    elif THINK_TAG.startswith(stripped):
                        continue  # still an ambiguous prefix — wait for more chars
                    else:
                        resolved_think = True
                        if buf:
                            yield buf
                        continue
                if in_think:
                    if "</think>" in buf:
                        buf = buf.split("</think>", 1)[1]
                        in_think = False
                        resolved_think = True
                        if buf:
                            yield buf
                    continue  # still inside <think>, nothing to emit yet
                continue
            yield piece

    got_any = False
    try:
        async for piece in _stream_from(get_llm(model_type, temperature)):
            got_any = True
            yield piece
    except Exception as e:
        print(f"Primary model streaming failed: {e}")

    if got_any:
        if sources_md:
            yield sources_md
        return

    try:
        async for piece in _stream_from(get_llm("fast", temperature)):
            got_any = True
            yield piece
    except Exception as e:
        print(f"Fallback model streaming also failed: {e}")

    if not got_any:
        yield "I'm having trouble reaching the model right now. Please try again in a moment."
    elif sources_md:
        yield sources_md

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
            "iterations": 0
        }

async def _handle_chat(request: ChatRequest):
    session_id = request.session_id
    
    if session_id not in sessions:
        sessions[session_id] = {"messages": []}
        
    session = sessions[session_id]
    llm = get_llm(request.model_type, request.temperature)

    # 1. Manage memory size
    session["messages"] = await summarize_memory(session["messages"], llm)

    # 2. Append the new human message
    session["messages"].append(HumanMessage(content=request.message))

    # 3. Streaming requests get real-time node-by-node progress from the graph itself —
    #    hand off to the generator now instead of running the graph synchronously first.
    if request.stream:
        return StreamingResponse(
            generate_stream(request, session, session_id),
            media_type="text/event-stream"
        )

    # 4. Non-streaming (plain JSON) path. Small talk / short messages skip the plan-execute-reflect
    #    loop entirely — no reason to pay for 5+ sequential LLM calls to answer "hi" or "thanks".
    if is_simple_message(request.message):
        final_response = await answer_directly(
            request.message, session["messages"], request.model_type, request.temperature
        )
        final_state = {"completion_score": 100, "iteration": 1}
    else:
        # 4. Setup Initial State for the LangGraph Workflow
        initial_state = {
            "messages": session["messages"],
            "model_type": request.model_type,
            "temperature": request.temperature,
            "max_iterations": request.max_iterations,
            "iteration": 0,
            "completion_score": 0,
            "completed_steps": [],
            "plan": []
        }

        # 5. Size the timeout to the worst-case number of sequential LLM calls this request
        #    could trigger, instead of a flat 20s that's fine for small tasks but too short
        #    for anything that needs multiple plan/execute/reflect iterations.
        timeout = graph_timeout_seconds(request.model_type, request.max_iterations)

        try:
            # Executes the full Goal->Plan->Execute->Reflect loop
            final_state = await asyncio.wait_for(app_graph.ainvoke(initial_state), timeout=timeout)
            final_response = final_state.get("response", "Task completed but no response was formulated.")
        except asyncio.TimeoutError:
            print(f"[{session_id}] Workflow timed out after {timeout:.0f}s. Falling back to direct answer.")
            final_response = await answer_directly(
                request.message, session["messages"], request.model_type, request.temperature
            )
            final_state = {"completion_score": 100, "iteration": 1}
        except Exception as e:
            print(f"[{session_id}] Workflow failed: {e}. Falling back to direct answer.")
            final_response = await answer_directly(
                request.message, session["messages"], request.model_type, request.temperature
            )
            final_state = {"completion_score": 100, "iteration": 1}

    # 5. Append AI final answer to memory
    session["messages"].append(AIMessage(content=final_response))

    # 6. Return the plain JSON response (streaming requests already returned in step 3)
    return {
        "response": final_response,
        "session_id": session_id,
        "goal_progress": final_state.get("completion_score", 100),
        "completed": final_state.get("completion_score", 100) >= 100,
        "iterations": final_state.get("iteration", 1)
    }

# ==================================================================================
# CODE MODE SECTION  —  fully separate from the normal chat section above.
#
# This section only handles "code with reasoning" requests. It has its own
# models, its own session memory, its own LangGraph workflow, and its own
# FastAPI endpoint. Nothing here touches or is touched by the normal /chat flow.
# ==================================================================================

# ----------------------------------------------------------------------
# CODE MODE: Structured Output Models
#
# One shape per stage of the Claude AI coding workflow:
#
#   Idea -> Plan -> Code -> Test -> Review -> Fix -> Commit
#
# IdeaAnalysis and CodePlan split what used to be a single combined
# "think about it" call into two real stages (understand the ask, THEN decide
# the concrete next step) — the same read-before-you-touch-anything judgment
# an engineer makes, just modeled as two explicit steps instead of one. Code
# reuses CodeExecutorOutput's shape conceptually (see code_node). Test and
# Review are separate, honest passes — no artificial 0-100 "completion
# percentage" score; the judgment calls (already_done, passed, looks_correct)
# are what drive the loop. Fix reuses the Code stage's generation path. Commit
# is a lightweight plain-text summary, not a structured call (see commit_node).
# ----------------------------------------------------------------------

class IdeaAnalysis(BaseModel):
    understanding: str = Field(description="Plain-language restatement of what's actually being asked, the way an engineer would summarize a ticket to themselves before opening an editor")
    relevant_context: str = Field(default="", description="What in the existing code (if any) is relevant here: which functions, files, or behavior this touches and what they currently do. Empty string if there's no existing code yet.")
    root_cause: str = Field(default="", description="For a bug report: the actual underlying cause of the problem, not just the symptom. Empty string if this isn't a bug fix.")
    already_done: bool = Field(default=False, description="True only if, having looked at what already exists, nothing further is actually needed")

class CodePlan(BaseModel):
    next_step: str = Field(description="The exact, concrete next action to take — specific enough to act on immediately, e.g. 'fix the off-by-one in the pagination loop' or 'build the login form component'")
    is_multi_file: bool = Field(description="True only if this genuinely needs several distinct files/pages/modules that cannot reasonably live in one runnable file. False for anything that fits naturally in a single file/component/instant preview.")
    estimated_file_count: int = Field(default=1, description="Rough number of separate files needed to build this well. Use 1 for single-file tasks.")
    target_file: str = Field(default="", description="The filename next_step targets, e.g. 'index.html', 'auth.js'. Leave empty for single-file tasks.")

class CodeExecutorOutput(BaseModel):
    code: str = Field(description="The generated code for the current step, complete and runnable")
    language: str = Field(description="Programming language of the code, e.g. python, javascript, html, jsx, css")
    filename: str = Field(default="", description="The filename this code belongs to for multi-file tasks, e.g. 'index.html'. Empty string for single-file tasks.")
    explanation: str = Field(description="Brief explanation of what the code does")
    is_frontend: bool = Field(description="True if this code renders a UI in a browser (html/css/js/react/vue/etc), false if it is backend/server/CLI/script code")

class TestReport(BaseModel):
    passed: bool = Field(description="True if the code, reasoned through against realistic unit/integration/edge-case tests, would actually pass them")
    test_notes: str = Field(default="", description="What was checked (concrete test cases and edge cases considered) and the result — concise, like a test-run summary")
    issues: str = Field(default="", description="Specific, concrete failures found — exact enough to act on directly. Empty string if none.")

class CodeReview(BaseModel):
    looks_correct: bool = Field(description="False if there's a real bug, missed requirement, or broken edge case in what was just written")
    issues: str = Field(default="", description="Specific, concrete problems found — exact enough to act on directly. Empty string if none.")
    notes: str = Field(default="", description="Anything else worth mentioning about quality or trade-offs, even when looks_correct is True")

# ----------------------------------------------------------------------
# CODE MODE: Graph State
# ----------------------------------------------------------------------

class CodeAgentState(TypedDict):
    messages: List[BaseMessage]
    goal: str
    current_step: str
    target_file: str
    analysis_notes: str
    test_notes: str
    test_passed: bool
    review_notes: str
    needs_fix: bool
    already_done: bool
    completion_score: int
    iteration: int
    max_iterations: int
    code: str
    language: str
    is_frontend: bool
    explanation: str
    response: str
    commit_message: str
    model_key: str
    temperature: float
    is_multi_file: bool
    estimated_file_count: int
    files: Dict[str, str]
    file_languages: Dict[str, str]
    last_error: Optional[str]
    web_search_query: str
    web_search_results: str

# ----------------------------------------------------------------------
# CODE MODE: Session Memory (kept separate from the normal chat sessions)
# ----------------------------------------------------------------------

code_sessions: Dict[str, Dict[str, Any]] = {}

# ----------------------------------------------------------------------
# CODE MODE: Model Router — three models, all selectable by the user
# ----------------------------------------------------------------------

CODE_MODEL_MAP = {
    "kimi": "nvidia/nemotron-3-ultra-550b-a55b",   # high-end reasoning and coding
    "glm": "deepseek-ai/deepseek-v4-pro",          # fast response with code
    "kimik2.6": "poolside/laguna-xs-2.1",          # real Kimi K2.6 — runs WITH thinking on, see below
}

# "kimi" and "glm" ship with "Thinking" mode ON by default on NVIDIA NIM. That's
# the actual reason Code Mode was returning no real answers for them: for
# structured JSON calls (goal extraction, planning, execution, reflection,
# evaluation), when the model's reasoning trace doesn't finish closing before
# max_tokens is hit, the API hands back an empty/null final `content` — the
# whole token budget went to reasoning_content instead of the answer. That's a
# documented NIM behavior (chat_template_kwargs), not something retries fix,
# since the same elaborate system prompt makes the model "think" the same way
# every attempt. Different model families use different toggle keys ("thinking"
# vs "enable_thinking"); sending both every time — instead of keying off a
# hardcoded model-name list — means this keeps working no matter which models
# CODE_MODEL_MAP points to; unrecognized keys are harmlessly ignored by the
# chat template.
CODE_THINKING_OFF_KWARGS = {"thinking": False, "enable_thinking": False}

# kimik2.6 is the one deliberate exception: thinking is left ON for it (same
# max_tokens=32768 budget as the other two models below — nothing about the
# budget changes, the model just gets to spend part of it on a visible
# reasoning trace instead of having that trace suppressed). This reintroduces
# the empty-content risk described above for kimik2.6's structured calls inside
# the coding graph (goal/plan/execute/reflect/evaluate) if its reasoning runs
# long — that tradeoff is intentional here, not an oversight.
CODE_THINKING_ON_KWARGS = {"thinking": True, "enable_thinking": True}

def get_code_llm(model_key: str, temperature: float = 0.2) -> ChatNVIDIA:
    """Routes to one of the three Code Mode models. Uses the same NVIDIA_API_KEY as the rest of the app."""
    key = (model_key or "").strip().lower()
    model_name = CODE_MODEL_MAP.get(key, CODE_MODEL_MAP["kimi"])  # default to the high-reasoning model
    thinking_kwargs = CODE_THINKING_ON_KWARGS if key == "kimik2.6" else CODE_THINKING_OFF_KWARGS
    return ChatNVIDIA(
        model=model_name,
        temperature=temperature,
        max_tokens=32768,
        timeout=280,
        model_kwargs={"chat_template_kwargs": thinking_kwargs} if thinking_kwargs else {},
    )

# ----------------------------------------------------------------------
# CODE MODE: Reasoning Level -> Max Iterations
# ----------------------------------------------------------------------

CODE_REASONING_ITERATIONS = {
    "low": 2,
    "medium": 3,
    "high": 4,
    "max": 5,
}

def get_code_max_iterations(reasoning_level: str) -> int:
    """Starting floor for iterations. For multi-file builds this gets raised dynamically
    by plan_node once it estimates how many files are actually needed (see there)
    — a 15-file app needs more passes than any fixed reasoning-level ceiling allows for."""
    return CODE_REASONING_ITERATIONS.get((reasoning_level or "").strip().lower(), CODE_REASONING_ITERATIONS["medium"])

# ----------------------------------------------------------------------
# CODE MODE: "Simple message" check — deliberately NOT the same heuristic as
# is_simple_message() used by the normal chat section. That one treats ANY
# message <=20 chars as small talk, which wrongly swallows real coding asks
# like "fix this bug" or "add a button" (both under 20 chars) — those would
# skip the whole plan/execute loop (and the seeded prior-code context) and
# go straight to a single freeform LLM call with no guarantee of returning
# code at all. Here only genuine greetings/acknowledgements count as simple.
# ----------------------------------------------------------------------

CODE_MODE_GREETINGS = ("hi", "hello", "hey", "yo", "sup", "thanks", "thank you", "ok", "okay", "bye")

def is_simple_code_message(message: str) -> bool:
    text = message.strip().lower()
    return any(text == g or text.startswith(g + " ") or text.startswith(g + ",") for g in CODE_MODE_GREETINGS)

# ----------------------------------------------------------------------
# CODE MODE: System Prompt — intentionally left empty, to be filled in separately
# ----------------------------------------------------------------------

CODE_SYSTEM_PROMPT = """You are an elite software engineering AI — a dedicated build/plan/execute coding agent held to the same bar as the best coding assistants available today (Claude-level engineering judgment, not a generic code-completion model). Your only job is solving real software engineering tasks end-to-end: understanding what's actually being built, researching what you don't already know for certain, planning the architecture, and then writing code that a senior engineer would approve without a second pass.

Your domain covers:
- Programming, software architecture, algorithms and data structures
- Debugging and root-cause analysis
- Refactoring and code review
- API design, databases, DevOps
- Testing, performance optimization, security
- Documentation, system design

Never switch into a general, non-technical assistant role. If a request is genuinely unrelated to software engineering, politely refuse and explain that you are a dedicated coding agent.

--------------------------------------------------
PRIMARY OBJECTIVE
--------------------------------------------------

Produce correct, maintainable, production-quality software — not a toy sketch that merely resembles an answer. Optimize for correctness, robustness, readability, maintainability, scalability, security, and performance, in that rough order of priority. Quality always beats speed; never trade a correct answer for a shorter one.

--------------------------------------------------
RESEARCH BEFORE YOU BUILD
--------------------------------------------------

Treat "what does this actually require" as a real research question, not something to pattern-match from memory alone:

- Read the full request and any existing code/history in context before deciding what to build. Ground every claim about existing code, a library, or an API in what was actually shown to you — never invent a function signature, config key, or file that wasn't given.
- If the state below includes "Web Search Results," that content was fetched moments ago because the task involves something your training data can't be trusted for — a current library version, a recent API/framework change, up-to-date best practice, a changelog, or similar. Treat it as ground truth over your own recollection, use it to inform the plan and the code you write, and don't silently ignore it.
- If that section is empty or says "None," it just means this particular request wasn't judged to need a search — it does NOT mean you're unable to search. Never tell the user you "can't access the internet" or "can't check for the latest version" as a blanket statement. If a real dependency's current version genuinely matters and you're not certain of it, name that one specific uncertainty and state your best-known/assumed version rather than claiming a general inability to look things up.
- If you're not certain a library, API, or language feature behaves the way you're about to rely on, say so plainly (as a one-line assumption) rather than asserting it as fact and hoping it compiles. Never randomly guess an API surface.
- When something is missing (target language/framework version, data shape, scale, existing file layout), pick the most reasonable default, state it in one line, and keep moving — don't block on a question you can answer yourself with a sane assumption.
- You are this app's dedicated coding agent — not ChatGPT/GPT/Codex or any OpenAI product. Never claim a "knowledge cutoff of Feb 2025" or send the user to platform.openai.com or an OpenAI changelog; that's not this system. You also do NOT have a callable search/browsing tool to invoke yourself — any web search already ran automatically before your turn, and results (if any) are given to you as plain text above. Never emit a tool call, function call, or JSON like {"tool": "search", ...} as your answer — respond with real prose and real code only, never invocation syntax pretending to call something you don't have.

--------------------------------------------------
THINK, THEN DESIGN, THEN CODE
--------------------------------------------------

Before producing code:
1. Understand the entire problem — not just the first sentence of it.
2. Identify real constraints and edge cases, including ones the user didn't spell out.
3. Design the shape of the solution before writing a line of implementation.
4. Prefer the simplest design that actually satisfies the requirements over a clever one that impresses but overcomplicates.
5. Avoid unnecessary abstraction layers; avoid needless dependencies.
6. Stay in scope — change only what the task asked for. If you notice an unrelated problem worth fixing, name it briefly instead of rewriting it uninvited.

--------------------------------------------------
ENGINEERING STANDARDS
--------------------------------------------------

Write code as if someone else will maintain it for years: modular, readable, testable, reusable, documented, and typed wherever the language supports it. Avoid duplicated logic, magic numbers, deep nesting, hidden side effects, and global mutable state. Prefer pure functions, composition, dependency injection, descriptive naming, and explicit behavior over implicit magic.

--------------------------------------------------
BUG PREVENTION & SECURITY
--------------------------------------------------

Actively check for null references, race conditions, resource leaks, off-by-one errors, integer overflow, invalid assumptions, concurrency issues, and API misuse — catch these before they ship, not after. Apply real security discipline by default: input validation, output escaping, safe authentication/authorization patterns, injection-attack prevention, proper secrets handling, and secure defaults. Never generate insecure production code when a secure alternative is just as easy to write.

--------------------------------------------------
PERFORMANCE, TESTING, ERROR HANDLING
--------------------------------------------------

Consider time/space complexity and scalability where it matters, without prematurely optimizing where it doesn't. Write code that's easy to test, and suggest concrete unit/integration tests or edge cases when that adds real value. Handle failures explicitly with meaningful errors instead of silent or swallowed exceptions.

--------------------------------------------------
COMMUNICATION
--------------------------------------------------

Be concise but technically complete — explain the reasoning behind a non-obvious decision (e.g. "used a queue here instead of polling because X"), skip explaining the obvious, and never pad the response with filler. Focus on the engineering.

--------------------------------------------------
MODIFYING, DEBUGGING, REFACTORING, WRITING NEW CODE
--------------------------------------------------

- Modifying existing code: preserve public APIs, behavior, and compatibility unless breaking changes were explicitly requested. Minimize unnecessary diffs.
- Debugging: find the actual root cause and explain why the bug happened — a fix that only hides the symptom is not acceptable.
- Refactoring: improve readability, maintainability, and architecture without changing observable behavior.
- Writing new code: prefer complete, production-ready implementations over toy examples or placeholders; avoid pseudocode unless it was explicitly requested.

--------------------------------------------------
OUTPUT REQUIREMENTS
--------------------------------------------------

Generated code should compile/run whenever possible, follow language and ecosystem conventions, include all necessary imports and dependencies, and be complete — never omit sections and call it "left as an exercise" unless the user explicitly asked for a partial sketch.

--------------------------------------------------
FINAL VALIDATION (DO THIS BEFORE YOU RESPOND)
--------------------------------------------------

Before finishing, verify against the actual requirement — not a simplified version of it you found easier to satisfy:
✓ Requirements satisfied  ✓ Logic is correct  ✓ Edge cases considered
✓ Security reviewed  ✓ Performance acceptable  ✓ Code is maintainable
✓ No unnecessary complexity  ✓ Nothing was silently invented or assumed without saying so

Only then produce the final answer.

--------------------------------------------------
OUTPUT FORMAT CONSTRAINT (CRITICAL — READ THIS)
--------------------------------------------------

You can only return ONE code block per turn (one 'code' + one 'language' field).
There is no mechanism to deliver separate files in the same turn.

Because of this:
- For any frontend/web task (HTML, CSS, JS, React, Vue, Svelte, a "website",
  "app", "page", "component", "UI"), you MUST produce a single self-contained
  .html file with ALL CSS inside a <style> tag and ALL JS inside a <script>
  tag, in the <head>/<body> of that same file. The 'language' field for
  every such task MUST be "html" — never "css", "javascript", "jsx", "tsx",
  "vue", or "svelte" on their own. Those languages cannot run standalone in
  a browser preview with no build step; only a plain .html file can.
  NEVER write <link rel="stylesheet" href="styles.css"> or
  <script src="app.js"> — those files will never exist, and the page will
  silently render unstyled and non-functional.
- If the user explicitly asks for React/Vue/Svelte source code itself (not a
  preview of it running), that's fine to return as its own language — but
  if the goal is to SHOW or PREVIEW a working UI, always compile it down to
  vanilla HTML/CSS/JS (or load the framework from a CDN <script> tag inside
  the same HTML file) rather than returning bare framework source.
- If a task genuinely requires multiple real files (e.g. a Python package),
  pick the single most important file for this step and say in your
  explanation which other files still need separate follow-up turns —
  do not silently drop files with no mention.

--------------------------------------------------
UI / VISUAL DESIGN STANDARDS (CRITICAL — APPLIES ANY TIME YOU BUILD, DESIGN,
OR GENERATE A UI: A WEBSITE, APP, PAGE, DASHBOARD, COMPONENT, OR TOOL)
--------------------------------------------------

Icons and symbols:

- NEVER use emoji characters anywhere in generated UI — not in buttons,
  headers, nav items, labels, placeholders, empty states, toasts, or any
  other on-screen text. This rule has no exceptions.
- Use real icon assets from established open-source icon sets only —
  e.g. Lucide, Feather Icons, Heroicons, Font Awesome (Free), Bootstrap
  Icons, Material Symbols, or Simple Icons for brand/tech logos. Inline
  the SVG markup directly or load the set from its official CDN.
- Never fake an icon with an emoji, a random Unicode glyph, or a text
  abbreviation standing in for one.

Images:

- Whenever a design calls for a photo, hero image, avatar, product shot,
  or any other picture, you MUST emit a real `<img src="...">` element
  that actually loads — NEVER a bare link/anchor with text like "image"
  or "photo", NEVER a filename that points at a local asset that was not
  provided to you (e.g. "images/hero.jpg", "./photo1.png"), and NEVER a
  placeholder string in place of a working URL.
- Since you cannot browse for a specific real photo, use a stable
  image-generation/placeholder service that returns an actual image byte
  stream at request time, so the `<img>` tag renders something real
  instead of a broken icon or a link. Good defaults:
  - `https://picsum.photos/seed/<unique-seed>/<width>/<height>` for
    generic photographic filler (vary the seed per image so they differ).
  - `https://source.unsplash.com/<width>x<height>/?<topic-keywords>` when
    the image should match a topic/theme (e.g. "coffee", "mountains").
  - For avatars, `https://i.pravatar.cc/<size>?img=<1-70>`.
- Always set `alt` text describing the image, and always set explicit
  width/height (or CSS) so the layout doesn't jump while the image loads.
- Never wrap the only content of a card/section in an `<a>` tag with no
  `<img>` inside it when the design calls for a picture — the visible
  element must be the image itself, not a text link standing in for it.

Color and theme:

- Match the look of modern, production AI-assistant interfaces (ChatGPT,
  Claude, Gemini-style): true dark backgrounds in the near-black / dark
  charcoal range (roughly #0d0d0d–#212121), high-contrast readable text,
  clearly defined surface/border layers, and one deliberate accent color
  for interactive elements (links, primary buttons, active states).
- Support both a dark theme and a light theme; default to dark unless the
  user says otherwise, and include a working theme toggle when reasonable.
- NEVER use pastel, washed-out, or "paled" color palettes.
- Avoid both extremes: don't ship a bare, flat, single-color minimal
  template, and don't ship a cluttered, noisy layout either. Aim for the
  same polished, considered density as a real shipped product.

--------------------------------------------------
CLOSING DISCIPLINE
--------------------------------------------------

Work like a senior engineer scoping a real change, not someone
pattern-matching to the nearest example. Before handing back the final
answer, do one last honest pass: does this genuinely solve what was
asked, with nothing invented, no unstated assumptions, and no shortcut
that would embarrass you in code review?
"""

# ----------------------------------------------------------------------
# CODE MODE: Frontend vs Backend Detection (drives whether a live preview
# is shown, the same way Gemini Canvas / Claude Artifacts only preview
# renderable frontend code and just show a code block for backend code)
# ----------------------------------------------------------------------

FRONTEND_CODE_LANGUAGES = {
    "html", "htm", "css", "scss", "sass", "javascript", "js", "jsx",
    "typescript", "ts", "tsx", "vue", "svelte", "react",
}
BACKEND_CODE_LANGUAGES = {
    "python", "py", "java", "c", "cpp", "c++", "csharp", "c#", "go", "golang",
    "rust", "ruby", "php", "sql", "bash", "shell", "sh", "kotlin", "swift",
    "scala", "perl", "r", "dart", "elixir", "haskell", "lua",
}

_FRONTEND_CODE_SIGNALS = (
    "<html", "<!doctype html", "<div", "<body", "document.getelementbyid",
    "usestate(", "import react", "from 'react'", "<template>", "createroot",
    "addeventlistener", "queryselector", "export default function",
)
_BACKEND_CODE_SIGNALS = (
    "def ", "import flask", "@app.route", "public static void main",
    "using system;", "func main(", "package main", "import fastapi",
    "select * from", "#include <", "class ", "require(",
)

def classify_code_target(language: str, code: str) -> bool:
    """
    Returns True if the code should be treated as frontend (preview-able), False if backend.
    Trusts the explicit language first; falls back to a content heuristic only if the
    language is missing or ambiguous.
    """
    lang = (language or "").strip().lower()
    if lang in FRONTEND_CODE_LANGUAGES:
        return True
    if lang in BACKEND_CODE_LANGUAGES:
        return False

    code_lower = (code or "").lower()
    fe_score = sum(1 for s in _FRONTEND_CODE_SIGNALS if s in code_lower)
    be_score = sum(1 for s in _BACKEND_CODE_SIGNALS if s in code_lower)
    if fe_score > be_score:
        return True
    return False  # default: no preview when unclear (safer than a broken preview)

# ----------------------------------------------------------------------
# CODE MODE: Single-file bundling guard
#
# The "always one self-contained HTML file" rule (see OUTPUT FORMAT CONSTRAINT
# above) is enforced only by the prompt, so on a follow-up turn ("add a navbar")
# the model sometimes answers just the narrow slice it was asked for instead of
# re-emitting the whole bundled page. When that happens this does one cheap
# follow-up call to merge the new snippet back into the previous full file, so
# the frontend Canvas always gets a complete, runnable single file.
# ----------------------------------------------------------------------

async def merge_html_bundle(prior_code: str, new_snippet: str, model_key: str, temperature: float) -> str:
    if not prior_code:
        return new_snippet  # nothing to merge into (first turn) — return as-is

    llm = get_code_llm(model_key, temperature)
    prompt = (
        "Merge the 'New Snippet' into the 'Existing File' below. Produce ONE complete, "
        "self-contained HTML file with ALL CSS inside a single <style> tag and ALL JS inside "
        "a single <script> tag, both inside that same file. Never use <link rel=\"stylesheet\"> "
        "or <script src=...> pointing at separate files. Preserve everything from the existing "
        "file that the new snippet doesn't change. Return ONLY the raw HTML — no explanation, "
        "no markdown code fences.\n\n"
        f"Existing File:\n{prior_code}\n\nNew Snippet:\n{new_snippet}"
    )
    try:
        res = await llm.ainvoke([HumanMessage(content=prompt)])
        merged = strip_thinking(res.content).strip()
        fence_match = CODE_FENCE_RE.search(merged)  # strip fences if the model added them anyway
        if fence_match:
            merged = fence_match.group(2).strip()
        return merged if merged else new_snippet
    except Exception as e:
        print(f"[CodeMode] HTML merge pass failed: {e}")
        return new_snippet  # fall back to whatever we had rather than losing the answer

async def bundle_into_html(code: str, language: str, goal: str, prior_code: str, model_key: str, temperature: float) -> str:
    """Converts a non-HTML frontend answer (bare CSS, bare JS, a JSX component, a Vue SFC,
    etc.) into one self-contained HTML file. Nothing but plain HTML can be dropped into an
    iframe and previewed with no build step, so whatever framework/language the model chose,
    this compiles it down to vanilla HTML/CSS/JS (or a CDN <script> tag for the framework,
    if the snippet genuinely needs the framework itself) inside a single file."""
    llm = get_code_llm(model_key, temperature)
    context = f"Existing HTML file to merge into:\n{prior_code}\n\n" if prior_code else ""
    prompt = (
        f"The snippet below is written in {language}, which cannot run standalone in a "
        "browser preview with no build step. Convert it into ONE complete, self-contained "
        "HTML file: put all CSS inside a single <style> tag and all JS inside a single "
        "<script> tag, both inside that same file. Prefer plain vanilla JS/CSS equivalents; "
        "only load a framework from a CDN <script> tag if the snippet genuinely can't work "
        f"without it. Goal: {goal}\n\n{context}"
        f"Snippet ({language}):\n{code}\n\n"
        "Return ONLY the raw HTML — no explanation, no markdown code fences."
    )
    try:
        res = await llm.ainvoke([HumanMessage(content=prompt)])
        bundled = strip_thinking(res.content).strip()
        fence_match = CODE_FENCE_RE.search(bundled)
        if fence_match:
            bundled = fence_match.group(2).strip()
        return bundled if bundled else code
    except Exception as e:
        print(f"[CodeMode] bundle_into_html pass failed: {e}")
        return code  # fall back to the unconverted snippet rather than losing the answer

# A deliberately generous keyword list: false positives just cost one extra cheap LLM
# call (add_missing_js is a no-op-safe repair), false negatives silently ship a dead
# page, so err toward triggering the check.
_JS_INTERACTIVITY_KEYWORDS = (
    "click", "button", "toggle", "interactive", "calculator", "todo", "to-do",
    "game", "quiz", "form", "validate", "validation", "filter", "search",
    "drag", "drop", "animate", "animation", "counter", "timer", "slider",
    "carousel", "modal", "popup", "fetch", "api call", "dynamic", "add task",
    "delete task", "sort", "score", "submit", "login", "sign up", "chatbot",
    "calendar", "clock", "stopwatch", "convert", "converter", "generator",
)

def _needs_js(goal: str, current_step: str) -> bool:
    """Heuristic: does the goal/current step imply behavior that requires JavaScript?"""
    text = f"{goal} {current_step}".lower()
    return any(keyword in text for keyword in _JS_INTERACTIVITY_KEYWORDS)

async def add_missing_js(code: str, goal: str, current_step: str, model_key: str, temperature: float) -> str:
    """Repair pass for an HTML answer that has no <script> tag at all despite the goal
    clearly needing interactive behavior (e.g. a todo list that can't add/remove items).
    Adds the missing JavaScript inline, preserving the existing HTML/CSS untouched."""
    llm = get_code_llm(model_key, temperature)
    prompt = (
        "The HTML file below is missing the JavaScript needed to make it actually work. "
        f"Goal: {goal}\nCurrent step: {current_step}\n\n"
        "Add the necessary JavaScript inside a single <script> tag in the same file so the "
        "page is fully functional — do not just style it, make it actually do what the goal "
        "describes. Preserve the existing HTML and CSS as-is. Never use <script src=...> "
        "pointing at a separate file. Return ONLY the raw HTML — no explanation, no markdown "
        f"code fences.\n\nExisting File:\n{code}"
    )
    try:
        res = await llm.ainvoke([HumanMessage(content=prompt)])
        repaired = strip_thinking(res.content).strip()
        fence_match = CODE_FENCE_RE.search(repaired)
        if fence_match:
            repaired = fence_match.group(2).strip()
        return repaired if repaired else code
    except Exception as e:
        print(f"[CodeMode] add_missing_js repair pass failed: {e}")
        return code  # fall back to the unrepaired version rather than losing the answer

# ----------------------------------------------------------------------
# CODE MODE: Single-file frontend enforcement — runs ONCE per user turn
#
# Guards 0/1/2 (bundle non-HTML into HTML, restore dropped style/script, add missing JS)
# used to run inside code_node itself, i.e. on every internal think/act
# loop iteration. Only the LAST iteration's output actually reaches the user (each
# iteration overwrites "code"/"language" in state), so re-running these checks on every
# earlier iteration wasted calls for nothing AND stacked several extra sequential LLM
# calls inside a single node — which is what pushed some turns past the graph's timeout
# budget and produced a silent "no code" fallback instead of a slow-but-correct answer.
# Running it once here, on the final result only, fixes both problems.
# ----------------------------------------------------------------------

async def enforce_single_file_frontend(
    code: str, language: str, is_frontend: bool,
    goal_hint: str, prior_code: str, model_key: str, temperature: float
) -> tuple[str, str, bool]:
    if not code or not is_frontend:
        return code, language, is_frontend

    lang_lower = language.strip().lower()

    # Any non-HTML frontend language (bare CSS/JS, JSX, Vue SFC, etc.) can't be previewed
    # standalone in an iframe with no build step — bundle it into one real HTML file.
    if lang_lower in FRONTEND_CODE_LANGUAGES and lang_lower not in ("html", "htm"):
        code = await bundle_into_html(code, language, goal_hint, prior_code, model_key, temperature)
        language, lang_lower = "html", "html"

    if lang_lower in ("html", "htm"):
        # Merge back anything the prior file had that this answer silently dropped.
        if prior_code:
            code_lower, prior_lower = code.lower(), prior_code.lower()
            looks_like_full_page = ("<html" in code_lower) or ("<!doctype" in code_lower)
            dropped_style = ("<style" in prior_lower) and ("<style" not in code_lower)
            dropped_script = ("<script" in prior_lower) and ("<script" not in code_lower)
            if not looks_like_full_page or dropped_style or dropped_script:
                code = await merge_html_bundle(prior_code, code, model_key, temperature)

        # Add JS once, at the end, if the overall goal clearly needs interactivity and
        # the final answer still has none.
        if "<script" not in code.lower() and _needs_js(goal_hint, ""):
            code = await add_missing_js(code, goal_hint, "", model_key, temperature)

    return code, language, is_frontend

def rebuild_response_with_code(final_response: str, new_code: str, new_language: str) -> str:
    """After enforce_single_file_frontend changes the code, the fenced code block inside
    final_response (built earlier, before post-processing) is stale — rebuild it, keeping
    whatever prose/explanation came before the fence."""
    explanation_part = final_response.split("```")[0].strip() if "```" in final_response else final_response.strip()
    return f"{explanation_part}\n\n```{new_language}\n{new_code}\n```".strip()

# ----------------------------------------------------------------------
# CODE MODE: Structured LLM Execution Helper (mirrors execute_llm_structured,
# but uses CODE_SYSTEM_PROMPT instead of the GoalAI persona)
# ----------------------------------------------------------------------

async def execute_code_llm_structured(llm: ChatNVIDIA, prompt_str: str, pydantic_model, state: dict, retries: int = 2):
    """Executes an LLM call and ensures structured Pydantic output.

    Returns (parsed_result, error_message). On success error_message is None.
    On failure parsed_result is None and error_message holds the real reason
    (e.g. an auth/rate-limit/timeout error from the model call, or the actual
    parsing failure) — previously this was only printed to server logs, so
    every failure looked identical to the caller and the user just saw a
    generic canned message with no way to tell what actually went wrong.
    """
    parser = PydanticOutputParser(pydantic_object=pydantic_model)
    format_instructions = parser.get_format_instructions()

    base_system = CODE_SYSTEM_PROMPT if CODE_SYSTEM_PROMPT.strip() else "You are an elite, Claude-level coding assistant. Research and plan before writing code, ground every claim in what you actually know, never fabricate an API, and return structured, well-formatted, production-quality output."
    # CODE_SYSTEM_PROMPT is a static module-level constant, so "now" can't be baked
    # into it at import time — a long-running server would hand the model a date
    # that's stale by however long it's been up. Stamp the real current date/time
    # on fresh, per-call, the same way format_code_context() does.
    base_system = f"Current Date & Time: {get_current_datetime_str()} — this is the real, current date/time; trust it completely (what year it is, how recent a library/framework version is, whether Web Search Results below are stale) rather than assuming a date from your own training.\n\n" + base_system

    # format_instructions is raw JSON-schema text full of literal { } characters.
    # It must be handed to ChatPromptTemplate as a template VARIABLE (filled in at
    # .ainvoke time), never concatenated into the template string itself — otherwise
    # LangChain tries to parse every brace in the schema as a template placeholder
    # and throws on every single call. This mirrors execute_llm_structured above,
    # which already does it the safe way.
    prompt = ChatPromptTemplate.from_messages([
        ("system", base_system + "\n\n{format_instructions}"),
        ("user", prompt_str)
    ])

    chain = prompt | llm

    last_error = "Unknown error"
    last_raw_content = ""
    for attempt in range(retries):
        try:
            res = await chain.ainvoke({"format_instructions": format_instructions, **state})
            last_raw_content = res.content or ""
            content = strip_thinking(last_raw_content).strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            return parser.parse(content.strip()), None
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            print(f"[CodeMode] Structured Parsing Retry {attempt + 1}/{retries} failed: {e} | Raw content: {res.content[:300] if 'res' in dir() else 'N/A'}")
            await asyncio.sleep(0.5)

    # Last-ditch salvage: a truncated/malformed JSON response (e.g. hit max_tokens mid-object)
    # very often still contains a perfectly good — or partially-written — code block inside
    # the raw text. Pulling that out beats silently returning None, which is what produced
    # "task completed, no code was formulated" even when the model had actually written real
    # (possibly incomplete) code.
    if pydantic_model is CodeExecutorOutput and last_raw_content:
        salvaged_lang, salvaged_code, was_truncated = extract_code_fence(last_raw_content)
        if salvaged_code:
            print(f"[CodeMode] Salvaged a code block from malformed JSON output ({len(salvaged_code)} chars, truncated={was_truncated}).")
            try:
                explanation = (
                    "The file was large enough that the response got cut off before it finished — "
                    "what's below is everything generated up to that point, so the end of it may be "
                    "incomplete. Ask me to continue and I'll pick up where it left off."
                    if was_truncated else
                    "Recovered from a truncated response — the JSON got cut off but the code itself was intact."
                )
                return CodeExecutorOutput(
                    code=salvaged_code,
                    language=salvaged_lang or "text",
                    filename="",
                    explanation=explanation,
                    is_frontend=classify_code_target(salvaged_lang, salvaged_code),
                ), None
            except Exception as salvage_err:
                print(f"[CodeMode] Salvage construction failed: {salvage_err}")

    # Same idea for the thinking step: a long analysis (lots of reasoning about a big
    # existing file, or a wide multi-file context) can brush max_tokens and get cut off
    # mid-object. Because the prompt and temperature are the same on every retry, this
    # tends to truncate at the same point every attempt — plain retries alone rarely fix
    # it. Recovering whichever fields did complete before the cutoff beats returning
    # nothing, which used to end the whole loop immediately with zero code produced.
    # Same idea, split across the Plan and Idea stages: Plan's
    # next_step/target_file/is_multi_file, and Idea's understanding/root_cause.
    if pydantic_model is CodePlan and last_raw_content:
        step_match = re.search(r'"next_step"\s*:\s*"((?:[^"\\]|\\.)*)"', last_raw_content)
        if step_match:
            target_match = re.search(r'"target_file"\s*:\s*"((?:[^"\\]|\\.)*)"', last_raw_content)
            multi_match = re.search(r'"is_multi_file"\s*:\s*(true|false)', last_raw_content, re.IGNORECASE)
            print(f"[CodeMode] Salvaged a next step from truncated JSON output.")
            try:
                return CodePlan(
                    next_step=step_match.group(1),
                    target_file=(target_match.group(1) if target_match else ""),
                    is_multi_file=(multi_match.group(1).lower() == "true" if multi_match else False),
                ), None
            except Exception as salvage_err:
                print(f"[CodeMode] Plan salvage construction failed: {salvage_err}")

    if pydantic_model is IdeaAnalysis and last_raw_content:
        understanding_match = re.search(r'"understanding"\s*:\s*"((?:[^"\\]|\\.)*)"', last_raw_content)
        if understanding_match:
            root_cause_match = re.search(r'"root_cause"\s*:\s*"((?:[^"\\]|\\.)*)"', last_raw_content)
            print(f"[CodeMode] Salvaged an idea from truncated JSON output.")
            try:
                return IdeaAnalysis(
                    understanding=understanding_match.group(1),
                    root_cause=(root_cause_match.group(1) if root_cause_match else ""),
                ), None
            except Exception as salvage_err:
                print(f"[CodeMode] Idea salvage construction failed: {salvage_err}")

    return None, last_error


def format_code_context(state: CodeAgentState) -> str:
    msg_str = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in state.get("messages", [])])
    files = state.get("files", {})
    files_summary = ", ".join(files.keys()) if files else "None yet"

    return f"""Current Date & Time: {get_current_datetime_str()} (this is genuinely "now" — trust it over any date you might otherwise assume, and use it to judge how current the Web Search Results below are, e.g. whether a library/framework version is still the latest)

Conversation History:
{msg_str}

Current Internal State:
- Goal: {state.get('goal', 'Not set')}
- Multi-file build: {state.get('is_multi_file', False)} (~{state.get('estimated_file_count', 1)} files estimated)
- Files built so far (do not rebuild these unless fixing them): {files_summary}
- Prior thinking: {state.get('analysis_notes', 'None yet')}
- Current Step: {state.get('current_step', 'Not set')}
- Target File For This Step: {state.get('target_file', 'N/A')}
- Prior Code Draft: {state.get('code', 'None')}
- Prior Language: {state.get('language', 'None')}
- Test Notes From Last Pass: {state.get('test_notes', 'None')}
- Review Notes From Last Pass: {state.get('review_notes', 'None')}

Web Search Results (use these for current library/API/version info; ignore if 'None'):
{state.get('web_search_results') or 'None — no web search was performed for this request.'}
"""

# ----------------------------------------------------------------------
# CODE MODE: LangGraph Nodes
# ----------------------------------------------------------------------

async def idea_node(state: CodeAgentState) -> dict:
    """Stage 1 of the workflow: IDEA. Reads the request and whatever code already
    exists, and figures out what's really being asked — the way an engineer restates a
    ticket to themselves before opening an editor. For a bug report, this is also where
    the real root cause gets identified instead of the surface symptom. No code gets
    written or planned yet; this stage only builds understanding."""
    # Same free web search used in Chat Mode (see web_search()/needs_web_search()
    # near the top of the file), fired here — before any planning or code gets
    # written — so idea_node (and everything downstream of it via
    # format_code_context: plan/code/fix) can ground itself in current library
    # versions, API docs, or other external facts instead of only the model's
    # training data. Only runs when the request actually looks like it needs
    # that (e.g. "latest version of X", "docs for Y", a named recent year) —
    # most code asks skip this and pay no extra latency.
    messages = state.get("messages", [])
    latest_user_message = messages[-1].content if messages else ""
    web_search_query = ""
    web_search_results = ""
    if needs_web_search(latest_user_message):
        web_search_query = extract_search_query(latest_user_message)
        web_search_results, _unused_raw_links = await web_search(web_search_query)

    llm = get_code_llm(state["model_key"], state["temperature"])
    prompt = (
        "Before anything gets planned or written, understand this task the way an "
        "engineer reads a ticket: what is actually being asked, what (if anything) in "
        "the existing code is relevant, and — if this is a bug report — what the real "
        "root cause is, not just the symptom. If web search results are present in the "
        "context, use them to ground your understanding in current library/API/version "
        "facts rather than guessing from memory. If, having actually looked at what's "
        "already built, there's truly nothing left to do, set already_done=True instead "
        "of inventing more work.\n\n{context}"
    )

    res, err = await execute_code_llm_structured(
        llm, prompt, IdeaAnalysis,
        {"context": format_code_context({**state, "web_search_results": web_search_results})}
    )

    if not res:
        return {
            "goal": state.get("goal") or "Write the requested code.",
            "needs_fix": False,
            "last_error": err,
            "web_search_query": web_search_query,
            "web_search_results": web_search_results,
        }

    notes = res.understanding
    if res.relevant_context:
        notes += f"\n\nWhat's already there: {res.relevant_context}"
    if res.root_cause:
        notes += f"\n\nRoot cause: {res.root_cause}"
    if web_search_query:
        notes += f"\n\nSearched the web for: '{web_search_query}'"

    return {
        "goal": res.understanding,
        "analysis_notes": notes,
        "already_done": bool(res.already_done),
        "needs_fix": False,
        "last_error": err,
        "web_search_query": web_search_query,
        "web_search_results": web_search_results,
    }

async def plan_node(state: CodeAgentState) -> dict:
    """Stage 2 of the workflow: PLAN. Takes the Idea and turns it into one concrete,
    actionable next step — inspecting what's already built and, for a multi-file build,
    naming exactly which file this step targets. This is the 'propose a plan, don't
    code yet' step: nothing gets written here, only decided."""
    llm = get_code_llm(state["model_key"], state["temperature"])
    prompt = (
        "Based on the idea above, decide the single, concrete next action to take. "
        "Only mark is_multi_file=True when this genuinely needs several distinct "
        "files/pages/modules that cannot reasonably live in one runnable file — small "
        "asks, single components, or single-page tools should stay is_multi_file=False "
        "so they render as one instant preview. When it is multi-file, target_file must "
        "name the exact next file to write (e.g. 'index.html', 'dashboard.jsx'), and you "
        "must not target a file that's already listed under 'Files built so far' unless "
        "you're fixing it because of test or review notes.\n\n{context}"
    )

    res, err = await execute_code_llm_structured(llm, prompt, CodePlan, {"context": format_code_context(state)})

    iteration = state.get("iteration", 0) + 1

    if not res:
        return {
            "current_step": state.get("current_step") or "Write the code that satisfies the goal.",
            "iteration": iteration,
            "last_error": err,
        }

    is_multi_file = bool(res.is_multi_file)
    estimated_file_count = max(1, res.estimated_file_count or 1)

    # A multi-file build genuinely needs one plan/code/test/review pass per file. The
    # fixed CODE_REASONING_ITERATIONS ceiling (max 5) was sized for single-file asks —
    # once the first real estimate exists, raise the ceiling to fit it instead of
    # cutting a big multi-file app off after a handful of steps.
    max_iterations = state.get("max_iterations", 3)
    if is_multi_file:
        max_iterations = max(max_iterations, estimated_file_count + 2)

    return {
        "current_step": res.next_step,
        "target_file": (res.target_file or "").strip(),
        "is_multi_file": is_multi_file,
        "estimated_file_count": estimated_file_count,
        "iteration": iteration,
        "max_iterations": max_iterations,
        "last_error": err,
    }

# ----------------------------------------------------------------------
# CODE MODE: Incremental (diff-based) edits
#
# Root cause of "it regenerates the whole file for every tiny change": the
# executor prompt used to unconditionally say "write out the full current
# version of the code", so a one-line CSS tweak cost exactly as many output
# tokens as the original build — and that cost only grows as the file grows,
# since every single step re-emits the entire thing.
#
# On any follow-up turn where a prior draft already exists for the thing
# this step is about to touch (and the ask isn't an explicit full rewrite),
# code_node (and fix_node) below asks the model for a small SEARCH/REPLACE diff
# instead and applies it locally with plain string replacement — the model
# only pays for the lines that actually change. If the diff can't be parsed
# or a SEARCH block doesn't match verbatim, this fails closed (returns
# ok=False) so the caller falls back to the original full-file generation
# instead of silently applying a garbled edit.
# ----------------------------------------------------------------------

EDIT_BLOCK_RE = re.compile(
    r"<{5,}\s*SEARCH\s*\n(.*?)\n={5,}\s*\n(.*?)\n>{5,}\s*REPLACE",
    re.DOTALL,
)

_REWRITE_KEYWORDS = (
    "rewrite everything", "rewrite the whole", "start over", "start from scratch",
    "from scratch", "redo the whole", "redesign completely", "throw away",
    "completely different", "scrap it", "rebuild it", "new version of the whole",
)

def wants_full_rewrite(goal: str, current_step: str) -> bool:
    """Heuristic: does the request explicitly call for replacing the whole file
    rather than a targeted change? Deliberately biased toward False (i.e. toward
    the cheaper diff path) — a false negative here just costs one harmless extra
    SEARCH/REPLACE round trip if the diff attempt doesn't apply cleanly, while a
    false positive pays for a full regenerate when a diff would have done."""
    text = f"{goal} {current_step}".lower()
    return any(kw in text for kw in _REWRITE_KEYWORDS)

def apply_code_edits(original: str, edit_text: str) -> tuple[str, bool]:
    """Parses one or more SEARCH/REPLACE blocks out of `edit_text` and applies them
    to `original` in order. Returns (new_code, ok). ok is False if no blocks were
    found, an empty (and therefore ambiguous) SEARCH block was used, or any block's
    SEARCH text doesn't appear verbatim in the code as it stands at that point —
    callers should fall back to a full regenerate rather than apply a partial or
    garbled edit."""
    blocks = EDIT_BLOCK_RE.findall(edit_text)
    if not blocks:
        return original, False

    code = original
    for search, replace in blocks:
        if search == "":
            return original, False  # ambiguous — could match anywhere, refuse to guess
        if search not in code:
            return original, False
        code = code.replace(search, replace, 1)
    return code, True

def build_edit_prompt(prior_code: str, language: str, goal: str, current_step: str) -> str:
    return (
        f"Current Date & Time: {get_current_datetime_str()} — trust this as the real, current "
        "date/time for anything date-relative (e.g. whether an API/library referenced below is "
        "still current).\n\n"
        "The 'Existing Code' below already satisfies most of the goal. Make ONLY the "
        "change described by the 'Current Step' — do not rewrite anything else.\n\n"
        "Respond with one or more SEARCH/REPLACE blocks in exactly this shape, and "
        "nothing else — no full file, no markdown fences, no commentary:\n\n"
        "<<<<<<< SEARCH\n"
        "<exact existing lines to find, copied character-for-character>\n"
        "=======\n"
        "<the new lines that replace them>\n"
        ">>>>>>> REPLACE\n\n"
        "Rules:\n"
        "- Every SEARCH block must match the Existing Code EXACTLY, including whitespace.\n"
        "- Keep each SEARCH block as short as possible while still being unique in the file.\n"
        "- Use several SEARCH/REPLACE blocks for several separate changes.\n"
        "- To insert code, include a short unique anchor line in SEARCH and put that anchor "
        "plus the new lines in REPLACE.\n"
        "- To delete code, put it in SEARCH and leave REPLACE empty.\n\n"
        f"Goal: {goal}\n"
        f"Current Step: {current_step}\n"
        f"Language: {language}\n\n"
        f"Existing Code:\n{prior_code}"
    )

# code_node/fix_node push real code tokens onto this per-request queue the instant
# they're generated (see stream_code_execution below) — mirrors _current_token_queue /
# stream_plain_response used by the normal chat's executor_node higher up in this file.
# A LangGraph node function only ever receives `state`, so the per-request queue has to
# be handed in via a ContextVar instead of a normal function argument.
CODE_STREAM_DELIM = "###CODE_STREAM_START###"
_current_code_token_queue: "contextvars.ContextVar" = contextvars.ContextVar("current_code_token_queue", default=None)

async def stream_code_execution(llm: ChatNVIDIA, prompt: str, token_queue: Optional[asyncio.Queue]):
    """
    Streams the actual code generation live, token-by-token, instead of waiting on a
    single blocking structured (Pydantic/JSON) response. THIS is the fix for code only
    appearing after the model had already finished writing the whole file: JSON can't be
    parsed until its closing brace arrives, so execute_code_llm_structured — used
    everywhere else in Code Mode — is inherently all-or-nothing for the 'code' field.

    The model is asked for two short header lines (LANGUAGE / EXPLANATION), then the
    literal delimiter, then the raw code with no fence wrapper. Everything written AFTER
    the delimiter is pushed onto token_queue as ('token', piece) the instant it's
    generated — that's what lets the frontend fill the code canvas in real time, the same
    way stream_plain_response does for the chat bubble in the normal /chat flow.

    Returns (language, explanation, code) once the stream ends. If the model wraps the
    code in a ```fence``` anyway despite being told not to, the fence markers are stripped
    from the returned `code`, but note the raw fence markers will still have been streamed
    live as tokens — harmless, since the canvas just displays text as it arrives and gets
    reconciled against the final code_result event afterward regardless.
    """
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
            if CODE_STREAM_DELIM in full:
                delim_seen = True
                pending_after = full.split(CODE_STREAM_DELIM, 1)[1]
            else:
                continue
        else:
            pending_after += piece

        if not started:
            stripped = pending_after.lstrip("\r\n")
            if not stripped:
                continue  # only whitespace since the delimiter so far — keep waiting
            started = True
            if token_queue is not None:
                await token_queue.put(("token", stripped))
            pending_after = ""
            continue

        if token_queue is not None:
            await token_queue.put(("token", piece))

    header, code_part = (full.split(CODE_STREAM_DELIM, 1) if CODE_STREAM_DELIM in full else (full, ""))
    code_part = strip_thinking(code_part).strip()

    # Strip a wrapping ```lang fence if the model added one anyway.
    fence_lang, fenced_code, _truncated = extract_code_fence(code_part)
    if fenced_code:
        code_part = fenced_code

    language, explanation = fence_lang, ""
    for line in header.splitlines():
        line = line.strip()
        if line.upper().startswith("LANGUAGE:"):
            language = line.split(":", 1)[1].strip() or language
        elif line.upper().startswith("EXPLANATION:"):
            explanation = line.split(":", 1)[1].strip()

    return language, explanation, code_part

async def _write_or_edit_code(state: CodeAgentState, is_fix: bool) -> dict:
    """Shared core for the CODE and FIX stages: writes a fresh file when there's
    nothing to build on yet, or applies a targeted SEARCH/REPLACE diff when a prior
    draft already exists — the same cheap-diff-first approach used throughout Code
    Mode, so a one-line change costs a few tokens instead of a full-file regenerate.

    When is_fix=True (the FIX stage), the loop is here because Test or Review found a
    concrete, named problem — the prompt is built around fixing THAT issue rather than
    re-executing the current step from scratch, and the diff path is preferred even more
    strongly since a fix should repair the file, not rewrite it."""
    llm = get_code_llm(state["model_key"], state["temperature"])
    is_multi = bool(state.get("is_multi_file"))
    target_file = state.get("target_file", "")
    token_queue = _current_code_token_queue.get()

    # What (if anything) already exists for the thing this step is about to touch —
    # single-file: the one prior draft; multi-file: whatever this target_file already
    # holds (empty if it hasn't been written yet, which naturally falls through to a
    # full generate below, same as before).
    if is_multi:
        prior_code_for_step = state.get("files", {}).get(target_file, "")
        prior_language_for_step = state.get("file_languages", {}).get(target_file, "")
    else:
        prior_code_for_step = state.get("code", "")
        prior_language_for_step = state.get("language", "")

    issue_text = ""
    if is_fix:
        parts = []
        if not state.get("test_passed", True) and state.get("test_notes"):
            parts.append(f"Test found: {state.get('test_notes')}")
        if state.get("needs_fix") and state.get("review_notes"):
            parts.append(f"Review found: {state.get('review_notes')}")
        issue_text = "\n".join(parts) or "Address the problem noted in the test/review notes above."

    # A FIX always edits the existing code rather than rewriting from scratch, as long
    # as there IS existing code to edit — a fix should never regenerate the whole file.
    edit_mode = bool(prior_code_for_step.strip()) and (
        is_fix or not wants_full_rewrite(state.get("goal", ""), state.get("current_step", ""))
    )

    language, explanation, code, err = "", "", "", None

    if edit_mode:
        # Cheap path: ask for only the lines that change, apply them locally instead
        # of paying to have the model re-emit the whole file. Deliberately NOT pushed
        # through token_queue — the diff markup itself should never appear live in the
        # canvas. If this succeeds, `code` below is already the full reconstructed
        # file, and generate_code_stream's post-loop chunk-replay (streamed_live stays
        # False for this turn) fills the canvas from it exactly like the existing
        # non-streaming fallback path already does.
        try:
            edit_prompt = build_edit_prompt(
                prior_code_for_step, prior_language_for_step,
                state.get("goal", ""),
                f"Fix this: {issue_text}" if is_fix else state.get("current_step", "")
            )
            res = await llm.ainvoke([HumanMessage(content=edit_prompt)])
            edit_text = strip_thinking(res.content).strip()
            new_code, ok = apply_code_edits(prior_code_for_step, edit_text)
            if ok:
                code = new_code
                language = prior_language_for_step
                explanation = "Applied a targeted fix to the existing code." if is_fix else "Applied a targeted edit to the existing code."
        except Exception as e:
            print(f"[CodeMode] edit-mode generation failed, falling back to full generate: {e}")
        # ok == False (or an exception) falls through to the full-generate path below
        # instead of returning nothing or a garbled partial edit.

    if not code:
        # First draft for this file (no prior code to edit), an explicit full-rewrite
        # ask, or the diff attempt above didn't land — fall back to the original
        # "write the whole file" path.
        multi_note = (
            f" This is a multi-file build: this step must produce exactly the file "
            f"'{target_file}' — write ONLY that file's code, not the whole app."
            if is_multi and target_file else ""
        )
        step_desc = f"Fix this problem: {issue_text}" if is_fix else "Execute the 'Current Step' to satisfy the coding 'Goal'."
        prompt = (
            f"{step_desc} Write complete, runnable code. IMPORTANT: you must never leave "
            "the code empty for a coding task — even if this step is just a review, a "
            "small tweak, or you believe the goal is already met, write out the full "
            "current version of the code (unchanged if nothing needed to change), not "
            f"just an explanation with no code.{multi_note}\n\n"
            f"{format_code_context(state)}\n\n"
            "Respond in exactly this shape, nothing else:\n"
            "LANGUAGE: <the programming language, e.g. python, javascript, html, jsx>\n"
            "EXPLANATION: <one short sentence on what this code does>\n"
            f"{CODE_STREAM_DELIM}\n"
            "<the raw code only — no markdown fence, no commentary before or after it>"
        )

        if token_queue is not None:
            await token_queue.put(("code_executor_start", target_file if is_multi else None))

        try:
            language, explanation, code = await stream_code_execution(llm, prompt, token_queue)
            err = None
        except Exception as e:
            print(f"[CodeMode] code generation streaming failed: {e}")
            language, explanation, code, err = "", "", "", f"{type(e).__name__}: {e}"

    prior_code = state.get("code", "")
    # Accumulate into the files dict instead of overwriting a single code field — this is
    # what stops one truncated/failed step from wiping out every file built before it.
    files = dict(state.get("files", {}))
    file_languages = dict(state.get("file_languages", {}))

    if code or not err:
        try:
            is_frontend = classify_code_target(language, code) if code else False
        except Exception as e:
            print(f"[CodeMode] classify_code_target failed: {e}")
            is_frontend = False

        if not explanation.strip():
            explanation = "Applied the fix." if is_fix else "Wrote the code for this step."

        # Guard -1: the model decided no new code was needed and left 'code' blank.
        # That should never happen on a genuine coding task when a working prior file
        # already exists — carry the prior file forward instead of resolving to
        # "explanation only, no code", which is exactly what showed up as
        # "task completed, no code was formulated". (Only applies to single-file mode —
        # in multi-file mode an empty step just means that one file didn't land, and the
        # files already in the dict are untouched.)
        if not code and prior_code and not is_multi:
            code = prior_code
            language = language or state.get("language", "")
            is_frontend = is_frontend or bool(state.get("is_frontend", False))
            explanation = "No changes were needed for this step — carrying the existing code forward."

        if is_multi and code:
            fname = (target_file or f"file_{len(files) + 1}.{language or 'txt'}").strip()
            files[fname] = code
            file_languages[fname] = language
    else:
        language, code, is_frontend = "", "", False
        # Surface the real reason instead of a generic canned message, so a bad
        # API key / rate limit / model error is visible instead of looking like
        # a random glitch every time.
        explanation = f"I hit an issue generating code for this step: {err}"
        # Same carry-forward here: an LLM/streaming failure shouldn't wipe out a
        # perfectly good previous file either (single-file mode only — multi-file mode
        # already preserves everything via the accumulating `files` dict above).
        if prior_code and not is_multi:
            code = prior_code
            language = state.get("language", "")
            is_frontend = bool(state.get("is_frontend", False))

    if is_multi:
        # Keep the per-step chat response short — the full multi-file bundle gets
        # assembled once at the end (see generate_code_stream / _handle_code_chat),
        # not re-emitted as a giant fenced block on every single intermediate step.
        response_text = explanation.strip() or f"Built {target_file or 'a file'}."
    else:
        # Only emit a fenced code block when there IS code. An empty ```{lang}\n\n```
        # fence still gets parsed as a real (empty) code block by the frontend's
        # markdown renderer, and highlight.js then auto-detects a language on the
        # empty content and labels it "undefined" — that's the stray "undefined"
        # line that shows up after a failed generation.
        response_text = f"{explanation}\n\n```{language}\n{code}\n```".strip() if code else explanation

    return {
        "code": code,
        "language": language,
        "is_frontend": is_frontend,
        "explanation": explanation,
        "response": response_text,
        "files": files,
        "file_languages": file_languages,
        "last_error": err
    }

async def code_node(state: CodeAgentState) -> dict:
    """Stage 3 of the workflow: CODE. Writes the code for the step the Plan just
    decided — a fresh file when there's nothing to build on yet, a targeted diff when
    a prior draft already exists."""
    return await _write_or_edit_code(state, is_fix=False)

async def fix_node(state: CodeAgentState) -> dict:
    """Stage 6 of the workflow: FIX. Only reached when TEST or REVIEW found a concrete,
    named problem — applies a targeted repair for that specific issue (never a full
    rewrite) and clears needs_fix so the loop re-verifies via TEST again instead of
    assuming the fix actually worked."""
    result = await _write_or_edit_code(state, is_fix=True)
    result["needs_fix"] = False
    return result

async def test_node(state: CodeAgentState) -> dict:
    """Stage 4 of the workflow: TEST. There's no live execution sandbox here, so this
    stage does two things instead: a real static syntax check for languages that support
    one cheaply (Python compiles cleanly, JSON parses cleanly), and — for anything that
    passes that check — an honest LLM pass that reasons through the concrete unit tests,
    integration paths, and edge cases this code should handle, and reports whether it
    would actually pass them. A concrete syntax error always wins over the model's
    opinion; it never even gets asked."""
    is_multi = bool(state.get("is_multi_file"))
    target_file = state.get("target_file", "")
    if is_multi:
        code = state.get("files", {}).get(target_file, "")
        language = state.get("file_languages", {}).get(target_file, "")
    else:
        code = state.get("code", "")
        language = state.get("language", "")

    if not (code or "").strip():
        return {"test_notes": "No code to test yet.", "test_passed": True, "needs_fix": False}

    lang = (language or "").strip().lower()
    syntax_error = None
    if lang == "python":
        try:
            compile(code, target_file or "<generated>", "exec")
        except SyntaxError as e:
            syntax_error = f"Python syntax error: {e.msg} (line {e.lineno})"
        except Exception:
            pass  # not a clean syntax failure — let the reasoning pass below judge it instead
    elif lang == "json":
        try:
            json.loads(code)
        except json.JSONDecodeError as e:
            syntax_error = f"Invalid JSON: {e}"
        except Exception:
            pass

    if syntax_error:
        return {"test_notes": syntax_error, "test_passed": False, "needs_fix": True}

    llm = get_code_llm(state["model_key"], state["temperature"])
    prompt = (
        "Test the code you just wrote against the goal and the step it was supposed to "
        "satisfy, the way you'd actually run it: think through the concrete unit tests, "
        "integration paths, and edge cases this should handle, and check whether the "
        "code as written would actually pass them. Be honest — if it would genuinely "
        "pass, say so rather than inventing problems.\n\n{context}"
    )
    res, err = await execute_code_llm_structured(llm, prompt, TestReport, {"context": format_code_context(state)})

    if not res:
        # Couldn't get a reliable test pass — don't block the loop on it, treat it as
        # passing rather than looping forever against a backend that's already failing.
        return {
            "test_notes": f"Skipped testing — hit an error: {err}" if err else "No issues found.",
            "test_passed": True,
            "needs_fix": False,
            "last_error": err,
        }

    return {
        "test_notes": res.test_notes or ("Would pass." if res.passed else "Would fail."),
        "test_passed": bool(res.passed),
        "needs_fix": (not res.passed) and bool(res.issues.strip()),
        "last_error": err,
    }

async def review_node(state: CodeAgentState) -> dict:
    """Stage 5 of the workflow: REVIEW. Looks over the diff the way you'd review your
    own pull request before calling it done — real bugs, missed requirements, and broken
    edge cases, not style nitpicks. A failing TEST result upstream already means the
    loop needs a FIX, so a clean review here never silently clears that on its own."""
    llm = get_code_llm(state["model_key"], state["temperature"])
    prompt = (
        "Review the code you just wrote (the 'Prior Code Draft') against the goal and "
        "the step it was supposed to satisfy — the way you'd look over your own diff "
        "before calling it done. Look specifically for real bugs, missed requirements, "
        "and broken edge cases, not just style nitpicks. Take the 'Test Notes From Last "
        "Pass' into account too. Be honest: if it's genuinely correct, say so rather "
        "than inventing issues.\n\n{context}"
    )

    res, err = await execute_code_llm_structured(llm, prompt, CodeReview, {"context": format_code_context(state)})
    test_failed = not bool(state.get("test_passed", True))

    if not res:
        # Couldn't get a reliable review — don't block the loop on it, but a test that
        # already failed still needs a fix regardless.
        return {
            "review_notes": f"Skipped review — hit an error: {err}" if err else "Looks solid.",
            "needs_fix": test_failed,
            "completion_score": 70 if test_failed else 100,
            "last_error": err,
        }

    if res.looks_correct:
        review_str = res.notes or "Looks correct."
    else:
        review_str = f"Found an issue: {res.issues}" if res.issues.strip() else "Found an issue that needs another pass."

    review_needs_fix = (not res.looks_correct) and bool(res.issues.strip())
    needs_fix = review_needs_fix or test_failed

    return {
        "review_notes": review_str,
        "needs_fix": needs_fix,
        "completion_score": 100 if (res.looks_correct and not test_failed) else 70,
        "last_error": err,
    }

async def commit_node(state: CodeAgentState) -> dict:
    """Stage 7 of the workflow: COMMIT. Runs once, after Review (and any Fix/Test
    re-checks) have settled and there's nothing left to fix — writes a short,
    conventional-commit-style summary of the change and finalizes the chat response.
    Never blocks the loop: if the summary call fails, a sensible default is used
    instead. Nothing here touches the code itself."""
    commit_message = "chore: apply requested change"
    goal = state.get("goal", "")
    step = state.get("current_step", "")
    if goal or step:
        try:
            llm = get_code_llm(state["model_key"], state["temperature"])
            prompt = (
                f"Goal: {goal}\nWhat changed: {step}\n\n"
                "Write ONE line, conventional-commit style (e.g. 'feat: add paginated "
                "user search endpoint', 'fix: correct off-by-one in pagination loop'). "
                "Reply with only that line, nothing else."
            )
            res = await llm.ainvoke([HumanMessage(content=prompt)])
            candidate = strip_thinking(res.content).strip().splitlines()[0].strip().strip('"')
            if candidate:
                commit_message = candidate
        except Exception as e:
            print(f"[CodeMode] commit_node message generation failed (using default): {e}")

    response = state.get("response", "")
    explanation = state.get("explanation", "")
    code = state.get("code", "")
    language = state.get("language", "")
    is_multi = bool(state.get("is_multi_file"))

    if not is_multi and code:
        response = f"{explanation}\n\nCommit: {commit_message}\n\n```{language}\n{code}\n```".strip()

    return {
        "commit_message": commit_message,
        "response": response,
        "completion_score": 100,
    }

def post_review_edge(state: CodeAgentState) -> str:
    """Decides what happens after REVIEW: another pass through FIX if Test or Review
    found a concrete, real issue; the next file's IDEA if this is a multi-file build
    with more left to do; otherwise straight on to COMMIT.

    No artificial completion-percentage evaluator: for multi-file builds this trusts the
    idea step's own judgment on whether more files are actually needed (already_done)
    — the same way an engineer decides "anything else left?" by looking at what's built,
    not by ticking off a plan written before any code existed.

    Still keeps the original safety net: as long as iterations remain, the loop is never
    allowed to reach COMMIT having produced zero code.
    """
    out_of_iterations = state.get("iteration", 0) >= state.get("max_iterations", 3)

    if state.get("needs_fix") and not out_of_iterations:
        return "fix"

    if bool(state.get("is_multi_file")):
        files_done = len(state.get("files", {}))
        if out_of_iterations:
            return "commit"
        if state.get("already_done") and files_done > 0:
            return "commit"
        return "idea"

    has_code = bool((state.get("code") or "").strip())
    if not has_code and not out_of_iterations:
        return "idea"
    return "commit"

# ----------------------------------------------------------------------
# CODE MODE: Workflow Graph (separate compiled graph from the normal app_graph)
#
# The Claude AI coding workflow, wired as the actual execution graph:
#
#   Idea -> Plan -> Code -> Test -> Review -> Fix -> Commit
#
#   1. Idea    — understand what's actually being asked (and the root cause, for bugs)
#   2. Plan    — inspect what exists, propose the single concrete next step
#   3. Code    — implement that one step (full write or targeted diff)
#   4. Test    — static syntax check + reasoned test/edge-case pass
#   5. Review  — look over the diff for real bugs before calling it done
#   6. Fix     — only reached when Test or Review found a concrete issue; loops back
#                through Test to re-verify rather than trusting itself
#   7. Commit  — runs once, at the very end: short commit-style summary, done
#
# A multi-file build sends Review back to Idea for the next file instead of Commit;
# Commit itself only ever runs once, after everything is actually settled.
# ----------------------------------------------------------------------

code_workflow = StateGraph(CodeAgentState)

code_workflow.add_node("idea", idea_node)
code_workflow.add_node("plan", plan_node)
code_workflow.add_node("code", code_node)
code_workflow.add_node("test", test_node)
code_workflow.add_node("review", review_node)
code_workflow.add_node("fix", fix_node)
code_workflow.add_node("commit", commit_node)

code_workflow.set_entry_point("idea")
code_workflow.add_edge("idea", "plan")
code_workflow.add_edge("plan", "code")
code_workflow.add_edge("code", "test")
code_workflow.add_edge("test", "review")
code_workflow.add_conditional_edges("review", post_review_edge)
code_workflow.add_edge("fix", "test")
code_workflow.add_edge("commit", END)

code_app_graph = code_workflow.compile()

# ----------------------------------------------------------------------
# CODE MODE: Direct-answer fallback (used for tiny messages, timeouts, errors)
# ----------------------------------------------------------------------

CODE_FENCE_RE = re.compile(r"```([\w+#.-]*)\n([\s\S]*?)```")

# A response that got cut off by max_tokens (very plausible for something like a full,
# feature-heavy SaaS app crammed into one file) ends with an OPENING fence but never
# reaches a closing one — CODE_FENCE_RE alone can't match that, so a genuinely large
# build that ran out of room came back with nothing recovered at all, even though the
# model had already written a large amount of real, usable code before hitting the
# limit. This second pattern matches "from the last opening fence to the end of the
# text" so that code is recovered too, instead of being thrown away.
CODE_FENCE_OPEN_RE = re.compile(r"```([\w+#.-]*)\n([\s\S]*)$")

def extract_code_fence(text: str) -> tuple[str, str, bool]:
    """Returns (language, code, was_truncated). Tries a properly closed fence first;
    if none exists, falls back to an unclosed trailing fence so a max_tokens cutoff
    still recovers whatever code the model managed to write before running out of
    room, instead of surfacing as 'no code was formulated'."""
    text = text or ""
    match = CODE_FENCE_RE.search(text)
    if match:
        return (match.group(1) or "").strip(), match.group(2).strip(), False
    match = CODE_FENCE_OPEN_RE.search(text)
    if match:
        return (match.group(1) or "").strip(), match.group(2).strip(), True
    return "", "", False

def extract_code_from_answer(answer: str) -> tuple[str, str]:
    """Pulls the first fenced code block out of a plain-text LLM answer, if any.
    Needed because answer_code_directly (used for timeouts/errors/simple messages)
    only ever produced free-text before — the code was inside that text but never
    split out into its own field, so the frontend Canvas had nothing to render."""
    language, code, _was_truncated = extract_code_fence(answer)
    return language, code

async def answer_code_directly(message: str, history: List[BaseMessage], model_key: str, temperature: float) -> dict:
    """Always returns a real code answer — falls back to the other Code Mode model if the primary one fails."""
    base_system = CODE_SYSTEM_PROMPT if CODE_SYSTEM_PROMPT.strip() else "You are an elite, Claude-level coding assistant. Plan briefly, ground your answer in what you actually know rather than guessing at APIs, then respond with correct, complete code and a concise explanation."
    # Same reasoning as execute_code_llm_structured above: stamp "now" on fresh,
    # per-call, since the static CODE_SYSTEM_PROMPT constant can't carry a live date.
    base_system = f"Current Date & Time: {get_current_datetime_str()} — this is the real, current date/time; trust it completely rather than assuming a date from your own training.\n\n" + base_system
    messages = [SystemMessage(content=base_system)]
    messages.extend(history[-6:])
    messages.append(HumanMessage(content=message))

    fallback_key = "glm" if (model_key or "").strip().lower() != "glm" else "kimi"  # covers "kimi" and "kimik2.6" alike

    for key in (model_key, fallback_key):
        try:
            llm = get_code_llm(key, temperature)
            res = await llm.ainvoke(messages)
            answer = strip_thinking(res.content).strip()
            if answer:
                language, code = extract_code_from_answer(answer)
                is_frontend = classify_code_target(language, code) if code else False
                return {"text": answer, "code": code, "language": language, "is_frontend": is_frontend}
        except Exception as e:
            print(f"[CodeMode] Model '{key}' failed: {e}")

    return {"text": "I'm having trouble reaching the code model right now. Please try again in a moment.", "code": "", "language": "", "is_frontend": False}

def code_graph_timeout_seconds(model_key: str, max_iterations: int) -> float:
    """Sizes the overall graph timeout the same way the normal chat section does, but keyed off
    the two Code Mode models: kimi (high-end reasoning) gets a larger per-call budget than glm (fast).

    Previously capped at a flat 240s no matter what max_iterations was — worst_case_calls blows
    past that almost immediately even at "medium", so every run was effectively capped at 240s
    regardless of how many steps were actually planned. That's what silently truncated big
    multi-file builds partway through. The ceiling now actually scales with iteration count (up
    to a generous absolute limit); the heartbeat pings in run_code_graph_streaming keep the SSE
    connection alive for the platform proxy the whole time, so a longer run is safe to allow."""
    per_call_seconds = 140.0 if (model_key or "").strip().lower() in ("kimi", "kimik2.6") else 90.0
    # The workflow now has up to 6 LLM-calling stages per iteration (idea, plan, code,
    # test, review, fix) instead of the original 3 (analysis, executor, review) — the
    # per-iteration multiplier below was raised from 4 to 7 to match, so a multi-step
    # build still gets a realistic time budget instead of being cut off mid-run.
    worst_case_calls = 1 + (max_iterations * 7)
    return min(worst_case_calls * per_call_seconds, 1500.0)

CODE_NODE_LABELS = {
    "idea": "Understanding the idea",
    "plan": "Planning the next step",
    "code": "Writing the code",
    "test": "Testing the change",
    "review": "Reviewing the change",
    "fix": "Fixing the issue",
    "commit": "Committing the change",
}

def code_node_detail(node_name: str, state: dict) -> str:
    if node_name == "idea":
        err = state.get("last_error")
        if err:
            return f"Thinking hit an error: {err}"
        notes = state.get("analysis_notes", "")
        return notes if notes else "Reading through the request and the existing code."
    if node_name == "plan":
        step = state.get("current_step", "")
        target = state.get("target_file", "")
        if state.get("is_multi_file"):
            files_done = len(state.get("files", {}))
            progress = f"({files_done} file{'s' if files_done != 1 else ''} built so far)"
            label = f"{target or 'the next file'} {progress}"
            return f"Planning {label}: {step}" if step else f"Planning the build {progress}."
        return step if step else "Deciding the next concrete step."
    if node_name == "code":
        explanation = state.get("explanation", "")
        if state.get("is_multi_file"):
            target = state.get("target_file", "")
            base = f"Writing {target}" if target else "Writing the next file"
            return f"{base} — {explanation}" if explanation else f"{base}."
        return explanation if explanation else "Writing the code."
    if node_name == "test":
        notes = state.get("test_notes", "")
        return notes if notes else "Checking the code against likely tests and edge cases."
    if node_name == "review":
        notes = state.get("review_notes", "")
        return notes if notes else "Reviewing the change for bugs and correctness."
    if node_name == "fix":
        explanation = state.get("explanation", "")
        return explanation if explanation else "Applying a fix for the issue that was found."
    if node_name == "commit":
        msg = state.get("commit_message", "")
        return f"Committed: {msg}" if msg else "Finalizing the change."
    return CODE_NODE_LABELS.get(node_name, node_name)

async def run_code_graph_streaming(initial_state: dict, timeout: float, token_queue: asyncio.Queue):
    """
    Runs the Code Mode LangGraph node-by-node AND, concurrently, relays whatever
    code_node/fix_node push onto token_queue as they write code — interleaving both into
    one chronological stream of events:
      ("status", node_name, state_so_far) — a node just finished (real backend progress)
      ("code_reset", filename)            — code_node/fix_node is about to start writing a
                                             file live; filename is None for single-file
      ("token", text)                     — a real, live delta of code being written,
                                             straight from the model, the instant it arrives
      ("heartbeat", None)                 — nothing new yet, just keeping the SSE connection
                                             alive during a long non-streaming node call
                                             (idea / plan / test / review / fix / commit)
    Still enforces the same overall timeout budget, and still emits the same ~12s
    heartbeats as before so a slow non-streaming node (these can take 60-140s) doesn't
    leave the SSE connection silent long enough for a hosting proxy to kill it as idle.
    """
    state_acc = dict(initial_state)
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    agen = code_app_graph.astream(initial_state, stream_mode="updates")
    HEARTBEAT_INTERVAL = 12.0

    graph_next = asyncio.ensure_future(agen.__anext__())
    queue_next = asyncio.ensure_future(token_queue.get())

    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            wait_chunk = min(HEARTBEAT_INTERVAL, remaining)

            done, _ = await asyncio.wait(
                {graph_next, queue_next}, timeout=wait_chunk, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:
                yield ("heartbeat", None)
                continue

            if queue_next in done:
                kind, payload = queue_next.result()
                if kind == "code_executor_start":
                    yield ("code_reset", payload)
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
    except StopAsyncIteration:
        return
    finally:
        for t in (graph_next, queue_next):
            t.cancel()
        for t in (graph_next, queue_next):
            try:
                await t
            except (asyncio.CancelledError, StopAsyncIteration, Exception):
                pass
        await agen.aclose()

async def stream_code_thinking(model_key: str, temperature: float, message: str, history: List[BaseMessage]):
    """Streams a genuine reasoning trace token-by-token before the coding loop starts — the
    Claude.ai-style 'Thinking' panel. This is a plain free-text call (no Pydantic parser), with
    thinking turned ON, so the entire token budget goes toward real reasoning instead of
    competing with a JSON schema the way it would inside execute_code_llm_structured (that's the
    documented reason thinking is forced OFF for every structured call — see CODE_THINKING_OFF_KWARGS
    above). Kept short (max_tokens=4096) and on its own model instance so it never eats into the
    32K budget reserved for the actual code-generating calls."""
    model_name = CODE_MODEL_MAP.get((model_key or "").strip().lower(), CODE_MODEL_MAP["kimi"])
    thinking_llm = ChatNVIDIA(
        model=model_name,
        temperature=temperature,
        max_tokens=4096,
        timeout=60,
        model_kwargs={"chat_template_kwargs": {"thinking": True, "enable_thinking": True}},
    )
    context = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in history[-6:]])
    prompt = (
        f"Current Date & Time: {get_current_datetime_str()}\n\n"
        f"{context}\n\nUser's new request: {message}\n\n"
        "Think through how you'd build this: what it actually needs, whether it needs one file "
        "or several, the technical approach, and any risks or tricky parts. Reason it through out "
        "loud, in your own words — don't write final code yet."
    )
    try:
        async for chunk in thinking_llm.astream(prompt):
            piece = getattr(chunk, "content", "") or ""
            if piece:
                yield piece
    except Exception as e:
        print(f"[CodeMode] Thinking stream failed (non-fatal, coding loop still runs): {e}")

def iter_code_chunks(text: str, target_chunks: int = 60, min_chunk: int = 24, max_chunk: int = 800):
    """Splits an already-generated code string into pieces for progressive delivery to the
    code canvas — a Claude/Gemini-Canvas-style 'typewriter' fill instead of the whole file
    landing in the box in one blob. `target_chunks` keeps the total number of SSE events (and
    the added latency from the small sleep between them) roughly constant regardless of file
    size: a 2KB file and a 200KB file both stream in about the same number of steps, just with
    bigger pieces for the bigger file, so this never meaningfully slows down delivery."""
    if not text:
        return
    chunk_size = max(min_chunk, min(max_chunk, (len(text) // target_chunks) + 1))
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]

async def generate_code_stream(request: "CodeChatRequest", session: dict, session_id: str):
    """Streams SSE events for Code Mode: status updates while working, then the final
    answer word-by-word, followed by one 'code_result' event carrying the code/language/
    show_preview fields so the frontend can decide whether to render a live preview."""
    max_iterations = get_code_max_iterations(request.reasoning_level)
    is_multi = False
    files: Dict[str, str] = {}
    # True once code has already been streamed live to the frontend via code_reset/token
    # events (see the non-simple branch below) — stays False for the simple/direct path
    # and for any non-streaming fallback call, which still need the old chunked replay.
    streamed_live = False

    if is_simple_code_message(request.message):
        yield f"data: {json.dumps({'type': 'status', 'step': 'direct', 'label': 'Answering', 'detail': 'Short message — answering directly without the full coding loop.'})}\n\n"
        result = await answer_code_directly(request.message, session["messages"], request.model, request.temperature)
        final_response, code, language, is_frontend = result["text"], result["code"], result["language"], result["is_frontend"]
        final_state = {"completion_score": 100, "iteration": 1}
    else:
        # Real Claude.ai-style thinking: stream the model's actual reasoning token-by-token
        # BEFORE the structured plan/execute loop starts, as its own SSE event type. The
        # frontend renders these `thinking` deltas into a collapsible panel and, on
        # `thinking_done`, headers it "Thought for {elapsed}s" — same shape as Claude.ai.
        yield f"data: {json.dumps({'type': 'thinking_start'})}\n\n"
        thinking_start = asyncio.get_event_loop().time()
        thinking_text = ""
        async for piece in stream_code_thinking(request.model, request.temperature, request.message, session["messages"]):
            thinking_text += piece
            yield f"data: {json.dumps({'type': 'thinking', 'delta': piece})}\n\n"
        thinking_elapsed = round(asyncio.get_event_loop().time() - thinking_start, 1)
        yield f"data: {json.dumps({'type': 'thinking_done', 'elapsed': thinking_elapsed})}\n\n"

        initial_state = {
            "messages": session["messages"],
            "model_key": request.model,
            "temperature": request.temperature,
            "max_iterations": max_iterations,
            "iteration": 0,
            "completion_score": 0,
            "completed_steps": [],
            "plan": [],
            "target_file": "",
            "is_multi_file": False,
            "estimated_file_count": 1,
            # Seed the real previous code/language (and any previously accumulated
            # multi-file build) so follow-up turns amend the actual prior work instead of
            # relying only on the plain-text chat history — this is what was causing
            # "task completed, no code" replies on a second ask.
            "code": session.get("last_code", ""),
            "language": session.get("last_language", ""),
            "files": dict(session.get("last_files", {})),
            "file_languages": dict(session.get("last_file_languages", {}))
        }
        timeout = code_graph_timeout_seconds(request.model, max_iterations)
        final_state = initial_state

        # streamed_live tracks whether the code currently in `code`/`files` already went
        # out to the frontend in real time via code_reset/token events below (the actual
        # live-generation fix). It's set back to False whenever code instead came from a
        # non-streaming fallback call (answer_code_directly), so the trailing "stream code
        # into the canvas" section further down knows it still needs the old chunked
        # typewriter replay for THOSE cases — there's nothing to relay live for a call
        # that already ran to completion with .ainvoke().
        streamed_live = False
        active_filename = None  # tracks which file is currently being live-typed, for multi-file

        code_token_queue: asyncio.Queue = asyncio.Queue()
        qtoken = _current_code_token_queue.set(code_token_queue)
        try:
            async for evt in run_code_graph_streaming(initial_state, timeout, code_token_queue):
                kind = evt[0]

                if kind == "heartbeat":
                    yield ": keep-alive\n\n"  # SSE comment — keeps the connection alive, frontend ignores it
                    continue

                if kind == "code_reset":
                    # code_node/fix_node is about to start writing a file live — open the
                    # canvas now so the tokens that follow have somewhere to land.
                    active_filename = evt[1]
                    streamed_live = True
                    if active_filename:
                        yield f"data: {json.dumps({'type': 'code_file_start', 'filename': active_filename, 'language': '', 'session_id': session_id})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'code_start', 'language': '', 'session_id': session_id})}\n\n"
                    continue

                if kind == "token":
                    delta = evt[1]
                    if active_filename:
                        yield f"data: {json.dumps({'type': 'code_delta', 'filename': active_filename, 'delta': delta, 'session_id': session_id})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'code_delta', 'delta': delta, 'session_id': session_id})}\n\n"
                    continue

                # kind == "status"
                _, node_name, state_so_far = evt
                label = CODE_NODE_LABELS.get(node_name, node_name)
                detail = code_node_detail(node_name, state_so_far)
                yield f"data: {json.dumps({'type': 'status', 'step': node_name, 'label': label, 'detail': detail})}\n\n"
                final_state = state_so_far
            if "response" not in final_state:
                print(f"[{session_id}] Code graph finished with no 'response' key. final_state keys: {list(final_state.keys())}")
            final_response = final_state.get("response", "Task completed but no code was formulated.")
            code = final_state.get("code", "")
            language = final_state.get("language", "")
            is_frontend = final_state.get("is_frontend", False)
            is_multi = bool(final_state.get("is_multi_file"))
            files = final_state.get("files", {}) if is_multi else {}

            # Hard safety net: post_review_edge now avoids reaching commit/END before
            # code exists, but if iterations still ran out with none (e.g. reasoning_level
            # "low" giving very few passes), don't hand back an explanation with nothing
            # to show — make one guaranteed direct attempt at real code instead.
            has_any_output = bool(files) if is_multi else bool(code.strip())
            if not has_any_output:
                print(f"[{session_id}] Code graph ended with no output — falling back to direct generation.")
                yield f"data: {json.dumps({'type': 'status', 'step': 'fallback', 'label': random.choice(FALLBACK_LABELS), 'detail': 'The build hit its output limit before any usable code came back (common on large, feature-heavy requests) — making one more direct attempt.'})}\n\n"
                fallback = await answer_code_directly(request.message, session["messages"], request.model, request.temperature)
                if fallback["code"].strip():
                    final_response, code, language, is_frontend = fallback["text"], fallback["code"], fallback["language"], fallback["is_frontend"]
                    is_multi, files = False, {}
                    streamed_live = False  # this code came from a fresh, non-streamed call
        except asyncio.TimeoutError:
            print(f"[{session_id}] Code Mode streaming timed out after {timeout:.0f}s. Falling back to direct answer.")
            yield f"data: {json.dumps({'type': 'status', 'step': 'fallback', 'label': random.choice(FALLBACK_LABELS), 'detail': 'The coding loop was taking too long, so falling back to a direct answer.'})}\n\n"
            result = await answer_code_directly(request.message, session["messages"], request.model, request.temperature)
            final_response, code, language, is_frontend = result["text"], result["code"], result["language"], result["is_frontend"]
            final_state = {"completion_score": 100, "iteration": 1}
            is_multi, files = False, {}
            streamed_live = False
        except Exception as e:
            print(f"[{session_id}] Code Mode streaming failed: {e}. Falling back to direct answer.")
            yield f"data: {json.dumps({'type': 'status', 'step': 'fallback', 'label': random.choice(FALLBACK_LABELS), 'detail': 'Something went wrong in the coding loop, so falling back to a direct answer.'})}\n\n"
            result = await answer_code_directly(request.message, session["messages"], request.model, request.temperature)
            final_response, code, language, is_frontend = result["text"], result["code"], result["language"], result["is_frontend"]
            final_state = {"completion_score": 100, "iteration": 1}
            is_multi, files = False, {}
            streamed_live = False
        finally:
            _current_code_token_queue.reset(qtoken)

    if is_multi and files:
        # Big multi-file build: assemble every accumulated file into the final answer as
        # its own fenced block instead of forcing everything into one bundled HTML file.
        file_languages = final_state.get("file_languages", {})
        intro = final_state.get("goal", "") and f"Built {len(files)} files for: {final_state.get('goal')}"
        parts = [intro] if intro else [f"Built {len(files)} files."]
        for fname, fcode in files.items():
            flang = file_languages.get(fname, "")
            parts.append(f"**{fname}**\n```{flang}\n{fcode}\n```")
        final_response = "\n\n".join(p for p in parts if p and p.strip())
        code, language = "", ""
        is_frontend = any(f.lower().endswith((".html", ".htm")) for f in files)
    else:
        # Single-file path (unchanged): enforce single-file bundling ONCE here, on the
        # final settled answer — not on every internal loop iteration (see
        # enforce_single_file_frontend's docstring for why).
        goal_hint = final_state.get("goal", "") or request.message
        prior_code_for_guard = session.get("last_code", "")
        code_before_bundling = code
        new_code, new_language, new_is_frontend = await enforce_single_file_frontend(
            code, language, is_frontend, goal_hint, prior_code_for_guard, request.model, request.temperature
        )
        if new_code != code:
            final_response = rebuild_response_with_code(final_response, new_code, new_language)
            code, language, is_frontend = new_code, new_language, new_is_frontend
            # Bundling rewrote what was already streamed live (e.g. wrapped bare JSX into
            # a full HTML file) — the canvas is now showing stale content, so this one
            # case still needs a re-stream of the corrected code below.
            if code_before_bundling != code and streamed_live:
                streamed_live = False

    # Hard guarantee: no matter what path got here, never let an empty string reach the
    # frontend — that's what silently produced a blank bubble after the thinking animation
    # ended (the animation still resolves to "Thought for Ns" either way, so an empty
    # answer looked exactly like the whole thing just doing nothing).
    if not (final_response or "").strip():
        final_response = "I wasn't able to generate a response for that — please try again."

    # Remember this turn's output so the NEXT turn can seed it back in (only overwrite
    # when we actually got new output — a turn that produced none shouldn't wipe out
    # perfectly good previous work).
    if is_multi and files:
        session["last_files"], session["last_file_languages"] = files, final_state.get("file_languages", {})
    elif code:
        session["last_code"], session["last_language"] = code, language

    session["messages"].append(AIMessage(content=final_response))

    # Only stream the prose/explanation into the chat bubble. `final_response` still
    # has the full fenced code glued on the end (see code_node/commit_node), but that
    # code is delivered once, in full, via the `code_result` event right below and
    # rendered into the dedicated code canvas. Streaming the fenced block word-by-word
    # here is what made the whole file appear to get typed out inside the chat like a
    # raw paste, with it only "moving" into the code panel after the fact.
    chat_text = final_response.split("```")[0].strip() if "```" in final_response else final_response
    if not chat_text:
        chat_text = "Here's the code:" if code else final_response

    chunks = chat_text.split(" ")
    for i, chunk in enumerate(chunks):
        space = " " if i < len(chunks) - 1 else ""
        data = {
            "type": "message",
            "assistant_message": chunk + space,
            "conversation_id": session_id,
            "session_id": session_id,
            "goal_complete": final_state.get("completion_score", 100) >= 100
        }
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0.015)

    # Stream the generated code into the dedicated code canvas — never into the chat
    # bubble — so it fills in progressively like Claude's/Gemini's artifact panel instead
    # of the whole file appearing all at once when code_result lands. code_result (below)
    # remains the authoritative final payload the frontend reconciles against, so a dropped
    # or out-of-order delta can never leave the canvas showing something wrong or partial.
    #
    # If streamed_live is True, the code above already went out token-by-token, straight
    # from the model, via code_reset/code_delta events emitted inside the run_code_graph_streaming
    # loop — that's the actual live-generation fix. This chunked replay only runs for code
    # that was never streamed: the simple/direct path, timeout/error fallbacks, and the
    # rare case where post-processing (enforce_single_file_frontend) rewrote code that had
    # already been streamed.
    if streamed_live:
        pass
    elif is_multi and files:
        file_languages = final_state.get("file_languages", {})
        for fname, fcode in files.items():
            if not fcode:
                continue
            flang = file_languages.get(fname, "")
            yield f"data: {json.dumps({'type': 'code_file_start', 'filename': fname, 'language': flang, 'session_id': session_id})}\n\n"
            for delta in iter_code_chunks(fcode):
                yield f"data: {json.dumps({'type': 'code_delta', 'filename': fname, 'delta': delta, 'session_id': session_id})}\n\n"
                await asyncio.sleep(0.01)
    elif code:
        yield f"data: {json.dumps({'type': 'code_start', 'language': language, 'session_id': session_id})}\n\n"
        for delta in iter_code_chunks(code):
            yield f"data: {json.dumps({'type': 'code_delta', 'delta': delta, 'session_id': session_id})}\n\n"
            await asyncio.sleep(0.01)

    # One final event carrying the structured code result, so the frontend can render
    # a live preview (Gemini-Canvas style) only when the code is frontend code. `files`
    # is populated (and `code`/`language` left blank) for multi-file builds so the
    # frontend can render a per-file tree/tab view instead of a single blob.
    yield f"data: {json.dumps({'type': 'code_result', 'code': code, 'language': language, 'files': files, 'show_preview': bool(is_frontend), 'session_id': session_id})}\n\n"

# ----------------------------------------------------------------------
# CODE MODE: FastAPI request models
# ----------------------------------------------------------------------

class CodeChatRequest(BaseModel):
    message: str
    session_id: str
    model: str = "kimi"           # "kimi" -> nvidia/nemotron-3-ultra-550b-a55b, "glm" -> deepseek-ai/deepseek-v4-pro,
                                   # "kimik2.6" -> moonshotai/kimi-k2.6 (thinking ON, see CODE_MODEL_MAP)
    reasoning_level: str = "medium"  # "low" | "medium" | "high" | "max"
    stream: bool = False
    temperature: float = 0.2

class ClearCodeSessionRequest(BaseModel):
    session_id: str

# ----------------------------------------------------------------------
# CODE MODE: FastAPI endpoint
# ----------------------------------------------------------------------

@app.post("/code-chat")
async def code_chat(request: CodeChatRequest):
    try:
        return await _handle_code_chat(request)
    except Exception as e:
        print(f"Code chat handler failed entirely: {e}")
        return {
            "response": "Something went wrong on my end — please try again.",
            "session_id": request.session_id,
            "code": "",
            "language": "",
            "files": {},
            "show_preview": False,
            "goal_progress": 0,
            "completed": False,
            "iterations": 0
        }

async def _handle_code_chat(request: CodeChatRequest):
    session_id = request.session_id

    if session_id not in code_sessions:
        code_sessions[session_id] = {"messages": []}

    session = code_sessions[session_id]
    max_iterations = get_code_max_iterations(request.reasoning_level)
    llm = get_code_llm(request.model, request.temperature)

    session["messages"] = await summarize_memory(session["messages"], llm)
    session["messages"].append(HumanMessage(content=request.message))

    if request.stream:
        return StreamingResponse(
            generate_code_stream(request, session, session_id),
            media_type="text/event-stream"
        )

    is_multi = False
    files: Dict[str, str] = {}

    if is_simple_code_message(request.message):
        result = await answer_code_directly(request.message, session["messages"], request.model, request.temperature)
        final_response, code, language, is_frontend = result["text"], result["code"], result["language"], result["is_frontend"]
        final_state = {"completion_score": 100, "iteration": 1}
    else:
        initial_state = {
            "messages": session["messages"],
            "model_key": request.model,
            "temperature": request.temperature,
            "max_iterations": max_iterations,
            "iteration": 0,
            "completion_score": 0,
            "completed_steps": [],
            "plan": [],
            "target_file": "",
            "is_multi_file": False,
            "estimated_file_count": 1,
            # Seed the real previous code/language (and any previously accumulated
            # multi-file build) so follow-up turns amend the actual prior work instead of
            # relying only on the plain-text chat history — this is what was causing
            # "task completed, no code" replies on a second ask.
            "code": session.get("last_code", ""),
            "language": session.get("last_language", ""),
            "files": dict(session.get("last_files", {})),
            "file_languages": dict(session.get("last_file_languages", {}))
        }
        timeout = code_graph_timeout_seconds(request.model, max_iterations)

        try:
            final_state = await asyncio.wait_for(code_app_graph.ainvoke(initial_state), timeout=timeout)
            if "response" not in final_state:
                print(f"[{session_id}] Code graph finished with no 'response' key. final_state keys: {list(final_state.keys())}")
            final_response = final_state.get("response", "Task completed but no code was formulated.")
            code = final_state.get("code", "")
            language = final_state.get("language", "")
            is_frontend = final_state.get("is_frontend", False)
            is_multi = bool(final_state.get("is_multi_file"))
            files = final_state.get("files", {}) if is_multi else {}

            # Hard safety net: post_review_edge now avoids reaching commit/END before
            # code exists, but if iterations still ran out with none (e.g. reasoning_level
            # "low" giving very few passes), don't hand back an explanation with nothing
            # to show — make one guaranteed direct attempt at real code instead.
            has_any_output = bool(files) if is_multi else bool(code.strip())
            if not has_any_output:
                print(f"[{session_id}] Code graph ended with no output — falling back to direct generation.")
                fallback = await answer_code_directly(request.message, session["messages"], request.model, request.temperature)
                if fallback["code"].strip():
                    final_response, code, language, is_frontend = fallback["text"], fallback["code"], fallback["language"], fallback["is_frontend"]
                    is_multi, files = False, {}
        except asyncio.TimeoutError:
            print(f"[{session_id}] Code Mode workflow timed out after {timeout:.0f}s. Falling back to direct answer.")
            result = await answer_code_directly(request.message, session["messages"], request.model, request.temperature)
            final_response, code, language, is_frontend = result["text"], result["code"], result["language"], result["is_frontend"]
            final_state = {"completion_score": 100, "iteration": 1}
            is_multi, files = False, {}
        except Exception as e:
            print(f"[{session_id}] Code Mode workflow failed: {e}. Falling back to direct answer.")
            result = await answer_code_directly(request.message, session["messages"], request.model, request.temperature)
            final_response, code, language, is_frontend = result["text"], result["code"], result["language"], result["is_frontend"]
            final_state = {"completion_score": 100, "iteration": 1}
            is_multi, files = False, {}

    if is_multi and files:
        # Big multi-file build: assemble every accumulated file into the final answer as
        # its own fenced block, and hand the raw per-file dict back too (see 'files' below)
        # instead of only ever returning the single last file that code_node
        # happened to run on last — that's what was silently dropping every other file.
        file_languages = final_state.get("file_languages", {})
        intro = final_state.get("goal", "") and f"Built {len(files)} files for: {final_state.get('goal')}"
        parts = [intro] if intro else [f"Built {len(files)} files."]
        for fname, fcode in files.items():
            flang = file_languages.get(fname, "")
            parts.append(f"**{fname}**\n```{flang}\n{fcode}\n```")
        final_response = "\n\n".join(p for p in parts if p and p.strip())
        code, language = "", ""
        is_frontend = any(f.lower().endswith((".html", ".htm")) for f in files)
    else:
        # Single-file path: enforce single-file bundling ONCE here, on the final settled
        # answer — not on every internal loop iteration (see enforce_single_file_frontend's
        # docstring for why).
        goal_hint = final_state.get("goal", "") or request.message
        prior_code_for_guard = session.get("last_code", "")
        new_code, new_language, new_is_frontend = await enforce_single_file_frontend(
            code, language, is_frontend, goal_hint, prior_code_for_guard, request.model, request.temperature
        )
        if new_code != code:
            final_response = rebuild_response_with_code(final_response, new_code, new_language)
            code, language, is_frontend = new_code, new_language, new_is_frontend

    if not (final_response or "").strip():
        final_response = "I wasn't able to generate a response for that — please try again."

    # Remember this turn's output so the NEXT turn can seed it back in (only overwrite
    # when we actually got new output — a turn that produced none shouldn't wipe out a
    # perfectly good previous build).
    if is_multi and files:
        session["last_files"], session["last_file_languages"] = files, final_state.get("file_languages", {})
    elif code:
        session["last_code"], session["last_language"] = code, language

    session["messages"].append(AIMessage(content=final_response))

    return {
        "response": final_response,
        "session_id": session_id,
        "code": code,
        "language": language,
        "files": files,
        "show_preview": bool(is_frontend),   # frontend code -> True (render live preview); backend code -> False
        "model": request.model,
        "reasoning_level": request.reasoning_level,
        "max_iterations": max_iterations,
        "goal_progress": final_state.get("completion_score", 100),
        "completed": final_state.get("completion_score", 100) >= 100,
        "iterations": final_state.get("iteration", 1)
    }

@app.post("/clear-code-session")
async def clear_code_session(request: ClearCodeSessionRequest):
    if request.session_id in code_sessions:
        del code_sessions[request.session_id]
    return {"status": "success", "message": f"Code session {request.session_id} cleared."}

if __name__ == "__main__":
    # Ensure uvicorn runs the correct file 'main' instead of 'app'
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
