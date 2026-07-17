# ==================================================
# CONFIGURATION
# ==================================================
import os
import json
import re
import random
import asyncio
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


async def execute_llm_structured(llm: ChatNVIDIA, prompt_str: str, pydantic_model, state: dict, retries: int = 2):
    """Executes an LLM call and ensures structured Pydantic output."""
    parser = PydanticOutputParser(pydantic_object=pydantic_model)
    format_instructions = parser.get_format_instructions()
    
    system_prompt = (
        """You are GoalAI, an intelligent goal-oriented assistant.

Your primary responsibility is not merely answering questions but helping the user successfully achieve their goal.

Always identify the user's objective before responding.

Treat every conversation as part of an ongoing mission.

You should continuously monitor progress toward the user's objective and adapt your strategy whenever new information appears.

## Core Principles

• Always understand the user's real objective before generating an answer.

• Maintain awareness of the overall goal throughout the conversation.

• Break complex goals into manageable subtasks.

• Decide the best next action instead of trying to solve everything at once.

• After every response, internally assess whether the goal has been achieved.

• If the goal is incomplete, continue working toward it in future turns.

• If the user's goal changes, immediately update your understanding and continue from the new objective.

Never lose sight of the user's primary objective.

---

## Goal Understanding

For every message determine:

- Primary goal
- Secondary goals
- User constraints
- Missing information
- Desired final outcome

Ask clarifying questions only when necessary.

Avoid unnecessary questions if enough information already exists.

---

## Planning

Create an internal strategy before responding.

The strategy should:

- identify subtasks
- prioritize them
- choose the best next action
- minimize unnecessary work

Plans may change whenever new information appears.

---

## Execution

Focus only on the next useful action.

Produce responses that move the user closer to completing the goal.

Avoid irrelevant information.

Avoid unnecessary verbosity.

---

## Reflection

After generating a response, internally evaluate whether it:

- answered the user's request
- respected constraints
- remained accurate
- moved the user closer to the goal

If improvements are needed, incorporate them into future responses.

---

## Goal Tracking

Maintain awareness of:

Current goal

Completed progress

Remaining work

Current context

Conversation history

Always use previous conversation context whenever it is relevant.

---

## Adaptation

If the user provides new information:

Update your understanding.

Revise your strategy.

Continue toward the updated goal.

Never restart unless the user explicitly requests it.

---

## Conversation Style

Respond naturally like ChatGPT.

Be clear.

Be concise.

Be helpful.

Avoid robotic language.

Avoid repeatedly mentioning goals or planning.

The user should experience a smooth conversation without seeing your internal planning process.

---

## Accuracy

Do not invent facts.

If uncertain, clearly state uncertainty.

If required information is missing, ask for it.

Prefer correctness over confidence.

---

## Completion

When the user's goal is fully achieved:

Provide the final result.

Mention any remaining considerations only if they are useful.

Then wait for the user's next instruction.

Never continue unnecessary work after the goal has been completed.

Your success is measured by how effectively you help the user achieve their objective while keeping the interaction natural, efficient, and focused.

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
    
    return f"""Conversation History:
{msg_str}

Current Internal State:
- Goal: {state.get('goal', 'Not set')}
- Constraints: {state.get('constraints', [])}
- Plan: {state.get('plan', [])}
- Completed Steps: {state.get('completed_steps', [])}
- Current Step: {state.get('current_step', 'Not set')}
- Prior Response Draft: {state.get('response', 'None')}
- Reflection on Draft: {state.get('reflection', 'None')}
"""

# ==================================================
# LANGGRAPH NODES
# ==================================================

async def understand_goal_node(state: AgentState) -> dict:
    llm = get_llm(state["model_type"], state["temperature"])
    prompt = "Analyze this conversation and extract the core goal, intent, constraints, and missing info.\n\n{context}"
    
    res = await execute_llm_structured(llm, prompt, GoalExtraction, {"context": format_context(state)})
    
    return {
        "goal": res.main_goal if res else "Provide a helpful response.",
        "hidden_intent": res.hidden_intent if res else "",
        "constraints": res.constraints if res else [],
        "iteration": 0,
        "completed_steps": [],
        "plan": state.get("plan", [])
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
    llm = get_llm(state["model_type"], state["temperature"])
    prompt = "Execute the 'Current Step' to satisfy the user's 'Goal'. Provide your generated response text and reasoning.\n\n{context}"
    
    res = await execute_llm_structured(llm, prompt, ExecutorOutput, {"context": format_context(state)})
    
    new_completed = state.get("completed_steps", []) + [state.get("current_step", "")]
    
    return {
        "response": res.response if res else "I apologize, I encountered an issue formulating my answer.",
        "executor_reasoning": res.reasoning_summary if res else "",
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
    """Decides whether to end the reasoning loop or continue planning/executing."""
    if state.get("completion_score", 0) >= 100:
        return END
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

def node_detail(node_name: str, state: dict) -> str:
    """
    Turns whatever a node actually produced into a real sentence of reasoning text,
    so the thinking UI shows genuine content instead of a decorative status word.
    """
    if node_name == "understand_goal":
        goal = state.get("goal", "")
        return f"Understanding the goal: {goal}" if goal else "Understanding the goal."
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

async def run_graph_streaming(initial_state: dict, timeout: float):
    """
    Runs the LangGraph workflow node-by-node, yielding (node_name, state_so_far) the moment each
    node actually finishes — this is real backend progress, not a simulated/canned timer. Enforces
    the same overall timeout budget as the non-streaming path so a slow run still falls back cleanly.
    """
    state_acc = dict(initial_state)
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    agen = app_graph.astream(initial_state, stream_mode="updates")
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            chunk = await asyncio.wait_for(agen.__anext__(), timeout=remaining)
            node_name, node_output = next(iter(chunk.items()))
            state_acc.update(node_output)
            yield node_name, state_acc
    except StopAsyncIteration:
        return
    finally:
        await agen.aclose()

async def generate_stream(request: "ChatRequest", session: dict, session_id: str):
    """
    Streams two kinds of SSE events to the frontend:
      - {"type": "status", "step": <node name>, "label": <text>}  while the agent is actually working
      - {"type": "message", "assistant_message": <word chunk>, ...}  once the final answer is ready
    The status events mirror whichever LangGraph node just finished, so the "thought process" shown
    in the UI is synced to real backend state instead of a client-side simulated cycle.
    """
    if is_simple_message(request.message):
        yield f"data: {json.dumps({'type': 'status', 'step': 'direct', 'label': 'Answering', 'detail': 'This is a short message, so answering directly without the full planning loop.'})}\n\n"
        final_response = await answer_directly(
            request.message, session["messages"], request.model_type, request.temperature
        )
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

        try:
            async for node_name, state_so_far in run_graph_streaming(initial_state, timeout):
                label = NODE_LABELS.get(node_name, node_name)
                detail = node_detail(node_name, state_so_far)
                yield f"data: {json.dumps({'type': 'status', 'step': node_name, 'label': label, 'detail': detail})}\n\n"
                final_state = state_so_far
            final_response = final_state.get("response", "Task completed but no response was formulated.")
        except asyncio.TimeoutError:
            print(f"[{session_id}] Streaming workflow timed out after {timeout:.0f}s. Falling back to direct answer.")
            yield f"data: {json.dumps({'type': 'status', 'step': 'fallback', 'label': random.choice(FALLBACK_LABELS), 'detail': 'The full reasoning loop was taking too long, so falling back to a direct answer.'})}\n\n"
            final_response = await answer_directly(
                request.message, session["messages"], request.model_type, request.temperature
            )
            final_state = {"completion_score": 100, "iteration": 1}
        except Exception as e:
            print(f"[{session_id}] Streaming workflow failed: {e}. Falling back to direct answer.")
            yield f"data: {json.dumps({'type': 'status', 'step': 'fallback', 'label': random.choice(FALLBACK_LABELS), 'detail': 'Something went wrong in the reasoning loop, so falling back to a direct answer.'})}\n\n"
            final_response = await answer_directly(
                request.message, session["messages"], request.model_type, request.temperature
            )
            final_state = {"completion_score": 100, "iteration": 1}

    # Now that we actually have the final answer, save it to memory
    session["messages"].append(AIMessage(content=final_response))

    # Stream the final answer word-by-word (ChatGPT-style typing effect)
    chunks = final_response.split(" ")
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
        await asyncio.sleep(0.015) # Simulated typing delay

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
    messages = [SystemMessage(content="You are GoalAI, a helpful, friendly assistant. Respond naturally and concisely.")]
    messages.extend(history[-6:])
    messages.append(HumanMessage(content=message))

    # Try the requested model first
    try:
        llm = get_llm(model_type, temperature)
        res = await llm.ainvoke(messages)
        answer = strip_thinking(res.content).strip()
        if answer:
            return answer
    except Exception as e:
        print(f"Primary model failed: {e}")

    # Fallback: always try the fast model before giving up
    try:
        fallback_llm = get_llm("fast", temperature)
        res = await fallback_llm.ainvoke(messages)
        answer = strip_thinking(res.content).strip()
        if answer:
            return answer
    except Exception as e:
        print(f"Fallback model also failed: {e}")

    # Absolute last resort — user still gets a real response, never a crash
    return "I'm having trouble reaching the model right now. Please try again in a moment."

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
# ----------------------------------------------------------------------

class CodeGoalExtraction(BaseModel):
    main_goal: str = Field(description="The coding task the user actually wants solved")
    language_hint: str = Field(description="Programming language / framework implied by the request, if any (empty string if unclear)")
    constraints: List[str] = Field(description="Technical constraints or requirements to follow")
    missing_information: List[str] = Field(description="What we need to ask the user, if anything")

class CodePlannerOutput(BaseModel):
    plan: List[str] = Field(description="Ordered list of technical steps to build the solution")
    current_step: str = Field(description="The single immediate next step to execute")
    approach: str = Field(description="Technical approach / design decision for this step")

class CodeExecutorOutput(BaseModel):
    code: str = Field(description="The generated code for the current step, complete and runnable")
    language: str = Field(description="Programming language of the code, e.g. python, javascript, html, jsx, css")
    explanation: str = Field(description="Brief explanation of what the code does")
    is_frontend: bool = Field(description="True if this code renders a UI in a browser (html/css/js/react/vue/etc), false if it is backend/server/CLI/script code")

class CodeReflectorOutput(BaseModel):
    quality: str = Field(description="Assessment of the code quality")
    correctness: str = Field(description="Is the code correct / free of bugs?")
    bugs_found: str = Field(description="Any bugs, edge cases, or issues found")
    improvements: str = Field(description="Actionable advice to improve the code")

class CodeEvaluatorOutput(BaseModel):
    completion_percentage: int = Field(description="0 to 100 representing how complete the coding task is")
    confidence: float = Field(description="Confidence in this evaluation (0.0 to 1.0)")
    should_continue: bool = Field(description="Whether more iterations are needed to finish the task")

# ----------------------------------------------------------------------
# CODE MODE: Graph State
# ----------------------------------------------------------------------

class CodeAgentState(TypedDict):
    messages: List[BaseMessage]
    goal: str
    constraints: List[str]
    plan: List[str]
    current_step: str
    completed_steps: List[str]
    reflection: str
    completion_score: int
    iteration: int
    max_iterations: int
    code: str
    language: str
    is_frontend: bool
    explanation: str
    response: str
    model_key: str
    temperature: float

# ----------------------------------------------------------------------
# CODE MODE: Session Memory (kept separate from the normal chat sessions)
# ----------------------------------------------------------------------

code_sessions: Dict[str, Dict[str, Any]] = {}

# ----------------------------------------------------------------------
# CODE MODE: Model Router — only two models, both selectable by the user
# ----------------------------------------------------------------------

CODE_MODEL_MAP = {
    "kimi": "nvidia/nemotron-3-ultra-550b-a55b",   # high-end reasoning and coding
    "glm": "deepseek-ai/deepseek-v4-pro",          # fast response with code
}

# Both models ship with "Thinking" mode ON by default on NVIDIA NIM. That's the
# actual reason Code Mode was returning no real answers: for structured JSON
# calls (goal extraction, planning, execution, reflection, evaluation), when the
# model's reasoning trace doesn't finish closing before max_tokens is hit, the
# API hands back an empty/null final `content` — the whole token budget went to
# reasoning_content instead of the answer. That's a documented NIM behavior for
# both models (chat_template_kwargs), not something retries fix, since the same
# elaborate system prompt makes the model "think" the same way every attempt.
# Different model families use different toggle keys ("thinking" vs "enable_thinking").
# Sending both every time — instead of keying off a hardcoded model-name list —
# means this keeps working no matter which models CODE_MODEL_MAP points to;
# unrecognized keys are harmlessly ignored by the chat template.
CODE_THINKING_OFF_KWARGS = {"thinking": False, "enable_thinking": False}

def get_code_llm(model_key: str, temperature: float = 0.2) -> ChatNVIDIA:
    """Routes to one of the two Code Mode models. Uses the same NVIDIA_API_KEY as the rest of the app."""
    key = (model_key or "").strip().lower()
    model_name = CODE_MODEL_MAP.get(key, CODE_MODEL_MAP["kimi"])  # default to the high-reasoning model
    thinking_kwargs = CODE_THINKING_OFF_KWARGS
    return ChatNVIDIA(
        model=model_name,
        temperature=temperature,
        max_tokens=16384,
        timeout=120,
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

CODE_SYSTEM_PROMPT = """You are an elite software engineering AI whose only responsibility is solving software engineering tasks.

Your domain is strictly limited to:
- Programming
- Software architecture
- Algorithms and data structures
- Debugging
- Refactoring
- Code review
- API design
- Databases
- DevOps
- Testing
- Performance optimization
- Security
- Documentation
- System design

Never switch into a general assistant role.

If a request is unrelated to software engineering, politely refuse and explain that you are a dedicated coding agent.

--------------------------------------------------
PRIMARY OBJECTIVE
--------------------------------------------------

Produce correct, maintainable, production-quality software.

Optimize for:

- correctness
- robustness
- readability
- maintainability
- scalability
- security
- performance

Never optimize for short answers over correct answers.

Quality is always more important than speed.

--------------------------------------------------
THINKING PRINCIPLES
--------------------------------------------------

Before producing code:

1. Understand the entire problem.
2. Infer missing technical details when reasonable.
3. Identify constraints.
4. Identify edge cases.
5. Design before coding.
6. Prefer simple solutions over clever ones.
7. Avoid unnecessary abstractions.
8. Produce deterministic solutions.

Never randomly guess APIs.

If uncertain, clearly state assumptions.

--------------------------------------------------
ENGINEERING STANDARDS
--------------------------------------------------

Write code as if it will be maintained for years.

Code should be:

- modular
- readable
- testable
- reusable
- documented
- typed whenever possible

Avoid:

- duplicated logic
- magic numbers
- deeply nested code
- hidden side effects
- global mutable state

Prefer:

- pure functions
- composition
- dependency injection
- descriptive naming
- explicit behavior

--------------------------------------------------
CODE QUALITY
--------------------------------------------------

Every implementation should aim for:

Correctness first.

Then:

- simplicity
- readability
- maintainability
- efficiency

Never intentionally introduce:

- dead code
- unreachable branches
- duplicated logic
- unnecessary complexity

--------------------------------------------------
BUG PREVENTION
--------------------------------------------------

Actively search for:

- null references
- race conditions
- resource leaks
- off-by-one errors
- integer overflow
- invalid assumptions
- concurrency issues
- API misuse
- security vulnerabilities

Prevent them before they happen.

--------------------------------------------------
SECURITY
--------------------------------------------------

Always consider:

- input validation
- output escaping
- authentication
- authorization
- injection attacks
- secrets management
- secure defaults
- least privilege

Never generate insecure production code when a secure alternative exists.

--------------------------------------------------
PERFORMANCE
--------------------------------------------------

When writing algorithms:

Consider:

Time complexity

Space complexity

Scalability

Avoid premature optimization unless required.

--------------------------------------------------
TESTING
--------------------------------------------------

Every implementation should be easy to test.

When appropriate, suggest:

- unit tests
- integration tests
- edge cases
- failure cases

--------------------------------------------------
ERROR HANDLING
--------------------------------------------------

Handle failures explicitly.

Prefer meaningful errors over silent failures.

Never swallow exceptions without reason.

--------------------------------------------------
COMMUNICATION
--------------------------------------------------

Be concise but technically complete.

Avoid unnecessary filler.

Explain only when useful.

Focus on engineering.

--------------------------------------------------
WHEN MODIFYING EXISTING CODE
--------------------------------------------------

Preserve:

- public APIs
- behavior
- compatibility

unless the task explicitly requests breaking changes.

Minimize unnecessary modifications.

--------------------------------------------------
WHEN DEBUGGING
--------------------------------------------------

Find the root cause.

Do not merely patch symptoms.

Explain why the issue occurred.

--------------------------------------------------
WHEN REFACTORING
--------------------------------------------------

Improve:

- readability
- maintainability
- architecture

without changing behavior.

--------------------------------------------------
WHEN WRITING NEW CODE
--------------------------------------------------

Prefer production-ready implementations over toy examples.

Avoid placeholders.

Avoid pseudocode unless explicitly requested.

--------------------------------------------------
OUTPUT REQUIREMENTS
--------------------------------------------------

Generated code should:

compile whenever possible

follow language conventions

include imports

include necessary dependencies

be complete

not omit important sections

--------------------------------------------------
FINAL VALIDATION
--------------------------------------------------

Before finishing, internally verify:

✓ Requirements satisfied

✓ Logic is correct

✓ Edge cases considered

✓ Security reviewed

✓ Performance acceptable

✓ Code is maintainable

✓ No unnecessary complexity

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
# used to run inside code_executor_node itself, i.e. on every internal planner/executor
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

    base_system = CODE_SYSTEM_PROMPT if CODE_SYSTEM_PROMPT.strip() else "You are a coding assistant that returns structured, well-formatted output."

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

            return parser.parse(content.strip()), None
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            print(f"[CodeMode] Structured Parsing Retry {attempt + 1}/{retries} failed: {e} | Raw content: {res.content[:300] if 'res' in dir() else 'N/A'}")
            await asyncio.sleep(0.5)

    return None, last_error

def format_code_context(state: CodeAgentState) -> str:
    msg_str = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in state.get("messages", [])])

    return f"""Conversation History:
{msg_str}

Current Internal State:
- Goal: {state.get('goal', 'Not set')}
- Constraints: {state.get('constraints', [])}
- Plan: {state.get('plan', [])}
- Completed Steps: {state.get('completed_steps', [])}
- Current Step: {state.get('current_step', 'Not set')}
- Prior Code Draft: {state.get('code', 'None')}
- Prior Language: {state.get('language', 'None')}
- Reflection on Draft: {state.get('reflection', 'None')}
"""

# ----------------------------------------------------------------------
# CODE MODE: LangGraph Nodes
# ----------------------------------------------------------------------

async def understand_code_goal_node(state: CodeAgentState) -> dict:
    llm = get_code_llm(state["model_key"], state["temperature"])
    prompt = "Analyze this conversation and extract the core coding goal, language hint, constraints, and missing info.\n\n{context}"

    res, err = await execute_code_llm_structured(llm, prompt, CodeGoalExtraction, {"context": format_code_context(state)})

    return {
        "goal": res.main_goal if res else "Write the requested code.",
        "constraints": res.constraints if res else [],
        "iteration": 0,
        "completed_steps": [],
        "plan": state.get("plan", []),
        "last_error": err
    }

async def code_planner_node(state: CodeAgentState) -> dict:
    llm = get_code_llm(state["model_key"], state["temperature"])
    prompt = "Based on the goal and completed steps, create or update the technical plan. Identify the single immediate next step.\n\n{context}"

    res, err = await execute_code_llm_structured(llm, prompt, CodePlannerOutput, {"context": format_code_context(state)})

    iteration = state.get("iteration", 0) + 1

    return {
        "plan": res.plan if res else state.get("plan", []),
        "current_step": res.current_step if res else "Write the code that satisfies the goal.",
        "iteration": iteration,
        "last_error": err
    }

async def code_executor_node(state: CodeAgentState) -> dict:
    llm = get_code_llm(state["model_key"], state["temperature"])
    prompt = (
        "Execute the 'Current Step' to satisfy the coding 'Goal'. Return complete, runnable "
        "code plus its language and a brief explanation. IMPORTANT: the 'code' field must "
        "never be left empty for a coding task — even if this step is just a review, a small "
        "tweak, or you believe the goal is already met, return the full current version of the "
        "code (unchanged if nothing needed to change), not just an explanation with no code."
        "\n\n{context}"
    )

    res, err = await execute_code_llm_structured(llm, prompt, CodeExecutorOutput, {"context": format_code_context(state)})

    new_completed = state.get("completed_steps", []) + [state.get("current_step", "")]
    prior_code = state.get("code", "")

    if res:
        language = res.language
        code = res.code
        try:
            is_frontend = res.is_frontend if res.is_frontend is not None else classify_code_target(language, code)
        except Exception as e:
            print(f"[CodeMode] classify_code_target failed: {e}")
            is_frontend = bool(res.is_frontend)
        explanation = res.explanation

        # Guard -1: the model decided no new code was needed and left 'code' blank.
        # That should never happen on a genuine coding task when a working prior file
        # already exists — carry the prior file forward instead of resolving to
        # "explanation only, no code", which is exactly what showed up as
        # "task completed, no code was formulated".
        if not code and prior_code:
            code = prior_code
            language = language or state.get("language", "")
            is_frontend = is_frontend or bool(state.get("is_frontend", False))
            if not explanation.strip():
                explanation = "No changes were needed for this step — carrying the existing code forward."
    else:
        language, code, is_frontend = "", "", False
        # Surface the real reason instead of a generic canned message, so a bad
        # API key / rate limit / model error is visible instead of looking like
        # a random glitch every time.
        explanation = f"I hit an issue generating code for this step: {err}" if err else "I hit an issue generating code for this step."
        # Same carry-forward here: an LLM/parsing failure shouldn't wipe out a
        # perfectly good previous file either.
        if prior_code:
            code = prior_code
            language = state.get("language", "")
            is_frontend = bool(state.get("is_frontend", False))

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
        "completed_steps": new_completed,
        "last_error": err
    }

async def code_reflector_node(state: CodeAgentState) -> dict:
    llm = get_code_llm(state["model_key"], state["temperature"])
    prompt = "Review the 'Prior Code Draft' against the 'Goal' and 'Constraints'. Check correctness, bugs, and quality.\n\n{context}"

    res, err = await execute_code_llm_structured(llm, prompt, CodeReflectorOutput, {"context": format_code_context(state)})

    reflection_str = "Looks solid."
    if res:
        reflection_str = f"Quality: {res.quality} | Correctness: {res.correctness} | Bugs: {res.bugs_found} | Improvements: {res.improvements}"
    elif err:
        reflection_str = f"Skipped review — the review step hit an error: {err}"

    return {
        "reflection": reflection_str,
        "last_error": err
    }

async def code_evaluator_node(state: CodeAgentState) -> dict:
    llm = get_code_llm(state["model_key"], state["temperature"])
    prompt = "Based on the reflection, evaluate if the coding goal is now fully achieved (0-100 completion).\n\n{context}"

    res, err = await execute_code_llm_structured(llm, prompt, CodeEvaluatorOutput, {"context": format_code_context(state)})

    # Defaults to 100 on failure so the loop stops instead of retrying against
    # a backend that's already failing, rather than implying the task is done.
    score = res.completion_percentage if res else 100

    return {
        "completion_score": score,
        "last_error": err
    }

def code_decision_edge(state: CodeAgentState) -> str:
    """Decides whether to end the code reasoning loop or continue planning/executing.

    Bug fix: this used to end the loop purely on the evaluator's own opinion of
    "is this done?", with no check that code had actually been produced. If the
    planner's first step was something non-code-producing (e.g. "clarify
    requirements") and the evaluator still scored it 100% complete, the loop
    ended having never generated any code — surfacing as "task completed but
    no code was formulated" with nothing to show. Now: as long as iterations
    remain, an empty 'code' field forces at least one more planner/executor
    pass before the loop is allowed to end.
    """
    has_code = bool((state.get("code") or "").strip())
    out_of_iterations = state.get("iteration", 0) >= state.get("max_iterations", 3)

    if not has_code and not out_of_iterations:
        return "code_planner"
    if state.get("completion_score", 0) >= 100:
        return END
    if out_of_iterations:
        return END
    return "code_planner"

# ----------------------------------------------------------------------
# CODE MODE: Workflow Graph (separate compiled graph from the normal app_graph)
# ----------------------------------------------------------------------

code_workflow = StateGraph(CodeAgentState)

code_workflow.add_node("understand_code_goal", understand_code_goal_node)
code_workflow.add_node("code_planner", code_planner_node)
code_workflow.add_node("code_executor", code_executor_node)
code_workflow.add_node("code_reflector", code_reflector_node)
code_workflow.add_node("code_evaluator", code_evaluator_node)

code_workflow.set_entry_point("understand_code_goal")
code_workflow.add_edge("understand_code_goal", "code_planner")
code_workflow.add_edge("code_planner", "code_executor")
code_workflow.add_edge("code_executor", "code_reflector")
code_workflow.add_edge("code_reflector", "code_evaluator")
code_workflow.add_conditional_edges("code_evaluator", code_decision_edge)

code_app_graph = code_workflow.compile()

# ----------------------------------------------------------------------
# CODE MODE: Direct-answer fallback (used for tiny messages, timeouts, errors)
# ----------------------------------------------------------------------

CODE_FENCE_RE = re.compile(r"```([\w+#.-]*)\n([\s\S]*?)```")

def extract_code_from_answer(answer: str) -> tuple[str, str]:
    """Pulls the first fenced code block out of a plain-text LLM answer, if any.
    Needed because answer_code_directly (used for timeouts/errors/simple messages)
    only ever produced free-text before — the code was inside that text but never
    split out into its own field, so the frontend Canvas had nothing to render."""
    match = CODE_FENCE_RE.search(answer or "")
    if not match:
        return "", ""
    return (match.group(1) or "").strip(), match.group(2).strip()

async def answer_code_directly(message: str, history: List[BaseMessage], model_key: str, temperature: float) -> dict:
    """Always returns a real code answer — falls back to the other Code Mode model if the primary one fails."""
    base_system = CODE_SYSTEM_PROMPT if CODE_SYSTEM_PROMPT.strip() else "You are a coding assistant. Respond with code and a brief explanation."
    messages = [SystemMessage(content=base_system)]
    messages.extend(history[-6:])
    messages.append(HumanMessage(content=message))

    fallback_key = "glm" if (model_key or "").strip().lower() != "glm" else "kimi"

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
    the two Code Mode models: kimi (high-end reasoning) gets a larger per-call budget than glm (fast)."""
    per_call_seconds = 140.0 if (model_key or "").strip().lower() == "kimi" else 90.0
    worst_case_calls = 1 + (max_iterations * 4)
    return min(worst_case_calls * per_call_seconds, 240.0)

CODE_NODE_LABELS = {
    "understand_code_goal": "Understanding the coding goal",
    "code_planner": "Planning the implementation",
    "code_executor": "Writing the code",
    "code_reflector": "Reviewing the code",
    "code_evaluator": "Confirming completion",
}

def code_node_detail(node_name: str, state: dict) -> str:
    if node_name == "understand_code_goal":
        err = state.get("last_error")
        if err:
            return f"Understanding the goal hit an error: {err}"
        goal = state.get("goal", "")
        return f"Understanding the goal: {goal}" if goal else "Understanding the coding goal."
    if node_name == "code_planner":
        err = state.get("last_error")
        if err:
            return f"Planning hit an error: {err}"
        step = state.get("current_step", "")
        return f"Planning the next step: {step}" if step else "Planning the implementation."
    if node_name == "code_executor":
        explanation = state.get("explanation", "")
        return explanation if explanation else "Writing code for the current step."
    if node_name == "code_reflector":
        reflection = state.get("reflection", "")
        return reflection if reflection else "Reviewing the code for bugs and quality."
    if node_name == "code_evaluator":
        score = state.get("completion_score", None)
        return f"Completion check: {score}% of the task is done." if score is not None else "Checking whether the task is complete."
    return CODE_NODE_LABELS.get(node_name, node_name)

async def run_code_graph_streaming(initial_state: dict, timeout: float):
    """Streams (node_name, state) tuples from the LangGraph run. Also yields periodic
    (None, state_acc) heartbeats every ~12s so a single slow node call (these can take
    60-140s with zero bytes sent in between) doesn't leave the SSE connection silent long
    enough for a hosting platform's proxy (e.g. Render) to kill it as idle. That silent
    kill — not a Python exception — is what shows up as 'thinking animation finishes,
    then nothing': the connection dies mid-node, so there's nothing to catch or log."""
    state_acc = dict(initial_state)
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    agen = code_app_graph.astream(initial_state, stream_mode="updates")
    HEARTBEAT_INTERVAL = 12.0
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            wait_chunk = min(HEARTBEAT_INTERVAL, remaining)
            try:
                chunk = await asyncio.wait_for(agen.__anext__(), timeout=wait_chunk)
            except asyncio.TimeoutError:
                if wait_chunk >= remaining:
                    raise
                yield None, state_acc  # heartbeat only — no node finished yet
                continue
            node_name, node_output = next(iter(chunk.items()))
            state_acc.update(node_output)
            yield node_name, state_acc
    except StopAsyncIteration:
        return
    finally:
        await agen.aclose()

async def generate_code_stream(request: "CodeChatRequest", session: dict, session_id: str):
    """Streams SSE events for Code Mode: status updates while working, then the final
    answer word-by-word, followed by one 'code_result' event carrying the code/language/
    show_preview fields so the frontend can decide whether to render a live preview."""
    max_iterations = get_code_max_iterations(request.reasoning_level)

    if is_simple_code_message(request.message):
        yield f"data: {json.dumps({'type': 'status', 'step': 'direct', 'label': 'Answering', 'detail': 'Short message — answering directly without the full coding loop.'})}\n\n"
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
            # Seed the real previous code/language so follow-up turns amend the actual
            # prior file instead of relying only on the plain-text chat history — this is
            # what was causing "task completed, no code" replies on a second ask.
            "code": session.get("last_code", ""),
            "language": session.get("last_language", "")
        }
        timeout = code_graph_timeout_seconds(request.model, max_iterations)
        final_state = initial_state

        try:
            async for node_name, state_so_far in run_code_graph_streaming(initial_state, timeout):
                if node_name is None:
                    yield ": keep-alive\n\n"  # SSE comment — keeps the connection alive, frontend ignores it
                    continue
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

            # Hard safety net: code_decision_edge now avoids ending the loop before code
            # exists, but if iterations still ran out with none (e.g. reasoning_level
            # "low" giving very few passes), don't hand back an explanation with nothing
            # to show — make one guaranteed direct attempt at real code instead.
            if not code.strip():
                print(f"[{session_id}] Code graph ended with empty code — falling back to direct generation.")
                yield f"data: {json.dumps({'type': 'status', 'step': 'fallback', 'label': random.choice(FALLBACK_LABELS), 'detail': 'The coding loop finished without code, so falling back to a direct answer.'})}\n\n"
                fallback = await answer_code_directly(request.message, session["messages"], request.model, request.temperature)
                if fallback["code"].strip():
                    final_response, code, language, is_frontend = fallback["text"], fallback["code"], fallback["language"], fallback["is_frontend"]
        except asyncio.TimeoutError:
            print(f"[{session_id}] Code Mode streaming timed out after {timeout:.0f}s. Falling back to direct answer.")
            yield f"data: {json.dumps({'type': 'status', 'step': 'fallback', 'label': random.choice(FALLBACK_LABELS), 'detail': 'The coding loop was taking too long, so falling back to a direct answer.'})}\n\n"
            result = await answer_code_directly(request.message, session["messages"], request.model, request.temperature)
            final_response, code, language, is_frontend = result["text"], result["code"], result["language"], result["is_frontend"]
            final_state = {"completion_score": 100, "iteration": 1}
        except Exception as e:
            print(f"[{session_id}] Code Mode streaming failed: {e}. Falling back to direct answer.")
            yield f"data: {json.dumps({'type': 'status', 'step': 'fallback', 'label': random.choice(FALLBACK_LABELS), 'detail': 'Something went wrong in the coding loop, so falling back to a direct answer.'})}\n\n"
            result = await answer_code_directly(request.message, session["messages"], request.model, request.temperature)
            final_response, code, language, is_frontend = result["text"], result["code"], result["language"], result["is_frontend"]
            final_state = {"completion_score": 100, "iteration": 1}

    # Enforce single-file bundling ONCE here, on the final settled answer — not on every
    # internal loop iteration (see enforce_single_file_frontend's docstring for why).
    goal_hint = final_state.get("goal", "") or request.message
    prior_code_for_guard = session.get("last_code", "")
    new_code, new_language, new_is_frontend = await enforce_single_file_frontend(
        code, language, is_frontend, goal_hint, prior_code_for_guard, request.model, request.temperature
    )
    if new_code != code:
        final_response = rebuild_response_with_code(final_response, new_code, new_language)
        code, language, is_frontend = new_code, new_language, new_is_frontend

    # Hard guarantee: no matter what path got here, never let an empty string reach the
    # frontend — that's what silently produced a blank bubble after the thinking animation
    # ended (the animation still resolves to "Thought for Ns" either way, so an empty
    # answer looked exactly like the whole thing just doing nothing).
    if not (final_response or "").strip():
        final_response = "I wasn't able to generate a response for that — please try again."

    # Remember this turn's code so the NEXT turn can seed it back in (only overwrite
    # when we actually got new code — a turn that produced none shouldn't wipe out a
    # perfectly good previous file).
    if code:
        session["last_code"], session["last_language"] = code, language

    session["messages"].append(AIMessage(content=final_response))

    chunks = final_response.split(" ")
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

    # One final event carrying the structured code result, so the frontend can render
    # a live preview (Gemini-Canvas style) only when the code is frontend code.
    yield f"data: {json.dumps({'type': 'code_result', 'code': code, 'language': language, 'show_preview': bool(is_frontend), 'session_id': session_id})}\n\n"

# ----------------------------------------------------------------------
# CODE MODE: FastAPI request models
# ----------------------------------------------------------------------

class CodeChatRequest(BaseModel):
    message: str
    session_id: str
    model: str = "kimi"           # "kimi" -> moonshotai/kimi-k2.6, "glm" -> z-ai/glm-5.2
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
            # Seed the real previous code/language so follow-up turns amend the actual
            # prior file instead of relying only on the plain-text chat history — this is
            # what was causing "task completed, no code" replies on a second ask.
            "code": session.get("last_code", ""),
            "language": session.get("last_language", "")
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

            # Hard safety net: code_decision_edge now avoids ending the loop before code
            # exists, but if iterations still ran out with none (e.g. reasoning_level
            # "low" giving very few passes), don't hand back an explanation with nothing
            # to show — make one guaranteed direct attempt at real code instead.
            if not code.strip():
                print(f"[{session_id}] Code graph ended with empty code — falling back to direct generation.")
                fallback = await answer_code_directly(request.message, session["messages"], request.model, request.temperature)
                if fallback["code"].strip():
                    final_response, code, language, is_frontend = fallback["text"], fallback["code"], fallback["language"], fallback["is_frontend"]
        except asyncio.TimeoutError:
            print(f"[{session_id}] Code Mode workflow timed out after {timeout:.0f}s. Falling back to direct answer.")
            result = await answer_code_directly(request.message, session["messages"], request.model, request.temperature)
            final_response, code, language, is_frontend = result["text"], result["code"], result["language"], result["is_frontend"]
            final_state = {"completion_score": 100, "iteration": 1}
        except Exception as e:
            print(f"[{session_id}] Code Mode workflow failed: {e}. Falling back to direct answer.")
            result = await answer_code_directly(request.message, session["messages"], request.model, request.temperature)
            final_response, code, language, is_frontend = result["text"], result["code"], result["language"], result["is_frontend"]
            final_state = {"completion_score": 100, "iteration": 1}

    # Enforce single-file bundling ONCE here, on the final settled answer — not on every
    # internal loop iteration (see enforce_single_file_frontend's docstring for why).
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

    # Remember this turn's code so the NEXT turn can seed it back in (only overwrite
    # when we actually got new code — a turn that produced none shouldn't wipe out a
    # perfectly good previous file).
    if code:
        session["last_code"], session["last_language"] = code, language

    session["messages"].append(AIMessage(content=final_response))

    return {
        "response": final_response,
        "session_id": session_id,
        "code": code,
        "language": language,
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
