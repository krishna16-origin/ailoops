# ==================================================
# CONFIGURATION
# ==================================================
import os
import json
import re
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
        model_name = "z-ai/glm-5.2"
    elif model_type_clean == "reasoning":
        model_name = "nvidia/nemotron-3-ultra-550b-a55b"
        
    return ChatNVIDIA(model=model_name, temperature=temperature, max_tokens=4096, timeout=15)

def strip_thinking(text: str) -> str:
    """Removes <think>...</think> reasoning blocks some models emit before the real answer."""
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    return text.strip()


async def execute_llm_structured(llm: ChatNVIDIA, prompt_str: str, pydantic_model, state: dict, retries: int = 3):
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
    if state.get("iteration", 0) >= 5:
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
    max_iterations: int = 5

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

async def generate_stream(response_text: str, state: dict, session_id: str):
    """Yields chunks of the final response to simulate a ChatGPT-like streaming experience."""
    # Chunk by words to simulate token streaming
    chunks = response_text.split(" ")
    for i, chunk in enumerate(chunks):
        space = " " if i < len(chunks) - 1 else ""
        data = {
            "assistant_message": chunk + space,
            "conversation_id": session_id,
            "session_id": session_id,
            "goal_complete": state.get("completion_score", 100) >= 100
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

    # 3. Setup Initial State for the LangGraph Workflow
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

    # 4. Run the workflow with a strict 20-second timeout to prevent overthinking
    try:
        # Executes the full Goal->Plan->Execute->Reflect loop
        final_state = await asyncio.wait_for(app_graph.ainvoke(initial_state), timeout=20.0)
        final_response = final_state.get("response", "Task completed but no response was formulated.")
    except asyncio.TimeoutError:
        print(f"[{session_id}] Workflow timed out after 20 seconds. Falling back to direct answer.")
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

    # 6. Return response (Streamed or JSON)
    if request.stream:
        return StreamingResponse(
            generate_stream(final_response, final_state, session_id),
            media_type="text/event-stream"
        )
    else:
        return {
            "response": final_response,
            "session_id": session_id,
            "goal_progress": final_state.get("completion_score", 100),
            "completed": final_state.get("completion_score", 100) >= 100,
            "iterations": final_state.get("iteration", 1)
        }

if __name__ == "__main__":
    # Ensure uvicorn runs the correct file 'main' instead of 'app'
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
