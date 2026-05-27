"""
Smart Home AI — Agentic Supervisor Architecture
Orchestrator oversees EmbeddingAI and ToolCaller in inner retry loops.
If final results are unsatisfactory, orch re-triggers the entire pipeline.

Graph flow:
  orchestrator
      ├─→ embedding_ai  ←─ inner retry (orch reviews nodes)
      ├─→ tool_caller   ←─ inner retry (orch reviews plan)
      ├─→ executor
      └─→ responder  OR  back to embedding_ai (outer retry)
"""

import json
import chromadb
from chromadb.utils import embedding_functions
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional

# ─────────────────────────────────────────────────────────
# IoT Devices (demo)
# ─────────────────────────────────────────────────────────

class IoTDevices:
    # ─── LIGHTS ───
    @staticmethod
    def light_on_bedroom():
        print(f"  [IoT] 💡 Light ON  → Bedroom")
        return "Bedroom light turned on"

    @staticmethod
    def light_off_bedroom():
        print(f"  [IoT] 🌑 Light OFF → Bedroom")
        return "Bedroom light turned off"

    @staticmethod
    def light_on_hall():
        print(f"  [IoT] 💡 Light ON  → Hall")
        return "Hall light turned on"

    @staticmethod
    def light_off_hall():
        print(f"  [IoT] 🌑 Light OFF → Hall")
        return "Hall light turned off"

    @staticmethod
    def light_on_kitchen():
        print(f"  [IoT] 💡 Light ON  → Kitchen")
        return "Kitchen light turned on"

    @staticmethod
    def light_off_kitchen():
        print(f"  [IoT] 🌑 Light OFF → Kitchen")
        return "Kitchen light turned off"

    # ─── TV ───
    @staticmethod
    def tv_on_hall():
        print(f"  [IoT] 📺 TV ON   → Hall")
        return "Hall TV turned on"

    @staticmethod
    def tv_off_hall():
        print(f"  [IoT] 📺 TV OFF  → Hall")
        return "Hall TV turned off"

    @staticmethod
    def tv_on_bedroom():
        print(f"  [IoT] 📺 TV ON   → Bedroom")
        return "Bedroom TV turned on"

    @staticmethod
    def tv_off_bedroom():
        print(f"  [IoT] 📺 TV OFF  → Bedroom")
        return "Bedroom TV turned off"

    # ─── FAN ───
    @staticmethod
    def fan_on_bedroom():
        print(f"  [IoT] 🌀 Fan ON   → Bedroom")
        return "Bedroom fan turned on"

    @staticmethod
    def fan_off_bedroom():
        print(f"  [IoT] ⛔ Fan OFF  → Bedroom")
        return "Bedroom fan turned off"

    @staticmethod
    def fan_on_hall():
        print(f"  [IoT] 🌀 Fan ON   → Hall")
        return "Hall fan turned on"

    @staticmethod
    def fan_off_hall():
        print(f"  [IoT] ⛔ Fan OFF  → Hall")
        return "Hall fan turned off"

    # ─── AC / THERMOSTAT ───
    @staticmethod
    def ac_set_temp_bedroom(temp: int = 22):
        print(f"  [IoT] ❄️  AC → Bedroom {temp}°C")
        return f"Bedroom AC set to {temp}°C"

    @staticmethod
    def thermostat_set(temp: int = 24):
        print(f"  [IoT] 🌡️  Thermostat → {temp}°C")
        return f"Thermostat set to {temp}°C"

    # ─── DOORS ───
    @staticmethod
    def door_lock_front():
        print(f"  [IoT] 🔒 Lock     → Front Door")
        return "Front door locked"

    @staticmethod
    def door_unlock_front():
        print(f"  [IoT] 🔓 Unlock   → Front Door")
        return "Front door unlocked"

    @staticmethod
    def door_lock_back():
        print(f"  [IoT] 🔒 Lock     → Back Door")
        return "Back door locked"

    @staticmethod
    def door_unlock_back():
        print(f"  [IoT] 🔓 Unlock   → Back Door")
        return "Back door unlocked"


DEVICE_MAP = {
    # Lights
    "light_on_bedroom":      IoTDevices.light_on_bedroom,
    "light_off_bedroom":     IoTDevices.light_off_bedroom,
    "light_on_hall":         IoTDevices.light_on_hall,
    "light_off_hall":        IoTDevices.light_off_hall,
    "light_on_kitchen":      IoTDevices.light_on_kitchen,
    "light_off_kitchen":     IoTDevices.light_off_kitchen,
    # TV
    "tv_on_hall":            IoTDevices.tv_on_hall,
    "tv_off_hall":           IoTDevices.tv_off_hall,
    "tv_on_bedroom":         IoTDevices.tv_on_bedroom,
    "tv_off_bedroom":        IoTDevices.tv_off_bedroom,
    # Fan
    "fan_on_bedroom":        IoTDevices.fan_on_bedroom,
    "fan_off_bedroom":       IoTDevices.fan_off_bedroom,
    "fan_on_hall":           IoTDevices.fan_on_hall,
    "fan_off_hall":          IoTDevices.fan_off_hall,
    # AC / Thermostat
    "ac_set_temp_bedroom":   IoTDevices.ac_set_temp_bedroom,
    "thermostat_set":        IoTDevices.thermostat_set,
    # Doors
    "door_lock_front":       IoTDevices.door_lock_front,
    "door_unlock_front":     IoTDevices.door_unlock_front,
    "door_lock_back":        IoTDevices.door_lock_back,
    "door_unlock_back":      IoTDevices.door_unlock_back,
}

TOOL_SCHEMA = "\n".join([
    "[LIGHTS]",
    "light_on_bedroom()       - turn on bedroom light",
    "light_off_bedroom()      - turn off bedroom light",
    "light_on_hall()          - turn on hall light",
    "light_off_hall()         - turn off hall light",
    "light_on_kitchen()       - turn on kitchen light",
    "light_off_kitchen()      - turn off kitchen light",
    "[TV]",
    "tv_on_hall()             - turn on hall TV",
    "tv_off_hall()            - turn off hall TV",
    "tv_on_bedroom()          - turn on bedroom TV",
    "tv_off_bedroom()         - turn off bedroom TV",
    "[FAN]",
    "fan_on_bedroom()         - turn on bedroom fan",
    "fan_off_bedroom()        - turn off bedroom fan",
    "fan_on_hall()            - turn on hall fan",
    "fan_off_hall()           - turn off hall fan",
    "[AC & THERMOSTAT]",
    "ac_set_temp_bedroom(temp) - set bedroom AC temperature (default 22°C)",
    "thermostat_set(temp)    - set whole-house thermostat (default 24°C)",
    "[DOORS]",
    "door_lock_front()        - lock front door",
    "door_unlock_front()      - unlock front door",
    "door_lock_back()         - lock back door",
    "door_unlock_back()       - unlock back door",
])

# ─────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query:              str
    refined_query:      str

    # embedding loop
    relevant_nodes:     List[str]
    embed_feedback:     Optional[str]   # orch critique → embedding retry
    embed_retries:      int

    # tool caller loop
    tool_plan:          Optional[str]   # JSON string
    tool_feedback:      Optional[str]   # orch critique → tool retry
    tool_retries:       int

    # executor
    action_results:     List[str]

    # outer loop
    outer_retries:      int
    satisfied:          bool

    final_response:     Optional[str]
    # query classification
    is_action:          bool
    # chat history for context window
    chat_history:       List[str]
    # orchestrator clarification request
    pending_question:   Optional[str]
    pending_question_asked: bool

MAX_EMBED_RETRIES = 2
MAX_TOOL_RETRIES  = 2
MAX_OUTER_RETRIES = 2

# ─────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────

orch_llm = OllamaLLM(model="llama3.2:3b",      temperature=0.1)
tool_llm = OllamaLLM(model="qwen2.5-coder:7b", temperature=0.0)

# ─────────────────────────────────────────────────────────
# ChromaDB
# ─────────────────────────────────────────────────────────

def init_chroma() -> chromadb.Collection:
    client = chromadb.Client()
    try:
        ef = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text",
        )
    except Exception:
        ef = embedding_functions.DefaultEmbeddingFunction()

    col = client.get_or_create_collection("iot_devices", embedding_function=ef)

    docs = [
        # Bedroom Lights
        {"id": "light_on_bedroom",       "doc": "turn on bedroom light illuminate bedroom brighten bedroom room switch on lights"},
        {"id": "light_off_bedroom",      "doc": "turn off bedroom light darken bedroom switch off lights dim bedroom"},
        # Hall Lights
        {"id": "light_on_hall",          "doc": "turn on hall light illuminate hall brighten hallway living room switch on lights"},
        {"id": "light_off_hall",         "doc": "turn off hall light darken hall switch off lights dim living room"},
        # Kitchen Lights
        {"id": "light_on_kitchen",       "doc": "turn on kitchen light illuminate kitchen brighten kitchen switch on lights"},
        {"id": "light_off_kitchen",      "doc": "turn off kitchen light darken kitchen switch off lights dim kitchen"},
        # Hall TV
        {"id": "tv_on_hall",             "doc": "turn on hall TV television living room watch start TV watch television"},
        {"id": "tv_off_hall",            "doc": "turn off hall TV television living room stop watching end TV"},
        # Bedroom TV
        {"id": "tv_on_bedroom",          "doc": "turn on bedroom TV television bedroom watch start bedroom TV"},
        {"id": "tv_off_bedroom",         "doc": "turn off bedroom TV television bedroom stop watching end bedroom TV"},
        # Bedroom Fan
        {"id": "fan_on_bedroom",         "doc": "turn on bedroom fan start fan airflow ventilate cool bedroom"},
        {"id": "fan_off_bedroom",        "doc": "turn off bedroom fan stop fan disable fan bedroom"},
        # Hall Fan
        {"id": "fan_on_hall",            "doc": "turn on hall fan start fan airflow ventilate cool living room hall"},
        {"id": "fan_off_hall",           "doc": "turn off hall fan stop fan disable fan hall living room"},
        # AC & Thermostat
        {"id": "ac_set_temp_bedroom",    "doc": "set bedroom AC temperature air conditioner cooling bedroom degrees cold"},
        {"id": "thermostat_set",         "doc": "set thermostat home temperature heating cooling control whole house"},
        # Front Door
        {"id": "door_lock_front",        "doc": "lock front door secure entrance deadbolt front door"},
        {"id": "door_unlock_front",      "doc": "unlock front door open entrance allow entry front door"},
        # Back Door
        {"id": "door_lock_back",         "doc": "lock back door secure back entrance deadbolt back door"},
        {"id": "door_unlock_back",       "doc": "unlock back door open back entrance allow entry back door"},
    ]

    existing = col.get()["ids"]
    new = [d for d in docs if d["id"] not in existing]
    if new:
        col.add(ids=[d["id"] for d in new], documents=[d["doc"] for d in new])
        print(f"[ChromaDB] Indexed {len(new)} IoT nodes.\n")

    return col

# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _extract_json(raw: str) -> str:
    """Pull first [...] block from LLM output."""
    s = raw.find("[")
    e = raw.rfind("]") + 1
    if s == -1:
        return "[]"
    try:
        blob = raw[s:e]
        json.loads(blob)
        return blob
    except Exception:
        return "[]"


def _append_chat_message(history: List[str], speaker: str, text: str, max_history: int = 10) -> List[str]:
    return (history + [f"{speaker}: {text}"])[-max_history:]


def _recent_chat_context(history: List[str], size: int = 5) -> str:
    return "\n".join(history[-size:])


def _classify_query(query: str) -> tuple[bool, Optional[str]]:
    """
    Classify if query is an ACTION (control command) or INFO (question/listing).
    Returns: (is_action, direct_response_if_info)
    """
    q = query.lower().strip()
    info_keywords = ["what", "which", "list", "show", "tell", "available", "help", "status", "get", "how many", "is there", "where"]
    action_keywords = ["turn", "switch", "set", "lock", "unlock", "on", "off", "dim", "brighten", "enable", "disable", "activate", "deactivate"]

    # If it's a question mark, treat as info
    if q.endswith("?"):
        return False, None

    # Specific quick responses
    if "nodes in the bedroom" in q or "devices in the bedroom" in q or ("bedroom" in q and "list" in q):
        return False, _get_bedroom_devices()
    if "all devices" in q or "available devices" in q or "list all devices" in q:
        return False, _get_all_devices()

    if any(kw in q for kw in info_keywords):
        return False, None
    if any(kw in q for kw in action_keywords):
        return True, None
    # Default to non-action (safer)
    return False, None


def _get_bedroom_devices() -> str:
    """Get all bedroom devices."""
    bedroom_devices = [k for k in DEVICE_MAP.keys() if "bedroom" in k]
    if not bedroom_devices:
        return "No bedroom devices registered."
    device_list = "\n  • ".join(bedroom_devices)
    return f"Bedroom devices:\n  • {device_list}"


def _get_all_devices() -> str:
    """Get all available devices grouped by category."""
    categories = {
        "Lights": [k for k in DEVICE_MAP.keys() if "light" in k],
        "TV": [k for k in DEVICE_MAP.keys() if "tv" in k],
        "Fans": [k for k in DEVICE_MAP.keys() if "fan" in k],
        "AC/Thermostat": [k for k in DEVICE_MAP.keys() if "ac" in k or "thermostat" in k],
        "Doors": [k for k in DEVICE_MAP.keys() if "door" in k],
    }
    result = "Available devices:\n"
    for category, devices in categories.items():
        if devices:
            result += f"\n{category}:\n"
            for device in devices:
                result += f"  • {device}\n"
    return result

# ─────────────────────────────────────────────────────────
# Graph Nodes
# ─────────────────────────────────────────────────────────

# ── 1. Orchestrator (entry + outer supervisor) ──────────

def orchestrator_node(state: AgentState) -> AgentState:
    print(f"\n{'═'*55}")
    print(f"[Orch] Query: {state['query']}  (outer retry #{state['outer_retries']})")
    # First, classify the raw user query to avoid rewriting casual chat into actions
    is_action, direct_resp = _classify_query(state['query'])
    print(f"[Orch] initial_is_action={is_action} direct_response={'YES' if direct_resp else 'NO'}")

    # If it's not an action, skip the rewriter and return early; chat frontend already handles it.
    if not is_action:
        return {
            **state,
            "refined_query":   state['query'],
            "embed_feedback":  None,
            "tool_feedback":   None,
            "embed_retries":   0,
            "tool_retries":    0,
            "action_results":  [],
            "satisfied":       False,
            "is_action":       False,
        }

    # Otherwise, refine intent into a device-action using the LLM (only for actions)
    prompt = f"""You are a smart home orchestrator.
User query: "{state['query']}"
Rephrase as a clear, concise device-action intent. One sentence only."""
    refined = orch_llm.invoke(prompt).strip()
    print(f"[Orch] Refined: {refined}")

    # mark as action and continue
    return {
        **state,
        "refined_query":   refined,
        "embed_feedback":  None,
        "tool_feedback":   None,
        "embed_retries":   0,
        "tool_retries":    0,
        "action_results":  [],
        "satisfied":       False,
        "is_action":       True,
    }

# ── 2. Embedding AI ──────────────────────────────────────

def embedding_node(state: AgentState, col: chromadb.Collection) -> AgentState:
    feedback_ctx = f" (feedback: {state['embed_feedback']})" if state["embed_feedback"] else ""
    print(f"[EmbedAI] Searching{feedback_ctx} → '{state['refined_query']}'")

    # Augment query with feedback hint if retrying
    search_query = state["refined_query"]
    if state["embed_feedback"]:
        search_query += " " + state["embed_feedback"]

    results = col.query(query_texts=[search_query], n_results=4)
    nodes   = results["ids"][0] if results["ids"] else []
    print(f"[EmbedAI] Found nodes: {nodes}")
    return {**state, "relevant_nodes": nodes}

# ── 3. Orch reviews embedding results ───────────────────

def orch_review_embed(state: AgentState) -> AgentState:
    # Pragmatic check: if any relevant nodes were found, approve. Otherwise request clarification.
    if not state["relevant_nodes"]:
        question = "I couldn't identify which device you want to control. Can you please specify the room or device, such as bedroom light or front door lock?"
        print(f"[Orch→Embed] CLARIFY: {question}")
        return {**state, "embed_feedback": None, "embed_retries": 0, "pending_question": question, "pending_question_asked": False, "is_action": False}
    
    print(f"[Orch→Embed] APPROVE: Found {len(state['relevant_nodes'])} candidate device(s)")
    return {**state, "embed_feedback": None}  # approved

# ── 4. Tool Caller AI ────────────────────────────────────

def tool_caller_node(state: AgentState) -> AgentState:
    feedback_ctx = f" (fix: {state['tool_feedback']})" if state["tool_feedback"] else ""
    nodes_str    = ", ".join(state["relevant_nodes"])
    print(f"[ToolAI] Planning{feedback_ctx} for nodes: {nodes_str}")

    feedback_hint = f"\nPrevious plan was rejected: {state['tool_feedback']}. Fix it." \
                    if state["tool_feedback"] else ""

    prompt = f"""You are a smart home tool caller. Device functions are room-specific (no room parameter needed).
User intent: "{state['refined_query']}"
Candidate device functions: {nodes_str}

Available tools (room-specific - call as-is, no room parameter):
{TOOL_SCHEMA}
{feedback_hint}

Output ONLY a JSON array of actions. Format:
[{{"function": "func_name"}}, {{"function": "another_func", "args": {{"temp": 20}}}}]
Most functions take no arguments. Only thermostat/ac functions take temp parameter. No explanation."""

    raw  = tool_llm.invoke(prompt).strip()
    plan = _extract_json(raw)
    print(f"[ToolAI] Plan: {plan}")
    return {**state, "tool_plan": plan}

# ── 5. Orch reviews tool plan ────────────────────────────

def orch_review_tool(state: AgentState) -> AgentState:
    # Deterministic validation: check if all functions in plan exist in DEVICE_MAP
    try:
        plan = json.loads(state["tool_plan"] or "[]")
    except Exception:
        plan = []
    
    if not plan:
        reason = "Tool plan is empty or invalid JSON."
        print(f"[Orch→Tool] RETRY: {reason}")
        return {**state, "tool_feedback": reason, "tool_retries": state["tool_retries"] + 1}
    
    invalid_funcs = [a.get("function", "") for a in plan if a.get("function", "") not in DEVICE_MAP]
    if invalid_funcs:
        question = "I couldn't map your request to a supported device. Please tell me the room and device clearly, for example 'turn on bedroom light' or 'lock front door'."
        print(f"[Orch→Tool] CLARIFY: {question}")
        return {**state, "tool_feedback": None, "tool_retries": 0, "pending_question": question, "pending_question_asked": False, "is_action": False}
    
    print(f"[Orch→Tool] APPROVE: Plan has {len(plan)} valid action(s)")
    return {**state, "tool_feedback": None}  # approved

# ── 6. Executor ──────────────────────────────────────────

def executor_node(state: AgentState) -> AgentState:
    print(f"[Executor] Running plan...")
    results = []
    try:
        plan = json.loads(state["tool_plan"] or "[]")
    except Exception:
        plan = []

    if not plan:
        return {**state, "action_results": ["No valid actions found."]}

    for action in plan:
        fn   = action.get("function", "")
        args = action.get("args", {})
        if fn in DEVICE_MAP:
            try:
                res = DEVICE_MAP[fn](**args)
                results.append(res)
            except Exception as ex:
                results.append(f"Error: {fn} → {ex}")
        else:
            results.append(f"Unknown function: {fn}")

    print(f"[Executor] Results: {results}")
    return {**state, "action_results": results}

# ── 7. Orch reviews execution results ───────────────────

def orch_review_results(state: AgentState) -> AgentState:
    # Pragmatic check: if results contain no errors, mark as satisfied
    has_errors = any("Error" in r or "Unknown" in r for r in state["action_results"])
    
    if has_errors:
        print(f"[Orch→Results] RETRY: Execution had errors  (outer retry {state['outer_retries']}/{MAX_OUTER_RETRIES})")
        if state["outer_retries"] < MAX_OUTER_RETRIES:
            return {**state, "satisfied": False, "outer_retries": state["outer_retries"] + 1}
    
    print(f"[Orch→Results] APPROVE: Actions executed successfully  (outer retry {state['outer_retries']}/{MAX_OUTER_RETRIES})")
    return {**state, "satisfied": True}

# ── 8. Responder ─────────────────────────────────────────

def responder_node(state: AgentState) -> AgentState:
    results_str = "; ".join(state["action_results"]) or "No actions were taken."
    prompt = f"""Smart home task completed.
Actions done: {results_str}
Write one friendly sentence confirming to the user what was done."""
    reply = orch_llm.invoke(prompt).strip()
    print(f"\n[Response] ✅ {reply}")
    return {**state, "final_response": reply}


def route_after_orchestrator(state: AgentState) -> str:
    if state.get("pending_question"):
        return "clarify"
    if state.get("is_action"):
        return "proceed_to_embed"
    return "proceed_to_chat"


def llm_frontend_node(state: AgentState) -> AgentState:
    """Conversational front for users. Replies to info queries, forwards actions to orchestrator."""
    print("[LLMFront] Conversational front received user input")
    is_action, direct_resp = _classify_query(state.get("query", ""))
    current_history = state.get("chat_history", [])

    if state.get("pending_question") and not state.get("pending_question_asked"):
        current_history = _append_chat_message(current_history, "User", state["query"])
        reply = state["pending_question"]
        print(f"[LLMFront] Asking clarification: {reply}")
        updated_history = _append_chat_message(current_history, "Assistant", reply)
        return {**state, "final_response": reply, "satisfied": True, "chat_history": updated_history, "pending_question": state["pending_question"], "pending_question_asked": True, "is_action": False}

    if state.get("pending_question") and state.get("pending_question_asked"):
        current_history = _append_chat_message(current_history, "User", state["query"])
        state = {**state, "pending_question": None, "pending_question_asked": False, "chat_history": current_history}
        is_action, direct_resp = _classify_query(state.get("query", ""))
        print(f"[LLMFront] is_action={is_action} direct_response={'YES' if direct_resp else 'NO'}")
    else:
        current_history = _append_chat_message(current_history, "User", state["query"])
        print(f"[LLMFront] is_action={is_action} direct_response={'YES' if direct_resp else 'NO'}")

    if not is_action:
        if direct_resp:
            reply = direct_resp
        else:
            prompt = f"You are a friendly, simple smart-home assistant for non-technical users. Use the conversation history to respond clearly, but do not perform any device actions.\n\nConversation history:\n{_recent_chat_context(current_history, 5)}\n\nReply as Assistant in one friendly sentence."
            reply = orch_llm.invoke(prompt).strip()
        print(f"[LLMFront] Replying to user: {reply}")
        updated_history = _append_chat_message(current_history, "Assistant", reply)
        return {**state, "final_response": reply, "satisfied": True, "is_action": False, "chat_history": updated_history, "pending_question": None, "pending_question_asked": False}

    # For actions, forward to orchestrator (do not execute here)
    print("[LLMFront] Forwarding action to orchestrator")
    return {**state, "is_action": True, "chat_history": current_history, "pending_question": None, "pending_question_asked": False}


def route_after_frontend(state: AgentState) -> str:
    if state.get("is_action"):
        return "to_orch"
    return "frontend_end"

# ─────────────────────────────────────────────────────────
# Conditional Edges (routing logic)
# ─────────────────────────────────────────────────────────

def route_after_embed_review(state: AgentState) -> str:
    if state.get("pending_question"):
        return "clarify"
    if state["embed_feedback"] and state["embed_retries"] <= MAX_EMBED_RETRIES:
        print(f"[Router] Embed retry #{state['embed_retries']}")
        return "retry_embed"
    return "proceed_to_tool"

def route_after_tool_review(state: AgentState) -> str:
    if state.get("pending_question"):
        return "clarify"
    if state["tool_feedback"] and state["tool_retries"] <= MAX_TOOL_RETRIES:
        print(f"[Router] Tool retry #{state['tool_retries']}")
        return "retry_tool"
    return "proceed_to_exec"

def route_after_results(state: AgentState) -> str:
    if not state["satisfied"]:
        print(f"[Router] Outer retry — re-running full pipeline")
        return "outer_retry"
    return "respond"

# ─────────────────────────────────────────────────────────
# Build Graph
# ─────────────────────────────────────────────────────────

def build_graph(col: chromadb.Collection):
    def _embed(s): return embedding_node(s, col)

    g = StateGraph(AgentState)

    g.add_node("orchestrator",        orchestrator_node)
    g.add_node("llm_frontend",        llm_frontend_node)
    g.add_node("embedding_ai",        _embed)
    g.add_node("orch_review_embed",   orch_review_embed)
    g.add_node("tool_caller",         tool_caller_node)
    g.add_node("orch_review_tool",    orch_review_tool)
    g.add_node("executor",            executor_node)
    g.add_node("orch_review_results", orch_review_results)
    g.add_node("responder",           responder_node)

    # Entry: conversational front-end handles casual conversation and forwards actions
    g.set_entry_point("llm_frontend")
    g.add_conditional_edges(
        "llm_frontend",
        route_after_frontend,
        {"to_orch": "orchestrator", "frontend_end": END},
    )

    # Route after orchestrator: action -> embedding flow, clarification -> chat frontend
    g.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {"proceed_to_embed": "embedding_ai", "proceed_to_chat": "llm_frontend", "clarify": "llm_frontend"},
    )

    # Inner embedding loop
    g.add_edge("embedding_ai", "orch_review_embed")
    g.add_conditional_edges(
        "orch_review_embed",
        route_after_embed_review,
        {"retry_embed": "embedding_ai", "proceed_to_tool": "tool_caller", "clarify": "llm_frontend"},
    )

    # Inner tool loop
    g.add_edge("tool_caller", "orch_review_tool")
    g.add_conditional_edges(
        "orch_review_tool",
        route_after_tool_review,
        {"retry_tool": "tool_caller", "proceed_to_exec": "executor", "clarify": "llm_frontend"},
    )

    # Outer loop
    g.add_edge("executor", "orch_review_results")
    g.add_conditional_edges(
        "orch_review_results",
        route_after_results,
        {"outer_retry": "embedding_ai", "respond": "responder"},
    )

    g.add_edge("responder", END)

    return g.compile()

# ─────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────

def run_query(app, query: str, prev_state: Optional[AgentState] = None):
    if prev_state is None:
        init_state: AgentState = {
            "query":           query,
            "refined_query":   "",
            "relevant_nodes":  [],
            "embed_feedback":  None,
            "embed_retries":   0,
            "tool_plan":       None,
            "tool_feedback":   None,
            "tool_retries":    0,
            "action_results":  [],
            "outer_retries":   0,
            "satisfied":       False,
            "final_response":  None,
            "is_action":       False,
            "chat_history":    [],
            "pending_question": None,
            "pending_question_asked": False,
        }
    else:
        init_state = {
            **prev_state,
            "query":           query,
            "refined_query":   "",
            "embed_feedback":  None,
            "embed_retries":   0,
            "tool_plan":       None,
            "tool_feedback":   None,
            "tool_retries":    0,
            "action_results":  [],
            "outer_retries":   0,
            "satisfied":       False,
            "final_response": None,
        }
    return app.invoke(init_state)


if __name__ == "__main__":
    print("╔" + "═"*53 + "╗")
    print("║" + " "*53 + "║")
    print("║" + "  🏠 Smart Home Voice & Text Assistant 🏠".center(53) + "║")
    print("║" + " "*53 + "║")
    print("║" + "  Type what you want to do in plain English".center(53) + "║")
    print("║" + "  Example: 'Turn on bedroom light'".center(53) + "║")
    print("║" + "  Type 'exit' or 'quit' to stop".center(53) + "║")
    print("║" + " "*53 + "║")
    print("╚" + "═"*53 + "╝")
    print()

    col = init_chroma()
    app = build_graph(col)

    command_count = 0
    current_state: Optional[AgentState] = None
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            # Handle exit commands
            if user_input.lower() in ["exit", "quit", "bye", "goodbye", "q"]:
                print("\n" + "="*55)
                print("👋 Thank you for using Smart Home Assistant!")
                print("   Goodbye! 🏠")
                print("="*55)
                break
            
            # Skip empty inputs
            if not user_input:
                print("ℹ️  Please type a command (e.g., 'Turn on bedroom light')")
                continue
            
            command_count += 1
            print("\n" + "─"*55)
            print(f"[Command #{command_count}] Processing: {user_input}")
            print("─"*55)
            
            # Run the query through the agent and preserve session history
            current_state = run_query(app, user_input, current_state)
            
        except KeyboardInterrupt:
            print("\n\n" + "="*55)
            print("⚠️  Session interrupted by user")
            print("   Goodbye! 🏠")
            print("="*55)
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("ℹ️  Please try rephrasing your command.")
            continue