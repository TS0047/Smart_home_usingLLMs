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
    @staticmethod
    def light_on(room: str = "living room"):
        print(f"  [IoT] 💡 Light ON  → {room}")
        return f"Light ON in {room}"

    @staticmethod
    def light_off(room: str = "living room"):
        print(f"  [IoT] 🌑 Light OFF → {room}")
        return f"Light OFF in {room}"

    @staticmethod
    def fan_on(room: str = "bedroom"):
        print(f"  [IoT] 🌀 Fan ON   → {room}")
        return f"Fan ON in {room}"

    @staticmethod
    def fan_off(room: str = "bedroom"):
        print(f"  [IoT] ⛔ Fan OFF  → {room}")
        return f"Fan OFF in {room}"

    @staticmethod
    def ac_set_temp(temp: int = 22, room: str = "bedroom"):
        print(f"  [IoT] ❄️  AC {temp}°C → {room}")
        return f"AC set to {temp}°C in {room}"

    @staticmethod
    def door_lock(door: str = "main"):
        print(f"  [IoT] 🔒 Lock     → {door} door")
        return f"{door} door locked"

    @staticmethod
    def door_unlock(door: str = "main"):
        print(f"  [IoT] 🔓 Unlock   → {door} door")
        return f"{door} door unlocked"

    @staticmethod
    def tv_on(room: str = "living room"):
        print(f"  [IoT] 📺 TV ON   → {room}")
        return f"TV ON in {room}"

    @staticmethod
    def tv_off(room: str = "living room"):
        print(f"  [IoT] 📺 TV OFF  → {room}")
        return f"TV OFF in {room}"

    @staticmethod
    def thermostat_set(temp: int = 24):
        print(f"  [IoT] 🌡️  Thermostat → {temp}°C")
        return f"Thermostat set to {temp}°C"


DEVICE_MAP = {
    "light_on":       IoTDevices.light_on,
    "light_off":      IoTDevices.light_off,
    "fan_on":         IoTDevices.fan_on,
    "fan_off":        IoTDevices.fan_off,
    "ac_set_temp":    IoTDevices.ac_set_temp,
    "door_lock":      IoTDevices.door_lock,
    "door_unlock":    IoTDevices.door_unlock,
    "tv_on":          IoTDevices.tv_on,
    "tv_off":         IoTDevices.tv_off,
    "thermostat_set": IoTDevices.thermostat_set,
}

TOOL_SCHEMA = "\n".join([
    "light_on(room)          - turn on light",
    "light_off(room)         - turn off light",
    "fan_on(room)            - turn on fan",
    "fan_off(room)           - turn off fan",
    "ac_set_temp(temp, room) - set AC temperature",
    "door_lock(door)         - lock a door",
    "door_unlock(door)       - unlock a door",
    "tv_on(room)             - turn on TV",
    "tv_off(room)            - turn off TV",
    "thermostat_set(temp)    - set thermostat",
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
        {"id": "light_on",       "doc": "turn on light illuminate brighten room switch on lights"},
        {"id": "light_off",      "doc": "turn off light darken room switch off lights dim"},
        {"id": "fan_on",         "doc": "turn on fan start fan airflow ventilate cool"},
        {"id": "fan_off",        "doc": "turn off fan stop fan disable fan"},
        {"id": "ac_set_temp",    "doc": "set AC air conditioner temperature cooling degrees cold"},
        {"id": "door_lock",      "doc": "lock door secure entrance deadbolt"},
        {"id": "door_unlock",    "doc": "unlock door open entrance allow entry"},
        {"id": "tv_on",          "doc": "turn on TV television watch start TV"},
        {"id": "tv_off",         "doc": "turn off TV television stop watching"},
        {"id": "thermostat_set", "doc": "set thermostat home temperature heating cooling control"},
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

def _orch_review(context: str, question: str) -> str:
    """Generic orchestrator review — returns APPROVE or RETRY:<reason>."""
    prompt = f"""{context}
Question: {question}
Reply with exactly one of:
  APPROVE
  RETRY:<one-sentence reason>
No other text."""
    return orch_llm.invoke(prompt).strip()

# ─────────────────────────────────────────────────────────
# Graph Nodes
# ─────────────────────────────────────────────────────────

# ── 1. Orchestrator (entry + outer supervisor) ──────────

def orchestrator_node(state: AgentState) -> AgentState:
    print(f"\n{'═'*55}")
    print(f"[Orch] Query: {state['query']}  (outer retry #{state['outer_retries']})")

    prompt = f"""You are a smart home orchestrator.
User query: "{state['query']}"
Rephrase as a clear, concise device-action intent. One sentence only."""
    refined = orch_llm.invoke(prompt).strip()
    print(f"[Orch] Refined: {refined}")

    return {
        **state,
        "refined_query":  refined,
        "embed_feedback": None,
        "tool_feedback":  None,
        "embed_retries":  0,
        "tool_retries":   0,
        "action_results": [],
        "satisfied":      False,
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
    ctx = f"""User intent: "{state['refined_query']}"
Embedding AI returned these IoT device candidates: {state['relevant_nodes']}
Available devices: {list(DEVICE_MAP.keys())}"""

    verdict = _orch_review(ctx, "Are these device candidates sufficient to fulfill the intent?")
    print(f"[Orch→Embed] {verdict}")

    if verdict.startswith("RETRY"):
        reason = verdict[6:].strip()
        return {**state, "embed_feedback": reason, "embed_retries": state["embed_retries"] + 1}
    return {**state, "embed_feedback": None}  # approved

# ── 4. Tool Caller AI ────────────────────────────────────

def tool_caller_node(state: AgentState) -> AgentState:
    feedback_ctx = f" (fix: {state['tool_feedback']})" if state["tool_feedback"] else ""
    nodes_str    = ", ".join(state["relevant_nodes"])
    print(f"[ToolAI] Planning{feedback_ctx} for nodes: {nodes_str}")

    feedback_hint = f"\nPrevious plan was rejected: {state['tool_feedback']}. Fix it." \
                    if state["tool_feedback"] else ""

    prompt = f"""You are a smart home tool caller.
User intent: "{state['refined_query']}"
Candidate device functions: {nodes_str}

Available tools:
{TOOL_SCHEMA}
{feedback_hint}

Output ONLY a JSON array of actions. Format:
[{{"function": "func_name", "args": {{"param": "value"}}}}]
Default room = "living room" if unspecified. No explanation."""

    raw  = tool_llm.invoke(prompt).strip()
    plan = _extract_json(raw)
    print(f"[ToolAI] Plan: {plan}")
    return {**state, "tool_plan": plan}

# ── 5. Orch reviews tool plan ────────────────────────────

def orch_review_tool(state: AgentState) -> AgentState:
    ctx = f"""User intent: "{state['refined_query']}"
Tool Caller produced this JSON action plan: {state['tool_plan']}
Available device functions: {list(DEVICE_MAP.keys())}"""

    verdict = _orch_review(ctx, "Is this action plan correct and complete for the user intent?")
    print(f"[Orch→Tool] {verdict}")

    if verdict.startswith("RETRY"):
        reason = verdict[6:].strip()
        return {**state, "tool_feedback": reason, "tool_retries": state["tool_retries"] + 1}
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
    ctx = f"""User intent: "{state['refined_query']}"
Actions executed: {state['action_results']}"""

    verdict = _orch_review(ctx, "Did these actions fully and correctly satisfy the user intent?")
    print(f"[Orch→Results] {verdict}  (outer retry {state['outer_retries']}/{MAX_OUTER_RETRIES})")

    if verdict.startswith("RETRY") and state["outer_retries"] < MAX_OUTER_RETRIES:
        return {**state, "satisfied": False, "outer_retries": state["outer_retries"] + 1}
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

# ─────────────────────────────────────────────────────────
# Conditional Edges (routing logic)
# ─────────────────────────────────────────────────────────

def route_after_embed_review(state: AgentState) -> str:
    if state["embed_feedback"] and state["embed_retries"] <= MAX_EMBED_RETRIES:
        print(f"[Router] Embed retry #{state['embed_retries']}")
        return "retry_embed"
    return "proceed_to_tool"

def route_after_tool_review(state: AgentState) -> str:
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
    g.add_node("embedding_ai",        _embed)
    g.add_node("orch_review_embed",   orch_review_embed)
    g.add_node("tool_caller",         tool_caller_node)
    g.add_node("orch_review_tool",    orch_review_tool)
    g.add_node("executor",            executor_node)
    g.add_node("orch_review_results", orch_review_results)
    g.add_node("responder",           responder_node)

    # Entry
    g.set_entry_point("orchestrator")
    g.add_edge("orchestrator", "embedding_ai")

    # Inner embedding loop
    g.add_edge("embedding_ai", "orch_review_embed")
    g.add_conditional_edges(
        "orch_review_embed",
        route_after_embed_review,
        {"retry_embed": "embedding_ai", "proceed_to_tool": "tool_caller"},
    )

    # Inner tool loop
    g.add_edge("tool_caller", "orch_review_tool")
    g.add_conditional_edges(
        "orch_review_tool",
        route_after_tool_review,
        {"retry_tool": "tool_caller", "proceed_to_exec": "executor"},
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

def run_query(app, query: str):
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
    }
    return app.invoke(init_state)


if __name__ == "__main__":
    print("Starting Smart Home Agent (Agentic Supervisor Mode)...\n")
    col = init_chroma()
    app = build_graph(col)

    queries = [
        "Turn on the lights in the bedroom",
        "I'm going to sleep — set AC to 20 degrees and lock the front door",
        "Switch off the TV and fan in the living room",
    ]

    for q in queries:
        run_query(app, q)
        print()