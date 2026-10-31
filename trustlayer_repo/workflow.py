from langgraph.graph import StateGraph, END
from schemas import TrustLayerState
from rag import retrieve_evidence
from agents import run_planner, run_parallel_drafters, run_auditor
from database import log_step

async def planner_node(state: TrustLayerState):
    sub_queries = await run_planner(state["question"])
    log_step(state["session_id"], "Planner", state["retry_count"], "Split query", {"sub_queries": sub_queries})
    return {"sub_queries": sub_queries}

async def retriever_node(state: TrustLayerState):
    chunks = retrieve_evidence(state["sub_queries"], state["retry_count"])
    log_step(state["session_id"], "Retriever", state["retry_count"], f"Fetched {len(chunks)} chunks", {"sources": [f"{c['source_file']} p.{c['page_number']}" for c in chunks]})
    return {"retrieved_chunks": chunks}

async def drafters_node(state: TrustLayerState):
    ans_a, ans_b = await run_parallel_drafters(state["question"], state["retrieved_chunks"])
    log_step(state["session_id"], "Parallel Drafters", state["retry_count"], "Drafted A & B", {"agent_a": ans_a, "agent_b": ans_b})
    return {"agent_a_answer": ans_a, "agent_b_answer": ans_b}

async def auditor_node(state: TrustLayerState):
    audit = await run_auditor(state["question"], state["agent_a_answer"], state["agent_b_answer"], state["retrieved_chunks"])
    claims_dump = [c.model_dump() for c in audit.claims]
    
    verdict = audit.verdict
    retry_count = state["retry_count"]
    
    if verdict == "Retry" and retry_count >= 2:
        verdict = "Escalate"
        
    log_step(state["session_id"], "Auditor", retry_count, verdict, {"claims": claims_dump, "agents_agree": audit.agents_agree})
    return {
        "auditor_claims": claims_dump,
        "agents_agree": audit.agents_agree,
        "verdict": verdict,
        "final_answer": audit.final_cited_answer
    }

async def retry_increment_node(state: TrustLayerState):
    new_count = state["retry_count"] + 1
    unsupported = [c["claim"] for c in state["auditor_claims"] if c["status"] != "Supported"]
    return {"retry_count": new_count, "sub_queries": state["sub_queries"] + unsupported[:2]}

def route_after_audit(state: TrustLayerState):
    if state["verdict"] == "Release":
        return "release"
    elif state["verdict"] == "Retry" and state["retry_count"] < 2:
        return "retry"
    return "escalate"

builder = StateGraph(TrustLayerState)
builder.add_node("planner", planner_node)
builder.add_node("retriever", retriever_node)
builder.add_node("drafters", drafters_node)
builder.add_node("auditor", auditor_node)
builder.add_node("retry_step", retry_increment_node)

builder.set_entry_point("planner")
builder.add_edge("planner", "retriever")
builder.add_edge("retriever", "drafters")
builder.add_edge("drafters", "auditor")
builder.add_conditional_edges(
    "auditor",
    route_after_audit,
    {
        "release": END,
        "retry": "retry_step",
        "escalate": END
    }
)
builder.add_edge("retry_step", "retriever")

trustlayer_graph = builder.compile()
