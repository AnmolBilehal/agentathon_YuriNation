import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from database import init_db, get_logs
from rag import ingest_pdfs
from workflow import trustlayer_graph

app = FastAPI(title="TrustLayer API")
init_db()

class QueryRequest(BaseModel):
    question: str

@app.post("/ingest")
def ingest_endpoint():
    count = ingest_pdfs("./policies")
    return {"status": "ok", "chunks_indexed": count}

@app.post("/ask")
async def ask_policy(req: QueryRequest):
    session_id = str(uuid.uuid4())[:8]
    initial_state = {
        "session_id": session_id,
        "question": req.question,
        "sub_queries": [],
        "retrieved_chunks": [],
        "agent_a_answer": "",
        "agent_b_answer": "",
        "auditor_claims": [],
        "agents_agree": True,
        "retry_count": 0,
        "verdict": "Pending",
        "final_answer": ""
    }
    final_state = await trustlayer_graph.ainvoke(initial_state)
    final_state["audit_logs"] = get_logs(session_id)
    return final_state

@app.get("/audit-logs")
def fetch_logs():
    return get_logs()
