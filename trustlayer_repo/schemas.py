from typing import List, Literal, TypedDict
from pydantic import BaseModel, Field

class SourceChunk(BaseModel):
    source_file: str
    page_number: int
    text: str

class ClaimVerification(BaseModel):
    claim: str = Field(description="Atomic factual claim extracted from the drafted answers")
    status: Literal["Supported", "Unsupported", "Contradicted"]
    citation: str = Field(description="Format: filename.pdf p.X or 'None'")
    reasoning: str

class AuditorResult(BaseModel):
    claims: List[ClaimVerification]
    agents_agree: bool = Field(description="True if Agent A and Agent B agree on core policy facts")
    verdict: Literal["Release", "Retry", "Escalate"]
    final_cited_answer: str = Field(description="Verified synthesized answer with inline citations if Release, else reason for Retry/Escalate")

class PlannerResult(BaseModel):
    sub_queries: List[str] = Field(description="Atomic search queries split from the user question")

class TrustLayerState(TypedDict):
    session_id: str
    question: str
    sub_queries: List[str]
    retrieved_chunks: List[dict]
    agent_a_answer: str
    agent_b_answer: str
    auditor_claims: List[dict]
    agents_agree: bool
    retry_count: int
    verdict: str
    final_answer: str
