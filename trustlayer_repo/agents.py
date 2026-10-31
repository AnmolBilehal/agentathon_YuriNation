import os
import json
import asyncio
from dotenv import load_dotenv
from groq import AsyncGroq
from google import genai
from google.genai import types
from schemas import PlannerResult, AuditorResult

load_dotenv()
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        [f"[{c['source_file']} p.{c['page_number']}]: {c['text']}" for c in chunks]
    )

async def run_planner(question: str) -> list[str]:
    resp = await groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Split the policy question into 1 to 3 concise search queries. Return JSON matching {'sub_queries': ['...']}."},
            {"role": "user", "content": question}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    data = json.loads(resp.choices[0].message.content)
    return data.get("sub_queries", [question])

async def run_agent_a(question: str, chunks: list[dict]) -> str:
    context = format_context(chunks)
    resp = await groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are Policy Agent A. Answer strictly using the provided policy excerpts. Cite every claim inline like [policy.pdf p.4]. If evidence is missing, state that clearly."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.1
    )
    return resp.choices[0].message.content

async def run_agent_b(question: str, chunks: list[dict]) -> str:
    context = format_context(chunks)
    prompt = f"You are Policy Agent B. Answer strictly from the policy excerpts below and cite every claim inline like [policy.pdf p.4]. If unsupported, state that clearly.\n\nContext:\n{context}\n\nQuestion: {question}"
    resp = await asyncio.to_thread(
        gemini_client.models.generate_content,
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1)
    )
    return resp.text

async def run_parallel_drafters(question: str, chunks: list[dict]) -> tuple[str, str]:
    return await asyncio.gather(
        run_agent_a(question, chunks),
        run_agent_b(question, chunks)
    )

async def run_auditor(question: str, agent_a: str, agent_b: str, chunks: list[dict]) -> AuditorResult:
    context = format_context(chunks)
    prompt = f"""You are the TrustLayer Compliance Auditor.
1. Break the claims made by Agent A and Agent B into atomic claims.
2. Verify each claim strictly against the Source Evidence below.
   - 'Supported': Explicitly backed by a source chunk.
   - 'Unsupported': Not found in the source chunks.
   - 'Contradicted': Directly conflicts with source chunks or the other agent.
3. Check if Agent A and Agent B agree on the core policy answer.
4. Remember: Agent agreement alone does NOT guarantee correctness; source evidence determines verification.
5. Set verdict:
   - 'Release' ONLY if all material claims are 'Supported' by source evidence and agents agree.
   - 'Retry' if any claim is 'Unsupported' due to missing evidence.
   - 'Escalate' if claims are 'Contradicted' or agents directly disagree on policy rules.

Question: {question}
Agent A Answer: {agent_a}
Agent B Answer: {agent_b}
Source Evidence:
{context}"""

    resp = await asyncio.to_thread(
        gemini_client.models.generate_content,
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AuditorResult,
            temperature=0.0
        )
    )
    return AuditorResult.model_validate_json(resp.text)
