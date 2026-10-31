import streamlit as st
import httpx

API_URL = "http://localhost:8000"
st.set_page_config(page_title="TrustLayer | Agentic Policy Verification", layout="wide")

st.title("🛡️ TrustLayer — Agentic AI Verification Layer")
st.caption("Track: Cybersecurity & AI Governance | Team YuriNation")

with st.sidebar:
    st.header("📄 Knowledge Base")
    if st.button("Ingest Policy PDFs (./policies)"):
        res = httpx.post(f"{API_URL}/ingest", timeout=60.0).json()
        st.success(f"Indexed {res['chunks_indexed']} chunks into ChromaDB!")

col_q, col_btn = st.columns([5, 1])
with col_q:
    question = st.text_input("Ask a policy question...", placeholder="e.g., Does our gold health policy cover pre-existing dental surgery in month 2?")
with col_btn:
    st.write("")
    st.write("")
    send = st.button("Send", type="primary", use_container_width=True)

if send and question:
    with st.spinner("Running Planner → ChromaDB Retriever → Parallel LLMs → Auditor..."):
        res = httpx.post(f"{API_URL}/ask", json={"question": question}, timeout=120.0).json()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🤖 Agent A Answer (Llama-3.3-70B)")
        st.info(res["agent_a_answer"])
    with col_b:
        st.subheader("🤖 Agent B Answer (Gemini 2.5 Flash)")
        st.info(res["agent_b_answer"])

    st.subheader(f"🔍 AUDITOR (Retries used: {res['retry_count']}/2)")
    for idx, c in enumerate(res["auditor_claims"], 1):
        status = c["status"]
        badge = "✅ Supported" if status == "Supported" else f"❌ {status.upper()} (retry {res['retry_count']}/2)"
        st.markdown(f"**Claim {idx}:** {c['claim']} — **{badge}** *(Citation: `{c['citation']}`)*")
        st.caption(f"Reasoning: {c['reasoning']}")

    st.divider()

    verdict = res["verdict"]
    if verdict == "Release":
        st.success(f"**Verdict: Verified & Released** | Verification Status: Supported\n\n{res['final_answer']}")
    else:
        st.error(f"**Verdict: Escalated to Human Reviewer** | Agents Agree: {res['agents_agree']} | Retries: {res['retry_count']}/2")
        st.warning(f"**Reason / Summary for Human Agent:** {res['final_answer']}")
        st.button("🚨 Escalate to Human Queue (Sent with both drafts & evidence)", disabled=True)

    sources = sorted(list({f"{c['source_file']} p.{c['page_number']}" for c in res["retrieved_chunks"]}))
    st.markdown(f"**Sources:** {', '.join(sources) if sources else 'None'}")

    with st.expander("📜 View SQLite Compliance Audit Log"):
        st.json(res["audit_logs"])
