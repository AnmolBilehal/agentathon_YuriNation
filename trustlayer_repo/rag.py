import os
import chromadb
from chromadb.utils import embedding_functions
import pymupdf4llm

CHROMA_PATH = "./chroma_store"
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-base-en-v1.5"
)
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name="policy_docs",
    embedding_function=embedding_fn
)

def ingest_pdfs(pdf_Folder: str = "./policies"):
    os.makedirs(pdf_Folder, exist_ok=True)
    ids, docs, metadatas = [], [], []
    for fname in os.listdir(pdf_Folder):
        if not fname.endswith(".pdf"):
            continue
        fpath = os.path.join(pdf_Folder, fname)
        page_chunks = pymupdf4llm.to_markdown(fpath, page_chunks=True)
        for idx, page in enumerate(page_chunks):
            page_num = page.get("metadata", {}).get("page", idx + 1)
            text = page.get("text", "").strip()
            if not text:
                continue
            for i in range(0, len(text), 650):
                chunk_text = text[i:i + 800]
                ids.append(f"{fname}_p{page_num}_{i}")
                docs.append(chunk_text)
                metadatas.append({"source_file": fname, "page_number": page_num})
    if ids:
        collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
    return len(ids)

def retrieve_evidence(sub_queries: list[str], retry_count: int) -> list[dict]:
    top_k = 3 * (retry_count + 1)
    results = collection.query(query_texts=sub_queries, n_results=top_k)
    seen = set()
    chunks = []
    for doc_list, meta_list in zip(results.get("documents", []), results.get("metadatas", [])):
        for doc, meta in zip(doc_list, meta_list):
            key = f"{meta['source_file']}_p{meta['page_number']}_{doc[:40]}"
            if key not in seen:
                seen.add(key)
                chunks.append({
                    "source_file": meta["source_file"],
                    "page_number": meta["page_number"],
                    "text": doc
                })
    return chunks
