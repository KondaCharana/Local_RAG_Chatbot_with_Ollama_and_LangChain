# api_groq.py  (example)
import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel

# import your existing RAG helpers
from your_rag_module import get_or_create_coe_vector_db, COE_DOCS_SOURCE_DIR, COE_AI_DB_DIR, load_coe_documents
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

# ---------- CONFIG (env vars) ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.ai/v1/models/llama3.1/generate")  # set to correct endpoint
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama3.1")

# ---------- FastAPI ----------
app = FastAPI(title="CoE RAG using Groq")

# ---------- Load vector DB & retriever (same as before) ----------
vector_db = get_or_create_coe_vector_db(COE_DOCS_SOURCE_DIR, COE_AI_DB_DIR)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# ---------- Prompt template (same idea you used earlier) ----------
rag_prompt_template = ChatPromptTemplate.from_template("""
You are an AI assistant for the Center of Excellence AI Team.
Use ONLY the provided context below. If you cannot find the answer, say you don't have enough info.

Context:
{context}

Question:
{question}
""")

# ---------- Groq wrapper ----------
def call_groq(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    """
    Generic HTTP call to Groq model endpoint.
    Set GROQ_API_URL and GROQ_API_KEY in env.
    The exact request/response shape may differ by Groq API version — adjust if needed.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in environment")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL_NAME,
        # Many hosted LLMs accept a "prompt" or "inputs" field — adjust to Groq expected shape.
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # add other options Groq supports if needed
    }

    resp = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # The exact path to generated text depends on Groq API response.
    # Common shapes:
    #  - {"generated_text": "..." }
    #  - {"choices":[{"text":"..."}]}
    # Adjust accordingly. Below we try a few common keys:
    if isinstance(data, dict):
        if "generated_text" in data:
            return data["generated_text"]
        if "text" in data:
            return data["text"]
        if "choices" in data and len(data["choices"]) > 0 and "text" in data["choices"][0]:
            return data["choices"][0]["text"]
    # fallback: stringify
    return str(data)

# ---------- Pydantic model for /ask ----------
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list = []

# ---------- /ask endpoint (retriever + prompt build + call Groq) ----------
@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    # 1) Retrieve top-k context chunks (List[Document])
    docs = retriever.get_relevant_documents(req.question)  # many retrievers expose similarity_search or get_relevant_documents
    # If your retriever object has a different method name, use that (e.g., similarity_search)
    # convert docs into a single context string (you may want to include metadata like source/page)
    context_texts = []
    for d in docs:
        meta = d.metadata if hasattr(d, "metadata") else {}
        source_info = f"source:{meta.get('source','unknown')}, page:{meta.get('page','?')}"
        context_texts.append(f"[{source_info}] {d.page_content}")

    context_combined = "\n\n".join(context_texts)

    # 2) Build final prompt using your template
    prompt = rag_prompt_template.format_prompt(context=context_combined, question=req.question).to_string()

    # 3) Call Groq API
    try:
        answer_text = call_groq(prompt, max_tokens=512, temperature=0.0)
    except Exception as e:
        raise RuntimeError(f"Groq call failed: {e}")

    # 4) Return answer + source list
    sources = [d.metadata.get("source") for d in docs if hasattr(d, "metadata")]
    return AskResponse(answer=answer_text, sources=sources)
