# api_groq.py
import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from langchain.prompts import ChatPromptTemplate

# ------------------ STARTUP LOGS ------------------
print("🚀 Starting FastAPI application...")

# ------------------ CONFIG ------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = os.getenv(
    "GROQ_API_URL",
    "https://api.groq.ai/v1/models/llama3.1/generate"
)
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama3.1")

COE_DOCS_SOURCE_DIR = os.getenv("COE_DOCS_SOURCE_DIR")
COE_AI_DB_DIR = os.getenv("COE_AI_DB_DIR")

if not GROQ_API_KEY:
    print("⚠️ GROQ_API_KEY is not set")

# ------------------ FASTAPI APP ------------------
app = FastAPI(title="CoE RAG using Groq")
print("✅ FastAPI app object created")

# ------------------ LAZY RAG OBJECTS ------------------
vector_db = None
retriever = None

def get_retriever():
    """
    Lazy-load vector DB and retriever.
    This ensures Cloud Run can bind to PORT first.
    """
    global vector_db, retriever

    if retriever is None:
        print("📚 Loading vector database...")
        from your_rag_module import get_or_create_coe_vector_db

        vector_db = get_or_create_coe_vector_db(
            COE_DOCS_SOURCE_DIR,
            COE_AI_DB_DIR
        )
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        print("✅ Retriever initialized")

    return retriever

# ------------------ PROMPT ------------------
rag_prompt_template = ChatPromptTemplate.from_template("""
You are an AI assistant for the Center of Excellence AI Team.
Use ONLY the provided context below.
If the answer is not present, say you don't have enough information.

Context:
{context}

Question:
{question}
""")

# ------------------ GROQ CALL ------------------
def call_groq(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    response = requests.post(
        GROQ_API_URL,
        json=payload,
        headers=headers,
        timeout=120
    )
    response.raise_for_status()
    data = response.json()

    if isinstance(data, dict):
        if "generated_text" in data:
            return data["generated_text"]
        if "choices" in data:
            return data["choices"][0]["text"]

    return str(data)

# ------------------ API MODELS ------------------
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: List[str]

# ------------------ ENDPOINT ------------------
@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):

    retriever = get_retriever()   # 🔥 Lazy load happens HERE
    docs = retriever.invoke(req.question)

    context_blocks = []
    sources = []

    for d in docs:
        src = d.metadata.get("source", "unknown") if hasattr(d, "metadata") else "unknown"
        context_blocks.append(f"[{src}] {d.page_content}")
        sources.append(src)

    context = "\n\n".join(context_blocks)

    prompt = rag_prompt_template.format_prompt(
        context=context,
        question=req.question
    ).to_string()

    answer = call_groq(prompt)

    return AskResponse(answer=answer, sources=sources)
