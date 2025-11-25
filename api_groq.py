# api_groq.py
import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_core.prompts import ChatPromptTemplate

# ---- import your existing RAG functions ----
from chatbot_test1 import (
    get_or_create_coe_vector_db,
    COE_DOCS_SOURCE_DIR,
    COE_AI_DB_DIR
)

# ---- Load Vector DB ----
vector_db = get_or_create_coe_vector_db(COE_DOCS_SOURCE_DIR, COE_AI_DB_DIR)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# ---- Prompt template ----
rag_prompt_template = ChatPromptTemplate.from_template("""
You are an AI assistant for the Center of Excellence AI Team.
You must answer ONLY using this context.  
If answer not found → say "Not enough info."

Context:
{context}

Question:
{question}
""")

# ---- FastAPI APP ----
app = FastAPI(title="CoE RAG using Groq")

# ---- CORS (allow Streamlit to talk to backend) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Groq Config ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"   # most stable free one


# ---- Groq API Call ----
def call_groq(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }

    response = requests.post(GROQ_API_URL, json=payload, headers=headers)
    data = response.json()

    try:
        return data["choices"][0]["message"]["content"]
    except:
        return str(data)


# ---- Request & Response Models ----
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list


# ---- MAIN RAG ENDPOINT ----
@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):

    # 1) Retrieve docs
    docs = retriever.invoke(req.question)

    context_list = []
    sources = []

    for d in docs:
        meta = d.metadata or {}
        source = meta.get("source", "Unknown")
        context_list.append(f"[{source}] {d.page_content}")
        sources.append({"source": source, "content": d.page_content})

    final_context = "\n\n".join(context_list)

    # 2) Build prompt
    prompt = rag_prompt_template.format_prompt(
        context=final_context, 
        question=req.question
    ).to_string()

    # 3) Call Groq
    answer = call_groq(prompt)

    # 4) Return both
    return AskResponse(answer=answer, sources=sources)
