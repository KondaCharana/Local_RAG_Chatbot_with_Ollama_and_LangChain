# api_groq.py
import os
import time
import logging
import requests
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_core.prompts import ChatPromptTemplate

# -------------------------------------------------
# Logging (Cloud Run automatically captures stdout)
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logging.info("🚀 Starting CoE RAG API")

# -------------------------------------------------
# Environment variables (Cloud Run injects these)
# -------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama3.1")
GROQ_API_URL = os.getenv(
    "GROQ_API_URL",
    "https://api.groq.com/openai/v1/chat/completions"
)

COE_DOCS_SOURCE_DIR = os.getenv("COE_DOCS_SOURCE_DIR")
COE_AI_DB_DIR = os.getenv("COE_AI_DB_DIR")

if not GROQ_API_KEY:
    logging.warning("⚠️ GROQ_API_KEY is not set")

# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI(title="CoE RAG using Groq")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.info("✅ FastAPI app created")

# -------------------------------------------------
# Lazy RAG objects (CRITICAL for Cloud Run)
# -------------------------------------------------
vector_db = None
retriever = None

def get_retriever():
    """
    Lazily initialize vector DB & retriever.
    Prevents Cloud Run startup timeout.
    """
    global vector_db, retriever

    if retriever is None:
        logging.info("📚 Initializing vector database...")
        from chatbot_test1 import get_or_create_coe_vector_db

        vector_db = get_or_create_coe_vector_db(
            COE_DOCS_SOURCE_DIR,
