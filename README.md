

#  **Title:**

### **Advanced RAG Chatbot – LangChain, Chroma, Groq, FastAPI, Streamlit**

---

#  **Structure Includes Both Versions:**

### ** Version 1 — Local RAG with Ollama (Original)**

### ** Version 2 — Cloud RAG with Groq API + FastAPI + Streamlit (New Upgrade)**

---

---

#  **Advanced RAG Chatbot – LangChain, Chroma, Groq, FastAPI, Streamlit**

A complete **Retrieval-Augmented Generation (RAG)** framework built using modern AI tools.
This project contains **two powerful versions**:

---

#  **Version 1 – Local RAG with Ollama (Offline LLM)**

> Uses **LangChain + Chroma + Ollama**
> Runs **completely on your machine**, internet not needed.

#  **Version 2 – Cloud RAG with Groq API (FastAPI + Streamlit)**

> Uses **Groq’s ultra-fast hosted LLaMA models**
> Provides backend **REST API**
> Includes a **ChatGPT-style Streamlit frontend**
> Suitable for **cloud deployment (HuggingFace, Render, etc.)**

---

#  **Overview**

How to build a complete RAG pipeline:

✔ Load documents (PDF, text, Word, PPT, JSON…)
✔ Split content into chunks
✔ Generate embeddings using SentenceTransformers
✔ Store/search using Chroma Vector DB
✔ Integrate a Large Language Model (LLM)
✔ Build a conversational chatbot
✔ Provide both FastAPI backend + Streamlit UI

---

# 📂 **Project Structure**

```
.
├── CoE_Work_Docs/               # Sample documents
├── CoE_Knowledge_Base/          # Vector DB storage
│
│
├── api_groq.py                  # NEW backend (Groq + FastAPI + RAG)
├── streamlit_app.py             # NEW frontend (Chat UI)
│
├── chatbot_test1.py           # Shared RAG helper functions-older version
├── README.md
└── requirements.txt
```

---

#  **Version 1 — Local RAG with Ollama**

### ** Stack**

* LangChain
* ChromaDB
* Sentence Transformers
* Local LLM via Ollama (llama3, mistral, gemma etc.)

### ** Run (Local Only)**

Start Ollama server:

```
ollama run llama3
```

Run Python script:

```
python chatbot_test1.py
```

---

#  **Version 2 — Cloud RAG with Groq (New Upgrade )**

This upgraded version introduces:

### ✔️ FastAPI backend

### ✔️ Groq LLaMA models (super fast)

### ✔️ Streamlit front-end UI

### ✔️ Document retrieval viewer

### ✔️ Ready for deployment on HuggingFace Spaces

---

#  **Architecture**

Below is the RAG architecture used in both versions:

### (Using your uploaded architecture file)

 *Chatbot Architecture Image:*
`/mnt/data/8f890c04-9278-49c5-a5b7-22cafcf99aaf.png`

---

# 🛠️ **Run the NEW Version (Groq + FastAPI + Streamlit)**

## 1️ Install dependencies

```
pip install -r requirements.txt
```

## 2️ Set up Groq API key

Windows:

```
setx GROQ_API_KEY "your_key_here"
```

macOS/Linux:

```
export GROQ_API_KEY="your_key_here"
```

## 3️ Start FastAPI backend

```
uvicorn api_groq:app --reload
```

API Docs:

 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 4️ Start Streamlit chatbot UI

```
streamlit run streamlit_app.py
```

Frontend URL:

 [http://localhost:8501](http://localhost:8501)

---

# 💬 **Streamlit UI Features**

✔ ChatGPT-style chat interface
✔ Shows relevant retrieved chunks
✔ Pretty formatting
✔ API-powered chat
✔ Works locally or on cloud

---

#  **API Endpoint**

### `POST /ask`

**Request Body:**

```json
{
  "question": "What is vector store?"
}
```

**Response:**

```json
{
  "answer": "A vector store is ...",
  "retrieved_docs": [
    {"source": "doc1.txt", "page": 1, "text": "...."}
  ]
}
```

---

#  **Technologies Used**

### 🔹 LangChain

Text splitting, pipeline management, retrieval utilities.

### 🔹 ChromaDB

Vector store for chunk similarity search.

### 🔹 Sentence-Transformers

Embeddings for document chunks.

### 🔹 Groq LLaMA3

Hosted LLM for blazing-fast responses.

### 🔹 FastAPI

Backend REST API.

### 🔹 Streamlit

Frontend UI for chat.

---

#  **Future Improvements**

* Add file upload feature to rebuild vector DB
* Add streaming responses (token-by-token)
* Add authentication to FastAPI
* Deploy backend + frontend on HuggingFace
* Add Pinecone / Weaviate alternative vector databases
* Add Long-context models (LLaMA3-405B, Mistral-Large)

---

#  **Contributions**

PRs are welcome!
Feature suggestions are welcome!

---
