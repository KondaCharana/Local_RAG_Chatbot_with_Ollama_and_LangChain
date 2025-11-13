# 🤖 AI RAG Chatbot Framework (LangChain + Ollama)

An intelligent **Retrieval-Augmented Generation (RAG)** chatbot built with **LangChain**, **Ollama**, and **Hugging Face embeddings**.  
It allows local, private question answering using your own knowledge base.

---

## 🚀 Features
- Loads and processes `.txt`, `.pdf`, `.docx`, `.pptx`, and `.xlsx` files
- Builds a **local Chroma vector database**
- Integrates **Ollama LLMs (e.g., Llama 3.1)** for offline Q&A
- Auto-creates dummy CoE project data for demonstration
- Fully offline and private RAG pipeline

---

## 🧠 Tech Stack
| Component | Technology |
|------------|-------------|
| Language | Python 3.12+ |
| Framework | LangChain |
| LLM | Ollama (Llama 3.1) |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector DB | Chroma |
| File Handling | PyPDF, Docx2txt, Unstructured |
| Chat Interface | Command-line |

---

## ⚙️ Setup Instructions

# 1. Clone the repository
git clone https://github.com/<your-username>/AI_RAG_Chatbot_Framework.git
cd AI_RAG_Chatbot_Framework

# 2. Create a virtual environment
python -m venv env
env\Scripts\activate   # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Ollama
ollama serve
ollama pull llama3.1

# 5. Run the chatbot
python chatbot_test1.py
