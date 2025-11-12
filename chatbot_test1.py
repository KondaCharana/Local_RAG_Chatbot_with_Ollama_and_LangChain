import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader,UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama # For connecting to local Ollama models
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser

# --- Configuration ---
COE_AI_DB_DIR = "CoE_Knowledge_Base"
COE_DOCS_SOURCE_DIR = "CoE_Work_Docs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "llama3.1" # Ensure this matches the model you pulled in Ollama (e.g., llama3.1, llama2, mistral)

# --- Function to Load Documents (reused from previous script) ---
def load_coe_documents(source_dir: str) -> list:
    """
    Loads documents from the specified directory using appropriate LangChain loaders.
    Supports .txt, .pdf, and .docx files.
    """
    print(f"\nLoading documents from '{source_dir}'...")
    loaded_documents = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.endswith((".txt",".py")):
                    loader = TextLoader(file_path)
                elif file.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif file.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                elif file.endswith(".pptx"):
                    loader = UnstructuredPowerPointLoader(file_path)
                elif file.endswith(".json"):
                    from langchain_community.document_loaders import UnstructuredFileLoader
                    loader = UnstructuredFileLoader(file_path)
                elif file.endswith(".xlsx"):
                    from langchain_community.document_loaders import UnstructuredExcelLoader
                    loader = UnstructuredExcelLoader(file_path)
                else:
                    print(f"  Skipping unsupported file type: {file_path}")
                    continue
                loaded_documents.extend(loader.load())
                print(f"  Loaded: {file}")
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
    if not loaded_documents:
        print(f"No documents found or loaded from '{source_dir}'.")
    return loaded_documents

# --- Function to Create or Load the Vector Database (reused from previous script) ---
def get_or_create_coe_vector_db(docs_source_dir: str, db_persist_dir: str) -> Chroma:
    """
    Creates or loads the Chroma vector database for the CoE AI Team.
    It will load documents, chunk them, embed them, and store/load them.
    """
    print(f"\n--- Setting up CoE AI Knowledge Base ---")
    os.makedirs(docs_source_dir, exist_ok=True)
    os.makedirs(db_persist_dir, exist_ok=True)

    print(f"1. Initializing embedding model: {EMBEDDING_MODEL_NAME} (This may download the model)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("Embedding model initialized.")

    vectorstore = None
    if os.path.exists(os.path.join(db_persist_dir, "chroma.sqlite3")):
        print(f"2. Loading existing vector database from '{db_persist_dir}'...")
        vectorstore = Chroma(persist_directory=db_persist_dir, embedding_function=embeddings)
        print("Vector database loaded.")
    else:
        print(f"2. Creating new vector database in '{db_persist_dir}'...")

    documents = load_coe_documents(docs_source_dir)
    if not documents:
        if vectorstore:
            print("No new documents found to add. Using existing database.")
            return vectorstore
        else:
            print("Cannot create database: No documents found to process. Please add files to 'CoE_AI_Team_Work_Docs'.")
            exit()

    print("\n3. Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    if vectorstore is None:
        print(f"\n4. Populating new vector database with {len(chunks)} chunks...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_persist_dir
        )
        print("New vector database created and populated.")
    else:
        print(f"\n4. Adding {len(chunks)} new chunks to existing vector database...")
        vectorstore.add_documents(chunks)
        print("Existing vector database updated with new chunks.")
    
    vectorstore.persist()
    print(f"Vector database saved to disk in '{db_persist_dir}'.")

    return vectorstore

# --- Main Chatbot Logic ---
if __name__ == "__main__":
    # --- Ensure Ollama Server is Running ---
    print("--- IMPORTANT: Ensure your Ollama server is running and 'llama3.1' model is pulled ---")
    print("   You can run 'ollama run llama3.1' in your terminal to start it.")
    print("   If you have a different model, update OLLAMA_MODEL_NAME in the script.")
    
    # --- Create some dummy work documents for the CoE AI Team (if not already there) ---
    print(f"\nEnsuring '{COE_DOCS_SOURCE_DIR}' directory exists and contains dummy data...")
    os.makedirs(COE_DOCS_SOURCE_DIR, exist_ok=True)

    # Example: Project Report
    project_report_path = os.path.join(COE_DOCS_SOURCE_DIR, "Project_Alpha_Q1_Report.txt")
    if not os.path.exists(project_report_path):
        with open(project_report_path, "w") as f:
            f.write("""
            Project Alpha Q1 2024 Report:
            Objective: Develop a new anomaly detection model for network security.
            Key Results: Achieved 95% detection accuracy on test data. Implemented a real-time inference pipeline.
            Challenges: Data labeling bottleneck, integration with legacy systems.
            Next Steps: Focus on data augmentation, optimize model for edge deployment.
            Team Members: Alex, Ben, Carol.
            """)

    # Example: Research Notes (PDF content simulated as .txt)
    llm_research_path = os.path.join(COE_DOCS_SOURCE_DIR, "Latest_LLM_Research_Summary.pdf")
    if not os.path.exists(llm_research_path):
        with open(llm_research_path, "w") as f:
            f.write("""
            Summary of Latest LLM Research (May 2024):
            Recent advancements focus on multi-modal LLMs and improved long-context understanding.
            New techniques like "Tree-of-Thought" prompting show promise for complex reasoning.
            Concerns remain regarding hallucination and computational cost.
            """)
    
    # Example: Meeting Minutes (DOCX content simulated as .txt)
    meeting_minutes_path = os.path.join(COE_DOCS_SOURCE_DIR, "AI_Team_Meeting_2024_06_28.docx")
    if not os.path.exists(meeting_minutes_path):
        with open(meeting_minutes_path, "w") as f:
            f.write("""
            AI Team Meeting Minutes - June 28, 2024:
            Attendees: All AI Team members.
            Agenda: Project Beta kickoff, review of Q2 performance, planning for upcoming training.
            Decisions: Project Beta will use a hybrid cloud architecture. Training on advanced prompt engineering scheduled for July.
            Action Items: Ben to draft Project Beta's initial design document by July 5th.
            """)
    
    # Example: Course Info
    course_info_path = os.path.join(COE_DOCS_SOURCE_DIR, "python_course_info.txt")
    if not os.path.exists(course_info_path):
        with open(course_info_path, "w") as f:
            f.write("""
            Course Name: Advanced Python for Data Science
            Instructor: Dr. Anya Sharma
            Duration: 8 weeks
            Schedule: Tuesdays & Thursdays, 6 PM - 8 PM (Online)
            Prerequisites: Intermediate Python, basic statistics
            Content: Covers Pandas, NumPy, Scikit-learn, Matplotlib, and an introduction to deep learning with TensorFlow.
            Certification: Certificate of Completion upon successful project submission.
            Next Start Date: September 15, 2024
            """)

    print("\nDummy documents ensured for chatbot context.")

    # --- 1. Create or Load the CoE AI Vector Database ---
    # This will ensure your knowledge base is ready
    coe_vector_db = get_or_create_coe_vector_db(COE_DOCS_SOURCE_DIR, COE_AI_DB_DIR)

    # --- 2. Initialize Ollama LLM ---
    print(f"\n2. Initializing Ollama LLM with model: {OLLAMA_MODEL_NAME}...")
    # Ensure Ollama server is running and the model is pulled
    llm = ChatOllama(model=OLLAMA_MODEL_NAME)
    print("Ollama LLM initialized.")

    # --- 3. Create a Retriever from the Vector Database ---
    print("\n3. Creating retriever from vector database...")
    retriever = coe_vector_db.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks
    print("Retriever created.")

    # --- 4. Define the RAG Prompt Template ---
    # This template instructs the LLM to use the provided context
    rag_prompt_template = ChatPromptTemplate.from_template("""
    You are an AI assistant for the Center of Excellence AI Team.
    Your goal is to answer questions based ONLY on the provided context.
    If you cannot find the answer in the context, politely state that you don't have enough information.

    Context: {context}

    Question: {question}
    """)
    print("\n4. RAG Prompt Template defined.")

    # --- 5. Build the RAG Chain ---
    # This uses LangChain Expression Language (LCEL) to define the RAG workflow
    print("\n5. Building the RAG chain...")
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} # Pass question to retriever, and question itself
        | rag_prompt_template                                    # Format into the RAG prompt
        | llm                                                    # Send to Ollama LLM
        | StrOutputParser()                                      # Parse LLM's output as a string
    )
    print("RAG chain built.")

    # --- 6. Start the Chatbot Loop ---
    print("\n--- CoE AI Chatbot (Powered by Ollama & Local Vector DB) ---")
    print("Type your questions about CoE AI team work. Type 'exit' to quit.")

    while True:
        user_input = input("\nYour Question: ")
        if user_input.lower() == 'exit':
            print("Exiting chatbot. Goodbye!")
            break
        
        print("Thinking...")
        try:
            # Invoke the RAG chain with the user's question
            response = rag_chain.invoke(user_input)
            print(f"AI Assistant: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure your Ollama server is running and the model is available.")

