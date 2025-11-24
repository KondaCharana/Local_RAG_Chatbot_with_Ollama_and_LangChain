# streamlit_app.py
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"   # local FastAPI backend

st.set_page_config(page_title="CoE RAG Chatbot", layout="wide")

st.title("🤖 CoE AI RAG Chatbot (Groq Powered)")
st.write("Ask questions about CoE Team documents.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"🧑 **You:** {msg['content']}")
    else:
        st.markdown(f"🤖 **AI:** {msg['content']}")

        # Show retrieved documents
        with st.expander("📄 Retrieved Documents"):
            for i, doc in enumerate(msg["sources"]):
                st.markdown(f"**{i+1}. Source:** {doc['source']}")
                st.write(doc["content"])
                st.markdown("---")


# Input box
user_input = st.text_input("Your question:", key="input")

if st.button("Send"):
    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        # Add user msg to chat
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Call backend
        response = requests.post(API_URL, json={"question": user_input}).json()

        answer = response["answer"]
        retrieved_docs = response["sources"]

        # Add AI response
        st.session_state.messages.append(
            {"role": "ai", "content": answer, "sources": retrieved_docs}
        )

        st.rerun()
