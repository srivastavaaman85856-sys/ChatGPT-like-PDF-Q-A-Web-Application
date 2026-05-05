import streamlit as st
import requests
import tempfile
import os

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

st.set_page_config(page_title="Chat with PDF", layout="wide")

# ---------- CSS for Chat Bubbles ----------
st.markdown("""
    <style>
    .user-bubble {
        background-color: #DCF8C6;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        max-width: 75%;
        float: right;
        clear: both;
        color: black;
    }
    .bot-bubble {
        background-color: #F1F0F0;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        max-width: 75%;
        float: left;
        clear: both;
        color: black;
    }
    .clearfix { clear: both; }
    </style>
""", unsafe_allow_html=True)

st.title("📄 Chat with PDF (Ollama)")

# ---------- Session State Init ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "db" not in st.session_state:
    st.session_state.db = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# ---------- Sidebar: Upload Multiple PDFs ----------
with st.sidebar:
    st.header("📂 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        new_files = [f.name for f in uploaded_files if f.name not in st.session_state.processed_files]

        if new_files:
            with st.spinner("Processing PDFs..."):
                all_docs = []

                for uploaded_file in uploaded_files:
                    if uploaded_file.name in st.session_state.processed_files:
                        continue

                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        pdf_path = tmp_file.name

                    # Load and split
                    loader = PyPDFLoader(pdf_path)
                    documents = loader.load()

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=100
                    )
                    docs = splitter.split_documents(documents)
                    all_docs.extend(docs)

                    # Cleanup temp
                    os.remove(pdf_path)
                    st.session_state.processed_files.append(uploaded_file.name)

                # Build or update FAISS index
                embeddings = OllamaEmbeddings(model="nomic-embed-text")

                if st.session_state.db is None:
                    st.session_state.db = FAISS.from_documents(all_docs, embeddings)
                else:
                    new_db = FAISS.from_documents(all_docs, embeddings)
                    st.session_state.db.merge_from(new_db)

            st.success(f"✅ Processed {len(new_files)} new PDF(s)!")

    # Show processed files
    if st.session_state.processed_files:
        st.markdown("### 📋 Loaded PDFs:")
        for name in st.session_state.processed_files:
            st.markdown(f"- {name}")

    # Clear button
    if st.button("🗑️ Clear Everything"):
        st.session_state.chat_history = []
        st.session_state.db = None
        st.session_state.processed_files = []
        st.rerun()

# ---------- Main Chat Area ----------
if st.session_state.db is None:
    st.info("👈 Upload one or more PDFs from the sidebar to get started.")
else:
    # Display chat history
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f'<div class="user-bubble">🧑 {chat["content"]}</div><div class="clearfix"></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-bubble">🤖 {chat["content"]}</div><div class="clearfix"></div>', unsafe_allow_html=True)

    # Chat input at the bottom
    query = st.chat_input("Ask a question from your PDFs...")

    if query:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Retrieve relevant chunks
        relevant_docs = st.session_state.db.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Build prompt with last 6 messages as history
        history_text = ""
        for msg in st.session_state.chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

        prompt = f"""You are a helpful assistant. Answer ONLY from the given context.
If the answer is not in the context, say "I don't know based on the provided PDFs."

Context:
{context}

Conversation History:
{history_text}

Answer the last user question:"""

        # Call Ollama
        with st.spinner("Thinking..."):
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "gemma3",
                    "prompt": prompt,
                    "stream": False
                }
            )
            answer = response.json()["response"]

        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # Rerun to show updated chat
        st.rerun()