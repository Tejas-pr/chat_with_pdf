import os
import re
import uuid
import streamlit as st
import chromadb
import ollama
from pypdf import PdfReader

# Page Configuration
st.set_page_config(page_title="Chat with PDF (Local RAG)", page_icon="📄", layout="centered")
st.title("📄 Chat with your PDF")
st.caption("Powered by Local Ollama & ChromaDB — 100% Private & Offline")

EMBED_MODEL = "nomic-embed-text"

# --- HELPER FUNCTIONS ---
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text + "\n"
    # Clean whitespace
    return re.sub(r'\s+', ' ', raw_text).strip()

def chunk_text(text, chunk_size=1200, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build_vector_store(chunks):
    client = chromadb.Client()
    # Create unique collection for this upload
    collection_name = f"doc_{uuid.uuid4().hex[:8]}"
    collection = client.create_collection(name=collection_name)

    for i, chunk in enumerate(chunks):
        res = ollama.embed(model=EMBED_MODEL, input=chunk)
        embedding = res["embeddings"][0]
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk]
        )
    return collection

# --- SIDEBAR: CONTROLS & FILE UPLOAD ---
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox(
        "Choose Model",
        ["llama3.2:latest", "qwen3.5:9b"],
        index=0
    )

    st.divider()
    st.header("📁 Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "collection" not in st.session_state:
    st.session_state.collection = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None

# --- PROCESS UPLOADED PDF ---
if uploaded_file and (st.session_state.current_file != uploaded_file.name):
    with st.spinner("📖 Reading & indexing PDF into ChromaDB..."):
        text = extract_text(uploaded_file)
        chunks = chunk_text(text)
        collection = build_vector_store(chunks)

        st.session_state.collection = collection
        st.session_state.current_file = uploaded_file.name
        st.session_state.messages = []  # Reset chat for new file
        st.success(f"Indexed **{uploaded_file.name}** ({len(chunks)} chunks ready)!")

# --- DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- CHAT INPUT & RESPONSE ---
if prompt := st.chat_input("Ask a question about your PDF..."):
    if not st.session_state.collection:
        st.warning("⚠️ Please upload a PDF first from the sidebar!")
        st.stop()

    # 1. Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Retrieve relevant chunks
    query_embed = ollama.embed(model=EMBED_MODEL, input=prompt)["embeddings"][0]
    results = st.session_state.collection.query(
        query_embeddings=[query_embed],
        n_results=4
    )
    context = "\n---\n".join(results["documents"][0])

    # 3. System prompt
    system_prompt = f"""You are a helpful assistant answering questions about a document.
Assume the user may be the author if they ask "who am I" or "my name".
Use the following context to answer:
{context}

If the answer is not in the context, politely say you don't find it.
Keep answers clear and concise."""

    messages_to_send = [{"role": "system", "content": system_prompt}]
    # Add conversation history
    for m in st.session_state.messages:
        messages_to_send.append({"role": m["role"], "content": m["content"]})

    # 4. Stream response into UI
    with st.chat_message("assistant"):
        stream = ollama.chat(
            model=model_choice,
            messages=messages_to_send,
            stream=True
        )

        def stream_tokens():
            for chunk in stream:
                yield chunk["message"]["content"]

        response_text = st.write_stream(stream_tokens())

    # 5. Save assistant reply to memory
    st.session_state.messages.append({"role": "assistant", "content": response_text})
