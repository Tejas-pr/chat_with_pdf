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

# --- 1. EXTRACT & CHUNK WITH PAGE METADATA ---
def extract_and_chunk_pdf(pdf_file, chunk_size=1200, overlap=200):
    reader = PdfReader(pdf_file)
    chunks = []
    metadatas = []

    for page_idx, page in enumerate(reader.pages):
        raw_text = page.extract_text()
        if not raw_text:
            continue
        cleaned_text = re.sub(r'\s+', ' ', raw_text).strip()

        start = 0
        while start < len(cleaned_text):
            end = start + chunk_size
            chunk = cleaned_text[start:end]
            chunks.append(chunk)
            # Store which page and file this chunk belongs to!
            metadatas.append({
                "page": page_idx + 1,  # 1-indexed for humans
                "filename": pdf_file.name
            })
            start += chunk_size - overlap

    return chunks, metadatas

# --- 2. VECTOR DATABASE WITH METADATA ---
def build_vector_store(chunks, metadatas):
    client = chromadb.Client()
    collection_name = f"doc_{uuid.uuid4().hex[:8]}"
    collection = client.create_collection(name=collection_name)

    for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
        res = ollama.embed(model=EMBED_MODEL, input=chunk)
        embedding = res["embeddings"][0]
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[meta]  # 👈 Storing page numbers
        )
    return client, collection

# --- 3. SIDEBAR CONTROLS ---
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

    if st.button("🧹 Clear Chat & Delete Document"):
        if st.session_state.client and st.session_state.collection:
            try:
                st.session_state.client.delete_collection(st.session_state.collection.name)
            except Exception:
                pass
        st.session_state.collection = None
        st.session_state.client = None
        st.session_state.current_file = None
        st.session_state.messages = []
        st.rerun()

# --- 4. INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "collection" not in st.session_state:
    st.session_state.collection = None

if "client" not in st.session_state:
    st.session_state.client = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None

# --- 5. INGEST UPLOADED PDF ---
if uploaded_file and (st.session_state.current_file != uploaded_file.name):
    with st.spinner("📖 Reading & indexing PDF into ChromaDB..."):
        chunks, metadatas = extract_and_chunk_pdf(uploaded_file)
        client, collection = build_vector_store(chunks, metadatas)

        st.session_state.client = client
        st.session_state.collection = collection
        st.session_state.current_file = uploaded_file.name
        st.session_state.messages = []  # Reset chat for new file
        st.success(f"Indexed **{uploaded_file.name}** ({len(chunks)} chunks across {len(PdfReader(uploaded_file).pages)} pages)!")

# --- 6. DISPLAY CHAT HISTORY WITH CITATIONS ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If assistant message has source citations, render them!
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 View Sources & Citations"):
                for src in msg["sources"]:
                    st.markdown(f"**📄 Page {src['page']}** (`{src['filename']}`)")
                    st.caption(src["text"])
                    st.divider()

# --- 7. CHAT INPUT & STREAMING ---
if prompt := st.chat_input("Ask a question about your PDF..."):
    if not st.session_state.collection:
        st.warning("⚠️ Please upload a PDF first from the sidebar!")
        st.stop()

    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Query ChromaDB (returns documents AND metadatas)
    query_embed = ollama.embed(model=EMBED_MODEL, input=prompt)["embeddings"][0]
    results = st.session_state.collection.query(
        query_embeddings=[query_embed],
        n_results=4
    )
    retrieved_chunks = results["documents"][0]
    retrieved_metadatas = results["metadatas"][0]

    context = "\n---\n".join(retrieved_chunks)

    # Prepare sources list for display
    sources = [
        {"page": meta["page"], "filename": meta["filename"], "text": doc}
        for doc, meta in zip(retrieved_chunks, retrieved_metadatas)
    ]

    # Grounded prompt
    system_prompt = f"""You are a helpful assistant answering questions about a document.
Assume the user may be the author if they ask "who am I" or "my name".
Use the following context to answer:
{context}

If the answer is not in the context, politely say you don't find it.
Keep answers clear and concise."""

    messages_to_send = [{"role": "system", "content": system_prompt}]
    for m in st.session_state.messages:
        messages_to_send.append({"role": m["role"], "content": m["content"]})

    # Stream response
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

        # Show citations immediately under the answer!
        with st.expander("📚 View Sources & Citations"):
            for src in sources:
                st.markdown(f"**📄 Page {src['page']}** (`{src['filename']}`)")
                st.caption(src["text"])
                st.divider()

    # Save to history with sources attached!
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "sources": sources
    })
