import os
import re
import chromadb
import ollama
from pypdf import PdfReader

PDF_PATH = "Tejas_P_R_Resume.pdf"
EMBED_MODEL = "nomic-embed-text"
# LLM_MODEL = "qwen3.5:9b"
LLM_MODEL = "llama3.2:latest"

# --- 1. EXTRACT & CLEAN TEXT FROM PDF ---
def extract_text(pdf_path):
    print(f"📖 Reading {pdf_path}...")
    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text + "\n"
    
    # 🧼 CLEANING: Replace excessive newlines and weird spaces with single spaces
    cleaned_text = re.sub(r'\s+', ' ', raw_text).strip()
    return cleaned_text

# --- 2. CHUNK TEXT ---
def chunk_text(text, chunk_size=600, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    print(f"✂️  Split cleaned document into {len(chunks)} chunks.")
    return chunks

# --- 3. VECTOR DATABASE (CHROMADB) SETUP ---
def build_vector_store(chunks):
    print("🧠 Creating embeddings and storing in ChromaDB...")
    client = chromadb.Client()
    collection = client.create_collection(name="pdf_knowledge")

    for i, chunk in enumerate(chunks):
        response = ollama.embed(model=EMBED_MODEL, input=chunk)
        embedding = response["embeddings"][0]

        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk]
        )
    print("✅ Indexing complete!")
    return collection

# --- 4. RETRIEVE AND GENERATE (WITH STREAMING & MEMORY) ---
def chat_with_pdf(collection, user_question, history):
    print("🔍 Searching document & thinking...\n", end="", flush=True)
    # 4a. Embed question
    query_embed = ollama.embed(model=EMBED_MODEL, input=user_question)["embeddings"][0]

    # 4b. Find top 4 chunks
    results = collection.query(
        query_embeddings=[query_embed],
        n_results=7
    )
    retrieved_chunks = results["documents"][0]
    context = "\n---\n".join(retrieved_chunks)

    # 4c. Strict RAG System Prompt with context
    system_message = {
        "role": "system",
        "content": f"""You are a helpful assistant answering questions about a document.
        Use the following context to answer questions:
        {context}

        If the answer is not in the context, say "I don't find that information in the document."
        Keep answers concise and clear."""
    }

    # 4d. Build the full message thread: [System Message] + [Past History] + [New Question]
    messages_to_send = [system_message] + history + [{"role": "user", "content": user_question}]

    # 4e. Stream the response word-by-word
    stream = ollama.chat(
        model=LLM_MODEL,
        messages=messages_to_send,
        stream=True  # ⚡ Enables live token streaming
    )

    full_response = ""
    for chunk in stream:
        token = chunk["message"]["content"]
        print(token, end="", flush=True)  # Print word immediately
        full_response += token

    print()  # Print a clean newline at the end
    return full_response

# --- MAIN LOOP ---
if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: {PDF_PATH} not found!")
        exit(1)

    text = extract_text(PDF_PATH)
    chunks = chunk_text(text)
    collection = build_vector_store(chunks)

    print("\n" + "="*50)
    print("🚀 Ready! Ask questions about your PDF (type 'exit' to quit):")
    print("="*50 + "\n")

    conversation_history = []

    while True:
        question = input("\n🧑 You: ")
        if question.strip().lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        if not question.strip():
            continue

        # Call with streaming and history
        answer = chat_with_pdf(collection, question, conversation_history)
        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": answer})
