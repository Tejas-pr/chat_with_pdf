import os
import re
import chromadb
import ollama
from pypdf import PdfReader

PDF_PATH = "Tejas_P_R_Resume.pdf"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3.5:9b"
# LLM_MODEL = "llama3.2:3b"

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

# --- 4. RETRIEVE AND GENERATE (RAG) ---
def chat_with_pdf(collection, user_question):
    # 4a. Embed question
    query_embed = ollama.embed(model=EMBED_MODEL, input=user_question)["embeddings"][0]

    # 4b. Find top 4 chunks (better coverage!)
    results = collection.query(
        query_embeddings=[query_embed],
        n_results=4
    )
    retrieved_chunks = results["documents"][0]
    context = "\n---\n".join(retrieved_chunks)

    # 4c. Strict RAG prompt
    prompt = f"""You are an assistant answering questions about a resume/document.
Use ONLY the context below. If you are not sure, say you don't know.

Context:
{context}

Question:
{user_question}
"""

    # 4d. Fast generation
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"], context

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

    while True:
        question = input("\n🧑 You: ")
        if question.strip().lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        if not question.strip():
            continue

        print("\n🤖 Thinking...")
        answer, context_used = chat_with_pdf(collection, question)
        print(f"\n💡 Answer:\n{answer}")
