import os
import chromadb
import ollama
from pypdf import PdfReader

PDF_PATH = "Tejas_P_R_Resume.pdf"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3.5:9b"

# --- 1. EXTRACT TEXT FROM PDF ---
def extract_text(pdf_path):
    print(f"📖 Reading {pdf_path}...")
    reader = PdfReader(pdf_path)
    full_text = ""
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text

# --- 2. CHUNK TEXT ---
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    print(f"✂️  Split document into {len(chunks)} chunks.")
    return chunks

# --- 3. VECTOR DATABASE (CHROMADB) SETUP ---
def build_vector_store(chunks):
    print("🧠 Creating embeddings and storing in ChromaDB...")
    client = chromadb.Client()
    collection = client.create_collection(name="pdf_knowledge")

    for i, chunk in enumerate(chunks):
        # Convert chunk to numbers (embedding)
        response = ollama.embed(model=EMBED_MODEL, input=chunk)
        embedding = response["embeddings"][0]

        # Store in ChromaDB
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk]
        )
    print("✅ Indexing complete!")
    return collection

# --- 4. RETRIEVE AND GENERATE (RAG) ---
def chat_with_pdf(collection, user_question):
    # 4a. Embed the user's question
    query_embed = ollama.embed(model=EMBED_MODEL, input=user_question)["embeddings"][0]

    # 4b. Find the 2 most relevant chunks
    results = collection.query(
        query_embeddings=[query_embed],
        n_results=2
    )
    retrieved_chunks = results["documents"][0]
    context = "\n---\n".join(retrieved_chunks)

    # 4c. Construct the prompt with retrieved context
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context below.
If the answer is not in the context, say "I don't find that information in the document."

Context:
{context}

Question:
{user_question}
"""

    # 4d. Generate answer using your local LLM
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"], context

# --- MAIN INTERACTIVE LOOP ---
if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: {PDF_PATH} not found in this directory!")
        exit(1)

    # Ingestion phase (runs once at start)
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
        
        # Optional: inspect what context was actually retrieved
        print("\n" + "-"*40)
        print("🔍 [RAG Context Retrieved Under The Hood]:")
        print(context_used)
        print("-"*(40))
