# 📄 Local RAG: Chat With Your PDF

A 100% private, local, and offline **Retrieval-Augmented Generation (RAG)** application built from scratch using Python, Ollama, ChromaDB, and Streamlit.

---

## 🧠 What is RAG? (The Core Concept)

Think of standard LLMs (like ChatGPT, Llama, Gemini) taking a **closed-book exam**:
* They only know what they memorized during pre-training.
* They have never seen your private PDF, resume, or company documents.
* Asking them about private data leads to guesswork (**hallucination**).

**RAG (Retrieval-Augmented Generation)** converts the LLM into a student taking an **open-book exam**:
1. **Retrieve**: When a question is asked, search your document and grab the most relevant chunks.
2. **Augment**: Inject those chunks directly into the model's prompt as context.
3. **Generate**: The LLM synthesizes an accurate answer grounded strictly in your document.

---

## 🏗️ Architecture & Pipeline

```
                     ┌───────────────────────┐
                     │     Your PDF File     │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ 1. Extract & Clean    │ (pypdf + regex whitespace normalization)
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ 2. Text Chunking      │ (1200 char chunks with 200 char overlap)
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ 3. Vector Embeddings  │ (Ollama: nomic-embed-text)
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ 4. Vector Storage     │ (ChromaDB: Local SQLite / Parquet)
                     └───────────┬───────────┘
                                 │
       ════════════════════ WHEN A USER ASKS A QUESTION ════════════════════
                                 │
                     ┌───────────────────────┐
                     │ 5. Semantic Search    │ (Cosine similarity on query embedding)
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ 6. Grounded Prompt    │ (Inject top chunks into system prompt)
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ 7. LLM Response       │ (Ollama: llama3.2 / qwen3.5 with token streaming)
                     └───────────────────────┘
```

---

## 📚 What I Learned Building This

### 1. Text Extraction & Cleaning
* Raw PDFs often contain scattered linebreaks, odd kerning, and weird whitespace.
* Using `re.sub(r'\s+', ' ', raw_text)` prevents fragmented words and keeps sentences cohesive.

### 2. Chunking Strategy & The "Galuru" Bug 🐛
* **Why chunk size matters**: Naive character chunking (`chunk_size=500` with no cleaning) sliced `"Bengaluru"` into `"galuru"` at chunk boundaries. The AI saw `"galuru | 2020-2024"` at the top of the chunk and concluded that the person's name was **"galuru"**!
* **The fix**: Using sensible chunk sizes (`1200` characters) with an overlap (`200` characters) ensures critical concepts and names never get cut in half.

### 3. Embeddings vs Generative Models
* **Generative models** (`llama3.2`, `qwen3.5`): Think, write, and converse.
* **Embedding models** (`nomic-embed-text`): Pure mathematical encoders that map text into a 768-dimensional semantic space where similar ideas sit close to each other.

### 4. Vector Database (ChromaDB) & Disk Persistence
* In-memory databases (`chromadb.Client()`) re-embed documents on every single restart.
* Persistent databases (`chromadb.PersistentClient(path="./chroma_db")`) store vectors on disk.
* Smart loading checks `collection.count() > 0` to skip re-indexing, loading pre-computed documents in **0.01 seconds**.

### 5. Conversational Memory & Streaming
* **Statelessness**: LLMs have no internal memory between queries. We maintain memory by appending user and assistant messages into a conversation list (`messages`).
* **Live Streaming**: Using `stream=True` prints tokens as they are generated, giving the responsive ChatGPT typing feel.

### 6. The "Identity / Pronoun" Mismatch in RAG
* When a user asks *"What is my name?"*, the LLM doesn't inherently know that the user is the author of the resume.
* Clarifying the system prompt (*"Assume the user is the author/subject of the document when they refer to 'I' or 'my'"*) resolves pronoun ambiguity.

---

## 🛠️ Tech Stack

* **Language**: Python 3.9+
* **LLM Runtime**: [Ollama](https://ollama.ai/)
* **Generative Models**: Meta `llama3.2:latest` (3B) / Alibaba `qwen3.5:9b`
* **Embedding Model**: `nomic-embed-text` (768 dimensions)
* **Vector Database**: [ChromaDB](https://www.trychroma.com/)
* **PDF Parser**: `pypdf`
* **Web UI**: [Streamlit](https://streamlit.io/)

---

## 🚀 How to Run

### 1. Prerequisites
Ensure Ollama is running and download the models:
```bash
ollama pull nomic-embed-text
ollama pull llama3.2:latest
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install pypdf chromadb ollama streamlit
```

### 3. Option A: Run Terminal Interactive CLI
```bash
python main.py
```

### 4. Option B: Run Streamlit Browser Web App
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to drag-and-drop any PDF and chat in real-time!
