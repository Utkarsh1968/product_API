# 🧠 Product Q&A API (RAG-based)

This is a backend API that answers user questions about a specific set of products using a custom **Retrieval-Augmented Generation (RAG)** pipeline. It uses:

- Sentence-transformers for local embedding generation
- FAISS for in-memory vector similarity search
- Gemini or OpenAI for LLM-based answer generation
- FastAPI to expose the entire workflow as a REST API

---

## ⚙️ How It Works

1. **Product data** is loaded from a JSON file.
2. Descriptions and features are broken into small, meaningful text chunks.
3. Each chunk is converted to an embedding vector using a sentence transformer model (e.g., `all-MiniLM-L6-v2`).
4. The user sends a question.
5. The system generates an embedding for the question and retrieves the top-k most relevant chunks using cosine similarity with FAISS.
6. The system constructs a prompt using the retrieved chunks and the user question.
7. The prompt is passed to a Large Language Model (LLM) like Gemini 
8. The LLM responds with an answer **based only on the given context**.

---

## 🗂️ Project Structure

```text
project/
├── app/
│   ├── server.py           # FastAPI application
│   ├── loader.py         # Load and read product data
│   ├── embedder.py       # Local embedding generator
│   ├── vector_store.py   # In-memory FAISS vector store
│   ├── retriever.py      # Retrieval of top-k relevant chunks
│   └── llm_client.py     # Interact with Gemini LLM
├── data/
│   └── products.json     # Dataset of product info
├── .env                  # Environment file with API keys
├── requirements.txt      # Required Python packages
└── README.md
```





## 🧠 Chunking Strategy

We apply a **hybrid chunking strategy**:

- Use NLTK’s `sent_tokenize()` for normal-length, well-punctuated text.
- Use overlapping word-based chunking for long or poorly punctuated text (e.g., 40-word chunks with 10-word overlap).

This approach ensures:
- Meaningful, semantically rich chunks
- Better embedding quality
- Improved accuracy in retrieval and answering

---

## 📦 Installation

### 1. Install Required Libraries

```bash
pip install -r requirements.txt

```
### 2. Add API Keys
Create a .env file in the root of your project and add your API keys:
<pre>
GEMINI_API_KEY = your_gemini_key_here
</pre>



---
## 🚀 Run the API Server
```bash
uvicorn App.server:app --reload
```
---
## 📬 Example API Call

### Endpoint
```bash
POST /query
```
### Request Body
```bash
{
  "question": "What are the features of the Smartphone X200?"
}
```
### Response
```bash
{
  "answer": "Cutting-edge smartphone with a stunning display and powerful performance. Equipped with the latest processor, long battery life, and a sleek, lightweight design.",
  "source_product_ids": ["1","2"]
}
```
---
## 🧠 Models Used
- **Embedding model:** all-MiniLM-L6-v2 (via sentence-transformers)

- **LLM:** gemini-2.0-flash

## ✅ Design Decisions
- All embeddings are generated locally using pre-trained transformer models (no API calls for embeddings).

- No high-level frameworks like LangChain or LlamaIndex were used — all logic (chunking, retrieval, prompting) is implemented from scratch.

- LLM service is configurable (Gemini or OpenAI).

- Supports clean logging, modular structure, and is async-ready using FastAPI.
