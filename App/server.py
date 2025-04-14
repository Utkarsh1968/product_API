from fastapi import FastAPI
from pydantic import BaseModel
from App.loader import load_json, chunk_text
from App.embedder import Embedder
from App.vector_store import vectorStore
from App.retriever import retrieve_chunks
from App.llm_client import construct_prompt, get_llm_answer
import logging

logging.basicConfig(level=logging.INFO)
app = FastAPI()

# Initialize components
products = load_json("Data/Product.json")
chunks = [chunk for product in products for chunk in chunk_text(product)]
texts = [c[1] for c in chunks]
embedder = Embedder()
embeddings = embedder.embed(texts)
vector_store = vectorStore(dim=embeddings[0].shape[0])
vector_store.add(embeddings, chunks)

class Query(BaseModel):
    question: str

@app.post("/query")
async def query_endpoint(query: Query):
    logging.info(f"Received: {query.question}")
    top_chunks = retrieve_chunks(query.question, embedder, vector_store)
    prompt = construct_prompt(query.question, top_chunks)
    answer = get_llm_answer(prompt)
    product_ids = list(set(pid for pid, _ in top_chunks))
    return {"answer": answer, "source_product_ids": product_ids}
