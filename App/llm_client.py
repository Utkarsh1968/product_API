import requests
import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def construct_prompt(question, retrieved_chunks):
    context = "\n".join(f"- ({pid}): {chunk}" for pid, chunk in retrieved_chunks)
    prompt = f"""
You are a product assistant. Use only the context below to answer the user's question.
If not answerable, say "Sorry, not found in the product info."

Context:
{context}

Question: {question}
Answer:
"""
    return prompt

def get_llm_answer(prompt):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }
    try:
        res = requests.post(f"{url}?key={GEMINI_API_KEY}", json=payload, headers=headers, timeout=10)
        res.raise_for_status()
        return res.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"LLM Error: {e}"
