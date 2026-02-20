import os
import requests
import json
from sentence_transformers import SentenceTransformer
import faiss
from rag_module import MedicalIndexer

class MedicalOrchestrator:
    def __init__(self, indexer, api_key, base_url="http://localhost:8001", model_name = "gemma-3-27b"):
        self.indexer = indexer
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json" 
        }

    def query(self, user_question, top_k=5):
        # 1. Local Search on Laptop
        query_vec = self.indexer.encoder.encode([user_question], convert_to_numpy=True)
        distances, indices = self.indexer.index.search(query_vec, top_k)
        
        context = "\n---\n".join([self.indexer.metadata[i]['text'] for i in indices[0] if i != -1])

        # 2. Remote Inference on Server
        prompt = (
            "Use the provided context to answer the question. If unsure, say 'Incomplete data'.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {user_question}"
        )

        response = requests.post(
            f"{self.base_url}/v1/chat/completions", headers = self.headers,
            json={
                "model": f"{self.model_name}",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 2048
            },
            timeout=120
        )
        return response.json()['choices'][0]['message']['content']

# --- Usage Example ---

# api_key="sk-no-key-required"         # Use this dummy key if hosting llm locally 
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")


indexer = MedicalIndexer(db_folder='medical_db')
indexer.load()
orchestrator = MedicalOrchestrator(indexer, api_key, base_url="https://api.openai.com", model_name="gpt-4o-mini")
answer = orchestrator.query("Which patients have a history of walnut allergies?")
print(f"\nAI ANSWER:\n{answer}")