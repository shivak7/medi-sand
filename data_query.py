import requests
import json
from sentence_transformers import SentenceTransformer
import faiss
from rag_module import MedicalIndexer
import requests

class MedicalOrchestrator:
    def __init__(self, indexer, server_url="http://viropa:8001"):
        self.indexer = indexer
        self.server_url = server_url

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
            f"{self.server_url}/v1/chat/completions",
            json={
                "model": "gemma-3-27b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 2048
            },
            timeout=120
        )
        return response.json()['choices'][0]['message']['content']

# --- Usage Example ---
indexer = MedicalIndexer(db_folder='medical_db')
indexer.load()
orchestrator = MedicalOrchestrator(indexer=indexer)
answer = orchestrator.query("Which patients have a history of walnut allergies?")
print(f"\nAI ANSWER:\n{answer}")