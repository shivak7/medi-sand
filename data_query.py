import requests
import json
from sentence_transformers import SentenceTransformer
import faiss

class MedicalRAGOrchestrator:
    def __init__(self, db_folder="medical_db", server_url="http://viropa:8001"):
        self.server_url = server_url
        
        # 1. Load Local Embedding Model (Laptop CPU)
        self.encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        
        # 2. Load the Local Index and Metadata
        self.index = faiss.read_index(f"{db_folder}/index.faiss")
        with open(f"{db_folder}/metadata.json", "r") as f:
            self.metadata = json.load(f)
            
        print(f"Orchestrator ready. Knowledge base: {len(self.metadata)} chunks.")

    def retrieve_context(self, query, top_k=4):
        """Finds the most relevant snippets locally on the laptop."""
        # Embed the query
        query_vector = self.encoder.encode([query], convert_to_numpy=True)
        
        # Search the FAISS index
        distances, indices = self.index.search(query_vector, top_k)
        
        # Pull the text from metadata
        relevant_chunks = [self.metadata[i] for i in indices[0] if i != -1]
        return "\n---\n".join(relevant_chunks)

    def ask_question(self, user_query):
        """The full loop: Retrieve -> Prompt -> Server Call."""
        
        # Step 1: Get local context
        context = self.retrieve_context(user_query)
        
        # Step 2: Build the RAG Prompt
        # We use a 'System' role to ground the model's behavior
        prompt = (
            "You are a clinical data assistant. Use the provided context snippets to answer "
            "the user's question accurately. If the answer isn't in the context, state that "
            "you don't have enough information.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"USER QUESTION: {user_query}"
        )

        # Step 3: Send to the Dual 3090 Server
        print("Sending focused context to server...")
        response = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json={
                "model": "gemma-3-27b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1, # Keep it deterministic for medical data
                "max_tokens": 1024
            }
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error from server: {response.text}"

# --- Usage Example ---
orchestrator = MedicalRAGOrchestrator(db_folder="./hospital_v1_index")
answer = orchestrator.ask_question("Which patients have a history of walnut allergies?")
print(f"\nAI ANSWER:\n{answer}")