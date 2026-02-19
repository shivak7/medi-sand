This reference document compiles all the modules built for my **Local PC/Mac +  Dual 3090 Server** architecture.

---

# **Local Medical Intelligence Pipeline: Reference**

**Architecture:** Local PC/Mac (Docling + FAISS + Embeddings)  Server (Gemma 3 27B + llama.cpp)

---

### **0. Data:** 
Artificially created medical data records made using Gemini 3.0. Primary fields are:

Patient: <patient id code> \
Gender identity: <M/F> \
DOB: <yyyy-mm-dd> \
Date of visit: \
Reason for hospitalization: \
ICD10 code: \
Vital signs: \
Known allergies: \
Pre-existing conditions: \
Medication history: \
Treatment received: \
Medicine prescribed: \
Discharge Status: \
Follow up: <yes/no> \
Follow up date: <yyyy-mm-dd/NA>
Doctor's notes: 

---

### **1. Server Setup (The Host)**

Run this command on the GPU server to ensure the 128k context window and GPU offloading are active.

```bash
# Ensure you are using the build with CUDA support
./build/bin/llama-server \
  -m gemma-3-27b-it.gguf \
  -c 128000 \
  -ngl 99 \
  -sm row \
  --jinja \
  --port 8001 \
  --host 0.0.0.0

```

---

### **2. Module: The Hybrid Indexer (Laptop)**

This module handles chunking, embedding, and persisting your medical data. It uses metadata to allow for accurate "Counting" queries.

```python
import os, json, faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

class MedicalIndexer:
    def __init__(self, db_folder="medical_db", model="BAAI/bge-small-en-v1.5"):
        self.db_folder = db_folder
        self.encoder = SentenceTransformer(model, device="cpu") # Laptop friendly
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []

    def add_documents(self, chunks, metadata_list):
        embeddings = self.encoder.encode(chunks, convert_to_numpy=True)
        self.index.add(embeddings)
        for text, meta in zip(chunks, metadata_list):
            self.metadata.append({"text": text, "meta": meta})

    def save(self):
        if not os.path.exists(self.db_folder): os.makedirs(self.db_folder)
        faiss.write_index(self.index, f"{self.db_folder}/index.faiss")
        with open(f"{self.db_folder}/metadata.json", "w") as f:
            json.dump(self.metadata, f)

    def load(self):
        if os.path.exists(f"{self.db_folder}/index.faiss"):
            self.index = faiss.read_index(f"{self.db_folder}/index.faiss")
            with open(f"{self.db_folder}/metadata.json", "r") as f:
                self.metadata = json.load(f)
            return True
        return False

```

---

### **3. Module: The RAG Orchestrator (Laptop  Server)**

This script performs the "Retrieve and Prompt" logic, sending focused context to your 27B model.

```python
import requests

class MedicalOrchestrator:
    def __init__(self, indexer, server_url="http://localhost:8001"):
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

```

---

### **4. Module: Structured Batch Extraction**

Use this for the "Scenario A" lane (direct processing of smaller PDFs).

```python
def extract_structured_data(markdown_content, server_url="http://localhost:8001"):
    prompt = (
        "Extract every patient record from the text below. "
        "Return ONLY a JSON array of objects. Schema: [patient_id, icd10_codes, summary].\n\n"
        f"TEXT:\n{markdown_content}"
    )

    resp = requests.post(
        f"{server_url}/v1/chat/completions",
        json={
            "model": "gemma-3-27b",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}
        }
    )
    return resp.json()['choices'][0]['message']['content']

```

---

### **Summary of Logic Flow**

1. **Ingest:** Docling converts PDF  Markdown locally on your laptop.
2. **Route:** Check token count via `localhost:8001/tokenize`.
3. **Path A (Small):** Send full Markdown to Server for JSON extraction.
4. **Path B (Massive):** * Chunk text locally.
* Embed and save to FAISS index locally.
* Retrieve top-5 chunks locally.
* Send 5 chunks + question to Server for the final answer.



**You're all set!** Your dual 3090s are now the engine for a very powerful, private medical research tool. Happy coding!