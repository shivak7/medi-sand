
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class PersistableMedicalRAGIndexer:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", device="cpu"):
        print(f"Initializing Embedding Model: {model_name}...")
        self.encoder = SentenceTransformer(model_name, device=device)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        
        # We start with None and initialize properly during 'load' or 'add'
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []

    def save(self, folder_path="medical_db"):
        """Saves both the FAISS index and the text metadata to a folder."""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        index_file = os.path.join(folder_path, "index.faiss")
        meta_file = os.path.join(folder_path, "metadata.json")

        # 1. Save the Vector Index (C++ binary format)
        faiss.write_index(self.index, index_file)
        
        # 2. Save the Text Metadata (JSON format)
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=4)
        
        print(f"✅ Database persisted to '{folder_path}'")

    def load(self, folder_path="medical_db"):
        """Loads a previously saved index and metadata."""
        index_file = os.path.join(folder_path, "index.faiss")
        meta_file = os.path.join(folder_path, "metadata.json")

        if os.path.exists(index_file) and os.path.exists(meta_file):
            # 1. Load the Vector Index
            self.index = faiss.read_index(index_file)
            
            # 2. Load the Metadata
            with open(meta_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            print(f"🚀 Loaded {len(self.metadata)} chunks from disk. Ready for queries.")
            return True
        else:
            print("⚠️ No existing index found in this folder.")
            return False

    def add_to_index(self, chunks):
        """Converts chunks to vectors and updates the index."""
        embeddings = self.encoder.encode(chunks, convert_to_numpy=True)
        self.index.add(embeddings)
        self.metadata.extend(chunks)