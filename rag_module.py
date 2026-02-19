import os, json, faiss
import numpy as np
from sentence_transformers import SentenceTransformer


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
