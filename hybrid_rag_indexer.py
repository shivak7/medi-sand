import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class HybridMedicalIndexer:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", device="cpu"):
        self.encoder = SentenceTransformer(model_name, device=device)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        # self.metadata now stores a list of dictionaries instead of raw strings
        self.metadata = [] 

    def add_to_index(self, chunks, meta_tags):
        """
        chunks: List of strings (the text)
        meta_tags: List of dicts (e.g., [{'id': 'P1', 'condition': 'Diabetes'}, ...])
        """
        if len(chunks) != len(meta_tags):
            raise ValueError("The number of chunks must match the number of metadata tags.")

        embeddings = self.encoder.encode(chunks, convert_to_numpy=True)
        self.index.add(embeddings)
        
        # Combine text and tags for each entry
        for i in range(len(chunks)):
            self.metadata.append({
                "text": chunks[i],
                "meta": meta_tags[i]
            })

    def count_by_condition(self, condition_name):
        """
        The key to your question: This skips the LLM and Top-K entirely.
        It scans the structured metadata to give an 100% accurate count.
        """
        count = sum(1 for item in self.metadata if item['meta'].get('condition') == condition_name)
        return count

    def filtered_search(self, query, condition_filter, top_k=5):
        """
        Performs a semantic search BUT only within chunks that match a metadata filter.
        """
        query_vector = self.encoder.encode([query], convert_to_numpy=True)
        # Note: Standard FAISS doesn't do complex filtering internally; 
        # in a million-record scenario, you would use a DB like 'ChromaDB' or 'Qdrant'
        # For this local demo, we search, then filter the results.
        distances, indices = self.index.search(query_vector, top_k * 10) # Get more candidates
        
        results = []
        for idx in indices[0]:
            if idx != -1:
                item = self.metadata[idx]
                if item['meta'].get('condition') == condition_filter:
                    results.append(item['text'])
            if len(results) >= top_k:
                break
        return results