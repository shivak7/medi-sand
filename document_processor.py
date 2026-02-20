import os
import json
import requests
from docling.document_converter import DocumentConverter
from rag_module import MedicalIndexer
from langchain_text_splitters import RecursiveCharacterTextSplitter
#import tiktoken



class DocumentHandler:

    def __init__(self, api_key, base_url, is_local, model_name, chunk_size, chunk_overlap, db_folder):
        
        self.api_key = api_key
        self.base_url = base_url
        self.is_local = is_local
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.db_folder = db_folder

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json" 
        }

    def check_server_health(server_url):
        """Verifies if a local llama.cpp server is reachable."""
        try:
            response = requests.get(f"{server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False


    def get_auto_metadata(self, markdown_content):
        """
        Uses Gemma 3 4B to identify the 'header' info of a medical record.
        """
        # We only need the first part of the document to find metadata
        header_text = markdown_content[:2000] 
        
        prompt = (
            "Identify the following fields from this medical record header: "
            "patient_id, primary_condition, visit_year. "
            "Return ONLY a JSON object."
        )

        response = requests.post(
            f"{self.base_url}/v1/chat/completions", headers=self.headers,
            json={
                "model": f"{self.model_name}", # Use a smaller model for speed/cost
                "messages": [{"role": "user", "content": f"{prompt}\n\nTEXT:\n{header_text}"}],
                "response_format": {"type": "json_object"}
            }
        )
        return response.json()['choices'][0]['message']['content']
    
    def process_document_mixed_pipeline(self, file_path):

        if self.is_local:
            if not self.check_server_health(self.base_url):
                return {"lane": "ERROR", "content": "Server unreachable."}

        print(f"Processing: {file_path}")
        converter = DocumentConverter()
        result = converter.convert(file_path)
        markdown_content = result.document.export_to_markdown()

        meta_json_str = self.get_auto_metadata(markdown_content)
        file_meta = json.loads(meta_json_str)
        indexer = MedicalIndexer(db_folder=self.db_folder)
        # Process all your existing PDFs
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_text(markdown_content)
        metadata_list = [file_meta for _ in range(len(chunks))]
        indexer.add_documents(chunks, metadata_list=metadata_list)
        # Save it to disk
        indexer.save()
        return {"lane": "RAG", "content": "Saved to VectorDB + Metadata."}
    
    def batch_process_folder(self, input_dir):
        """Loops through all PDFs in a folder and saves one master JSON."""
        files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
        
        for filename in files:
            file_path = os.path.join(input_dir, filename)
            result = self.process_document_mixed_pipeline(file_path)
            if result["lane"] == "ERROR":
                print(result)
                
        print(f"✅ Batch processing complete.") # {len(all_results)} patients saved to master_extraction.json")


# api_key="sk-no-key-required"         # Use this dummy key if hosting llm locally 
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")


db_creator = DocumentHandler(api_key=api_key, base_url="https://api.openai.com", is_local=False, model_name="gpt-4o-mini", chunk_size=1000, chunk_overlap=150, db_folder = 'medical_db')
db_creator.batch_process_folder(input_dir="./raw_records/")