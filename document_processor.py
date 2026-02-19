import os
import json
import requests
from docling.document_converter import DocumentConverter
from rag_module import MedicalIndexer
from langchain_text_splitters import RecursiveCharacterTextSplitter 

def check_server_health(server_url):
    """Verifies the llama.cpp server is reachable."""
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def count_tokens_via_server(text, server_url):
    """Uses the llama.cpp server to count tokens accurately."""
    try:
        response = requests.post(
            f"{server_url}/tokenize",
            json={"content": text},
            timeout=10
        )
        return len(response.json().get("tokens", []))
    except Exception as e:
        print(f"Tokenization failed: {e}")
        return len(text) // 4 
    
def get_auto_metadata(markdown_content, server_url="http://viropa:8001"):
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
        f"{server_url}/v1/chat/completions",
        json={
            "model": "gemma-3-4b", # Using the smaller model for speed/cost
            "messages": [{"role": "user", "content": f"{prompt}\n\nTEXT:\n{header_text}"}],
            "response_format": {"type": "json_object"}
        }
    )
    return response.json()['choices'][0]['message']['content']

def process_document_mixed_pipeline(file_path, server_url="http://viropa:8001", model_name="gemma-3-27b"):
    if not check_server_health(server_url):
        return {"lane": "ERROR", "content": "Server unreachable."}

    print(f"Processing: {file_path}")
    converter = DocumentConverter()
    result = converter.convert(file_path)
    markdown_content = result.document.export_to_markdown()
    
    num_tokens = count_tokens_via_server(markdown_content, server_url)
    THRESHOLD = 100#102_400 
    
    if num_tokens <= THRESHOLD:
        # THE FIX: We specifically ask for an ARRAY of objects to get all 10 records
        prompt = (
            "Analyze the following medical document. It contains multiple patient records. "
            "Extract details for EVERY patient found and return them ONLY as a valid JSON array. "
            "Schema for each object: [patient_id, dob, icd10_codes, vitals, triage_priority, summary].\n\n"
            f"RECORDS:\n{markdown_content}"
        )

        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            },
            timeout=120 # Increased for multi-patient generation
        )
        
        raw_data = response.json()
        # Parse content string into actual JSON list
        try:
            content_str = raw_data['choices'][0]['message']['content']
            parsed_json = json.loads(content_str)
            # Handle case where model wraps array in a root key
            if isinstance(parsed_json, dict) and len(parsed_json) == 1:
                parsed_json = list(parsed_json.values())[0]
            return {"lane": "DIRECT", "data": parsed_json, "count": len(parsed_json)}
        except:
            return {"lane": "ERROR", "content": "Failed to parse JSON response."}
    else:
        meta_json_str = get_auto_metadata(markdown_content, server_url)
        file_meta = json.loads(meta_json_str)
        indexer = MedicalIndexer(db_folder='medical_db')
        # Process all your existing PDFs
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_text(markdown_content)
        metadata_list = [file_meta for _ in range(len(chunks))]
        indexer.add_documents(chunks, metadata_list=metadata_list)
        # Save it to disk
        indexer.save()
        return {"lane": "RAG", "content": "File too large."}

def batch_process_folder(input_dir, server_url="http://viropa:8001", model_name="gemma-3-27b"):
    """Loops through all PDFs in a folder and saves one master JSON."""
    all_results = []
    files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    
    for filename in files:
        file_path = os.path.join(input_dir, filename)
        result = process_document_mixed_pipeline(file_path, server_url, model_name)
        if result["lane"] == "DIRECT":
            all_results.extend(result["data"])
            
    with open("master_extraction.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"✅ Batch complete. {len(all_results)} patients saved to master_extraction.json")

# Usage
batch_process_folder("./raw_records/")