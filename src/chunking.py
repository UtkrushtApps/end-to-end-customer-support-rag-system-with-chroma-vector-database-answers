import json
from typing import List, Dict

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap  # Overlap to preserve context
    return chunks

def chunk_documents(cleaned_docs_path: str, output_path: str, chunk_size: int = 300, overlap: int = 50):
    with open(cleaned_docs_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    all_chunks = []
    chunk_id = 0
    for doc in docs:
        doc_chunks = chunk_text(doc['text'], chunk_size, overlap)
        for i, chunk in enumerate(doc_chunks):
            all_chunks.append({
                'id': f"{doc['source']}_chunk_{i}",
                'source': doc['source'],
                'chunk_id': i,
                'text': chunk
            })
            chunk_id += 1
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Chunked {len(docs)} documents into {len(all_chunks)} chunks.")

if __name__ == "__main__":
    chunk_documents('data/cleaned_docs.json', 'data/chunks.json', chunk_size=75, overlap=10)
