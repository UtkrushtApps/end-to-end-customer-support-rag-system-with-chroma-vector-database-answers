import glob
import os
import re
from typing import List, Dict

def read_documents(directory: str) -> List[Dict]:
    files = glob.glob(os.path.join(directory, '*'))
    documents = []
    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            text = f.read()
            documents.append({'source': os.path.basename(fpath), 'text': text})
    return documents

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    text = re.sub(r'\u201c|\u201d|\u2018|\u2019', '"', text)  # Normalize quotes
    return text.strip()

def deduplicate(docs: List[Dict]) -> List[Dict]:
    seen = set()
    unique_docs = []
    for doc in docs:
        t = doc['text']
        if t not in seen:
            unique_docs.append(doc)
            seen.add(t)
    return unique_docs

def preprocess_documents(directory: str) -> List[Dict]:
    docs = read_documents(directory)
    for doc in docs:
        doc['text'] = clean_text(doc['text'])
    docs = deduplicate(docs)
    return docs

def save_cleaned_docs(docs: List[Dict], output_path: str):
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    docs = preprocess_documents('data/documents/')
    save_cleaned_docs(docs, 'data/cleaned_docs.json')
    print(f"Preprocessed and saved {len(docs)} unique cleaned documents.")
