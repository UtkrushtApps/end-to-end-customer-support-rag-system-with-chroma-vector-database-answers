import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def embed_chunks(input_path: str, output_path: str, model_name: str = 'all-MiniLM-L6-v2'):
    with open(input_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    model = SentenceTransformer(model_name)
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64, normalize_embeddings=True)
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings[i].tolist()
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Embedded {len(chunks)} chunks.")

if __name__ == "__main__":
    embed_chunks('data/chunks.json', 'data/chunks_embedded.json')
