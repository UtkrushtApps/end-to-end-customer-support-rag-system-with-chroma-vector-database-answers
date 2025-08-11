import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
from typing import List, Dict, Tuple
import time

class RAGRetriever:
    def __init__(self, chroma_host='localhost', port=8000, collection_name='support-docs', model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.client = Client(Settings(chroma_api_impl="rest", chroma_server_host=chroma_host, chroma_server_http_port=port))
        self.collection = self.client.get_collection(collection_name, embedding_function=None)

    def embed_query(self, query: str):
        emb = self.model.encode([query], normalize_embeddings=True)
        return emb[0]

    def search(self, query: str, top_k: int = 4) -> Tuple[List[Dict], float]:
        start_time = time.time()
        query_emb = self.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k,
            include=["metadatas", "documents", "distances", "ids"])
        elapsed = time.time() - start_time
        retrieved = []
        for i in range(len(results['ids'][0])):
            retrieved.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        return retrieved, elapsed
