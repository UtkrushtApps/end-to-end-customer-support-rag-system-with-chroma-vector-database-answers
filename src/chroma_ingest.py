import json
from chromadb import Client
from chromadb.config import Settings
from tqdm import tqdm

def create_and_ingest(collection_name: str, input_path: str):
    client = Client(Settings(chroma_api_impl="rest", chroma_server_host="localhost", chroma_server_http_port=8000))
    collection = client.get_or_create_collection(collection_name, embedding_function=None)
    with open(input_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    batch_size = 64
    embeddings = []
    ids = []
    metadatas = []
    documents = []
    for i, chunk in tqdm(enumerate(chunks), total=len(chunks)):
        embeddings.append(chunk['embedding'])
        ids.append(chunk['id'])
        metadatas.append({
            'source': chunk['source'],
            'chunk_id': chunk['chunk_id']
        })
        documents.append(chunk['text'])
        if len(embeddings) == batch_size or i == len(chunks) - 1:
            collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
            embeddings, ids, metadatas, documents = [], [], [], []
    print(f"Ingested {len(chunks)} chunks into collection '{collection_name}'.")

if __name__ == "__main__":
    create_and_ingest('support-docs', 'data/chunks_embedded.json')
