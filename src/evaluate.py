import time
import json
from retrieval import RAGRetriever
from prompting import build_prompt
from generation import instantiate_llm, generate_rag_answer
from typing import List

def recall_at_k(retrieved_chunks: List[dict], gold_sources: List[str], k: int) -> float:
    hits = 0
    for chunk in retrieved_chunks[:k]:
        if chunk['metadata']['source'] in gold_sources:
            hits += 1
    return hits / min(len(gold_sources), k)

def precision_at_k(retrieved_chunks: List[dict], gold_sources: List[str], k: int) -> float:
    hits = 0
    for chunk in retrieved_chunks[:k]:
        if chunk['metadata']['source'] in gold_sources:
            hits += 1
    return hits / k

def gold_sources_from_query(query: str) -> List[str]:
    # A mapping from sample queries to source files
    gold = {
        "password": ["faq.txt"],
        "browsers": ["faq.txt"],
        "can't log in": ["troubleshooting_guide.txt", "faq.txt"],
        "update": ["product_manual.txt"],
        "help": ["faq.txt"]
    }
    q = query.lower()
    matched = []
    for key in gold:
        if key in q:
            matched.extend(gold[key])
    if not matched:
        matched = ["faq.txt"]  # Default for fallback
    return matched

def run_evaluation(top_k=4):
    retriever = RAGRetriever()
    llm_pipe = instantiate_llm()
    with open('sample_queries.txt', 'r', encoding='utf-8') as f:
        queries = [l.strip() for l in f if l.strip()]
    logs = []
    for query in queries:
        gold_sources = gold_sources_from_query(query)
        retrieved, latency = retriever.search(query, top_k)
        recall = recall_at_k(retrieved, gold_sources, top_k)
        precision = precision_at_k(retrieved, gold_sources, top_k)
        prompt_data = build_prompt(query, retrieved)
        answer = generate_rag_answer(llm_pipe, prompt_data['prompt'])
        print(f"\nQuery: {query}\nAnswer: {answer}\n[Recall@{top_k}: {recall:.2f} | Precision@{top_k}: {precision:.2f} | Latency: {latency:.3f}s]")
        logs.append({
            'query': query,
            'answer': answer,
            'recall_k': recall,
            'precision_k': precision,
            'latency': latency,
            'retrieved_metadata': prompt_data['metadata'],
        })
    with open('logs/eval_log.json', 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=2)
if __name__ == "__main__":
    import os
    os.makedirs('logs', exist_ok=True)
    run_evaluation()
