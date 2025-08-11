# Solution Steps

1. Source or create realistic support documents (FAQs, guides, manuals) and place them in 'data/documents/'.

2. Implement a preprocessing script ('src/preprocess.py') that reads, cleans, and de-duplicates the documents, outputting a cleaned document file.

3. Implement a chunking script ('src/chunking.py') for splitting cleaned text into overlapping chunks with robust metadata, saving as 'data/chunks.json'.

4. Implement embedding logic ('src/embedding.py') using SentenceTransformers to embed each text chunk and save embeddings into 'data/chunks_embedded.json'.

5. Write a Chroma ingestion script ('src/chroma_ingest.py') that connects to Chroma on localhost:8000, creates a collection, and uploads embeddings, documents, and metadata in batches.

6. Develop retrieval logic ('src/retrieval.py') to embed incoming queries, perform semantic search in Chroma, and return top-k most relevant chunks with metadata and score.

7. Implement context assembly and robust prompting ('src/prompting.py') to format system/user/assistant prompts that present context with explicit citations.

8. Build a generation module ('src/generation.py') that loads a local LLM (using transformers pipeline) and generates answers from the composed prompt.

9. Write an evaluation script ('src/evaluate.py') to execute end-to-end retrieval and generation for sampled queries, calculating and logging recall@k, precision@k, and query latency.

10. Run the pipeline end-to-end: preprocess documents, chunk and embed, ingest into Chroma, retrieve for sample queries, generate cited answers, and report evaluation metrics.

