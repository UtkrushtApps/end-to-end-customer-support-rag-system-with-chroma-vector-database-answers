def build_prompt(query: str, contexts: list) -> dict:
    """
    Build the system/user/assistant formatted prompt for LLM generation.
    Ensures that references/citations to document sources and chunk_ids are present.
    """
    context_str = "\n\n".join([
        f"[Source: {c['metadata']['source']}, chunk: {c['metadata']['chunk_id']}] {c['document']}"
        for c in contexts
    ])
    system_prompt = (
        "You are a helpful customer support agent for a tech startup. Use ONLY the provided context to answer the user's question. "
        "Cite the relevant sources and chunk numbers in your answer by referencing [Source, chunk].\n" 
        "If the answer is not found in the context, say you do not know.\n\n"
        f"Context:\n{context_str}\n\n"
        f"User question: {query}\nAnswer: "
    )
    return {"prompt": system_prompt, "metadata": [c['metadata'] for c in contexts]}
