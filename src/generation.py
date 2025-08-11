from transformers import pipeline
import torch
import os

def instantiate_llm():
    device = 0 if torch.cuda.is_available() else -1
    # Using meta-llama/Llama-2-7b-chat-hf or similar would require heavier infra, so use distilgpt2 for demo.
    pipe = pipeline(
        "text-generation",
        model='distilgpt2',
        device=device,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=50256  # For GPT2
    )
    return pipe

def generate_rag_answer(llm_pipe, prompt: str) -> str:
    output = llm_pipe(prompt, truncation=True)[0]["generated_text"]
    # Extract only the answer after the prompt if extra text is present
    ans = output[len(prompt):].strip() if output.startswith(prompt) else output.strip()
    return ans.split('\n')[0]  # Return only the first line (short answer)
