from models.vector_db import embed_texts
from sentence_transformers import SentenceTransformer
import torch

from app.config import COLLECTION_NAME


def retrieve_relevant_docs(client, query, top_k=3):
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = encoder.encode([query])[0]
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    docs = [hit.payload["text"] for hit in results]
    return docs


def generate_chat_answer(model, tokenizer, query, docs):
    context = "\n\n".join(docs)

    prompt = f"""<|im_start|>system
        Você é um assistente útil que responde perguntas baseado no contexto fornecido.<|im_end|>
        <|im_start|>user
        Contexto:
        {context}
        
        Pergunta: {query}<|im_end|>
        <|im_start|>assistant
        """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return response.strip()