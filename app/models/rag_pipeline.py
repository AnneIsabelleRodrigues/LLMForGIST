from sentence_transformers import SentenceTransformer
import torch

from app.config import COLLECTION_NAME, EMBEDDING_MODEL

encoder = SentenceTransformer(EMBEDDING_MODEL)


def retrieve_relevant_docs(client, query, top_k=3):
    query_vector = encoder.encode(query, convert_to_tensor=False).tolist()

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )

    docs = [hit.payload["text"] for hit in results]
    return docs


def generate_chat_answer(model, tokenizer, query, docs):
    context = "\n".join(docs)

    # --- Formato de Prompt Otimizado para Qwen (ChatML) ---
    # É fundamental que este formato seja EXATAMENTE o que o modelo Qwen foi treinado.

    system_prompt = "Você é um assistente útil e preciso. Responda à pergunta do usuário estritamente baseado nas informações fornecidas no contexto."

    user_prompt = f"Contexto:\n{context}\n\nPergunta: {query}"

    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.8,
            do_sample=True,
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return response.strip()