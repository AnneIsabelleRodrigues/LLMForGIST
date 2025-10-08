from sentence_transformers import SentenceTransformer

from config import QDRANT_CLUSTER_URL, QDRANT_API_KEY, HUGGINGFACE_TOKEN
from models.dataset_loader import load_local_dataset
from models.base_model import load_base_model
from models.fine_tune import fine_tune
from models.vector_db import init_qdrant, add_documents, load_texts_from_file
from models.rag_pipeline import retrieve_relevant_docs, generate_chat_answer

if __name__ == '__main__':

    tokenizer, model = load_base_model()

    # dataset = load_local_dataset("data/dataset")
    #
    # model = fine_tune(model, tokenizer, dataset)

    client = init_qdrant(url=QDRANT_CLUSTER_URL, api_key=QDRANT_API_KEY)

    texts = load_texts_from_file("data/documents/summary.txt")
    add_documents(client, texts)

    query = "Em poucas palavras, o que é GIST?"
    docs = retrieve_relevant_docs(client, query)
    print(docs)
    answer = generate_chat_answer(model, tokenizer, query, docs)

    print("\nResposta:\n", answer)
