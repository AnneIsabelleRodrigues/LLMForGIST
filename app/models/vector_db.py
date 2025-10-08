import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
import torch

from app.config import COLLECTION_NAME, VECTOR_DIMENSION

def load_texts_from_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    with open(filepath, "r", encoding="cp1252", errors="replace") as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]
    return texts

def init_qdrant(collection_name=COLLECTION_NAME, url=None, api_key=None):
    client = QdrantClient(url=url, api_key=api_key)
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=Distance.COSINE)
    )
    return client

def embed_texts(texts):
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = encoder.encode(texts)
    return vectors

def add_documents(client, texts):
    vectors = embed_texts(texts)

    payload = [{"text": t} for t in texts]

    client.upload_collection(
        collection_name=COLLECTION_NAME,
        vectors=vectors.tolist(),
        payload=payload
    )