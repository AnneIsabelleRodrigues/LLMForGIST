import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from typing import List
import csv

from app.config import COLLECTION_NAME, VECTOR_DIMENSION, MISTRAL_API_KEY, EMBEDDING_MODEL

embedder = SentenceTransformer(EMBEDDING_MODEL)

def load_documents_for_embedding(folder_path: str) -> List[str]:
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Pasta não encontrada: {folder_path}")

    supported_exts = (".txt", ".md", ".csv", ".jsonl", ".pdf")
    all_texts = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()

            if ext not in supported_exts:
                continue

            try:
                content = []

                if ext == ".pdf":
                    reader = PdfReader(file_path)
                    for page in reader.pages:
                        content.append(page.extract_text())

                elif ext == ".csv":
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if any(row):
                                content.append(", ".join(row))

                else:  # .txt, .md, .jsonl, etc.
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read().splitlines()

                for line in content:
                    stripped_line = line.strip()
                    if stripped_line:
                        all_texts.append(stripped_line)

            except Exception as e:
                print(f"[Aviso] Falha ao ler {file_path}: {e}")
                continue

    unique_texts = list(set(all_texts))

    return unique_texts

def chunk_texts(texts: List[str], chunk_size: int = 1024, chunk_overlap: int = 100) -> List[str]:

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text("\n\n".join(texts))

    return [chunk.strip() for chunk in chunks if chunk.strip()]

def init_qdrant(collection_name=COLLECTION_NAME, url=None, api_key=None, recreate=False):
    client = QdrantClient(url=url, api_key=api_key)
    if recreate:
        client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=Distance.COSINE)
        )

    return client

def embed_texts(texts):
    vectors = embedder.encode(texts, convert_to_tensor=False).tolist()
    return vectors


def add_documents(client: QdrantClient, texts):
    vectors = embed_texts(texts)

    points = [
        PointStruct(
            id=idx,
            vector=vector,
            payload={"text": text}
        )
        for idx, (vector, text) in enumerate(zip(vectors, texts))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)

def search_documents(client: QdrantClient, query: str, mistral_api_key=MISTRAL_API_KEY, limit=3):
    query_embedding = embedder.encode(query).tolist()

    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=limit
    )

    return [
        {
            "score": hit.score,
            "text": hit.payload.get("text")
        }
        for hit in hits
    ]