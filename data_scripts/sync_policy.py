import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from importlib import import_module
from importlib.util import find_spec

if find_spec("langchain_huggingface"):
    HuggingFaceEmbeddings = import_module("langchain_huggingface").HuggingFaceEmbeddings
else:
    HuggingFaceEmbeddings = import_module("langchain_community.embeddings").HuggingFaceEmbeddings

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from app.core.config import (
    COLLECTION_POLICIES,
    PROJECT_ROOT,
    QDRANT_URL,
    TEXT_EMBEDDING_MODEL,
    TEXT_VECTOR_SIZE,
)


def _load_policy_blocks() -> list[str]:
    policy_path = PROJECT_ROOT / "app" / "data" / "policy.txt"
    raw_policy = policy_path.read_text(encoding="utf-8")
    raw_policy = re.sub(r"\r\n?", "\n", raw_policy).strip()

    title_match = re.match(r"^(.*?)(?=\n\s*1\.)", raw_policy, flags=re.DOTALL)
    title = title_match.group(1).strip() if title_match else ""
    body = raw_policy[len(title):].strip() if title else raw_policy

    sections = [
        match.group(0).strip()
        for match in re.finditer(
            r"(?ms)^\s*\d+\.\s+.*?(?=^\s*\d+\.\s+|\Z)",
            body,
        )
    ]

    if sections:
        if title:
            sections[0] = f"{title}\n\n{sections[0]}"
        return [re.sub(r"\n{3,}", "\n\n", section).strip() for section in sections]

    return [block.strip() for block in raw_policy.split("\n\n") if block.strip()]


def sync_policies():
    print("=== Sync FShop policies to Qdrant ===")
    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Collection: {COLLECTION_POLICIES}")
    print(f"Embedding: {TEXT_EMBEDDING_MODEL} ({TEXT_VECTOR_SIZE} dims)")

    policies_data = _load_policy_blocks()
    if not policies_data:
        raise RuntimeError("No policy content found in app/data/policy.txt")

    client = QdrantClient(url=QDRANT_URL)
    embeddings = HuggingFaceEmbeddings(model_name=TEXT_EMBEDDING_MODEL)

    if client.collection_exists(COLLECTION_POLICIES):
        client.delete_collection(COLLECTION_POLICIES)

    client.create_collection(
        collection_name=COLLECTION_POLICIES,
        vectors_config=VectorParams(size=TEXT_VECTOR_SIZE, distance=Distance.COSINE),
    )

    documents = [
        Document(page_content=text, metadata={"source": "policy_manual", "chunk": index})
        for index, text in enumerate(policies_data, start=1)
    ]

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_POLICIES,
        embedding=embeddings,
    )
    vector_store.add_documents(documents)

    print(f"Synced {len(documents)} policy chunks successfully.")


if __name__ == "__main__":
    sync_policies()
