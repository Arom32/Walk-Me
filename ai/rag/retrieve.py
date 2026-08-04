"""강원 관광 RAG 검색."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

AI_DIR = Path(__file__).resolve().parents[1]
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from rag import resolve_chroma_dir
from rag.index import COLLECTION_NAME, DEFAULT_EMBED_MODEL


@dataclass
class RetrievedDoc:
    id: str
    text: str
    distance: float | None
    metadata: dict


class TravelRetriever:
    def __init__(
        self,
        chroma_dir: Path | None = None,
        embed_model: str = DEFAULT_EMBED_MODEL,
        collection_name: str = COLLECTION_NAME,
    ):
        import chromadb
        from chromadb.utils import embedding_functions

        self.chroma_dir = Path(chroma_dir) if chroma_dir else resolve_chroma_dir()
        if not self.chroma_dir.exists():
            raise FileNotFoundError(
                f"Chroma 없음: {self.chroma_dir}\n먼저: python -m rag.index"
            )

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embed_model,
        )
        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = client.get_collection(
            name=collection_name,
            embedding_function=ef,
        )

    def search(
        self,
        query: str,
        k: int = 5,
        doc_type: str | None = None,
        region: str | None = None,
    ) -> list[RetrievedDoc]:
        where = None
        clauses = []
        if doc_type:
            clauses.append({"doc_type": doc_type})
        if region:
            clauses.append({"region": region})
        if len(clauses) == 1:
            where = clauses[0]
        elif len(clauses) > 1:
            where = {"$and": clauses}

        kwargs = {
            "query_texts": [query],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        result = self.collection.query(**kwargs)
        docs: list[RetrievedDoc] = []
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for i, doc_id in enumerate(ids):
            docs.append(
                RetrievedDoc(
                    id=doc_id,
                    text=documents[i] or "",
                    distance=distances[i] if i < len(distances) else None,
                    metadata=metadatas[i] or {},
                )
            )
        return docs


def format_context(docs: list[RetrievedDoc]) -> str:
    if not docs:
        return "(검색된 관광 정보 없음)"
    blocks = []
    for i, d in enumerate(docs, 1):
        blocks.append(f"[참고 {i}]\n{d.text}")
    return "\n\n".join(blocks)
