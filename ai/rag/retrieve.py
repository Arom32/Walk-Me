"""강원 관광 RAG 검색."""

from __future__ import annotations

import re
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

        if doc_type == "place":
            exact = self._find_exact_place_match(query, region)
            if exact is not None and exact.id not in {d.id for d in docs}:
                docs.insert(0, exact)
                docs = docs[:k]

        return docs

    def _find_exact_place_match(
        self, query: str, region: str | None
    ) -> RetrievedDoc | None:
        """벡터 검색 top-k가 놓칠 수 있는 정확한 장소명을, 이름 목록을
        직접 훑어서 찾는다 (예: "속초항"이 벡터 검색에서 밀려나는 경우).
        임베딩 계산이 필요 없는 메타데이터 조회라 비용이 낮다."""
        where: dict = {"doc_type": "place"}
        if region:
            where = {"$and": [{"doc_type": "place"}, {"region": region}]}
        got = self.collection.get(where=where, include=["metadatas"])
        ids = got.get("ids") or []
        metadatas = got.get("metadatas") or []

        q_norm = re.sub(r"\s+", "", query)
        best_id, best_len = None, 0
        for doc_id, meta in zip(ids, metadatas):
            name = (meta or {}).get("place_name") or ""
            name_norm = re.sub(r"\s+", "", name)
            if len(name_norm) < 3:
                continue
            if name_norm in q_norm and len(name_norm) > best_len:
                best_id, best_len = doc_id, len(name_norm)

        if best_id is None:
            return None
        fetched = self.collection.get(ids=[best_id], include=["documents", "metadatas"])
        fetched_docs = fetched.get("documents") or []
        fetched_metas = fetched.get("metadatas") or []
        if not fetched_docs:
            return None
        return RetrievedDoc(
            id=best_id,
            text=fetched_docs[0] or "",
            distance=0.0,
            metadata=fetched_metas[0] or {},
        )


def format_context(docs: list[RetrievedDoc]) -> str:
    if not docs:
        return "(검색된 관광 정보 없음)"
    blocks = []
    for i, d in enumerate(docs, 1):
        blocks.append(f"[참고 {i}]\n{d.text}")
    return "\n\n".join(blocks)
