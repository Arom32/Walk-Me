"""강원 관광 문서 → Chroma 벡터 인덱스 빌드.

사용:
  cd ai
  pip install -r rag/requirements.txt
  python -m rag.index
  python -m rag.index --places-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AI_DIR = Path(__file__).resolve().parents[1]
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from rag import resolve_chroma_dir, resolve_processed_dir

DEFAULT_EMBED_MODEL = "jhgan/ko-sroberta-multitask"
COLLECTION_NAME = "gangwon_travel"


def load_jsonl(path: Path) -> list[dict]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def build_index(
    processed_dir: Path,
    chroma_dir: Path,
    embed_model: str,
    places_only: bool = False,
    reset: bool = True,
) -> int:
    import chromadb
    from chromadb.utils import embedding_functions

    place_path = processed_dir / "gangwon_place_docs.jsonl"
    trip_path = processed_dir / "gangwon_trip_docs.jsonl"
    if not place_path.exists():
        raise SystemExit(f"없음: {place_path} — 먼저 build_gangwon_rag_docs.py 실행")

    docs = load_jsonl(place_path)
    if not places_only and trip_path.exists():
        docs.extend(load_jsonl(trip_path))

    print(f"문서 수: {len(docs):,}")
    print(f"임베딩: {embed_model}")
    print(f"Chroma: {chroma_dir}")

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"기존 컬렉션 삭제: {COLLECTION_NAME}")
        except Exception:
            pass

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embed_model,
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for i, doc in enumerate(docs):
        doc_id = str(doc.get("id") or f"doc_{i}")
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        ids.append(doc_id)
        documents.append(text)
        meta = {
            "doc_type": str(doc.get("doc_type") or ""),
            "place_name": str(doc.get("place_name") or ""),
            "region": str(doc.get("region") or ""),
            "visit_type": str(doc.get("visit_type") or ""),
            "visit_count": int(doc.get("visit_count") or 0),
        }
        metadatas.append(meta)

    # Chroma batch limit 회피
    batch = 200
    for start in range(0, len(ids), batch):
        end = start + batch
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  indexed {min(end, len(ids)):,}/{len(ids):,}")

    print(f"✅ 완료: {collection.count():,} docs → {chroma_dir}")
    return collection.count()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Gangwon travel RAG index")
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--chroma-dir", type=Path, default=None)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--places-only", action="store_true")
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()

    processed = args.processed_dir or resolve_processed_dir()
    chroma = args.chroma_dir or resolve_chroma_dir()
    build_index(
        processed_dir=processed,
        chroma_dir=chroma,
        embed_model=args.embed_model,
        places_only=args.places_only,
        reset=not args.no_reset,
    )


if __name__ == "__main__":
    main()
