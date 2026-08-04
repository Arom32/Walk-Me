# ai/rag

강원 관광 데이터 RAG입니다. AI Hub 여행 CSV → 장소/코스 문서 → Chroma 검색.

팀 `ai/` 구조:

```text
ai/
  llm/   # 사투리 텍스트 LLM
  rag/   # 관광 사실 검색  ← 여기
  tts/   # CosyVoice 음성
```

## 여기 있는 것

| 경로 | 설명 |
|------|------|
| `extract_travel_csv.py` | AI Hub zip → `backend/data/extracted` |
| `build_gangwon_rag_docs.py` | CSV → `data/processed/*.jsonl` |
| `index.py` | jsonl → Chroma (`data/chroma`) |
| `retrieve.py` / `ask.py` | 검색 + (선택) 사투리 변환 |
| `data/processed/` | 생성 문서 (로컬) |
| `data/chroma/` | 벡터 DB (로컬, git 제외) |

## 사용

```bash
cd ai
pip install -r rag/requirements.txt

# 문서가 이미 있으면 인덱스만
python -m rag.index

# 검색
python -m rag.ask "속초 가볼 만한 곳" --places-only --region 속초

# 사투리 변환까지 (팀원 LoRA / env)
python -m rag.ask "속초 가볼 만한 곳" --places-only --with-llm
```

Windows:

```bat
cd C:\Users\DS\Downloads\Walk-Me-main\ai
python -m rag.ask "속초 가볼 만한 곳" --places-only --region 속초
```

## 데이터 경로

- 원본 CSV: `backend/data/raw`, `backend/data/extracted` (기존 위치 유지)
- RAG 산출물: `ai/rag/data/processed`, `ai/rag/data/chroma`
- Chroma만 Drive에서 받으려면 `.env`의 `CHROMA_DIR`로 지정

## 파이프라인 한눈에

```text
질문 → rag.ask (검색) → 표준어 초안(장소명 고정)
                      → (선택) ai/llm LoRA 사투리 변환
                      → (선택) ai/tts CosyVoice 음성
```
