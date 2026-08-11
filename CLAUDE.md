# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> 응답은 한국어로

## 프로젝트: 워메! (Walk-Me!)

팀 TTF. 관광객의 질문에 지역 사투리로 답하는 대화형 AI 관광 도슨트 서비스.

최종 구상은 촬영(이미지)·음성·텍스트 입력을 모두 포함하지만, **현재 구현 우선순위는 텍스트 질문 경로**다. 이미지·음성 입력은 추후 계획이며, 지금 코드(`ai/pipeline.py`, backend `/guide` API)도 텍스트 질문 기준으로만 동작한다.

## 파이프라인 / 핵심 설계 원칙

```
질문 → ai/rag    관광 사실 검색 (Chroma, 표준어 초안 + 장소명 고정)
     → ai/llm    팀 Gemma LoRA로 말투만 사투리화
     → ai/tts    CosyVoice zero-shot 음성 합성
     → backend   /guide API로 텍스트·wav 통합 제공
```

**RAG = 사실(장소명) / LoRA = 말투 / TTS = 음성**, 역할을 엄격히 분리한다.

- 프롬프트만으로 LLM에게 사투리·관광 가이드를 시키는 방식은 실험 결과 환각(가짜 장소, 가짜 사투리)이 심해 제품 경로로 채택하지 않음 — 그래서 RAG가 사실을 표준어 초안으로 고정하고, LoRA는 말투 변환만 담당하도록 분리했다.
- 기본 모델: `google/gemma-4-E2B-it` + 팀 강원 LoRA. (gemma-2 계열은 폐기, gemma-4-12B는 크기 문제로 보류)

## 디렉토리 구조

| 경로 | 역할 |
|---|---|
| `ai/rag/` | 관광 사실 검색. AI Hub 여행 CSV → 문서화 → Chroma 벡터 검색 |
| `ai/llm/` | 사투리 변환 LoRA 학습/추론 코드 |
| `ai/tts/` | CosyVoice 기반 음성 합성 (kangwon 파인튜닝 체크포인트) |
| `ai/pipeline.py` | RAG → LLM → TTS 통합 CLI |
| `backend/src/` | FastAPI 서버 (`main.py`), 설정(`config.py`), DB 모델 |
| `frontend/` | 아직 스텁 상태 |

공식 모델/데이터 경로:

| 용도 | 경로 |
|---|---|
| RAG 벡터 DB | `ai/rag/data/chroma/` |
| LLM LoRA | `ai/llm/lora_output/final/` (환경변수 `LORA_PATH`로 override) |
| TTS 체크포인트 | `ai/tts/models/kangwon/` |
| TTS 프롬프트 wav | `ai/tts/prompts/*.wav` |

## 자주 쓰는 명령어

RAG 인덱싱/검색:
```bash
cd ai
python -m rag.index
python -m rag.ask "속초 가볼 만한 곳" --places-only --region 속초
```

전체 파이프라인 (RAG + 사투리 + TTS):
```bash
cd ai
python pipeline.py "속초 가볼 만한 곳" --places-only --with-llm --tts
```
TTS 없이 텍스트만 확인하려면 `--tts`를 빼면 된다.

TTS 품질 확인 (사실상의 테스트 — self-clone 결과가 깨지는지 먼저 확인):
```bash
cd ai/tts
python smoke_test.py
```

백엔드 실행:
```bash
cd backend
set PYTHONPATH=%CD%;%CD%\..\ai
uvicorn src.main:app --host 0.0.0.0 --port 8000
```
- `GET /health`, `POST /guide` (`{"question": "...", "with_llm": true, "tts": true}`), `GET /guide/audio/{filename}`
- DB 포함 통째로 띄우려면 리포 루트에서 `docker-compose up` (Postgres + web)

**환경은 두 개로 나뉜다**: TTS는 `conda activate cosyvoice` (Python 3.10, torch 2.3.1+cu121) — 학습 때와 동일한 버전으로 고정되어 있으니 `ai/tts/requirements.txt`를 임의로 올리지 말 것. 백엔드/RAG/LLM은 이 conda env와 무관한 별도 환경에서 돌아간다.

## TTS 배경 (요약)

- 처음엔 v1(CosyVoice-300M-SFT)으로 Full SFT를 시도했으나 노이즈가 심해 실패, 폐기됨.
- **v3(Fun-CosyVoice3-0.5B) + LLM LoRA가 최종 확정 스펙.** 최종 체크포인트는 loss/acc 같은 정량 지표가 아니라 실제로 들어보고(리스닝 테스트) 선정했다 — 이 프로젝트에서 정량 지표는 최종본 판단 기준으로 신뢰하기 어려웠다.
- v1과 v3는 yaml·tokenizer·vocoder 구성이 전부 다르다. 섞어 쓰면 음성이 깨지므로 `ai/tts/models/kangwon/`은 항상 v3 구성(`cosyvoice3.yaml` + `speech_tokenizer_v3.onnx` + `CosyVoice-BlankEN/`)으로 유지한다.
- 알려진 미해결 증상: `family=v3`로 정상 감지되는데도 self-clone(자기복제) 결과가 깨지는 사례가 보고된 적 있다. 원인은 아직 확정되지 않았고 계속 조사 중이므로, 여기서 특정 해결 방향을 단정하지 않는다. TTS가 이상하면 우선 `python smoke_test.py`로 재현되는지 확인하고 최신 상태를 팀에 확인할 것.

## 참고 문서

- `ai/README.md` — 파이프라인/설계 원칙, CLI, API 사용법
- `ai/tts/README.md` — TTS 환경 구성, 음성 깨짐 트러블슈팅
- `ai/rag/README.md` — RAG 데이터 경로, 인덱싱 절차
