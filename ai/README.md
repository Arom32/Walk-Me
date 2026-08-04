# Walk-Me AI 파이프라인
<<<<<<< Updated upstream

```text
질문 → rag (관광 사실) → llm LoRA/템플릿 (사투리) → tts CosyVoice (음성)
```

## 설계 원칙 (LLM 실험 결론)

프롬프트만으로 사투리·관광 가이드를 시키면 실패한다.

=======
```text
질문 → rag (관광 사실) → llm LoRA (사투리) → tts CosyVoice (음성)
질문 → rag (관광 사실) → llm LoRA/템플릿 (사투리) → tts CosyVoice (음성)
```
## 설계 원칙 (LLM 실험 결론)
프롬프트만으로 사투리·관광 가이드를 시키면 실패한다.
>>>>>>> Stashed changes
| 시도 | 결과 | Walk-Me에 대한 함의 |
|------|------|---------------------|
| Llama3-8B + 사투리 system prompt | 환각, 어색한 어미, 영어 누출, 반복 | 제품 경로로 쓰지 않음 (충청 등 타 방언 프롬프트도 동일) |
| EEVE-Korean Instruct | 한국어는 자연스러우나 사투리는 가짜/혼재 | RAG를 EEVE로 대체하지 않음 |
| Gemma2 방언 FT (제주 중심 변환기) | 표준↔방언 변환용; 강원·관광 사실과 별개 | 선택적 실험만; 기본 스택 아님 |
<<<<<<< Updated upstream

**공식 분리:** RAG = 사실(장소명) · LoRA/템플릿 = 말투 · CosyVoice = 음성.  
기본 모델은 `gemma-4-E2B-it` + 팀 강원 LoRA. gemma-2는 폐기, gemma-4-12B는 크기상 보류.

## CLI

```bat
cd ai
python pipeline.py "속초 가볼 만한 곳" --places-only --with-llm --tts
```

TTS 없이 텍스트만:

```bat
python pipeline.py "속초 가볼 만한 곳" --places-only --with-llm
```

## API

=======
**공식 분리:** RAG = 사실(장소명) · LoRA/템플릿 = 말투 · CosyVoice = 음성.  
기본 모델은 `gemma-4-E2B-it` + 팀 강원 LoRA. gemma-2는 폐기, gemma-4-12B는 크기상 보류.
## CLI
>>>>>>> Stashed changes
```bat
cd backend
set PYTHONPATH=%CD%;%CD%\..\ai
uvicorn src.main:app --port 8000
```

- `GET /health`
- `POST /guide` — `{"question":"속초 가볼 만한 곳","with_llm":true,"tts":true}`
- `GET /guide/audio/{filename}` — 합성 wav

## 사전 준비

| 경로 | 내용 |
|------|------|
| `ai/llm/lora_output/final/` | Gemma LoRA |
| `ai/rag/data/chroma/` | 관광 인덱스 |
<<<<<<< Updated upstream
| `ai/tts/models/kangwon/` | CosyVoice **v1(300M-SFT)** 체크포인트 (`cosyvoice.yaml` + `speech_tokenizer_v1`) |
| `ai/tts/prompts/*.wav` | zero-shot 프롬프트 |

TTS가 깨지면 `ai/tts/README.md` 참고. CosyVoice3 패키지로 돌리면 학습과 불일치합니다.
품질 확인: `cd ai/tts && python smoke_test.py`
=======
| `ai/tts/models/kangwon/` | CosyVoice 체크포인트 |
| `ai/tts/prompts/*.wav` | zero-shot 프롬프트 |
>>>>>>> Stashed changes
