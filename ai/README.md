# Walk-Me AI 파이프라인

```text
질문 → rag (관광 사실) → llm LoRA/템플릿 (사투리) → tts CosyVoice (음성)
```

## 설계 원칙 (LLM 실험 결론)

프롬프트만으로 사투리·관광 가이드를 시키면 실패한다.

| 시도 | 결과 | Walk-Me에 대한 함의 |
|------|------|---------------------|
| Llama3-8B + 사투리 system prompt | 환각, 어색한 어미, 영어 누출, 반복 | 제품 경로로 쓰지 않음 (충청 등 타 방언 프롬프트도 동일) |
| EEVE-Korean Instruct | 한국어는 자연스러우나 사투리는 가짜/혼재 | RAG를 EEVE로 대체하지 않음 |
| Gemma2 방언 FT (제주 중심 변환기) | 표준↔방언 변환용; 강원·관광 사실과 별개 | 선택적 실험만; 기본 스택 아님 |

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

```bat
cd backend
set PYTHONPATH=%CD%;%CD%\..\ai
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

- `GET /health`
- `POST /guide` — `{"question":"속초 가볼 만한 곳","with_llm":true,"tts":true}`
- `GET /guide/audio/{filename}` — 합성 wav

## 사전 준비

| 경로 | 내용 |
|------|------|
| `ai/llm/lora_output/final/` | Gemma LoRA |
| `ai/rag/data/chroma/` | 관광 인덱스 |
| `ai/tts/models/kangwon/` | CosyVoice **v3 (Fun-CosyVoice3-0.5B)** 체크포인트 (`cosyvoice3.yaml` + `speech_tokenizer_v3` + `CosyVoice-BlankEN/`) |
| `ai/tts/prompts/*.wav` | zero-shot 프롬프트 |

### TTS 모델 준비 (`ai/tts/models/kangwon/`)

Drive의 `kangwon.zip`을 `ai/tts/models/kangwon/`에 그대로 풀면 아래 구성이 전부 포함되어 있어 따로 받을 게 없습니다:

```text
models/kangwon/
  cosyvoice3.yaml
  speech_tokenizer_v3.onnx
  campplus.onnx
  CosyVoice-BlankEN/           ← Qwen 토크나이저 디렉토리
  llm.pt                       ← 파인튜닝 결과
  flow.pt
  hift.pt
```

TTS 전용 conda 환경도 필요합니다 (백엔드/RAG/LLM과는 별도):

```bash
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
cd ai/tts
pip install -r requirements.txt
```

풀어놓은 뒤 반드시 아래로 정상 동작 확인:

```bash
cd ai/tts
python smoke_test.py
```

`outputs/smoke_self_clone.wav`가 깨지면 문장 길이 문제가 아니라 **버전/환경 불일치**입니다 (v1 패키지가 섞여 있는 경우가 흔함). `ai/tts/README.md`의 트러블슈팅 표 참고.
