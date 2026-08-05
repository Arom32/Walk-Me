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

## 환경 (conda 2개로 분리, 필수)

`google/gemma-4-E2B-it`(LLM)는 `transformers>=5.12.1`이 있어야 로딩되는데, CosyVoice는 학습 재현을 위해 `transformers==4.51.3`에 고정돼 있다 — 두 요구사항이 한 Python 프로세스 안에서 절대 동시에 만족될 수 없다. 그래서 conda env를 완전히 둘로 나눈다:

| env | 용도 | 설치 |
|---|---|---|
| `walkme-llm` | RAG + LLM(LoRA) + backend + `pipeline.py` 실행 | `pip install -r ai/llm/requirements.txt -r ai/rag/requirements.txt -r backend/requirements.txt` |
| `cosyvoice` | TTS(CosyVoice) 전용 | `ai/tts/requirements.txt` (아래 "TTS 모델 준비" 참고) |

`pipeline.py`와 backend는 `walkme-llm` env에서 실행하면 된다. TTS는 `ai/tts/subprocess_client.py`가 내부적으로 `conda run -n cosyvoice python synthesize.py ...`를 서브프로세스로 호출해 처리하므로, **`cosyvoice` env를 따로 activate할 필요는 없고 conda에 그 env가 존재하기만 하면 된다.**

과거에 `except Exception`으로 LLM 로딩 실패가 조용히 삼켜져서 RAG 템플릿 문장으로만 "성공"한 것처럼 보이던 문제가 있었다 — 지금은 실패 시 `[경고] LoRA 사투리 변환 실패, RAG 템플릿으로 폴백: ...`가 찍히니, 이 경고가 보이면 `walkme-llm` env가 아니라 `cosyvoice` env(또는 transformers가 낡은 다른 env)에서 실행 중이라는 뜻이다.

## CLI

```bash
conda activate walkme-llm
cd ai
python pipeline.py "속초 가볼 만한 곳" --places-only --with-llm --tts
```

TTS 없이 텍스트만:

```bash
python pipeline.py "속초 가볼 만한 곳" --places-only --with-llm
```

## API

```bash
conda activate walkme-llm
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

TTS 전용 conda 환경(`cosyvoice`)도 한 번 만들어둬야 합니다 — `pipeline.py`/backend가 `walkme-llm` env에서 실행되면서 이 env를 서브프로세스로 자동 호출합니다 (위 "환경" 섹션 참고). 직접 activate해서 쓸 일은 smoke test 때뿐입니다:

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
