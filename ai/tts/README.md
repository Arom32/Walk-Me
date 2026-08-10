# ai/tts

강원도 사투리 TTS (CosyVoice + kangwon 체크포인트).

**이 폴더의 코드는 `cosyvoice` conda env(transformers==4.51.3 고정)에서만 돌아갑니다.** `ai/llm`이 요구하는 `transformers>=5.12.1`(gemma-4 로딩용)과는 같은 프로세스에 절대 공존할 수 없어서, `pipeline.py`/backend는 이 폴더를 직접 import하지 않고 `ai/tts/subprocess_client.py`를 통해 `synthesize.py`를 별도 프로세스로 실행합니다 (자세한 배경은 `ai/README.md`의 "환경" 섹션 참고). `synthesize.py`를 서브프로세스 진입점으로 직접 호출하고 싶으면:
```bash
conda run -n cosyvoice python synthesize.py "합성할 문장" --out outputs/test.wav
```

## 음성이 깨질 때

학습 스펙과 추론 패키지 버전이 다른 경우(v1/v3 혼용 등)가 대부분입니다. 원인 비교표·상세 진단은 [`doc/tts-voice-quality.md`](../../doc/tts-voice-quality.md) 참고. `synthesize.py`가 모델 폴더를 보고 v1/v3를 감지하고, v3가 아니면 경고를 냅니다.

**올바른 kangwon 구성 (v3):**

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

Drive의 `kangwon.zip`을 그대로 풀어 넣으면 위 파일이 전부 포함되어 있습니다 — 베이스 onnx/yaml을 따로 받아올 필요 없습니다.

## 환경 (학습과 동일 권장)

| 항목 | 팀원 |
|------|------|
| OS | Ubuntu / WSL |
| env | `conda activate cosyvoice` |
| Python | 3.10 |
| torch | 2.3.1+cu121 |
| transformers | 4.51.3 |

**처음 만들 때:**

```bash
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
cd ai/tts
pip install -r requirements.txt
```

`requirements.txt`가 `--extra-index-url https://download.pytorch.org/whl/cu121`로 torch/torchaudio cu121 빌드를 받아옵니다. 버전은 `Model_TTS/CosyVoice/requirements.txt`(학습 때 쓴 전체 목록) 기준으로 고정되어 있으니 임의로 올리지 마세요 — deepspeed/onnxruntime-gpu/modelscope 등도 그 버전 그대로 맞아야 합니다.

Windows Python 3.13 + 최신 torch 는 추가 위험 요인입니다.

## 품질 확인

```bat
cd ai\tts
python smoke_test.py
```

1. `outputs/smoke_self_clone.wav` — 프롬프트와 같은 문장 자기복제  
   - 이것도 깨지면: **버전/환경** 문제 (문장 길이와 무관) → [`doc/tts-voice-quality.md`](../../doc/tts-voice-quality.md) 참고  
   - 괜찮으면: RAG 사투리 전체 문장으로 진행 (`--text "..."`)

문장은 **자르지 않습니다.**

## 프롬프트 (AI Hub dialect)

- wav: `prompts/st_set2_collectorgw185_speakergw1744_63_9.wav` (16kHz mono)
- text:  
  `아까 내가 사이즈 먹고 그를 때부터 분멩이 택택할 거라고 했는데 나한테 어림도 웂으니까 하나 더 큰 거 주서요`
  (`synthesize.py`가 v3용 `<|endofprompt|>` 프리픽스를 자동으로 붙입니다)

## 파이프라인 (RAG + 사투리 + TTS)

`pipeline.py`/backend는 `walkme-llm` env에서 실행합니다 (이 TTS 폴더의 `cosyvoice` env가 아닙니다 — 자세한 건 `ai/README.md` "환경" 섹션 참고). TTS는 내부에서 `cosyvoice` env를 서브프로세스로 자동 호출하므로 따로 activate할 필요는 없습니다.

```bat
conda activate walkme-llm
cd ai

REM 1) kangwon 가중치(v3) → models\kangwon\
REM 2) 프롬프트 wav → prompts\

python pipeline.py "속초 가볼 만한 곳" --places-only --with-llm --tts
```

텍스트만 (WSL 없이도 가능):

```bat
python pipeline.py "속초 가볼 만한 곳" --places-only --with-llm
```

TTS만 직접 (cosyvoice env에서, v3 kangwon):

```bat
conda activate cosyvoice
cd ai/tts
python smoke_test.py --text "여기에 사투리 답변 전체"
```

API (walkme-llm env에서):

```bat
conda activate walkme-llm
cd backend
set PYTHONPATH=%CD%;%CD%\..\ai
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

`POST /guide` body 예: `{"question":"속초 가볼 만한 곳","tts":true}`
