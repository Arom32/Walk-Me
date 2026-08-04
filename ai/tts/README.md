# ai/tts

강원도 사투리 TTS (CosyVoice + kangwon 체크포인트).

## 음성이 깨질 때 — 가장 흔한 원인

정확한 학습 스펙/체크포인트 선정 경위는 `Model_TTS/CLAUDE.md` 10번 섹션 참고. 추론 패키지 버전이 실제 학습 베이스와 다르면 음성이 깨집니다.

| | 실제 학습 (정상 기준) | 호환 안 되는 패키지 |
|--|----------------------|-------------------|
| 베이스 | **CosyVoice 3.0 (Fun-CosyVoice3-0.5B)** | CosyVoice-300M-SFT (v1) |
| yaml | `cosyvoice3.yaml` | `cosyvoice.yaml` |
| tokenizer | `speech_tokenizer_v3.onnx` | `speech_tokenizer_v1.onnx` |
| 프롬프트 텍스트 | `You are a helpful assistant.<\|endofprompt\|>...` (자동 처리) | dialect 문장만 |
| vocoder | CosyVoice3 CausalHiFT | hift / hifigan (v1) |

v1(CosyVoice-300M-SFT) 기반 Full SFT 시도는 이 프로젝트에서 노이즈가 심해 실패로 판정, 폐기됐습니다 (`docs/attempt1_cosyvoice1_failure.md`). 이후 v3 + LoRA로 전환해 성공한 게 지금의 kangwon 체크포인트입니다.

`synthesize.py` 가 모델 폴더를 보고 v1/v3를 감지하고, v3가 아니면 경고를 냅니다.

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
   - 이것도 깨지면: **버전/환경** 문제 (문장 길이와 무관)  
   - 괜찮으면: RAG 사투리 전체 문장으로 진행 (`--text "..."`)

문장은 **자르지 않습니다.**

## 프롬프트 (AI Hub dialect)

- wav: `prompts/st_set2_collectorgw185_speakergw1744_63_9.wav` (16kHz mono)
- text:  
  `아까 내가 사이즈 먹고 그를 때부터 분멩이 택택할 거라고 했는데 나한테 어림도 웂으니까 하나 더 큰 거 주서요`
  (`synthesize.py`가 v3용 `<|endofprompt|>` 프리픽스를 자동으로 붙입니다)

## 파이프라인 (RAG + 사투리 + TTS)

```bat
cd ai

REM 1) kangwon 가중치(v3) → models\kangwon\
REM 2) 프롬프트 wav → prompts\

python pipeline.py "속초 가볼 만한 곳" --places-only --with-llm --tts
```

텍스트만 (WSL 없이도 가능):

```bat
python pipeline.py "속초 가볼 만한 곳" --places-only --with-llm
```

TTS만 직접 (가능하면 WSL cosyvoice + v3 kangwon):

```bat
cd ai/tts
python smoke_test.py --text "여기에 사투리 답변 전체"
```

API:

```bat
cd backend
set PYTHONPATH=%CD%;%CD%\..\ai
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

`POST /guide` body 예: `{"question":"속초 가볼 만한 곳","tts":true}`
