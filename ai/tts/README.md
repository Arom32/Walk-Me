# ai/tts

강원도 사투리 TTS (CosyVoice + kangwon 체크포인트).

## 음성이 깨질 때 — 가장 흔한 원인

팀원 학습 노트와 **추론 패키지 버전이 다르면** 음성이 깨집니다.

| | 팀원 학습 (정상 기준) | 깨지기 쉬운 패키지 |
|--|----------------------|-------------------|
| 베이스 | **CosyVoice-300M-SFT** | Fun-CosyVoice3 / BlankEN |
| yaml | `cosyvoice.yaml` | `cosyvoice3.yaml` |
| tokenizer | `speech_tokenizer_v1.onnx` | `speech_tokenizer_v3.onnx` |
| 프롬프트 텍스트 | dialect 문장만 | `You are a helpful assistant.<\|endofprompt\|>...` |
| vocoder | hift / hifigan (v1) | CosyVoice3 CausalHiFT |

`synthesize.py` 가 모델 폴더를 보고 v1/v3를 감지하고, v3면 경고를 냅니다.

**올바른 kangwon 구성 (v1):**

```text
models/kangwon/
  cosyvoice.yaml              ← 300M-SFT 쪽
  speech_tokenizer_v1.onnx
  campplus.onnx
  llm.pt                      ← 파인튜닝 결과
  flow.pt
  hift.pt                     ← 또는 hifigan.pt 를 hift.pt 로 복사/심볼릭
  (spk2info.pt 선택)
```

베이스 onnx/yaml 은 `~/Corner-ttf/CosyVoice/pretrained_models/CosyVoice-300M-SFT/` 에서 가져오고,
`llm.pt` / `flow.pt` / `hift.pt` 만 파인튜닝 `exp/...` average 결과로 교체하세요.

## 환경 (학습과 동일 권장)

| 항목 | 팀원 |
|------|------|
| OS | Ubuntu / WSL |
| env | `conda activate cosyvoice` |
| Python | 3.10 |
| torch | 2.3.1+cu121 |
| transformers | 4.51.3 |

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
- text (v1):  
  `아까 내가 사이즈 먹고 그를 때부터 분멩이 택택할 거라고 했는데 나한테 어림도 웂으니까 하나 더 큰 거 주서요`

## 파이프라인

```bat
REM 텍스트 (Windows OK)
cd ai
python pipeline.py "속초 가볼 만한 곳" --places-only --with-llm

REM TTS (가능하면 WSL cosyvoice + v1 kangwon)
python smoke_test.py --text "여기에 사투리 답변 전체"
```
