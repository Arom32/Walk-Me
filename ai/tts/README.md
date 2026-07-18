# ai/tts

강원도 사투리 TTS 추론 코드입니다. 학습 자체는 별도 저장소에서 완료되었고, 여기에는 그 결과물로 실제 추론을 돌리는 데 필요한 **최소한의 것만** 옮겨왔습니다 — `Model_TTS` 전체(235GB)가 아니라 코드 2.9MB + 참조 파일 몇 개입니다.

## 여기 있는 것

- `cosyvoice/` — CosyVoice 추론 코드 (Apache 2.0, `LICENSE-cosyvoice` 참고)
- `third_party/Matcha-TTS/matcha/` — CosyVoice가 내부적으로 쓰는 flow-matching 컴포넌트 (MIT, `LICENSE-matcha` 참고)
- `requirements.txt` — 추론에만 필요한 의존성 (`Model_TTS/CosyVoice/requirements.txt`에서 학습/UI용 무거운 패키지 뺀 버전)
- `inference_example.py` — 최소 추론 예제 (`AutoModel` → `inference_zero_shot`)

## 여기 없는 것 (Drive에서 받아야 함)

체크포인트(5.1G)와 프롬프트 wav는 git에 올리지 않았습니다. 체크포인트는 파일 크기 때문에(개별 파일이 GitHub 100MB 제한을 넘음), 프롬프트 wav는 AI Hub 원본 데이터라 이용정책상(제3자 제공 금지, 비상업적 목적만 허용 — https://www.aihub.or.kr/intrcn/guid/usagepolicy.do) git처럼 누구나 접근 가능한 곳에 두면 안 되기 때문입니다.

### 1) 체크포인트 → `models/kangwon/`

- Drive 경로: `Corner-TTF/TTS-Model/` 
- kangwon.zip (약 4.5 GB)
- 다음 파일들을 그대로 `ai/tts/models/kangwon/` 밑에 풀어넣으면 됩니다 
- 드라이브에서 다운 받은 파일로 압축 해제 후 덮어 씌워도 됩니다.

| 파일 | 크기 |
|---|---|
| `llm.pt` | 1.9G |
| `flow.pt` | 1.3G |
| `speech_tokenizer_v3.onnx` | 925M |
| `CosyVoice-BlankEN/` (폴더 그대로) | 947M |
| `hift.pt` | 80M |
| `campplus.onnx` | 27M |
| `cosyvoice3.yaml` | 8.0K |

라이선스: Apache 2.0 (HuggingFace `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` 모델 카드 기준) — 재배포/파생 문제없음.

### 2) 프롬프트 wav → `prompts/`

- Drive 경로: `Corner-TTF/TTS-Model/` 
- 파일: `st_set2_collectorgw185_speakergw1744_63_9.wav` ( 302KB )

- 발화 텍스트(zero-shot 추론 시 `prompt_text`로 그대로 사용): `아까 내가 사이즈 먹고 그를 때부터 분멩이 택택할 거라고 했는데 나한테 어림도 웂으니까 하나 더 큰 거 주서요`

### 받은 후

```
ai/tts/models/kangwon/   ← 체크포인트 파일들
ai/tts/prompts/          ← wav 파일
```

이렇게 채워지면 `conda activate cosyvoice`(또는 `requirements.txt`로 새로 만든 환경)에서 `python inference_example.py` 실행 시 바로 동작합니다.
