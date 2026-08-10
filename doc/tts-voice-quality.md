# TTS 음성이 깨질 때 — 가장 흔한 원인

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

`python smoke_test.py`로 재현되는지 먼저 확인하세요 (`ai/tts/README.md` "품질 확인" 참고). `outputs/smoke_self_clone.wav`가 깨지면 문장 길이 문제가 아니라 버전/환경 불일치입니다 (v1 패키지가 섞여 있는 경우가 흔함).

알려진 미해결 증상으로, `family=v3`로 정상 감지되는데도 self-clone(자기복제) 결과가 깨지는 사례가 보고된 적 있습니다. 원인은 아직 확정되지 않았고 계속 조사 중입니다 — 재현되면 팀에 최신 상태를 확인하세요.
