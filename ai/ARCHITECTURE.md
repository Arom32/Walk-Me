"""
Walk-Me 역할 분담 (팀원 문서 정리)

## 텍스트
- gemma-2-2b-it: 사용 안 함
- gemma-4-12b: 너무 큼
- gemma-4-E2B/e4b + LoRA: 사투리 말투
- 프롬프트만으로는 관광 사실 할루시네이션 → RAG 필수

파이프라인:
  질문 → ai/rag (사실) → ai/llm LoRA (말투) → ai/tts (음성)

## 음성 (CosyVoice) — 버전 주의
- 데이터: AI Hub 중·노년층 방언 (강원 따라말하기), train ~68k / valid ~8.5k
- 학습 베이스: **CosyVoice-300M-SFT (v1)** + `speech_tokenizer_v1` + `cosyvoice.yaml`
- 추론도 반드시 **같은 v1 스택**. CosyVoice3(yaml/tokenizer/BlankEN)로 돌리면 깨짐
- 프롬프트 예시: st_set2_collectorgw185_speakergw1744_63_9
- 환경: Ubuntu/WSL + conda cosyvoice + Python 3.10 + torch 2.3.1
- 품질 확인: `ai/tts/smoke_test.py` (자기복제 → 관광 문장)
"""
