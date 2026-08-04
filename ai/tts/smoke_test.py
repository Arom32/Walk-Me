"""TTS 품질 스모크 테스트.

1) 프롬프트 wav와 동일한 문장 자기복제
   → 깨지면: CosyVoice 버전(v1 vs v3) 또는 Python/torch 환경 문제
2) --text 로 관광 사투리 전체 문장 합성 (자르지 않음)

권장:
  conda activate cosyvoice   # Python 3.10, torch 2.3.1
  models/kangwon = CosyVoice-300M-SFT(v1) yaml/tokenizer + FT 가중치

사용:
  cd ai/tts
  python smoke_test.py
  python smoke_test.py --text "속초에서 가볼 만한 곳으로는 속초등대가 있습니."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TTS_DIR = Path(__file__).resolve().parent
AI_DIR = TTS_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))
if str(TTS_DIR) not in sys.path:
    sys.path.insert(0, str(TTS_DIR))

CANONICAL_DIALECT = (
    "아까 내가 사이즈 먹고 그를 때부터 분멩이 택택할 거라고 했는데 "
    "나한테 어림도 웂으니까 하나 더 큰 거 주서요"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=None, help="합성 문장 (없으면 자기복제)")
    parser.add_argument("--out", default=None, help="출력 wav")
    args = parser.parse_args()

    try:
        from tts.synthesize import (
            DEFAULT_PROMPT_TRANSCRIPT,
            DEFAULT_PROMPT_WAV,
            _resolve_model_dir,
            detect_cosyvoice_family,
            synthesize,
        )
    except ImportError as e:
        # Windows에 구버전 synthesize.py 만 있는 경우
        print("[오류] tts/synthesize.py 가 최신이 아닙니다.")
        print("  Mac/repo 의 ai/tts/synthesize.py 를 Windows 에 덮어쓰세요.")
        print("  필요 함수: detect_cosyvoice_family, prompt_text_for_family")
        raise SystemExit(f"ImportError: {e}") from e

    model_dir = _resolve_model_dir()
    family = detect_cosyvoice_family(model_dir)
    print("=" * 60)
    print("model_dir:", model_dir, "family=", family)
    print("프롬프트 wav:", DEFAULT_PROMPT_WAV, "exists=", DEFAULT_PROMPT_WAV.exists())
    print("코드 전사문:", DEFAULT_PROMPT_TRANSCRIPT)
    if DEFAULT_PROMPT_TRANSCRIPT != CANONICAL_DIALECT:
        print("[경고] 전사문이 AI Hub dialect 와 다릅니다.")
    if family == "v3":
        print(
            "[경고] CosyVoice3 감지 — 팀 학습은 300M-SFT(v1) 입니다. "
            "깨지면 models/kangwon 을 v1 구성으로 바꾸세요. (README 참고)"
        )
    print("=" * 60)

    text = args.text or CANONICAL_DIALECT
    out = args.out
    if out is None:
        tag = "self_clone" if args.text is None else "custom"
        out = str(TTS_DIR / "outputs" / f"smoke_{tag}.wav")

    path = synthesize(text, out_path=out)
    print("결과:", path)
    print("self_clone 이 깨지면 문장 문제가 아니라 체크포인트/환경 문제입니다.")


if __name__ == "__main__":
    main()
