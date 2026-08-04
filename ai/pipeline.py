"""Walk-Me 가이드 파이프라인: RAG → 사투리 LLM → (선택) CosyVoice TTS.

사용:
  cd ai
  python pipeline.py "속초 가볼 만한 곳" --places-only --with-llm --tts

TTS만 (이미 만든 문장):
  python pipeline.py --tts-only "속초등대를 먼저 가보면 좋습니."
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

AI_DIR = Path(__file__).resolve().parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))


def _free_llm() -> None:
    """VRAM 확보를 위해 RAG LLM 언로드 (있을 때만)."""
    try:
        import rag.ask as ask_mod
        import torch

        ask_mod._model = None
        ask_mod._tokenizer = None
        ask_mod._loaded_with_lora = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[pipeline] LLM VRAM 해제")
    except Exception as e:
        print(f"[pipeline] LLM 해제 스킵: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-Me RAG + dialect + TTS")
    parser.add_argument("question", nargs="?", default="속초 가볼 만한 곳")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--region", default=None)
    parser.add_argument("--places-only", action="store_true", default=True)
    parser.add_argument("--with-llm", action="store_true", default=True)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--tts", action="store_true", help="사투리 답변을 wav로 합성")
    parser.add_argument("--tts-only", default=None, help="이 문장만 TTS (RAG 생략)")
    parser.add_argument(
        "--out",
        default=None,
        help="wav 출력 경로 (기본: ai/tts/outputs/guide_*.wav)",
    )
    args = parser.parse_args()

    if args.tts_only:
        from tts.synthesize import synthesize

        path = synthesize(args.tts_only, out_path=args.out)
        print("=" * 60)
        print("[TTS only]")
        print(args.tts_only)
        print("audio:", path)
        return

    from rag.ask import answer_question

    result = answer_question(
        args.question,
        k=args.k,
        region=args.region,
        places_only=args.places_only,
        with_llm=args.with_llm,
        no_lora=args.no_lora,
    )

    print("=" * 60)
    print("질문:", result["question"])
    if result["region"]:
        print("지역:", result["region"])
    print("=" * 60)
    print("[표준어]")
    print(result["standard"])
    print("=" * 60)
    print("[사투리 답변]")
    print(result["answer"])
    print("=" * 60)

    if args.tts:
        _free_llm()
        from tts.synthesize import synthesize

        # 생성한 사투리 답변 전체를 그대로 읽음 (자르지 않음)
        speak = result["answer"]
        print("[TTS 입력]")
        print(speak)
        path = synthesize(speak, out_path=args.out)
        print("[음성]")
        print(path)
        result["audio_path"] = str(path)


if __name__ == "__main__":
    main()
