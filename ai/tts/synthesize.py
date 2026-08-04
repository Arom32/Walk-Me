"""CosyVoice(kangwon) 음성 합성 래퍼.

사전 준비:
  ai/tts/models/kangwon/   ← llm.pt, flow.pt, ...
  ai/tts/prompts/*.wav     ← zero-shot 프롬프트
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import numpy as np

TTS_DIR = Path(__file__).resolve().parent

if str(TTS_DIR) not in sys.path:
    sys.path.insert(0, str(TTS_DIR))
_matcha = TTS_DIR / "third_party" / "Matcha-TTS"
if str(_matcha) not in sys.path:
    sys.path.insert(0, str(_matcha))

DEFAULT_MODEL_DIR = TTS_DIR / "models" / "kangwon"
DEFAULT_PROMPT_WAV = (
    TTS_DIR / "prompts" / "st_set2_collectorgw185_speakergw1744_63_9.wav"
)
DEFAULT_PROMPT_TRANSCRIPT = (
    "아까 내가 사이즈 먹고 그를 때부터 분멩이 택택할 거라고 했는데 "
    "나한테 어림도 웂으니까 하나 더 큰 거 주서요"
)
# CosyVoice3 전용. v1(300M-SFT)은 transcript 만 씀.
COSYVOICE3_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
DEFAULT_PROMPT_TEXT = COSYVOICE3_PROMPT_PREFIX + DEFAULT_PROMPT_TRANSCRIPT
DEFAULT_OUT_DIR = TTS_DIR / "outputs"

_cosyvoice = None


def _resolve_prompt_wav() -> Path:
    env = os.environ.get("TTS_PROMPT_WAV", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    if DEFAULT_PROMPT_WAV.exists():
        return DEFAULT_PROMPT_WAV
    prompts = TTS_DIR / "prompts"
    if prompts.exists():
        wavs = sorted(prompts.glob("*.wav"))
        if wavs:
            return wavs[0]
    raise FileNotFoundError(
        f"프롬프트 wav 없음:\n  {DEFAULT_PROMPT_WAV}\n  또는 env TTS_PROMPT_WAV"
    )


def _resolve_model_dir() -> Path:
    env = os.environ.get("TTS_MODEL_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_MODEL_DIR.resolve()


def detect_cosyvoice_family(model_dir: Path) -> str:
    """반환: 'v1' | 'v2' | 'v3' | 'unknown'

    팀원 학습 노트 기준 강원 파인튜닝은 CosyVoice-300M-SFT (v1,
    speech_tokenizer_v1, cosyvoice.yaml). CosyVoice3 패키지로 돌리면 깨짐.
    """
    if (model_dir / "cosyvoice.yaml").exists():
        return "v1"
    if (model_dir / "cosyvoice2.yaml").exists():
        return "v2"
    if (model_dir / "cosyvoice3.yaml").exists():
        return "v3"
    return "unknown"


def _warn_if_version_mismatch(model_dir: Path, family: str) -> None:
    has_v1_tok = (model_dir / "speech_tokenizer_v1.onnx").exists()
    has_v3_tok = (model_dir / "speech_tokenizer_v3.onnx").exists()
    has_blank = (model_dir / "CosyVoice-BlankEN").exists()
    print(f"[TTS] family={family} tokenizer_v1={has_v1_tok} tokenizer_v3={has_v3_tok} BlankEN={has_blank}")
    if family == "v3":
        print(
            "[TTS][경고] cosyvoice3.yaml 감지.\n"
            "  팀원 학습은 CosyVoice-300M-SFT (v1, speech_tokenizer_v1, cosyvoice.yaml) 입니다.\n"
            "  v3로 추론하면 음성이 깨질 수 있습니다.\n"
            "  → models/kangwon 을 v1 구성으로 맞추세요:\n"
            "     cosyvoice.yaml + speech_tokenizer_v1.onnx + llm.pt/flow.pt/hift.pt\n"
            "     (베이스: pretrained_models/CosyVoice-300M-SFT 의 onnx/yaml + FT 가중치)"
        )
    if family == "v1" and has_v3_tok and not has_v1_tok:
        print("[TTS][경고] v1 yaml 인데 tokenizer 가 v3 만 있습니다. 학습과 불일치합니다.")


def prompt_text_for_family(family: str, transcript: str | None = None) -> str:
    text = (transcript or DEFAULT_PROMPT_TRANSCRIPT).strip()
    if family == "v3":
        if "<|endofprompt|>" not in text:
            return COSYVOICE3_PROMPT_PREFIX + text
        return text
    # v1 / v2: endofprompt 없이 dialect 전사만
    if "<|endofprompt|>" in text:
        text = text.split("<|endofprompt|>", 1)[-1].strip()
    return text


def get_cosyvoice():
    global _cosyvoice
    if _cosyvoice is not None:
        return _cosyvoice

    model_dir = _resolve_model_dir()
    if not model_dir.exists():
        raise FileNotFoundError(f"TTS 모델 없음: {model_dir}")

    family = detect_cosyvoice_family(model_dir)
    _warn_if_version_mismatch(model_dir, family)

    from cosyvoice.cli.cosyvoice import AutoModel

    print(f"[TTS] model_dir={model_dir}")
    _cosyvoice = AutoModel(model_dir=str(model_dir))
    _cosyvoice._walkme_family = family  # type: ignore[attr-defined]
    return _cosyvoice


def _save_wav(path: Path, speech, sample_rate: int) -> None:
    import soundfile as sf

    wav = speech.detach().cpu().float()
    if wav.ndim == 2:
        if wav.shape[0] <= 4:
            wav = wav.mean(dim=0)
        else:
            wav = wav[:, 0]
    wav_np = wav.numpy()
    peak = float(np.max(np.abs(wav_np))) if wav_np.size else 0.0
    if peak > 1.0:
        wav_np = wav_np / peak
    sf.write(str(path), wav_np, sample_rate)


def synthesize(
    text: str,
    *,
    out_path: Path | str | None = None,
    prompt_text: str | None = None,
    prompt_wav: Path | str | None = None,
) -> Path:
    """생성 문장 그대로 합성 (잘라내지 않음)."""
    import torch

    text = (text or "").strip()
    if not text:
        raise ValueError("합성할 텍스트가 비어 있습니다.")

    # 줄바꿈만 공백으로 — 내용은 그대로
    tts_text = " ".join(line.strip() for line in text.splitlines() if line.strip())
    print(f"[TTS] text={tts_text!r}")

    cosy = get_cosyvoice()
    family = getattr(cosy, "_walkme_family", None) or detect_cosyvoice_family(
        _resolve_model_dir()
    )
    p_wav = Path(prompt_wav) if prompt_wav else _resolve_prompt_wav()
    env_prompt = os.environ.get("TTS_PROMPT_TEXT", "").strip()
    if prompt_text:
        p_text = prompt_text_for_family(family, prompt_text)
    elif env_prompt:
        p_text = prompt_text_for_family(family, env_prompt)
    else:
        p_text = prompt_text_for_family(family, DEFAULT_PROMPT_TRANSCRIPT)
    print(f"[TTS] prompt_text={p_text!r}")

    if out_path is None:
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEFAULT_OUT_DIR / f"guide_{uuid.uuid4().hex[:10]}.wav"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    pieces = []
    # 한국어는 전처리로 글자를 바꾸지 않음 — 생성 문장 그대로 토크나이즈
    use_frontend = not any("\uac00" <= ch <= "\ud7a3" for ch in tts_text)
    for chunk in cosy.inference_zero_shot(
        tts_text,
        p_text,
        str(p_wav),
        stream=False,
        text_frontend=use_frontend,
    ):
        speech = chunk["tts_speech"].detach().cpu().float()
        if speech.ndim == 2 and speech.shape[0] > 1:
            speech = speech.mean(dim=0, keepdim=True)
        dur = speech.shape[-1] / float(cosy.sample_rate)
        print(f"[TTS] chunk duration={dur:.2f}s")
        pieces.append(speech)

    if not pieces:
        raise RuntimeError("TTS 출력이 비어 있습니다.")

    merged = torch.cat(pieces, dim=-1)
    _save_wav(out_path, merged, cosy.sample_rate)
    print(f"[TTS] saved {out_path} ({merged.shape[-1] / cosy.sample_rate:.2f}s)")
    return out_path
