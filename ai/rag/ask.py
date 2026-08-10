"""강원 관광 RAG 질의.

사용 (repo 루트 또는 ai/ 에서):
  cd ai
  python -m rag.ask "속초 가볼 만한 곳" --places-only --region 속초
  python -m rag.ask "속초 가볼 만한 곳" --places-only --with-llm
  python -m rag.ask "속초 가볼 만한 곳" --places-only --with-llm --no-lora
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

AI_DIR = Path(__file__).resolve().parents[1]
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from rag.retrieve import RetrievedDoc, TravelRetriever, format_context

_model = None
_tokenizer = None
_loaded_with_lora: bool | None = None


def _resolve_model_paths() -> tuple[str, str]:
    """팀원 설정(src.config)을 기본으로 쓰고, env가 있으면 덮어씀."""
    import os

    base = os.environ.get("BASE_MODEL", "").strip()
    lora = os.environ.get("LORA_PATH", "").strip()
    repo = Path(__file__).resolve().parents[2]
    try:
        sys.path.insert(0, str(repo / "backend"))
        from src.config import settings

        if not base:
            base = settings.BASE_MODEL
        if not lora:
            lora = str(settings.lora_path())
    except Exception:
        if not base:
            base = "google/gemma-4-E2B-it"
        if not lora:
            lora = str(repo / "ai" / "llm" / "lora_output" / "final")
    return base, lora


def _adapter_base_model(lora_dir: str) -> str | None:
    cfg = Path(lora_dir) / "adapter_config.json"
    if not cfg.exists():
        return None
    try:
        return json.loads(cfg.read_text(encoding="utf-8")).get(
            "base_model_name_or_path"
        )
    except Exception:
        return None


def load_llm(use_lora: bool = True):
    global _model, _tokenizer, _loaded_with_lora
    if _model is not None and _loaded_with_lora == use_lora:
        import torch

        if torch.cuda.is_available() and next(_model.parameters()).device.type != "cuda":
            print("[LLM] CPU에 대기 중이던 모델을 GPU로 복귀")
            _model.to("cuda")
        return _model, _tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base, lora = _resolve_model_paths()
    lora_path = Path(lora) if lora else None

    if use_lora and lora_path and lora_path.exists():
        adapter_base = _adapter_base_model(str(lora_path))
        if adapter_base and adapter_base != base:
            print(
                f"[경고] LoRA는 '{adapter_base}'용인데 BASE_MODEL은 '{base}'입니다.\n"
                f"       adapter_config.json 기준으로 BASE_MODEL을 맞춥니다."
            )
            base = adapter_base
        print(f"[LLM] base={base}")
        print(f"[LLM] LoRA={lora_path}")
    else:
        print(f"[LLM] base={base} (LoRA 없음)")
        if use_lora and lora_path and not lora_path.exists():
            print(f"[경고] LoRA 경로 없음: {lora_path}")
            use_lora = False

    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    quant_kwargs = {}
    if torch.cuda.is_available():
        # gemma-4-E2B-it은 텍스트+비전+오디오 타워를 다 가진 멀티모달 모델이라
        # bf16 풀로드만으로 ~11.4GB를 써서(12GB 카드엔 거의 꽉 참) TTS가 쓸 VRAM이
        # 안 남는다. 4bit로 줄여 여유를 만든다.
        from llm.model_config import bnb_config

        quant_kwargs["quantization_config"] = bnb_config
    model = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        **quant_kwargs,
    )
    if use_lora and lora_path and lora_path.exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(lora_path))
    else:
        print("[LLM] base 모델만 사용")

    model.eval()
    _model, _tokenizer, _loaded_with_lora = model, tokenizer, use_lora
    return model, tokenizer


def release_llm_gpu_memory() -> None:
    """LLM을 지우지 않고 GPU에서만 CPU로 내린다 (host RAM에 유지).

    상주 서버(backend)에서 TTS 차례에 VRAM을 비워줄 때 쓴다. `_model = None`으로
    지우면 다음 요청 때 디스크에서 처음부터(수십 GB) 다시 읽어야 하므로, 대신
    `to("cpu")`로만 내려서 디스크 재접근 없이 다음 `load_llm()` 호출에서 GPU로
    복귀시킬 수 있게 한다.
    """
    import gc

    import torch

    if _model is None:
        return
    if torch.cuda.is_available() and next(_model.parameters()).device.type == "cuda":
        _model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        print("[LLM] GPU 메모리 반납 (모델은 CPU RAM에 유지, 디스크 재접근 없음)")


def generate(
    prompt: str,
    max_new_tokens: int = 160,
    temperature: float = 0.3,
    use_lora: bool = True,
) -> str:
    import torch

    model, tokenizer = load_llm(use_lora=use_lora)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": 1.2,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    ).strip()


def extract_place_cards(docs: list[RetrievedDoc]) -> list[dict]:
    cards = []
    seen = set()
    for d in docs:
        name = (d.metadata.get("place_name") or "").strip()
        if not name:
            m = re.search(r"장소:\s*(.+)", d.text)
            name = m.group(1).strip() if m else ""
        if not name or name in seen:
            continue
        if any(x in name for x in ("아파트", "터미널", "휴게소", "신고", "주차장")):
            continue
        seen.add(name)
        card = {
            "name": name,
            "region": d.metadata.get("region") or "",
            "visit_type": d.metadata.get("visit_type") or "",
            "address": "",
            "extra": [],
        }
        for line in d.text.splitlines():
            if line.startswith("주소:"):
                card["address"] = line.replace("주소:", "", 1).strip()
            if line.startswith("관련 음식") or line.startswith("평균 만족도"):
                card["extra"].append(line)
        cards.append(card)
    return cards


def build_standard_answer(question: str, cards: list[dict]) -> str:
    if not cards:
        return "검색된 관광 정보가 없어서 확실히 추천하기 어렵습니다."

    lines = [
        f"질문 '{question}' 기준으로, 실제 여행 기록에 많이 나온 장소입니다.",
        "추천 장소:",
    ]
    for i, c in enumerate(cards[:5], 1):
        bit = f"{i}. {c['name']}"
        if c.get("visit_type"):
            bit += f" ({c['visit_type']})"
        if c.get("address"):
            bit += f" — {c['address']}"
        lines.append(bit)
    lines.append("위 장소 이름은 그대로 두고 방문하시면 됩니다.")
    return "\n".join(lines)


def _object_particle(name: str) -> str:
    """을/를 — 마지막 글자 받침 유무."""
    if not name:
        return "를"
    code = ord(name[-1])
    if 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 != 0:
        return "을"
    return "를"


def build_short_for_convert(cards: list[dict], region: str | None = None) -> str:
    """표준어 짧은 추천 문장 (장소명 포함)."""
    names = [c["name"] for c in cards[:5]]
    if not names:
        return "추천할 장소를 찾지 못했습니다."
    where = region or "강원"
    joined = ", ".join(names)
    particle = _object_particle(names[0])
    return (
        f"{where}에서 가볼 만한 곳으로는 {joined}이 있습니다. "
        f"특히 {names[0]}{particle} 먼저 가보면 좋습니다."
    )


def assemble_dialect_with_names(names: list[str], region: str | None) -> str:
    """장소명은 코드가 직접 넣고, 말미만 사투리 틀로 고정 (환각 방지)."""
    where = region or "강원"
    joined = ", ".join(names)
    first = names[0]
    particle = _object_particle(first)
    return (
        f"{where}에서 가볼 만한 곳으로는 {joined}이 있습니다. "
        f"특히 {first}{particle} 먼저 가보면 좋습니다."
    )


def build_masked_dialect_template(names: list[str], region: str | None) -> tuple[str, dict[str, str]]:
    """장소명을 플레이스홀더로 가린 문장 틀. LoRA가 어미만 사투리로 바꾸게 하고
    장소명 자체는 절대 못 건드리게 해서 환각을 막는다."""
    where = region or "강원"
    placeholders = [f"[장소{i + 1}]" for i in range(len(names))]
    joined = ", ".join(placeholders)
    particle = _object_particle(names[0])  # 을/를 판단은 실제 이름 기준
    template = (
        f"{where}에서 가볼 만한 곳으로는 {joined}이 있습니다. "
        f"특히 {placeholders[0]}{particle} 먼저 가보면 좋습니다."
    )
    return template, dict(zip(placeholders, names))


def dialectize_body(masked_template: str, mapping: dict[str, str]) -> tuple[str, bool]:
    """본문 틀의 어미만 LoRA로 사투리화. 실패하면 ("", False) — 호출부가 표준 틀로 폴백."""
    prompt = (
        "아래 문장의 어미(문장 끝맺음)만 강원도 사투리 말투로 바꿔라.\n"
        "대괄호로 감싼 [장소N] 표시는 절대 지우거나 바꾸지 말고 그대로 유지할 것.\n"
        "새로운 문장·정보·장소를 추가하지 말고 원문 문장 구조와 순서를 유지할 것.\n"
        "변환 문장만 출력.\n\n"
        f"원문: {masked_template}\n\n"
        "사투리:"
    )
    try:
        out = generate(prompt, max_new_tokens=160, temperature=0, use_lora=True)
        out = out.strip().splitlines()[0].strip()
    except Exception as e:
        print(f"[경고] 본문 사투리 변환 실패, 표준 틀로 폴백: {e!r}")
        return "", False

    expected = set(mapping.keys())
    found = set(re.findall(r"\[장소\d+\]", out))
    bad = any(x in out for x in ("http", "영어", "〈", "<P"))
    if bad or not out or len(out) > 300 or found != expected:
        return "", False

    for placeholder, name in mapping.items():
        out = out.replace(placeholder, name)
    return out, True


def convert_to_dialect(
    standard_short: str,
    must_keep_names: list[str],
    use_lora: bool,
    region: str | None = None,
) -> tuple[str, bool]:
    """
    LoRA에 장소명을 넣지 않는다.
    - 본문: 장소명 고정 삽입 + 사투리 말미 템플릿 (항상 성공)
    - (선택) LoRA로 장소명 없는 짧은 인사만 앞에 붙임
    """
    names = [n for n in must_keep_names[:5] if n]
    if not names:
        return standard_short, False

    body = assemble_dialect_with_names(names, region)  # 안전한 표준어 폴백, 항상 준비해둠

    if not use_lora:
        return body, True

    masked_template, mapping = build_masked_dialect_template(names, region)
    dialect_body, body_ok = dialectize_body(masked_template, mapping)
    if body_ok:
        body = dialect_body

    # 장소명 없는 문장만 LoRA에 넘겨 말투 샘플 → 실패해도 body는 유지
    opener_src = (
        f"{region or '강원'} 여행 오셨으면 천천히 둘러보시면 좋습니다."
    )
    prompt = (
        "아래 한 문장만 강원도 사투리 말투로 바꿔라.\n"
        "고유명사·영어·새 장소 이름 넣지 말 것. 변환 문장만 출력.\n\n"
        f"원문: {opener_src}\n\n"
        "사투리:"
    )
    try:
        opener = generate(
            prompt, max_new_tokens=60, temperature=0, use_lora=True
        )
        opener = opener.strip().splitlines()[0].strip()
        bad = any(
            x in opener
            for x in ("신발", "헐겁", "http", "영어", "〈", "<P")
        )
        # 너무 길거나 빈 응답이면 버림
        if bad or not opener or len(opener) > 80:
            return body, True
        return f"{opener}\n{body}", True
    except Exception as e:
        print(f"[경고] LoRA 사투리 변환 실패, RAG 템플릿으로 폴백: {e!r}")
        return body, True


def answer_question(
    question: str,
    *,
    k: int = 5,
    region: str | None = None,
    places_only: bool = True,
    with_llm: bool = True,
    no_lora: bool = False,
) -> dict:
    """질문 → RAG 검색 → (선택) 사투리 답변. API/파이프라인용."""
    if not region:
        for r in ("속초", "강릉", "춘천", "양양", "평창", "원주", "동해", "삼척"):
            if r in question:
                region = r
                break

    if places_only is True or any(
        x in question for x in ("가볼", "명소", "관광", "해변", "시장", "장소")
    ):
        places_only = True

    retriever = TravelRetriever()
    docs = retriever.search(
        question,
        k=k,
        doc_type="place" if places_only else None,
        region=region,
    )
    context = format_context(docs)
    cards = extract_place_cards(docs)
    standard = build_standard_answer(question, cards)
    short = build_short_for_convert(cards, region=region)
    names = [c["name"] for c in cards]

    dialect = standard
    dialect_ok = False
    if with_llm:
        dialect, dialect_ok = convert_to_dialect(
            short, names, use_lora=not no_lora, region=region
        )

    return {
        "question": question,
        "region": region,
        "places": cards[:5],
        "context": context,
        "standard": standard,
        "short": short,
        "dialect": dialect,
        "dialect_ok": dialect_ok,
        "answer": dialect if with_llm else standard,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gangwon travel RAG ask")
    parser.add_argument("question", nargs="?", default="속초에서 가볼 만한 곳 추천해줘")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--region", default=None)
    parser.add_argument("--places-only", action="store_true")
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="사투리 변환 시 LoRA 미사용(base만). LoRA가 이상할 때 권장",
    )
    args = parser.parse_args()

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
        print("지역 필터:", result["region"])
    print("=" * 60)
    print(result["context"])
    print("=" * 60)
    print("[표준어 초안 — RAG]")
    print(result["standard"])
    print("=" * 60)

    if args.with_llm:
        print("[변환용 짧은 원문]")
        print(result["short"])
        print("=" * 60)
        if result["dialect_ok"]:
            print("[사투리 답변]")
        else:
            print("[사투리 변환 실패 → 표준어 초안 유지]")
        print(result["dialect"])
    else:
        print("(사투리 변환은 --with-llm)")


if __name__ == "__main__":
    main()
