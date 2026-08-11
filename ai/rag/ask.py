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

    # CPU 동적 양자화(quantize_dynamic)의 ONEDNN 커널은 float32 입력만 받는다
    # (bf16을 넣으면 "data type of input should be float" 에러). 그래서 CPU
    # 경로는 float32로 로드한다 — 어차피 아래서 Linear 레이어를 int8로 양자화해
    # 최종 메모리는 float32 그대로 두는 것보다 훨씬 작아진다.
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
        if not torch.cuda.is_available():
            # CPU 머신: bitsandbytes 4bit(nf4)는 CUDA 커널 전용이라 못 쓴다.
            # 대신 LoRA를 base에 합친 뒤(merge_and_unload) 순수 torch 내장
            # 동적 int8 양자화를 적용 — 별도 패키지 설치 없이 Linear 레이어
            # 메모리를 bf16 대비 추가로 크게 줄인다.
            # quantize_dynamic은 기본값(inplace=False)이면 모델을 deepcopy한
            # 뒤 바꿔서 원본+사본이 동시에 떠 순간 메모리가 2배로 튄다
            # (32GB RAM 머신에서 실제로 OOM 발생) — inplace=True로 복사를 없앤다.
            import gc

            model = model.merge_and_unload()
            gc.collect()
            torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
            )
            gc.collect()
            print("[LLM] CPU 전용 환경: LoRA merge + int8 동적 양자화 적용")
    else:
        print("[LLM] base 모델만 사용")
        if not torch.cuda.is_available():
            import gc

            torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
            )
            gc.collect()
            print("[LLM] CPU 전용 환경: int8 동적 양자화 적용")

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


_CARD_EXTRA_PREFIXES = ("관련 음식", "평균 만족도", "평균 체류", "주요 활동", "방문 기록 수")


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
            if line.startswith(_CARD_EXTRA_PREFIXES):
                card["extra"].append(line)
        cards.append(card)
    return cards


_GENERIC_PLACE_NAMES = ("다이소", "이마트", "농협", "편의점")
_NEARBY_KEYWORDS = ("근처", "주변", "말고", "다른")


def match_question_place(question: str, cards: list[dict]) -> dict | None:
    """질문이 이미 검색된 특정 장소를 콕 집어 물은 건지 판별한다.

    공백을 제거하고 부분 문자열로 비교한다 ("외옹치 해변" ↔ "외옹치해변").
    2자 미만 장소명은 오탐(예: 장소명이 "거기"인 경우)을 막기 위해 제외하고,
    "근처"/"주변"처럼 다른 곳을 찾는 질문은 매칭하지 않는다.
    "경포"/"경포대"처럼 이름이 겹치면 가장 긴 이름이 우선한다.
    """
    q = re.sub(r"\s+", "", question)
    if any(k in q for k in _NEARBY_KEYWORDS):
        return None

    best: dict | None = None
    best_len = 0
    for card in cards:
        name = re.sub(r"\s+", "", card.get("name") or "")
        if len(name) < 3 or name in _GENERIC_PLACE_NAMES:
            continue
        if name not in q:
            continue
        if len(name) > best_len:
            has_detail = card.get("visit_type") or card.get("address") or card.get("extra")
            if not has_detail:
                continue
            best, best_len = card, len(name)
    return best


def card_field(card: dict, prefix: str) -> str:
    """extra 라인에서 '접두어: 값'의 값만 꺼낸다. 없으면 빈 문자열."""
    for line in card.get("extra", []):
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip() if ":" in line else ""
    return ""


_SMALLTALK_KEYWORDS = (
    "안녕",
    "반가워",
    "반갑",
    "고마워",
    "고맙",
    "감사",
    "수고",
    "잘가",
    "잘 가",
    "이름이 뭐",
    "누구",
    "뭐하고 있",
    "뭐해",
    "심심",
    "잘 지내",
)
_FOOD_KEYWORDS = ("음식", "맛집", "먹", "맛있", "식당", "배고")

_SMALLTALK_RESPONSES = (
    "어이구, 반갑습니더~ 나는 강원도서 나고 자란 사람이여. 여행 얘기라믄 뭐든 물어보이소.",
    "고맙긴 뭘, 나도 얘기하는 게 즐겁습니더. 어데 궁금한 데 있으믄 편하게 물어보이소.",
    "허허, 심심하믄 강원도 얘기나 좀 하까요? 맛집이든 갈 만한 데든 물어보이소.",
    "잘 지내지예~ 강원도 여행 준비하는 거라믄 내가 아는 만큼 알려드릴께요.",
)

_NO_MATCH_DISTANCE = 0.65  # 실측 기준: 정상 관광 질문 top-1 거리는 ~0.63 이하, 오프토픽은 ~0.69 이상

# 신뢰도 게이트에 걸린 질문(오프토픽/미지원)에 쓰는 응대. LLM 자유 생성은 실제로
# 써 보니 "게이저 타운" 같은 가짜 장소명이나 영어/마크다운 gibberish를 지어내는
# 사례가 확인돼서 채택하지 않는다 (환각 방지 원칙 위반) — 대신 스몰토크와 같은
# 패턴으로, 흔한 주제(날씨/나이/코딩 등)는 키워드로 감지해 그에 맞는 고정 문구를,
# 나머지는 일반 문구 풀에서 골라 다양성만 확보한다.
_NO_MATCH_CATEGORIES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("날씨", "비 와", "눈 와", "기온"),
        (
            "날씨는 지도 잘 모르겠고예, 강원도 여행 얘기면 자신 있습니더.",
            "일기예보는 지 담당이 아이라예. 어디 갈지는 잘 알려드릴 수 있어예.",
        ),
    ),
    (
        ("몇 살", "나이가", "생일"),
        (
            "나이는 비밀입니더~ 그것보다 강원도 여행 얘기나 하입시더.",
        ),
    ),
    (
        ("코드", "파이썬", "프로그래밍", "알고리즘"),
        (
            "그런 건 지가 하나도 모르겠습니더. 강원도 여행 얘기라믄 뭐든 물어보이소.",
        ),
    ),
    (
        ("심심", "아무말"),
        (
            "심심하믄 강원도 얘기나 하까요? 맛집이든 갈 만한 데든 물어보이소.",
        ),
    ),
)
_NO_MATCH_RESPONSES = (
    "아이고, 그건 지도 잘 모르겠습니더. 강원도 여행 얘기로 다시 물어봐 주이소.",
    "그거는 내 담당이 아이라가 잘 모르겠어예. 강원도 어디 갈지 물어보이소.",
    "그건 좀 어려운 질문이네예. 강원도 여행이라면 뭐든 답해드릴 수 있는데예.",
    "허허, 그거는 잘 모르겠고예. 강원도 관광지나 맛집 얘기 좀 해볼까예?",
    "음... 그건 지 전문이 아이라서예. 강원도 여행 궁금한 거 있으믄 편하게 물어보이소.",
    "그거는 대답하기 어렵네예. 강원도 여행 준비하는 거믄 제가 도와드릴 수 있어예.",
)


def pick_no_match_reply(question: str) -> str:
    for keywords, templates in _NO_MATCH_CATEGORIES:
        if any(k in question for k in keywords):
            return templates[hash(question) % len(templates)]
    return _NO_MATCH_RESPONSES[hash(question) % len(_NO_MATCH_RESPONSES)]


def classify_intent(question: str) -> str:
    """RAG 검색 전에 질문 종류를 나눈다. 사담은 RAG를 아예 안 태우고,
    음식 질문은 같은 RAG 검색 결과에서 다른 필드(관련 음식)를 답변에 쓴다."""
    q = question.strip()
    if len(q) <= 20 and any(k in q for k in _SMALLTALK_KEYWORDS):
        return "smalltalk"
    if any(k in q for k in _FOOD_KEYWORDS):
        return "food"
    return "general"


def smalltalk_reply(question: str) -> str:
    return _SMALLTALK_RESPONSES[hash(question) % len(_SMALLTALK_RESPONSES)]


def extract_foods(card: dict) -> list[str]:
    for line in card.get("extra", []):
        if line.startswith("관련 음식"):
            after = line.split(":", 1)[1] if ":" in line else ""
            return [x.strip() for x in after.split(",") if x.strip()]
    return []


def build_food_standard_answer(question: str, cards: list[dict]) -> tuple[str, list[tuple[str, list[str]]]]:
    """RAG가 찾은 장소들의 '관련 음식/구매' 필드만 그대로 옮겨 적는다 — 새 사실을 지어내지 않는다."""
    entries = [(c["name"], extract_foods(c)) for c in cards[:5]]
    entries = [(name, foods) for name, foods in entries if foods]

    if not entries:
        return (
            "검색된 장소들에서 음식 관련 기록을 찾지 못했습니다.",
            [],
        )

    lines = [
        f"질문 '{question}' 기준으로, 실제 방문 기록에 남아 있는 음식 정보입니다.",
        "관련 음식:",
    ]
    for name, foods in entries:
        lines.append(f"- {name}: {', '.join(foods[:4])}")
    lines.append("가시는 길에 참고해보세요.")
    return "\n".join(lines), entries


def assemble_food_dialect(entries: list[tuple[str, list[str]]], region: str | None) -> str:
    where = region or "강원"
    top_name, top_foods = entries[0]
    foods_joined = ", ".join(top_foods[:3])
    body = f"{where}서 뭐 좀 무봐야겠다카믄 {foods_joined} 겉은 게 있습니더. 특히 {top_name} 쪽서 마이 묵는다 카데."
    if len(entries) > 1:
        # 표준어 답변(build_food_standard_answer)은 entries를 최대 5개까지 다 보여주는데
        # 여기서 [1:3]으로 잘라서 사투리 답변만 식당이 덜 나오던 버그 — 표준어와 범위를 맞춘다.
        others = ", ".join(n for n, _ in entries[1:])
        body += f"\n{others}에도 먹을 게 있습니더."
    return body


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
    lines.append("여행하실 때 참고해보세요.")
    return "\n".join(lines)


def build_place_detail_standard(question: str, card: dict) -> str:
    """단일 장소에 대한 상세 답변. 문서에 있는 값만 그대로 옮겨 적는다 — 새 문장을 지어내지 않는다."""
    name = card["name"]
    lines = [f"질문 '{question}' 기준으로, 실제 여행 기록에 남아 있는 {name} 정보입니다."]
    head = name
    if card.get("visit_type"):
        head += f" ({card['visit_type']})"
    lines.append(head)
    if card.get("address"):
        lines.append(f"- 주소: {card['address']}")
    dwell = card_field(card, "평균 체류")
    if dwell:
        lines.append(f"- 평균 체류: {dwell}")
    satisfaction = card_field(card, "평균 만족도")
    if satisfaction:
        lines.append(f"- 평균 만족도: {satisfaction}")
    activity = card_field(card, "주요 활동")
    if activity:
        lines.append(f"- 주요 활동: {activity}")
    food = card_field(card, "관련 음식")
    if food:
        lines.append(f"- 관련 음식/구매: {food}")
    if len(lines) <= 2:
        lines.append("상세 기록이 많지 않아 참고용으로만 봐주세요.")
    else:
        lines.append("여행하실 때 참고해보세요.")
    return "\n".join(lines)


def _has_batchim(text: str) -> bool:
    if not text:
        return False
    code = ord(text[-1])
    return 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 != 0


def _object_particle(name: str) -> str:
    """을/를 — 마지막 글자 받침 유무."""
    if not name:
        return "를"
    return "을" if _has_batchim(name) else "를"


def _topic_particle(name: str) -> str:
    """은/는 — 마지막 글자 받침 유무."""
    if not name:
        return "는"
    return "은" if _has_batchim(name) else "는"


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
        f"특히 {names[0]}{particle} 추천합니다."
    )


def build_place_detail_body(
    card: dict, region: str | None = None
) -> tuple[str, str, dict[str, str]]:
    """단일 장소 상세 문장을 (표준어 본문, [장소N] 마스킹 틀, 매핑)으로 함께 만든다.

    이름뿐 아니라 유형/주소/만족도/활동 값까지 전부 [장소N]으로 마스킹해서
    LoRA에는 사실이 하나도 안 보이게 한다 (리스트 모드보다도 환각 여지가 적다).
    플레이스홀더는 최대 5개 — dialectize_body의 검증 실패율을 낮추기 위함.
    """
    name = card["name"]
    where = card.get("region") or region or "강원"
    visit_type = card.get("visit_type") or ""
    address = card.get("address") or ""
    satisfaction = card_field(card, "평균 만족도")
    activity = card_field(card, "주요 활동")
    food = card_field(card, "관련 음식")

    mapping: dict[str, str] = {}

    def slot(value: str) -> str:
        ph = f"[장소{len(mapping) + 1}]"
        mapping[ph] = value
        return ph

    name_ph = slot(name)
    if visit_type:
        type_ph = slot(visit_type)
        std_sentences = [f"{where}에 있는 {name}{_topic_particle(name)} {visit_type}입니다."]
        mask_sentences = [f"{where}에 있는 {name_ph}{_topic_particle(name)} {type_ph}입니다."]
    else:
        std_sentences = [f"{where}에 있는 {name}입니다."]
        mask_sentences = [f"{where}에 있는 {name_ph}입니다."]

    if len(mapping) < 5 and (address or satisfaction):
        std_bits, mask_bits = [], []
        if address and len(mapping) < 5:
            ph = slot(address)
            std_bits.append(f"주소는 {address}")
            mask_bits.append(f"주소는 {ph}")
        if satisfaction and len(mapping) < 5:
            ph = slot(satisfaction)
            std_bits.append(f"평균 만족도는 {satisfaction}")
            mask_bits.append(f"평균 만족도는 {ph}")
        if std_bits:
            std_sentences.append(", ".join(std_bits) + "입니다.")
            mask_sentences.append(", ".join(mask_bits) + "입니다.")

    if len(mapping) < 5 and activity:
        ph = slot(activity)
        std_sentences.append(f"주로 {activity} 같은 걸 하는 곳으로 기록돼 있습니다.")
        mask_sentences.append(f"주로 {ph} 같은 걸 하는 곳으로 기록돼 있습니다.")
    elif len(mapping) < 5 and food:
        ph = slot(food)
        std_sentences.append(f"관련 기록으로는 {food} 같은 게 남아 있습니다.")
        mask_sentences.append(f"관련 기록으로는 {ph} 같은 게 남아 있습니다.")

    return " ".join(std_sentences), " ".join(mask_sentences), mapping


def _tone_dialect(text: str) -> str:
    """이미 만든 표준어 문장의 어미만 기계적으로 사투리 어미로 바꾼다 — 이 코드베이스에
    이미 쓰이는(스몰토크/음식 응답에 실제로 있는) 어미만 재사용하고 새로 지어내지 않는다.
    LoRA 생성이 실패해도(오늘 실측상 대부분 그럼) 폴백이 표준어로 안 들리게 하기 위함.
    사실 값은 절대 안 건드림 — 문장 끝 술어만 문자열 치환."""
    text = text.replace("있습니다", "있습니더")
    text = text.replace("합니다", "합니더")
    text = text.replace("입니다", "이래요")
    return text


def assemble_dialect_with_names(names: list[str], region: str | None) -> str:
    """장소명은 코드가 직접 넣고, 말미만 사투리 틀로 고정 (환각 방지)."""
    where = region or "강원"
    joined = ", ".join(names)
    first = names[0]
    particle = _object_particle(first)
    return _tone_dialect(
        f"{where}에서 가볼 만한 곳으로는 {joined}이 있습니다. "
        f"특히 {first}{particle} 추천합니다."
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
        f"특히 {placeholders[0]}{particle} 추천합니다."
    )
    return template, dict(zip(placeholders, names))


def _label_phrases(masked_template: str) -> list[str]:
    """각 [장소N] 플레이스홀더 바로 앞의 "라벨" 어절(예: "평균 만족도는")을 뽑는다.
    이 라벨이 깨지면 뒤따르는 사실 값이 뭘 가리키는지 못 알아보게 되므로, 사투리
    변환 후에도 그대로 남아있어야 하는 최소 요구사항으로 쓴다.

    첫 플레이스홀더 앞의 구절(예: "가볼 만한 곳으로는")은 문장 오프닝 절이라
    어미처럼 사투리 스타일로 유동적으로 바뀌는 걸 실제로 관찰했기 때문에 제외한다
    — "곳으로는"→"곳으로,"처럼 바뀌는 건 정상 동작이지 오염이 아니다."""
    parts = re.split(r"(\[장소\d+\])", masked_template)
    labels = []
    for i in range(3, len(parts), 2):
        preceding = parts[i - 1]
        words = preceding.strip().split()
        last_word = words[-1].strip(",.") if words else ""
        if len(last_word) >= 2:
            labels.append(last_word)
    return labels


def _dialectize_body_attempt(
    masked_template: str, mapping: dict[str, str], temperature: float
) -> tuple[str, bool]:
    prompt = (
        "아래 문장의 어미(문장 끝맺음)만 강원도 사투리 말투로 바꿔라.\n"
        "대괄호로 감싼 [장소N] 표시는 절대 지우거나 바꾸지 말고 그대로 유지할 것.\n"
        "새로운 문장·정보·장소를 추가하지 말고 원문 문장 구조와 순서를 유지할 것.\n"
        "변환 문장만 출력.\n\n"
        f"원문: {masked_template}\n\n"
        "사투리:"
    )
    try:
        out = generate(prompt, max_new_tokens=160, temperature=temperature, use_lora=True)
        out = out.strip().splitlines()[0].strip()
    except Exception as e:
        print(f"[경고] 본문 사투리 변환 실패(temperature={temperature}): {e!r}")
        return "", False

    expected = set(mapping.keys())
    found = set(re.findall(r"\[장소\d+\]", out))
    bad = any(x in out for x in ("http", "영어", "〈", "<P", "사투리", "원문", "말투"))
    labels_ok = all(label in out for label in _label_phrases(masked_template))
    # "가보면"→"가믄", "좋습니다"→"좋제" 같은 활용 변화는 정상이지만, "먼저"/"추천"은
    # 실사용 중 "메머가"/"메번"/"택름합니다" 등으로 반복해서 깨지는 게 확인된 단어라
    # 별도로 보존을 요구한다 (플레이스홀더 뒤 구절이라 _label_phrases가 못 잡음).
    _PROTECTED_CONNECTORS = ("먼저", "추천")
    connector_ok = all(
        (word not in masked_template) or (word in out) for word in _PROTECTED_CONNECTORS
    )
    if bad or not out or len(out) > 300 or found != expected or not labels_ok or not connector_ok:
        return "", False

    for placeholder, name in mapping.items():
        out = out.replace(placeholder, name)
    return out, True


def dialectize_body(masked_template: str, mapping: dict[str, str]) -> tuple[str, bool]:
    """본문 틀의 어미만 LoRA로 사투리화. 실패하면 ("", False) — 호출부가 표준 틀로 폴백.

    temperature=0(결정론적)으로 먼저 시도한다 — 같은 입력이면 항상 같은 결과라
    빠르고 재현 가능하다. 검증(플레이스홀더/라벨/"먼저" 보존)에 걸리면, 결정론적
    시도가 매번 똑같이 깨지는 것뿐일 수 있으므로 무작위성을 조금씩 높여 재시도한다.
    검증 기준은 시도마다 동일하게 적용되므로 재시도가 안전성을 낮추지 않는다 —
    통과 못 하면 여러 번 시도해도 결국 안전한 표준어 폴백으로 끝난다."""
    for temperature in (0, 0.4, 0.7):
        out, ok = _dialectize_body_attempt(masked_template, mapping, temperature)
        if ok:
            return out, True
    return "", False


def convert_to_dialect(
    standard_short: str,
    must_keep_names: list[str],
    use_lora: bool,
    region: str | None = None,
    *,
    fallback_body: str | None = None,
    masked: tuple[str, dict[str, str]] | None = None,
    include_opener: bool = True,
    convert_body: bool = True,
) -> tuple[str, bool]:
    """
    LoRA에 실제 사실 값을 넣지 않는다.
    - 본문: 사실 고정 삽입 + 사투리 말미 템플릿 (항상 성공)
    - (선택) LoRA로 사실 없는 짧은 인사만 앞에 붙임

    include_opener=False면 인사를 아예 안 붙인다 — 대화가 이미 진행 중인
    후속 턴에서 매번 "여행 오셨으면" 인사가 반복되는 걸 막기 위함.

    convert_body=False면 본문 자체는 LoRA에 안 태우고 fallback_body(표준어)를
    그대로 쓴다 — 장소 상세 답변처럼 마스킹 문장이 길고 사실 밀도가 높으면
    LoRA가 마스킹 안 된 일반 단어까지 오타로 깨뜨리는 사례가 있어서, 그런
    경우엔 본문 정확도를 우선한다 (오프너만 사투리로 붙임).

    fallback_body/masked를 넘기면 리스트 모드 전용 템플릿
    (assemble_dialect_with_names/build_masked_dialect_template) 대신 그걸 쓴다
    — 장소 상세 답변 등 다른 문장 모양을 그대로 사투리 변환 파이프라인에 태우기 위함.
    """
    names = [n for n in must_keep_names[:5] if n]
    if not names:
        return standard_short, False

    body = fallback_body if fallback_body is not None else assemble_dialect_with_names(names, region)

    if not use_lora:
        return body, True

    if convert_body:
        if masked is not None:
            masked_template, mapping = masked
        else:
            masked_template, mapping = build_masked_dialect_template(names, region)
        dialect_body, body_ok = dialectize_body(masked_template, mapping)
        if body_ok:
            body = dialect_body

    if not include_opener:
        return body, True

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
            for x in ("신발", "헐겁", "http", "영어", "〈", "<P", "사투리", "원문", "말투")
        )
        # 오프너는 원문에 사실이 전혀 없는 문장이라, 숫자나 다른 지역명이 새로
        # 등장하면 사실을 지어냈다는 신호로 보고 버린다.
        has_digit = any(ch.isdigit() for ch in opener)
        leaked_region = any(
            r in opener for r in (*_REGION_NAMES, *_OTHER_REGIONS) if r != (region or "")
        )
        # 너무 길거나 빈 응답이면 버림
        if bad or has_digit or leaked_region or not opener or len(opener) > 80:
            return body, True
        return f"{opener}\n{body}", True
    except Exception as e:
        print(f"[경고] LoRA 사투리 변환 실패, RAG 템플릿으로 폴백: {e!r}")
        return body, True




_REGION_NAMES = ("속초", "강릉", "춘천", "양양", "평창", "원주", "동해", "삼척")
# 강원도 밖 시/도 — 전체 17개 광역 행정구역 중 강원도를 뺀 나머지 고정 목록이라
# 사투리 표현처럼 무한히 늘어나지 않는다 (whack-a-mole 아님).
_OTHER_REGIONS = (
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
)


def _reply_only(question: str, region: str | None, reply: str, context: str = "") -> dict:
    """스몰토크/비지원 지역/신뢰도 게이트처럼 '장소 정보 없이 문장 하나만' 돌려주는
    응답 3곳의 공통 모양. RAG 사실 없이 고정/생성 문구만 있는 경우 전용."""
    return {
        "question": question,
        "region": region,
        "places": [],
        "context": context,
        "standard": reply,
        "short": reply,
        "dialect": reply,
        "dialect_ok": True,
        "answer": reply,
    }


def answer_question(
    question: str,
    *,
    k: int = 5,
    region: str | None = None,
    places_only: bool = True,
    with_llm: bool = True,
    no_lora: bool = False,
    history: list[dict] | None = None,
) -> dict:
    """질문 → RAG 검색 → (선택) 사투리 답변. API/파이프라인용.

    history: 같은 대화 세션의 이전 턴들, [{"question": "..."}, ...] (오래된 것 -> 최신 순).
    "그중에", "거기" 같은 후속 질문을 지원하기 위해서만 쓴다 — RAG 검색 질의 보강과
    지역 유지(carry-over)에만 관여하고, LLM 프롬프트에 사실을 직접 주입하지는 않는다
    (사실은 항상 이번 턴의 RAG 검색 결과에서만 가져온다 — 환각 방지 원칙 유지).
    """
    history = history or []
    intent = classify_intent(question)

    if intent == "smalltalk":
        return _reply_only(question, region, smalltalk_reply(question))

    # 알려진 오프토픽 카테고리(날씨/나이/코딩/심심)는 RAG 신뢰도 게이트보다 먼저 확정
    # 처리한다. region=None(대화 첫 턴 등)이면 지역 필터 없이 검색이 더 넓은 코퍼스를
    # 도는데, 그러면 오프토픽 질문의 top-1 거리도 같이 낮아져 _NO_MATCH_DISTANCE
    # 임계값을 통과해버리는 게 실측으로 확인됨 (예: "지금 날씨 어떄요?" region=None →
    # distance 0.645, region="속초" → 0.71). 새 키워드를 늘리는 게 아니라 이미 있는
    # 카테고리를 검색 전에 앞당겨 적용하는 것이라 두더지잡기가 아니다.
    for keywords, templates in _NO_MATCH_CATEGORIES:
        if any(k in question for k in keywords):
            reply = templates[hash(question) % len(templates)]
            return _reply_only(question, region, reply)

    # 질문에 지역이 명시돼 있으면 항상 최우선 — 프론트가 이전에 선택된 지역(칩 등)을
    # region 파라미터로 계속 넘기고 있어도, "양양 해변 추천"처럼 질문 안에서 다른
    # 지역을 콕 집으면 그게 이겨야 한다 (안 그러면 넘어온 region이 "강릉"으로 굳어
    # 있을 때 "양양" 질문에도 계속 강릉으로 답해버리는 버그가 남는다).
    explicit_region = next((r for r in _REGION_NAMES if r in question), None)
    other_region = next((r for r in _OTHER_REGIONS if r in question), None)
    if explicit_region:
        region = explicit_region
    elif other_region:
        # 강원도 밖 지역을 콕 집어 물으면, 넘어온/이전 턴 지역을 이어받지 말고
        # 지원 범위 밖이라고 답한다 — 프론트가 넘긴 region 파라미터가 이미 다른
        # 지역으로 굳어 있어도(예: "강릉") 질문의 명시적 언급이 항상 이긴다.
        reply = (
            f"{other_region}는 지가 아직 잘 모르는 동네라예. 강원도 여행 얘기로 물어봐 주시믄 좋겠습니더."
        )
        return _reply_only(question, region, reply)
    elif not region:
        # 후속 질문이 지역을 다시 언급하지 않으면 직전 턴의 지역을 유지한다.
        for turn in reversed(history):
            prev_q = (turn.get("question") or "") if isinstance(turn, dict) else ""
            found = next((r for r in _REGION_NAMES if r in prev_q), None)
            if found:
                region = found
                break

    if places_only is True or any(
        x in question for x in ("가볼", "명소", "관광", "해변", "시장", "장소")
    ):
        places_only = True

    # 검색 질의 보강: 직전 질문을 덧붙여 지칭 표현("그중", "거기" 등)의 문맥을 살린다.
    search_query = question
    if history:
        prev_q = (history[-1].get("question") or "") if isinstance(history[-1], dict) else ""
        if prev_q:
            search_query = f"{prev_q} {question}"

    retriever = TravelRetriever()
    docs = retriever.search(
        search_query,
        k=k,
        doc_type="place" if places_only else None,
        region=region,
    )
    context = format_context(docs)
    cards = extract_place_cards(docs)

    if places_only and (not docs or (docs[0].distance is not None and docs[0].distance > _NO_MATCH_DISTANCE)):
        return _reply_only(question, region, pick_no_match_reply(question), context=context)

    if intent == "food":
        food_matched = match_question_place(question, cards)
        food_cards = [food_matched] if food_matched is not None else cards
        standard, food_entries = build_food_standard_answer(question, food_cards)
        if food_entries:
            dialect = assemble_food_dialect(food_entries, region)
            dialect_ok = True
        else:
            dialect = standard
            dialect_ok = False
        short = standard
        if food_matched is not None:
            cards = [food_matched] + [c for c in cards if c["name"] != food_matched["name"]]
    else:
        matched = match_question_place(question, cards)
        if matched is not None:
            standard = build_place_detail_standard(question, matched)
            short, masked_template, mapping = build_place_detail_body(matched, region=region)

            dialect = standard
            dialect_ok = False
            if with_llm:
                dialect, dialect_ok = convert_to_dialect(
                    short,
                    [matched["name"]],
                    use_lora=not no_lora,
                    region=region,
                    fallback_body=_tone_dialect(short),
                    masked=(masked_template, mapping),
                    include_opener=not history,
                )
            cards = [matched] + [c for c in cards if c["name"] != matched["name"]]
        else:
            standard = build_standard_answer(question, cards)
            short = build_short_for_convert(cards, region=region)
            names = [c["name"] for c in cards]

            dialect = standard
            dialect_ok = False
            if with_llm:
                dialect, dialect_ok = convert_to_dialect(
                    short, names, use_lora=not no_lora, region=region,
                    include_opener=not history,
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
