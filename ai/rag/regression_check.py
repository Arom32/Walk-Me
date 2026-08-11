"""오늘 발견한 버그들의 회귀 방지 체크. pytest 아님 — 반복문 + assert 모음.

사용 (walkme-llm env):
  cd ai
  python -m rag.regression_check          # 전체 (LLM 케이스 포함, 느림)
  python -m rag.regression_check --quick   # RAG만, LLM 로딩 없이 빠르게

--quick 없이 돌리는 케이스는 dialectize_body를 실제로 태워야만 검증되는 것들이다
(오프너 생략, "만족도"→"웂족도" 급 오염 방지) — with_llm=False로만 도는 회귀 스크립트는
이 버그들을 하나도 못 잡는다는 게 리뷰에서 지적된 맹점이라 반드시 포함한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

AI_DIR = Path(__file__).resolve().parents[1]
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from rag.ask import (
    _NO_MATCH_CATEGORIES,
    _NO_MATCH_RESPONSES,
    _SMALLTALK_RESPONSES,
    answer_question,
    build_place_detail_body,
    dialectize_body,
)

_ALL_NO_MATCH_REPLIES = set(_NO_MATCH_RESPONSES) | {
    t for _, templates in _NO_MATCH_CATEGORIES for t in templates
}

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def run_shape_cases() -> None:
    cases = [
        ("리스트 모드", "속초 가볼 만한 곳", "속초", [], "실제 여행 기록에 많이 나온 장소입니다"),
        ("장소 상세 (공백 매칭, 외옹치해변)", "외옹치 해변에는 뭐가 있어요?", "속초", [], "실제 여행 기록에 남아 있는 외옹치해변 정보"),
        ("장소 상세 (정확이름 검색, 속초항)", "속초항은 뭐가 유명해요?", "속초", [], "실제 여행 기록에 남아 있는 속초항 정보"),
        ("장소 상세 (영금정)", "영금정 가는 길 알려줘", "속초", [], "실제 여행 기록에 남아 있는 영금정 정보"),
        ("food 리스트", "속초 맛집", "속초", [], "실제 방문 기록에 남아 있는 음식 정보"),
        ("food 상세 (시골할머니)", "시골할머니는 어때요 뭐가 제일 맛있어요?", "속초", [], "시골할머니:"),
    ]
    for name, q, region, hist, expect_substr in cases:
        r = answer_question(q, region=region, history=hist, with_llm=False)
        check(name, expect_substr in r["standard"], f"got: {r['standard'][:80]!r}")


def run_smalltalk_cases() -> None:
    for name, q in [("감사 인사", "감사해요"), ("정체성 질문", "누구세요"), ("일반 인사", "안녕")]:
        r = answer_question(q, region="속초", history=[], with_llm=False)
        check(name, r["dialect"] in _SMALLTALK_RESPONSES, f"got: {r['dialect']!r}")


def run_no_match_cases() -> None:
    for name, q in [("오프토픽 날씨", "오늘 날씨 어때요"), ("오프토픽 코드요청", "파이썬 코드 짜줘")]:
        r = answer_question(q, region="속초", places_only=True, history=[], with_llm=False)
        check(name, r["dialect"] in _ALL_NO_MATCH_REPLIES, f"got: {r['dialect']!r}")

    # region=None(대화 첫 턴 등)이면 지역 필터 없는 검색이라 오프토픽 질문도 top-1
    # distance가 임계값 밑으로 내려가 새어나가던 버그 — 카테고리 키워드를 RAG보다
    # 앞서 체크하도록 고쳐서 region 유무와 무관하게 걸려야 한다.
    for name, q in [
        ("오프토픽 날씨 (region=None)", "지금 날씨 어떄요?"),
        ("오프토픽 나이 (region=None)", "너 몇 살이야"),
    ]:
        r = answer_question(q, region=None, places_only=True, history=[], with_llm=False)
        check(name, r["dialect"] in _ALL_NO_MATCH_REPLIES, f"got: {r['dialect']!r}")


def run_unsupported_region_cases() -> None:
    for name, q in [("서울 맛집", "서울 맛집 알려주세요"), ("서울 관광", "서울 관광으로는 어디가 좋아요")]:
        r = answer_question(q, region=None, history=[{"question": "속초 가볼 만한 곳"}], with_llm=False)
        check(
            name,
            "서울" in r["dialect"] and "모르는 동네" in r["dialect"],
            f"got: {r['dialect']!r}",
        )


def run_region_override_cases() -> None:
    """프론트가 넘긴(스티키 칩 등) region 파라미터보다 질문에 명시된 지역이 항상
    이겨야 한다 — "강릉"이 넘어와도 "양양 해변 추천"이면 양양으로 답해야 함."""
    r = answer_question("양양 해변 추천", region="강릉", history=[], with_llm=False)
    check("스티키 region보다 질문의 명시 지역 우선", r["region"] == "양양", f"got region: {r['region']!r}")

    r2 = answer_question("서울 맛집", region="강릉", history=[], with_llm=False)
    check(
        "스티키 region이어도 미지원 지역 질문은 거절",
        "서울" in r2["dialect"] and "모르는 동네" in r2["dialect"],
        f"got: {r2['dialect']!r}",
    )

    r3 = answer_question("가볼 만한 곳 추천", region="강릉", history=[], with_llm=False)
    check("질문에 지역 언급 없으면 넘어온 region 유지", r3["region"] == "강릉", f"got region: {r3['region']!r}")


def run_opener_cases() -> None:
    """with_llm=True 필요 — 느림. 오프너 반복 버그 회귀 확인."""
    r_first = answer_question("속초 가볼 만한 곳", region="속초", history=[], with_llm=True)
    check(
        "첫 턴 오프너 있음",
        len(r_first["dialect"].splitlines()) >= 2,
        f"got: {r_first['dialect'][:60]!r}",
    )
    r_followup = answer_question(
        "속초등대는 뭐가 좋아요?",
        region="속초",
        history=[{"question": "속초 가볼 만한 곳"}],
        with_llm=True,
    )
    check(
        "후속 턴 오프너 없음",
        "여행 오셨" not in r_followup["dialect"] and "이르시" not in r_followup["dialect"],
        f"got: {r_followup['dialect'][:60]!r}",
    )


def run_label_preservation_regression() -> None:
    """with_llm=True 필요 — 느림. 실제 오염 버그("만족도"->"웂족도") 재현 케이스로
    dialectize_body의 라벨 보존 검증이 실제로 작동하는지 직접 확인한다."""
    card = {
        "name": "속초등대",
        "region": "속초",
        "visit_type": "자연관광지",
        "address": "강원 속초시 영금정로5길 8-28",
        "extra": ["평균 만족도: 3.75/5", "주요 활동: 단순 구경 / 산책 / 걷기"],
    }
    _, masked, mapping = build_place_detail_body(card, region="속초")
    out, ok = dialectize_body(masked, mapping)
    if ok:
        check("라벨 보존 검증 (성공시 라벨 온전)", "만족" in out, f"got: {out!r}")
    else:
        check("라벨 보존 검증 (실패시 안전 폴백)", out == "", f"got: {out!r}")


if __name__ == "__main__":
    quick = "--quick" in sys.argv

    run_shape_cases()
    run_smalltalk_cases()
    run_no_match_cases()
    run_unsupported_region_cases()
    run_region_override_cases()

    if not quick:
        print("--- 아래는 --with-llm 필요한 케이스, LoRA 모델 로딩으로 느릴 수 있음 ---")
        run_opener_cases()
        run_label_preservation_regression()
    else:
        print("--quick: LLM 필요 케이스(오프너/라벨보존) 스킵함")

    print()
    if FAILURES:
        print(f"FAIL: {len(FAILURES)}건 실패 — {FAILURES}")
        sys.exit(1)
    print("모든 케이스 통과")
