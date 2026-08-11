"""ai/rag/ask.py의 _NO_MATCH_DISTANCE(0.65) 임계값을 라벨링된 held-out
질문 세트로 검증한다. pytest 아님 — 그냥 스크립트.

사용 (walkme-llm env):
  cd ai
  python -m rag.eval_no_match_threshold

원래 0.65는 세션 중 수동으로 관찰한 18개 질문으로 잡은 값이었다. 여기서는 그중 일부 +
이번에 새로 추가한 질문들(다른 지역/다른 표현)을 합쳐서 좀 더 넓게 검증한다 — 그래도
완전한 자동 라벨링 데이터셋은 아니고, 여전히 사람이 만든 소규모 세트라는 한계는 있다.
"""

from __future__ import annotations

import sys
from pathlib import Path

AI_DIR = Path(__file__).resolve().parents[1]
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from rag.ask import _NO_MATCH_DISTANCE
from rag.retrieve import TravelRetriever

# (질문, region) — 실제 관광/음식 질문. 라벨: on_topic
ON_TOPIC = [
    ("속초 가볼 만한 곳", "속초"),
    ("외옹치 해변에는 뭐가 있어요?", "속초"),
    ("속초등대는 뭐가 유명해요", "속초"),
    ("영금정 가는 길 알려줘", "속초"),
    ("속초해수욕장 주차 가능해요?", "속초"),
    ("속초 맛집", "속초"),
    ("청초호유원지는 어떤 곳이에요", "속초"),
    ("강릉 커피거리는 뭐가 있나요", "강릉"),
    ("평창 스키장 어디가 좋아요", "평창"),
    ("삼척 가볼 만한 곳", "삼척"),
    ("동해 맛집 추천", "동해"),
    ("원주 명소 알려줘", "원주"),
    ("양양 서핑하기 좋은 곳", "양양"),
    ("춘천 닭갈비 맛집", "춘천"),
]

# (질문, region) — 오프토픽/무관 질문. 라벨: off_topic
OFF_TOPIC = [
    ("오늘 날씨 어때요", "속초"),
    ("파이썬 코드 짜줘", "속초"),
    ("오늘 기분이 어때", "속초"),
    ("너 몇 살이야", "속초"),
    ("심심한데 아무말이나 해줘", "속초"),
    ("주식 투자 어떻게 해", "속초"),
    ("서울 맛집 알려줘", None),
    ("오늘 저녁 뭐 먹을지 추천해줘 집에서", "속초"),
    ("영어 단어 하나만 알려줘", "속초"),
    ("게임 추천해줘", "속초"),
]


def top1_distance(retriever: TravelRetriever, question: str, region: str | None) -> float | None:
    docs = retriever.search(question, k=3, doc_type="place", region=region)
    if not docs or docs[0].distance is None:
        return None
    return docs[0].distance


def main() -> None:
    retriever = TravelRetriever()

    on_dists = []
    print(f"=== ON-TOPIC (임계값 {_NO_MATCH_DISTANCE} 이하로 나와야 정상) ===")
    for q, region in ON_TOPIC:
        d = top1_distance(retriever, q, region)
        on_dists.append(d)
        mark = "OK" if d is not None and d <= _NO_MATCH_DISTANCE else "MISCLASSIFIED(오탐: no-match로 잘못 걸림)"
        print(f"  [{mark}] {d}  {q!r}")

    off_dists = []
    print(f"\n=== OFF-TOPIC (임계값 {_NO_MATCH_DISTANCE} 초과로 나와야 정상) ===")
    for q, region in OFF_TOPIC:
        d = top1_distance(retriever, q, region)
        off_dists.append(d)
        mark = "OK" if d is not None and d > _NO_MATCH_DISTANCE else "MISCLASSIFIED(누락: 리스트로 새어나감)"
        print(f"  [{mark}] {d}  {q!r}")

    on_valid = [d for d in on_dists if d is not None]
    off_valid = [d for d in off_dists if d is not None]
    on_errors = sum(1 for d in on_valid if d > _NO_MATCH_DISTANCE)
    off_errors = sum(1 for d in off_valid if d <= _NO_MATCH_DISTANCE)

    print("\n=== 요약 ===")
    print(f"on-topic  distance: min={min(on_valid):.4f} max={max(on_valid):.4f} (n={len(on_valid)}, 오탐 {on_errors}건)")
    print(f"off-topic distance: min={min(off_valid):.4f} max={max(off_valid):.4f} (n={len(off_valid)}, 누락 {off_errors}건)")

    if max(on_valid) < min(off_valid):
        mid = (max(on_valid) + min(off_valid)) / 2
        print(f"\n두 그룹이 겹치지 않음 — 여유 구간 [{max(on_valid):.4f}, {min(off_valid):.4f}], 중간값 {mid:.4f}")
        print(f"현재 임계값 {_NO_MATCH_DISTANCE}가 이 구간 안에 있는지: {'예' if max(on_valid) < _NO_MATCH_DISTANCE < min(off_valid) else '아니오 — 조정 필요'}")
    else:
        print(f"\n경고: 두 그룹이 겹침 (on-topic max={max(on_valid):.4f} >= off-topic min={min(off_valid):.4f}) — 단일 임계값으로 완벽히 못 가름, 오차 있음을 감안할 것")

    total_errors = on_errors + off_errors
    print(f"\n총 {len(ON_TOPIC) + len(OFF_TOPIC)}건 중 오분류 {total_errors}건")


if __name__ == "__main__":
    main()
