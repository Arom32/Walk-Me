"""
AI Hub 여행 CSV(TL/VL) → 강원 RAG 문서(jsonl) 생성.

출력:
  ai/rag/data/processed/gangwon_place_docs.jsonl   # 장소 단위 (메인)
  ai/rag/data/processed/gangwon_trip_docs.jsonl    # 여행 코스 단위 (보조)
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

AI_DIR = Path(__file__).resolve().parents[1]
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from rag import EXTRACTED_DIR as EXTRACTED
from rag import PROCESSED_DIR as OUT_DIR

SPLITS = ("TL", "VL")

# 방문지 유형 중 RAG에 덜 유용한 것
SKIP_VISIT_TYPES = {"22", "24"}  # 친구/친지집, 숙소
SKIP_NAME_PATTERNS = re.compile(
    r"(자택|우리집|친구집|친지|집$|휴게소|고속도로|톨게이트|주유소)"
)


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_code_map(path: Path) -> dict[tuple[str, str], str]:
    rows = load_csv(path)
    return {(r["cd_a"], r["cd_b"]): r["cd_nm"] for r in rows}


def is_gangwon(row: dict) -> bool:
    blob = " ".join(
        [
            row.get("ROAD_NM_ADDR") or "",
            row.get("LOTNO_ADDR") or "",
            row.get("TRAVEL_STATUS_DESTINATION") or "",
        ]
    )
    return "강원" in blob


def code_name(codes: dict, group: str, value: str | None, default: str = "") -> str:
    if not value:
        return default
    return codes.get((group, str(value).strip()), default or str(value))


def place_key(row: dict) -> str:
    poi = (row.get("POI_NM") or "").strip()
    name = (row.get("VISIT_AREA_NM") or "").strip()
    return poi or name


def normalize_place_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip()


def should_skip_visit(row: dict) -> bool:
    vtype = (row.get("VISIT_AREA_TYPE_CD") or "").strip()
    if vtype in SKIP_VISIT_TYPES:
        return True
    name = place_key(row)
    if not name:
        return True
    if SKIP_NAME_PATTERNS.search(name):
        return True
    return False


def safe_float(v: str | None) -> float | None:
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def avg(nums: list[float]) -> float | None:
    return round(sum(nums) / len(nums), 2) if nums else None


def merge_visits(codes: dict) -> tuple[list[dict], dict[tuple[str, str], list[dict]], dict[str, dict]]:
    """TL+VL 방문지/활동/여행객 로드 후 강원 방문만 반환."""
    visits: list[dict] = []
    # VISIT_AREA_ID alone is NOT unique across travels
    seen_keys: set[tuple[str, str]] = set()
    activities_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    travellers: dict[str, dict] = {}

    for split in SPLITS:
        base = EXTRACTED / split
        visit_rows = load_csv(base / "tn_visit_area_info.csv")
        act_rows = load_csv(base / "tn_activity_his.csv")
        trav_rows = load_csv(base / "tn_traveller_master.csv")
        travel_rows = load_csv(base / "tn_travel.csv")

        travel_by_id = {r["TRAVEL_ID"]: r for r in travel_rows}
        for r in trav_rows:
            travellers.setdefault(r["TRAVELER_ID"], r)

        for r in act_rows:
            key = (r["TRAVEL_ID"], r["VISIT_AREA_ID"])
            activities_by_key[key].append(r)

        for r in visit_rows:
            key = (r["TRAVEL_ID"], r["VISIT_AREA_ID"])
            if key in seen_keys:
                continue
            if not is_gangwon(r):
                continue
            if should_skip_visit(r):
                continue
            r = dict(r)
            r["_split"] = split
            r["_travel"] = travel_by_id.get(r["TRAVEL_ID"], {})
            r["_key"] = key
            visits.append(r)
            seen_keys.add(key)

    return visits, activities_by_key, travellers


def activity_summaries(
    act_rows: list[dict], codes: dict
) -> tuple[list[str], list[str]]:
    types: list[str] = []
    details: list[str] = []
    for a in act_rows:
        tname = code_name(codes, "ACT", a.get("ACTIVITY_TYPE_CD"))
        if tname:
            types.append(tname)
        dtl = (a.get("ACTIVITY_DTL") or "").strip()
        if dtl:
            for part in re.split(r"[;|/]", dtl):
                part = part.strip()
                if part:
                    details.append(part)
    return types, details


def build_place_docs(
    visits: list[dict],
    activities_by_key: dict[tuple[str, str], list[dict]],
    codes: dict,
) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for v in visits:
        key = normalize_place_name(place_key(v))
        buckets[key].append(v)

    docs: list[dict] = []
    for name, rows in buckets.items():
        addrs = [
            (r.get("ROAD_NM_ADDR") or r.get("LOTNO_ADDR") or "").strip()
            for r in rows
        ]
        addrs = [a for a in addrs if a]
        addr = Counter(addrs).most_common(1)[0][0] if addrs else ""

        types = [
            code_name(codes, "VIS", r.get("VISIT_AREA_TYPE_CD"))
            for r in rows
            if r.get("VISIT_AREA_TYPE_CD")
        ]
        vtype = Counter(t for t in types if t).most_common(1)
        vtype = vtype[0][0] if vtype else ""

        dwell = [safe_float(r.get("RESIDENCE_TIME_MIN")) for r in rows]
        dwell = [d for d in dwell if d is not None]
        sat = [safe_float(r.get("DGSTFN")) for r in rows]
        sat = [s for s in sat if s is not None]
        revisit = [safe_float(r.get("REVISIT_INTENTION")) for r in rows]
        revisit = [x for x in revisit if x is not None]
        recommend = [safe_float(r.get("RCMDTN_INTENTION")) for r in rows]
        recommend = [x for x in recommend if x is not None]

        act_types: list[str] = []
        act_details: list[str] = []
        for r in rows:
            t, d = activity_summaries(activities_by_key.get(r["_key"], []), codes)
            act_types.extend(t)
            act_details.extend(d)

        top_acts = [x for x, _ in Counter(act_types).most_common(5)]
        top_foods = [x for x, _ in Counter(act_details).most_common(8)]

        # 시군 추정
        region = ""
        for token in ("강릉", "속초", "춘천", "원주", "평창", "강원", "홍천", "정선", "양양", "삼척", "동해", "태백", "인제", "고성", "영월", "철원", "화천", "양구", "횡성"):
            if token in addr or token in name:
                region = token if token != "강원" else region
                if token != "강원":
                    break
        if not region and "강원" in addr:
            region = "강원"

        lines = [
            f"장소: {name}",
            f"지역: {region or '강원'}",
        ]
        if vtype:
            lines.append(f"유형: {vtype}")
        if addr:
            lines.append(f"주소: {addr}")
        lines.append(f"방문 기록 수: {len(rows)}건")
        if dwell:
            lines.append(f"평균 체류: {avg(dwell)}분")
        if sat:
            lines.append(f"평균 만족도: {avg(sat)}/5")
        if revisit:
            lines.append(f"재방문 의향: {avg(revisit)}/5")
        if recommend:
            lines.append(f"추천 의향: {avg(recommend)}/5")
        if top_acts:
            lines.append(f"주요 활동: {', '.join(top_acts)}")
        if top_foods:
            lines.append(f"관련 음식/구매: {', '.join(top_foods)}")

        text = "\n".join(lines)
        docs.append(
            {
                "id": f"place::{name}",
                "doc_type": "place",
                "place_name": name,
                "region": region or "강원",
                "visit_type": vtype,
                "address": addr,
                "visit_count": len(rows),
                "avg_dwell_min": avg(dwell),
                "avg_satisfaction": avg(sat),
                "text": text,
            }
        )

    docs.sort(key=lambda d: (-d["visit_count"], d["place_name"]))
    return docs


def build_trip_docs(
    visits: list[dict],
    activities_by_key: dict[tuple[str, str], list[dict]],
    travellers: dict[str, dict],
    codes: dict,
) -> list[dict]:
    by_travel: dict[str, list[dict]] = defaultdict(list)
    for v in visits:
        by_travel[v["TRAVEL_ID"]].append(v)

    docs: list[dict] = []
    for travel_id, rows in by_travel.items():
        rows = sorted(rows, key=lambda r: int(r.get("VISIT_ORDER") or 0))
        if len(rows) < 2:
            continue

        travel_meta = rows[0].get("_travel") or {}
        traveler_id = travel_meta.get("TRAVELER_ID") or ""
        # travel_id often like b_b000323 → traveler b000323
        if not traveler_id and "_" in travel_id:
            traveler_id = travel_id.split("_", 1)[-1]
        trav = travellers.get(traveler_id, {})

        stops: list[str] = []
        for r in rows[:12]:
            nm = normalize_place_name(place_key(r))
            vtype = code_name(codes, "VIS", r.get("VISIT_AREA_TYPE_CD"))
            dwell = r.get("RESIDENCE_TIME_MIN") or "?"
            stops.append(f"{nm}({vtype}, {dwell}분)" if vtype else f"{nm}({dwell}분)")

        gender = trav.get("GENDER") or ""
        age = trav.get("AGE_GRP") or ""
        accompany = trav.get("TRAVEL_STATUS_ACCOMPANY") or ""
        residence = trav.get("TRAVEL_STATUS_RESIDENCE") or ""
        mvmn = travel_meta.get("MVMN_NM") or ""

        lines = [
            f"강원 여행 코스 예시 (여행ID: {travel_id})",
        ]
        profile_bits = [x for x in [f"{age}대" if age else "", gender, accompany, f"거주:{residence}" if residence else ""] if x]
        if profile_bits:
            lines.append(f"여행자: {', '.join(profile_bits)}")
        if mvmn:
            lines.append(f"이동수단: {mvmn}")
        lines.append(f"방문 순서: {' → '.join(stops)}")

        # 활동/음식 요약
        details: list[str] = []
        for r in rows:
            _, d = activity_summaries(activities_by_key.get(r["_key"], []), codes)
            details.extend(d)
        top_details = [x for x, _ in Counter(details).most_common(6)]
        if top_details:
            lines.append(f"주요 음식/구매: {', '.join(top_details)}")

        docs.append(
            {
                "id": f"trip::{travel_id}",
                "doc_type": "trip",
                "travel_id": travel_id,
                "stop_count": len(rows),
                "text": "\n".join(lines),
            }
        )

    docs.sort(key=lambda d: (-d["stop_count"], d["travel_id"]))
    return docs


def write_jsonl(path: Path, docs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def main() -> None:
    for split in SPLITS:
        visit_path = EXTRACTED / split / "tn_visit_area_info.csv"
        if not visit_path.exists():
            raise SystemExit(f"누락: {visit_path} — 먼저 CSV를 해제하세요.")

    codes = load_code_map(EXTRACTED / "TL" / "tc_codeb.csv")
    print("코드 로드:", len(codes))

    visits, activities_by_key, travellers = merge_visits(codes)
    print(f"강원 방문(필터 후): {len(visits):,}건")

    place_docs = build_place_docs(visits, activities_by_key, codes)
    trip_docs = build_trip_docs(visits, activities_by_key, travellers, codes)

    place_path = OUT_DIR / "gangwon_place_docs.jsonl"
    trip_path = OUT_DIR / "gangwon_trip_docs.jsonl"
    write_jsonl(place_path, place_docs)
    write_jsonl(trip_path, trip_docs)

    print(f"장소 문서: {len(place_docs):,} → {place_path}")
    print(f"코스 문서: {len(trip_docs):,} → {trip_path}")
    print("\n--- 장소 샘플 (상위 3) ---")
    for doc in place_docs[:3]:
        print(doc["text"])
        print("---")
    if trip_docs:
        print("\n--- 코스 샘플 ---")
        print(trip_docs[0]["text"])


if __name__ == "__main__":
    main()
