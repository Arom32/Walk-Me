"""AI Hub zip.part0 → backend/data/extracted/{TL,VL}/*.csv (영문 파일명, POI master 제외).

사용: cd ai && python -m rag.extract_travel_csv
"""

from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

AI_DIR = Path(__file__).resolve().parents[1]
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from rag import EXTRACTED_DIR, RAW_DIR

RENAMES = [
    (r"^tn_visit_area_info", "tn_visit_area_info.csv"),
    (r"^tn_traveller_master", "tn_traveller_master.csv"),
    (r"^tn_travel_", "tn_travel.csv"),
    (r"^tn_activity_his", "tn_activity_his.csv"),
    (r"^tn_activity_consume", "tn_activity_consume_his.csv"),
    (r"^tn_lodge_consume", "tn_lodge_consume_his.csv"),
    (r"^tn_companion_info", "tn_companion_info.csv"),
    (r"^tn_move_his", "tn_move_his.csv"),
    (r"^tn_mvmn_consume", "tn_mvmn_consume_his.csv"),
    (r"^tn_adv_consume", "tn_adv_consume_his.csv"),
    (r"^tn_tour_photo", "tn_tour_photo.csv"),
    (r"^tn_poi_master", "tn_poi_master.csv"),
    (r"^tc_sgg_", "tc_sgg.csv"),
    (r"^tc_codea_", "tc_codea.csv"),
    (r"^tc_codeb_", "tc_codeb.csv"),
]


def english_name(member: str) -> str | None:
    base = Path(member).name
    if not base or base.endswith("/"):
        return None
    for pat, out in RENAMES:
        if re.match(pat, base):
            return out
    return None


def extract(zip_path: Path, out_dir: Path, skip_poi: bool = True) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            en = english_name(info.filename)
            if not en:
                continue
            if skip_poi and en == "tn_poi_master.csv":
                print(f"  skip POI master ({info.file_size / 1024 / 1024:.0f}MB)")
                continue
            dest = out_dir / en
            with z.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"  {en}  {dest.stat().st_size / 1024:.0f}KB")


def resolve_zip(stem: str) -> Path:
    candidates = [
        RAW_DIR / f"{stem}.zip.part0",
        RAW_DIR / f"{stem}.zip",
        EXTRACTED_DIR / f"{stem}.zip",
        EXTRACTED_DIR / f"{stem}.zip.part0",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"{stem} zip not found under {RAW_DIR}")


def main() -> None:
    extract(resolve_zip("TL_csv"), EXTRACTED_DIR / "TL")
    extract(resolve_zip("VL_csv"), EXTRACTED_DIR / "VL")
    print("done")


if __name__ == "__main__":
    main()
