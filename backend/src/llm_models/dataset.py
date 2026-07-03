import os
import json
import random

INPUT_DIR = "../../data/train/kw_1"
TRAIN_FILE = "../../data/kw_1_train.jsonl"
VAL_FILE = "../../data/kw_1_val.jsonl"

VAL_RATIO = 0.05
SEED = 42
MIN_LEN = 3  # 너무 짧은 문장 필터링 기준 (글자 수)

SYSTEM_PROMPTS = [
    "표준어를 강원도 사투리로 변환하는 AI이다.",
    "너는 표준어 문장을 강원도 방언으로 옮기는 번역기야.",
    "입력된 표준어 문장을 자연스러운 강원도 사투리로 변환해라.",
    "표준어 문장을 강원도 사투리 문장으로 바꾸는 역할을 수행한다.",
    "다음 표준어 문장을 강원도 사투리로 자연스럽게 바꿔줘.",
]


def load_samples():
    """INPUT_DIR의 JSON 파일들을 읽어 (standard, dialect) 쌍 리스트로 반환."""

    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".json"))

    samples = []
    seen = set()
    skipped = 0

    for file in files:
        path = os.path.join(INPUT_DIR, file)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        try:
            standard = data["transcription"]["standard"].strip()
            dialect = data["script"]["value"].strip()
        except (KeyError, TypeError) as e:
            print(f"  [스킵] {file} - 필드 오류: {e}")
            skipped += 1
            continue

        # 빈 문자열 필터링
        if not standard or not dialect:
            skipped += 1
            continue

        # 변환이 안 된(원문 그대로인) 샘플 필터링
        if standard == dialect:
            skipped += 1
            continue

        # 너무 짧은 문장 필터링
        if len(standard) < MIN_LEN or len(dialect) < MIN_LEN:
            skipped += 1
            continue

        # 중복 제거
        key = (standard, dialect)
        if key in seen:
            skipped += 1
            continue
        seen.add(key)

        samples.append({"standard": standard, "dialect": dialect})

    print(f"\n총 파일 수: {len(files)}")
    print(f"수집된 샘플: {len(samples)}개")
    print(f"스킵된 샘플: {skipped}개\n")

    return samples


def write_jsonl(path, subset):
    """(standard, dialect) 쌍 리스트를 messages 포맷 jsonl로 저장."""

    with open(path, "w", encoding="utf-8") as writer:
        for s in subset:
            sample = {
                "messages": [
                    {
                        "role": "system",
                        "content": random.choice(SYSTEM_PROMPTS),
                    },
                    {
                        "role": "user",
                        "content": s["standard"],
                    },
                    {
                        "role": "assistant",
                        "content": s["dialect"],
                    },
                ]
            }
            writer.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"{path} 저장 완료 ({len(subset)}개)")


def create_jsonl():
    samples = load_samples()

    if not samples:
        print("저장할 샘플이 없습니다. INPUT_DIR 및 필드 매핑을 확인하세요.")
        return

    random.seed(SEED)
    random.shuffle(samples)

    split_idx = int(len(samples) * (1 - VAL_RATIO))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    write_jsonl(TRAIN_FILE, train_samples)
    write_jsonl(VAL_FILE, val_samples)


if __name__ == "__main__":
    create_jsonl()