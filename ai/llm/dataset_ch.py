import os
import json
import random

INPUT_FILE = "./data/kw_1_train.jsonl"
TRAIN_FILE = "./data/kw_1_train_10000.jsonl"
VAL_FILE = "./data/kw_1_val_10000.jsonl"

VAL_RATIO = 0.05
SEED = 42
MIN_LEN = 3          # 너무 짧은 문장 필터링 기준
MAX_SAMPLES = 10000  # 사용할 최대 샘플 수

SYSTEM_PROMPTS = [
    "표준어를 강원도 사투리로 변환하는 AI이다.",
    "너는 항상 강원도 사투리로 자연스럽게 대답하는 AI다.",
    "입력된 표준어 문장을 자연스러운 강원도 사투리로 변환해라.",
]

def load_samples():
    """jsonl 파일에서 최대 MAX_SAMPLES개를 읽어온다."""
    print(os.path.abspath(INPUT_FILE))
    samples = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)

            standard = sample["messages"][1]["content"]
            dialect = sample["messages"][2]["content"]

            samples.append({
                "standard": standard,
                "dialect": dialect
            })

            if len(samples) >= MAX_SAMPLES:
                break

    print(f"수집된 샘플 : {len(samples)}개")
    print(samples[:3])
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
        print("저장할 샘플이 없습니다.")
        return

    # 샘플 한 번 더 섞어서 train/valid 분할
    random.seed(SEED)
    random.shuffle(samples)

    split_idx = int(len(samples) * (1 - VAL_RATIO))

    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"Train : {len(train_samples)}")
    print(f"Valid : {len(val_samples)}")

    write_jsonl(TRAIN_FILE, train_samples)
    write_jsonl(VAL_FILE, val_samples)


if __name__ == "__main__":
    create_jsonl()