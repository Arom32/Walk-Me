import os
import json

INPUT_DIR = "../../data/train"
OUTPUT_FILE = "../../data/train.jsonl"

SYSTEM_PROMPT = "표준어를 강원도 사투리로 변환하는 AI이다."


def create_jsonl():

    files = sorted(
        [f for f in os.listdir(INPUT_DIR)
         if f.endswith(".json")]
    )

    count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as writer:

        for file in files:

            path = os.path.join(INPUT_DIR, file)

            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            try:

                standard = data["transcription"]["standard"].strip()

                dialect = data["script"]["value"].strip()

                if standard == "" or dialect == "":
                    continue

                sample = {

                    "messages":[

                        {
                            "role":"system",
                            "content":SYSTEM_PROMPT
                        },

                        {
                            "role":"user",
                            "content":standard
                        },

                        {
                            "role":"assistant",
                            "content":dialect
                        }

                    ]

                }

                writer.write(
                    json.dumps(sample, ensure_ascii=False)
                    + "\n"
                )

                count += 1

            except KeyError:
                continue

    print(f"{count}개 저장 완료")


if __name__ == "__main__":

    create_jsonl()