#데이터셋 처리-바꿔야 함
from datasets import load_dataset

def get_dataset():

    dataset = load_dataset(
        "json",
        data_files="../data/train.jsonl"
    )

    return dataset