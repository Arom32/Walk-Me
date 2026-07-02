#lora학습
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="../../data/train.jsonl"
)