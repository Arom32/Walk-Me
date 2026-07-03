import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from model_config import lora_config, bnb_config

# ------------------------------------------------------------------
# 경로 / 하이퍼파라미터 설정
# ------------------------------------------------------------------

MODEL_NAME = "google/gemma-4-12b-it"  # 실제 사용하는 모델 이름으로 수정

TRAIN_FILE = "../../data/kw_1_train.jsonl"
VAL_FILE = "../../data/kw_1_val.jsonl"

OUTPUT_DIR = "../../checkpoints/kw_lora"

MAX_SEQ_LENGTH = 512  # 사투리 변환 태스크는 문장 단위라 길지 않음. 데이터 분포 보고 조정

NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8  # 실질 배치 크기 = PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STEPS = 200
EVAL_STEPS = 200


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # k-bit 학습을 위한 준비 (그래디언트 체크포인팅 + 레이어 캐스팅)
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # 학습 중에는 캐시 비활성화 (체크포인팅과 충돌 방지)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def load_datasets():
    data_files = {"train": TRAIN_FILE, "validation": VAL_FILE}
    dataset = load_dataset("json", data_files=data_files)
    return dataset["train"], dataset["validation"]


def formatting_func(example, tokenizer):
    """messages(system/user/assistant) -> 모델 chat template 문자열로 변환."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return text


def main():
    model, tokenizer = load_model_and_tokenizer()
    train_dataset, val_dataset = load_datasets()

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        bf16=True,
        optim="paged_adamw_8bit",  # 4bit 양자화 모델과 궁합 좋음, VRAM 절약
        report_to="none",  # wandb 등 쓰면 "wandb"로 변경
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,  # 문장 단위 짧은 샘플이라 packing은 끔 (필요시 True로)
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=lambda ex: formatting_func(ex, tokenizer),
        processing_class=tokenizer,
    )

    trainer.train()

    # 최종 어댑터(LoRA) 저장
    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"학습 완료. 최종 어댑터 저장 위치: {final_dir}")


if __name__ == "__main__":
    main()