from peft import LoraConfig

lora_config = LoraConfig(

    r=16,  # r=8 or 16이 적정 64는 vram너무 잡아먹음

    lora_alpha=32, #ΔW × alpha/r ->lora효과 2배적용 (16*2)

    lora_dropout=0.05, #과적합 방지

    bias="none",  #메모리 최적화.. 덜 쓰려고

    task_type="CAUSAL_LM",

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]  #gate,up,down_proj은 적용 x -> vram과다 사용방지
)