import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "google/gemma-4-E2B-it"
LORA_PATH = "./lora_output/final"   # 자신의 checkpoint

SYSTEM_PROMPT = """
너는 강원도 토박이 AI 비서다.

규칙
- 모든 답변은 강원도 사투리로 한다.
- 사용자의 질문에 자연스럽게 대답한다.
- 표준어로 답하지 않는다.
- 답변 내용은 정확하게 유지한다.
"""
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH
)

model.eval()


print(model.peft_config)
for name, param in model.named_parameters():
    if "lora" in name.lower():
        print(name, param.shape, param.abs().mean().item())
        break
def ask(question):

    messages = [
        # {
        #     "role": "system",
        #     "content": SYSTEM_PROMPT
        # },
        {
            "role": "user",
            "content": question
        }
    ]
    print(type(question))
    print(repr(question))

    print(messages)
    print(type(messages))
    print(type(messages[0]["content"]))
    inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,   # 추가
    ).to(model.device)
    print("=== 실제 모델에 들어가는 프롬프트 ===")
    print(tokenizer.decode(inputs["input_ids"][0]))
    print("=" * 60)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,   # 수정
            max_new_tokens=256,
            temperature=0.2,    #수정 전에는 0.7이었음
            top_p=0.9,
            do_sample=True, #수정 전에는 True였음
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # 입력 부분 제외
    output = outputs[0][inputs["input_ids"].shape[-1]:]

    answer = tokenizer.decode(
        output,
        skip_special_tokens=True
    )

    return answer.strip()

while True:

    question = input("질문 : ")

    if question.lower() == "exit":
        break

    answer = ask(question)

    print("\n답변")
    print(answer)
    print("=" * 60)