#추론
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

import torch

MODEL_NAME = "google/gemma-4-E4B-it"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

from prompt import SYSTEM_PROMPT


def make_messages(user_input):

    return [

        {
            "role":"system",
            "content":SYSTEM_PROMPT
        },

        {
            "role":"user",
            "content":user_input
        }

    ]
    
def generate(user_input):

    messages = make_messages(user_input)

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][input_len:],
        skip_special_tokens=True,
    )

    return response.strip()