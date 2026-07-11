import torch
from transformers import ( AutoProcessor, AutoModelForCausalLM )
from model_config import bnb_config

MODEL_ID = "google/gemma-4-12b-it"

def get_model():

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(

        MODEL_ID,

        quantization_config=bnb_config,

        device_map="auto"

    )

    return processor, model

if __name__ == "__main__":
    # 코드 작동 테스트
    model, processor = get_model()
    print("연결완료")