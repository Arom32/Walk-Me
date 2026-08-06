import sys
from pathlib import Path

sys.path.append("third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import AutoModel

try:
    import soundfile as sf
except ImportError:
    sf = None
import torchaudio

MODEL_DIR = "./models/kangwon"
PROMPT_WAV = "./prompts/st_set2_collectorgw185_speakergw1744_63_9.wav"
PROMPT_TRANSCRIPT = (
    "아까 내가 사이즈 먹고 그를 때부터 분멩이 택택할 거라고 했는데 "
    "나한테 어림도 웂으니까 하나 더 큰 거 주서요"
)


def _prompt_for_model(model_dir: str) -> str:
    root = Path(model_dir)
    if (root / "cosyvoice3.yaml").exists():
        return "You are a helpful assistant.<|endofprompt|>" + PROMPT_TRANSCRIPT
    return PROMPT_TRANSCRIPT


def main(text: str):
    prompt_text = _prompt_for_model(MODEL_DIR)
    print("prompt_text:", prompt_text)
    cosyvoice = AutoModel(model_dir=MODEL_DIR)
    for i, j in enumerate(
        cosyvoice.inference_zero_shot(
            text, prompt_text, PROMPT_WAV, stream=False, text_frontend=False
        )
    ):
        out_path = f"out_{i}.wav"
        wav = j["tts_speech"].detach().cpu()
        if sf is not None:
            arr = wav.mean(dim=0).numpy() if wav.ndim == 2 else wav.numpy()
            sf.write(out_path, arr, cosyvoice.sample_rate)
        else:
            torchaudio.save(out_path, wav if wav.ndim == 2 else wav.unsqueeze(0), cosyvoice.sample_rate)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main("오늘 날씨가 참 좋네요.")
