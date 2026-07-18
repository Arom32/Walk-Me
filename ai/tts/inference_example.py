import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

MODEL_DIR = './models/kangwon'

# Drive에서 받은 프롬프트 wav를 ./prompts/ 에 넣고 파일명을 맞춰줄 것 (README 참고)
PROMPT_WAV = './prompts/st_set2_collectorgw185_speakergw1744_63_9.wav'
PROMPT_TEXT = '아까 내가 사이즈 먹고 그를 때부터 분멩이 택택할 거라고 했는데 나한테 어림도 웂으니까 하나 더 큰 거 주서요'


def main(text: str):
    cosyvoice = AutoModel(model_dir=MODEL_DIR)
    for i, j in enumerate(cosyvoice.inference_zero_shot(text, PROMPT_TEXT, PROMPT_WAV, stream=False)):
        out_path = f'out_{i}.wav'
        torchaudio.save(out_path, j['tts_speech'], cosyvoice.sample_rate)
        print(f'saved {out_path}')


if __name__ == '__main__':
    main('오늘 날씨가 참 좋네요.')
