# 워메! (Walk-Me!) — 환경 설정 및 실행 가이드

사투리 기반 대화형 AI 관광 도슨트 서비스. 파이프라인 전체 개요는 `Walk-Me/CLAUDE.md`, 컴포넌트별 상세는 `ai/README.md` · `ai/tts/README.md` · `ai/rag/README.md` 참고. 이 문서는 **처음 이 리포를 받은 사람이 처음부터 끝까지 따라 하면 실행까지 되는 것**을 목표로 합니다. 설치·실행 중 에러가 나면 [`doc/troubleshooting.md`](doc/troubleshooting.md), TTS 음성 품질 문제는 [`doc/tts-voice-quality.md`](doc/tts-voice-quality.md) 참고.

## 0. 왜 conda env가 2개인가

`ai/llm`(Gemma-4 LoRA)과 `ai/tts`(CosyVoice)는 요구하는 `transformers` 버전이 정면충돌합니다 — 하나의 Python 환경에 같이 설치할 수 없습니다. 그래서:

| env | 용도 | 담당 |
|---|---|---|
| `walkme-llm` | RAG + LLM(LoRA) + backend, `pipeline.py` 실행 주체 | 사람이 직접 activate해서 씀 |
| `cosyvoice` | TTS(CosyVoice) 전용 | `pipeline.py`/backend가 서브프로세스로 자동 호출 — 직접 activate할 일은 TTS 단독 테스트할 때뿐 |

**둘 다 만들어놔야** 전체 파이프라인이 돌아갑니다. 아래 순서대로 하나씩 만듭니다.

## 1. 사전 준비물

- conda (miniconda/anaconda)
- NVIDIA GPU + 드라이버 (`nvidia-smi`로 확인)
- Google Drive 자산 접근 권한 (`.env.example`의 `DRIVE_ASSETS_FOLDER_ID`)

## 2. `cosyvoice` env (TTS) 만들기

```bash
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
cd ai/tts
pip install --no-cache-dir -r requirements.txt
```

설치 중 에러가 나면 [`doc/troubleshooting.md`](doc/troubleshooting.md) 참고.

설치 후 검증 (필수):

```bash
python smoke_test.py
```

`outputs/smoke_self_clone.wav`가 정상 생성되면 (에러 없이, 수십 초 안에 끝남) 성공입니다. 콘솔에 `family=v3`가 찍히는지도 확인하세요.

## 3. `walkme-llm` env (RAG + LLM + backend) 만들기

```bash
conda create -n walkme-llm -y python=3.12
conda activate walkme-llm
pip install --no-cache-dir -r ai/llm/requirements.txt -r ai/rag/requirements.txt -r backend/requirements.txt
```

설치 중 에러(numpy/Python 버전 불일치, `psycopg2` 빌드 실패 등)가 나면 [`doc/troubleshooting.md`](doc/troubleshooting.md) 참고. DB 없이 로컬 개발만 할 거면 `psycopg2` 대신 `psycopg2-binary`를 쓰면 되는데, 이 대체 설치 커맨드도 같은 문서에 있습니다.

## 4. 모델/데이터 자산 배치

`.env.example`을 복사해 `.env` 생성 후 필요시 경로 override:

```bash
cp .env.example .env
```

| 경로 | 내용 | 출처 |
|---|---|---|
| `ai/tts/models/kangwon/` | CosyVoice v3 파인튜닝 체크포인트 (`cosyvoice3.yaml`+`speech_tokenizer_v3.onnx`+`campplus.onnx`+`CosyVoice-BlankEN/`+`llm.pt`/`flow.pt`/`hift.pt`) | Drive `kangwon.zip` 그대로 풀기 |
| `ai/llm/lora_output/final/` | Gemma-4 LoRA 어댑터 | Drive |
| `ai/rag/data/chroma/` | 관광 정보 벡터 인덱스 | Drive (또는 `python -m rag.index`로 로컬 생성) |
| `ai/tts/prompts/*.wav` | zero-shot 화자 프롬프트 wav | Drive |

`google/gemma-4-E2B-it` 베이스 모델을 처음 받을 때 다운로드가 멈추면:

```bash
HF_HUB_DISABLE_XET=1 python pipeline.py "속초 가볼 만한 곳" --places-only --with-llm
```

모델을 전부 한 번씩 받아둔 뒤에는 아래처럼 `HF_HUB_OFFLINE=1`을 걸어서 실행하는 걸 권장합니다 (캐시가 없는 첫 다운로드 단계에서는 빼야 함):

```bash
HF_HUB_OFFLINE=1 python pipeline.py "속초 가볼 만한 곳" --places-only --with-llm --tts
```

두 옵션이 왜 필요한지(각각 어떤 문제를 우회하는지)는 [`doc/troubleshooting.md`](doc/troubleshooting.md) 참고.

## 5. 실행

### RAG만 (텍스트, LoRA 없이)

```bash
conda activate walkme-llm
cd ai
python -m rag.ask "속초 가볼 만한 곳" --places-only --region 속초
```

### RAG + 사투리(LoRA), TTS 없이

```bash
./run_pipeline_safe.sh "속초 가볼 만한 곳" --places-only --with-llm
```

### 전체 파이프라인 (RAG + 사투리 + TTS)

```bash
./run_pipeline_safe.sh "속초 가볼 만한 곳" --places-only --with-llm --tts
```

`cosyvoice` env를 따로 activate할 필요 없습니다 — `ai/tts/subprocess_client.py`가 내부적으로 `conda run -n cosyvoice ...`로 알아서 호출합니다. **정상이면 TTS 부분은 수십 초 안에 끝납니다.** 훨씬 오래 걸리거나 타임아웃 에러가 나면 [`doc/troubleshooting.md`](doc/troubleshooting.md) 참고.

`run_pipeline_safe.sh`는 `pipeline.py`를 그대로 감싸되 cgroup 메모리 상한(스왑 금지)을 겁니다 — TTS 구간에서 RAM이 부족해지면 스왑 스래싱으로 컴퓨터 전체가 먹통되는 대신, 그 프로세스만 즉시 OOM-kill되고 에러가 뜹니다. 한도는 `WALKME_PIPELINE_MEM_LIMIT`(파이프라인 전체, 기본 16G), `WALKME_TTS_MEM_LIMIT`(TTS 서브프로세스 자체, 기본 10G) 두 환경변수로 조절합니다. `systemd-run`이 없는 환경(WSL2/systemd 아닌 곳)에서는 경고만 찍고 한도 없이 그대로 실행됩니다.

### 백엔드 API

```bash
conda activate walkme-llm
cd backend
set PYTHONPATH=%CD%;%CD%\..\ai   # Windows
# export PYTHONPATH=$PWD:$PWD/../ai   # macOS/Linux
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

- `GET /health`
- `POST /guide` — `{"question": "속초 가볼 만한 곳", "with_llm": true, "tts": true}`
- `GET /guide/audio/{filename}`

DB까지 포함해서 통째로 띄우려면 리포 루트에서:

```bash
docker-compose up
```

## 6. 뭔가 이상할 때

설치·다운로드·실행 중 에러는 [`doc/troubleshooting.md`](doc/troubleshooting.md), TTS 음성 품질 문제는 [`doc/tts-voice-quality.md`](doc/tts-voice-quality.md) 참고. 실행 중 메모리/디스크 사용량이 튀며 컴퓨터가 버벅이면 위 5번의 `WALKME_PIPELINE_MEM_LIMIT`/`WALKME_TTS_MEM_LIMIT` 한도를 낮춰서 재시도해보세요.
