# 워메! (Walk-Me!) — 환경 설정 및 실행 가이드

사투리 기반 대화형 AI 관광 도슨트 서비스. 파이프라인 전체 개요는 `Walk-Me/CLAUDE.md`, 컴포넌트별 상세는 `ai/README.md` · `ai/tts/README.md` · `ai/rag/README.md` 참고. 이 문서는 **처음 이 리포를 받은 사람이 처음부터 끝까지 따라 하면 실행까지 되는 것**을 목표로 합니다.

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

### 설치 중 흔한 에러와 대처

- **`ModuleNotFoundError: No module named 'pkg_resources'` (openai-whisper 빌드 실패)**
  최신 setuptools가 `pkg_resources`를 지원 안 해서 생김. 아래로 먼저 해결한 뒤 재시도:
  ```bash
  pip install "setuptools<81"
  pip install --no-cache-dir --no-build-isolation -r requirements.txt
  ```
- **torch 버전이 의도치 않게 바뀜 (예: `torch==2.3.1`이어야 하는데 다른 버전으로 깔림)**
  개별 패키지를 버전 없이 따로 설치하면 torch가 딸려 올라갈 수 있습니다. 항상 `requirements.txt`에 정확한 버전을 추가하고 `-r requirements.txt`로 전체를 다시 설치하세요 (개별 `pip install <package>` 지양).

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

### 설치 중 흔한 에러와 대처

- **`numpy==2.5.0`을 못 찾음 / Python 버전 불일치**
  `ai/llm/requirements.txt`의 `numpy==2.5.0`은 **Python 3.12 이상**이 필요합니다. env를 3.12로 만들었는지 확인하세요.
- **`psycopg2` 빌드 실패 (`pg_config executable not found`)**
  PostgreSQL 개발 헤더(`libpq-dev`)가 로컬에 없어서 나는 에러입니다. `backend`의 `/guide` 등 핵심 API는 DB 없이도 동작하니, 로컬 개발용으로는 `psycopg2` 대신 미리 빌드된 걸 씁니다:
  ```bash
  grep -v "^psycopg2==" backend/requirements.txt > /tmp/backend-req.txt
  pip install --no-cache-dir -r ai/llm/requirements.txt -r ai/rag/requirements.txt -r /tmp/backend-req.txt psycopg2-binary==2.9.11
  ```
  (실제 DB 연결까지 테스트하려면 `docker-compose up`으로 Postgres를 띄우고 `libpq-dev`를 설치해 진짜 `psycopg2`를 쓰세요 — `backend/Dockerfile` 참고.)

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

**HuggingFace에서 `google/gemma-4-E2B-it` 베이스 모델을 처음 받을 때**: 용량이 크고(10GB+), `hf_xet` 전송 백엔드가 이 환경에서 종종 다운로드가 멈추는 문제가 있었습니다. 멈추면 아래처럼 xet을 끄고 재시도하세요 (huggingface_hub가 이어받기를 지원하니 그냥 재실행하면 됨):

```bash
HF_HUB_DISABLE_XET=1 python pipeline.py "속초 가볼 만한 곳" --places-only --with-llm
```

**필요한 모델을 전부 한 번씩 받아둔 뒤에는** (`google/gemma-4-E2B-it`, `jhgan/ko-sroberta-multitask` 등) `HF_HUB_OFFLINE=1`을 걸어서 실행하는 걸 권장합니다. 이 모델들이 로컬에 이미 캐시돼 있어도, huggingface_hub/SentenceTransformer는 매번 HF Hub에 메타데이터를 재확인하려고 네트워크를 탑니다 — 평소엔 금방 끝나지만, 이 환경에서 HF Hub 응답이 느려지면 **타임아웃 없이 무한정 멈춰버립니다** (RAG 검색기 초기화 단계에서 특히 자주 겪음, Ctrl+C로 끊어야 함). `HF_HUB_OFFLINE=1`은 캐시만 쓰고 이 네트워크 확인을 아예 건너뜁니다:

```bash
HF_HUB_OFFLINE=1 python pipeline.py "속초 가볼 만한 곳" --places-only --with-llm --tts
```

(처음 자산을 받는 단계에서는 당연히 빼야 합니다 — 캐시가 없는 상태에서 offline 모드면 다운로드 자체가 안 됩니다.)

## 5. 실행

### RAG만 (텍스트, LoRA 없이)

```bash
conda activate walkme-llm
cd ai
python -m rag.ask "속초 가볼 만한 곳" --places-only --region 속초
```

### RAG + 사투리(LoRA), TTS 없이

```bash
python pipeline.py "속초 가볼 만한 곳" --places-only --with-llm
```

### 전체 파이프라인 (RAG + 사투리 + TTS)

```bash
python pipeline.py "속초 가볼 만한 곳" --places-only --with-llm --tts
```

`cosyvoice` env를 따로 activate할 필요 없습니다 — `ai/tts/subprocess_client.py`가 내부적으로 `conda run -n cosyvoice ...`로 알아서 호출합니다. **정상이면 TTS 부분은 수십 초 안에 끝납니다.** 훨씬 오래 걸리거나 아래 같은 에러가 나면 GPU를 못 잡고 CPU로 밀린 것입니다:

```
TTS 서브프로세스가 120초 안에 끝나지 않아 강제 종료했습니다...
```

이럴 땐 `nvidia-smi`로 GPU 메모리를 누가 잡고 있는지 확인하세요. 특히 `walkme-llm` 쪽에서 LLM을 로딩한 채로 다른 요청을 또 보내면(같은 프로세스를 재사용하지 않는 한) VRAM이 꽉 차서 TTS가 CPU로 밀릴 수 있습니다 — 이런 경우 GPU 프로세스를 정리하고 다시 시도하세요.

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

## 6. 뭔가 이상할 때 체크리스트

1. **"[경고] LoRA 사투리 변환 실패, RAG 템플릿으로 폴백"이 뜬다** → `walkme-llm` env가 아니라 다른(transformers 낡은) env에서 실행 중일 가능성. `conda activate walkme-llm` 확인.
2. **TTS 음성이 깨진다 / self-clone이 이상하다** → `ai/tts/README.md`의 v1/v3 버전 불일치 체크리스트 참고. `python smoke_test.py`로 재현부터.
3. **TTS가 비정상적으로 오래 걸린다(수 분 이상)** → GPU 경합/CPU 폴백 의심, `nvidia-smi`로 확인.
4. **`ai/tts` 쪽 새 `ModuleNotFoundError`가 뜬다** → `Model_TTS/CosyVoice/requirements.txt`(원본 학습 리포)에서 같은 패키지의 정확한 버전을 찾아 `ai/tts/requirements.txt`에 추가 후 재설치. (이 리포는 학습 스택 대비 추론에 필요한 것만 추린 서브셋이라 가끔 하나씩 빠져 있을 수 있음)
5. **RAG 검색기 초기화 단계(`TravelRetriever()`)에서 응답 없이 멈춘다** → 임베딩 모델(`jhgan/ko-sroberta-multitask`)이 로컬에 캐시돼 있어도 HF Hub에 메타데이터를 재확인하려고 네트워크를 타는데, 이게 타임아웃 없이 무한정 멈출 수 있습니다. `HF_HUB_OFFLINE=1`로 재실행 (위 4번 항목 참고).
