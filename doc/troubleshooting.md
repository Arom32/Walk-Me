# 트러블슈팅 (설치 · 자산 다운로드 · 실행)

env 설치, 모델/데이터 자산 준비, 실행 중 겪을 수 있는 문제 모음. 실행 방법 자체는 리포 루트 `README.md` 참고. TTS 음성이 깨지는 문제는 별도 문서 [`tts-voice-quality.md`](./tts-voice-quality.md) 참고.

## `cosyvoice` env 설치 에러

- **`ModuleNotFoundError: No module named 'pkg_resources'` (openai-whisper 빌드 실패)**
  최신 setuptools가 `pkg_resources`를 지원 안 해서 생김. 아래로 먼저 해결한 뒤 재시도:
  ```bash
  pip install "setuptools<81"
  pip install --no-cache-dir --no-build-isolation -r requirements.txt
  ```
- **torch 버전이 의도치 않게 바뀜 (예: `torch==2.3.1`이어야 하는데 다른 버전으로 깔림)**
  개별 패키지를 버전 없이 따로 설치하면 torch가 딸려 올라갈 수 있습니다. 항상 `requirements.txt`에 정확한 버전을 추가하고 `-r requirements.txt`로 전체를 다시 설치하세요 (개별 `pip install <package>` 지양).

## `walkme-llm` env 설치 에러

- **`numpy==2.5.0`을 못 찾음 / Python 버전 불일치**
  `ai/llm/requirements.txt`의 `numpy==2.5.0`은 **Python 3.12 이상**이 필요합니다. env를 3.12로 만들었는지 확인하세요.
- **`psycopg2` 빌드 실패 (`pg_config executable not found`)**
  PostgreSQL 개발 헤더(`libpq-dev`)가 로컬에 없어서 나는 에러입니다. `backend`의 `/guide` 등 핵심 API는 DB 없이도 동작하니, 로컬 개발용으로는 `psycopg2` 대신 미리 빌드된 걸 씁니다:
  ```bash
  grep -v "^psycopg2==" backend/requirements.txt > /tmp/backend-req.txt
  pip install --no-cache-dir -r ai/llm/requirements.txt -r ai/rag/requirements.txt -r /tmp/backend-req.txt psycopg2-binary==2.9.11
  ```
  (실제 DB 연결까지 테스트하려면 `docker-compose up`으로 Postgres를 띄우고 `libpq-dev`를 설치해 진짜 `psycopg2`를 쓰세요 — `backend/Dockerfile` 참고.)

## 모델 자산 다운로드 문제

**HuggingFace에서 `google/gemma-4-E2B-it` 베이스 모델을 처음 받을 때**: 용량이 크고(10GB+), `hf_xet` 전송 백엔드가 이 환경에서 종종 다운로드가 멈추는 문제가 있었습니다. 멈추면 아래처럼 xet을 끄고 재시도하세요 (huggingface_hub가 이어받기를 지원하니 그냥 재실행하면 됨):

```bash
HF_HUB_DISABLE_XET=1 python pipeline.py "속초 가볼 만한 곳" --places-only --with-llm
```

**필요한 모델을 전부 한 번씩 받아둔 뒤에는** (`google/gemma-4-E2B-it`, `jhgan/ko-sroberta-multitask` 등) `HF_HUB_OFFLINE=1`을 걸어서 실행하는 걸 권장합니다. 이 모델들이 로컬에 이미 캐시돼 있어도, huggingface_hub/SentenceTransformer는 매번 HF Hub에 메타데이터를 재확인하려고 네트워크를 탑니다 — 평소엔 금방 끝나지만, 이 환경에서 HF Hub 응답이 느려지면 **타임아웃 없이 무한정 멈춰버립니다** (RAG 검색기 초기화 단계에서 특히 자주 겪음, Ctrl+C로 끊어야 함). `HF_HUB_OFFLINE=1`은 캐시만 쓰고 이 네트워크 확인을 아예 건너뜁니다:

```bash
HF_HUB_OFFLINE=1 python pipeline.py "속초 가볼 만한 곳" --places-only --with-llm --tts
```

(처음 자산을 받는 단계에서는 당연히 빼야 합니다 — 캐시가 없는 상태에서 offline 모드면 다운로드 자체가 안 됩니다.)

## 실행 중 문제 체크리스트

1. **"[경고] LoRA 사투리 변환 실패, RAG 템플릿으로 폴백"이 뜬다** → `walkme-llm` env가 아니라 다른(transformers 낡은) env에서 실행 중일 가능성. `conda activate walkme-llm` 확인.

   과거엔 `ai/rag/ask.py`의 `convert_to_dialect()`에 LLM 호출 실패를 완전히 삼키는 `except Exception: return body, True`가 있었습니다. 이 때문에 env 분리 전에는 gemma-4 로딩이 매번 실패해도 에러 없이 RAG 템플릿 문장(장소명 + 고정 사투리 어미)으로 조용히 폴백됐고, 팀은 이걸 "LoRA가 정상 동작한다"고 착각했었습니다. 지금은 실패 시 위 경고를 찍도록 고쳐져 있으니, 이 경고가 보이면 잘못된 env에서 실행 중이라는 뜻입니다.

2. **TTS 음성이 깨진다 / self-clone이 이상하다** → [`tts-voice-quality.md`](./tts-voice-quality.md)의 v1/v3 버전 불일치 체크리스트 참고. `python smoke_test.py`로 재현부터.
3. **TTS가 비정상적으로 오래 걸린다(수 분 이상)** → 정상이면 TTS는 수십 초 안에 끝납니다. 훨씬 오래 걸리거나 아래 같은 에러가 나면 GPU를 못 잡고 CPU로 밀린 것입니다:
   ```
   TTS 서브프로세스가 120초 안에 끝나지 않아 강제 종료했습니다...
   ```
   `nvidia-smi`로 GPU 메모리를 누가 잡고 있는지 확인하세요. 특히 `walkme-llm` 쪽에서 LLM을 로딩한 채로 다른 요청을 또 보내면(같은 프로세스를 재사용하지 않는 한) VRAM이 꽉 차서 TTS가 CPU로 밀릴 수 있습니다 — 이런 경우 GPU 프로세스를 정리하고 다시 시도하세요.
4. **`ai/tts` 쪽 새 `ModuleNotFoundError`가 뜬다** → `Model_TTS/CosyVoice/requirements.txt`(원본 학습 리포)에서 같은 패키지의 정확한 버전을 찾아 `ai/tts/requirements.txt`에 추가 후 재설치. (이 리포는 학습 스택 대비 추론에 필요한 것만 추린 서브셋이라 가끔 하나씩 빠져 있을 수 있음)
5. **RAG 검색기 초기화 단계(`TravelRetriever()`)에서 응답 없이 멈춘다** → 임베딩 모델(`jhgan/ko-sroberta-multitask`)이 로컬에 캐시돼 있어도 HF Hub에 메타데이터를 재확인하려고 네트워크를 타는데, 이게 타임아웃 없이 무한정 멈출 수 있습니다. `HF_HUB_OFFLINE=1`로 재실행 (위 "모델 자산 다운로드 문제" 참고).
