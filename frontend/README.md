# frontend

React + Vite + TypeScript. 워메(Walk-Me!) 강원도 사투리 여행 도슨트 채팅 화면 — 질문 → 사투리 답변 · 추천 장소 · (선택) TTS 음성.

## 실행

```bash
npm install
npm run dev
```

`http://localhost:5173` 접속. 개발 서버는 `/guide`, `/health` 요청을 `vite.config.ts`의 proxy 설정을 통해 `http://localhost:8000`(backend)으로 넘긴다. 백엔드를 먼저 띄워둘 것:

```bash
cd ../backend
set PYTHONPATH=%CD%;%CD%\..\ai
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

백엔드를 다른 주소에서 띄웠다면 `.env.example`을 `.env`로 복사해 `VITE_API_BASE_URL`을 채운다.

## 구조

- `src/App.tsx` — 대화 상태 관리, `/guide` 호출, 현재 턴을 중앙 스테이지에 표시
- `src/components/Blob.tsx` — 상태(유휴/생각중/말하는중)에 따라 모양이 변하는 중앙 물방울, 말할 때는 실제 TTS 오디오 진폭에 반응
- `src/components/PlacesRow.tsx` — 최신 응답의 추천 장소를 가로 스크롤 카드로 표시
- `src/components/HistoryList.tsx` — 이전 대화를 접어서 보여주는 히스토리
- `src/components/Composer.tsx` — 입력창 + 빠른 질문 칩 + 강원도 지역 칩 + TTS 토글
- `src/api.ts` — `POST /guide` 호출, 오디오 URL 조합
- `src/types.ts` — 백엔드 `GuideRequest`/`GuideResponse` 타입 + `ChatMessage`/`GANGWON_REGIONS`

지역 칩은 실제 RAG가 인식하는 강원도 시/군(`ai/rag/ask.py`의 region 목록)과 맞춰뒀다. 다른 도(경상/전라 등) UI는 넣지 않았다 — 백엔드가 강원도 전용이라 없는 기능을 보여주면 안 되기 때문. 이미지·음성 입력, 지도 연동 등은 아직 범위 밖 (CLAUDE.md 참고 — 현재 구현 우선순위는 텍스트 질문 경로).

## 빌드

```bash
npm run build
```
