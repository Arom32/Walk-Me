import { useMemo, useRef, useState } from "react";
import { fetchGuide, audioUrl } from "./api";
import type { ChatMessage, HistoryTurn } from "./types";
import Persona, { type PersonaState } from "./components/Persona";
import PlacesRow from "./components/PlacesRow";
import HistorySidebar from "./components/HistorySidebar";
import Composer from "./components/Composer";
import AudioWaveform from "./components/AudioWaveform";
import "./App.css";

const INTRO: ChatMessage = {
  id: "intro",
  role: "assistant",
  text: "어서 오드래오, 무사가 궁금해가꼬 왔는가?",
};

/** 완료된 질문/답변 턴만 골라 백엔드로 보낼 히스토리를 만든다 (인사말·대기중·에러 턴 제외). */
function buildHistory(messages: ChatMessage[]): HistoryTurn[] {
  const turns: HistoryTurn[] = [];
  let pendingQuestion: string | null = null;

  for (const m of messages) {
    if (m.role === "user") {
      pendingQuestion = m.text;
    } else if (pendingQuestion && !m.pending && !m.error) {
      turns.push({ question: pendingQuestion, answer: m.text });
      pendingQuestion = null;
    } else if (pendingQuestion) {
      pendingQuestion = null;
    }
  }
  return turns;
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([INTRO]);
  const [input, setInput] = useState("");
  const [region, setRegion] = useState<string | null>(null);
  const [wantTts, setWantTts] = useState(false);
  const [loading, setLoading] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const { stageUser, stageAssistant, history } = useMemo(() => {
    const last = messages[messages.length - 1];
    const secondLast = messages[messages.length - 2];
    if (secondLast?.role === "user") {
      return {
        stageUser: secondLast,
        stageAssistant: last,
        history: messages.slice(0, messages.length - 2),
      };
    }
    return {
      stageUser: null as ChatMessage | null,
      stageAssistant: last,
      history: messages.slice(0, messages.length - 1),
    };
  }, [messages]);

  async function handleSubmit() {
    const question = input.trim();
    if (!question || loading) return;

    const conversationHistory = buildHistory(messages);

    const userMsg: ChatMessage = { id: `u-${Date.now()}`, role: "user", text: question };
    const pendingId = `a-${Date.now()}`;
    const pendingMsg: ChatMessage = { id: pendingId, role: "assistant", text: "", pending: true };

    setMessages((prev) => [...prev, userMsg, pendingMsg]);
    setInput("");
    setLoading(true);
    setIsSpeaking(false);

    try {
      const res = await fetchGuide({
        question,
        region: region ?? undefined,
        places_only: true,
        with_llm: true,
        tts: wantTts,
        history: conversationHistory,
      });

      setMessages((prev) =>
        prev.map((m) =>
          m.id === pendingId
            ? {
                ...m,
                pending: false,
                text: res.answer,
                standard: res.standard,
                places: res.places,
                audioUrl: res.audio_url,
                ttsError: res.tts_error,
              }
            : m,
        ),
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다.";
      setMessages((prev) =>
        prev.map((m) => (m.id === pendingId ? { ...m, pending: false, error: message } : m)),
      );
    } finally {
      setLoading(false);
    }
  }

  const personaState: PersonaState = loading ? "thinking" : isSpeaking ? "speaking" : "idle";
  const showStage = !!stageAssistant;
  const hasAnswered = messages.some((m) => m.role === "user");

  function handleReset() {
    if (audioRef.current) {
      audioRef.current.pause();
    }
    setMessages([INTRO]);
    setInput("");
    setRegion(null);
    setLoading(false);
    setIsSpeaking(false);
  }

  return (
    <div className="page">
      <header className="brand-header">
        <button
          type="button"
          className="brand-logo-button"
          onClick={handleReset}
          aria-label="처음으로 돌아가기"
          title="처음으로 돌아가기"
        >
          <img src="/logo.png" alt="워메 Walk Me!" className="brand-logo" />
        </button>

        {history.length > 0 && (
          <button type="button" className="history-open-btn" onClick={() => setHistoryOpen(true)}>
            대화 기록
          </button>
        )}
      </header>

      <HistorySidebar messages={history} open={historyOpen} onClose={() => setHistoryOpen(false)} />

      <main className={`stage ${hasAnswered ? "stage-compact" : ""}`}>
        <Persona state={personaState} audioEl={audioRef.current} size={hasAnswered ? 148 : 252} />

        <div className="stage-scroll scroll-thin">
          {showStage && stageAssistant && (
            <div className="stage-answer">
              {stageUser && <p className="stage-question">{stageUser.text}</p>}

              {stageAssistant.pending ? (
                <p className="stage-thinking">가만있어 보자, 곰곰 생각허는 중이라</p>
              ) : stageAssistant.error ? (
                <p className="stage-error">{stageAssistant.error}</p>
              ) : (
                <>
                  <p className="stage-text">{stageAssistant.text}</p>

                  {stageAssistant.standard && stageAssistant.standard !== stageAssistant.text && (
                    <details className="stage-standard">
                      <summary>표준어 원문</summary>
                      <p>{stageAssistant.standard}</p>
                    </details>
                  )}

                  {stageAssistant.audioUrl && (
                    <>
                      <audio
                        ref={audioRef}
                        autoPlay
                        src={audioUrl(stageAssistant.audioUrl)}
                        onPlay={() => setIsSpeaking(true)}
                        onPause={() => setIsSpeaking(false)}
                        onEnded={() => setIsSpeaking(false)}
                      />
                      <AudioWaveform audioEl={audioRef.current} active={isSpeaking} />
                    </>
                  )}
                  {stageAssistant.ttsError && (
                    <p className="stage-tts-error">음성 생성 실패: {stageAssistant.ttsError}</p>
                  )}

                  {stageAssistant.places && <PlacesRow places={stageAssistant.places} />}
                </>
              )}
            </div>
          )}
        </div>
      </main>

      <Composer
        value={input}
        onChange={setInput}
        onSubmit={handleSubmit}
        region={region}
        onRegionChange={setRegion}
        wantTts={wantTts}
        onWantTtsChange={setWantTts}
        disabled={loading}
      />
    </div>
  );
}
