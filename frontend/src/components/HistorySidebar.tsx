import type { ChatMessage } from "../types";

function pairUp(messages: ChatMessage[]) {
  const pairs: { user?: ChatMessage; assistant?: ChatMessage }[] = [];
  for (const m of messages) {
    if (m.role === "user") {
      pairs.push({ user: m });
    } else if (pairs.length > 0 && !pairs[pairs.length - 1].assistant) {
      pairs[pairs.length - 1].assistant = m;
    } else {
      pairs.push({ assistant: m });
    }
  }
  return pairs;
}

export default function HistorySidebar({
  messages,
  open,
  onClose,
}: {
  messages: ChatMessage[];
  open: boolean;
  onClose: () => void;
}) {
  const pairs = pairUp(messages);

  return (
    <>
      <div
        className={`history-backdrop ${open ? "history-backdrop-open" : ""}`}
        onClick={onClose}
        aria-hidden="true"
      />
      <aside className={`history-sidebar ${open ? "history-sidebar-open" : ""}`}>
        <div className="history-sidebar-head">
          <h2>이전 대화</h2>
          <button type="button" className="history-close-btn" onClick={onClose} aria-label="닫기">
            ✕
          </button>
        </div>

        <div className="history-sidebar-body scroll-thin">
          {pairs.length === 0 ? (
            <p className="history-empty">아직 나눈 대화가 없습니다.</p>
          ) : (
            <ul className="history-list">
              {pairs.map((pair, i) => (
                <li key={i} className="history-item">
                  {pair.user && <p className="history-q">{pair.user.text}</p>}
                  {pair.assistant && !pair.assistant.pending && !pair.assistant.error && (
                    <p className="history-a">{pair.assistant.text}</p>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>
    </>
  );
}
