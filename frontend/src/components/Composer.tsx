import { useRef, type KeyboardEvent } from "react";
import { GANGWON_REGIONS } from "../types";

interface ComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  region: string | null;
  onRegionChange: (region: string | null) => void;
  wantTts: boolean;
  onWantTtsChange: (value: boolean) => void;
  disabled: boolean;
}

const QUICK_QUESTIONS = ["속초 가볼 만한 곳", "강릉 여행 명소", "양양 해변 추천", "평창 관광지 알려주세요"];

export default function Composer({
  value,
  onChange,
  onSubmit,
  region,
  onRegionChange,
  wantTts,
  onWantTtsChange,
  disabled,
}: ComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSubmit();
    }
  }

  function autoGrow(el: HTMLTextAreaElement) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  }

  return (
    <div className="composer">
      <div className="composer-quick scroll-thin">
        {QUICK_QUESTIONS.map((q) => (
          <button
            key={q}
            type="button"
            className="chip"
            disabled={disabled}
            onClick={() => onChange(q)}
          >
            {q}
          </button>
        ))}
      </div>

      <div className="composer-row">
        <textarea
          ref={textareaRef}
          className="composer-input"
          rows={1}
          value={value}
          placeholder="강원도 여행에 대해 물어보세요"
          disabled={disabled}
          onChange={(e) => {
            onChange(e.target.value);
            autoGrow(e.target);
          }}
          onKeyDown={handleKeyDown}
        />
        <button
          type="button"
          className="composer-send"
          disabled={disabled || !value.trim()}
          onClick={onSubmit}
          aria-label="전송"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path
              d="M2 8h11.5M9 3.5 13.5 8 9 12.5"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </div>

      <div className="composer-footer">
        <div className="region-chips scroll-thin">
          <button
            type="button"
            className={`chip chip-region ${region === null ? "chip-active" : ""}`}
            onClick={() => onRegionChange(null)}
          >
            전체
          </button>
          {GANGWON_REGIONS.map((r) => (
            <button
              key={r}
              type="button"
              className={`chip chip-region ${region === r ? "chip-active" : ""}`}
              onClick={() => onRegionChange(r === region ? null : r)}
            >
              {r}
            </button>
          ))}
        </div>

        <label className="tts-toggle">
          <input
            type="checkbox"
            checked={wantTts}
            onChange={(e) => onWantTtsChange(e.target.checked)}
          />
          <span className="tts-toggle-track">
            <span className="tts-toggle-thumb" />
          </span>
          <span className="tts-toggle-label">음성</span>
        </label>
      </div>
    </div>
  );
}
