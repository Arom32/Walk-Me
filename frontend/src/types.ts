export interface HistoryTurn {
  question: string;
  answer?: string;
}

export interface GuideRequest {
  question: string;
  region?: string;
  k?: number;
  places_only?: boolean;
  with_llm?: boolean;
  no_lora?: boolean;
  tts?: boolean;
  history?: HistoryTurn[];
}

export interface Place {
  name?: string;
  address?: string;
  visit_type?: string;
}

export interface GuideResponse {
  question: string;
  region: string | null;
  places: Place[];
  standard: string;
  answer: string;
  dialect_ok: boolean;
  audio_url: string | null;
  audio_path?: string;
  tts_error?: string;
}

export const GANGWON_REGIONS = [
  "속초",
  "강릉",
  "춘천",
  "양양",
  "평창",
  "원주",
  "동해",
  "삼척",
] as const;

export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  text: string;
  standard?: string;
  places?: Place[];
  audioUrl?: string | null;
  ttsError?: string;
  pending?: boolean;
  error?: string;
}
