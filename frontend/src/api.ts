import type { GuideRequest, GuideResponse } from "./types";

// dev 서버는 vite.config.ts의 proxy를 통해 /guide, /health를 backend:8000으로 넘긴다.
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

export async function fetchGuide(req: GuideRequest): Promise<GuideResponse> {
  const res = await fetch(`${API_BASE}/guide`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`서버 오류 (${res.status}): ${detail || res.statusText}`);
  }

  return res.json() as Promise<GuideResponse>;
}

export function audioUrl(path: string): string {
  return `${API_BASE}${path}`;
}
