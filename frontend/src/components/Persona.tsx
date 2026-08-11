import { useEffect, useState } from "react";
import { getAnalyser } from "../lib/audioAnalyser";

export type PersonaState = "idle" | "thinking" | "speaking";

// 처음(대기) 상태는 항상 1행 1번째 + 2행 5번째로 고정 — 응답 풀과 겹치면 안 됨.
const IDLE_POOL = ["/persona/idle.png", "/persona/cover-mouth.png"];

const THINKING_POOL = [
  "/persona/thinking.png",
  "/persona/point-up.png",
  "/persona/present.png",
  "/persona/cheek-side.png",
  "/persona/speak-open.png",
];

// idle-alt.png는 IDLE_POOL에서 쓰므로 여기서는 빼서 겹치지 않게 한다.
const SPEAKING_POOL = [
  "/persona/call-out.png",
  "/persona/arms-crossed.png",
  "/persona/speak-laugh.png",
  "/persona/smile-big.png",
];

function pickTwo(pool: string[]): [string, string] {
  if (pool.length <= 2) return [pool[0], pool[1] ?? pool[0]];
  const a = Math.floor(Math.random() * pool.length);
  let b = Math.floor(Math.random() * (pool.length - 1));
  if (b >= a) b += 1;
  return [pool[a], pool[b]];
}

/**
 * state가 (다시) 시작될 때마다 후보 목록에서 딱 2장만 무작위로 골라
 * 그 둘만 번갈아 보여준다 — 화질이 고르지 않은 사진이 섞여 있어서
 * 매번 전부 다 보여주기보단 두 장만 안정적으로 쓰는 쪽을 택했다.
 */
function usePosePair(pool: string[], intervalMs: number, enabled: boolean) {
  const [pair, setPair] = useState<[string, string]>(() => pickTwo(pool));
  const [i, setI] = useState(0);

  useEffect(() => {
    if (!enabled) return;
    setPair(pickTwo(pool));
    setI(0);
    // pool은 상수 배열이라 참조만 바뀌지 않으면 되므로 enabled만 의존성으로 둔다.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled]);

  useEffect(() => {
    if (!enabled) return;
    const id = setInterval(() => setI((v) => (v + 1) % 2), intervalMs);
    return () => clearInterval(id);
  }, [enabled, intervalMs]);

  return pair[i];
}

/** 말하는 동안 실제 TTS 오디오 진폭에 맞춰 살짝 커졌다 작아지는 펄스 값(0~1)을 낸다. */
function useAudioLevel(state: PersonaState, audioEl: HTMLAudioElement | null | undefined) {
  const [level, setLevel] = useState(0);

  useEffect(() => {
    if (state !== "speaking" || !audioEl) {
      setLevel(0);
      return;
    }
    const analyser = getAnalyser(audioEl);
    if (!analyser) return;

    const data = new Uint8Array(analyser.frequencyBinCount);
    let raf = 0;
    const tick = () => {
      analyser.getByteFrequencyData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i++) sum += data[i];
      setLevel(sum / data.length / 255);
      raf = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(raf);
  }, [state, audioEl]);

  return level;
}

export default function Persona({
  state,
  audioEl,
  size = 148,
}: {
  state: PersonaState;
  audioEl?: HTMLAudioElement | null;
  size?: number;
}) {
  const idleFrame = usePosePair(IDLE_POOL, 4800, state === "idle");
  const thinkingFrame = usePosePair(THINKING_POOL, 2200, state === "thinking");
  const speakingFrame = usePosePair(SPEAKING_POOL, 1800, state === "speaking");
  const level = useAudioLevel(state, audioEl);

  const src = state === "thinking" ? thinkingFrame : state === "speaking" ? speakingFrame : idleFrame;

  const style = {
    "--size": `${size}px`,
    "--level": level,
  } as React.CSSProperties;

  return (
    <div className="persona-pulse" style={style}>
      <div className={`persona persona-${state}`}>
        <img key={src} src={src} alt="강원도 사투리 도슨트" className="persona-photo" />
      </div>
    </div>
  );
}
