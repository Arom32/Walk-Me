let sharedCtx: AudioContext | null = null;
const analyserCache = new WeakMap<HTMLAudioElement, AnalyserNode>();

/**
 * 오디오 엘리먼트 하나당 MediaElementSource는 한 번만 만들 수 있어서
 * (두 번째 호출은 예외를 던짐) 여러 컴포넌트가 같은 <audio>를 분석해야 할 때
 * (립싱크용 Persona, 파형용 AudioWaveform 등) 이 캐시를 공유해서 재사용한다.
 */
export function getAnalyser(audioEl: HTMLAudioElement): AnalyserNode | null {
  try {
    if (!sharedCtx) sharedCtx = new AudioContext();
    if (sharedCtx.state === "suspended") void sharedCtx.resume();

    let analyser = analyserCache.get(audioEl);
    if (!analyser) {
      const source = sharedCtx.createMediaElementSource(audioEl);
      analyser = sharedCtx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.6;
      source.connect(analyser);
      analyser.connect(sharedCtx.destination);
      analyserCache.set(audioEl, analyser);
    }
    return analyser;
  } catch {
    return null;
  }
}
