import { useEffect, useRef } from "react";
import { getAnalyser } from "../lib/audioAnalyser";

const BAR_COUNT = 32;

export default function AudioWaveform({
  audioEl,
  active,
}: {
  audioEl: HTMLAudioElement | null;
  active: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !audioEl) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const analyser = getAnalyser(audioEl);
    if (!analyser) return;

    const dpr = window.devicePixelRatio || 1;
    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    const color = getComputedStyle(document.documentElement).getPropertyValue("--on-glass-accent").trim() || "#6cd0ef";
    const data = new Uint8Array(analyser.frequencyBinCount);
    const step = Math.max(1, Math.floor(data.length / BAR_COUNT));
    const barGap = 3;
    const barWidth = (width - barGap * (BAR_COUNT - 1)) / BAR_COUNT;

    let raf = 0;
    const draw = () => {
      analyser.getByteFrequencyData(data);
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = color;

      for (let i = 0; i < BAR_COUNT; i++) {
        const level = data[i * step] / 255;
        const barH = Math.max(3, level * height);
        const x = i * (barWidth + barGap);
        const y = (height - barH) / 2;
        const r = Math.min(barWidth / 2, 2);
        ctx.beginPath();
        ctx.roundRect(x, y, barWidth, barH, r);
        ctx.fill();
      }
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [audioEl]);

  return (
    <canvas
      ref={canvasRef}
      className={`waveform ${active ? "waveform-active" : ""}`}
      style={{ width: 220, height: 34 }}
    />
  );
}
