import { useId } from "react";
import { useGlide, useWidth } from "./useGlide";

// FilterResponse in Vital's Filter skin: line from #ffb74d to #ff8180 with a bloom, translucent fill.
// The curve is an illustration, not Vital's filter model: a 12 dB/octave low-pass whose corner sits at the
// cutoff knob's position on a log-frequency axis (Vital's cutoff knob is linear in semitones), with resonance
// as the peak and dry/wet blending in the flat dry signal. Drive is not drawn. When envelope 2 moves the cutoff
// (its peak value differs from the resting cutoff), a fainter dashed curve marks the peak and a faint band the
// range swept between them.

interface Props {
  cutoff: number;
  /** cutoff at envelope 2's peak, on the same 0-1 scale */
  peak: number;
  resonance: number;
  mix: number;
  on: boolean;
  names: string[];
  height: number;
}

const DB_TOP = 18;
const DB_BOTTOM = -36;
const SEMITONES = 128; // Vital's cutoff spans MIDI notes 8 to 136
/** below this difference between peak and cutoff the envelope does not move the filter */
const SWEEP_MIN = 0.002;

function response(x: number, cutoff: number, resonance: number, mix: number) {
  const ratio = Math.pow(2, ((x - cutoff) * SEMITONES) / 12);
  const q = 0.707 * Math.pow(2, resonance * 3.2);
  const lp = 1 / Math.sqrt((1 - ratio * ratio) ** 2 + (ratio / q) ** 2);
  const mag = mix * lp + (1 - mix);
  return 20 * Math.log10(Math.max(mag, 1e-4));
}

export function FilterDisplay({ cutoff, peak, resonance, mix, on, names, height }: Props) {
  const gid = useId().replace(/:/g, "");
  const [ref, width] = useWidth<HTMLDivElement>(300);
  const [c, r, m, p] = useGlide([cutoff, resonance, mix, peak], names);
  const w = Math.max(width, 100);
  const h = height;
  const pad = 6;
  const yOf = (db: number) => pad + ((DB_TOP - Math.max(DB_BOTTOM - 6, Math.min(DB_TOP + 4, db))) / (DB_TOP - DB_BOTTOM)) * (h - 2 * pad);

  const N = 120;
  const curve = (corner: number) => {
    const pts: string[] = [];
    for (let i = 0; i <= N; i++) {
      const x = i / N;
      pts.push(`${i === 0 ? "M" : "L"} ${(x * w).toFixed(2)} ${yOf(response(x, corner, r, m)).toFixed(2)}`);
    }
    return pts.join(" ");
  };
  const line = curve(c);
  const sweep = Math.abs(p - c) > SWEEP_MIN;
  const fill = `${line} L ${w} ${h} L 0 ${h} Z`;
  // Octave grid: Vital's analysis views use faint vertical lines.
  const grid = Array.from({ length: 10 }, (_, i) => ((i + 1) * 12) / SEMITONES).map((x) => x * w);
  const zero = yOf(0);
  const stroke = on ? `url(#fline-${gid})` : "var(--color-arc-off)";

  return (
    <div ref={ref} className="relative w-full overflow-hidden rounded-[6px] bg-widget" style={{ height }} data-testid="filter-display">
      <svg width={w} height={h} className="block">
        <defs>
          <linearGradient id={`fline-${gid}`} x1="0" x2="1" y1="0" y2="0">
            <stop offset="0" stopColor="#ffb74d" />
            <stop offset="1" stopColor="#ff8180" />
          </linearGradient>
          <linearGradient id={`ffill-${gid}`} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor="#ffb74d" stopOpacity={on ? 0.32 : 0.1} />
            <stop offset="1" stopColor="#ff8180" stopOpacity={on ? 0.04 : 0.02} />
          </linearGradient>
        </defs>
        {grid.map((x) => (
          <line key={x} x1={x} x2={x} y1={0} y2={h} stroke="rgba(255,255,255,0.045)" strokeWidth={1} />
        ))}
        <line x1={0} x2={w} y1={zero} y2={zero} stroke="rgba(255,255,255,0.08)" strokeWidth={1} />
        {sweep && (
          <g data-testid="filter-sweep">
            <rect
              x={Math.min(c, p) * w}
              width={Math.abs(p - c) * w}
              y={0}
              height={h}
              fill={on ? "rgba(255,183,77,0.06)" : "rgba(255,255,255,0.025)"}
            />
            <path
              d={curve(p)}
              fill="none"
              stroke={on ? "#ffb74d" : "var(--color-arc-off)"}
              strokeOpacity={on ? 0.45 : 0.5}
              strokeWidth={1.5}
              strokeDasharray="4 3"
              strokeLinejoin="round"
            />
            <line x1={p * w} x2={p * w} y1={0} y2={h} stroke={on ? "rgba(255,183,77,0.25)" : "rgba(255,255,255,0.07)"} strokeWidth={1} strokeDasharray="2 3" />
          </g>
        )}
        <path d={fill} fill={`url(#ffill-${gid})`} />
        {on && <path d={line} fill="none" stroke={stroke} strokeWidth={4} opacity={0.5} style={{ filter: "blur(3px)" }} />}
        <path d={line} fill="none" stroke={stroke} strokeWidth={2} strokeLinejoin="round" />
        <line x1={c * w} x2={c * w} y1={0} y2={h} stroke={on ? "rgba(255,183,77,0.35)" : "rgba(255,255,255,0.1)"} strokeWidth={1} strokeDasharray="2 3" />
      </svg>
      <span className="pointer-events-none absolute top-[6px] right-[9px] text-[10px] tracking-[0.06em] text-white/30 uppercase">
        {on ? "approx. response" : "filter off"}
      </span>
    </div>
  );
}
