import { useId } from "react";
import { useGlide, useWidth } from "./useGlide";

// EnvelopeEditor in Vital's Envelope skin: Widget Background, a 2 px line fading from #1de9b6 to #40c4ff with
// the soft bloom Vital's line renderer gives it, a translucent fill of the same colours, ring markers in
// #64ffda at the breakpoints, and faint vertical time lines (Lighten Screen).

interface Props {
  attack: number;
  decay: number;
  sustain: number;
  /** null when the envelope's release is not one of the editable controls (ENV 2) */
  release: number | null;
  names: string[];
  height: number;
}

const PAD_X = 10;
const PAD_Y = 11;
// Vital's default decay/release power (curvature) is -2 (exponential-looking falls); attack power 0 is linear.
const curve = (t: number, power: number) => (Math.abs(power) < 1e-3 ? t : (Math.exp(power * t) - 1) / (Math.exp(power) - 1));

export function EnvelopeDisplay({ attack, decay, sustain, release, names, height }: Props) {
  const gid = useId().replace(/:/g, "");
  const [ref, width] = useWidth<HTMLDivElement>(400);
  const [a, d, s, r] = useGlide([attack, decay, sustain, release ?? 0.5], names);

  const w = Math.max(width, 100);
  const h = height;
  const innerW = w - 2 * PAD_X;
  const top = PAD_Y;
  const bottom = h - PAD_Y;
  const yOf = (level: number) => bottom - level * (bottom - top);

  // Each stage gets width in proportion to its knob position (Vital's times are already power-scaled on the
  // knob), plus a fixed stretch for the held part of the note.
  const unit = innerW / 4.2;
  const xa = PAD_X + unit * (0.04 + a * 0.96);
  const xd = xa + unit * (0.04 + d * 0.96);
  const xs = xd + unit * 0.7;
  const xr = Math.min(w - PAD_X, xs + unit * (0.04 + r * 0.96));

  const N = 28;
  const pts: string[] = [`M ${PAD_X} ${bottom}`, `L ${xa} ${top}`];
  for (let i = 1; i <= N; i++) pts.push(`L ${xa + (xd - xa) * (i / N)} ${yOf(1 - (1 - s) * curve(i / N, -2))}`);
  pts.push(`L ${xs} ${yOf(s)}`);
  const sustainPath = pts.join(" ");
  const rel: string[] = [`M ${xs} ${yOf(s)}`];
  for (let i = 1; i <= N; i++) rel.push(`L ${xs + (xr - xs) * (i / N)} ${yOf(s * (1 - curve(i / N, -2)))}`);
  const releasePath = rel.join(" ");
  const fillPath = `${sustainPath} ${rel.slice(1).join(" ")} L ${xr} ${bottom} Z`;
  const grid = Array.from({ length: 11 }, (_, i) => PAD_X + (innerW / 12) * (i + 1));
  const fixedRelease = release === null;

  return (
    <div ref={ref} className="relative w-full overflow-hidden rounded-[6px] bg-widget" style={{ height }}>
      <svg width={w} height={h} className="block">
        <defs>
          <linearGradient id={`line-${gid}`} x1="0" x2="1" y1="0" y2="0">
            <stop offset="0" stopColor="#1de9b6" />
            <stop offset="1" stopColor="#40c4ff" />
          </linearGradient>
          <linearGradient id={`fill-${gid}`} x1="0" x2="1" y1="0" y2="0">
            <stop offset="0" stopColor="#1de9b6" stopOpacity="0.24" />
            <stop offset="1" stopColor="#40c4ff" stopOpacity="0.24" />
          </linearGradient>
          <linearGradient id={`fade-${gid}`} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor="white" stopOpacity="1" />
            <stop offset="1" stopColor="white" stopOpacity="0.25" />
          </linearGradient>
          <mask id={`mask-${gid}`}>
            <rect x="0" y="0" width={w} height={h} fill={`url(#fade-${gid})`} />
          </mask>
        </defs>
        {grid.map((x) => (
          <line key={x} x1={x} x2={x} y1={0} y2={h} stroke="rgba(255,255,255,0.045)" strokeWidth={1} />
        ))}
        <line x1={xs} x2={xs} y1={0} y2={h} stroke="rgba(255,255,255,0.12)" strokeWidth={1} strokeDasharray="2 3" />
        <path d={fillPath} fill={`url(#fill-${gid})`} mask={`url(#mask-${gid})`} />
        <g style={{ filter: "blur(3px)" }} opacity={0.55}>
          <path d={sustainPath} fill="none" stroke={`url(#line-${gid})`} strokeWidth={4} strokeLinejoin="round" />
          {!fixedRelease && <path d={releasePath} fill="none" stroke={`url(#line-${gid})`} strokeWidth={4} />}
        </g>
        <path d={sustainPath} fill="none" stroke={`url(#line-${gid})`} strokeWidth={2} strokeLinejoin="round" />
        <path
          d={releasePath}
          fill="none"
          stroke={`url(#line-${gid})`}
          strokeWidth={2}
          strokeLinejoin="round"
          opacity={fixedRelease ? 0.35 : 1}
          strokeDasharray={fixedRelease ? "3 3" : undefined}
        />
        <Marker x={xa} y={top} />
        <Marker x={xd} y={yOf(s)} />
        <Marker x={xs} y={yOf(s)} />
        {!fixedRelease && <Marker x={xr} y={bottom} />}
      </svg>
      {fixedRelease && (
        <span className="pointer-events-none absolute top-[6px] right-[9px] text-[10px] tracking-[0.06em] text-white/30 uppercase">
          release not editable
        </span>
      )}
    </div>
  );
}

function Marker({ x, y }: { x: number; y: number }) {
  return <circle cx={x} cy={y} r={3.6} fill="var(--color-widget)" stroke="#64ffda" strokeWidth={1.8} />;
}
