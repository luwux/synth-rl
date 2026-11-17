// The server sends display text with units, already rounded to three significant digits: "1240 Hz", "0.0064 s",
// "56.7 %", "-6 dB", "3 voices", "on", "saw to square, 18%". The numbers are shown as sent (rounding them again
// would turn "0.0064 s" into "0 s"); only the layout changes: Hz from 1000 up as kHz, and "%" without a space.

/** Three significant digits without trailing zeros: 1240 -> "1.24", 12500 -> "12.5", 1000 -> "1". */
function threeDigits(x: number): string {
  const s = x.toPrecision(3);
  return s.includes(".") && !s.includes("e") ? s.replace(/0+$/, "").replace(/\.$/, "") : String(Number(s));
}

export function prettyDisplay(text: string | undefined | null): string {
  const t = (text ?? "").trim();
  const m = t.match(/^([-+]?\d*\.?\d+)\s*(Hz|%)$/);
  if (!m) return t;
  if (m[2] === "%") return `${m[1]}%`;
  const hz = Number(m[1]);
  return Math.abs(hz) >= 1000 ? `${threeDigits(hz / 1000)} kHz` : t;
}
