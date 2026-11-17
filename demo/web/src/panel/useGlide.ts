import { useEffect, useRef, useState } from "react";
import { animate, useMotionValue, useMotionValueEvent } from "motion/react";
import { useStore } from "../state/store";
import { prefersReducedMotion } from "../lib/hooks";

/**
 * Values for a display (envelope, filter curve) that glide with the knobs: over 600 ms when an agent round moved
 * one of `names`, instantly otherwise.
 */
export function useGlide(target: number[], names: string[]): number[] {
  const glowSeq = useStore((s) => s.glow.seq);
  const moved = useStore((s) => names.some((n) => s.glow.from[n] !== undefined));
  const t = useMotionValue(1);
  const [shown, setShown] = useState(target);
  const from = useRef(target);
  const to = useRef(target);
  const lastSeq = useRef(glowSeq);
  useMotionValueEvent(t, "change", (k) => {
    setShown(from.current.map((f, i) => f + (to.current[i] - f) * k));
  });
  const key = target.join(",");
  useEffect(() => {
    const agent = moved && lastSeq.current !== glowSeq;
    lastSeq.current = glowSeq;
    to.current = target;
    if (agent && !prefersReducedMotion()) {
      from.current = shown;
      t.jump(0);
      const a = animate(t, 1, { duration: 0.6, ease: [0.77, 0, 0.175, 1] });
      return () => a.stop();
    }
    from.current = target;
    t.jump(1);
    setShown(target);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, glowSeq]);
  return shown;
}

/** Track an element's width. */
export function useWidth<T extends HTMLElement>(fallback = 300) {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width] as const;
}
