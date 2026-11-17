import { useEffect, useState } from "react";

/** Whether Shift is held, for fine knob drags (Vital uses the same modifier). */
export function useShiftKey() {
  const [shift, setShift] = useState(false);
  useEffect(() => {
    const on = (e: KeyboardEvent) => setShift(e.shiftKey);
    const blur = () => setShift(false);
    window.addEventListener("keydown", on);
    window.addEventListener("keyup", on);
    window.addEventListener("blur", blur);
    return () => {
      window.removeEventListener("keydown", on);
      window.removeEventListener("keyup", on);
      window.removeEventListener("blur", blur);
    };
  }, []);
  return shift;
}

export function prefersReducedMotion() {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export function isTypingTarget(el: EventTarget | null) {
  if (!(el instanceof HTMLElement)) return false;
  return el.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName);
}
