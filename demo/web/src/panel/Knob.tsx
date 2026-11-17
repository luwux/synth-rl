import { useEffect, useId, useRef, useState } from "react";
import { KnobHeadless, useKnobKeyboardControls } from "react-knob-headless";
import { animate, motion, useMotionValue, useTransform, type MotionValue } from "motion/react";
import { Tooltip } from "radix-ui";
import type { Control } from "../api/types";
import { useStore } from "../state/store";
import { prefersReducedMotion, useShiftKey } from "../lib/hooks";
import { BIPOLAR, S, SHORT, STEPS } from "./layout";
import { prettyDisplay } from "../lib/format";

// Vital's rotary: 0.8 pi either side of 12 o'clock, arc 32 px across and 2.4 px thick inside a 40 px body,
// a white hand half the radius long, and 1.4x thickness on hover (synth_slider.cpp, default.vitalskin).
const MAX_ARC = 0.8 * Math.PI;
const BOX = 48;
const C = BOX / 2;
const ARC_R = 16;
const BODY_R = 20;
const THICK = 2.4;
const HAND = 8.6;

const angleOf = (t: number) => -MAX_ARC + 2 * MAX_ARC * Math.min(1, Math.max(0, t));
const polar = (r: number, a: number) => [C + r * Math.sin(a), C - r * Math.cos(a)] as const;
function arcPath(r: number, a0: number, a1: number) {
  if (Math.abs(a1 - a0) < 0.002) a1 = a0 + 0.002;
  const [x0, y0] = polar(r, a0);
  const [x1, y1] = polar(r, a1);
  const large = Math.abs(a1 - a0) > Math.PI ? 1 : 0;
  const sweep = a1 > a0 ? 1 : 0;
  return `M ${x0.toFixed(3)} ${y0.toFixed(3)} A ${r} ${r} 0 ${large} ${sweep} ${x1.toFixed(3)} ${y1.toFixed(3)}`;
}
function handPath(a: number) {
  const [x0, y0] = polar(ARC_R - HAND, a);
  const [x1, y1] = polar(ARC_R, a);
  return `M ${x0.toFixed(3)} ${y0.toFixed(3)} L ${x1.toFixed(3)} ${y1.toFixed(3)}`;
}
const TRACK = arcPath(ARC_R, -MAX_ARC, MAX_ARC);

const EASE_IN_OUT = [0.77, 0, 0.175, 1] as const;
const ROUND = (v: number) => Math.round(v * 10000) / 10000;

interface Props {
  name: string;
  control: Control | undefined;
  accent: string;
  /** false when the section's power switch is off: Vital greys the arcs */
  active: boolean;
  locked: boolean;
  /** use this motion value for the displayed value, so a display (the oscillator's wave) can follow the knob */
  shown?: MotionValue<number>;
}

export function Knob({ name, control, accent, active, locked, shown: shared }: Props) {
  const id = useId();
  const pending = useStore((s) => s.pending[name]);
  const glowSeq = useStore((s) => s.glow.seq);
  const glowFrom = useStore((s) => s.glow.from[name]);
  const flashSeq = useStore((s) => (s.flash?.name === name ? s.flash.seq : 0));
  const setControl = useStore((s) => s.setControl);
  const setPending = useStore((s) => s.setPending);
  const shift = useShiftKey();

  const steps = STEPS[name];
  const bipolar = BIPOLAR.has(name);
  const target = pending ?? control?.value ?? 0;

  const [drag, setDrag] = useState<number | null>(null);
  const [hover, setHover] = useState(false);
  const dragRef = useRef<{ start: number; value: number } | null>(null);
  const keyTimer = useRef<number | undefined>(undefined);

  // Displayed value: jumps for manual, undo and new-patch changes; glides over ~600 ms when an agent round moved it.
  const own = useMotionValue(target);
  const shown = shared ?? own;
  const lastSeq = useRef(glowSeq);
  useEffect(() => {
    if (drag !== null) return;
    const agentMoved = glowFrom !== undefined && lastSeq.current !== glowSeq;
    lastSeq.current = glowSeq;
    if (agentMoved && !prefersReducedMotion()) {
      shown.jump(glowFrom);
      const a = animate(shown, target, { duration: 0.6, ease: EASE_IN_OUT });
      return () => a.stop();
    }
    shown.jump(target);
  }, [target, glowSeq, glowFrom, drag, shown]);

  const valueD = useTransform(shown, (v) => arcPath(ARC_R, bipolar ? 0 : -MAX_ARC, angleOf(v)));
  const handD = useTransform(shown, (v) => handPath(angleOf(v)));

  // Glow: a burst while the knob travels, then a steady halo until the next change.
  const [glowPhase, setGlowPhase] = useState<"none" | "burst" | "steady">("none");
  useEffect(() => {
    if (glowFrom === undefined) {
      setGlowPhase("none");
      return;
    }
    setGlowPhase("burst");
    const t = window.setTimeout(() => setGlowPhase("steady"), 900);
    return () => window.clearTimeout(t);
  }, [glowFrom, glowSeq]);

  // Flash when the change row in the chat is clicked.
  const ringRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!flashSeq || !ringRef.current) return;
    ringRef.current.animate(
      [
        { opacity: 0, transform: "scale(0.92)" },
        { opacity: 1, transform: "scale(1)", offset: 0.25 },
        { opacity: 0, transform: "scale(1.18)" },
      ],
      { duration: 650, iterations: 2, easing: "cubic-bezier(0.23, 1, 0.32, 1)" },
    );
  }, [flashSeq]);

  const commit = (value: number) => {
    if (!control) return;
    const v = ROUND(value);
    if (Math.abs(v - control.value) < 1e-6) {
      setPending(name, null);
      return;
    }
    void setControl(name, v);
  };

  const onPointerDown = () => {
    if (locked || !control) return;
    dragRef.current = { start: target, value: target };
    setDrag(target);
    const up = () => {
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
      const d = dragRef.current;
      dragRef.current = null;
      setDrag(null);
      if (d && Math.abs(d.value - d.start) > 1e-6) {
        setPending(name, d.value);
        commit(d.value);
      }
    };
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
  };

  const onChange = (v: number) => {
    if (!dragRef.current) return;
    dragRef.current.value = v;
    setDrag(v);
    shown.jump(v);
  };

  const roundFn = (v: number) => (steps ? Math.round(v * (steps.count - 1)) / (steps.count - 1) : v);
  const value = drag ?? target;
  const keyboard = useKnobKeyboardControls({
    valueRaw: value,
    valueMin: 0,
    valueMax: 1,
    step: steps ? 1 / (steps.count - 1) : 0.01,
    stepLarger: steps ? 4 / (steps.count - 1) : 0.1,
    onValueRawChange: (v) => {
      if (locked || !control) return;
      const r = roundFn(v);
      setPending(name, r);
      window.clearTimeout(keyTimer.current);
      keyTimer.current = window.setTimeout(() => commit(r), 350);
    },
  });

  const dragging = drag !== null;
  const color = active ? accent : "var(--color-arc-off)";
  const thick = hover || dragging ? THICK * 1.4 : THICK;
  const label = SHORT[name] ?? control?.label ?? name;
  const stepValue = steps ? Math.round(roundFn(value) * (steps.count - 1)) + steps.first : 0;
  const popupText = dragging || pending !== undefined
    ? steps
      ? `${stepValue}`
      : `${Math.round(value * 1000) / 10}%`
    : prettyDisplay(control?.display) || (steps ? `${stepValue}` : `${Math.round(value * 1000) / 10}%`);
  const glowing = glowPhase !== "none";

  return (
    <div
      className="relative flex min-w-0 flex-1 flex-col items-center"
      data-control={name}
      data-glow={glowPhase}
      onPointerEnter={() => setHover(true)}
      onPointerLeave={() => setHover(false)}
    >
      <Tooltip.Root open={(hover || dragging) && !!control} delayDuration={0}>
        <Tooltip.Trigger asChild>
          <KnobHeadless
            id={id}
            aria-label={control?.label ?? name}
            className={`relative mt-[2px] outline-none ${locked || !control ? "cursor-not-allowed" : "cursor-ns-resize"}`}
            style={{ width: BOX * S, height: BOX * S }}
            valueRaw={value}
            valueMin={0}
            valueMax={1}
            dragSensitivity={shift ? 0.0008 : 0.006}
            valueRawRoundFn={(v) => ROUND(roundFn(v))}
            valueRawDisplayFn={() => popupText}
            onValueRawChange={onChange}
            onPointerDown={onPointerDown}
            includeIntoTabOrder
            {...keyboard}
          >
            <div ref={ringRef} className="knob-ring pointer-events-none absolute inset-0 rounded-full opacity-0" style={{ boxShadow: `0 0 0 2px ${accent}, 0 0 14px ${accent}` }} />
            {steps ? (
              <StepBox value={stepValue} color={color} hover={hover || dragging} glowing={glowing} />
            ) : (
              <svg width={BOX * S} height={BOX * S} viewBox={`0 0 ${BOX} ${BOX}`} className="block overflow-visible">
                <circle cx={C} cy={C} r={BODY_R} fill="var(--color-knob-body)" />
                <circle cx={C} cy={C} r={BODY_R - 0.5} fill="none" stroke="var(--color-knob-border)" strokeWidth={1} />
                <path d={TRACK} fill="none" stroke="var(--shadow)" strokeWidth={THICK + 1} strokeLinecap="round" />
                <path d={TRACK} fill="none" stroke="var(--color-track)" strokeWidth={thick} strokeLinecap="round" className="knob-stroke" />
                <motion.path
                  d={valueD}
                  fill="none"
                  stroke={color}
                  strokeWidth={thick + 3.5}
                  strokeLinecap="round"
                  className="knob-glow"
                  style={{ filter: "blur(3px)" }}
                />
                <motion.path d={valueD} fill="none" stroke={color} strokeWidth={thick} strokeLinecap="round" className="knob-stroke" />
                {glowFrom !== undefined && <GhostTick value={glowFrom} />}
                <motion.path d={handD} fill="none" stroke="#ffffff" strokeWidth={thick} strokeLinecap="round" className="knob-stroke" />
              </svg>
            )}
          </KnobHeadless>
        </Tooltip.Trigger>
        <Tooltip.Portal>
          <Tooltip.Content
            side="top"
            sideOffset={2}
            className="vital-popup pointer-events-none z-50 rounded-[4px] border border-black bg-popup px-2 py-[3px] text-[13px] leading-[17px] text-text shadow-[0_4px_14px_rgba(0,0,0,0.45)]"
          >
            <span className="text-icon">{control?.label}</span>
            <span className="ml-2 tabular-nums text-white">{popupText}</span>
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>
      <div
        className="label-pill mt-[1px] flex w-full items-center justify-center px-1 uppercase leading-none tracking-[0.02em] whitespace-nowrap"
        style={{
          height: 18 * S,
          borderRadius: 9 * S,
          fontSize: 10.5 * S,
          ...(glowing ? { color: accent, background: `color-mix(in srgb, ${accent} 16%, var(--color-label))` } : {}),
        }}
      >
        {label}
      </div>
    </div>
  );
}

function GhostTick({ value }: { value: number }) {
  const a = angleOf(value);
  const [x0, y0] = polar(ARC_R + 2.2, a);
  const [x1, y1] = polar(BODY_R + 2.4, a);
  return (
    <path
      d={`M ${x0} ${y0} L ${x1} ${y1}`}
      stroke="rgba(255,255,255,0.6)"
      strokeWidth={1.6}
      strokeLinecap="round"
      className="ghost-tick"
    />
  );
}

/** Vital draws counts (unison voices) as a text component instead of a knob. */
function StepBox({ value, color, hover, glowing }: { value: number; color: string; hover: boolean; glowing: boolean }) {
  return (
    <div className="flex h-full w-full items-center justify-center">
      <div
        className="flex items-center justify-center bg-field tabular-nums text-white"
        style={{
          width: 44 * S,
          height: 30 * S,
          borderRadius: 6 * S,
          fontSize: 15 * S,
          boxShadow: glowing
            ? `inset 0 0 0 1.5px ${color}, 0 0 10px color-mix(in srgb, ${color} 45%, transparent)`
            : hover
              ? `inset 0 0 0 1px ${color}`
              : "inset 0 0 0 1px rgba(255,255,255,0.04)",
          transition: "box-shadow 300ms ease",
        }}
      >
        <span className="step-value">{value}</span>
      </div>
    </div>
  );
}
