import type { ReactNode } from "react";
import { Switch } from "radix-ui";
import type { Control } from "../api/types";
import { useStore } from "../state/store";
import { S } from "./layout";

// SynthSection: body #4c4f52 with 4 px rounding and a hairline border, a heading band (Body Heading Background)
// as wide as Title Width (32 px), heading text in Lato Light 14, and the power button at the start of the heading.
// Vital's default heading is sideways: a strip on the left with the title reading upwards and the power button on
// top; filters use a band across the top.

const TITLE = 32 * S;
const HEADING_FONT = 15.5;

interface Props {
  title: string;
  accent: string;
  heading: string;
  sideways?: boolean;
  power?: { control: Control | undefined; locked: boolean };
  className?: string;
  style?: React.CSSProperties;
  children: ReactNode;
}

export function Section({ title, accent, heading, sideways, power, className = "", style, children }: Props) {
  const body = `relative flex min-w-0 rounded-[4px] bg-body shadow-[0_0_0_1px_var(--hairline),0_1px_3px_rgba(0,0,0,0.25)] ${className}`;
  if (sideways) {
    return (
      <section className={`${body} flex-row`} style={style} aria-label={title}>
        <header className="relative flex shrink-0 flex-col items-center rounded-l-[4px]" style={{ background: heading, width: TITLE }}>
          {power && <PowerButton control={power.control} accent={accent} locked={power.locked} />}
          <h2
            className="pointer-events-none flex min-h-0 flex-1 rotate-180 items-center justify-center font-light tracking-[0.08em] whitespace-nowrap text-heading-text [writing-mode:vertical-rl]"
            style={{ fontSize: HEADING_FONT, paddingTop: 4, paddingBottom: 4 }}
          >
            {title}
          </h2>
        </header>
        <div className="flex min-w-0 flex-1 flex-col justify-center">{children}</div>
      </section>
    );
  }
  return (
    <section className={`${body} flex-col`} style={style} aria-label={title}>
      <header className="relative flex shrink-0 items-center rounded-t-[4px]" style={{ background: heading, height: TITLE * 0.94 }}>
        {power && <PowerButton control={power.control} accent={accent} locked={power.locked} />}
        <h2
          className="pointer-events-none absolute inset-y-0 flex items-center justify-center font-light tracking-[0.06em] text-heading-text"
          style={{ left: power ? 22 * S : 0, right: 0, fontSize: HEADING_FONT }}
        >
          {title}
        </h2>
      </header>
      {children}
    </section>
  );
}

function PowerButton({ control, accent, locked }: { control: Control | undefined; accent: string; locked: boolean }) {
  const setControl = useStore((s) => s.setControl);
  const pending = useStore((s) => (control ? s.pending[control.name] : undefined));
  const glowed = useStore((s) => (control ? s.glow.from[control.name] !== undefined : false));
  const on = (pending ?? control?.value ?? 0) > 0.5;
  return (
    <Switch.Root
      checked={on}
      disabled={locked || !control}
      onCheckedChange={(checked) => control && void setControl(control.name, checked ? 1 : 0)}
      aria-label={control?.label ?? "section switch"}
      data-control={control?.name}
      className="power group relative z-10 flex shrink-0 items-center justify-center rounded-full outline-none disabled:cursor-not-allowed"
      style={{ width: 30 * S, height: 30 * S }}
    >
      {glowed && (
        <span
          className="power-glow absolute rounded-full"
          style={{ width: 22 * S, height: 22 * S, boxShadow: `0 0 0 1.5px ${accent}, 0 0 12px ${accent}` }}
        />
      )}
      <span
        className="absolute rounded-full bg-white/0 transition-[background-color] duration-150 group-hover:bg-[var(--lighten)] group-active:bg-[var(--overlay)] group-disabled:bg-white/0"
        style={{ width: 22 * S, height: 22 * S }}
      />
      <Switch.Thumb
        className="relative block rounded-full transition-[background-color,box-shadow] duration-200"
        style={{
          width: 14 * S,
          height: 14 * S,
          background: on ? accent : "var(--color-power-off)",
          boxShadow: on ? `0 0 6px color-mix(in srgb, ${accent} 45%, transparent)` : "none",
        }}
      />
    </Switch.Root>
  );
}
