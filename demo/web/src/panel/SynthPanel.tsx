import { useMotionValue } from "motion/react";
import { useStore, controlOf } from "../state/store";
import { Knob } from "./Knob";
import { Section } from "./Section";
import { EnvelopeDisplay } from "./Envelope";
import { FilterDisplay } from "./FilterDisplay";
import { OscWave } from "./OscWave";
import { ROWS, SKINS, type SectionDef } from "./layout";

const DISPLAY_H = 108;
/** an oscillator's wave view: as tall as a knob with its label, and as wide as this many knobs; at the narrowest
 * page (1280 px) it gives way to the knobs, down to about 75 px */
const WAVE_H = 82;
const WAVE_SLOTS = 2;

export function SynthPanel() {
  return (
    <div className="flex flex-col gap-[6px]">
      {ROWS.map((row, i) => (
        <div key={i} className="flex gap-[6px]">
          {row.map((def) => (
            <PanelSection key={def.group} def={def} />
          ))}
        </div>
      ))}
    </div>
  );
}

function PanelSection({ def }: { def: SectionDef }) {
  const state = useStore((s) => s.state);
  const locked = useStore((s) => s.active !== null || s.conn !== "ok");
  const pendingPower = useStore((s) => (def.power ? s.pending[def.power] : undefined));
  const skin = SKINS[def.group];
  const powerControl = def.power ? controlOf(state, def.power) : undefined;
  const active = def.power ? (pendingPower ?? powerControl?.value ?? 0) > 0.5 : true;
  const value = (n: string) => controlOf(state, n)?.value ?? 0;
  const { envelope: env, filter, wave } = def;
  // The FRAME knob's displayed value, shared with the wave view so the wave follows drags and glides.
  const frameShown = useMotionValue(wave ? value(wave.frame) : 0);
  // Width follows the number of knobs; single-knob effects get room for their title and power button, a
  // sideways heading strip takes some width of its own, and an oscillator's wave view that of WAVE_SLOTS knobs.
  const grow = def.knobs.length + (def.knobs.length === 1 ? 0.5 : 0) + (def.sideways ? 0.4 : 0) + (wave ? WAVE_SLOTS : 0);
  const knobs = def.knobs.map((name) => (
    <Knob
      key={name}
      name={name}
      control={controlOf(state, name)}
      accent={skin.accent}
      active={active}
      locked={locked}
      shown={name === wave?.frame ? frameShown : undefined}
    />
  ));

  return (
    <Section
      title={def.title}
      accent={skin.accent}
      heading={skin.heading}
      sideways={def.sideways}
      power={def.power ? { control: powerControl, locked } : undefined}
      style={{ flexGrow: grow, flexBasis: 0 }}
    >
      {env && (
        <div className="px-[6px] pt-[6px]">
          <EnvelopeDisplay
            attack={value(env.attack)}
            decay={value(env.decay)}
            sustain={value(env.sustain)}
            release={env.release ? value(env.release) : null}
            names={[env.attack, env.decay, env.sustain, ...(env.release ? [env.release] : [])]}
            height={DISPLAY_H}
          />
        </div>
      )}
      {filter && (
        <div className="px-[6px] pt-[6px]">
          <FilterDisplay
            cutoff={value(filter.cutoff)}
            peak={value(filter.peak)}
            resonance={value(filter.resonance)}
            mix={value(filter.mix)}
            on={active}
            names={[filter.cutoff, filter.peak, filter.resonance, filter.mix, ...(def.power ? [def.power] : [])]}
            height={DISPLAY_H}
          />
        </div>
      )}
      {wave ? (
        <div className="flex items-center gap-[6px] p-[6px]">
          <div className="min-w-0" style={{ flex: `${WAVE_SLOTS} 1 0` }}>
            <OscWave
              table={state?.current.wavetables?.[wave.table]}
              frame={frameShown}
              accent={skin.accent}
              on={active}
              height={WAVE_H}
              testId={`osc-wave-${wave.table.replace("osc_", "")}`}
            />
          </div>
          <div className="flex gap-[6px]" style={{ flex: `${def.knobs.length} 1 0` }}>
            {knobs}
          </div>
        </div>
      ) : (
        <div className="flex gap-[6px] px-[6px] pt-[6px] pb-[6px]">{knobs}</div>
      )}
    </Section>
  );
}
