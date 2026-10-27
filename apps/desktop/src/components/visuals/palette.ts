// Canvas / WebGL cannot consume CSS custom properties directly, so the visual
// panels read the token values from computed style once per draw and fall back
// to the checked-in token snapshot when the element is detached.

export interface VisualPalette {
  ground: string;
  surface: string;
  surface2: string;
  raised: string;
  ink: string;
  inkStrong: string;
  muted: string;
  faint: string;
  line: string;
  lineStrong: string;
  accent: string;
  accentSoft: string;
  accentDeep: string;
  attention: string;
  mlp: string;
  positive: string;
  negative: string;
  warning: string;
  reasoning: string;
}

const FALLBACK: VisualPalette = {
  ground: "#17130f",
  surface: "#1c1712",
  surface2: "#211a14",
  raised: "#271f17",
  ink: "#f2e7d5",
  inkStrong: "#fff8ec",
  muted: "#a89780",
  faint: "#7c6b59",
  line: "#33291f",
  lineStrong: "#49392a",
  accent: "#d08a4e",
  accentSoft: "#e8b57a",
  accentDeep: "#9c5a2e",
  attention: "#d08a4e",
  mlp: "#9aa36b",
  positive: "#8faa76",
  negative: "#c96a50",
  warning: "#d8b36a",
  reasoning: "#b9a98e",
};

const VAR_NAMES: Record<keyof VisualPalette, string> = {
  ground: "--ground",
  surface: "--surface",
  surface2: "--surface-2",
  raised: "--raised",
  ink: "--ink",
  inkStrong: "--ink-strong",
  muted: "--muted",
  faint: "--faint",
  line: "--line",
  lineStrong: "--line-strong",
  accent: "--accent",
  accentSoft: "--accent-soft",
  accentDeep: "--accent-deep",
  attention: "--attention",
  mlp: "--mlp",
  positive: "--positive",
  negative: "--negative",
  warning: "--warning",
  reasoning: "--reasoning",
};

export function readPalette(element: Element | null): VisualPalette {
  if (!element || typeof window === "undefined") return FALLBACK;
  const styles = window.getComputedStyle(element);
  const out = { ...FALLBACK };
  for (const key of Object.keys(VAR_NAMES) as (keyof VisualPalette)[]) {
    const value = styles.getPropertyValue(VAR_NAMES[key]).trim();
    if (value) out[key] = value;
  }
  return out;
}

/** Convert a #rgb / #rrggbb token value to an rgba() string for canvas / WebGL. */
export function withAlpha(color: string, alpha: number): string {
  const parsed = parseHex(color);
  if (!parsed) return color;
  const [r, g, b] = parsed;
  return `rgba(${r}, ${g}, ${b}, ${clampAlpha(alpha)})`;
}

/** Blend two hex colours; useful for flat fills that still track the palette. */
export function mixHex(from: string, to: string, t: number): string {
  const a = parseHex(from);
  const b = parseHex(to);
  if (!a || !b) return from;
  const k = clampAlpha(t);
  const channel = (i: number) => Math.round(a[i] + (b[i] - a[i]) * k);
  return `rgb(${channel(0)}, ${channel(1)}, ${channel(2)})`;
}

export function parseHex(color: string): [number, number, number] | null {
  const hex = color.trim();
  if (!hex.startsWith("#")) return null;
  const body = hex.slice(1);
  if (body.length === 3) {
    const r = Number.parseInt(body[0] + body[0], 16);
    const g = Number.parseInt(body[1] + body[1], 16);
    const b = Number.parseInt(body[2] + body[2], 16);
    return Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b) ? null : [r, g, b];
  }
  if (body.length === 6) {
    const r = Number.parseInt(body.slice(0, 2), 16);
    const g = Number.parseInt(body.slice(2, 4), 16);
    const b = Number.parseInt(body.slice(4, 6), 16);
    return Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b) ? null : [r, g, b];
  }
  return null;
}

function clampAlpha(value: number): number {
  if (!Number.isFinite(value)) return 1;
  return Math.min(1, Math.max(0, value));
}
