// Small formatting helpers shared by the visual panels.
// Numbers in the UI are rendered with `.num` / tabular figures; these helpers
// only decide the text, never the emphasis.

export function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

export function clamp01(value: number): number {
  return clamp(value, 0, 1);
}

export function fmtInt(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return Math.round(value).toLocaleString("en-US");
}

export function fmtFixed(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

export function fmtWeight(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  if (value === 0) return "0.000";
  if (value < 0.001) return value.toExponential(1);
  return value.toFixed(3);
}

export function fmtSigned(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const text = Math.abs(value).toFixed(digits);
  return value > 0 ? `+${text}` : value < 0 ? `-${text}` : `0.${"0".repeat(digits)}`;
}

export function fmtPercent(part: number | null | undefined, total: number | null | undefined, digits = 1): string {
  if (part == null || total == null || !Number.isFinite(part) || !Number.isFinite(total) || total <= 0) {
    return "—";
  }
  return `${((part / total) * 100).toFixed(digits)}%`;
}

export function fmtBytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 100 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

/** Shorten a display label without inventing an ellipsis character cluster. */
export function shorten(text: string, max: number): string {
  if (max <= 0) return "";
  const chars = Array.from(text);
  if (chars.length <= max) return text;
  return `${chars.slice(0, max).join("")}…`;
}

/** A token label is usually one or two pieces; collapse whitespace for display. */
export function tokenLabel(token: string | undefined, max = 6): string {
  if (token == null) return "—";
  const cleaned = token.replace(/\s+/g, " ").trim();
  if (cleaned.length === 0) return "∅";
  return shorten(cleaned, max);
}

export function range(start: number, end: number): number[] {
  const out: number[] = [];
  for (let i = start; i < end; i += 1) out.push(i);
  return out;
}
