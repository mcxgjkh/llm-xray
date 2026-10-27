import type { SVGProps } from "react";

const paths: Record<string, string[]> = {
  observatory: ["M3 18h18", "M12 3v12", "M7.5 7.5c2.4-2.4 6.6-2.4 9 0", "M4.5 12c4-4 11-4 15 0", "M9 18l3 3 3-3"],
  attention: [
    "M4 4h16v16H4z",
    "M4 9h16",
    "M4 14.5h16",
    "M9 4v16",
    "M14.5 4v16",
  ],
  vectors: ["M12 3l7 4v10l-7 4-7-4V7z", "M5 7l7 4 7-4", "M12 11v10"],
  memory: ["M5 4h14v16H5z", "M8 4v16", "M11 8h5", "M11 12h5", "M11 16h3"],
  replay: ["M4 12a8 8 0 1 0 2.3-5.6", "M4 4v5h5", "M10 9l5 3-5 3z"],
  settings: [
    "M12 8.5A3.5 3.5 0 1 0 12 15.5 3.5 3.5 0 0 0 12 8.5z",
    "M19 12a7 7 0 0 0-.1-1l2-1.5-2-3.4-2.3 1a7 7 0 0 0-1.7-1L14.6 3h-4l-.3 2.1a7 7 0 0 0-1.7 1l-2.3-1-2 3.4L6.3 11a7 7 0 0 0 0 2l-2 1.5 2 3.4 2.3-1a7 7 0 0 0 1.7 1l.3 2.1h4l.3-2.1a7 7 0 0 0 1.7-1l2.3 1 2-3.4-2-1.5c.1-.3.1-.7.1-1z",
  ],
  docs: ["M5 3h9l5 5v13H5z", "M14 3v5h5", "M8 12h8", "M8 16h6"],
  send: ["M3 11l18-8-8 18-2-8z", "M11 13l10-10"],
  stop: ["M6 6h12v12H6z"],
  play: ["M8 5l11 7-11 7z"],
  pause: ["M8 5v14", "M16 5v14"],
  download: ["M12 3v12", "M7 10l5 5 5-5", "M4 20h16"],
  plus: ["M12 5v14", "M5 12h14"],
  refresh: ["M20 8a8 8 0 1 0 1 7", "M20 3v5h-5"],
  sound: ["M4 9v6h4l5 4V5L8 9z", "M16 9c1.5 1.7 1.5 4.3 0 6", "M18.5 6.5c2.8 3.2 2.8 7.8 0 11"],
  soundOff: ["M4 9v6h4l5 4V5L8 9z", "M17 9l5 6", "M22 9l-5 6"],
  crt: ["M3 5h18v12H3z", "M8 21h8", "M12 17v4", "M6 8h8"],
  collapse: ["M15 5l-7 7 7 7"],
  expand: ["M9 5l7 7-7 7"],
  close: ["M6 6l12 12", "M18 6L6 18"],
  chevronDown: ["M6 9l6 6 6-6"],
  chevronRight: ["M9 6l6 6-6 6"],
  search: ["M10.5 4a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13z", "M15.5 15.5L21 21"],
  spark: ["M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"],
  layers: ["M12 3l9 5-9 5-9-5z", "M3 13l9 5 9-5", "M3 17l9 5 9-5"],
  token: ["M4 7h16", "M4 12h10", "M4 17h13", "M18 14l3 3-3 3"],
  plug: ["M9 3v5", "M15 3v5", "M6 8h12v3a6 6 0 0 1-12 0z", "M12 17v4"],
  info: ["M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z", "M12 11v5", "M12 8h.01"],
  warning: ["M12 3l10 18H2z", "M12 9v5", "M12 17h.01"],
};

export type IconName = keyof typeof paths;

export function Icon({
  name,
  size = 16,
  ...props
}: { name: IconName | string; size?: number } & SVGProps<SVGSVGElement>) {
  const lines = paths[name] ?? paths.info;
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...props}
    >
      {lines.map((d) => (
        <path key={d} d={d} />
      ))}
    </svg>
  );
}

export function BrandMark({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true">
      <circle cx="16" cy="16" r="12.5" stroke="var(--line-strong)" strokeWidth="1" />
      <path d="M9 9l14 14M23 9L9 23" stroke="var(--accent)" strokeWidth="1.4" strokeLinecap="round" />
      <circle cx="16" cy="16" r="3.1" stroke="var(--accent-soft)" strokeWidth="1" />
      <circle cx="16" cy="16" r="1.1" fill="var(--accent-soft)" />
      <path d="M16 1.5v3M16 27.5v3M1.5 16h3M27.5 16h3" stroke="var(--muted)" strokeWidth="1" />
    </svg>
  );
}
