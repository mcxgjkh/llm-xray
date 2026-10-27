export function Sparkline({
  values,
  width = 108,
  height = 28,
  min = 0,
  max,
  color = "var(--accent)",
  label,
}: {
  values: (number | null)[];
  width?: number;
  height?: number;
  min?: number;
  max?: number;
  color?: string;
  label?: string;
}) {
  const clean = values.filter((value): value is number => value !== null && Number.isFinite(value));
  if (clean.length < 2) {
    return <div className="sparkline-empty" style={{ width, height }} aria-label={label ?? "暂无趋势"} />;
  }
  const lo = min;
  const hi = Math.max(max ?? Math.max(...clean), lo + 0.0001);
  const span = hi - lo;
  const points = values.map((value, index) => {
    const x = (index / (values.length - 1)) * width;
    const y = height - 2 - (((value ?? lo) - lo) / span) * (height - 4);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const last = points.at(-1)?.split(",") ?? ["0", "0"];
  return (
    <svg
      className="sparkline"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={label ?? "趋势"}
    >
      <path d={`M${points.join(" L")}`} fill="none" stroke={color} strokeWidth="1.4" />
      <circle cx={last[0]} cy={last[1]} r="1.8" fill={color} />
    </svg>
  );
}
