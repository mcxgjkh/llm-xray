import type { KVInfo, Probe } from "../types";

/**
 * Probe snapshots carry only the compact KV fields. Prefer the full estimate
 * from the backend; otherwise expand the compact form with conservative
 * model defaults so the map can still draw its frame.
 */
export function resolveKv(kv: KVInfo | null, probeKv: Probe["kv"] | null | undefined): KVInfo | null {
  if (kv) return kv;
  if (!probeKv) return null;
  return {
    ...probeKv,
    layers: 32,
    kv_heads: 8,
    head_dim: 128,
    dtype: "f16",
    estimated: true,
  };
}
