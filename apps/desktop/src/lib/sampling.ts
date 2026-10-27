import type { SamplingParams, TokenOption } from "../types";

export interface Distributions {
  options: TokenOption[];
  entropy: number;
  kept: number;
  chosenProb: number | null;
}

/**
 * Recompute a display distribution from the model's raw top-k log-probs.
 * This never changes tokens already emitted; it answers "what would this
 * sampler have done with the same logits".
 */
export function resample(
  topk: TokenOption[],
  params: Pick<SamplingParams, "temperature" | "top_p" | "top_k" | "repetition_penalty">,
  recentTokens: string[],
  chosenToken?: string,
): Distributions {
  if (!topk.length) return { options: [], entropy: 0, kept: 0, chosenProb: null };
  const temp = Math.max(0.01, params.temperature);
  const counts = new Map<string, number>();
  for (const token of recentTokens.slice(-96)) {
    counts.set(token, (counts.get(token) ?? 0) + 1);
  }
  const penalty = Math.max(0.5, params.repetition_penalty);
  let items = topk.map((item) => {
    let logit = item.logprob;
    const seen = counts.get(item.token) ?? 0;
    if (seen > 0 && penalty !== 1) {
      logit = logit > 0 ? logit / (penalty * (1 + 0.08 * (seen - 1))) : logit * penalty * (1 + 0.08 * (seen - 1));
    }
    return { ...item, logit: logit / temp };
  });

  items.sort((a, b) => b.logit - a.logit);
  const k = Math.max(1, Math.min(params.top_k || items.length, items.length));
  items = items.slice(0, k);

  const max = items[0]?.logit ?? 0;
  let total = 0;
  const exp = items.map((item) => {
    const e = Math.exp(item.logit - max);
    total += e;
    return { item, e };
  });
  let options: TokenOption[] = exp.map(({ item, e }, i) => ({
    token: item.token,
    id: item.id,
    logprob: item.logprob,
    prob: e / total,
    rank: i + 1,
  }));

  const p = Math.max(0.01, Math.min(1, params.top_p || 1));
  let cumulative = 0;
  let cut = options.length;
  for (let i = 0; i < options.length; i += 1) {
    cumulative += options[i].prob;
    if (cumulative >= p) {
      cut = i + 1;
      break;
    }
  }
  options = options.slice(0, cut);
  const sum = options.reduce((acc, item) => acc + item.prob, 0) || 1;
  options = options.map((item) => ({ ...item, prob: item.prob / sum }));

  const entropy = -options.reduce((acc, item) => {
    const value = Math.max(item.prob, 1e-12);
    return acc + value * Math.log2(value);
  }, 0);

  let chosenProb: number | null = null;
  if (chosenToken) {
    const direct = options.find((item) => item.token === chosenToken);
    if (direct) chosenProb = direct.prob;
    else {
      const raw = topk.find((item) => item.token === chosenToken);
      if (raw) {
        const keptFind = options.find((item) => item.token === raw.token);
        chosenProb = keptFind ? keptFind.prob : 0;
      }
    }
  }

  return { options, entropy, kept: options.length, chosenProb };
}

export function formatProb(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (value === 0) return "0";
  if (value < 0.001) return value.toExponential(1);
  return value.toFixed(digits);
}

export function formatPct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}
