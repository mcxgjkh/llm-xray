import { describe, expect, it } from "vitest";
import { resample } from "./sampling";
import type { SamplingParams, TokenOption } from "../types";

function option(token: string, logprob: number): TokenOption {
  return { token, logprob, prob: Math.exp(logprob), rank: 1 };
}

const topk = [option("甲", -0.1), option("乙", -0.3), option("丙", -1.2), option("丁", -2.0)];

const base: SamplingParams = {
  temperature: 1,
  top_p: 1,
  top_k: 40,
  repetition_penalty: 1,
  max_tokens: 64,
  seed: null,
};

describe("resample", () => {
  it("normalizes probabilities", () => {
    const result = resample(topk, base, [], "甲");
    const sum = result.options.reduce((acc, item) => acc + item.prob, 0);
    expect(sum).toBeCloseTo(1, 6);
    expect(result.chosenProb).toBeCloseTo(result.options[0].prob, 6);
  });

  it("sharpens with low temperature and flattens with high temperature", () => {
    const cold = resample(topk, { ...base, temperature: 0.1 }, []);
    const hot = resample(topk, { ...base, temperature: 2 }, []);
    expect(cold.options[0].prob).toBeGreaterThan(hot.options[0].prob);
    expect(cold.entropy).toBeLessThan(hot.entropy);
  });

  it("truncates to top-k", () => {
    const result = resample(topk, { ...base, top_k: 2 }, []);
    expect(result.options).toHaveLength(2);
    expect(result.kept).toBe(2);
  });

  it("truncates to top-p nucleus", () => {
    const result = resample(topk, { ...base, top_p: 0.55 }, []);
    expect(result.options.length).toBeLessThan(topk.length);
    expect(result.options.reduce((acc, item) => acc + item.prob, 0)).toBeCloseTo(1, 6);
  });

  it("penalizes recently repeated tokens", () => {
    const clean = resample(topk, base, [], "甲");
    const repeated = resample(topk, { ...base, repetition_penalty: 1.4 }, ["甲", "甲"], "甲");
    const firstClean = clean.options.find((item) => item.token === "甲")!.prob;
    const firstRepeated = repeated.options.find((item) => item.token === "甲")!.prob;
    expect(firstRepeated).toBeLessThan(firstClean);
  });

  it("returns an empty distribution without candidates", () => {
    const result = resample([], base, []);
    expect(result.options).toEqual([]);
    expect(result.chosenProb).toBeNull();
  });
});
