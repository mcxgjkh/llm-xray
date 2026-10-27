import type { ModelInfo, RuntimeMode } from "../types";

export interface MemoryEstimate {
  totalGb: number;
  weightsGb: number;
  kvGb: number;
  context: number;
  device: "GPU" | "CPU";
  detail: string;
}

function modelScale(idOrPath: string): "8b" | "1.5b" | "0.5b" | "unknown" {
  const value = idOrPath.toLowerCase();
  if (value.includes("8b")) return "8b";
  if (value.includes("1.5b") || value.includes("1_5b")) return "1.5b";
  if (value.includes("0.5b") || value.includes("0_5b")) return "0.5b";
  return "unknown";
}

/**
 * Conservative order-of-magnitude estimate for the KV cache plus weights.
 * LM Studio memory is managed by its own runtime; the useful number while
 * moving max-tokens is the KV growth. Native mode also pays for the weights.
 */
export function estimateMemory(
  mode: RuntimeMode,
  model: ModelInfo | undefined,
  nativeModel: string,
  maxTokens: number,
  device: string,
): MemoryEstimate {
  const isNative = mode === "native";
  const scale = modelScale(isNative ? nativeModel : model?.id ?? "");
  const onGpu = isNative ? device !== "cpu" : true;

  let weightsGb = 0.8;
  let kvBytesPerToken = 128 * 1024;
  let contextCap = 131072;
  if (scale === "8b") {
    weightsGb = isNative ? 32 : model?.quantization?.toUpperCase().includes("Q3") ? 4.6 : 5.2;
    kvBytesPerToken = 144 * 1024;
  } else if (scale === "1.5b") {
    weightsGb = isNative ? 6.2 : 1.8;
    kvBytesPerToken = 28 * 1024;
  } else if (scale === "0.5b") {
    weightsGb = isNative ? 2.1 : 1.1;
    kvBytesPerToken = 24 * 1024;
  } else if (!isNative) {
    weightsGb = 4.8;
  }

  const modelCap = isNative ? 32768 : model?.max_context_length ?? contextCap;
  contextCap = Math.max(512, modelCap);
  const context = Math.min(contextCap, 512 + Math.max(256, maxTokens));
  const kvGb = (context * kvBytesPerToken) / 1024 ** 3;
  const runtimeGb = isNative ? (onGpu ? 0.7 : 0.9) : 0.7;
  const totalGb = weightsGb + kvGb + runtimeGb;
  const deviceLabel = isNative ? (onGpu ? "GPU" : "CPU") : "GPU/VRAM";
  return {
    totalGb,
    weightsGb,
    kvGb,
    context,
    device: isNative ? (onGpu ? "GPU" : "CPU") : "GPU",
    detail: `估算：权重 ${weightsGb.toFixed(1)} GB + KV ${kvGb.toFixed(2)} GB（${context} tokens）· ${deviceLabel}`,
  };
}
