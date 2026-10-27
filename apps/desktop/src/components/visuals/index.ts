// Visual panels for LLM-Xray. Everything here is presentation-only: data arrives
// through props from the Lead's state layer, nothing fetches the API.
//
// Minimal integration example:
//
//   import {
//     AttentionHeatmap,
//     KVMemoryMap,
//     ParticleCloud3D,
//     ResidualBars,
//     SonificationToggle,
//     useSonification,
//   } from "./components/visuals";
//
//   const sound = useSonification({ enabled, entropy, chosenProb, step });
//   <AttentionHeatmap probe={probe} layer={layer} head={head}
//     history={probeHistory} maxColumns={32}
//     onLayerChange={setLayer} onHeadChange={setHead} />
//   <AttentionHeatmap probe={probe} layer={layer} head={head} compact /> // 16 columns
//   <ParticleCloud3D points={projection} activeStep={step} reducer={reducer} />
//   <KVMemoryMap kv={kv} contextTokens={contextLength} />
//   <ResidualBars residual={probe?.residual ?? []} selectedToken={activeToken} />
//   <NeuralTopology probe={probe} activeToken={activeToken} />
//   <SonificationToggle enabled={enabled} onToggle={setEnabled} entropy={entropy}
//     chosenProb={chosenProb} />
//
// Styles are imported by styles/global.css via styles/visuals.css.

export { AttentionHeatmap } from "./AttentionHeatmap";
export type { AttentionHeatmapProps, HeadTypeMeta, HeatRow } from "./AttentionHeatmap";
export { HEAD_TYPES } from "./AttentionHeatmap";

export { ParticleCloud3D } from "./ParticleCloud3D";
export type { ParticleCloud3DProps } from "./ParticleCloud3D";

export { NeuralTopology } from "./NeuralTopology";
export type { NeuralTopologyProps } from "./NeuralTopology";

export { KVMemoryMap } from "./KVMemoryMap";
export type { KVMemoryMapProps, ColumnMode as KVColumnMode } from "./KVMemoryMap";

export { ResidualBars } from "./ResidualBars";
export type { ResidualBarsProps, ResidualRow } from "./ResidualBars";

export { SonificationToggle } from "./SonificationToggle";
export type { SonificationToggleProps } from "./SonificationToggle";

export {
  useSonification,
  unlockAudio,
  isAudioSupported,
  isAudioUnlocked,
  disposeSharedAudio,
} from "./Sonification";
export type { SonificationInput, SonificationState } from "./Sonification";

export type { VisualPalette } from "./palette";
