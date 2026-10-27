// Opt-in Web Audio sonification for the entropy stream.
//
// Design constraints from docs/SPEC.md and the visual brief:
// - silent by default; the AudioContext is only resumed from a user gesture.
// - never plays when the document is hidden or before the user enables it.
// - short tones, low volume, no loop, no autoplay.
// - low entropy -> crisp high, short tone; high entropy -> low filtered noise + chord.

import { useCallback, useEffect, useRef, useState } from "react";
import { clamp01 } from "./format";

export interface SonificationInput {
  enabled: boolean;
  entropy: number | null;
  chosenProb?: number | null;
  step: number | null;
}

export interface SonificationState {
  supported: boolean;
  ready: boolean;
  active: boolean;
  unlock: () => Promise<boolean>;
}

type AudioContextCtor = typeof AudioContext;

let sharedContext: AudioContext | null = null;
let sharedMaster: GainNode | null = null;
let sharedNoise: AudioBuffer | null = null;
let mountedHooks = 0;

function resolveCtor(): AudioContextCtor | null {
  if (typeof window === "undefined") return null;
  const candidate = window as Window & { webkitAudioContext?: AudioContextCtor };
  return window.AudioContext ?? candidate.webkitAudioContext ?? null;
}

export function isAudioSupported(): boolean {
  return resolveCtor() !== null;
}

function buildNoiseBuffer(context: AudioContext): AudioBuffer {
  const length = Math.max(1, Math.floor(context.sampleRate * 0.5));
  const buffer = context.createBuffer(1, length, context.sampleRate);
  const data = buffer.getChannelData(0);
  for (let index = 0; index < data.length; index += 1) {
    data[index] = Math.random() * 2 - 1;
  }
  return buffer;
}

function ensureContext(): AudioContext | null {
  const Ctor = resolveCtor();
  if (!Ctor) return null;
  if (!sharedContext || sharedContext.state === "closed") {
    sharedContext = new Ctor();
    sharedMaster = sharedContext.createGain();
    sharedMaster.gain.value = 0.8;
    sharedMaster.connect(sharedContext.destination);
    sharedNoise = buildNoiseBuffer(sharedContext);
  }
  return sharedContext;
}

/**
 * Must be invoked directly from a pointer/keyboard gesture. Returns the running
 * context, or null when the environment has no Web Audio or resume was blocked.
 */
export async function unlockAudio(): Promise<AudioContext | null> {
  const context = ensureContext();
  if (!context) return null;
  if (context.state === "suspended") {
    try {
      await context.resume();
    } catch {
      return null;
    }
  }
  return context.state === "running" ? context : null;
}

export function isAudioUnlocked(): boolean {
  return sharedContext?.state === "running";
}

function scheduleDisconnect(seconds: number, nodes: AudioNode[], sources: AudioScheduledSourceNode[]): void {
  const delay = Math.max(0, Math.ceil(seconds * 1000));
  window.setTimeout(() => {
    for (const source of sources) {
      try {
        source.disconnect();
      } catch {
        // Already disconnected when the context was closed.
      }
    }
    for (const node of nodes) {
      try {
        node.disconnect();
      } catch {
        // Already disconnected when the context was closed.
      }
    }
  }, delay);
}

function playStep(context: AudioContext, entropy: number, chosenProb: number): void {
  if (typeof document !== "undefined" && document.visibilityState !== "visible") return;
  if (context.state !== "running") return;
  const master = sharedMaster ?? context.destination;
  // Runtime entropy is nats; 4 nats is already a very flat distribution here.
  const normalizedEntropy = clamp01(entropy / 4);
  const probability = clamp01(chosenProb);
  const now = context.currentTime;
  const duration = normalizedEntropy > 0.55 ? 0.22 : 0.11;

  const envelope = context.createGain();
  const peak = 0.03 + 0.04 * (1 - normalizedEntropy) * (0.4 + 0.6 * probability);
  envelope.gain.setValueAtTime(0.0001, now);
  envelope.gain.exponentialRampToValueAtTime(Math.max(0.0002, peak), now + 0.014);
  envelope.gain.exponentialRampToValueAtTime(0.0001, now + duration);
  envelope.connect(master);

  const sources: AudioScheduledSourceNode[] = [];
  const nodes: AudioNode[] = [envelope];

  if (normalizedEntropy < 0.5) {
    // Confident distribution: a clean, high, short ping.
    const oscillator = context.createOscillator();
    oscillator.type = "sine";
    oscillator.frequency.setValueAtTime(620 + (1 - normalizedEntropy) * 880 + probability * 160, now);
    oscillator.connect(envelope);
    oscillator.start(now);
    oscillator.stop(now + duration + 0.02);
    sources.push(oscillator);
  } else {
    // Uncertain distribution: a low, breathy chord with filtered noise.
    const chord = context.createGain();
    chord.gain.value = 0.55;
    chord.connect(envelope);
    nodes.push(chord);
    const base = 146.83 + (1 - normalizedEntropy) * 40;
    [base, base * 1.335, base * 1.682].forEach((frequency, index) => {
      const oscillator = context.createOscillator();
      oscillator.type = "triangle";
      oscillator.frequency.setValueAtTime(frequency, now);
      oscillator.detune.setValueAtTime(index % 2 === 0 ? -4 : 4, now);
      oscillator.connect(chord);
      oscillator.start(now);
      oscillator.stop(now + duration + 0.02);
      sources.push(oscillator);
    });
    const filter = context.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.setValueAtTime(2200 - normalizedEntropy * 1500, now);
    filter.Q.value = 0.7;
    filter.connect(envelope);
    nodes.push(filter);
    if (sharedNoise) {
      const noise = context.createBufferSource();
      noise.buffer = sharedNoise;
      noise.loop = true;
      noise.connect(filter);
      noise.start(now);
      noise.stop(now + duration + 0.02);
      sources.push(noise);
    }
  }

  scheduleDisconnect(duration + 0.08, nodes, sources);
}

/**
 * React hook: call `unlock` from a click handler, then flip `enabled`.
 * A step only sounds when it actually changes; enabling does not replay history.
 */
export function useSonification({
  enabled,
  entropy,
  chosenProb = null,
  step,
}: SonificationInput): SonificationState {
  const [supported, setSupported] = useState(false);
  const [ready, setReady] = useState(false);
  const lastStep = useRef<number | null>(null);
  const lastPlayedAt = useRef(0);
  const wasEnabled = useRef(false);

  useEffect(() => {
    setSupported(resolveCtor() !== null);
    mountedHooks += 1;
    return () => {
      mountedHooks = Math.max(0, mountedHooks - 1);
      if (mountedHooks === 0 && sharedContext) {
        const context = sharedContext;
        sharedContext = null;
        sharedMaster = null;
        sharedNoise = null;
        void context.close().catch(() => undefined);
      }
    };
  }, []);

  const unlock = useCallback(async () => {
    const context = await unlockAudio();
    const ok = context != null;
    setReady(ok && enabled);
    return ok;
  }, [enabled]);

  useEffect(() => {
    if (!enabled) {
      wasEnabled.current = false;
      lastStep.current = step;
      setReady(false);
      return;
    }
    const context = sharedContext;
    if (!context || context.state !== "running") {
      setReady(false);
      return;
    }
    setReady(true);
    if (!wasEnabled.current) {
      // Enabling mid-stream should not fire a retroactive tone for the current step.
      wasEnabled.current = true;
      lastStep.current = step;
      return;
    }
    if (step == null || step === lastStep.current) {
      lastStep.current = step;
      return;
    }
    const now = typeof performance !== "undefined" ? performance.now() : Date.now();
    if (now - lastPlayedAt.current < 90) return;
    lastStep.current = step;
    lastPlayedAt.current = now;
    playStep(context, entropy ?? 0, chosenProb ?? 0.5);
  }, [enabled, entropy, chosenProb, step]);

  return { supported, ready, active: enabled && ready, unlock };
}

/** Test / teardown helper; the hook closes the shared context when it unmounts. */
export function disposeSharedAudio(): void {
  if (!sharedContext) return;
  const context = sharedContext;
  sharedContext = null;
  sharedMaster = null;
  sharedNoise = null;
  void context.close().catch(() => undefined);
}
