import { useEffect, useState } from "react";
import { fmtFixed } from "./format";
import { isAudioSupported, unlockAudio } from "./Sonification";

export interface SonificationToggleProps {
  enabled: boolean;
  onToggle: (next: boolean) => void;
  entropy: number | null;
  chosenProb: number | null;
}

/**
 * Compact toggle for the entropy sonification. The click handler unlocks the
 * AudioContext synchronously inside the gesture before asking the parent to
 * enable the stream; nothing is ever played before that gesture.
 */
export function SonificationToggle({ enabled, onToggle, entropy, chosenProb }: SonificationToggleProps) {
  const [supported, setSupported] = useState(true);

  useEffect(() => {
    setSupported(isAudioSupported());
  }, []);

  const handleClick = () => {
    if (!supported) return;
    if (enabled) {
      onToggle(false);
      return;
    }
    // Stay inside the gesture: browsers only resume audio from real input.
    // Enable the stream after resume settles so the hook sees a running context.
    void unlockAudio()
      .then(() => onToggle(true))
      .catch(() => onToggle(true));
  };

  return (
    <button
      type="button"
      className={`lx-sonify${enabled ? " is-on" : ""}`}
      aria-pressed={enabled}
      disabled={!supported}
      onClick={handleClick}
      title={
        supported
          ? "声化默认关闭；首次开启需要一次点击以解锁音频，不自动播放"
          : "当前环境不支持 Web Audio"
      }
    >
      <svg
        className="lx-sonify__icon"
        viewBox="0 0 16 16"
        width="14"
        height="14"
        aria-hidden="true"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        {enabled ? (
          <>
            <path d="M2.5 6.2h2.2L7.8 3.6v8.8L4.7 9.8H2.5Z" />
            <path d="M10.2 5.6a3.4 3.4 0 0 1 0 4.8" />
            <path d="M12.1 3.9a6 6 0 0 1 0 8.2" />
          </>
        ) : (
          <>
            <path d="M2.5 6.2h2.2L7.8 3.6v8.8L4.7 9.8H2.5Z" />
            <path d="m11 6 3 4" />
            <path d="m14 6-3 4" />
          </>
        )}
      </svg>
      <span className="lx-sonify__label">{supported ? "声化" : "声化不可用"}</span>
      <span className="lx-sonify__sep" aria-hidden="true">
        /
      </span>
      <span className="lx-sonify__value num">熵 {entropy == null ? "—" : fmtFixed(entropy, 2)}</span>
      {chosenProb != null ? (
        <span className="lx-sonify__prob num">p {fmtFixed(chosenProb, 2)}</span>
      ) : null}
      <span className="lx-sonify__state">{enabled ? "开" : "关"}</span>
    </button>
  );
}

export default SonificationToggle;
