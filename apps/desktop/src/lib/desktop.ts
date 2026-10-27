// Native bridge exposed by desktop/launcher.py through pywebview's js_api.
// It is absent in a normal browser tab, where the Blob fallback is used.

export interface DesktopSaveApi {
  save_text_file?: (
    filename: string,
    content: string,
    path?: string,
  ) => Promise<string | null> | string | null;
}

declare global {
  interface Window {
    pywebview?: { api?: DesktopSaveApi };
  }
}

export function desktopSaveApi(): DesktopSaveApi | null {
  const api = window.pywebview?.api;
  return api && typeof api.save_text_file === "function" ? api : null;
}
