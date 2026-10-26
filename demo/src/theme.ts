// The dark tokens of the HTML report (editast/report/html.py), so the frame around it matches.
export const theme = {
  page: "#0d0d0d",
  surface: "#1a1a19",
  ink: "#f0efec",
  ink2: "#c3c2b7",
  muted: "#898781",
  border: "rgba(255,255,255,.10)",
  blue: "#3987e5",
  green: "#199e70",
  purple: "#9085e9",
  playhead: "#ff5c5c",
  ui: 'system-ui, -apple-system, "Segoe UI", sans-serif',
  mono: '"Cascadia Mono", Consolas, ui-monospace, monospace',
};

export const clamp01 = (x: number) => Math.min(1, Math.max(0, x));

export const easeInOut = (p: number) => (p < 0.5 ? 2 * p * p : 1 - (-2 * p + 2) ** 2 / 2);
