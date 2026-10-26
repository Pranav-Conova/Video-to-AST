import { useMemo } from "react";
import { clamp01, easeInOut, theme } from "./theme";

const LINE_H = 27;
const CONTEXT_ROWS = 2; // lines kept above the current shot
const SCROLL_SECONDS = 0.4;
const CUT_FLASH = 0.15; // seconds a zero-length cut stays lit

type Line = { text: string; kind: string; a: number | null; b: number | null };

const SPAN = /(\d+(?:\.\d+)?)s\.\.(\d+(?:\.\d+)?)s/;

// The timeline part of a compact EditScript: section blocks, without the analyzer's `params {...}`.
const parse = (src: string): Line[] => {
  const out: Line[] = [];
  let inside = false;
  for (const raw of src.split(/\r?\n/)) {
    if (/^ {2}section /.test(raw)) inside = true;
    else if (inside && /^ {2}\S/.test(raw) && !/^ {2}\}/.test(raw)) break; // overlays, patterns, ...
    if (!inside) continue;
    const text = raw.slice(2).replace(/ params \{.*\}(?= \{$|$)/, "");
    const m = SPAN.exec(text);
    out.push({
      text,
      kind: text.trim().split(" ")[0] ?? "",
      a: m ? parseFloat(m[1]) : null,
      b: m ? parseFloat(m[2]) : null,
    });
  }
  return out;
};

const isActive = (line: Line, t: number) => {
  if (line.a === null || line.b === null || line.kind === "}") return false;
  if (line.b - line.a < 1e-6) return t >= line.a && t < line.a + CUT_FLASH;
  return t >= line.a && t < line.b;
};

const TOKEN = /("(?:[^"\\]|\\.)*")|(#[0-9A-Fa-f]{6}\b)|((?<![A-Za-z0-9_])[+-]?\d+(?:\.\d+)?(?:s|x|deg)?(?![A-Za-z0-9_]))/g;

// Same colours as the EditScript view in the HTML report.
const Highlighted: React.FC<{ text: string }> = ({ text }) => {
  const indent = /^\s*/.exec(text)![0];
  const rest = text.slice(indent.length);
  const kw = /^[A-Za-z_]\w*/.exec(rest)?.[0] ?? "";
  const parts: React.ReactNode[] = [indent];
  if (kw) parts.push(<span key="kw" style={{ color: theme.purple, fontWeight: 600 }}>{kw}</span>);
  const body = rest.slice(kw.length);
  let pos = 0;
  for (const m of body.matchAll(TOKEN)) {
    parts.push(body.slice(pos, m.index));
    parts.push(
      <span key={m.index} style={{ color: m[1] ? theme.green : theme.blue }}>
        {m[0]}
      </span>,
    );
    pos = m.index! + m[0].length;
  }
  parts.push(body.slice(pos));
  return <>{parts}</>;
};

export const EditScriptPanel: React.FC<{ text: string; t: number; style: React.CSSProperties }> = ({
  text,
  t,
  style,
}) => {
  const lines = useMemo(() => parse(text), [text]);
  const shotRows = lines.flatMap((l, i) => (l.kind === "shot" ? [i] : []));
  const cur = shotRows.findIndex((i) => isActive(lines[i], t));
  const k = cur >= 0 ? cur : Math.max(0, shotRows.findIndex((i) => (lines[i].a ?? 0) > t) - 1);
  const row = (j: number) => Math.max(0, shotRows[Math.max(0, j)] - CONTEXT_ROWS) * LINE_H;
  // Scroll to the new shot over SCROLL_SECONDS after it starts.
  const p = easeInOut(clamp01((t - (lines[shotRows[k]]?.a ?? 0)) / SCROLL_SECONDS));
  const y = row(k - 1) + (row(k) - row(k - 1)) * p;

  return (
    <div
      style={{
        position: "absolute",
        overflow: "hidden",
        borderRadius: 12,
        border: `1px solid ${theme.border}`,
        backgroundColor: theme.surface,
        ...style,
      }}
    >
      <div style={{ position: "absolute", left: 0, right: 0, top: 12 - y }}>
        {lines.map((line, i) => {
          const current = i === shotRows[k];
          const active = isActive(line, t);
          return (
            <div
              key={i}
              style={{
                height: LINE_H,
                lineHeight: `${LINE_H}px`,
                paddingLeft: 16,
                whiteSpace: "pre",
                fontFamily: theme.mono,
                fontSize: 16,
                color: theme.ink2,
                opacity: active || current ? 1 : 0.45,
                backgroundColor: current ? "rgba(57,135,229,.20)" : active ? "rgba(255,255,255,.06)" : undefined,
                boxShadow: current ? `inset 3px 0 0 ${theme.blue}` : undefined,
              }}
            >
              <Highlighted text={line.text} />
            </div>
          );
        })}
      </div>
    </div>
  );
};
