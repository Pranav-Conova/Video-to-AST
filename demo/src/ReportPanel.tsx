import { useCallback, useLayoutEffect, useRef, useState } from "react";
import { IFrame, staticFile, useDelayRender } from "remotion";
import { clamp01, easeInOut, theme } from "./theme";

// Shows the real HTML report in an iframe, cropped and zoomed so its timeline lanes, down to
// the one named `lastLane`, fill the panel. Each frame scrolls the timeline a page at a time (like an
// editor following its playhead), draws the playhead, and outlines every block whose time
// span contains it.

const SVG_NS = "http://www.w3.org/2000/svg";
const PAGE_W = 640; // report px the page is laid out at; wider than the visible part
const MARGIN = 6; // report px kept around the timeline
const BELOW = 3; // report px kept below the last lane: half the gap, so the next lane stays hidden
const LEAD = 36; // report px of timeline left of the playhead after a page turn
const TAIL = 56; // a page turns when the playhead gets this close to the right edge
const TURN_SECONDS = 0.4;
const FLASH_SECONDS = 0.2; // instants (cuts) stay outlined this long; drops 3x as long

// Hide scrollbars in the capture; everything else is the report's own styling.
const CAPTURE_CSS = `html,body{overflow:hidden}.scroll{scrollbar-width:none}`;

type Item = { a: number; b: number; drop: boolean; x: number; y: number; w: number; h: number };
type Measured = {
  pps: number;
  x0: number;
  zoom: number;
  left: number; // the timeline's top-left corner in the page
  top: number;
  viewW: number; // visible width of the scrolling lanes
  maxScroll: number;
  height: number;
  items: Item[];
  scroller: HTMLElement;
  overlay: SVGGElement;
};

const SPAN = /(\d+(?:\.\d+)?)–(\d+(?:\.\d+)?)s/;
const DROP = /drop at (\d+(?:\.\d+)?)s/;

const span = (tip: string): [number, number, boolean] | null => {
  const m = SPAN.exec(tip);
  if (m) return [parseFloat(m[1]), parseFloat(m[2]), false];
  const d = DROP.exec(tip);
  return d ? [parseFloat(d[1]), parseFloat(d[1]), true] : null;
};

const measure = (doc: Document, lastLane: string, panelW: number, panelH: number): Measured | null => {
  const svg = doc.querySelector<SVGSVGElement>("svg.lanes");
  const scroller = doc.querySelector<HTMLElement>(".scroll");
  const tl = doc.querySelector<HTMLElement>(".tl");
  if (!svg || !scroller || !tl) return null;
  const box = svg.getBoundingClientRect();
  const items: Item[] = [];
  svg.querySelectorAll<SVGElement>("[data-tip]").forEach((el) => {
    const s = span(el.dataset.tip ?? "");
    if (!s) return;
    const r = el.getBoundingClientRect();
    items.push({ a: s[0], b: s[1], drop: s[2], x: r.left - box.left, y: r.top - box.top, w: r.width, h: r.height });
  });
  const overlay = doc.createElementNS(SVG_NS, "g");
  svg.append(overlay);
  const tlBox = tl.getBoundingClientRect();
  // Lane labels and lane groups come in the same order.
  const labels = [...doc.querySelectorAll("svg.labels text.lane")].map((n) => n.textContent);
  const lanes = svg.querySelectorAll(":scope > g[transform]");
  const last = lanes[labels.indexOf(lastLane)] ?? svg;
  const zoom = panelH / (last.getBoundingClientRect().bottom - tlBox.top + MARGIN + BELOW);
  const viewW = panelW / zoom - MARGIN - (scroller.getBoundingClientRect().left - tlBox.left);
  return {
    pps: parseFloat(svg.dataset.pps ?? "60"),
    x0: parseFloat(svg.dataset.x0 ?? "8"),
    zoom,
    left: tlBox.left,
    top: tlBox.top,
    viewW,
    maxScroll: scroller.scrollWidth - viewW,
    height: box.height,
    items,
    scroller,
    overlay,
  };
};

// Scroll offset at time t: fixed pages, each turn eased over TURN_SECONDS.
const pageScroll = (m: Measured, t: number, start: number) => {
  const x = m.x0 + t * m.pps;
  const s0 = Math.max(0, m.x0 + start * m.pps - LEAD);
  const step = m.viewW - LEAD - TAIL;
  const first = s0 + m.viewW - TAIL; // x of the first page turn
  let scroll = s0;
  if (x >= first) {
    const k = Math.floor((x - first) / step) + 1;
    const turnT = (first + (k - 1) * step - m.x0) / m.pps;
    scroll = s0 + (k - 1) * step + step * easeInOut(clamp01((t - turnT) / TURN_SECONDS));
  }
  return Math.min(m.maxScroll, scroll);
};

const el = (doc: Document, name: string, attrs: Record<string, string | number>) => {
  const node = doc.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
};

export const ReportPanel: React.FC<{
  src: string;
  t: number;
  start: number;
  lastLane: string;
  style: React.CSSProperties & { width: number; height: number };
}> = ({ src, t, start, lastLane, style }) => {
  const ref = useRef<HTMLIFrameElement>(null);
  const { delayRender, continueRender, cancelRender } = useDelayRender();
  const [handle] = useState(() => delayRender("Measuring the report timeline"));
  const [m, setM] = useState<Measured | null>(null);

  const onLoad = useCallback(() => {
    const doc = ref.current?.contentDocument;
    if (!doc) return cancelRender(new Error("report iframe has no document"));
    doc.documentElement.dataset.theme = "dark";
    const css = doc.createElement("style");
    css.textContent = CAPTURE_CSS;
    doc.head.append(css);
    const measured = measure(doc, lastLane, style.width - 2, style.height - 2); // inside the border
    if (!measured) return cancelRender(new Error(`${src} has no timeline`));
    setM(measured);
    continueRender(handle);
  }, [cancelRender, continueRender, handle, src, lastLane, style.width, style.height]);

  useLayoutEffect(() => {
    if (!m) return;
    const doc = m.overlay.ownerDocument;
    m.scroller.scrollLeft = pageScroll(m, t, start);
    const marks: Element[] = [];
    for (const it of m.items) {
      const instant = it.b - it.a < 1e-6;
      const lit = instant
        ? t >= it.a && t < it.a + FLASH_SECONDS * (it.drop ? 3 : 1)
        : t >= it.a && t < it.b;
      if (lit)
        marks.push(
          el(doc, "rect", {
            x: it.x - 1, y: it.y - 1, width: it.w + 2, height: it.h + 2, rx: 4,
            fill: "none", stroke: "#fff", "stroke-width": 2,
          }),
        );
    }
    const x = m.x0 + t * m.pps;
    marks.push(el(doc, "line", { x1: x, x2: x, y1: 0, y2: m.height, stroke: theme.playhead, "stroke-width": 2 }));
    marks.push(el(doc, "path", { d: `M${x - 6},0 L${x + 6},0 L${x},8 Z`, fill: theme.playhead }));
    m.overlay.replaceChildren(...marks);
  }, [m, t, start]);

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
      <IFrame
        ref={ref}
        src={staticFile(src)}
        onLoad={onLoad}
        style={{
          position: "absolute",
          left: m ? -(m.left - MARGIN) * m.zoom : 0,
          top: m ? -(m.top - MARGIN) * m.zoom : 0,
          width: PAGE_W,
          height: 2400,
          border: 0,
          transform: `scale(${m?.zoom ?? 1})`,
          transformOrigin: "0 0",
        }}
      />
    </div>
  );
};
