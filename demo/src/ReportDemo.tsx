import { Video } from "@remotion/media";
import { AbsoluteFill, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { EditScriptPanel } from "./EditScriptPanel";
import { ReportPanel } from "./ReportPanel";
import { theme } from "./theme";

export type ReportDemoProps = {
  video: string; // files in public/
  report: string;
  edit: string;
  script: string;
  command: string; // shown above the report
  credit: string; // shown above the video
  start: number; // seconds of the source shown
  end: number;
  // Filled in by calculateMetadata.
  scriptText?: string;
  aspect?: number;
  letterbox?: [number, number];
};

const PAD = 28;
const LEFT_W = 580;
const LABEL_H = 36;

const Label: React.FC<{
  x: number;
  y: number;
  w: number;
  title: string;
  detail: string;
  mono?: boolean;
  right?: string;
}> = ({ x, y, w, title, detail, mono, right }) => (
  <div
    style={{
      position: "absolute",
      left: x,
      top: y,
      width: w,
      height: LABEL_H - 10,
      display: "flex",
      alignItems: "baseline",
      gap: 12,
      whiteSpace: "nowrap",
    }}
  >
    <span
      style={{
        fontSize: 16,
        fontWeight: 600,
        letterSpacing: "0.06em",
        textTransform: "uppercase",
        color: theme.ink,
      }}
    >
      {title}
    </span>
    <span
      style={{
        fontSize: 17,
        color: theme.muted,
        fontFamily: mono ? theme.mono : theme.ui,
        overflow: "hidden",
        textOverflow: "ellipsis",
        flex: "1 1 auto",
      }}
    >
      {detail}
    </span>
    {right ? (
      <span
        style={{
          fontSize: 17,
          fontFamily: theme.mono,
          color: theme.ink2,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {right}
      </span>
    ) : null}
  </div>
);

const timecode = (t: number) => {
  const m = Math.floor(t / 60);
  const s = (t - 60 * m).toFixed(2);
  return `${m}:${s.length < 5 ? "0" + s : s}`;
};

export const ReportDemo: React.FC<ReportDemoProps> = (props) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  const t = props.start + frame / fps;

  const aspect = props.aspect ?? 16 / 9;
  const [lbTop, lbBottom] = props.letterbox ?? [0, 0];
  const fullH = LEFT_W / aspect; // the whole frame, bars included
  const videoH = Math.round(fullH * (1 - lbTop - lbBottom));
  const videoTop = PAD + LABEL_H;
  const scriptTop = videoTop + videoH + 20 + LABEL_H;
  const rightX = PAD + LEFT_W + PAD;
  const rightW = width - rightX - PAD;

  return (
    <AbsoluteFill style={{ backgroundColor: theme.page, color: theme.ink, fontFamily: theme.ui }}>
      <Label x={PAD} y={PAD} w={LEFT_W} title="Source" detail={props.credit} right={timecode(t)} />
      <div
        style={{
          position: "absolute",
          left: PAD,
          top: videoTop,
          width: LEFT_W,
          height: videoH,
          overflow: "hidden",
          borderRadius: 10,
          backgroundColor: "#000",
        }}
      >
        <Video
          src={staticFile(props.video)}
          muted
          trimBefore={Math.round(props.start * fps)}
          style={{ position: "absolute", left: 0, top: -fullH * lbTop, width: LEFT_W, height: fullH }}
        />
      </div>

      <Label x={PAD} y={scriptTop - LABEL_H} w={LEFT_W} title="Edit AST" detail="EditScript at the playhead" />
      <EditScriptPanel
        text={props.scriptText ?? ""}
        t={t}
        style={{ left: PAD, top: scriptTop, width: LEFT_W, height: height - PAD - scriptTop }}
      />

      <Label x={rightX} y={PAD} w={rightW} title="HTML report" detail={props.command} mono />
      <ReportPanel
        src={props.report}
        t={t}
        start={props.start}
        style={{ left: rightX, top: videoTop, width: rightW, height: height - PAD - videoTop }}
      />
    </AbsoluteFill>
  );
};
