import { Video } from "@remotion/media";
import {
  AbsoluteFill,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { EditScriptPanel } from "./EditScriptPanel";
import { ReportPanel } from "./ReportPanel";
import { theme } from "./theme";

export type ReportDemoProps = {
  video: string; // files in public/
  report: string;
  edit: string;
  script: string;
  command: string; // shown above the report
  lastLane: string; // the report shows its timeline lanes down to this one
  credit: string; // shown above the video
  start: number; // seconds of the source shown
  end: number;
  videoWidth?: number; // landscape videos: the left column's width (portrait videos fill its height)
  // Filled in by calculateMetadata.
  scriptText?: string;
  aspect?: number;
  letterbox?: [number, number];
};

const PAD = 28;
const LABEL_H = 36;
const SCRIPT_H = 200; // EditScript panel under the report, in the portrait layout

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
  const shown = 1 - lbTop - lbBottom; // the part of the frame between the bars
  const videoTop = PAD + LABEL_H;
  const bottom = height - PAD;
  // Landscape: the video over the EditScript on the left, the report on the right.
  // Portrait: the video takes the left column's full height, and the EditScript sits under
  // the report.
  const portrait = aspect < 1;
  const videoH = portrait
    ? bottom - videoTop
    : Math.round(((props.videoWidth ?? 580) / aspect) * shown);
  const fullH = videoH / shown; // the whole frame, bars included
  const leftW = portrait
    ? Math.round(fullH * aspect)
    : (props.videoWidth ?? 580);
  const rightX = PAD + leftW + PAD;
  const rightW = width - rightX - PAD;
  const script = portrait
    ? { left: rightX, top: bottom - SCRIPT_H, width: rightW, height: SCRIPT_H }
    : {
        left: PAD,
        top: videoTop + videoH + 20 + LABEL_H,
        width: leftW,
        height: bottom - (videoTop + videoH + 20 + LABEL_H),
      };
  const report = {
    left: rightX,
    top: videoTop,
    width: rightW,
    height: (portrait ? script.top - 20 - LABEL_H : bottom) - videoTop,
  };

  return (
    <AbsoluteFill
      style={{
        backgroundColor: theme.page,
        color: theme.ink,
        fontFamily: theme.ui,
      }}
    >
      <Label
        x={PAD}
        y={PAD}
        w={leftW}
        title="Source"
        detail={props.credit}
        right={timecode(t)}
      />
      <div
        style={{
          position: "absolute",
          left: PAD,
          top: videoTop,
          width: leftW,
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
          style={{
            position: "absolute",
            left: 0,
            top: -fullH * lbTop,
            width: leftW,
            height: fullH,
          }}
        />
      </div>

      <Label
        x={script.left}
        y={script.top - LABEL_H}
        w={script.width}
        title="Edit AST"
        detail="EditScript at the playhead"
      />
      <EditScriptPanel text={props.scriptText ?? ""} t={t} style={script} />

      <Label
        x={rightX}
        y={PAD}
        w={rightW}
        title="HTML report"
        detail={props.command}
        mono
      />
      <ReportPanel
        src={props.report}
        t={t}
        start={props.start}
        lastLane={props.lastLane}
        style={report}
      />
    </AbsoluteFill>
  );
};
