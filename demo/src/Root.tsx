import { CalculateMetadataFunction, Composition, staticFile } from "remotion";
import { ReportDemo, ReportDemoProps } from "./ReportDemo";

type Layout = { letterbox_top?: number; letterbox_bottom?: number };
type Node = { type: string; layout?: Layout | null; items?: Node[] };

const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
};

// Loads the Edit AST and the EditScript, and sets the frame rate and length from them.
const calculateMetadata: CalculateMetadataFunction<ReportDemoProps> = async ({
  props,
  abortSignal,
}) => {
  const [edit, script] = await Promise.all([
    fetch(staticFile(props.edit), { signal: abortSignal }).then((r) => r.json()),
    fetch(staticFile(props.script), { signal: abortSignal }).then((r) => r.text()),
  ]);
  const shots = (edit.timeline.sections as Node[]).flatMap((sec) =>
    (sec.items ?? []).filter((it) => it.type === "shot"),
  );
  const layouts = shots.map((s) => s.layout ?? {});
  const fps: number = edit.media.fps;
  return {
    fps,
    durationInFrames: Math.round((props.end - props.start) * fps),
    props: {
      ...props,
      scriptText: script,
      aspect: edit.media.width / edit.media.height,
      // The video panel crops away the letterbox bars most shots have.
      letterbox: [
        median(layouts.map((l) => l.letterbox_top ?? 0)),
        median(layouts.map((l) => l.letterbox_bottom ?? 0)),
      ],
    },
  };
};

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="ReportDemo"
      component={ReportDemo}
      durationInFrames={581}
      fps={24}
      width={1280}
      height={800}
      defaultProps={{
        video: "sintel_trailer.mp4",
        report: "sintel_trailer.report.html",
        edit: "sintel_trailer.edit.json",
        script: "sintel_trailer.compact.edit",
        command: "editast analyze sintel_trailer.mp4",
        credit: "Sintel trailer · © Blender Foundation · CC-BY 3.0",
        start: 12.7,
        end: 36.9,
      }}
      calculateMetadata={calculateMetadata}
    />
  );
};
