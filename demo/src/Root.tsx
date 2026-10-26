import { CalculateMetadataFunction, Composition, staticFile } from "remotion";
import { ReportDemo, ReportDemoProps } from "./ReportDemo";

type Layout = { letterbox_top?: number; letterbox_bottom?: number };
type Node = { type: string; layout?: Layout | null; items?: Node[] };

const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
};

// The GIFs are 12 fps, so 24 fps is plenty whatever the video's own rate; <Video> samples it by time.
const FPS = 24;

// Loads the Edit AST and the EditScript, and sets the length from the shown part of the video.
const calculateMetadata: CalculateMetadataFunction<ReportDemoProps> = async ({
  props,
  abortSignal,
}) => {
  const [edit, script] = await Promise.all([
    fetch(staticFile(props.edit), { signal: abortSignal }).then((r) =>
      r.json(),
    ),
    fetch(staticFile(props.script), { signal: abortSignal }).then((r) =>
      r.text(),
    ),
  ]);
  const shots = (edit.timeline.sections as Node[]).flatMap((sec) =>
    (sec.items ?? []).filter((it) => it.type === "shot"),
  );
  const layouts = shots.map((s) => s.layout ?? {});
  return {
    durationInFrames: Math.round((props.end - props.start) * FPS),
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
    <>
      {/* docs/demo.gif, in the README's film trailer example */}
      <Composition
        id="Dune"
        component={ReportDemo}
        durationInFrames={581}
        fps={FPS}
        width={1280}
        height={800}
        defaultProps={{
          video: "dune_trailer.mp4",
          report: "dune_trailer.report.html",
          edit: "dune_trailer.edit.json",
          script: "dune_trailer.compact.edit",
          command: "editast analyze dune_trailer.mp4",
          credit: "Dune trailer · © Warner Bros. Pictures & Legendary",
          lastLane: "energy",
          start: 72.7,
          end: 96.9,
        }}
        calculateMetadata={calculateMetadata}
      />
      {/* docs/car-edit-demo.gif, at the top of the README */}
      <Composition
        id="CarEdit"
        component={ReportDemo}
        durationInFrames={346}
        fps={FPS}
        width={1280}
        height={800}
        defaultProps={{
          video: "car_edit.mp4",
          report: "car_edit.report.html",
          edit: "car_edit.edit.json",
          script: "car_edit.compact.edit",
          command: "editast analyze youtube.com/shorts/N-va3tJzyT0",
          credit: "by urboyliam",
          lastLane: "beats",
          start: 0,
          end: 14.4,
        }}
        calculateMetadata={calculateMetadata}
      />
    </>
  );
};
