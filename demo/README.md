# README demos

The two GIFs in the main README are rendered from this
[Remotion](https://www.remotion.dev/) project:

| GIF | Composition | Shows |
|---|---|---|
| `docs/car-edit-demo.gif` | `CarEdit` | all 14.4 s of a vertical BMW edit by urboyliam |
| `docs/demo.gif` | `Dune` | 24 s of the Dune trailer (72.7–96.9 s) |

Each plays the video next to its compact EditScript and the real HTML report from
`editast analyze`, which it loads in an iframe. On every frame it moves a playhead
across the report's timeline, turns the timeline's pages, and outlines the shots,
transitions, effects and text under the playhead.

## Rebuild the GIFs

Needs Node 20+ and the editast Python environment (see the main README).

```bash
cd demo
npm install

python prepare.py https://www.youtube.com/watch?v=n9xhJrPXop4 --name dune_trailer
npm run frames      # renders out/frames/*.png at 880x550
python make_gif.py  # writes ../docs/demo.gif

python prepare.py https://www.youtube.com/shorts/N-va3tJzyT0 --name car_edit
npm run frames:car
python make_gif.py out/car-frames ../docs/car-edit-demo.gif
```

`prepare.py` downloads a video, analyses it, and puts the video, its Edit AST,
compact EditScript and HTML report in `public/`. `npm run dev` opens Remotion
Studio to preview the compositions.

YouTube decides which encode it serves. The Dune GIF was made from the H.264 1080p
stream; an AV1 download gives the same cuts, with a few motion effects that differ.

The props in `src/Root.tsx` choose the files, the part of the video shown (`start`
and `end`), the last report lane shown (`lastLane`; lanes below it are cropped so
the text stays readable) and, for a landscape video, its width (`videoWidth`;
full-frame footage in motion makes the GIF large). A portrait video fills the left
column's height, and its EditScript moves under the report. To use another video,
run `prepare.py` on it and add a composition.

## Licences

The videos belong to their owners: the Dune trailer to Warner Bros. Pictures and
Legendary, the BMW edit to urboyliam. The GIFs show them to illustrate the
analysis; they are not covered by this repository's licence. Remotion is free for
individuals and companies of up to 3 people; larger companies need a
[company license](https://www.remotion.dev/license).
