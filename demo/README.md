# README demo

`docs/demo.gif`, at the top of the main README, is rendered from this
[Remotion](https://www.remotion.dev/) project. The composition in `src/` plays the
Sintel trailer next to its compact EditScript and the real HTML report from
`editast analyze`, which it loads in an iframe. On every frame it moves a playhead
across the report's timeline, turns the timeline's pages, and outlines the shots,
transitions and effects under the playhead.

## Rebuild the GIF

Needs Node 20+ and the editast Python environment (see the main README).

```bash
cd demo
npm install
python prepare.py   # downloads the Sintel trailer (4 MB) and analyses it into public/
npm run frames      # renders out/frames/*.png at 880x550
python make_gif.py  # writes ../docs/demo.gif
```

`npm run dev` opens Remotion Studio to preview the composition. It shows 12.7–36.9 s
of the trailer, set by the `start` and `end` props in `src/Root.tsx`.

To use another video, put it in `public/` with its `.edit.json`, `.report.html` and
compact EditScript (`editast show clip.edit.json --format compact`), and change the
props. The layout assumes a landscape video.

## Licences

Sintel © Blender Foundation, CC-BY 3.0, [durian.blender.org](https://durian.blender.org/).
Remotion is free for individuals and companies of up to 3 people; larger companies
need a [company license](https://www.remotion.dev/license).
